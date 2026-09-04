import { createHash } from 'node:crypto';

import { embedBackfillLockId } from './embed-backfill-lock.ts';
import { embedBatchWithBackoff } from './embed-retry.ts';
import { wrapChunkTextsForStoredMode } from './embedding-context.ts';
import { getEmbeddingDimensions, getEmbeddingModel } from './ai/gateway.ts';
import {
  isDbLockHandleForEngine,
  syncLockId,
  type DbLockHandle,
} from './db-lock.ts';
import type { BrainEngine } from './engine.ts';
import {
  createRequiredMigrationEmbedPairs,
  isRequiredMigrationEmbedAllowlistRunActive,
  RequiredMigrationEmbedAllowlistError,
  runWithRequiredMigrationEmbedPairs,
  type RequiredMigrationEmbedAllowlistRun,
} from './required-migration-embed-allowlist.ts';
import {
  assertRequiredMigrationWireAttemptHookActive,
  getActiveRequiredMigrationWireAttemptHook,
  isRequiredMigrationEmbedAllowlistModeRequired,
  type RequiredMigrationNativeAttemptHook,
} from './required-migration-wire-attempt.ts';
import { quoteIdentifier, resolveActiveEmbeddingColumnFromEngine } from './search/embedding-column.ts';

export interface RequiredMigrationEmbedAdmission {
  readonly sourceId: string;
  readonly locks: readonly [DbLockHandle, DbLockHandle];
  readonly allowlist: RequiredMigrationEmbedAllowlistRun;
  readonly wireAttemptHook: RequiredMigrationNativeAttemptHook;
}

interface RequiredMigrationEmbedAdmissionOpts {
  stale?: boolean;
  all?: boolean;
  slug?: string;
  slugs?: string[];
  sourceId?: string;
  dryRun?: boolean;
  singleFlight?: boolean;
  heldLocks?: DbLockHandle[];
}

interface RequiredMigrationEmbedResult {
  embedded: number;
  total_chunks: number;
  pages_processed: number;
}

interface RequiredMigrationEmbedRunOpts {
  batchSize?: number;
  signal?: AbortSignal;
  onProgress?: (done: number, total: number, embedded: number) => void;
}

export const REQUIRED_MIGRATION_EMBED_MODEL = 'openrouter:google/gemini-embedding-001';
export const REQUIRED_MIGRATION_EMBED_DIMENSIONS = 1024;
export const REQUIRED_MIGRATION_EMBED_BATCH_SIZE = 100;

function contextFault(): never {
  throw new RequiredMigrationEmbedAllowlistError('context');
}

function assertRequiredMigrationEmbeddingTarget(): void {
  if (
    getEmbeddingModel() !== REQUIRED_MIGRATION_EMBED_MODEL ||
    getEmbeddingDimensions() !== REQUIRED_MIGRATION_EMBED_DIMENSIONS ||
    !isRequiredMigrationEmbedAllowlistModeRequired() ||
    Boolean(process.env.OPENROUTER_BASE_URL?.trim())
  ) {
    contextFault();
  }
}

function exactAllowlistSource(run: RequiredMigrationEmbedAllowlistRun): string {
  const sources = new Set(Array.from(run.entries.values(), entry => entry.identity.source_id));
  if (sources.size !== 1) contextFault();
  return sources.values().next().value as string;
}

async function refreshExactLocks(locks: readonly DbLockHandle[]): Promise<void> {
  for (const lock of locks) {
    let owned: boolean;
    try {
      owned = await lock.refresh();
    } catch {
      contextFault();
    }
    if (!owned) contextFault();
  }
}

/** Fail-closed admission for the REQUIRED per-source embed drain. */
export async function admitRequiredMigrationEmbedRun(
  engine: BrainEngine,
  run: RequiredMigrationEmbedAllowlistRun,
  opts: RequiredMigrationEmbedAdmissionOpts,
): Promise<RequiredMigrationEmbedAdmission> {
  const wireAttemptHook = getActiveRequiredMigrationWireAttemptHook();
  if (!wireAttemptHook) contextFault();
  const sourceId = exactAllowlistSource(run);
  try {
    if (
      opts.stale !== true || opts.sourceId !== sourceId || opts.singleFlight !== true ||
      opts.dryRun !== false || opts.all !== undefined || opts.slug !== undefined || opts.slugs !== undefined ||
      opts.heldLocks?.length !== 2
    ) {
      contextFault();
    }
    assertRequiredMigrationEmbeddingTarget();

    const expectedIds = [syncLockId(sourceId), embedBackfillLockId(sourceId)] as const;
    const byId = new Map(opts.heldLocks.map(lock => [lock?.id, lock] as const));
    if (byId.size !== 2 || expectedIds.some(id => !byId.has(id))) contextFault();
    const locks = expectedIds.map(id => byId.get(id)!) as unknown as [DbLockHandle, DbLockHandle];
    if (locks.some(lock => !isDbLockHandleForEngine(lock, engine))) contextFault();
    await refreshExactLocks(locks);
    return Object.freeze({
      sourceId,
      locks: Object.freeze(locks),
      allowlist: run,
      wireAttemptHook,
    });
  } finally {
    assertRequiredMigrationWireAttemptHookActive(wireAttemptHook);
  }
}

function sha256(text: string): string {
  return createHash('sha256').update(text, 'utf8').digest('hex');
}

function md5(text: string): string {
  return createHash('md5').update(text, 'utf8').digest('hex');
}

function chunkKey(sourceId: string, slug: string, chunkSource: string, chunkIndex: number): string {
  return JSON.stringify([sourceId, slug, chunkSource, chunkIndex]);
}

interface RequiredMigrationProjectionRow {
  page_id: number | string;
  slug: string;
  chunk_index: number | string;
  chunk_source: string;
  chunk_text: string;
  model: string | null;
  embedded_text_hash: string | null;
  embedding_is_null: boolean;
  dimensions: number | string | null;
}

interface RequiredMigrationProjectionState {
  pending: ReadonlySet<string>;
  satisfied: number;
}

async function inspectRequiredMigrationProjection(
  engine: BrainEngine,
  admission: RequiredMigrationEmbedAdmission,
  embeddingColumn: { name: string; dimensions: number },
  model: string,
): Promise<RequiredMigrationProjectionState> {
  const column = quoteIdentifier(embeddingColumn.name);
  const rows = await engine.executeRaw<RequiredMigrationProjectionRow>(
    `SELECT p.id AS page_id, p.slug, c.chunk_index, c.chunk_source, c.chunk_text,
            c.model, c.embedded_text_hash,
            (c.${column} IS NULL) AS embedding_is_null,
            CASE WHEN c.${column} IS NULL THEN NULL ELSE vector_dims(c.${column})::int END AS dimensions
       FROM pages p
       JOIN content_chunks c ON c.page_id = p.id
      WHERE p.source_id = $1
      ORDER BY p.slug, c.chunk_source, c.chunk_index, c.id`,
    [admission.sourceId],
  );
  if (rows.length !== admission.allowlist.entries.size) contextFault();

  const seen = new Set<string>();
  const pending = new Set<string>();
  let satisfied = 0;
  for (const row of rows) {
    const pageId = Number(row.page_id);
    const chunkIndex = Number(row.chunk_index);
    if (
      !Number.isSafeInteger(pageId) || pageId <= 0 ||
      typeof row.slug !== 'string' || row.slug.length === 0 ||
      !Number.isSafeInteger(chunkIndex) || chunkIndex < 0 ||
      typeof row.chunk_source !== 'string' || row.chunk_source.length === 0 ||
      typeof row.chunk_text !== 'string'
    ) {
      contextFault();
    }
    const key = chunkKey(admission.sourceId, row.slug, row.chunk_source, chunkIndex);
    if (seen.has(key)) contextFault();
    seen.add(key);
    const entry = admission.allowlist.entries.get(key);
    if (!entry || sha256(row.chunk_text) !== entry.stored_text_sha256) contextFault();

    if (row.embedding_is_null === true) {
      pending.add(key);
    } else if (
      row.embedding_is_null === false &&
      row.model === model &&
      Number(row.dimensions) === embeddingColumn.dimensions &&
      row.embedded_text_hash === md5(row.chunk_text)
    ) {
      satisfied++;
    } else {
      contextFault();
    }
  }
  if (seen.size !== admission.allowlist.entries.size) contextFault();
  return { pending, satisfied };
}

async function assertPendingProjectionIsDrainable(
  engine: BrainEngine,
  admission: RequiredMigrationEmbedAdmission,
  pending: ReadonlySet<string>,
): Promise<void> {
  const seen = new Set<string>();
  const batchSize = 2000;
  let afterPageId = 0;
  let afterChunkIndex = -1;
  for (;;) {
    const batch = await engine.listStaleChunks({
      batchSize,
      afterPageId,
      afterChunkIndex,
      sourceId: admission.sourceId,
    });
    if (batch.length === 0) break;
    for (const row of batch) {
      if (row.source_id !== admission.sourceId) contextFault();
      const key = chunkKey(row.source_id, row.slug, row.chunk_source, row.chunk_index);
      const entry = admission.allowlist.entries.get(key);
      if (!pending.has(key) || seen.has(key) || !entry || sha256(row.chunk_text) !== entry.stored_text_sha256) {
        contextFault();
      }
      seen.add(key);
    }
    const last = batch[batch.length - 1];
    if (
      last.page_id < afterPageId ||
      (last.page_id === afterPageId && last.chunk_index <= afterChunkIndex)
    ) {
      contextFault();
    }
    afterPageId = last.page_id;
    afterChunkIndex = last.chunk_index;
    if (batch.length < batchSize) break;
  }
  if (seen.size !== pending.size) contextFault();
}

/** REQUIRED-only stale loop: provider output flows exclusively through vector CAS. */
export async function runRequiredMigrationEmbedDrain(
  engine: BrainEngine,
  admission: RequiredMigrationEmbedAdmission,
  result: RequiredMigrationEmbedResult,
  opts: RequiredMigrationEmbedRunOpts = {},
): Promise<void> {
  assertRequiredMigrationWireAttemptHookActive(admission.wireAttemptHook);
  try {
    await runRequiredMigrationEmbedDrainActive(engine, admission, result, opts);
  } finally {
    assertRequiredMigrationWireAttemptHookActive(admission.wireAttemptHook);
  }
}

async function runRequiredMigrationEmbedDrainActive(
  engine: BrainEngine,
  admission: RequiredMigrationEmbedAdmission,
  result: RequiredMigrationEmbedResult,
  opts: RequiredMigrationEmbedRunOpts,
): Promise<void> {
  assertRequiredMigrationEmbeddingTarget();
  const model = getEmbeddingModel();
  const resolved = await resolveActiveEmbeddingColumnFromEngine(engine);
  if (resolved.embeddingModel && resolved.embeddingModel !== model) contextFault();
  const embeddingColumn = resolved.embeddingModel
    ? resolved
    : { ...resolved, dimensions: getEmbeddingDimensions(), embeddingModel: model };
  if (
    !isRequiredMigrationEmbedAllowlistRunActive(admission.allowlist) ||
    exactAllowlistSource(admission.allowlist) !== admission.sourceId
  ) {
    contextFault();
  }
  const initial = await inspectRequiredMigrationProjection(engine, admission, embeddingColumn, model);
  if (initial.pending.size + initial.satisfied !== admission.allowlist.entries.size) contextFault();
  await assertPendingProjectionIsDrainable(engine, admission, initial.pending);

  const requestedBatchSize = opts.batchSize ?? REQUIRED_MIGRATION_EMBED_BATCH_SIZE;
  if (!Number.isSafeInteger(requestedBatchSize) || requestedBatchSize <= 0) contextFault();
  const batchSize = Math.min(requestedBatchSize, REQUIRED_MIGRATION_EMBED_BATCH_SIZE);
  const staleCount = initial.pending.size;
  let afterPageId = 0;
  let afterChunkIndex = -1;
  let processedPages = 0;

  while (!opts.signal?.aborted) {
    const batch = await engine.listStaleChunks({
      batchSize,
      afterPageId,
      afterChunkIndex,
      sourceId: admission.sourceId,
    });
    if (batch.length === 0) break;
    const last = batch[batch.length - 1];
    afterPageId = last.page_id;
    afterChunkIndex = last.chunk_index;
    result.total_chunks += batch.length;

    const pages = new Map<string, { rows: typeof batch; indexes: number[] }>();
    for (let index = 0; index < batch.length; index++) {
      const row = batch[index];
      if (row.source_id !== admission.sourceId) contextFault();
      const page = pages.get(row.slug);
      if (page) {
        page.rows.push(row);
        page.indexes.push(index);
      } else {
        pages.set(row.slug, { rows: [row], indexes: [index] });
      }
    }

    const providerTexts = new Array<string>(batch.length);
    for (const [slug, pageBatch] of pages) {
      if (opts.signal?.aborted) contextFault();
      const page = await engine.getPage(slug, { sourceId: admission.sourceId });
      if (!page) contextFault();
      const pageTexts = wrapChunkTextsForStoredMode(page, pageBatch.rows);
      for (let index = 0; index < pageTexts.length; index++) {
        providerTexts[pageBatch.indexes[index]] = pageTexts[index];
      }
    }
    if (providerTexts.some(text => typeof text !== 'string')) contextFault();

    const pairs = createRequiredMigrationEmbedPairs(batch, providerTexts);
    const embeddings = await runWithRequiredMigrationEmbedPairs(
      pairs,
      () => embedBatchWithBackoff(providerTexts, { abortSignal: opts.signal }),
    );
    if (
      embeddings.length !== batch.length ||
      embeddings.some(embedding => !(embedding instanceof Float32Array) || embedding.length !== embeddingColumn.dimensions)
    ) {
      contextFault();
    }
    if (opts.signal?.aborted) contextFault();

    // Provider success is not authority to write: synchronously re-prove
    // BOTH exact handles immediately before the single batch CAS primitive.
    await refreshExactLocks(admission.locks);
    await engine.updateRequiredMigrationChunkEmbeddings({
      embeddingColumn,
      model,
      locks: admission.locks.map(lock => ({ id: lock.id, acquired_at: lock.acquiredAt })),
      rows: batch.map((row, index) => ({
        page_id: row.page_id,
        chunk_index: row.chunk_index,
        chunk_source: row.chunk_source,
        expected_text_sha256: sha256(row.chunk_text),
        expected_text: row.chunk_text,
        embedding: embeddings[index],
      })),
    });
    result.embedded += batch.length;
    processedPages += pages.size;
    result.pages_processed += pages.size;
    opts.onProgress?.(processedPages, staleCount, result.embedded);

    if (batch.length < batchSize) break;
  }

  if (opts.signal?.aborted) contextFault();
  await refreshExactLocks(admission.locks);
  const terminal = await inspectRequiredMigrationProjection(engine, admission, embeddingColumn, model);
  if (terminal.pending.size !== 0 || terminal.satisfied !== admission.allowlist.entries.size) contextFault();
}
