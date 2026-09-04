import { createHash } from 'crypto';
import { isDbLockHandleForEngine, syncLockId, type DbLockHandle } from './db-lock.ts';
import type { BrainEngine } from './engine.ts';
import { assertValidSourceId } from './source-id.ts';
import type { ChunkInput } from './types.ts';

export type RequiredMigrationChunkSource = ChunkInput['chunk_source'];

export interface RequiredMigrationFileContract {
  pageProvenance: {
    source_kind: string;
    source_uri: string;
    ingested_via: string;
  };
  rawData: {
    source: string;
    data: {
      record_token: string;
      lifecycle: 'active' | 'expired' | 'superseded';
      invalidation_reason: string | null;
      superseded_by: string | null;
      body_chars: number;
      body_bytes: number;
      anchor_sha256: string;
    };
  };
  expectedProjection: {
    markdown_sha256: string;
    page_hash: string;
    chunk_count: number;
    chunks: ReadonlyArray<{
      chunk_source: RequiredMigrationChunkSource;
      chunk_index: number;
      chunk_sha256: string;
    }>;
  };
}

export type RequiredMigrationFileSidecarResolver = (
  relativePath: string,
) => RequiredMigrationFileContract | undefined | Promise<RequiredMigrationFileContract | undefined>;

/** Base projection contract passed through the import layer. */
export interface RequiredMigrationSyncOptions {
  mode: 'required';
  fileSidecarResolver?: RequiredMigrationFileSidecarResolver;
}

/** Runtime-only sync contract. The caller owns and refreshes this exact lock. */
export interface RequiredMigrationSyncRuntimeOptions extends RequiredMigrationSyncOptions {
  sourceId: string;
  sourceLock: DbLockHandle;
  signal: AbortSignal;
}

export class RequiredMigrationContractError extends Error {
  constructor(message: string) {
    super(`[required-migration] ${message}`);
    this.name = 'RequiredMigrationContractError';
  }
}

export function assertRequiredMigrationSyncOptions(
  opts: RequiredMigrationSyncOptions,
): asserts opts is RequiredMigrationSyncOptions & { fileSidecarResolver: RequiredMigrationFileSidecarResolver } {
  if (opts?.mode !== 'required') throw new RequiredMigrationContractError('mode must be exactly "required"');
  if (typeof opts.fileSidecarResolver !== 'function') {
    throw new RequiredMigrationContractError('fileSidecarResolver is required');
  }
}

export function assertRequiredMigrationSyncRuntimeOptions(
  opts: RequiredMigrationSyncRuntimeOptions,
): asserts opts is RequiredMigrationSyncRuntimeOptions & { fileSidecarResolver: RequiredMigrationFileSidecarResolver } {
  assertRequiredMigrationSyncOptions(opts);
  try {
    assertValidSourceId(opts.sourceId);
  } catch (error) {
    throw new RequiredMigrationContractError(error instanceof Error ? error.message : String(error));
  }
  const expectedLockId = syncLockId(opts.sourceId);
  const lock = opts.sourceLock;
  if (
    !lock || typeof lock !== 'object' || lock.id !== expectedLockId ||
    typeof lock.acquiredAt !== 'string' || lock.acquiredAt.length === 0 ||
    typeof lock.refresh !== 'function' || typeof lock.release !== 'function'
  ) {
    throw new RequiredMigrationContractError(`sourceLock must be the exact externally-held ${expectedLockId} handle`);
  }
  const signal = opts.signal;
  if (
    !signal || typeof signal.aborted !== 'boolean' ||
    typeof signal.addEventListener !== 'function' || signal.aborted
  ) {
    throw new RequiredMigrationContractError('a live runner-owned AbortSignal is required');
  }
}

export async function assertRequiredMigrationSyncRuntimeLock(
  engine: BrainEngine,
  opts: RequiredMigrationSyncRuntimeOptions,
): Promise<void> {
  assertRequiredMigrationSyncRuntimeOptions(opts);
  if (!isDbLockHandleForEngine(opts.sourceLock, engine)) {
    throw new RequiredMigrationContractError('sourceLock was not acquired from the target engine');
  }
  let owned: boolean;
  try {
    owned = await opts.sourceLock.refresh();
  } catch (error) {
    throw new RequiredMigrationContractError(
      `sourceLock validation failed (${error instanceof Error ? error.message : String(error)})`,
    );
  }
  if (!owned) {
    throw new RequiredMigrationContractError(`sourceLock no longer owns ${syncLockId(opts.sourceId)}`);
  }
  if (opts.signal.aborted) {
    throw new RequiredMigrationContractError('runner-owned AbortSignal fired during lock validation');
  }
}

export function requiredMigrationProjectionContract(
  opts: RequiredMigrationSyncRuntimeOptions,
): RequiredMigrationSyncOptions & { fileSidecarResolver: RequiredMigrationFileSidecarResolver } {
  assertRequiredMigrationSyncRuntimeOptions(opts);
  return { mode: 'required', fileSidecarResolver: opts.fileSidecarResolver };
}

const SHA256_HEX_RE = /^[a-f0-9]{64}$/;
const REQUIRED_SIDECAR_KEYS = [
  'anchor_sha256', 'body_bytes', 'body_chars', 'invalidation_reason',
  'lifecycle', 'record_token', 'superseded_by',
] as const;

function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  const obj = value as Record<string, unknown>;
  return `{${Object.keys(obj).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(obj[key])}`).join(',')}}`;
}

function sha256Text(value: string): string {
  return createHash('sha256').update(value, 'utf8').digest('hex');
}

function requireSha256(value: unknown, field: string, relativePath: string): asserts value is string {
  if (typeof value !== 'string' || !SHA256_HEX_RE.test(value)) {
    throw new RequiredMigrationContractError(`${relativePath}: invalid ${field}`);
  }
}

function validateRequiredMigrationFileContract(value: RequiredMigrationFileContract, relativePath: string): void {
  if (!value || typeof value !== 'object') {
    throw new RequiredMigrationContractError(`${relativePath}: resolver returned no contract`);
  }
  const provenance = value.pageProvenance;
  if (
    !provenance ||
    typeof provenance.source_kind !== 'string' || provenance.source_kind.length === 0 ||
    typeof provenance.source_uri !== 'string' || provenance.source_uri.length === 0 ||
    typeof provenance.ingested_via !== 'string' || provenance.ingested_via.length === 0
  ) {
    throw new RequiredMigrationContractError(`${relativePath}: incomplete page provenance`);
  }
  const rawData = value.rawData;
  const data = rawData?.data;
  if (!rawData || typeof rawData.source !== 'string' || rawData.source.length === 0 || !data || typeof data !== 'object') {
    throw new RequiredMigrationContractError(`${relativePath}: incomplete raw_data sidecar`);
  }
  if (canonicalJson(Object.keys(data).sort()) !== canonicalJson([...REQUIRED_SIDECAR_KEYS].sort())) {
    throw new RequiredMigrationContractError(`${relativePath}: raw_data sidecar keys do not match the safe contract`);
  }
  requireSha256(data.record_token, 'rawData.data.record_token', relativePath);
  requireSha256(data.anchor_sha256, 'rawData.data.anchor_sha256', relativePath);
  if (!['active', 'expired', 'superseded'].includes(data.lifecycle)) {
    throw new RequiredMigrationContractError(`${relativePath}: invalid rawData.data.lifecycle`);
  }
  if (data.invalidation_reason !== null && typeof data.invalidation_reason !== 'string') {
    throw new RequiredMigrationContractError(`${relativePath}: invalid rawData.data.invalidation_reason`);
  }
  if (data.superseded_by !== null && typeof data.superseded_by !== 'string') {
    throw new RequiredMigrationContractError(`${relativePath}: invalid rawData.data.superseded_by`);
  }
  if (!Number.isSafeInteger(data.body_chars) || data.body_chars < 0) {
    throw new RequiredMigrationContractError(`${relativePath}: invalid rawData.data.body_chars`);
  }
  if (!Number.isSafeInteger(data.body_bytes) || data.body_bytes < 0) {
    throw new RequiredMigrationContractError(`${relativePath}: invalid rawData.data.body_bytes`);
  }

  const expected = value.expectedProjection;
  if (!expected || !Array.isArray(expected.chunks)) {
    throw new RequiredMigrationContractError(`${relativePath}: missing expected projection manifest`);
  }
  requireSha256(expected.markdown_sha256, 'expectedProjection.markdown_sha256', relativePath);
  requireSha256(expected.page_hash, 'expectedProjection.page_hash', relativePath);
  if (!Number.isSafeInteger(expected.chunk_count) || expected.chunk_count < 0 || expected.chunk_count !== expected.chunks.length) {
    throw new RequiredMigrationContractError(`${relativePath}: invalid expectedProjection.chunk_count`);
  }
  const validSources: ReadonlySet<string> = new Set(['compiled_truth', 'timeline', 'fenced_code', 'image_asset']);
  for (let i = 0; i < expected.chunks.length; i++) {
    const chunk = expected.chunks[i];
    if (chunk.chunk_index !== i || !validSources.has(chunk.chunk_source)) {
      throw new RequiredMigrationContractError(`${relativePath}: invalid ordered chunk identity at index ${i}`);
    }
    requireSha256(chunk.chunk_sha256, `expectedProjection.chunks[${i}].chunk_sha256`, relativePath);
  }
}

export async function resolveRequiredMigrationFileContract(
  opts: RequiredMigrationSyncOptions,
  relativePath: string,
): Promise<RequiredMigrationFileContract> {
  assertRequiredMigrationSyncOptions(opts);
  let contract: RequiredMigrationFileContract | undefined;
  try {
    contract = await opts.fileSidecarResolver(relativePath);
  } catch (error) {
    throw new RequiredMigrationContractError(
      `${relativePath}: resolver failed (${error instanceof Error ? error.message : String(error)})`,
    );
  }
  if (!contract) throw new RequiredMigrationContractError(`${relativePath}: resolver returned no contract`);
  validateRequiredMigrationFileContract(contract, relativePath);
  return contract;
}

export function assertRequiredMigrationProjection(
  contract: RequiredMigrationFileContract,
  relativePath: string,
  markdown: string,
  pageHash: string,
  chunks: ReadonlyArray<ChunkInput>,
): void {
  const expected = contract.expectedProjection;
  if (sha256Text(markdown) !== expected.markdown_sha256) {
    throw new RequiredMigrationContractError(`${relativePath}: post-transform Markdown projection mismatch`);
  }
  if (pageHash !== expected.page_hash) {
    throw new RequiredMigrationContractError(`${relativePath}: post-transform page projection mismatch`);
  }
  if (chunks.length !== expected.chunk_count) {
    throw new RequiredMigrationContractError(`${relativePath}: post-transform chunk count mismatch`);
  }
  for (let i = 0; i < chunks.length; i++) {
    const actual = chunks[i];
    const wanted = expected.chunks[i];
    if (actual.chunk_index !== wanted.chunk_index || actual.chunk_source !== wanted.chunk_source ||
        sha256Text(actual.chunk_text) !== wanted.chunk_sha256) {
      throw new RequiredMigrationContractError(`${relativePath}: post-transform ordered chunk projection mismatch at index ${i}`);
    }
  }
}

export async function assertRequiredMigrationContentSkipState(
  engine: BrainEngine,
  slug: string,
  sourceId: string,
  existing: { source_kind?: string | null; source_uri?: string | null; ingested_via?: string | null },
  contract: RequiredMigrationFileContract,
  relativePath: string,
): Promise<void> {
  const provenance = contract.pageProvenance;
  if (existing.source_kind !== provenance.source_kind || existing.source_uri !== provenance.source_uri ||
      existing.ingested_via !== provenance.ingested_via) {
    throw new RequiredMigrationContractError(`${relativePath}: content-skip page provenance mismatch`);
  }
  const sidecars = await engine.getRawData(slug, contract.rawData.source, { sourceId });
  if (sidecars.length !== 1 || canonicalJson(sidecars[0].data) !== canonicalJson(contract.rawData.data)) {
    throw new RequiredMigrationContractError(`${relativePath}: content-skip raw_data sidecar mismatch`);
  }
}
