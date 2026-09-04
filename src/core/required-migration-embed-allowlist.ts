import { AsyncLocalStorage } from 'node:async_hooks';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import type { ChunkInput, StaleChunkRow } from './types.ts';
import {
  REQUIRED_MIGRATION_EMBED_ALLOWLIST_MODE_ENV,
  REQUIRED_MIGRATION_PRIVACY_POLICY,
  isRequiredMigrationEmbedAllowlistModeRequired,
  runWithRequiredMigrationWireAttemptPairs,
} from './required-migration-wire-attempt.ts';

const SHA256_RE = /^[0-9a-f]{64}$/;

export const REQUIRED_EMBED_ALLOWLIST_ENV = {
  mode: REQUIRED_MIGRATION_EMBED_ALLOWLIST_MODE_ENV,
  path: 'GBRAIN_REQUIRED_MIGRATION_EMBED_ALLOWLIST_PATH',
  sha256: 'GBRAIN_REQUIRED_MIGRATION_EMBED_ALLOWLIST_SHA256',
} as const;

type StableChunkSource = ChunkInput['chunk_source'];
type RequiredMigrationEmbedRow = Omit<StaleChunkRow, 'chunk_source'> & {
  readonly chunk_source: StableChunkSource;
};

const STABLE_CHUNK_SOURCES: readonly StableChunkSource[] = [
  'compiled_truth',
  'timeline',
  'fenced_code',
  'image_asset',
];

function isStableChunkSource(value: unknown): value is StableChunkSource {
  return STABLE_CHUNK_SOURCES.includes(value as StableChunkSource);
}

export interface RequiredMigrationEmbedIdentity {
  readonly source_id: string;
  readonly page_id: number;
  readonly slug: string;
  readonly chunk_source: StableChunkSource;
  readonly chunk_index: number;
  readonly stored_text_sha256: string;
}

export interface RequiredMigrationEmbedPair {
  readonly text: string;
  readonly identity: RequiredMigrationEmbedIdentity;
}

export interface RequiredMigrationEmbedAllowlistEntry {
  readonly identity: {
    readonly source_id: string;
    readonly slug: string;
    readonly chunk_source: StableChunkSource;
    readonly chunk_index: number;
  };
  readonly stored_text_sha256: string;
  readonly provider_text_sha256: string;
  readonly provider_text_chars: number;
  readonly provider_text_bytes: number;
}

export type RequiredMigrationEmbedAllowlistFault =
  | 'context'
  | 'identity'
  | 'stored_hash'
  | 'provider_hash'
  | 'char'
  | 'byte'
  | 'missing'
  | 'extra'
  | 'duplicate'
  | 'truncation'
  | 'manifest';

export class RequiredMigrationEmbedAllowlistError extends Error {
  readonly code = 'required_migration_embed_allowlist';

  constructor(
    readonly fault: RequiredMigrationEmbedAllowlistFault,
    readonly identity?: string,
  ) {
    super(
      `[required-migration-embed] ${fault} fault` +
      (identity ? ` at ${identity}` : ''),
    );
    this.name = 'RequiredMigrationEmbedAllowlistError';
  }
}

export interface RequiredMigrationEmbedAllowlistRun {
  readonly mode: 'REQUIRED';
  readonly path: string;
  readonly sha256: string;
  readonly entries: ReadonlyMap<string, RequiredMigrationEmbedAllowlistEntry>;
  fault?: RequiredMigrationEmbedAllowlistError;
}

interface RequiredMigrationEmbedStore {
  readonly run: RequiredMigrationEmbedAllowlistRun;
  readonly pairs?: readonly RequiredMigrationEmbedPair[];
}

const store = new AsyncLocalStorage<RequiredMigrationEmbedStore>();

function sha256(text: string): string {
  return createHash('sha256').update(text, 'utf8').digest('hex');
}

function stableKey(identity: Pick<RequiredMigrationEmbedIdentity,
  'source_id' | 'slug' | 'chunk_source' | 'chunk_index'>): string {
  return JSON.stringify([
    identity.source_id,
    identity.slug,
    identity.chunk_source,
    identity.chunk_index,
  ]);
}

function identityLabel(identity: Pick<RequiredMigrationEmbedIdentity,
  'source_id' | 'page_id' | 'slug' | 'chunk_source' | 'chunk_index'>): string {
  return `${identity.source_id}/${identity.slug}/${identity.chunk_source}/${identity.chunk_index}` +
    `#page-${identity.page_id}`;
}

function fail(
  run: RequiredMigrationEmbedAllowlistRun,
  fault: RequiredMigrationEmbedAllowlistFault,
  identity?: Pick<RequiredMigrationEmbedIdentity,
    'source_id' | 'page_id' | 'slug' | 'chunk_source' | 'chunk_index'>,
): never {
  const error = new RequiredMigrationEmbedAllowlistError(
    fault,
    identity ? identityLabel(identity) : undefined,
  );
  run.fault ??= error;
  throw error;
}

function requireObject(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new RequiredMigrationEmbedAllowlistError('manifest');
  }
  return value as Record<string, unknown>;
}

function exactKeys(value: Record<string, unknown>, expected: readonly string[]): boolean {
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  return actual.length === wanted.length && actual.every((key, index) => key === wanted[index]);
}

function parseEntry(value: unknown): RequiredMigrationEmbedAllowlistEntry {
  const entry = requireObject(value);
  if (!exactKeys(entry, [
    'identity', 'stored_text_sha256', 'provider_text_sha256',
    'provider_text_chars', 'provider_text_bytes',
  ])) {
    throw new RequiredMigrationEmbedAllowlistError('manifest');
  }
  const identity = requireObject(entry.identity);
  if (!exactKeys(identity, ['source_id', 'slug', 'chunk_source', 'chunk_index'])) {
    throw new RequiredMigrationEmbedAllowlistError('manifest');
  }
  if (
    typeof identity.source_id !== 'string' || identity.source_id.length === 0 ||
    typeof identity.slug !== 'string' || identity.slug.length === 0 ||
    !isStableChunkSource(identity.chunk_source) ||
    !Number.isSafeInteger(identity.chunk_index) || Number(identity.chunk_index) < 0 ||
    typeof entry.stored_text_sha256 !== 'string' || !SHA256_RE.test(entry.stored_text_sha256) ||
    typeof entry.provider_text_sha256 !== 'string' || !SHA256_RE.test(entry.provider_text_sha256) ||
    !Number.isSafeInteger(entry.provider_text_chars) || Number(entry.provider_text_chars) < 0 ||
    !Number.isSafeInteger(entry.provider_text_bytes) || Number(entry.provider_text_bytes) < 0
  ) {
    throw new RequiredMigrationEmbedAllowlistError('manifest');
  }
  return Object.freeze({
    identity: Object.freeze({
      source_id: identity.source_id,
      slug: identity.slug,
      chunk_source: identity.chunk_source,
      chunk_index: Number(identity.chunk_index),
    }),
    stored_text_sha256: entry.stored_text_sha256,
    provider_text_sha256: entry.provider_text_sha256,
    provider_text_chars: Number(entry.provider_text_chars),
    provider_text_bytes: Number(entry.provider_text_bytes),
  });
}

export function loadRequiredMigrationEmbedAllowlistFromEnv(
  env: Record<string, string | undefined> = process.env,
): RequiredMigrationEmbedAllowlistRun | null {
  const mode = env[REQUIRED_EMBED_ALLOWLIST_ENV.mode];
  const artifactPath = env[REQUIRED_EMBED_ALLOWLIST_ENV.path];
  const expectedSha256 = env[REQUIRED_EMBED_ALLOWLIST_ENV.sha256];
  if (mode === undefined && artifactPath === undefined && expectedSha256 === undefined) return null;
  if (!isRequiredMigrationEmbedAllowlistModeRequired(env) || !artifactPath || !expectedSha256 || !SHA256_RE.test(expectedSha256)) {
    throw new RequiredMigrationEmbedAllowlistError('manifest');
  }

  const pinnedPath = resolve(artifactPath);
  const bytes = readFileSync(pinnedPath);
  const actualSha256 = createHash('sha256').update(bytes).digest('hex');
  if (actualSha256 !== expectedSha256) {
    throw new RequiredMigrationEmbedAllowlistError('manifest');
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(bytes.toString('utf8'));
  } catch {
    throw new RequiredMigrationEmbedAllowlistError('manifest');
  }
  const artifact = requireObject(parsed);
  if (!exactKeys(artifact, ['version', 'entries']) || artifact.version !== 1 || !Array.isArray(artifact.entries)) {
    throw new RequiredMigrationEmbedAllowlistError('manifest');
  }

  const entries = new Map<string, RequiredMigrationEmbedAllowlistEntry>();
  for (const rawEntry of artifact.entries) {
    const entry = parseEntry(rawEntry);
    const key = stableKey(entry.identity);
    if (entries.has(key)) throw new RequiredMigrationEmbedAllowlistError('duplicate');
    entries.set(key, entry);
  }
  return {
    mode: 'REQUIRED',
    path: pinnedPath,
    sha256: expectedSha256,
    entries,
  };
}

export async function runWithRequiredMigrationEmbedAllowlist<T>(
  run: RequiredMigrationEmbedAllowlistRun,
  fn: () => Promise<T>,
): Promise<T> {
  const result = await store.run({ run }, fn);
  if (run.fault) throw run.fault;
  return result;
}

/** True only inside the exact REQUIRED allowlist async context. */
export function isRequiredMigrationEmbedAllowlistRunActive(
  run: RequiredMigrationEmbedAllowlistRun,
): boolean {
  return store.getStore()?.run === run;
}

export function createRequiredMigrationEmbedPairs(
  rows: readonly RequiredMigrationEmbedRow[],
  texts: readonly string[],
): readonly RequiredMigrationEmbedPair[] | undefined {
  const active = store.getStore();
  if (!active) return undefined;
  if (rows.length < texts.length) fail(active.run, 'missing');
  if (rows.length > texts.length) fail(active.run, 'extra');

  const seen = new Set<string>();
  return Object.freeze(rows.map((row, index) => {
    const identity = Object.freeze({
      source_id: row.source_id,
      page_id: row.page_id,
      slug: row.slug,
      chunk_source: row.chunk_source,
      chunk_index: row.chunk_index,
      stored_text_sha256: sha256(row.chunk_text),
    });
    const key = stableKey(identity);
    if (seen.has(key)) fail(active.run, 'duplicate', identity);
    seen.add(key);
    return Object.freeze({ text: texts[index], identity });
  }));
}

export async function runWithRequiredMigrationEmbedPairs<T>(
  pairs: readonly RequiredMigrationEmbedPair[] | undefined,
  fn: () => Promise<T>,
): Promise<T> {
  const active = store.getStore();
  if (!active) return fn();
  if (!pairs) fail(active.run, 'context');
  if (active.run.fault) throw active.run.fault;
  return store.run({ run: active.run, pairs }, fn);
}

export async function runWithRequiredMigrationEmbedPairSlice<T>(
  texts: readonly string[],
  start: number,
  end: number,
  fn: () => Promise<T>,
): Promise<T> {
  const active = store.getStore();
  if (!active) return fn();
  const pairs = active.pairs;
  if (!pairs) fail(active.run, 'context');
  if (pairs.length < texts.length) fail(active.run, 'missing');
  if (pairs.length > texts.length) fail(active.run, 'extra');
  for (let i = 0; i < texts.length; i++) {
    if (pairs[i].text !== texts[i]) fail(active.run, 'identity', pairs[i].identity);
  }
  return store.run({ run: active.run, pairs: Object.freeze(pairs.slice(start, end)) }, fn);
}

function validatePairs(
  run: RequiredMigrationEmbedAllowlistRun,
  pairs: readonly RequiredMigrationEmbedPair[],
): void {
  const seen = new Set<string>();
  for (const pair of pairs) {
    const identity = pair.identity;
    if (
      !Number.isSafeInteger(identity.page_id) || identity.page_id <= 0 ||
      !Number.isSafeInteger(identity.chunk_index) || identity.chunk_index < 0
    ) {
      fail(run, 'identity', identity);
    }
    const key = stableKey(identity);
    if (seen.has(key)) fail(run, 'duplicate', identity);
    seen.add(key);
    const entry = run.entries.get(key);
    if (!entry) fail(run, 'missing', identity);
    if (identity.stored_text_sha256 !== entry.stored_text_sha256) fail(run, 'stored_hash', identity);
    if (sha256(pair.text) !== entry.provider_text_sha256) fail(run, 'provider_hash', identity);
    if (pair.text.length !== entry.provider_text_chars) fail(run, 'char', identity);
    if (Buffer.byteLength(pair.text, 'utf8') !== entry.provider_text_bytes) fail(run, 'byte', identity);
  }
}

export function bindRequiredMigrationProviderTexts(
  originalTexts: readonly string[],
  providerTexts: readonly string[],
): readonly RequiredMigrationEmbedPair[] | undefined {
  const active = store.getStore();
  if (!active) return undefined;
  const pairs = active.pairs;
  if (!pairs) fail(active.run, 'context');
  if (pairs.length < originalTexts.length) fail(active.run, 'missing');
  if (pairs.length > originalTexts.length) fail(active.run, 'extra');

  const bound = Object.freeze(pairs.map((pair, index) => {
    if (pair.text !== originalTexts[index]) fail(active.run, 'identity', pair.identity);
    if (providerTexts[index] !== originalTexts[index]) fail(active.run, 'truncation', pair.identity);
    return Object.freeze({ text: providerTexts[index], identity: pair.identity });
  }));
  validatePairs(active.run, bound);
  return bound;
}

export function assertRequiredMigrationTransportAttempt(
  texts: readonly string[],
  pairs: readonly RequiredMigrationEmbedPair[] | undefined,
): void {
  const active = store.getStore();
  if (!active) return;
  if (!pairs) fail(active.run, 'context');
  if (pairs.length < texts.length) fail(active.run, 'missing');
  if (pairs.length > texts.length) fail(active.run, 'extra');
  for (let i = 0; i < texts.length; i++) {
    if (pairs[i].text !== texts[i]) fail(active.run, 'identity', pairs[i].identity);
  }
  validatePairs(active.run, pairs);
}

export function runWithRequiredMigrationTransportAttempt<T>(
  texts: readonly string[],
  pairs: readonly RequiredMigrationEmbedPair[] | undefined,
  requiredOpenRouterModel: string | undefined,
  fn: () => T,
): T {
  if (isRequiredMigrationEmbedAllowlistModeRequired() && (!pairs || !requiredOpenRouterModel)) {
    throw new RequiredMigrationEmbedAllowlistError('context');
  }
  assertRequiredMigrationTransportAttempt(texts, pairs);
  return pairs && requiredOpenRouterModel
    ? runWithRequiredMigrationWireAttemptPairs(
      pairs,
      requiredOpenRouterModel,
      REQUIRED_MIGRATION_PRIVACY_POLICY,
      fn,
    )
    : fn();
}
