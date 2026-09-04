import { AsyncLocalStorage } from 'node:async_hooks';
import { createHash } from 'node:crypto';

import type { RequiredMigrationEmbedPair } from './required-migration-embed-allowlist.ts';

export const REQUIRED_MIGRATION_EMBED_ALLOWLIST_MODE_ENV = 'GBRAIN_REQUIRED_MIGRATION_EMBED_ALLOWLIST_MODE';
export const REQUIRED_MIGRATION_PRIVACY_POLICY = 'openrouter_zdr_data_denial' as const;

export type RequiredMigrationPrivacyPolicy = typeof REQUIRED_MIGRATION_PRIVACY_POLICY;

export interface RequiredMigrationNativeAttemptSummary {
  readonly sourceId: string;
  readonly orderedIdentitySetSha256: string;
  readonly itemCount: number;
  readonly utf16CharCount: number;
  readonly utf8ByteCount: number;
  readonly model: string;
  readonly privacyPolicy: RequiredMigrationPrivacyPolicy;
}

export type RequiredMigrationNativeAttemptFinish =
  | { readonly outcome: 'response'; readonly status: number }
  | { readonly outcome: 'transport_failure' };

export interface RequiredMigrationNativeAttemptHook {
  attemptStarted(summary: RequiredMigrationNativeAttemptSummary): void | Promise<void>;
  attemptFinished(
    summary: RequiredMigrationNativeAttemptSummary,
    finish: RequiredMigrationNativeAttemptFinish,
  ): void | Promise<void>;
}

export type RequiredMigrationWireAttemptFault =
  | 'context'
  | 'identity'
  | 'input'
  | 'model'
  | 'privacy';

export class RequiredMigrationWireAttemptError extends Error {
  readonly code = 'required_migration_wire_attempt';

  constructor(readonly fault: RequiredMigrationWireAttemptFault) {
    super(`[required-migration-wire-attempt] ${fault} fault`);
    this.name = 'RequiredMigrationWireAttemptError';
  }
}

interface RequiredMigrationWireAttemptStore {
  readonly hook?: RequiredMigrationNativeAttemptHook;
  readonly pairs?: readonly RequiredMigrationEmbedPair[];
  readonly model?: string;
  readonly privacyPolicy?: RequiredMigrationPrivacyPolicy;
}

const store = new AsyncLocalStorage<RequiredMigrationWireAttemptStore>();

function fail(fault: RequiredMigrationWireAttemptFault): never {
  throw new RequiredMigrationWireAttemptError(fault);
}

/** The sole activation anchor for REQUIRED migration wire enforcement. */
export function isRequiredMigrationEmbedAllowlistModeRequired(
  env: Record<string, string | undefined> = process.env,
): boolean {
  return env[REQUIRED_MIGRATION_EMBED_ALLOWLIST_MODE_ENV] === 'REQUIRED';
}

export function runWithRequiredMigrationWireAttemptHook<T>(
  hook: RequiredMigrationNativeAttemptHook,
  fn: () => T,
): T {
  return store.run({ hook }, fn);
}

export function getActiveRequiredMigrationWireAttemptHook(): RequiredMigrationNativeAttemptHook | undefined {
  return store.getStore()?.hook;
}

export function assertRequiredMigrationWireAttemptHookActive(
  expected?: RequiredMigrationNativeAttemptHook,
): RequiredMigrationNativeAttemptHook {
  const hook = getActiveRequiredMigrationWireAttemptHook();
  if (!hook || (expected && hook !== expected)) fail('context');
  return hook;
}

export function runWithRequiredMigrationWireAttemptPairs<T>(
  pairs: readonly RequiredMigrationEmbedPair[],
  model: string,
  privacyPolicy: RequiredMigrationPrivacyPolicy,
  fn: () => T,
): T {
  const active = store.getStore();
  return store.run({ hook: active?.hook, pairs, model, privacyPolicy }, fn);
}

function orderedIdentitySetSha256(pairs: readonly RequiredMigrationEmbedPair[]): string {
  const identities = pairs.map(pair => [
    pair.identity.source_id,
    pair.identity.page_id,
    pair.identity.slug,
    pair.identity.chunk_source,
    pair.identity.chunk_index,
    pair.identity.stored_text_sha256,
  ]);
  return createHash('sha256').update(JSON.stringify(identities), 'utf8').digest('hex');
}

function summarizeRequiredAttempt(
  body: Record<string, unknown>,
  active: RequiredMigrationWireAttemptStore,
): RequiredMigrationNativeAttemptSummary {
  const hook = active.hook;
  const pairs = active.pairs;
  if (!hook || !pairs || !active.model || !active.privacyPolicy || pairs.length === 0) fail('context');

  const inputs = body.input;
  if (!Array.isArray(inputs) || inputs.some(input => typeof input !== 'string')) fail('input');
  if (inputs.length !== pairs.length) fail('input');
  for (let index = 0; index < inputs.length; index++) {
    if (inputs[index] !== pairs[index].text) fail('input');
  }
  const providerModel = active.model.startsWith('openrouter:')
    ? active.model.slice('openrouter:'.length)
    : active.model;
  if (body.model !== providerModel) fail('model');

  const provider = body.provider;
  if (!provider || typeof provider !== 'object' || Array.isArray(provider)) fail('privacy');
  const privacy = provider as Record<string, unknown>;
  if (privacy.zdr !== true || privacy.data_collection !== 'deny') fail('privacy');

  const sourceIds = new Set(pairs.map(pair => pair.identity.source_id));
  if (sourceIds.size !== 1) fail('identity');
  const texts = inputs as string[];
  return Object.freeze({
    sourceId: sourceIds.values().next().value as string,
    orderedIdentitySetSha256: orderedIdentitySetSha256(pairs),
    itemCount: texts.length,
    utf16CharCount: texts.reduce((total, text) => total + text.length, 0),
    utf8ByteCount: texts.reduce((total, text) => total + Buffer.byteLength(text, 'utf8'), 0),
    model: active.model,
    privacyPolicy: active.privacyPolicy,
  });
}

export async function runRequiredMigrationNativeAttempt(
  body: unknown,
  nativeFetch: () => Promise<Response>,
): Promise<Response> {
  const active = store.getStore();
  if (!active?.hook && !active?.pairs) {
    if (isRequiredMigrationEmbedAllowlistModeRequired()) fail('context');
    return nativeFetch();
  }
  if (!body || typeof body !== 'object' || Array.isArray(body)) fail('input');

  const summary = summarizeRequiredAttempt(body as Record<string, unknown>, active);
  const hook = assertRequiredMigrationWireAttemptHookActive(active.hook);
  await hook.attemptStarted(summary);

  let response: Response;
  try {
    response = await nativeFetch();
  } catch (error) {
    await hook.attemptFinished(summary, { outcome: 'transport_failure' });
    throw error;
  }
  await hook.attemptFinished(summary, { outcome: 'response', status: response.status });
  return response;
}
