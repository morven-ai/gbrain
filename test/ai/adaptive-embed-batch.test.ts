/**
 * Tests for the adaptive embed batch system (PR #680, ships v0.28.7).
 *
 * Coverage matrix (per the eng-review plan):
 *
 *   1. Pure helpers exported from gateway.ts:
 *      - splitByTokenBudget pure-function semantics + chars_per_token threading
 *      - isTokenLimitError regex coverage
 *
 *   2. Recursion through public embed() with the AI-SDK transport stubbed.
 *      We do NOT call private functions; the test seam is the
 *      __setEmbedTransportForTests hook on the gateway.
 *
 *   3. Order preservation across recursive halving (left/right concat).
 *
 *   4. Terminal MIN_SUB_BATCH=1 — single text whose transport always fails
 *      must throw normalizeAIError, not loop forever.
 *
 *   5. OpenAI fast path (D3) — recipe with no max_batch_tokens calls the
 *      transport exactly once with no pre-split.
 *
 *   6. Shrink-on-miss adaptive cache (D8-A) — first miss halves the factor;
 *      after SHRINK_HEAL_AFTER successes the factor heals back toward the
 *      recipe-declared safety_factor.
 *
 *   7. Startup warning (D9-B) — gateway construction warns once for the
 *      configured embedding recipe when it is missing max_batch_tokens
 *      (excluding the OpenAI canonical fast-path recipe).
 */

import { afterAll, afterEach, beforeEach, describe, expect, mock, test } from 'bun:test';
import { createHash } from 'node:crypto';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  configureGateway,
  resetGateway,
  embed,
  splitByTokenBudget,
  capBatchItems,
  isTokenLimitError,
  __setEmbedTransportForTests,
  __getShrinkStateForTests,
} from '../../src/core/ai/gateway.ts';
import { AIConfigError, AITransientError } from '../../src/core/ai/errors.ts';
import { __setTestRecipesForTests } from '../../src/core/ai/recipes/index.ts';
import type { Recipe } from '../../src/core/ai/types.ts';
import { embedBatch } from '../../src/core/embedding.ts';
import {
  _setRateLimitFloorsForTests,
  embedBatchWithBackoff,
} from '../../src/core/embed-retry.ts';
import { __embedPageTextsForTests } from '../../src/commands/embed.ts';
import {
  REQUIRED_EMBED_ALLOWLIST_ENV,
  createRequiredMigrationEmbedPairs,
  loadRequiredMigrationEmbedAllowlistFromEnv,
  runWithRequiredMigrationEmbedAllowlist,
  runWithRequiredMigrationEmbedPairs,
  type RequiredMigrationEmbedAllowlistEntry,
  type RequiredMigrationEmbedAllowlistRun,
} from '../../src/core/required-migration-embed-allowlist.ts';
import type { StaleChunkRow } from '../../src/core/types.ts';

// The last test in this file leaves the gateway configured with a remote
// provider + fake key and a REAL embed transport. Without a final reset,
// that config leaks into whichever test file the shard runs next — the
// first downstream embed then makes a live HTTP call (broke master shard 6
// when #3022's new test file reshuffled shard composition). The bunfig
// legacy-embedding preload only re-applies its default when the gateway is
// UNCONFIGURED, so a configured-but-stale slot survives file boundaries.
afterAll(() => resetGateway());

// --------- Test helpers ---------

/**
 * Build an embedding-shape return for an arbitrary number of values. Each
 * embedding is `dims` floats, all set to a sentinel index so tests can
 * assert order preservation.
 */
function fakeEmbeddings(values: string[], dims: number): { embeddings: number[][] } {
  return {
    embeddings: values.map((_, i) =>
      // First slot encodes the input index so we can verify ordering.
      Array.from({ length: dims }, (_, j) => (j === 0 ? i : 0.1)),
    ),
  };
}

const VOYAGE_TOKEN_LIMIT_ERROR = new Error(
  "Request to model 'voyage-3-large' failed. The max allowed tokens per submitted batch is 120000.",
);

function configureVoyage(): void {
  configureGateway({
    embedding_model: 'voyage:voyage-3-large',
    embedding_dimensions: 1024,
    env: { VOYAGE_API_KEY: 'sk-fake' },
  });
}

function configureOpenAI(): void {
  configureGateway({
    embedding_model: 'openai:text-embedding-3-large',
    embedding_dimensions: 1536,
    env: { OPENAI_API_KEY: 'sk-fake' },
  });
}

function configureGoogle(): void {
  configureGateway({
    embedding_model: 'google:gemini-embedding-001',
    embedding_dimensions: 768,
    env: { GOOGLE_GENERATIVE_AI_API_KEY: 'fake' },
  });
}

// A recipe that declares an embedding touchpoint but omits every batch cap.
// Every shipped recipe now declares one (google gained max_batch_tokens), so
// the startup warning is exercised against this synthetic cap-less recipe —
// injected into the registry only for the duration of the test that needs it.
const CAPLESS_RECIPE: Recipe = {
  id: 'synthetic-capless',
  name: 'Synthetic cap-less (test fixture)',
  tier: 'openai-compat',
  implementation: 'openai-compatible',
  touchpoints: {
    embedding: {
      models: ['synthetic-embed-1'],
      default_dims: 768,
    },
  },
};

function configureCapless(): void {
  configureGateway({
    embedding_model: 'synthetic-capless:synthetic-embed-1',
    embedding_dimensions: 768,
    env: {},
  });
}

// --------- 1. Pure helpers ---------

describe('splitByTokenBudget (pure helper)', () => {
  test('single small text stays in one batch', () => {
    const result = splitByTokenBudget(['hello'], 120_000, 1);
    expect(result).toEqual([['hello']]);
  });

  test('texts fitting within budget stay in one batch', () => {
    const texts = Array.from({ length: 10 }, () => 'a'.repeat(1000));
    const result = splitByTokenBudget(texts, 96_000, 1);
    expect(result).toHaveLength(1);
    expect(result[0]).toHaveLength(10);
  });

  test('texts exceeding budget are split into multiple batches', () => {
    // chars_per_token=1, so each 50K-char text counts as 50K tokens.
    // Budget 96K → first text fits, second pushes over → new batch.
    const texts = ['a'.repeat(50_000), 'b'.repeat(50_000), 'c'.repeat(50_000)];
    const result = splitByTokenBudget(texts, 96_000, 1);
    expect(result).toHaveLength(3);
    expect(result.map(b => b.length)).toEqual([1, 1, 1]);
  });

  test('chars_per_token=4 (OpenAI density) packs 4× more chars per batch', () => {
    // Each 50K-char text = 12.5K tokens at chars_per_token=4. Budget 96K
    // tokens → 7 fit; the 8th would overflow into a new batch.
    const texts = Array.from({ length: 10 }, (_, i) => `${i}`.repeat(50_000));
    const result = splitByTokenBudget(texts, 96_000, 4);
    expect(result[0].length).toBe(7);
    expect(result[1].length).toBe(3);
  });

  test('default chars_per_token (4) when ratio omitted', () => {
    // Same payload as above without the explicit ratio.
    const texts = Array.from({ length: 10 }, (_, i) => `${i}`.repeat(50_000));
    const explicit = splitByTokenBudget(texts, 96_000, 4);
    const implicit = splitByTokenBudget(texts, 96_000);
    expect(implicit).toEqual(explicit);
  });

  test('empty input returns empty array', () => {
    expect(splitByTokenBudget([], 120_000, 1)).toEqual([]);
  });

  test('single text larger than budget still goes in a batch (split helper does not subdivide)', () => {
    const result = splitByTokenBudget(['a'.repeat(200_000)], 120_000, 1);
    expect(result).toHaveLength(1);
    expect(result[0]).toHaveLength(1);
  });

  test('zero or negative chars_per_token falls back to default', () => {
    const texts = ['a'.repeat(40_000)];
    expect(splitByTokenBudget(texts, 96_000, 0)).toEqual(splitByTokenBudget(texts, 96_000, 4));
    expect(splitByTokenBudget(texts, 96_000, -1)).toEqual(splitByTokenBudget(texts, 96_000, 4));
  });
});

describe('capBatchItems (hard COUNT cap helper)', () => {
  test('batch at or under the cap is returned as a single batch (no copy of contents)', () => {
    const texts = ['a', 'b', 'c'];
    expect(capBatchItems(texts, 3)).toEqual([texts]);
    expect(capBatchItems(texts, 10)).toEqual([texts]);
  });

  test('oversized batch splits into chunks of at most maxItems', () => {
    const texts = Array.from({ length: 100 }, (_, i) => `t${i}`);
    const result = capBatchItems(texts, 32);
    expect(result.map(b => b.length)).toEqual([32, 32, 32, 4]);
    expect(result.every(b => b.length <= 32)).toBe(true);
  });

  test('exact multiple splits evenly with no trailing empty batch', () => {
    const texts = Array.from({ length: 64 }, (_, i) => `t${i}`);
    expect(capBatchItems(texts, 32).map(b => b.length)).toEqual([32, 32]);
  });

  test('order is preserved across the split (concatenation round-trips)', () => {
    const texts = Array.from({ length: 70 }, (_, i) => `t${i}`);
    expect(capBatchItems(texts, 32).flat()).toEqual(texts);
  });

  test('maxItems <= 0 is a no-op (single batch) — never produces empty/infinite batches', () => {
    const texts = ['a', 'b', 'c'];
    expect(capBatchItems(texts, 0)).toEqual([texts]);
    expect(capBatchItems(texts, -5)).toEqual([texts]);
  });

  test('empty input returns a single empty batch', () => {
    expect(capBatchItems([], 32)).toEqual([[]]);
  });
});

describe('isTokenLimitError (pure helper)', () => {
  test('matches Voyage error format', () => {
    expect(isTokenLimitError(VOYAGE_TOKEN_LIMIT_ERROR)).toBe(true);
  });

  test('matches "token limit exceeded" variant', () => {
    expect(isTokenLimitError(new Error('Token limit exceeded for batch request'))).toBe(true);
  });

  test('matches "batch too many tokens" variant', () => {
    expect(isTokenLimitError(new Error('Batch contains too many tokens'))).toBe(true);
  });

  test('matches OpenAI embeddings "maximum request size" error (regression: PR ###)', () => {
    // Real error string returned by OpenAI's /v1/embeddings endpoint when the
    // sum of all input items exceeds 300k tokens. Without this match, gbrain's
    // recursive-halving safety net never engages on OpenAI and the queue stalls
    // forever on token-dense pages.
    const openaiErr = new Error(
      "Invalid 'input': maximum request size is 300000 tokens per request.",
    );
    expect(isTokenLimitError(openaiErr)).toBe(true);
  });

  test('matches generic "max tokens per request" phrasing', () => {
    expect(isTokenLimitError(new Error('Exceeded 300000 max tokens per request'))).toBe(true);
  });

  test('does not match unrelated errors', () => {
    expect(isTokenLimitError(new Error('Connection refused'))).toBe(false);
    expect(isTokenLimitError(new Error('Invalid API key'))).toBe(false);
    expect(isTokenLimitError(new Error('429 rate limited'))).toBe(false);
  });

  test('handles non-Error throwables', () => {
    expect(isTokenLimitError('Token limit exceeded')).toBe(true);
    expect(isTokenLimitError({ message: 'some other thing' })).toBe(false);
    expect(isTokenLimitError(null)).toBe(false);
    expect(isTokenLimitError(undefined)).toBe(false);
  });
});

// --------- 2-4. Recursion via embed() with stubbed transport ---------

describe('embed() recursion via stubbed transport', () => {
  beforeEach(() => resetGateway());
  afterEach(() => __setEmbedTransportForTests(null));

  test('halves on token-limit error and concatenates left+right in order', async () => {
    configureVoyage();

    const stub = mock(async ({ values }: { values: string[] }) => {
      // First call: full batch fails. Halved calls: succeed.
      if (values.length === 50) throw VOYAGE_TOKEN_LIMIT_ERROR;
      return fakeEmbeddings(values, 1024);
    });
    __setEmbedTransportForTests(stub as any);

    // Build 50 texts that each fit comfortably under any pre-split budget
    // (1 char ≈ 1 token in voyage's recipe; 0.5 × 120K = 60K char budget).
    const texts = Array.from({ length: 50 }, (_, i) => `t${i}`);
    const result = await embed(texts);

    // Stub fired 3 times: 1 fail (length 50) + 2 success (length 25 each).
    expect(stub).toHaveBeenCalledTimes(3);
    const callLengths = stub.mock.calls.map(([arg]) => (arg as { values: string[] }).values.length);
    expect(callLengths.sort((a, b) => a - b)).toEqual([25, 25, 50]);
    expect(result).toHaveLength(50);
  });

  test('preserves input order across halving boundaries', async () => {
    configureVoyage();

    const stub = mock(async ({ values }: { values: string[] }) => {
      if (values.length === 10) throw VOYAGE_TOKEN_LIMIT_ERROR;
      return fakeEmbeddings(values, 1024);
    });
    __setEmbedTransportForTests(stub as any);

    const texts = Array.from({ length: 10 }, (_, i) => String.fromCharCode(97 + i)); // a..j
    const result = await embed(texts);

    expect(result).toHaveLength(10);
    // The fakeEmbeddings helper encodes the within-call index in slot 0;
    // halved calls each receive sub-arrays of length 5, so slot 0 reads
    // [0,1,2,3,4,0,1,2,3,4] — that's the contract that proves order
    // preservation despite the embeddings being concatenated from two calls.
    const slotZero = result.map(v => v[0]);
    expect(slotZero).toEqual([0, 1, 2, 3, 4, 0, 1, 2, 3, 4]);
  });

  test('terminal case: single text always fails → normalizes and throws (no infinite loop)', async () => {
    configureVoyage();

    const stub = mock(async () => { throw VOYAGE_TOKEN_LIMIT_ERROR; });
    __setEmbedTransportForTests(stub as any);

    let caught: unknown = null;
    try {
      await embed(['just one text']);
    } catch (e) {
      caught = e;
    }
    expect(caught).not.toBeNull();
    expect(caught instanceof AIConfigError || caught instanceof AITransientError).toBe(true);
    // Stub fires once for the single-element batch; cannot halve further so
    // the recursion gives up at MIN_SUB_BATCH=1 and rethrows.
    expect(stub).toHaveBeenCalledTimes(1);
  });
});

// --------- 5. OpenAI fast path (D3) ---------

describe('embed() OpenAI fast path (no max_batch_tokens)', () => {
  beforeEach(() => resetGateway());
  afterEach(() => __setEmbedTransportForTests(null));

  test('recipe without max_batch_tokens calls transport exactly once with no partition', async () => {
    configureOpenAI();

    const stub = mock(async ({ values }: { values: string[] }) => fakeEmbeddings(values, 1536));
    __setEmbedTransportForTests(stub as any);

    const texts = Array.from({ length: 100 }, (_, i) => `text-${i}`);
    const result = await embed(texts);

    expect(stub).toHaveBeenCalledTimes(1);
    const callValues = (stub.mock.calls[0][0] as { values: string[] }).values;
    expect(callValues).toEqual(texts);
    expect(result).toHaveLength(100);
  });

  test('OpenAI fast path is unaffected by Voyage shrink state', async () => {
    // Configure Voyage first and trigger a shrink…
    configureVoyage();
    const voyageStub = mock(async ({ values }: { values: string[] }) => {
      if (values.length === 4) throw VOYAGE_TOKEN_LIMIT_ERROR;
      return fakeEmbeddings(values, 1024);
    });
    __setEmbedTransportForTests(voyageStub as any);
    await embed(['a', 'b', 'c', 'd']);
    expect(__getShrinkStateForTests('voyage')?.factor).toBe(0.25);

    // …then reconfigure to OpenAI. The shrink state belongs to the prior
    // gateway's lifecycle and must not leak.
    configureOpenAI();
    const openaiStub = mock(async ({ values }: { values: string[] }) => fakeEmbeddings(values, 1536));
    __setEmbedTransportForTests(openaiStub as any);
    await embed(['x', 'y']);
    expect(openaiStub).toHaveBeenCalledTimes(1);
    expect(__getShrinkStateForTests('voyage')).toBeUndefined();
  });
});

// --------- 6. Shrink-on-miss adaptive cache (D8-A) ---------

describe('shrink-on-miss adaptive cache', () => {
  beforeEach(() => resetGateway());
  afterEach(() => __setEmbedTransportForTests(null));

  test('first token-limit miss halves the recipe safety factor', async () => {
    configureVoyage();
    expect(__getShrinkStateForTests('voyage')).toBeUndefined();

    const stub = mock(async ({ values }: { values: string[] }) => {
      if (values.length === 4) throw VOYAGE_TOKEN_LIMIT_ERROR;
      return fakeEmbeddings(values, 1024);
    });
    __setEmbedTransportForTests(stub as any);

    await embed(['a', 'b', 'c', 'd']);
    // Voyage declares safety_factor=0.5; after one miss → 0.5 × 0.5 = 0.25.
    expect(__getShrinkStateForTests('voyage')?.factor).toBe(0.25);
  });

  test('factor floors at SHRINK_FLOOR (0.05) under repeated misses', async () => {
    configureVoyage();
    const stub = mock(async ({ values }: { values: string[] }) => {
      // Always throw on >1 to keep recursion going until MIN_SUB_BATCH=1
      // succeeds. That gives many shrink events per embed() call.
      if (values.length > 1) throw VOYAGE_TOKEN_LIMIT_ERROR;
      return fakeEmbeddings(values, 1024);
    });
    __setEmbedTransportForTests(stub as any);

    // 16 texts will recurse 4 levels deep, generating multiple shrink events.
    await embed(Array.from({ length: 16 }, (_, i) => `t${i}`));
    const factor = __getShrinkStateForTests('voyage')?.factor ?? -1;
    expect(factor).toBeGreaterThanOrEqual(0.05);
  });

  test('factor heals back toward declared safety_factor after enough wins', async () => {
    configureVoyage();
    const stub = mock(async ({ values }: { values: string[] }) => {
      // Once: fail at length 2, succeed everywhere else. Subsequent calls
      // all succeed.
      if (stub.mock.calls.length === 1 && values.length === 2) throw VOYAGE_TOKEN_LIMIT_ERROR;
      return fakeEmbeddings(values, 1024);
    });
    __setEmbedTransportForTests(stub as any);

    await embed(['a', 'b']); // 1 fail + 2 successes (length 1 each) → factor 0.25, wins 2
    const afterMiss = __getShrinkStateForTests('voyage')?.factor;
    expect(afterMiss).toBe(0.25);

    // Drive 10 more successful calls. SHRINK_HEAL_AFTER=10; on the 10th win
    // the factor multiplies by 1.5 (capped at the declared 0.5 ceiling).
    for (let i = 0; i < 8; i++) {
      await embed(['solo']);
    }
    const healed = __getShrinkStateForTests('voyage')?.factor ?? 0;
    // 0.25 × 1.5 = 0.375. Still below the recipe ceiling of 0.5; the next
    // round of 10 wins would bump it to min(0.5, 0.375 × 1.5) = 0.5.
    expect(healed).toBeCloseTo(0.375, 5);
  });

  test('healing path cannot exceed the recipe-declared safety_factor', async () => {
    configureVoyage();
    const stub = mock(async ({ values }: { values: string[] }) => {
      if (stub.mock.calls.length === 1) throw VOYAGE_TOKEN_LIMIT_ERROR;
      return fakeEmbeddings(values, 1024);
    });
    __setEmbedTransportForTests(stub as any);

    // Trigger one shrink, then drive enough wins to fully heal.
    await embed(['one', 'two']);
    for (let i = 0; i < 30; i++) {
      await embed(['solo']);
    }
    const factor = __getShrinkStateForTests('voyage')?.factor ?? 0;
    // Declared safety_factor is 0.5; healing must clamp at that ceiling.
    expect(factor).toBeLessThanOrEqual(0.5);
    expect(factor).toBeGreaterThan(0);
  });
});

// --------- 7. Startup warning (D9-B) ---------

describe('startup warning for recipes missing max_batch_tokens', () => {
  beforeEach(() => resetGateway());

  test('configured missing-cap recipe warns once; unrelated recipes stay quiet', () => {
    __setTestRecipesForTests([CAPLESS_RECIPE]);
    const warnings: string[] = [];
    const original = console.warn;
    console.warn = (msg: string) => warnings.push(String(msg));
    try {
      configureOpenAI();
      expect(warnings.length).toBe(0);
      configureCapless();
      const firstCallCount = warnings.length;
      // Reconfigure: the warning should NOT re-fire for the same recipe
      // within one process (we already told the operator).
      configureCapless();
      expect(warnings.length).toBe(firstCallCount);
    } finally {
      console.warn = original;
      __setTestRecipesForTests([]);
    }

    // The warning text should match the documented contract.
    const contractMatch = warnings.filter(w =>
      w.includes('[ai.gateway]') && w.includes('declares an embedding touchpoint'),
    );
    expect(contractMatch.length).toBe(1);

    // Voyage + google declare max_batch_tokens → suppressed. OpenAI is the
    // canonical fast-path recipe → also suppressed by id. Only the synthetic
    // cap-less recipe warns.
    expect(warnings.find(w => w.includes('"voyage"'))).toBeUndefined();
    expect(warnings.find(w => w.includes('"openai"'))).toBeUndefined();
    expect(warnings.find(w => w.includes('"google"'))).toBeUndefined();
    expect(warnings.find(w => w.includes('"synthetic-capless"'))).toBeDefined();
  });
});

// --------- 8. REQUIRED migration provider-boundary allowlist ---------

function digest(text: string): string {
  return createHash('sha256').update(text, 'utf8').digest('hex');
}

function staleRows(texts: readonly string[]): StaleChunkRow[] {
  return texts.map((text, index) => ({
    slug: 'wiki/required-example',
    chunk_index: index,
    chunk_text: text,
    chunk_source: 'compiled_truth',
    model: null,
    token_count: null,
    source_id: 'required-source',
    page_id: 1001,
  }));
}

function allowlistEntries(
  rows: readonly StaleChunkRow[],
  providerTexts: readonly string[],
): RequiredMigrationEmbedAllowlistEntry[] {
  return rows.map((row, index) => ({
    identity: {
      source_id: row.source_id,
      slug: row.slug,
      chunk_source: row.chunk_source,
      chunk_index: row.chunk_index,
    },
    stored_text_sha256: digest(row.chunk_text),
    provider_text_sha256: digest(providerTexts[index]),
    provider_text_chars: providerTexts[index].length,
    provider_text_bytes: Buffer.byteLength(providerTexts[index], 'utf8'),
  }));
}

function loadAllowlist(entries: readonly RequiredMigrationEmbedAllowlistEntry[]): {
  run: RequiredMigrationEmbedAllowlistRun;
  cleanup: () => void;
} {
  const dir = mkdtempSync(join(tmpdir(), 'gbrain-required-allowlist-'));
  const path = join(dir, 'allowlist.json');
  const raw = JSON.stringify({ version: 1, entries });
  writeFileSync(path, raw, { mode: 0o600 });
  try {
    const run = loadRequiredMigrationEmbedAllowlistFromEnv({
      [REQUIRED_EMBED_ALLOWLIST_ENV.mode]: 'REQUIRED',
      [REQUIRED_EMBED_ALLOWLIST_ENV.path]: path,
      [REQUIRED_EMBED_ALLOWLIST_ENV.sha256]: digest(raw),
    });
    if (!run) throw new Error('required allowlist fixture did not load');
    return { run, cleanup: () => rmSync(dir, { recursive: true, force: true }) };
  } catch (error) {
    rmSync(dir, { recursive: true, force: true });
    throw error;
  }
}

async function withRequiredPairs<T>(
  rows: readonly StaleChunkRow[],
  pairTexts: readonly string[],
  entries: readonly RequiredMigrationEmbedAllowlistEntry[],
  fn: () => Promise<T>,
): Promise<T> {
  const fixture = loadAllowlist(entries);
  try {
    return await runWithRequiredMigrationEmbedAllowlist(fixture.run, async () => {
      const pairs = createRequiredMigrationEmbedPairs(rows, pairTexts);
      return runWithRequiredMigrationEmbedPairs(pairs, fn);
    });
  } finally {
    fixture.cleanup();
  }
}

describe('REQUIRED migration immutable text/identity pairs', () => {
  beforeEach(() => {
    resetGateway();
    configureVoyage();
  });

  afterEach(() => {
    _setRateLimitFloorsForTests(null);
    __setEmbedTransportForTests(null);
  });

  test('pins the artifact bytes and rejects a manifest hash mismatch before transport', () => {
    const dir = mkdtempSync(join(tmpdir(), 'gbrain-required-allowlist-bad-'));
    const path = join(dir, 'allowlist.json');
    const raw = JSON.stringify({ version: 1, entries: [] });
    writeFileSync(path, raw, { mode: 0o600 });
    try {
      expect(() => loadRequiredMigrationEmbedAllowlistFromEnv({
        [REQUIRED_EMBED_ALLOWLIST_ENV.mode]: 'REQUIRED',
        [REQUIRED_EMBED_ALLOWLIST_ENV.path]: path,
        [REQUIRED_EMBED_ALLOWLIST_ENV.sha256]: '0'.repeat(64),
      })).toThrow('manifest fault');
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  test('accepts fenced_code manifest entries and reaches transport', async () => {
    const text = 'const answer = 42;';
    const rows = staleRows([text]).map((row) => ({
      ...row,
      chunk_source: 'fenced_code',
    })) as unknown as StaleChunkRow[];
    const transport = mock(async ({ values }: { values: string[] }) => fakeEmbeddings(values, 1024));
    __setEmbedTransportForTests(transport as any);

    const result = await withRequiredPairs(rows, [text], allowlistEntries(rows, [text]),
      () => embedBatch([text], { maxRetries: 0 }));

    expect(result).toHaveLength(1);
    expect(transport).toHaveBeenCalledTimes(1);
    expect((transport.mock.calls[0][0] as { values: string[] }).values).toEqual([text]);
  });

  test('rejects unsupported chunk_source as a manifest fault before transport', () => {
    const text = 'unsupported-source-payload';
    const rows = staleRows([text]);
    const [valid] = allowlistEntries(rows, [text]);
    const entries = [{
      ...valid,
      identity: { ...valid.identity, chunk_source: 'unsupported_source' },
    }] as unknown as RequiredMigrationEmbedAllowlistEntry[];
    const transport = mock(async ({ values }: { values: string[] }) => fakeEmbeddings(values, 1024));
    __setEmbedTransportForTests(transport as any);

    expect(() => loadAllowlist(entries)).toThrow('manifest fault');
    expect(transport).toHaveBeenCalledTimes(0);
  });

  test('context/hash/char/byte/missing/extra/duplicate/identity/truncation faults make zero transport calls', async () => {
    const transport = mock(async ({ values }: { values: string[] }) => fakeEmbeddings(values, 1024));
    __setEmbedTransportForTests(transport as any);

    const baseText = 'sensitive-payload-never-log';
    const rows = staleRows([baseText]);
    const valid = allowlistEntries(rows, [baseText]);
    const cases: Array<() => Promise<unknown>> = [
      async () => {
        const fixture = loadAllowlist(valid);
        try {
          await runWithRequiredMigrationEmbedAllowlist(fixture.run, () => embedBatch([baseText], { maxRetries: 0 }));
        } finally { fixture.cleanup(); }
      },
      () => withRequiredPairs(rows, [baseText], [{ ...valid[0], stored_text_sha256: '0'.repeat(64) }],
        () => embedBatch([baseText], { maxRetries: 0 })),
      () => withRequiredPairs(rows, [baseText], [{ ...valid[0], provider_text_sha256: '0'.repeat(64) }],
        () => embedBatch([baseText], { maxRetries: 0 })),
      () => withRequiredPairs(rows, [baseText], [{ ...valid[0], provider_text_chars: baseText.length + 1 }],
        () => embedBatch([baseText], { maxRetries: 0 })),
      () => withRequiredPairs(rows, [baseText], [{ ...valid[0], provider_text_bytes: Buffer.byteLength(baseText) + 1 }],
        () => embedBatch([baseText], { maxRetries: 0 })),
      () => withRequiredPairs(rows, [baseText], [], () => embedBatch([baseText], { maxRetries: 0 })),
      async () => {
        const twoTexts = ['first', 'second'];
        const twoRows = staleRows(twoTexts);
        return withRequiredPairs(twoRows, twoTexts, allowlistEntries(twoRows, twoTexts),
          () => embedBatch([twoTexts[0]], { maxRetries: 0 }));
      },
      async () => {
        const fixture = loadAllowlist([...valid, valid[0]]);
        fixture.cleanup();
      },
      async () => {
        const dupRows = [rows[0], rows[0]];
        return withRequiredPairs(dupRows, [baseText, baseText], valid,
          () => embedBatch([baseText, baseText], { maxRetries: 0 }));
      },
      () => withRequiredPairs(rows, [baseText], valid,
        () => embedBatch(['different-provider-text'], { maxRetries: 0 })),
      async () => {
        const longText = 'x'.repeat(8_001);
        const longRows = staleRows([longText]);
        const truncated = longText.slice(0, 8_000);
        return withRequiredPairs(longRows, [longText], allowlistEntries(longRows, [truncated]),
          () => embedBatch([longText], { maxRetries: 0 }));
      },
    ];

    for (const runCase of cases) {
      const callsBefore = transport.mock.calls.length;
      let message = '';
      try {
        await runCase();
        throw new Error('expected REQUIRED allowlist fault');
      } catch (error) {
        message = error instanceof Error ? error.message : String(error);
      }
      expect(transport.mock.calls.length).toBe(callsBefore);
      expect(message).not.toContain(baseText);
    }
  });

  test('100-item pagination slices identity pairs in the same order', async () => {
    const texts = Array.from({ length: 101 }, (_, index) => `payload-${index}`);
    const rows = staleRows(texts);
    const transport = mock(async ({ values }: { values: string[] }) => fakeEmbeddings(values, 1024));
    __setEmbedTransportForTests(transport as any);

    const result = await withRequiredPairs(rows, texts, allowlistEntries(rows, texts),
      () => embedBatch([...texts], { maxRetries: 0, onBatchComplete: () => {} }));

    expect(result).toHaveLength(101);
    expect(transport.mock.calls.map(([arg]) => (arg as { values: string[] }).values.length)).toEqual([100, 1]);
    expect(transport.mock.calls.flatMap(([arg]) => (arg as { values: string[] }).values)).toEqual(texts);
  });

  test('recursive token split and bounded retry revalidate the same immutable pairs', async () => {
    const texts = ['a', 'b', 'c', 'd'];
    const rows = staleRows(texts);
    let retryAttempt = 0;
    const transport = mock(async ({ values }: { values: string[] }) => {
      if (values.length === 4) throw VOYAGE_TOKEN_LIMIT_ERROR;
      if (values.length === 2 && values[0] === 'a' && retryAttempt++ === 0) {
        const error = new Error('429 rate limited; please try again in 0ms') as Error & { status: number };
        error.status = 429;
        throw error;
      }
      return fakeEmbeddings(values, 1024);
    });
    __setEmbedTransportForTests(transport as any);
    _setRateLimitFloorsForTests([1]);

    const result = await withRequiredPairs(rows, texts, allowlistEntries(rows, texts),
      () => embedBatchWithBackoff([...texts]));

    expect(result).toHaveLength(4);
    expect(transport.mock.calls.map(([arg]) => (arg as { values: string[] }).values)).toEqual([
      texts,
      ['a', 'b'],
      texts,
      ['a', 'b'],
      ['c', 'd'],
    ]);
  });

  test('permanent-error single fanout carries exactly one matching pair per call', async () => {
    const texts = ['fanout-a', 'fanout-b'];
    const rows = staleRows(texts);
    const transport = mock(async ({ values }: { values: string[] }) => {
      if (values.length > 1) throw new AIConfigError('permanent request fault');
      return fakeEmbeddings(values, 1024);
    });
    __setEmbedTransportForTests(transport as any);

    const fixture = loadAllowlist(allowlistEntries(rows, texts));
    try {
      const result = await runWithRequiredMigrationEmbedAllowlist(fixture.run, async () => {
        const pairs = createRequiredMigrationEmbedPairs(rows, texts);
        return __embedPageTextsForTests([...texts], {}, pairs);
      });
      expect(result.failed).toBe(0);
      expect(transport.mock.calls.map(([arg]) => (arg as { values: string[] }).values)).toEqual([
        texts,
        ['fanout-a'],
        ['fanout-b'],
      ]);
    } finally {
      fixture.cleanup();
    }
  });
});
