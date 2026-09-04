import { createHash } from 'node:crypto';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, beforeAll, beforeEach, describe, expect, test } from 'bun:test';

import { runEmbedCore } from '../src/commands/embed.ts';
import {
  __setEmbedTransportForTests,
  configureGateway,
  resetGateway,
} from '../src/core/ai/gateway.ts';
import { OPENROUTER_PRIVACY_STRICT_ENV } from '../src/core/ai/recipes/openrouter.ts';
import { embedBackfillLockId } from '../src/core/embed-backfill-lock.ts';
import { syncLockId, tryAcquireDbLock, type DbLockHandle } from '../src/core/db-lock.ts';
import { PGLiteEngine } from '../src/core/pglite-engine.ts';
import { REQUIRED_EMBED_ALLOWLIST_ENV } from '../src/core/required-migration-embed-allowlist.ts';
import { RequiredMigrationVectorCasMismatchError } from '../src/core/required-migration-vector-cas.ts';
import {
  runWithRequiredMigrationWireAttemptHook,
  type RequiredMigrationNativeAttemptHook,
} from '../src/core/required-migration-wire-attempt.ts';
import { resetPgliteState } from './helpers/reset-pglite.ts';

const MODEL = 'openrouter:google/gemini-embedding-001';
const DIMS = 1024;
const SOURCE_ID = 'default';
const SLUG = 'required/page';
const CHUNK_TEXT = 'exact required migration chunk';

let engine: PGLiteEngine;
let foreignEngine: PGLiteEngine;
let tmpHome: string;
let allowlistPath: string;
let pageId = 0;
let providerCalls = 0;
const savedEnv: Record<string, string | undefined> = {};
const wireAttemptHook: RequiredMigrationNativeAttemptHook = {
  attemptStarted() {},
  attemptFinished() {},
};

function sha256(value: string | Buffer): string {
  return createHash('sha256').update(value).digest('hex');
}

function installTransport(impl?: () => unknown): void {
  __setEmbedTransportForTests(async ({ values }: { values: string[] }) => {
    providerCalls++;
    if (impl) return impl() as never;
    return {
      embeddings: values.map(() => new Array(DIMS).fill(0.25)),
      usage: { tokens: values.length },
    } as never;
  });
}

function allowlistEntry(slug: string, text: string, chunkIndex = 0) {
  return {
    identity: {
      source_id: SOURCE_ID,
      slug,
      chunk_source: 'compiled_truth',
      chunk_index: chunkIndex,
    },
    stored_text_sha256: sha256(text),
    provider_text_sha256: sha256(text),
    provider_text_chars: text.length,
    provider_text_bytes: Buffer.byteLength(text, 'utf8'),
  };
}

function writeAllowlist(entries = [allowlistEntry(SLUG, CHUNK_TEXT)]): void {
  const bytes = Buffer.from(JSON.stringify({
    version: 1,
    entries,
  }));
  writeFileSync(allowlistPath, bytes);
  process.env[REQUIRED_EMBED_ALLOWLIST_ENV.mode] = 'REQUIRED';
  process.env[REQUIRED_EMBED_ALLOWLIST_ENV.path] = allowlistPath;
  process.env[REQUIRED_EMBED_ALLOWLIST_ENV.sha256] = sha256(bytes);
}

async function acquireLocks(target = engine): Promise<[DbLockHandle, DbLockHandle]> {
  const sync = await tryAcquireDbLock(target, syncLockId(SOURCE_ID), 60);
  const embed = await tryAcquireDbLock(target, embedBackfillLockId(SOURCE_ID), 60);
  if (!sync || !embed) throw new Error('test lock acquisition failed');
  return [sync, embed];
}

function requiredOpts(heldLocks: DbLockHandle[]) {
  return {
    stale: true,
    sourceId: SOURCE_ID,
    singleFlight: true,
    dryRun: false,
    quiet: true,
    heldLocks,
  } as const;
}

function runRequiredEmbedCore(
  target: PGLiteEngine,
  opts: Parameters<typeof runEmbedCore>[1],
) {
  return runWithRequiredMigrationWireAttemptHook(
    wireAttemptHook,
    () => runEmbedCore(target, opts),
  );
}

async function expectAdmissionRefusal(heldLocks: DbLockHandle[]): Promise<void> {
  await expect(runRequiredEmbedCore(engine, requiredOpts(heldLocks))).rejects.toThrow(/required-migration-(embed|wire-attempt)/);
  expect(providerCalls).toBe(0);
}

beforeAll(async () => {
  for (const key of [
    'GBRAIN_HOME', 'GBRAIN_EMBEDDING_MODEL', 'GBRAIN_EMBEDDING_DIMENSIONS',
    'OPENROUTER_API_KEY', 'OPENROUTER_BASE_URL', OPENROUTER_PRIVACY_STRICT_ENV, 'DATABASE_URL',
    ...Object.values(REQUIRED_EMBED_ALLOWLIST_ENV),
  ]) {
    savedEnv[key] = process.env[key];
    delete process.env[key];
  }
  tmpHome = mkdtempSync(join(tmpdir(), 'gbrain-required-embed-run-'));
  allowlistPath = join(tmpHome, 'allowlist.json');
  process.env.GBRAIN_HOME = tmpHome;
  process.env.OPENROUTER_API_KEY = 'or-test-fake';
  process.env[OPENROUTER_PRIVACY_STRICT_ENV] = '1';
  mkdirSync(join(tmpHome, '.gbrain'), { recursive: true });
  writeFileSync(join(tmpHome, '.gbrain', 'config.json'), JSON.stringify({
    engine: 'pglite',
    embedding_model: MODEL,
    embedding_dimensions: DIMS,
    openrouter_api_key: 'or-test-fake',
  }));
  resetGateway();
  configureGateway({
    embedding_model: MODEL,
    embedding_dimensions: DIMS,
    env: {
      OPENROUTER_API_KEY: 'or-test-fake',
      [OPENROUTER_PRIVACY_STRICT_ENV]: '1',
    },
  });
  engine = new PGLiteEngine();
  await engine.connect({ embedding_dimensions: DIMS } as never);
  await engine.initSchema();
  foreignEngine = new PGLiteEngine();
  await foreignEngine.connect({ embedding_dimensions: DIMS } as never);
  await foreignEngine.initSchema();
}, 60_000);

afterAll(async () => {
  __setEmbedTransportForTests(null);
  resetGateway();
  await engine.disconnect();
  await foreignEngine.disconnect();
  rmSync(tmpHome, { recursive: true, force: true });
  for (const [key, value] of Object.entries(savedEnv)) {
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
});

beforeEach(async () => {
  process.env[OPENROUTER_PRIVACY_STRICT_ENV] = '1';
  delete process.env.OPENROUTER_BASE_URL;
  configureGateway({
    embedding_model: MODEL,
    embedding_dimensions: DIMS,
    env: {
      OPENROUTER_API_KEY: 'or-test-fake',
      [OPENROUTER_PRIVACY_STRICT_ENV]: '1',
    },
  });
  await resetPgliteState(engine);
  await resetPgliteState(foreignEngine);
  await engine.putPage(SLUG, {
    type: 'note',
    title: 'Required page',
    compiled_truth: '# Required page\n\nBody must stay unchanged.',
    timeline: 'timeline-stable',
    frontmatter: { migration: 'fixture' },
  });
  await engine.upsertChunks(SLUG, [{
    chunk_index: 0,
    chunk_text: CHUNK_TEXT,
    chunk_source: 'compiled_truth',
    token_count: 7,
    language: 'markdown',
    symbol_name: 'stable-symbol',
    symbol_type: 'section',
    start_line: 2,
    end_line: 4,
    parent_symbol_path: ['root', 'stable'],
    doc_comment: 'stable-doc',
    symbol_name_qualified: 'root.stable-symbol',
    modality: 'text',
  }]);
  const rows = await engine.executeRaw<{ id: number }>(
    `SELECT id FROM pages WHERE source_id = $1 AND slug = $2`,
    [SOURCE_ID, SLUG],
  );
  pageId = Number(rows[0]?.id);
  providerCalls = 0;
  writeAllowlist();
  installTransport();
});

describe('REQUIRED embed exact dual-lock admission', () => {
  test('missing active native-attempt hook refuses before provider', async () => {
    const locks = await acquireLocks();
    try {
      await expect(runEmbedCore(engine, requiredOpts(locks))).rejects.toThrow(/required-migration-embed/);
      expect(providerCalls).toBe(0);
    } finally {
      await locks[1].release();
      await locks[0].release();
    }
  });

  test('exact stale/source/singleFlight/dryRun/target shape refuses every near miss before provider', async () => {
    const locks = await acquireLocks();
    const base = requiredOpts(locks);
    const invalid = [
      { ...base, stale: false },
      { ...base, sourceId: 'wrong-source' },
      { ...base, singleFlight: false },
      { ...base, dryRun: undefined },
      { ...base, all: false },
      { ...base, slug: '' },
      { ...base, slugs: [] },
    ];
    try {
      for (const opts of invalid) {
        await expect(runRequiredEmbedCore(engine, opts)).rejects.toThrow(/required-migration-embed/);
      }
      expect(providerCalls).toBe(0);
    } finally {
      await locks[1].release();
      await locks[0].release();
    }
  });

  test('model, dimensions, strict privacy, and base URL are hard-pinned before provider', async () => {
    const locks = await acquireLocks();
    const restoreTarget = () => {
      process.env[OPENROUTER_PRIVACY_STRICT_ENV] = '1';
      delete process.env.OPENROUTER_BASE_URL;
      configureGateway({
        embedding_model: MODEL,
        embedding_dimensions: DIMS,
        env: {
          OPENROUTER_API_KEY: 'or-test-fake',
          [OPENROUTER_PRIVACY_STRICT_ENV]: '1',
        },
      });
    };
    try {
      configureGateway({
        embedding_model: 'openrouter:google/text-embedding-004',
        embedding_dimensions: DIMS,
        env: {
          OPENROUTER_API_KEY: 'or-test-fake',
          [OPENROUTER_PRIVACY_STRICT_ENV]: '1',
        },
      });
      await expectAdmissionRefusal(locks);
      restoreTarget();

      configureGateway({
        embedding_model: MODEL,
        embedding_dimensions: DIMS + 1,
        env: {
          OPENROUTER_API_KEY: 'or-test-fake',
          [OPENROUTER_PRIVACY_STRICT_ENV]: '1',
        },
      });
      await expectAdmissionRefusal(locks);
      restoreTarget();

      delete process.env[OPENROUTER_PRIVACY_STRICT_ENV];
      await expectAdmissionRefusal(locks);
      restoreTarget();

      process.env.OPENROUTER_BASE_URL = 'https://example.invalid/api/v1';
      await expectAdmissionRefusal(locks);
    } finally {
      restoreTarget();
      await locks[1].release();
      await locks[0].release();
    }
  });

  test('missing or extra handles refuse before provider', async () => {
    const [sync, embed] = await acquireLocks();
    const extra = await tryAcquireDbLock(engine, 'gbrain-test-extra', 60);
    if (!extra) throw new Error('extra lock acquisition failed');
    try {
      await expectAdmissionRefusal([sync]);
      await expectAdmissionRefusal([sync, embed, extra]);
    } finally {
      await extra.release();
      await embed.release();
      await sync.release();
    }
  });

  test('wrong-key, foreign-engine, and released handles refuse before provider', async () => {
    const sync = await tryAcquireDbLock(engine, syncLockId(SOURCE_ID), 60);
    const wrong = await tryAcquireDbLock(engine, embedBackfillLockId('wrong-source'), 60);
    if (!sync || !wrong) throw new Error('wrong-key lock acquisition failed');
    try {
      await expectAdmissionRefusal([sync, wrong]);
    } finally {
      await wrong.release();
      await sync.release();
    }

    const foreign = await acquireLocks(foreignEngine);
    try {
      await expectAdmissionRefusal(foreign);
    } finally {
      await foreign[1].release();
      await foreign[0].release();
    }

    const released = await acquireLocks();
    await released[0].release();
    try {
      await expectAdmissionRefusal(released);
    } finally {
      await released[1].release();
    }
  });
});

describe('REQUIRED embed provider-to-CAS path', () => {
  test('malformed active embedding-column config refuses before provider', async () => {
    await engine.executeRaw(
      `INSERT INTO config (key, value) VALUES ('search_embedding_column', 'missing-required-column')
       ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value`,
    );
    const locks = await acquireLocks();
    try {
      await expect(runRequiredEmbedCore(engine, requiredOpts(locks))).rejects.toThrow();
      expect(providerCalls).toBe(0);
    } finally {
      await engine.executeRaw(`DELETE FROM config WHERE key = 'search_embedding_column'`);
      await locks[1].release();
      await locks[0].release();
    }
  });

  test('successful run uses CAS only and preserves page/chunk metadata', async () => {
    const before = await engine.executeRaw<Record<string, unknown>>(
      `SELECT p.compiled_truth, p.frontmatter, p.timeline, p.embedding_signature,
              p.contextual_retrieval_mode, c.page_id, c.chunk_index, c.chunk_source,
              c.chunk_text, c.token_count, c.language, c.symbol_name, c.symbol_type,
              c.start_line, c.end_line, c.parent_symbol_path, c.doc_comment,
              c.symbol_name_qualified, c.modality
         FROM content_chunks c JOIN pages p ON p.id = c.page_id
        WHERE c.page_id = $1 AND c.chunk_index = 0`,
      [pageId],
    );
    const calls = { getChunks: 0, upsertChunks: 0, signature: 0, restamp: 0, cas: 0 };
    const mutable = engine as unknown as Record<string, unknown>;
    const originalCas = engine.updateRequiredMigrationChunkEmbeddings.bind(engine);
    mutable.getChunks = async () => { calls.getChunks++; return []; };
    mutable.upsertChunks = async () => { calls.upsertChunks++; };
    mutable.setPageEmbeddingSignature = async () => { calls.signature++; };
    mutable.updatePageContextualRetrievalState = async () => { calls.restamp++; };
    mutable.updateRequiredMigrationChunkEmbeddings = async (...args: Parameters<typeof originalCas>) => {
      calls.cas++;
      return originalCas(...args);
    };
    const locks = await acquireLocks();
    try {
      const result = await runRequiredEmbedCore(engine, requiredOpts(locks));
      expect(result.embedded).toBe(1);
      expect(providerCalls).toBe(1);
      expect(calls).toEqual({ getChunks: 0, upsertChunks: 0, signature: 0, restamp: 0, cas: 1 });
      const after = await engine.executeRaw<Record<string, unknown>>(
        `SELECT p.compiled_truth, p.frontmatter, p.timeline, p.embedding_signature,
                p.contextual_retrieval_mode, c.page_id, c.chunk_index, c.chunk_source,
                c.chunk_text, c.token_count, c.language, c.symbol_name, c.symbol_type,
                c.start_line, c.end_line, c.parent_symbol_path, c.doc_comment,
                c.symbol_name_qualified, c.modality
           FROM content_chunks c JOIN pages p ON p.id = c.page_id
          WHERE c.page_id = $1 AND c.chunk_index = 0`,
        [pageId],
      );
      expect(after).toEqual(before);
      const vector = await engine.executeRaw<{ model: string; embedding_is_null: boolean }>(
        `SELECT model, (embedding IS NULL) AS embedding_is_null
           FROM content_chunks WHERE page_id = $1 AND chunk_index = 0`,
        [pageId],
      );
      expect(vector[0]?.model).toBe(MODEL);
      expect(vector[0]?.embedding_is_null).toBe(false);
    } finally {
      await locks[1].release();
      await locks[0].release();
      delete mutable.getChunks;
      delete mutable.upsertChunks;
      delete mutable.setPageEmbeddingSignature;
      delete mutable.updatePageContextualRetrievalState;
      delete mutable.updateRequiredMigrationChunkEmbeddings;
    }
  });

  test('multiple page contexts combine into one ordered provider batch and one atomic CAS', async () => {
    const fixtures = [
      { slug: SLUG, text: CHUNK_TEXT },
      { slug: 'required/page-b', text: 'second exact required chunk' },
      { slug: 'required/page-c', text: 'third exact required chunk' },
    ];
    for (const fixture of fixtures.slice(1)) {
      await engine.putPage(fixture.slug, {
        type: 'note',
        title: fixture.slug,
        compiled_truth: fixture.text,
        timeline: '',
        frontmatter: { migration: 'fixture' },
      });
      await engine.upsertChunks(fixture.slug, [{
        chunk_index: 0,
        chunk_text: fixture.text,
        chunk_source: 'compiled_truth',
        token_count: 7,
      }]);
    }
    writeAllowlist(fixtures.map(fixture => allowlistEntry(fixture.slug, fixture.text)));

    const providerBatches: string[][] = [];
    __setEmbedTransportForTests(async ({ values }: { values: string[] }) => {
      providerCalls++;
      providerBatches.push([...values]);
      return {
        embeddings: values.map(() => new Array(DIMS).fill(0.5)),
        usage: { tokens: values.length },
      } as never;
    });

    const mutable = engine as unknown as Record<string, unknown>;
    const originalCas = engine.updateRequiredMigrationChunkEmbeddings.bind(engine);
    const casBatchSizes: number[] = [];
    mutable.updateRequiredMigrationChunkEmbeddings = async (
      input: Parameters<typeof originalCas>[0],
    ) => {
      casBatchSizes.push(input.rows.length);
      return originalCas(input);
    };

    const locks = await acquireLocks();
    try {
      const result = await runRequiredEmbedCore(engine, requiredOpts(locks));
      expect(result.embedded).toBe(3);
      expect(result.pages_processed).toBe(3);
      expect(providerCalls).toBe(1);
      expect(providerBatches).toEqual([fixtures.map(fixture => fixture.text)]);
      expect(casBatchSizes).toEqual([3]);
    } finally {
      await locks[1].release();
      await locks[0].release();
      delete mutable.updateRequiredMigrationChunkEmbeddings;
    }
  });

  test('runner clamps oversized batch requests to ordered 100-chunk provider and CAS batches', async () => {
    const texts = Array.from({ length: 101 }, (_, index) => `required chunk ${index}`);
    await engine.upsertChunks(SLUG, texts.map((text, index) => ({
      chunk_index: index,
      chunk_text: text,
      chunk_source: 'compiled_truth' as const,
      token_count: 4,
    })));
    writeAllowlist(texts.map((text, index) => allowlistEntry(SLUG, text, index)));

    const providerBatchSizes: number[] = [];
    __setEmbedTransportForTests(async ({ values }: { values: string[] }) => {
      providerCalls++;
      providerBatchSizes.push(values.length);
      return {
        embeddings: values.map(() => new Array(DIMS).fill(0.75)),
        usage: { tokens: values.length },
      } as never;
    });

    const mutable = engine as unknown as Record<string, unknown>;
    const originalCas = engine.updateRequiredMigrationChunkEmbeddings.bind(engine);
    const casBatchSizes: number[] = [];
    mutable.updateRequiredMigrationChunkEmbeddings = async (
      input: Parameters<typeof originalCas>[0],
    ) => {
      casBatchSizes.push(input.rows.length);
      return originalCas(input);
    };

    const locks = await acquireLocks();
    try {
      const result = await runRequiredEmbedCore(engine, {
        ...requiredOpts(locks),
        batchSize: 1000,
      });
      expect(result.embedded).toBe(101);
      expect(providerCalls).toBe(2);
      expect(providerBatchSizes).toEqual([100, 1]);
      expect(casBatchSizes).toEqual([100, 1]);
    } finally {
      await locks[1].release();
      await locks[0].release();
      delete mutable.updateRequiredMigrationChunkEmbeddings;
    }
  });

  test('preflight refuses an extra database chunk before provider transport', async () => {
    await engine.upsertChunks(SLUG, [
      {
        chunk_index: 0,
        chunk_text: CHUNK_TEXT,
        chunk_source: 'compiled_truth',
        token_count: 7,
      },
      {
        chunk_index: 1,
        chunk_text: 'not present in the reviewed allowlist',
        chunk_source: 'compiled_truth',
        token_count: 7,
      },
    ]);
    const locks = await acquireLocks();
    try {
      await expect(runRequiredEmbedCore(engine, requiredOpts(locks))).rejects.toThrow(/required-migration-embed/);
      expect(providerCalls).toBe(0);
    } finally {
      await locks[1].release();
      await locks[0].release();
    }
  });

  test('preflight refuses stored-text drift before provider transport', async () => {
    await engine.executeRaw(
      `UPDATE content_chunks SET chunk_text = 'drifted before provider' WHERE page_id = $1 AND chunk_index = 0`,
      [pageId],
    );
    const locks = await acquireLocks();
    try {
      await expect(runRequiredEmbedCore(engine, requiredOpts(locks))).rejects.toThrow(/required-migration-embed/);
      expect(providerCalls).toBe(0);
    } finally {
      await locks[1].release();
      await locks[0].release();
    }
  });

  test('provider success is not terminal success without persisted exact vectors', async () => {
    const mutable = engine as unknown as Record<string, unknown>;
    mutable.updateRequiredMigrationChunkEmbeddings = async () => 1;
    const locks = await acquireLocks();
    try {
      await expect(runRequiredEmbedCore(engine, requiredOpts(locks))).rejects.toThrow(/required-migration-embed/);
      expect(providerCalls).toBe(1);
      expect(await engine.countStaleChunks({ sourceId: SOURCE_ID })).toBe(1);
    } finally {
      await locks[1].release();
      await locks[0].release();
      delete mutable.updateRequiredMigrationChunkEmbeddings;
    }
  });

  test('resume classifies an exact persisted vector as satisfied and sends nothing', async () => {
    const firstLocks = await acquireLocks();
    try {
      expect((await runRequiredEmbedCore(engine, requiredOpts(firstLocks))).embedded).toBe(1);
      expect(providerCalls).toBe(1);
    } finally {
      await firstLocks[1].release();
      await firstLocks[0].release();
    }

    providerCalls = 0;
    const resumeLocks = await acquireLocks();
    try {
      const result = await runRequiredEmbedCore(engine, requiredOpts(resumeLocks));
      expect(result.embedded).toBe(0);
      expect(providerCalls).toBe(0);
      expect(await engine.countStaleChunks({ sourceId: SOURCE_ID })).toBe(0);
    } finally {
      await resumeLocks[1].release();
      await resumeLocks[0].release();
    }
  });

  test.each([
    ['partial', () => ({ embeddings: [], usage: { tokens: 0 } })],
    ['failed', () => { throw new Error('provider failed'); }],
    ['null', () => ({ embeddings: [null], usage: { tokens: 0 } })],
  ])('provider %s result throws with CAS 0', async (_kind, response) => {
    installTransport(response);
    let casCalls = 0;
    const mutable = engine as unknown as Record<string, unknown>;
    mutable.updateRequiredMigrationChunkEmbeddings = async () => { casCalls++; return 1; };
    const locks = await acquireLocks();
    try {
      await expect(runRequiredEmbedCore(engine, requiredOpts(locks))).rejects.toThrow();
      expect(providerCalls).toBe(1);
      expect(casCalls).toBe(0);
      expect(await engine.countStaleChunks({ sourceId: SOURCE_ID })).toBe(1);
    } finally {
      await locks[1].release();
      await locks[0].release();
      delete mutable.updateRequiredMigrationChunkEmbeddings;
    }
  });

  test.each(['false', 'throw'] as const)('post-provider lock refresh %s throws with CAS 0', async (mode) => {
    const locks = await acquireLocks();
    const originalRefresh = locks[0].refresh;
    let refreshCalls = 0;
    locks[0].refresh = async () => {
      refreshCalls++;
      if (refreshCalls === 1) return originalRefresh();
      if (mode === 'false') return false;
      throw new Error('refresh failed');
    };
    let casCalls = 0;
    const mutable = engine as unknown as Record<string, unknown>;
    mutable.updateRequiredMigrationChunkEmbeddings = async () => { casCalls++; return 1; };
    try {
      await expect(runRequiredEmbedCore(engine, requiredOpts(locks))).rejects.toThrow(/required-migration-embed/);
      expect(providerCalls).toBe(1);
      expect(casCalls).toBe(0);
      expect(await engine.countStaleChunks({ sourceId: SOURCE_ID })).toBe(1);
    } finally {
      await locks[1].release();
      await locks[0].release();
      delete mutable.updateRequiredMigrationChunkEmbeddings;
    }
  });

  test('CAS mismatch after provider updates zero vector rows atomically', async () => {
    const locks = await acquireLocks();
    const originalRefresh = locks[0].refresh;
    let refreshCalls = 0;
    locks[0].refresh = async () => {
      refreshCalls++;
      if (refreshCalls === 2) {
        await engine.executeRaw(
          `UPDATE content_chunks SET chunk_text = 'raced text' WHERE page_id = $1 AND chunk_index = 0`,
          [pageId],
        );
      }
      return originalRefresh();
    };
    try {
      await expect(runRequiredEmbedCore(engine, requiredOpts(locks)))
        .rejects.toBeInstanceOf(RequiredMigrationVectorCasMismatchError);
      expect(providerCalls).toBe(1);
      const rows = await engine.executeRaw<{ embedding_is_null: boolean }>(
        `SELECT (embedding IS NULL) AS embedding_is_null
           FROM content_chunks WHERE page_id = $1 AND chunk_index = 0`,
        [pageId],
      );
      expect(rows[0]?.embedding_is_null).toBe(true);
    } finally {
      await locks[1].release();
      await locks[0].release();
    }
  });
});
