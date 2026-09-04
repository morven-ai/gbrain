import { createHash } from 'node:crypto';
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, test } from 'bun:test';

import { PGLiteEngine } from '../src/core/pglite-engine.ts';
import { tryAcquireDbLock, type DbLockHandle } from '../src/core/db-lock.ts';
import {
  RequiredMigrationVectorCasInputError,
  RequiredMigrationVectorCasMismatchError,
} from '../src/core/required-migration-vector-cas.ts';
import type { RequiredMigrationVectorCasRow } from '../src/core/required-migration-vector-cas.ts';
import type { ResolvedColumn } from '../src/core/types.ts';
import { resetPgliteState } from './helpers/reset-pglite.ts';

let engine: PGLiteEngine;
let embeddingColumn: ResolvedColumn;
let migrationEmbeddingColumn: ResolvedColumn;
let casLocks: [DbLockHandle, DbLockHandle];

const CAS_LOCK_IDS = ['gbrain-sync:cas-fixture', 'gbrain-embed-backfill:cas-fixture'] as const;

function lockFences(locks: readonly DbLockHandle[] = casLocks) {
  return locks.map(lock => ({ id: lock.id, acquired_at: lock.acquiredAt }));
}

function sha256(text: string): string {
  return createHash('sha256').update(text, 'utf8').digest('hex');
}

function md5(text: string): string {
  return createHash('md5').update(text, 'utf8').digest('hex');
}

beforeAll(async () => {
  engine = new PGLiteEngine();
  await engine.connect({});
  await engine.initSchema();
  const rows = await engine.executeRaw<{ dimensions: number }>(
    `SELECT atttypmod::int AS dimensions
       FROM pg_attribute
      WHERE attrelid = 'content_chunks'::regclass
        AND attname = 'embedding'
        AND attnum > 0
        AND NOT attisdropped`,
  );
  embeddingColumn = {
    name: 'embedding',
    type: 'vector',
    dimensions: Number(rows[0]?.dimensions),
    embeddingModel: 'openrouter:google/gemini-embedding-001',
  };
  await engine.executeRaw(
    `ALTER TABLE content_chunks ADD COLUMN IF NOT EXISTS embedding_required_migration vector(${embeddingColumn.dimensions})`,
  );
  migrationEmbeddingColumn = {
    ...embeddingColumn,
    name: 'embedding_required_migration',
  };
}, 30_000);

afterAll(async () => {
  await engine.disconnect();
});

beforeEach(async () => {
  await resetPgliteState(engine);
  const sync = await tryAcquireDbLock(engine, CAS_LOCK_IDS[0], 60);
  const embed = await tryAcquireDbLock(engine, CAS_LOCK_IDS[1], 60);
  if (!sync || !embed) throw new Error('failed to acquire CAS fixture locks');
  casLocks = [sync, embed];
});

afterEach(async () => {
  await Promise.all(casLocks.map(lock => lock.release()));
});

async function seedChunk(slug: string, text: string, chunkIndex = 0): Promise<number> {
  await engine.putPage(slug, {
    type: 'note',
    title: slug,
    compiled_truth: `# ${slug}\n\nPage body must stay unchanged.`,
    frontmatter: { migration: 'fixture' },
    timeline: 'timeline-stable',
  });
  await engine.upsertChunks(slug, [{
    chunk_index: chunkIndex,
    chunk_text: text,
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
    `SELECT id FROM pages WHERE slug = $1 AND source_id = 'default'`,
    [slug],
  );
  return Number(rows[0]?.id);
}

const CAS_FAULTS: Array<[string, (row: RequiredMigrationVectorCasRow) => void]> = [
  ['page_id', (row) => { row.page_id = 999_999; }],
  ['chunk_index', (row) => { row.chunk_index = 9; }],
  ['chunk_source', (row) => { row.chunk_source = 'timeline'; }],
  ['exact text', (row) => {
    row.expected_text = 'different but internally valid exact text';
    row.expected_text_sha256 = sha256(row.expected_text);
  }],
];

describe('required migration batch vector CAS — PGLite', () => {
  test('happy path updates only vector provenance fields', async () => {
    const text = 'Reviewed exact chunk text.';
    const pageId = await seedChunk('migration/happy', text);
    const before = await engine.executeRaw<Record<string, unknown>>(
      `SELECT p.compiled_truth, p.frontmatter, p.timeline,
              c.page_id, c.chunk_index, c.chunk_source, c.chunk_text,
              c.token_count, c.language, c.symbol_name, c.symbol_type,
              c.start_line, c.end_line, c.parent_symbol_path, c.doc_comment,
              c.symbol_name_qualified, c.modality
         FROM content_chunks c
         JOIN pages p ON p.id = c.page_id
        WHERE c.page_id = $1 AND c.chunk_index = 0`,
      [pageId],
    );

    const updated = await engine.updateRequiredMigrationChunkEmbeddings({
      embeddingColumn,
      model: 'openrouter:google/gemini-embedding-001',
      locks: lockFences(),
      rows: [{
        page_id: pageId,
        chunk_index: 0,
        chunk_source: 'compiled_truth',
        expected_text_sha256: sha256(text),
        expected_text: text,
        embedding: new Float32Array(embeddingColumn.dimensions).fill(0.25),
      }],
    });

    expect(updated).toBe(1);
    const after = await engine.executeRaw<Record<string, unknown>>(
      `SELECT p.compiled_truth, p.frontmatter, p.timeline,
              c.page_id, c.chunk_index, c.chunk_source, c.chunk_text,
              c.token_count, c.language, c.symbol_name, c.symbol_type,
              c.start_line, c.end_line, c.parent_symbol_path, c.doc_comment,
              c.symbol_name_qualified, c.modality
         FROM content_chunks c
         JOIN pages p ON p.id = c.page_id
        WHERE c.page_id = $1 AND c.chunk_index = 0`,
      [pageId],
    );
    expect(after).toEqual(before);

    const stamps = await engine.executeRaw<{
      model: string;
      embedded_at: Date | string | null;
      embedded_text_hash: string | null;
      dimensions: number;
    }>(
      `SELECT model, embedded_at, embedded_text_hash,
              vector_dims(embedding)::int AS dimensions
         FROM content_chunks
        WHERE page_id = $1 AND chunk_index = 0`,
      [pageId],
    );
    expect(stamps[0]?.model).toBe('openrouter:google/gemini-embedding-001');
    expect(stamps[0]?.embedded_at).not.toBeNull();
    expect(stamps[0]?.embedded_text_hash).toBe(md5(text));
    expect(Number(stamps[0]?.dimensions)).toBe(embeddingColumn.dimensions);
    expect(await engine.hasCompletePageEmbeddingProvenance(
      'migration/happy',
      { model: 'openrouter:google/gemini-embedding-001' },
    )).toBe(true);
  });

  test.each(CAS_FAULTS)('refuses CAS fault: %s and leaves embedding fields NULL', async (_name, mutate) => {
    const text = 'CAS fault target.';
    const pageId = await seedChunk(`migration/fault-${String(_name).replace(/\s+/g, '-')}`, text);
    const row: RequiredMigrationVectorCasRow = {
      page_id: pageId,
      chunk_index: 0,
      chunk_source: 'compiled_truth',
      expected_text_sha256: sha256(text),
      expected_text: text,
      embedding: new Float32Array(embeddingColumn.dimensions).fill(0.5),
    };
    mutate(row);
    const before = await engine.executeRaw<{
      embedding_is_null: boolean;
      model: string | null;
      embedded_at: Date | string | null;
      embedded_text_hash: string | null;
    }>(
      `SELECT (embedding IS NULL) AS embedding_is_null, model, embedded_at, embedded_text_hash
         FROM content_chunks WHERE page_id = $1 AND chunk_index = 0`,
      [pageId],
    );

    const error = await engine.updateRequiredMigrationChunkEmbeddings({
      embeddingColumn,
      model: 'openrouter:google/gemini-embedding-001',
      locks: lockFences(),
      rows: [row],
    }).catch((caught) => caught);
    expect(error).toBeInstanceOf(RequiredMigrationVectorCasMismatchError);
    expect(error.updatedRows).toBe(0);

    const rows = await engine.executeRaw<{
      embedding_is_null: boolean;
      model: string | null;
      embedded_at: Date | string | null;
      embedded_text_hash: string | null;
    }>(
      `SELECT (embedding IS NULL) AS embedding_is_null, model, embedded_at, embedded_text_hash
         FROM content_chunks WHERE page_id = $1 AND chunk_index = 0`,
      [pageId],
    );
    expect(rows).toEqual(before);
  });

  test('rejects an invalid expected hash before touching the database', async () => {
    const text = 'Hash validation target.';
    const pageId = await seedChunk('migration/hash-fault', text);
    const error = await engine.updateRequiredMigrationChunkEmbeddings({
      embeddingColumn,
      model: 'openrouter:google/gemini-embedding-001',
      locks: lockFences(),
      rows: [{
        page_id: pageId,
        chunk_index: 0,
        chunk_source: 'compiled_truth',
        expected_text_sha256: sha256('not the exact text'),
        expected_text: text,
        embedding: new Float32Array(embeddingColumn.dimensions).fill(0.5),
      }],
    }).catch((caught) => caught);
    expect(error).toBeInstanceOf(RequiredMigrationVectorCasInputError);
    expect(await engine.countStaleChunks()).toBe(1);
  });

  test('an expired lock fence refuses the whole CAS batch', async () => {
    const text = 'Expired lock fence target.';
    const pageId = await seedChunk('migration/expired-lock-fence', text);
    await engine.executeRaw(
      `UPDATE gbrain_cycle_locks SET ttl_expires_at = NOW() - INTERVAL '1 second' WHERE id = $1`,
      [casLocks[0].id],
    );

    const error = await engine.updateRequiredMigrationChunkEmbeddings({
      embeddingColumn,
      model: 'openrouter:google/gemini-embedding-001',
      locks: lockFences(),
      rows: [{
        page_id: pageId,
        chunk_index: 0,
        chunk_source: 'compiled_truth',
        expected_text_sha256: sha256(text),
        expected_text: text,
        embedding: new Float32Array(embeddingColumn.dimensions).fill(0.5),
      }],
    }).catch((caught) => caught);

    expect(error).toBeInstanceOf(RequiredMigrationVectorCasMismatchError);
    expect(error.updatedRows).toBe(0);
    expect(await engine.countStaleChunks()).toBe(1);
  });

  test('a successor acquisition makes the prior fence unable to write', async () => {
    const text = 'Successor lock fence target.';
    const pageId = await seedChunk('migration/successor-lock-fence', text);
    const priorFences = lockFences();
    await engine.executeRaw(
      `UPDATE gbrain_cycle_locks
          SET ttl_expires_at = NOW() - INTERVAL '2 hours',
              last_refreshed_at = NOW() - INTERVAL '2 hours'
        WHERE id = $1`,
      [casLocks[0].id],
    );
    const successor = await tryAcquireDbLock(engine, casLocks[0].id, 60);
    if (!successor) throw new Error('failed to acquire successor CAS fixture lock');

    try {
      expect(successor.acquiredAt).not.toBe(casLocks[0].acquiredAt);
      const error = await engine.updateRequiredMigrationChunkEmbeddings({
        embeddingColumn,
        model: 'openrouter:google/gemini-embedding-001',
        locks: priorFences,
        rows: [{
          page_id: pageId,
          chunk_index: 0,
          chunk_source: 'compiled_truth',
          expected_text_sha256: sha256(text),
          expected_text: text,
          embedding: new Float32Array(embeddingColumn.dimensions).fill(0.5),
        }],
      }).catch((caught) => caught);

      expect(error).toBeInstanceOf(RequiredMigrationVectorCasMismatchError);
      expect(error.updatedRows).toBe(0);
      expect(await engine.countStaleChunks()).toBe(1);
    } finally {
      await successor.release();
    }
  });

  test('existing active embedding refuses replacement', async () => {
    const text = 'Already embedded target.';
    const pageId = await seedChunk('migration/existing-vector', text);
    await engine.executeRaw(
      `UPDATE content_chunks
          SET embedding = ('[' || array_to_string(array_fill(0.75::real, ARRAY[$1::int]), ',') || ']')::vector,
              model = 'existing:model', embedded_at = '2026-01-01T00:00:00Z', embedded_text_hash = 'existing-hash'
        WHERE page_id = $2 AND chunk_index = 0`,
      [embeddingColumn.dimensions, pageId],
    );

    await expect(engine.updateRequiredMigrationChunkEmbeddings({
      embeddingColumn,
      model: 'openrouter:google/gemini-embedding-001',
      locks: lockFences(),
      rows: [{
        page_id: pageId,
        chunk_index: 0,
        chunk_source: 'compiled_truth',
        expected_text_sha256: sha256(text),
        expected_text: text,
        embedding: new Float32Array(embeddingColumn.dimensions).fill(0.5),
      }],
    })).rejects.toBeInstanceOf(RequiredMigrationVectorCasMismatchError);

    const rows = await engine.executeRaw<{ model: string; embedded_text_hash: string; first_value: number }>(
      `SELECT model, embedded_text_hash, (embedding::real[])[1] AS first_value
         FROM content_chunks WHERE page_id = $1 AND chunk_index = 0`,
      [pageId],
    );
    expect(rows[0]?.model).toBe('existing:model');
    expect(rows[0]?.embedded_text_hash).toBe('existing-hash');
    expect(Number(rows[0]?.first_value)).toBeCloseTo(0.75);
  });

  test('NULL guard applies to the explicit active column, not the legacy embedding column', async () => {
    const text = 'Explicit column target.';
    const pageId = await seedChunk('migration/explicit-column', text);
    await engine.executeRaw(
      `UPDATE content_chunks
          SET embedding = ('[' || array_to_string(array_fill(0.75::real, ARRAY[$1::int]), ',') || ']')::vector
        WHERE page_id = $2 AND chunk_index = 0`,
      [embeddingColumn.dimensions, pageId],
    );

    expect(await engine.updateRequiredMigrationChunkEmbeddings({
      embeddingColumn: migrationEmbeddingColumn,
      model: 'openrouter:google/gemini-embedding-001',
      locks: lockFences(),
      rows: [{
        page_id: pageId,
        chunk_index: 0,
        chunk_source: 'compiled_truth',
        expected_text_sha256: sha256(text),
        expected_text: text,
        embedding: new Float32Array(embeddingColumn.dimensions).fill(0.25),
      }],
    })).toBe(1);

    const rows = await engine.executeRaw<{ legacy_value: number; migration_value: number }>(
      `SELECT (embedding::real[])[1] AS legacy_value,
              (embedding_required_migration::real[])[1] AS migration_value
         FROM content_chunks WHERE page_id = $1 AND chunk_index = 0`,
      [pageId],
    );
    expect(Number(rows[0]?.legacy_value)).toBeCloseTo(0.75);
    expect(Number(rows[0]?.migration_value)).toBeCloseTo(0.25);
  });

  test('one mismatch in a two-row batch updates zero rows', async () => {
    const firstText = 'Batch first.';
    const secondText = 'Batch second.';
    const pageId = await seedChunk('migration/batch', firstText, 0);
    await engine.upsertChunks('migration/batch', [
      { chunk_index: 0, chunk_text: firstText, chunk_source: 'compiled_truth', token_count: 2 },
      { chunk_index: 1, chunk_text: secondText, chunk_source: 'compiled_truth', token_count: 2 },
    ]);

    const error = await engine.updateRequiredMigrationChunkEmbeddings({
      embeddingColumn,
      model: 'openrouter:google/gemini-embedding-001',
      locks: lockFences(),
      rows: [
        {
          page_id: pageId,
          chunk_index: 0,
          chunk_source: 'compiled_truth',
          expected_text_sha256: sha256(firstText),
          expected_text: firstText,
          embedding: new Float32Array(embeddingColumn.dimensions).fill(0.1),
        },
        {
          page_id: pageId,
          chunk_index: 1,
          chunk_source: 'wrong-source',
          expected_text_sha256: sha256(secondText),
          expected_text: secondText,
          embedding: new Float32Array(embeddingColumn.dimensions).fill(0.2),
        },
      ],
    }).catch((caught) => caught);
    expect(error).toBeInstanceOf(RequiredMigrationVectorCasMismatchError);
    expect(error.expectedRows).toBe(2);
    expect(error.updatedRows).toBe(0);

    const rows = await engine.executeRaw<{ n: number }>(
      `SELECT count(*)::int AS n FROM content_chunks WHERE page_id = $1 AND embedding IS NOT NULL`,
      [pageId],
    );
    expect(Number(rows[0]?.n)).toBe(0);
  });

  test('affected-row shortfall rolls back rows already updated in the transaction', async () => {
    const firstText = 'Rollback first.';
    const secondText = 'Rollback second.';
    const pageId = await seedChunk('migration/rollback', firstText, 0);
    await engine.upsertChunks('migration/rollback', [
      { chunk_index: 0, chunk_text: firstText, chunk_source: 'compiled_truth', token_count: 2 },
      { chunk_index: 1, chunk_text: secondText, chunk_source: 'compiled_truth', token_count: 2 },
    ]);
    await engine.executeRaw(`
      CREATE OR REPLACE FUNCTION required_migration_skip_second_update()
      RETURNS trigger AS $$
      BEGIN
        IF NEW.chunk_index = 1 THEN RETURN NULL; END IF;
        RETURN NEW;
      END;
      $$ LANGUAGE plpgsql
    `);
    await engine.executeRaw(`
      CREATE TRIGGER required_migration_skip_second_update
      BEFORE UPDATE OF embedding ON content_chunks
      FOR EACH ROW EXECUTE FUNCTION required_migration_skip_second_update()
    `);

    try {
      const error = await engine.updateRequiredMigrationChunkEmbeddings({
        embeddingColumn,
        model: 'openrouter:google/gemini-embedding-001',
        locks: lockFences(),
        rows: [
          {
            page_id: pageId,
            chunk_index: 0,
            chunk_source: 'compiled_truth',
            expected_text_sha256: sha256(firstText),
            expected_text: firstText,
            embedding: new Float32Array(embeddingColumn.dimensions).fill(0.1),
          },
          {
            page_id: pageId,
            chunk_index: 1,
            chunk_source: 'compiled_truth',
            expected_text_sha256: sha256(secondText),
            expected_text: secondText,
            embedding: new Float32Array(embeddingColumn.dimensions).fill(0.2),
          },
        ],
      }).catch((caught) => caught);
      expect(error).toBeInstanceOf(RequiredMigrationVectorCasMismatchError);
      expect(error.expectedRows).toBe(2);
      expect(error.updatedRows).toBe(1);

      const rows = await engine.executeRaw<{ embedded: number }>(
        `SELECT count(*) FILTER (WHERE embedding IS NOT NULL)::int AS embedded
           FROM content_chunks WHERE page_id = $1`,
        [pageId],
      );
      expect(Number(rows[0]?.embedded)).toBe(0);
    } finally {
      await engine.executeRaw(`DROP TRIGGER IF EXISTS required_migration_skip_second_update ON content_chunks`);
      await engine.executeRaw(`DROP FUNCTION IF EXISTS required_migration_skip_second_update()`);
    }
  });

  test('successful batch refuses re-execution and preserves the first stamps', async () => {
    const text = 'Execute exactly once.';
    const pageId = await seedChunk('migration/rerun', text);
    const input = {
      embeddingColumn,
      model: 'openrouter:google/gemini-embedding-001',
      locks: lockFences(),
      rows: [{
        page_id: pageId,
        chunk_index: 0,
        chunk_source: 'compiled_truth',
        expected_text_sha256: sha256(text),
        expected_text: text,
        embedding: new Float32Array(embeddingColumn.dimensions).fill(0.4),
      }],
    };
    await engine.updateRequiredMigrationChunkEmbeddings(input);
    const first = await engine.executeRaw<{ embedded_at: Date | string; embedded_text_hash: string; first_value: number }>(
      `SELECT embedded_at, embedded_text_hash, (embedding::real[])[1] AS first_value
         FROM content_chunks WHERE page_id = $1 AND chunk_index = 0`,
      [pageId],
    );

    input.rows[0].embedding.fill(0.9);
    await expect(engine.updateRequiredMigrationChunkEmbeddings(input))
      .rejects.toBeInstanceOf(RequiredMigrationVectorCasMismatchError);
    const second = await engine.executeRaw<{ embedded_at: Date | string; embedded_text_hash: string; first_value: number }>(
      `SELECT embedded_at, embedded_text_hash, (embedding::real[])[1] AS first_value
         FROM content_chunks WHERE page_id = $1 AND chunk_index = 0`,
      [pageId],
    );
    expect(second).toEqual(first);
  });
});
