import { createHash } from 'node:crypto';
import { describe, expect, test } from 'bun:test';

import { PostgresEngine } from '../src/core/postgres-engine.ts';
import {
  RequiredMigrationVectorCasInputError,
  RequiredMigrationVectorCasMismatchError,
  buildRequiredMigrationVectorCasStatement,
  executeRequiredMigrationVectorCas,
} from '../src/core/required-migration-vector-cas.ts';
import type { RequiredMigrationVectorCasInput } from '../src/core/required-migration-vector-cas.ts';
import type { ResolvedColumn } from '../src/core/types.ts';

const embeddingColumn: ResolvedColumn = {
  name: 'embedding_reviewed',
  type: 'halfvec',
  dimensions: 3,
  embeddingModel: 'openrouter:google/gemini-embedding-001',
};

function sha256(text: string): string {
  return createHash('sha256').update(text, 'utf8').digest('hex');
}

function validInput(): RequiredMigrationVectorCasInput {
  const expectedText = 'exact reviewed text';
  return {
    embeddingColumn,
    model: 'openrouter:google/gemini-embedding-001',
    locks: [
      { id: 'gbrain-sync:fixture', acquired_at: '1000.125' },
      { id: 'gbrain-embed-backfill:fixture', acquired_at: '1000.250' },
    ],
    rows: [{
      page_id: 11,
      chunk_index: 2,
      chunk_source: 'compiled_truth',
      expected_text_sha256: sha256(expectedText),
      expected_text: expectedText,
      embedding: new Float32Array([0.1, 0.2, 0.3]),
    }],
  };
}

describe('required migration vector CAS input validation', () => {
  test('rejects malformed or text-mismatched SHA-256 before SQL execution', () => {
    const malformed = validInput();
    malformed.rows[0].expected_text_sha256 = 'ABC';
    expect(() => buildRequiredMigrationVectorCasStatement(malformed)).toThrow(RequiredMigrationVectorCasInputError);

    const mismatched = validInput();
    mismatched.rows[0].expected_text_sha256 = sha256('different text');
    expect(() => buildRequiredMigrationVectorCasStatement(mismatched)).toThrow(/does not match expected_text/);
  });

  test('requires exactly two distinct, well-formed lock fences', () => {
    const missing = { ...validInput(), locks: undefined } as unknown as RequiredMigrationVectorCasInput;
    expect(() => buildRequiredMigrationVectorCasStatement(missing)).toThrow(/exactly two lock fences/);

    const one = validInput();
    one.locks = one.locks.slice(0, 1);
    expect(() => buildRequiredMigrationVectorCasStatement(one)).toThrow(/exactly two lock fences/);

    const three = validInput();
    three.locks = [...three.locks, { id: 'third', acquired_at: '1000.375' }];
    expect(() => buildRequiredMigrationVectorCasStatement(three)).toThrow(/exactly two lock fences/);

    const duplicate = validInput();
    duplicate.locks = [duplicate.locks[0], { ...duplicate.locks[1], id: duplicate.locks[0].id }];
    expect(() => buildRequiredMigrationVectorCasStatement(duplicate)).toThrow(/duplicate lock id/);

    const malformed = validInput();
    malformed.locks = [malformed.locks[0], { ...malformed.locks[1], acquired_at: '2026-08-30T00:00:00Z' }];
    expect(() => buildRequiredMigrationVectorCasStatement(malformed)).toThrow(/epoch-seconds text/);
  });

  test('rejects duplicate chunk identities', () => {
    const input = validInput();
    input.rows = [input.rows[0], { ...input.rows[0], embedding: new Float32Array([0.3, 0.2, 0.1]) }];
    expect(() => buildRequiredMigrationVectorCasStatement(input)).toThrow(/duplicate chunk identity/);
  });

  test('rejects invalid row and vector shapes', () => {
    const badPage = validInput();
    badPage.rows[0].page_id = 0;
    expect(() => buildRequiredMigrationVectorCasStatement(badPage)).toThrow(/page_id/);

    const wrongDimensions = validInput();
    wrongDimensions.rows[0].embedding = new Float32Array([0.1, 0.2]);
    expect(() => buildRequiredMigrationVectorCasStatement(wrongDimensions)).toThrow(/3 dimensions/);

    const nonFinite = validInput();
    nonFinite.rows[0].embedding = new Float32Array([0.1, Number.NaN, 0.3]);
    expect(() => buildRequiredMigrationVectorCasStatement(nonFinite)).toThrow(/finite values/);
  });

  test('requires explicit resolved embeddingColumn and provider:model', () => {
    const missingColumn = { ...validInput(), embeddingColumn: undefined } as unknown as RequiredMigrationVectorCasInput;
    expect(() => buildRequiredMigrationVectorCasStatement(missingColumn as RequiredMigrationVectorCasInput)).toThrow(/explicit resolved descriptor/);

    const missingModel = { ...validInput(), model: undefined } as unknown as RequiredMigrationVectorCasInput;
    expect(() => buildRequiredMigrationVectorCasStatement(missingModel as RequiredMigrationVectorCasInput)).toThrow(/provider:model/);

    const unresolvedColumn = validInput();
    unresolvedColumn.embeddingColumn = { ...embeddingColumn, embeddingModel: '' };
    expect(() => buildRequiredMigrationVectorCasStatement(unresolvedColumn)).toThrow(/resolved provider:model/);

    const mismatchedModel = validInput();
    mismatchedModel.model = 'openai:text-embedding-3-small';
    expect(() => buildRequiredMigrationVectorCasStatement(mismatchedModel)).toThrow(/exactly match embeddingColumn/);
  });
});

describe('required migration vector CAS SQL shape and Postgres parity', () => {
  test('SQL gates the whole batch and only assigns vector provenance columns', () => {
    const statement = buildRequiredMigrationVectorCasStatement(validInput());
    const normalized = statement.sql.replace(/\s+/g, ' ').trim();

    expect(normalized).toContain('c.page_id = i.page_id');
    expect(normalized).toContain('c.chunk_index = i.chunk_index');
    expect(normalized).toContain('c.chunk_source = i.chunk_source');
    expect(normalized).toContain('c.chunk_text = i.expected_text');
    expect(normalized).toContain("encode(sha256(convert_to(c.chunk_text, 'UTF8')), 'hex') = i.expected_text_sha256");
    expect(normalized).toContain('c."embedding_reviewed" IS NULL');
    expect(normalized).toContain('extract(epoch from l.acquired_at)::text = i.acquired_at');
    expect(normalized).toContain('l.ttl_expires_at > NOW()');
    expect(normalized).toContain('FOR UPDATE OF l');
    expect(normalized).toContain('(SELECT matched_count FROM lock_gate) = 2');
    expect(normalized).toContain('(SELECT matched_count FROM gate) = $13::integer');

    const setClause = normalized.match(/UPDATE content_chunks c SET (.+?) FROM input_rows i/)?.[1] ?? '';
    expect(setClause).toContain('"embedding_reviewed" = i.embedding_value');
    expect(setClause).toContain('model = $12::text');
    expect(setClause).toContain('embedded_at = now()');
    expect(setClause).toContain('embedded_text_hash = md5(i.expected_text)');
    expect(setClause).not.toMatch(/chunk_text\s*=/);
    expect(setClause).not.toMatch(/chunk_(index|source)\s*=/);
    expect(setClause).not.toMatch(/compiled_truth|frontmatter|metadata/);
  });

  test('PostgresEngine executes the same shared SQL statement without a connection', async () => {
    const input = validInput();
    const expected = buildRequiredMigrationVectorCasStatement(input);
    let capturedSql = '';
    let capturedParams: unknown[] = [];
    const engine = new PostgresEngine();
    Object.defineProperty(engine, 'executeRaw', {
      value: async (sql: string, params: unknown[]) => {
        capturedSql = sql;
        capturedParams = params;
        return [{ updated_count: 1 }];
      },
    });
    Object.defineProperty(engine, 'transaction', {
      value: async (fn: (tx: PostgresEngine) => Promise<unknown>) => fn(engine),
    });

    expect(await engine.updateRequiredMigrationChunkEmbeddings(input)).toBe(1);
    expect(capturedSql).toBe(expected.sql);
    expect(capturedParams).toEqual(expected.params);
  });

  test('unexpected affected-row count fails with the typed CAS error', async () => {
    const input = validInput();
    input.rows = [
      input.rows[0],
      {
        ...input.rows[0],
        chunk_index: 3,
        embedding: new Float32Array([0.3, 0.2, 0.1]),
      },
    ];

    const error = await executeRequiredMigrationVectorCas(
      async () => [{ updated_count: 1 }],
      input,
    ).catch((caught) => caught);
    expect(error).toBeInstanceOf(RequiredMigrationVectorCasMismatchError);
    expect(error.code).toBe('required_migration_vector_cas_mismatch');
    expect(error.expectedRows).toBe(2);
    expect(error.updatedRows).toBe(1);
  });
});
