import { createHash } from 'node:crypto';

import { MAX_DIMENSIONS, quoteIdentifier, validateColumnKey, vectorCastSuffix } from './search/embedding-column.ts';
import type { BrainEngine } from './engine.ts';
import type { ResolvedColumn } from './types.ts';

const SHA256_HEX_RE = /^[0-9a-f]{64}$/;
const LOCK_FENCE_RE = /^\d+(?:\.\d+)?$/;

export interface RequiredMigrationVectorCasLockFence {
  id: string;
  acquired_at: string;
}

export interface RequiredMigrationVectorCasRow {
  page_id: number;
  chunk_index: number;
  chunk_source: string;
  expected_text_sha256: string;
  expected_text: string;
  embedding: Float32Array;
}

export interface RequiredMigrationVectorCasInput {
  embeddingColumn: ResolvedColumn;
  model: string;
  locks: readonly RequiredMigrationVectorCasLockFence[];
  rows: readonly RequiredMigrationVectorCasRow[];
}

export class RequiredMigrationVectorCasInputError extends Error {
  readonly code = 'required_migration_vector_cas_invalid_input';

  constructor(
    readonly field: string,
    detail: string,
    readonly rowIndex?: number,
  ) {
    super(`Required migration vector CAS input invalid at ${field}${rowIndex === undefined ? '' : ` (row ${rowIndex})`}: ${detail}`);
    this.name = 'RequiredMigrationVectorCasInputError';
  }
}

export class RequiredMigrationVectorCasMismatchError extends Error {
  readonly code = 'required_migration_vector_cas_mismatch';

  constructor(
    readonly expectedRows: number,
    readonly updatedRows: number,
  ) {
    super(`Required migration vector CAS matched ${updatedRows}/${expectedRows} rows; database update refused`);
    this.name = 'RequiredMigrationVectorCasMismatchError';
  }
}

export interface RequiredMigrationVectorCasStatement {
  sql: string;
  params: unknown[];
  expectedRows: number;
}

type QueryRows = Array<{ updated_count: number | string }>;
export type RequiredMigrationVectorCasQuery = (sql: string, params: unknown[]) => Promise<QueryRows>;

function inputError(field: string, detail: string, rowIndex?: number): never {
  throw new RequiredMigrationVectorCasInputError(field, detail, rowIndex);
}

function validateEmbeddingColumn(value: unknown): asserts value is ResolvedColumn {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    inputError('embeddingColumn', 'an explicit resolved descriptor is required');
  }
  const column = value as Partial<ResolvedColumn>;
  if (typeof column.name !== 'string') inputError('embeddingColumn.name', 'must be a string');
  try {
    validateColumnKey(column.name);
  } catch (error) {
    inputError('embeddingColumn.name', error instanceof Error ? error.message : String(error));
  }
  if (column.type !== 'vector' && column.type !== 'halfvec') {
    inputError('embeddingColumn.type', 'must be vector or halfvec');
  }
  if (!Number.isInteger(column.dimensions) || (column.dimensions ?? 0) < 1 || (column.dimensions ?? 0) > MAX_DIMENSIONS) {
    inputError('embeddingColumn.dimensions', `must be an integer in [1, ${MAX_DIMENSIONS}]`);
  }
  if (typeof column.embeddingModel !== 'string') {
    inputError('embeddingColumn.embeddingModel', 'must be a string');
  }
  const modelSeparator = column.embeddingModel.indexOf(':');
  if (modelSeparator <= 0 || modelSeparator === column.embeddingModel.length - 1) {
    inputError('embeddingColumn.embeddingModel', 'must be a resolved provider:model string');
  }
}

function validateInput(input: RequiredMigrationVectorCasInput): void {
  if (!input || typeof input !== 'object' || Array.isArray(input)) {
    inputError('input', 'must be an object');
  }
  validateEmbeddingColumn(input.embeddingColumn);
  const modelSeparator = typeof input.model === 'string' ? input.model.indexOf(':') : -1;
  if (
    typeof input.model !== 'string' ||
    input.model.length === 0 ||
    input.model !== input.model.trim() ||
    modelSeparator <= 0 ||
    modelSeparator === input.model.length - 1
  ) {
    inputError('model', 'an explicit trimmed provider:model string is required');
  }
  if (input.model !== input.embeddingColumn.embeddingModel) {
    inputError('model', 'must exactly match embeddingColumn.embeddingModel');
  }
  if (!Array.isArray(input.locks) || input.locks.length !== 2) {
    inputError('locks', 'must contain exactly two lock fences');
  }
  const lockIds = new Set<string>();
  input.locks.forEach((lock, lockIndex) => {
    if (!lock || typeof lock !== 'object' || Array.isArray(lock)) {
      inputError('locks', 'must contain lock fence objects', lockIndex);
    }
    if (typeof lock.id !== 'string' || lock.id.length === 0 || lock.id !== lock.id.trim()) {
      inputError('locks.id', 'must be a non-empty trimmed string', lockIndex);
    }
    if (typeof lock.acquired_at !== 'string' || !LOCK_FENCE_RE.test(lock.acquired_at)) {
      inputError('locks.acquired_at', 'must be epoch-seconds text', lockIndex);
    }
    if (lockIds.has(lock.id)) inputError('locks', 'contains a duplicate lock id', lockIndex);
    lockIds.add(lock.id);
  });
  if (!Array.isArray(input.rows) || input.rows.length === 0) {
    inputError('rows', 'must be a non-empty array');
  }

  const identities = new Set<string>();
  input.rows.forEach((row, rowIndex) => {
    if (!row || typeof row !== 'object' || Array.isArray(row)) inputError('row', 'must be an object', rowIndex);
    if (!Number.isSafeInteger(row.page_id) || row.page_id <= 0) inputError('page_id', 'must be a positive safe integer', rowIndex);
    if (!Number.isSafeInteger(row.chunk_index) || row.chunk_index < 0) inputError('chunk_index', 'must be a non-negative safe integer', rowIndex);
    if (typeof row.chunk_source !== 'string' || row.chunk_source.length === 0) inputError('chunk_source', 'must be a non-empty string', rowIndex);
    if (typeof row.expected_text !== 'string') inputError('expected_text', 'must be a string', rowIndex);
    if (typeof row.expected_text_sha256 !== 'string' || !SHA256_HEX_RE.test(row.expected_text_sha256)) {
      inputError('expected_text_sha256', 'must be a lowercase 64-character SHA-256 hex digest', rowIndex);
    }
    const actualHash = createHash('sha256').update(row.expected_text, 'utf8').digest('hex');
    if (actualHash !== row.expected_text_sha256) {
      inputError('expected_text_sha256', 'does not match expected_text', rowIndex);
    }
    if (!(row.embedding instanceof Float32Array)) inputError('embedding', 'must be a Float32Array', rowIndex);
    if (row.embedding.length !== input.embeddingColumn.dimensions) {
      inputError('embedding', `must have ${input.embeddingColumn.dimensions} dimensions`, rowIndex);
    }
    for (const value of row.embedding) {
      if (!Number.isFinite(value)) inputError('embedding', 'must contain only finite values', rowIndex);
    }

    const identity = JSON.stringify([row.page_id, row.chunk_index, row.chunk_source]);
    if (identities.has(identity)) inputError('rows', 'contains a duplicate chunk identity', rowIndex);
    identities.add(identity);
  });
}

export function buildRequiredMigrationVectorCasStatement(
  input: RequiredMigrationVectorCasInput,
): RequiredMigrationVectorCasStatement {
  validateInput(input);

  const column = quoteIdentifier(input.embeddingColumn.name);
  const vectorCast = vectorCastSuffix(input.embeddingColumn);
  const params: unknown[] = [];
  const values: string[] = [];
  let paramIndex = 1;

  input.rows.forEach((row, inputOrder) => {
    values.push(
      `($${paramIndex++}::integer, $${paramIndex++}::bigint, $${paramIndex++}::integer, ` +
      `$${paramIndex++}::text, $${paramIndex++}::text, $${paramIndex++}::text, $${paramIndex++}${vectorCast})`,
    );
    params.push(
      inputOrder,
      row.page_id,
      row.chunk_index,
      row.chunk_source,
      row.expected_text_sha256,
      row.expected_text,
      `[${Array.from(row.embedding).join(',')}]`,
    );
  });
  const lockValues = input.locks.map(lock => {
    params.push(lock.id, lock.acquired_at);
    return `($${paramIndex++}::text, $${paramIndex++}::text)`;
  });
  const modelParam = `$${paramIndex++}::text`;
  const expectedRowsParam = `$${paramIndex++}::integer`;
  params.push(input.model, input.rows.length);

  return {
    expectedRows: input.rows.length,
    params,
    sql: `
WITH input_rows (
  input_order, page_id, chunk_index, chunk_source,
  expected_text_sha256, expected_text, embedding_value
) AS (
  VALUES ${values.join(',\n         ')}
), input_locks (lock_id, acquired_at) AS (
  VALUES ${lockValues.join(',\n         ')}
), held_locks AS MATERIALIZED (
  SELECT l.id
    FROM input_locks i
    JOIN gbrain_cycle_locks l ON l.id = i.lock_id
   WHERE extract(epoch from l.acquired_at)::text = i.acquired_at
     AND l.ttl_expires_at > NOW()
   FOR UPDATE OF l
), lock_gate AS (
  SELECT count(*)::integer AS matched_count
    FROM held_locks
), eligible AS (
  SELECT i.input_order, c.id
    FROM input_rows i
    JOIN content_chunks c
      ON c.page_id = i.page_id
     AND c.chunk_index = i.chunk_index
     AND c.chunk_source = i.chunk_source
   WHERE c.chunk_text = i.expected_text
     AND encode(sha256(convert_to(c.chunk_text, 'UTF8')), 'hex') = i.expected_text_sha256
     AND c.${column} IS NULL
), gate AS (
  SELECT count(*)::integer AS matched_count
    FROM eligible
), updated AS (
  UPDATE content_chunks c
     SET ${column} = i.embedding_value,
         model = ${modelParam},
         embedded_at = now(),
         embedded_text_hash = md5(i.expected_text)
    FROM input_rows i
    JOIN eligible e ON e.input_order = i.input_order
   WHERE c.id = e.id
     AND c.page_id = i.page_id
     AND c.chunk_index = i.chunk_index
     AND c.chunk_source = i.chunk_source
     AND c.chunk_text = i.expected_text
     AND encode(sha256(convert_to(c.chunk_text, 'UTF8')), 'hex') = i.expected_text_sha256
     AND c.${column} IS NULL
     AND (SELECT matched_count FROM lock_gate) = 2
     AND (SELECT matched_count FROM gate) = ${expectedRowsParam}
  RETURNING c.id
)
SELECT count(*)::integer AS updated_count FROM updated`,
  };
}

export async function executeRequiredMigrationVectorCas(
  query: RequiredMigrationVectorCasQuery,
  input: RequiredMigrationVectorCasInput,
): Promise<number> {
  const statement = buildRequiredMigrationVectorCasStatement(input);
  const rows = await query(statement.sql, statement.params);
  const updatedRows = Number(rows[0]?.updated_count ?? 0);
  if (!Number.isSafeInteger(updatedRows) || updatedRows !== statement.expectedRows) {
    throw new RequiredMigrationVectorCasMismatchError(statement.expectedRows, updatedRows);
  }
  return updatedRows;
}

export function executeRequiredMigrationVectorCasOnEngine(
  engine: BrainEngine,
  input: RequiredMigrationVectorCasInput,
): Promise<number> {
  return engine.transaction(
    (tx) => executeRequiredMigrationVectorCas(
      (query, params) => tx.executeRaw<{ updated_count: number | string }>(query, params),
      input,
    ),
  );
}
