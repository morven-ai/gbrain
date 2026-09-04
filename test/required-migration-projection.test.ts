import { afterEach, describe, expect, test } from 'bun:test';
import { createHash } from 'crypto';
import { mkdtempSync, rmSync, writeFileSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import type { BrainEngine } from '../src/core/engine.ts';
import {
  importFromFile,
} from '../src/core/import-file.ts';
import {
  RequiredMigrationContractError,
  requiredMigrationProjectionContract,
  type RequiredMigrationFileContract,
  type RequiredMigrationSyncRuntimeOptions,
} from '../src/core/required-migration-sync.ts';
import { syncLockId } from '../src/core/db-lock.ts';
import { parseMarkdown } from '../src/core/markdown.ts';
import { chunkText } from '../src/core/chunkers/recursive.ts';
import { resolveMaxChunkTokens } from '../src/core/embedding-input-limit.ts';
import { contentHash } from '../src/core/utils.ts';

const roots: string[] = [];

afterEach(() => {
  while (roots.length > 0) rmSync(roots.pop()!, { recursive: true, force: true });
});

function sha256(value: string): string {
  return createHash('sha256').update(value, 'utf8').digest('hex');
}

function contractFor(content: string, relativePath: string): RequiredMigrationFileContract {
  const slug = relativePath.replace(/\.md$/, '');
  const parsed = parseMarkdown(content, `${slug}.md`, { validate: true });
  parsed.tags.sort();
  const chunks = parsed.compiled_truth.trim()
    ? chunkText(parsed.compiled_truth, { maxTokens: resolveMaxChunkTokens() }).map((chunk, index) => ({
        chunk_source: 'compiled_truth' as const,
        chunk_index: index,
        chunk_sha256: sha256(chunk.text),
      }))
    : [];
  return {
    pageProvenance: {
      source_kind: 'mnemo-anchor',
      source_uri: `file:///fixture.ndjson?sha256=${'a'.repeat(64)}#record_token=${sha256(relativePath)}`,
      ingested_via: 'mnemo-recovery-v3',
    },
    rawData: {
      source: 'mnemo-recovery-v3',
      data: {
        record_token: sha256(relativePath),
        lifecycle: 'active',
        invalidation_reason: null,
        superseded_by: null,
        body_chars: parsed.compiled_truth.length,
        body_bytes: Buffer.byteLength(parsed.compiled_truth),
        anchor_sha256: 'a'.repeat(64),
      },
    },
    expectedProjection: {
      markdown_sha256: sha256(content),
      page_hash: contentHash({
        title: parsed.title,
        type: parsed.type,
        compiled_truth: parsed.compiled_truth,
        timeline: parsed.timeline,
        frontmatter: parsed.frontmatter,
        tags: parsed.tags,
      }),
      chunk_count: chunks.length,
      chunks,
    },
  };
}

function noWriteEngine(): { engine: BrainEngine; writes: string[] } {
  const writes: string[] = [];
  const engine = new Proxy({ kind: 'pglite' } as unknown as BrainEngine, {
    get(target, prop: string) {
      if (prop in target) return (target as unknown as Record<string, unknown>)[prop];
      if (prop === 'getPage') return async () => null;
      if (prop === 'findDuplicatePage') return undefined;
      if (prop === 'getConfig') return async () => null;
      if (prop === 'listConfigKeys') return async () => [];
      if (prop === 'transaction') return async () => { writes.push('transaction'); throw new Error('unexpected write'); };
      if (['putPage', 'putRawData', 'upsertChunks', 'deleteChunks'].includes(prop)) {
        return async () => { writes.push(prop); };
      }
      return async () => null;
    },
  });
  return { engine, writes };
}

function fixture(content: string): { filePath: string; relativePath: string } {
  const root = mkdtempSync(join(tmpdir(), 'gbrain-required-projection-'));
  roots.push(root);
  const relativePath = 'page.md';
  const filePath = join(root, relativePath);
  writeFileSync(filePath, content);
  return { filePath, relativePath };
}

describe('REQUIRED migration pre-write projection gate', () => {
  const content = ['---', 'type: concept', 'title: Fixture', '---', '', 'stable body'].join('\n');

  test('resolver contract missing fails before transaction/putPage', async () => {
    const { filePath, relativePath } = fixture(content);
    const { engine, writes } = noWriteEngine();

    await expect(importFromFile(engine, filePath, relativePath, {
      noEmbed: true,
      requiredMigration: { mode: 'required', fileSidecarResolver: () => undefined },
    })).rejects.toBeInstanceOf(RequiredMigrationContractError);
    expect(writes).toEqual([]);
  });

  test('incomplete expected manifest fails before transaction/putPage', async () => {
    const { filePath, relativePath } = fixture(content);
    const { engine, writes } = noWriteEngine();
    const incomplete = contractFor(content, relativePath) as unknown as Record<string, unknown>;
    delete incomplete.expectedProjection;

    await expect(importFromFile(engine, filePath, relativePath, {
      noEmbed: true,
      requiredMigration: {
        mode: 'required',
        fileSidecarResolver: () => incomplete as unknown as RequiredMigrationFileContract,
      },
    })).rejects.toBeInstanceOf(RequiredMigrationContractError);
    expect(writes).toEqual([]);
  });

  test.each(['markdown_hash', 'page_hash', 'chunk_count', 'ordered_chunk'] as const)(
    '%s mismatch fails with write=0',
    async (fault) => {
      const { filePath, relativePath } = fixture(content);
      const { engine, writes } = noWriteEngine();
      const contract = contractFor(content, relativePath);
      if (fault === 'markdown_hash') contract.expectedProjection.markdown_sha256 = 'b'.repeat(64);
      else if (fault === 'page_hash') contract.expectedProjection.page_hash = 'b'.repeat(64);
      else if (fault === 'chunk_count') {
        contract.expectedProjection.chunk_count = 0;
        contract.expectedProjection.chunks = [];
      } else {
        contract.expectedProjection.chunks = contract.expectedProjection.chunks.map((chunk, index) =>
          index === 0 ? { ...chunk, chunk_sha256: 'b'.repeat(64) } : chunk,
        );
      }

      await expect(importFromFile(engine, filePath, relativePath, {
        noEmbed: true,
        requiredMigration: { mode: 'required', fileSidecarResolver: () => contract },
      })).rejects.toBeInstanceOf(RequiredMigrationContractError);
      expect(writes).toEqual([]);
    },
  );
});

test('runtime lock fields are stripped from the import projection contract', () => {
  const runtime = {
    mode: 'required',
    sourceId: 'migration',
    sourceLock: {
      id: syncLockId('migration'),
      acquiredAt: '1',
      refresh: async () => true,
      release: async () => {},
    },
    signal: new AbortController().signal,
    fileSidecarResolver: () => undefined,
  } satisfies RequiredMigrationSyncRuntimeOptions;

  expect(requiredMigrationProjectionContract(runtime)).toEqual({
    mode: 'required',
    fileSidecarResolver: runtime.fileSidecarResolver,
  });
});
