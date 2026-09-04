import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, test } from 'bun:test';
import { execFileSync } from 'child_process';
import { createHash } from 'crypto';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'fs';
import { tmpdir } from 'os';
import { dirname, join } from 'path';
import { runSources } from '../src/commands/sources.ts';
import { performSync, runSync } from '../src/commands/sync.ts';
import { syncLockId, tryAcquireDbLock, type DbLockHandle } from '../src/core/db-lock.ts';
import { PGLiteEngine } from '../src/core/pglite-engine.ts';
import {
  importFromFile,
} from '../src/core/import-file.ts';
import {
  RequiredMigrationContractError,
  type RequiredMigrationFileContract,
  type RequiredMigrationSyncOptions,
  type RequiredMigrationSyncRuntimeOptions,
} from '../src/core/required-migration-sync.ts';
import { parseMarkdown } from '../src/core/markdown.ts';
import { chunkText } from '../src/core/chunkers/recursive.ts';
import { resolveMaxChunkTokens } from '../src/core/embedding-input-limit.ts';
import { contentHash } from '../src/core/utils.ts';
import { resetPgliteState } from './helpers/reset-pglite.ts';

let engine: PGLiteEngine;
let repo: string;

function git(args: string[]): void {
  execFileSync('git', args, { cwd: repo, stdio: 'pipe' });
}

function sha256(value: string): string {
  return createHash('sha256').update(value, 'utf8').digest('hex');
}

function markdown(title: string, body: string): string {
  return ['---', 'type: concept', `title: ${title}`, '---', '', body].join('\n');
}

function write(relativePath: string, content: string): void {
  const path = join(repo, relativePath);
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, content);
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
  const recordToken = sha256(relativePath);
  return {
    pageProvenance: {
      source_kind: 'mnemo-anchor',
      source_uri: `file:///fixture.ndjson?sha256=${'a'.repeat(64)}#record_token=${recordToken}`,
      ingested_via: 'mnemo-recovery-v3',
    },
    rawData: {
      source: 'mnemo-recovery-v3',
      data: {
        record_token: recordToken,
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

function resolverForRepo(calls: string[]): RequiredMigrationSyncOptions {
  return {
    mode: 'required',
    fileSidecarResolver: (relativePath) => {
      calls.push(relativePath);
      return contractFor(readFileSync(join(repo, relativePath), 'utf8'), relativePath);
    },
  };
}

function runtimeForRepo(
  calls: string[],
  sourceLock: DbLockHandle,
  signal: AbortSignal = new AbortController().signal,
): RequiredMigrationSyncRuntimeOptions {
  return {
    ...resolverForRepo(calls),
    sourceId: 'migration',
    sourceLock,
    signal,
  };
}

function fakeRuntime(overrides: Partial<RequiredMigrationSyncRuntimeOptions> = {}): RequiredMigrationSyncRuntimeOptions {
  return {
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
    ...overrides,
  };
}

async function addSource(): Promise<void> {
  await runSources(engine, ['add', 'migration', '--path', repo, '--no-federated', '--force']);
}

describe('REQUIRED migration sync PGLite fixture runtime', () => {
  beforeAll(async () => {
    engine = new PGLiteEngine();
    await engine.connect({});
    await engine.initSchema();
  }, 60_000);

  afterAll(async () => {
    await engine.disconnect();
  }, 60_000);

  beforeEach(async () => {
    await resetPgliteState(engine);
    repo = mkdtempSync(join(tmpdir(), 'gbrain-required-sync-'));
    git(['init', '-q']);
    git(['config', 'user.email', 'fixture@example.invalid']);
    git(['config', 'user.name', 'Fixture']);
  });

  afterEach(() => {
    rmSync(repo, { recursive: true, force: true });
  });

  test('full then incremental rename/add/modify threads resolver and persists exact sidecars', async () => {
    write('rename-me.md', markdown('Rename me', 'original rename body'));
    write('modify-me.md', markdown('Modify me', 'original modify body'));
    git(['add', '--', 'rename-me.md', 'modify-me.md']);
    git(['commit', '-qm', 'initial']);
    await addSource();

    const sourceLock = await tryAcquireDbLock(engine, syncLockId('migration'));
    expect(sourceLock).not.toBeNull();
    const db = (engine as unknown as {
      db: { query: (sql: string, params?: unknown[]) => Promise<{ rows: unknown[] }> };
    }).db;
    const originalQuery = db.query.bind(db);
    let reacquisitions = 0;
    db.query = async (sql, params) => {
      if (/^\s*INSERT INTO gbrain_cycle_locks/.test(sql)) reacquisitions++;
      return originalQuery(sql, params);
    };

    try {
      const fullCalls: string[] = [];
      await runSync(engine, [
        '--full', '--no-pull', '--no-embed', '--no-extract', '--repo', repo, '--source', 'migration',
      ], runtimeForRepo(fullCalls, sourceLock!));
      expect(fullCalls.sort()).toEqual(['modify-me.md', 'rename-me.md']);

      git(['mv', 'rename-me.md', 'renamed.md']);
      write('modify-me.md', markdown('Modify me', 'modified body'));
      write('added.md', markdown('Added', 'added body'));
      git(['add', '--', 'added.md', 'modify-me.md', 'renamed.md']);
      git(['commit', '-qm', 'rename add modify']);

      const incrementalCalls: string[] = [];
      await runSync(engine, [
        '--no-pull', '--no-embed', '--no-extract', '--repo', repo, '--source', 'migration',
      ], runtimeForRepo(incrementalCalls, sourceLock!));
      expect(incrementalCalls.sort()).toEqual(['added.md', 'modify-me.md', 'renamed.md']);
      expect(reacquisitions).toBe(0);

      expect(await engine.getPage('rename-me', { sourceId: 'migration' })).toBeNull();
      for (const relativePath of incrementalCalls) {
        const slug = relativePath.replace(/\.md$/, '');
        const contract = contractFor(readFileSync(join(repo, relativePath), 'utf8'), relativePath);
        const page = await engine.getPage(slug, { sourceId: 'migration' });
        expect(page?.source_kind).toBe(contract.pageProvenance.source_kind);
        expect(page?.source_uri).toBe(contract.pageProvenance.source_uri);
        expect(page?.ingested_via).toBe(contract.pageProvenance.ingested_via);
        const raw = await engine.getRawData(slug, contract.rawData.source, { sourceId: 'migration' });
        expect(raw).toHaveLength(1);
        expect(raw[0].data).toEqual(contract.rawData.data);
        const chunks = await engine.getChunks(slug, { sourceId: 'migration' });
        expect(chunks.map((chunk) => ({
          chunk_source: chunk.chunk_source,
          chunk_index: chunk.chunk_index,
          chunk_sha256: sha256(chunk.chunk_text),
        })) as unknown).toEqual(contract.expectedProjection.chunks);
      }
    } finally {
      db.query = originalQuery;
      await sourceLock!.release();
    }
  }, 120_000);

  test('runner-owned abort signal stops REQUIRED sync before draining the remaining files', async () => {
    write('first.md', markdown('First', 'first body'));
    write('second.md', markdown('Second', 'second body'));
    git(['add', '--', 'first.md', 'second.md']);
    git(['commit', '-qm', 'initial']);
    await addSource();

    const sourceLock = await tryAcquireDbLock(engine, syncLockId('migration'));
    expect(sourceLock).not.toBeNull();
    const controller = new AbortController();
    const calls: string[] = [];
    const baseResolver = resolverForRepo(calls).fileSidecarResolver!;
    const runtime: RequiredMigrationSyncRuntimeOptions = {
      mode: 'required',
      sourceId: 'migration',
      sourceLock: sourceLock!,
      signal: controller.signal,
      fileSidecarResolver: async (relativePath) => {
        const contract = await baseResolver(relativePath);
        if (calls.length === 1) controller.abort(new Error('lock-heartbeat-failed'));
        return contract;
      },
    };

    try {
      await expect(runSync(engine, [
        '--full', '--no-pull', '--no-embed', '--no-extract', '--repo', repo, '--source', 'migration',
      ], runtime)).rejects.toThrow('runner abort signal');
      const rows = await engine.executeRaw<{ n: number }>(
        `SELECT COUNT(*)::int AS n FROM pages WHERE source_id = 'migration'`,
      );
      expect(rows[0].n).toBeLessThan(2);
    } finally {
      await sourceLock!.release();
    }
  });

  test('runSync refuses REQUIRED mode without resolver before writes', async () => {
    write('page.md', markdown('Page', 'body'));
    git(['add', '--', 'page.md']);
    git(['commit', '-qm', 'initial']);
    await addSource();

    const sourceLock = await tryAcquireDbLock(engine, syncLockId('migration'));
    expect(sourceLock).not.toBeNull();
    let refreshCalls = 0;
    const originalRefresh = sourceLock!.refresh;
    sourceLock!.refresh = async (opts) => {
      refreshCalls++;
      return originalRefresh(opts);
    };

    try {
      await expect(runSync(engine, [
        '--full', '--no-pull', '--no-embed', '--no-extract', '--repo', repo, '--source', 'migration',
      ], {
        mode: 'required',
        sourceId: 'migration',
        sourceLock: sourceLock!,
        signal: new AbortController().signal,
      } as RequiredMigrationSyncRuntimeOptions)).rejects.toBeInstanceOf(RequiredMigrationContractError);
      expect(refreshCalls).toBe(0);
      const rows = await engine.executeRaw<{ n: number }>(`SELECT COUNT(*)::int AS n FROM pages WHERE source_id = 'migration'`);
      expect(rows[0].n).toBe(0);
    } finally {
      await sourceLock!.release();
    }
  });

  test('wrong source, missing lock, --all, and omitted --source refuse before sync writes', async () => {
    const missingLock = {
      ...fakeRuntime(),
      sourceLock: undefined,
    } as unknown as RequiredMigrationSyncRuntimeOptions;
    await expect(performSync(engine, {
      sourceId: 'migration',
      skipLock: true,
      requiredMigration: missingLock,
    })).rejects.toBeInstanceOf(RequiredMigrationContractError);
    await expect(performSync(engine, {
      sourceId: 'migration',
      requiredMigration: fakeRuntime(),
    })).rejects.toThrow('sourceLock was not acquired from the target engine');

    const otherEngine = new PGLiteEngine();
    await otherEngine.connect({});
    await otherEngine.initSchema();
    const foreignLock = await tryAcquireDbLock(otherEngine, syncLockId('migration'));
    expect(foreignLock).not.toBeNull();
    try {
      await expect(performSync(engine, {
        sourceId: 'migration',
        requiredMigration: runtimeForRepo([], foreignLock!),
      })).rejects.toThrow('sourceLock was not acquired from the target engine');
    } finally {
      await foreignLock!.release();
      await otherEngine.disconnect();
    }

    const releasedLock = await tryAcquireDbLock(engine, syncLockId('migration'));
    expect(releasedLock).not.toBeNull();
    await releasedLock!.release();
    await expect(performSync(engine, {
      sourceId: 'migration',
      requiredMigration: runtimeForRepo([], releasedLock!),
    })).rejects.toThrow('sourceLock no longer owns');

    await expect(runSync(engine, ['--source', 'migration'], fakeRuntime({
      sourceLock: { ...fakeRuntime().sourceLock, id: syncLockId('other') },
    }))).rejects.toBeInstanceOf(RequiredMigrationContractError);
    await expect(runSync(engine, ['--source', 'other'], fakeRuntime()))
      .rejects.toThrow('requires exactly one --source migration');
    await expect(runSync(engine, ['--all', '--source', 'migration'], fakeRuntime()))
      .rejects.toThrow('refuses --all');
    await expect(runSync(engine, [], fakeRuntime()))
      .rejects.toThrow('requires exactly one --source migration');
    const aborted = new AbortController();
    aborted.abort(new Error('lock-heartbeat-failed'));
    await expect(runSync(engine, ['--source', 'migration'], fakeRuntime({ signal: aborted.signal })))
      .rejects.toThrow('live runner-owned AbortSignal');
  });

  test('sidecar failure rolls back page and chunks in the same transaction', async () => {
    const relativePath = 'rollback.md';
    const content = markdown('Rollback', 'rollback body');
    write(relativePath, content);
    await addSource();
    const contract = contractFor(content, relativePath);
    const original = PGLiteEngine.prototype.putRawData;
    PGLiteEngine.prototype.putRawData = async function () {
      throw new Error('fixture sidecar failure');
    };
    try {
      await expect(importFromFile(engine, join(repo, relativePath), relativePath, {
        noEmbed: true,
        sourceId: 'migration',
        requiredMigration: { mode: 'required', fileSidecarResolver: () => contract },
      })).rejects.toThrow('fixture sidecar failure');
    } finally {
      PGLiteEngine.prototype.putRawData = original;
    }
    expect(await engine.getPage('rollback', { sourceId: 'migration' })).toBeNull();
    const chunks = await engine.executeRaw<{ n: number }>(
      `SELECT COUNT(*)::int AS n FROM content_chunks ch JOIN pages p ON p.id = ch.page_id WHERE p.source_id = 'migration'`,
    );
    expect(chunks[0].n).toBe(0);
  });

  test('unchanged content refuses provenance mismatch instead of auto-correcting', async () => {
    const relativePath = 'skip.md';
    const content = markdown('Skip', 'same body');
    write(relativePath, content);
    await addSource();
    const contract = contractFor(content, relativePath);
    await importFromFile(engine, join(repo, relativePath), relativePath, {
      noEmbed: true,
      sourceId: 'migration',
      requiredMigration: { mode: 'required', fileSidecarResolver: () => contract },
    });

    const mismatch = structuredClone(contract);
    mismatch.pageProvenance.source_uri += '&mismatch=1';
    await expect(importFromFile(engine, join(repo, relativePath), relativePath, {
      noEmbed: true,
      sourceId: 'migration',
      requiredMigration: { mode: 'required', fileSidecarResolver: () => mismatch },
    })).rejects.toThrow('content-skip page provenance mismatch');
    const page = await engine.getPage('skip', { sourceId: 'migration' });
    expect(page?.source_uri).toBe(contract.pageProvenance.source_uri);

    const sidecarMismatch = structuredClone(contract);
    sidecarMismatch.rawData.data.lifecycle = 'expired';
    await expect(importFromFile(engine, join(repo, relativePath), relativePath, {
      noEmbed: true,
      sourceId: 'migration',
      requiredMigration: { mode: 'required', fileSidecarResolver: () => sidecarMismatch },
    })).rejects.toThrow('content-skip raw_data sidecar mismatch');
    const raw = await engine.getRawData('skip', contract.rawData.source, { sourceId: 'migration' });
    expect(raw[0].data).toEqual(contract.rawData.data);
  });
});
