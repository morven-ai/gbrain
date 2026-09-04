import { describe, expect, test } from 'bun:test';
import { withEnv } from './helpers/with-env.ts';

import type { RequiredMigrationEmbedPair } from '../src/core/required-migration-embed-allowlist.ts';
import {
  REQUIRED_MIGRATION_EMBED_ALLOWLIST_MODE_ENV,
  REQUIRED_MIGRATION_PRIVACY_POLICY,
  runRequiredMigrationNativeAttempt,
  runWithRequiredMigrationWireAttemptHook,
  runWithRequiredMigrationWireAttemptPairs,
  type RequiredMigrationNativeAttemptFinish,
  type RequiredMigrationNativeAttemptHook,
  type RequiredMigrationNativeAttemptSummary,
} from '../src/core/required-migration-wire-attempt.ts';

const MODEL = 'google/gemini-embedding-001';

function pair(text: string, index = 0, sourceId = 'required-source'): RequiredMigrationEmbedPair {
  return Object.freeze({
    text,
    identity: Object.freeze({
      source_id: sourceId,
      page_id: 100 + index,
      slug: `required/page-${index}`,
      chunk_source: 'compiled_truth',
      chunk_index: index,
      stored_text_sha256: `${index}`.padStart(64, '0'),
    }),
  });
}

function body(inputs: readonly string[], overrides: Record<string, unknown> = {}) {
  return {
    model: MODEL,
    input: [...inputs],
    provider: { zdr: true, data_collection: 'deny' },
    ...overrides,
  };
}

function runRequiredAttempt(
  hook: RequiredMigrationNativeAttemptHook,
  pairs: readonly RequiredMigrationEmbedPair[],
  requestBody: unknown,
  nativeFetch: () => Promise<Response>,
) {
  return runWithRequiredMigrationWireAttemptHook(hook, () =>
    runWithRequiredMigrationWireAttemptPairs(
      pairs,
      MODEL,
      REQUIRED_MIGRATION_PRIVACY_POLICY,
      () => runRequiredMigrationNativeAttempt(requestBody, nativeFetch),
    ));
}

describe('REQUIRED native wire-attempt context', () => {
  test('ordinary native attempts stay unchanged without REQUIRED context', async () => {
    let fetches = 0;
    const response = await runRequiredMigrationNativeAttempt(undefined, async () => {
      fetches++;
      return new Response('{}', { status: 204 });
    });
    expect(response.status).toBe(204);
    expect(fetches).toBe(1);
  });

  test('REQUIRED mode refuses a missing wire context before native fetch', async () => {
    let fetches = 0;
    await withEnv({ [REQUIRED_MIGRATION_EMBED_ALLOWLIST_MODE_ENV]: 'REQUIRED' }, async () => {
      await expect(runRequiredMigrationNativeAttempt(undefined, async () => {
        fetches++;
        return new Response('{}', { status: 204 });
      })).rejects.toThrow(/context fault/);
      expect(fetches).toBe(0);
    });
  });

  test('missing hook and start faults refuse before native fetch', async () => {
    const pairs = [pair('fixture')];
    let fetches = 0;
    const nativeFetch = async () => {
      fetches++;
      return new Response('{}');
    };

    await expect(runWithRequiredMigrationWireAttemptPairs(
      pairs,
      MODEL,
      REQUIRED_MIGRATION_PRIVACY_POLICY,
      () => runRequiredMigrationNativeAttempt(body(['fixture']), nativeFetch),
    )).rejects.toThrow(/context fault/);

    await expect(runWithRequiredMigrationWireAttemptHook({
      attemptStarted() {},
      attemptFinished() {},
    }, () => runRequiredMigrationNativeAttempt(body(['fixture']), nativeFetch)))
      .rejects.toThrow(/context fault/);

    await expect(runRequiredAttempt({
      attemptStarted() { throw new Error('ledger start failed'); },
      attemptFinished() {},
    }, pairs, body(['fixture']), nativeFetch)).rejects.toThrow('ledger start failed');
    expect(fetches).toBe(0);
  });

  test('input, count, model, privacy, and identity drift refuse before hook or fetch', async () => {
    const pairs = [pair('first'), pair('second', 1)];
    let starts = 0;
    let fetches = 0;
    const hook: RequiredMigrationNativeAttemptHook = {
      attemptStarted() { starts++; },
      attemptFinished() {},
    };
    const cases = [
      body(['changed', 'second']),
      body(['first']),
      body(['first', 'second'], { model: 'google/other-model' }),
      body(['first', 'second'], { provider: { zdr: false, data_collection: 'allow' } }),
    ];
    for (const requestBody of cases) {
      await expect(runRequiredAttempt(hook, pairs, requestBody, async () => {
        fetches++;
        return new Response('{}');
      })).rejects.toThrow(/fault/);
    }

    const mixedSources = [pair('first'), pair('second', 1, 'other-source')];
    await expect(runRequiredAttempt(hook, mixedSources, body(['first', 'second']), async () => {
      fetches++;
      return new Response('{}');
    })).rejects.toThrow(/identity fault/);
    expect(starts).toBe(0);
    expect(fetches).toBe(0);
  });

  test('canonical OpenRouter model uses its provider segment for the native body', async () => {
    let fetches = 0;
    const hook: RequiredMigrationNativeAttemptHook = {
      attemptStarted() {},
      attemptFinished() {},
    };
    await runWithRequiredMigrationWireAttemptHook(hook, () =>
      runWithRequiredMigrationWireAttemptPairs(
        [pair('fixture')],
        'openrouter:google/gemini-embedding-001',
        REQUIRED_MIGRATION_PRIVACY_POLICY,
        () => runRequiredMigrationNativeAttempt(body(['fixture']), async () => {
          fetches++;
          return new Response('{}');
        }),
      ));
    expect(fetches).toBe(1);
  });

  test('hook-owned identity and cap refusals stop the current attempt before fetch', async () => {
    let expectedIdentityHash: string | undefined;
    let reservedChars = 0;
    let fetches = 0;
    const hook: RequiredMigrationNativeAttemptHook = {
      attemptStarted(summary) {
        expectedIdentityHash ??= summary.orderedIdentitySetSha256;
        if (summary.orderedIdentitySetSha256 !== expectedIdentityHash) throw new Error('identity drift');
        reservedChars += summary.utf16CharCount;
        if (reservedChars > 3) throw new Error('cap exceeded');
      },
      attemptFinished() {},
    };
    const nativeFetch = async () => {
      fetches++;
      return new Response('{}');
    };

    await runRequiredAttempt(hook, [pair('a')], body(['a']), nativeFetch);
    await expect(runRequiredAttempt(hook, [pair('a', 9)], body(['a']), nativeFetch))
      .rejects.toThrow('identity drift');
    await expect(runRequiredAttempt(hook, [pair('bbb')], body(['bbb']), nativeFetch))
      .rejects.toThrow('cap exceeded');
    expect(fetches).toBe(1);
  });

  test('every response and transport failure gets one finished callback', async () => {
    const starts: RequiredMigrationNativeAttemptSummary[] = [];
    const finishes: RequiredMigrationNativeAttemptFinish[] = [];
    const hook: RequiredMigrationNativeAttemptHook = {
      attemptStarted(summary) { starts.push(summary); },
      attemptFinished(_summary, finish) { finishes.push(finish); },
    };

    const response = await runRequiredAttempt(
      hook,
      [pair('response')],
      body(['response']),
      async () => new Response('{}', { status: 503 }),
    );
    expect(response.status).toBe(503);

    await expect(runRequiredAttempt(
      hook,
      [pair('failure')],
      body(['failure']),
      async () => { throw new Error('socket reset'); },
    )).rejects.toThrow('socket reset');

    expect(starts).toHaveLength(2);
    expect(starts[0]).toMatchObject({
      sourceId: 'required-source',
      itemCount: 1,
      utf16CharCount: 'response'.length,
      utf8ByteCount: Buffer.byteLength('response'),
      model: MODEL,
      privacyPolicy: REQUIRED_MIGRATION_PRIVACY_POLICY,
    });
    expect(finishes).toEqual([
      { outcome: 'response', status: 503 },
      { outcome: 'transport_failure' },
    ]);
  });

  test('finish failure withholds a successful response from the caller', async () => {
    let fetches = 0;
    await expect(runRequiredAttempt({
      attemptStarted() {},
      attemptFinished() { throw new Error('ledger finish failed'); },
    }, [pair('fixture')], body(['fixture']), async () => {
      fetches++;
      return new Response('{}', { status: 200 });
    })).rejects.toThrow('ledger finish failed');
    expect(fetches).toBe(1);
  });

  test('retry and recursive-split shaped sends are counted as separate native attempts', async () => {
    let starts = 0;
    let finishes = 0;
    let fetches = 0;
    const hook: RequiredMigrationNativeAttemptHook = {
      attemptStarted() { starts++; },
      attemptFinished() { finishes++; },
    };
    const sends = [
      [pair('a'), pair('b', 1)],
      [pair('a')],
      [pair('b', 1)],
    ];
    for (const pairs of sends) {
      await runRequiredAttempt(hook, pairs, body(pairs.map(item => item.text)), async () => {
        fetches++;
        return new Response('{}');
      });
    }
    expect({ starts, finishes, fetches }).toEqual({ starts: 3, finishes: 3, fetches: 3 });
  });
});
