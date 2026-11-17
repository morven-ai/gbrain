/**
 * E2E: gateway-loop resume reconciliation (fix-wave A).
 *
 * The gateway-native subagent loop persists each assistant turn but historically
 * never persisted the following tool-result user turn. A resumed job therefore
 * reloaded assistant tool-calls with no matching tool-result, and non-Anthropic
 * (openai-compat) providers reject that unbalanced history with
 * AI_MissingToolResultsError — dead-lettering the job. This wave:
 *   1. forward-persists the tool-result user turn (onToolResultTurn), and
 *   2. reconciles an already-corrupted transcript on resume by rebuilding the
 *      missing tool-result turns from settled subagent_tool_executions,
 *      re-dispatching idempotent-pending tools and throwing on non-idempotent.
 *
 * Hermetic: PGLite in-memory engine, gateway transport stubbed. Seeds use the
 * sanctioned `$N::text::jsonb` positional bind (NEVER JSON.stringify into a
 * bare `::jsonb`) so the seed is Postgres-safe too.
 *
 * Supersedes the resume/replay work in #1934 #2062 #2065 #2112 #2274 #2487
 * #2802 #2336 #2257 #2499.
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'bun:test';
import { PGLiteEngine } from '../../src/core/pglite-engine.ts';
import { resetPgliteState } from '../helpers/reset-pglite.ts';
import { makeSubagentHandler } from '../../src/core/minions/handlers/subagent.ts';
import type { MinionJobContext, ToolDef, ToolCtx } from '../../src/core/minions/types.ts';
import {
  __setChatTransportForTests,
  configureGateway,
  resetGateway,
  toModelMessages,
  type ChatBlock,
  type ChatMessage,
  type ChatResult,
} from '../../src/core/ai/gateway.ts';

let engine: PGLiteEngine;

beforeAll(async () => {
  engine = new PGLiteEngine();
  await engine.connect({});
  await engine.initSchema();
});
afterAll(async () => {
  __setChatTransportForTests(null);
  resetGateway();
  await engine.disconnect();
});
beforeEach(async () => {
  await resetPgliteState(engine);
  await engine.setConfig('version', '85');
  await engine.setConfig('agent.use_gateway_loop', 'true');
  configureGateway({
    chat_model: 'anthropic:claude-sonnet-4-6',
    embedding_model: 'openai:text-embedding-3-large',
    embedding_dimensions: 1536,
    expansion_model: 'anthropic:claude-haiku-4-5',
    env: { ANTHROPIC_API_KEY: 'stub', OPENAI_API_KEY: 'stub' },
  });
});

async function makeJob(prompt: string, model: string): Promise<{ jobId: number; ctx: MinionJobContext }> {
  const rows = await engine.executeRaw<{ id: number }>(
    `INSERT INTO minion_jobs (submission_authority, name, status, data, queue, priority, created_at)
     VALUES ('{"version":1,"kind":"application"}'::jsonb, 'subagent', 'active', $1::text::jsonb, 'default', 0, now()) RETURNING id`,
    [JSON.stringify({ prompt, model })],
  );
  const jobId = rows[0].id;
  const ctx: MinionJobContext = {
    id: jobId, name: 'subagent', data: { prompt, model }, attempts_made: 1, idempotency_key: null,
    signal: new AbortController().signal, deadlineAtMs: null, shutdownSignal: new AbortController().signal,
    updateProgress: async () => {}, updateTokens: async () => {}, log: async () => {},
    isActive: async () => true, readInbox: async () => [],
  };
  return { jobId, ctx };
}

function makeTools(executions: string[]): ToolDef[] {
  return [
    { name: 'search', description: 's', input_schema: { type: 'object' }, idempotent: true,
      async execute(input: unknown, _c: ToolCtx) { executions.push('search'); return { results: ['fresh'] }; } },
    { name: 'put_page', description: 'p', input_schema: { type: 'object' }, idempotent: false,
      async execute(_input: unknown, _c: ToolCtx) { executions.push('put_page'); return { saved: true }; } },
  ];
}

function buildHandler(toolRegistry: ToolDef[]) {
  return makeSubagentHandler({
    engine, config: {} as any, toolRegistry,
    makeAnthropic: () => ({ messages: { create: async () => { throw new Error('legacy path unused'); } } }) as any,
  });
}

async function seedMessage(jobId: number, idx: number, role: string, blocks: ChatBlock[]): Promise<void> {
  await engine.executeRaw(
    `INSERT INTO subagent_messages (job_id, message_idx, role, content_blocks, schema_version)
     VALUES ($1, $2, $3, $4::text::jsonb, 2)`,
    [jobId, idx, role, JSON.stringify(blocks)],
  );
}

async function seedExec(jobId: number, msgIdx: number, toolUseId: string, name: string, status: string, output: unknown, ordinal: number, error?: string): Promise<void> {
  await engine.executeRaw(
    `INSERT INTO subagent_tool_executions
       (job_id, message_idx, tool_use_id, tool_name, input, status, output, error, schema_version, ordinal)
     VALUES ($1, $2, $3, $4, $5::text::jsonb, $6, $7::text::jsonb, $8, 2, $9)`,
    [jobId, msgIdx, toolUseId, name, JSON.stringify({}), status, output == null ? null : JSON.stringify(output), error ?? null, ordinal],
  );
}

/** Assert every assistant tool-call turn is answered by a following tool-result. */
function assertBalanced(messages: ChatMessage[]): void {
  for (let i = 0; i < messages.length; i++) {
    const m = messages[i];
    if (m.role !== 'assistant' || typeof m.content === 'string') continue;
    const calls = m.content.filter((b): b is Extract<ChatBlock, { type: 'tool-call' }> => b.type === 'tool-call');
    if (calls.length === 0) continue;
    const next = messages[i + 1];
    expect(next, `assistant tool-call turn at ${i} must be followed by a tool-result turn`).toBeDefined();
    const answered = new Set(
      (typeof next!.content === 'string' ? [] : next!.content)
        .filter((b): b is Extract<ChatBlock, { type: 'tool-result' }> => b.type === 'tool-result')
        .map(b => b.toolCallId),
    );
    for (const c of calls) expect(answered.has(c.toolCallId), `tool-call ${c.toolCallId} unanswered`).toBe(true);
  }
  // The real provider path: this must not throw the ModelMessage schema error.
  expect(() => toModelMessages(messages)).not.toThrow();
}

describe('gateway resume reconciliation', () => {
  it('a pending journal write stays pending and replays its UUID before another model turn', async () => {
    let turns = 0; let pending = true; const identities: unknown[] = [];
    __setChatTransportForTests(async () => {
      turns++;
      return { text: turns === 1 ? '' : 'done', blocks: turns === 1
        ? [{ type: 'tool-call', toolCallId: 'stable-provider-call', toolName: 'brain_put_page', input: { slug: 'notes/pending-fixture', content: 'fixture' } }]
        : [{ type: 'text', text: 'done' }], stopReason: turns === 1 ? 'tool_calls' : 'end',
        usage: { input_tokens: 1, output_tokens: 1, cache_read_tokens: 0, cache_creation_tokens: 0 },
        model: 'openai:gpt-4o', providerId: 'openai' } satisfies ChatResult;
    });
    const { jobId, ctx } = await makeJob('persist the fixture', 'openai:gpt-4o');
    const registry: ToolDef[] = [{ name: 'brain_put_page', description: 'p', input_schema: { type: 'object' }, idempotent: true,
      execute: async input => {
        const id = (input as Record<string, unknown>).request_id; identities.push(id);
        if (typeof id !== 'string') throw new Error('Tool write identity must be persisted before execution');
        expect(id).toMatch(/^[0-9a-f-]{36}$/);
        const [row] = await engine.executeRaw<{ request_id: string }>("SELECT input->>'request_id' AS request_id FROM subagent_tool_executions WHERE job_id=$1", [jobId]);
        expect(row.request_id).toBe(id);
        return { request_id: id, state: pending ? 'queued' : 'committed', retry_after_ms: pending ? 100 : null };
      } }];
    // 40 s from its deadline the job is inside the replay reserve, so the pending write leaves at once (#5474).
    await expect(buildHandler(registry)({ ...ctx, deadlineAtMs: Date.now() + 40_000 })).rejects.toMatchObject({ code: 'write_pending' });
    expect((await engine.executeRaw<{ status: string }>('SELECT status FROM subagent_tool_executions WHERE job_id=$1', [jobId]))[0].status).toBe('pending');
    pending = false;
    await buildHandler(registry)(ctx);
    expect(turns).toBe(2); expect(identities).toHaveLength(2); expect(identities[1]).toBe(identities[0]);
    expect((await engine.executeRaw<{ status: string }>('SELECT status FROM subagent_tool_executions WHERE job_id=$1', [jobId]))[0].status).toBe('complete');
  });

  it('a resumed dispatch keeps replaying a still-pending write until it commits (#5474)', async () => {
    let turns = 0; let pendingReplies = 0; const seen: unknown[] = [];
    __setChatTransportForTests(async () => {
      turns++;
      return { text: '', blocks: turns === 1
        ? [{ type: 'tool-call', toolCallId: 'resume-call', toolName: 'brain_put_page', input: { slug: 'notes/resume-fixture', content: 'fixture' } }]
        : [{ type: 'text', text: 'finished' }], stopReason: turns === 1 ? 'tool_calls' : 'end',
        usage: { input_tokens: 1, output_tokens: 1, cache_read_tokens: 0, cache_creation_tokens: 0 },
        model: 'openai:gpt-4o', providerId: 'openai' } satisfies ChatResult;
    });
    const { jobId, ctx } = await makeJob('persist after resume', 'openai:gpt-4o');
    const registry: ToolDef[] = [{ name: 'brain_put_page', description: 'p', input_schema: { type: 'object' }, idempotent: true,
      execute: async input => {
        const request_id = (input as Record<string, unknown>).request_id; seen.push(request_id);
        return pendingReplies-- > 0 ? { request_id, state: 'running', retry_after_ms: 15 } : { request_id, state: 'committed', retry_after_ms: null };
      } }];
    pendingReplies = Number.MAX_SAFE_INTEGER;
    await expect(buildHandler(registry)({ ...ctx, deadlineAtMs: Date.now() + 40_000 })).rejects.toMatchObject({ code: 'write_pending' });
    pendingReplies = 3;
    await buildHandler(registry)(ctx);
    expect(turns).toBe(2);
    expect(seen).toHaveLength(5);
    expect(new Set(seen).size).toBe(1);
    const statuses = await engine.executeRaw<{ status: string }>('SELECT status FROM subagent_tool_executions WHERE job_id=$1', [jobId]);
    expect(statuses.map(row => row.status)).toEqual(['complete']);
  });

  it('forward-persists the tool-result user turn (idx 2) in a 2-turn flow', async () => {
    let turn = 0;
    __setChatTransportForTests(async () => {
      turn++;
      if (turn === 1) return {
        text: '', blocks: [{ type: 'tool-call', toolCallId: 'tc1', toolName: 'search', input: { q: 'x' } }] as ChatBlock[],
        stopReason: 'tool_calls', usage: { input_tokens: 1, output_tokens: 1, cache_read_tokens: 0, cache_creation_tokens: 0 },
        model: 'anthropic:claude-sonnet-4-6', providerId: 'anthropic',
      } satisfies ChatResult;
      return {
        text: 'done', blocks: [{ type: 'text', text: 'done' }] as ChatBlock[],
        stopReason: 'end', usage: { input_tokens: 1, output_tokens: 1, cache_read_tokens: 0, cache_creation_tokens: 0 },
        model: 'anthropic:claude-sonnet-4-6', providerId: 'anthropic',
      } satisfies ChatResult;
    });
    const { jobId, ctx } = await makeJob('go', 'anthropic:claude-sonnet-4-6');
    await buildHandler(makeTools([]))(ctx);

    const msgs = await engine.executeRaw<{ message_idx: number; role: string; content_blocks: unknown }>(
      `SELECT message_idx, role, content_blocks FROM subagent_messages WHERE job_id = $1 ORDER BY message_idx`, [jobId]);
    expect(msgs.map(m => [m.message_idx, m.role])).toEqual([[0, 'user'], [1, 'assistant'], [2, 'user'], [3, 'assistant']]);
    const toolResultTurn = typeof msgs[2].content_blocks === 'string' ? JSON.parse(msgs[2].content_blocks as string) : msgs[2].content_blocks;
    expect((toolResultTurn as any[])[0].type).toBe('tool-result');
    expect((toolResultTurn as any[])[0].toolCallId).toBe('tc1');
  });

  it('self-heals a pre-fix corrupted job from the stored output (no re-execute), transcript balanced', async () => {
    const { jobId, ctx } = await makeJob('resume me', 'openai:gpt-4o'); // non-Anthropic: strict pairing
    // Corrupted pre-fix state: seed user + assistant(tool-call), a complete
    // exec row, but NO tool-result user turn at idx 2.
    await seedMessage(jobId, 0, 'user', [{ type: 'text', text: 'resume me' }]);
    await seedMessage(jobId, 1, 'assistant', [{ type: 'tool-call', toolCallId: 'prov-tc-1', toolName: 'search', input: { q: 'x' } }]);
    await seedExec(jobId, 1, 'prov-tc-1', 'search', 'complete', { results: ['from-prior-run'] }, 0);

    let captured: ChatMessage[] = [];
    __setChatTransportForTests(async (opts) => {
      captured = opts.messages;
      return {
        text: 'recovered and done', blocks: [{ type: 'text', text: 'recovered and done' }] as ChatBlock[],
        stopReason: 'end', usage: { input_tokens: 1, output_tokens: 1, cache_read_tokens: 0, cache_creation_tokens: 0 },
        model: 'openai:gpt-4o', providerId: 'openai',
      } satisfies ChatResult;
    });
    const executions: string[] = [];
    const result = await buildHandler(makeTools(executions))(ctx);

    expect(result.result).toBe('recovered and done');
    expect(executions.length).toBe(0); // stored output reused, tool NOT re-run
    assertBalanced(captured);
    // The healed tool-result carries the real stored output.
    const healed = (captured[2].content as ChatBlock[])[0] as Extract<ChatBlock, { type: 'tool-result' }>;
    expect(healed.output).toEqual({ results: ['from-prior-run'] });

    // Durably persisted so the next resume stays balanced.
    const msgs = await engine.executeRaw<{ message_idx: number; role: string }>(
      `SELECT message_idx, role FROM subagent_messages WHERE job_id = $1 ORDER BY message_idx`, [jobId]);
    expect(msgs.map(m => [m.message_idx, m.role])).toEqual([[0, 'user'], [1, 'assistant'], [2, 'user'], [3, 'assistant']]);
  });

  it('heals MULTIPLE consecutive dangling assistant turns (pre-fix multi-turn corruption)', async () => {
    const { jobId, ctx } = await makeJob('multi', 'openai:gpt-4o');
    // Pre-fix loop persisted assistants at 1 and 3 (gaps at 2 = skipped user idx).
    await seedMessage(jobId, 0, 'user', [{ type: 'text', text: 'multi' }]);
    await seedMessage(jobId, 1, 'assistant', [{ type: 'tool-call', toolCallId: 'tc-a', toolName: 'search', input: {} }]);
    await seedExec(jobId, 1, 'tc-a', 'search', 'complete', { results: ['a'] }, 0);
    await seedMessage(jobId, 3, 'assistant', [{ type: 'tool-call', toolCallId: 'tc-b', toolName: 'search', input: {} }]);
    await seedExec(jobId, 3, 'tc-b', 'search', 'complete', { results: ['b'] }, 0);

    let captured: ChatMessage[] = [];
    __setChatTransportForTests(async (opts) => {
      captured = opts.messages;
      return { text: 'ok', blocks: [{ type: 'text', text: 'ok' }] as ChatBlock[], stopReason: 'end',
        usage: { input_tokens: 1, output_tokens: 1, cache_read_tokens: 0, cache_creation_tokens: 0 },
        model: 'openai:gpt-4o', providerId: 'openai' } satisfies ChatResult;
    });
    const executions: string[] = [];
    await buildHandler(makeTools(executions))(ctx);

    expect(executions.length).toBe(0);
    assertBalanced(captured);
    // Both dangling turns healed and persisted (idx 2 and 4).
    const msgs = await engine.executeRaw<{ message_idx: number; role: string }>(
      `SELECT message_idx, role FROM subagent_messages WHERE job_id = $1 ORDER BY message_idx`, [jobId]);
    expect(msgs.map(m => m.message_idx)).toContain(2);
    expect(msgs.map(m => m.message_idx)).toContain(4);
  });

  it('re-dispatches an idempotent tool that was still pending on resume', async () => {
    const { jobId, ctx } = await makeJob('redispatch', 'openai:gpt-4o');
    await seedMessage(jobId, 0, 'user', [{ type: 'text', text: 'redispatch' }]);
    await seedMessage(jobId, 1, 'assistant', [{ type: 'tool-call', toolCallId: 'tc-pending', toolName: 'search', input: {} }]);
    await seedExec(jobId, 1, 'tc-pending', 'search', 'pending', null, 0);

    __setChatTransportForTests(async () => ({ text: 'ok', blocks: [{ type: 'text', text: 'ok' }] as ChatBlock[], stopReason: 'end',
      usage: { input_tokens: 1, output_tokens: 1, cache_read_tokens: 0, cache_creation_tokens: 0 }, model: 'openai:gpt-4o', providerId: 'openai' } satisfies ChatResult));
    const executions: string[] = [];
    await buildHandler(makeTools(executions))(ctx);
    expect(executions).toEqual(['search']); // idempotent-pending re-executed once
  });

  it('throws on a non-idempotent tool still pending on resume', async () => {
    const { jobId, ctx } = await makeJob('unsafe', 'openai:gpt-4o');
    await seedMessage(jobId, 0, 'user', [{ type: 'text', text: 'unsafe' }]);
    await seedMessage(jobId, 1, 'assistant', [{ type: 'tool-call', toolCallId: 'tc-mut', toolName: 'put_page', input: {} }]);
    await seedExec(jobId, 1, 'tc-mut', 'put_page', 'pending', null, 0);

    __setChatTransportForTests(async () => ({ text: '', blocks: [] as ChatBlock[], stopReason: 'end',
      usage: { input_tokens: 0, output_tokens: 0, cache_read_tokens: 0, cache_creation_tokens: 0 }, model: 'openai:gpt-4o', providerId: 'openai' } satisfies ChatResult));
    await expect(buildHandler(makeTools([]))(ctx)).rejects.toThrow(/non-idempotent tool "put_page" pending on resume/i);
  });

  it('error-stubs a dangling tool-call whose tool is no longer registered', async () => {
    const { jobId, ctx } = await makeJob('gone tool', 'openai:gpt-4o');
    await seedMessage(jobId, 0, 'user', [{ type: 'text', text: 'gone tool' }]);
    await seedMessage(jobId, 1, 'assistant', [{ type: 'tool-call', toolCallId: 'tc-gone', toolName: 'removed_tool', input: {} }]);
    // No exec row and the tool isn't in the registry (only 'search'/'put_page').

    let captured: ChatMessage[] = [];
    __setChatTransportForTests(async (opts) => {
      captured = opts.messages;
      return { text: 'handled', blocks: [{ type: 'text', text: 'handled' }] as ChatBlock[], stopReason: 'end',
        usage: { input_tokens: 1, output_tokens: 1, cache_read_tokens: 0, cache_creation_tokens: 0 }, model: 'openai:gpt-4o', providerId: 'openai' } satisfies ChatResult;
    });
    const result = await buildHandler(makeTools([]))(ctx);
    expect(result.result).toBe('handled');
    assertBalanced(captured);
    const stub = (captured[2].content as ChatBlock[])[0] as Extract<ChatBlock, { type: 'tool-result' }>;
    expect(stub.isError).toBe(true);
    expect(String(stub.output)).toContain('removed_tool');
    // Persisted as a failed exec so the next resume is stable.
    const rows = await engine.executeRaw<{ status: string }>(
      `SELECT status FROM subagent_tool_executions WHERE job_id = $1 AND tool_use_id = 'tc-gone'`, [jobId]);
    expect(rows[0].status).toBe('failed');
  });

  it('terminal resume: a completed transcript returns its text without calling the model', async () => {
    const { jobId, ctx } = await makeJob('already done', 'openai:gpt-4o');
    await seedMessage(jobId, 0, 'user', [{ type: 'text', text: 'already done' }]);
    await seedMessage(jobId, 1, 'assistant', [{ type: 'text', text: 'the final answer' }]);

    let chatCalls = 0;
    __setChatTransportForTests(async () => { chatCalls++; return { text: 'SHOULD NOT RUN', blocks: [] as ChatBlock[], stopReason: 'end',
      usage: { input_tokens: 0, output_tokens: 0, cache_read_tokens: 0, cache_creation_tokens: 0 }, model: 'openai:gpt-4o', providerId: 'openai' } satisfies ChatResult; });
    const result = await buildHandler(makeTools([]))(ctx);
    expect(chatCalls).toBe(0);
    expect(result.result).toBe('the final answer');
    expect(result.stop_reason).toBe('end_turn');
  });
});

// 승인 회귀: 기존 actual-handler 시험에는 signed-only 중간 턴이나 재구성된
// system/tools가 없었다. PGLite의 제어 상태와 전송 복사본·남은 예산을 함께 검증한다.
// balanced crash resume도 예전 system/tools를 입증할 기록이 없어 보수적으로 제거한다.
describe('gateway 재개의 사고 이력과 제어 상태 분리', () => {
  const signed: ChatBlock = { type: 'reasoning', text: 'old', providerMetadata: { anthropic: { signature: 'old-signature' } } };
  const redacted: ChatBlock = { type: 'reasoning', text: '', providerMetadata: { anthropic: { redactedData: 'old-redacted' } } };
  const other: ChatBlock = { type: 'reasoning', text: 'keep', providerMetadata: { google: { thoughtSignature: 'keep' } } };
  for (const balanced of [false, true]) {
    for (const model of ['anthropic:claude-fable-5-2', 'openai:gpt-4o']) {
      it(`${model}, balanced=${balanced}: 빈 중간 턴 제거 후에도 두 소비 턴·한 호출 예산·번호 유지`, async () => {
        const { jobId, ctx } = await makeJob('resume', model);
        ctx.data = { ...ctx.data, max_turns: 3, system: 'rebuilt system' };
        await seedMessage(jobId, 0, 'user', [{ type: 'text', text: 'resume' }]);
        await seedMessage(jobId, 1, 'assistant', [signed, redacted]);
        await seedMessage(jobId, 2, 'user', [{ type: 'text', text: 'continue' }]);
        await seedMessage(jobId, 3, 'assistant', [signed, redacted, other, { type: 'text', text: 'keep text' }, { type: 'tool-call', toolCallId: 'old-call', toolName: 'search', input: {} }]);
        await seedExec(jobId, 3, 'old-call', 'search', 'complete', { results: ['stored'] }, 0);
        if (balanced) await seedMessage(jobId, 4, 'user', [{ type: 'tool-result', toolCallId: 'old-call', toolName: 'search', output: { results: ['stored'] } }]);
        const stored = await engine.executeRaw('SELECT * FROM subagent_messages WHERE job_id=$1 ORDER BY message_idx', [jobId]);
        let calls = 0;
        const fresh: ChatBlock = { type: 'reasoning', text: 'fresh', providerMetadata: { anthropic: { signature: 'fresh-signature' } } };
        __setChatTransportForTests(async opts => {
          calls++;
          const replay = structuredClone(opts.messages);
          expect(replay.filter(m => m.role === 'assistant')).toHaveLength(1);
          expect((replay.find(m => m.role === 'assistant')!.content as ChatBlock[])).toEqual([other, { type: 'text', text: 'keep text' }, { type: 'tool-call', toolCallId: 'old-call', toolName: 'search', input: {} }]);
          expect(opts.system).toContain('rebuilt system');
          assertBalanced(replay);
          return { text: '', blocks: [fresh, { type: 'tool-call', toolCallId: 'new-call', toolName: 'search', input: {} }], stopReason: 'tool_calls',
            usage: { input_tokens: 1, output_tokens: 1, cache_read_tokens: 0, cache_creation_tokens: 0 }, model, providerId: model.split(':')[0] } satisfies ChatResult;
        });
        const executions: string[] = [];
        const result = await buildHandler(makeTools(executions))(ctx);
        expect(calls).toBe(1);
        expect(result.turns_count).toBe(3);
        expect(result.stop_reason).toBe('max_turns');
        expect(executions).toEqual(['search']); // 완료된 old-call은 재실행하지 않는다.
        const after = await engine.executeRaw<any>('SELECT * FROM subagent_messages WHERE job_id=$1 ORDER BY message_idx', [jobId]);
        for (const row of stored) expect(after.find(m => m.message_idx === (row as any).message_idx)).toEqual(row);
        expect(after.map(m => [m.message_idx, m.role])).toEqual([[0, 'user'], [1, 'assistant'], [2, 'user'], [3, 'assistant'], [4, 'user'], [5, 'assistant'], [6, 'user']]);
        expect(JSON.stringify(after.find(m => m.message_idx === 5).content_blocks)).toContain('fresh-signature');
      });
    }
  }
  for (const reasoning of [signed, redacted]) {
    it(`tail ${JSON.stringify(reasoning.providerMetadata)}는 terminal 결과와 호출 0회 유지`, async () => {
      const { jobId, ctx } = await makeJob('terminal', 'anthropic:claude-fable-5-2');
      await seedMessage(jobId, 0, 'user', [{ type: 'text', text: 'terminal' }]);
      await seedMessage(jobId, 1, 'assistant', [reasoning]);
      let calls = 0;
      __setChatTransportForTests(async () => { calls++; throw new Error('terminal must not call'); });
      const result = await buildHandler(makeTools([]))(ctx);
      expect(calls).toBe(0);
      expect(result).toMatchObject({ result: '', turns_count: 1, stop_reason: 'error' });
    });
  }
  it('실제 handler→gateway→설치 SDK→HTTP: 재개 사고 제거 후 새 append-only 사고 보존', async () => {
    __setChatTransportForTests(null);
    const bodies: any[] = [];
    const server = Bun.serve({ hostname: '127.0.0.1', port: 0, async fetch(req) {
      const body = await req.json(); bodies.push(body);
      const response = { id: 'msg_fixture', type: 'message', role: 'assistant', model: body.model,
        content: bodies.length === 1 ? [
          { type: 'thinking', thinking: 'new thought', signature: 'new-signature' },
          { type: 'redacted_thinking', data: 'new-redacted' },
          { type: 'tool_use', id: 'fresh-call', name: 'search', input: {} },
        ] : [{ type: 'text', text: 'done' }],
        stop_reason: bodies.length === 1 ? 'tool_use' : 'end_turn', usage: { input_tokens: 1, output_tokens: 1 } };
      console.log(JSON.stringify({ body, response }));
      return Response.json(response);
    } });
    try {
      configureGateway({ chat_model: 'anthropic:claude-fable-5-2', env: { ANTHROPIC_API_KEY: 'fixture', ANTHROPIC_BASE_URL: `http://127.0.0.1:${server.port}/v1` } });
      const { jobId, ctx } = await makeJob('resume wire', 'anthropic:claude-fable-5-2');
      ctx.data = { ...ctx.data, system: 'current system', max_turns: 3 };
      await seedMessage(jobId, 0, 'user', [{ type: 'text', text: 'resume wire' }]);
      await seedMessage(jobId, 1, 'assistant', [signed, redacted, { type: 'tool-call', toolCallId: 'old-call', toolName: 'search', input: {} }]);
      await seedExec(jobId, 1, 'old-call', 'search', 'complete', { results: ['stored'] }, 0);
      const executions: string[] = [];
      const result = await buildHandler(makeTools(executions))(ctx);
      expect(result).toMatchObject({ result: 'done', turns_count: 2, stop_reason: 'end_turn' });
      expect(executions).toEqual(['search']);
      expect(bodies).toHaveLength(2);
      expect(JSON.stringify(bodies[0])).not.toContain('old-signature');
      expect(JSON.stringify(bodies[0])).not.toContain('old-redacted');
      const fresh = bodies[1].messages.flatMap((m: any) => m.content).filter((b: any) => ['thinking', 'redacted_thinking'].includes(b.type));
      expect(fresh).toEqual([{ type: 'thinking', thinking: 'new thought', signature: 'new-signature' }, { type: 'redacted_thinking', data: 'new-redacted' }]);
    } finally { server.stop(true); }
  });
});
