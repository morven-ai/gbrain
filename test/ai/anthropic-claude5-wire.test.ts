/**
 * Claude 5-generation request shape on the native Anthropic lane (#5688, #5690).
 *
 * Opus 5.5 and Sonnet 5.5 reject a forced `tool_choice` (`any` / `tool`) with a
 * 400, and every Claude model from Opus 4.7 / Sonnet 5 on rejects `temperature`
 * ("`temperature` is deprecated for this model"). `@ai-sdk/anthropic@3.0.74`
 * did not know any Claude 5 id, so it treated them as unknown models: it sent
 * `temperature` verbatim (decide, facts relink and brainstorm pin one) and
 * emulated `expand()`'s structured output with a forced `json` tool. Both calls
 * then 400 against the real API.
 *
 * These tests drive the real gateway (`chat()` and `expand()`) against a local
 * stub of the Messages API and assert the request body that leaves gbrain.
 */

import { afterAll, beforeAll, beforeEach, describe, expect, test } from 'bun:test';
import { chat, configureGateway, expand, resetGateway } from '../../src/core/ai/gateway.ts';

let server: ReturnType<typeof Bun.serve>;
let bodies: any[] = [];

beforeAll(() => {
  server = Bun.serve({
    port: 0,
    hostname: '127.0.0.1',
    async fetch(req) {
      const body = await req.json();
      bodies.push(body);
      const structured = body.output_config?.format !== undefined;
      return Response.json({
        id: 'msg_stub',
        type: 'message',
        role: 'assistant',
        model: body.model,
        content: [{ type: 'text', text: structured ? '{"queries":["alpha one","alpha two"]}' : 'ok' }],
        stop_reason: 'end_turn',
        usage: { input_tokens: 1, output_tokens: 1 },
      });
    },
  });
});

afterAll(() => {
  server.stop(true);
  resetGateway();
});

beforeEach(() => {
  bodies = [];
  resetGateway();
});

function configure(model: string): void {
  configureGateway({
    chat_model: `anthropic:${model}`,
    expansion_model: `anthropic:${model}`,
    env: { ANTHROPIC_API_KEY: 'stub-key', ANTHROPIC_BASE_URL: `http://127.0.0.1:${server.port}/v1` },
  });
}

describe('Claude 5-generation models on the native Anthropic lane', () => {
  for (const model of ['claude-opus-5-5', 'claude-sonnet-5-5', 'claude-opus-5', 'claude-sonnet-5', 'claude-opus-4-8']) {
    test(`${model}: chat() drops temperature instead of sending a field the model rejects`, async () => {
      configure(model);
      await chat({ model: `anthropic:${model}`, messages: [{ role: 'user', content: 'hi' }], temperature: 0, maxTokens: 16 });
      expect(bodies).toHaveLength(1);
      expect(bodies[0].model).toBe(model);
      expect('temperature' in bodies[0]).toBe(false);
    });
  }

  for (const model of ['claude-opus-5-5', 'claude-sonnet-5-5', 'claude-fable-5-2']) {
    test(`${model}: expand() uses native structured output, never a forced tool_choice`, async () => {
      configure(model);
      const out = await expand('alpha');
      expect(bodies).toHaveLength(1);
      expect(bodies[0].tool_choice).toBeUndefined();
      expect(bodies[0].output_config?.format?.type).toBe('json_schema');
      expect(out).toEqual(expect.arrayContaining(['alpha one', 'alpha two']));
      console.log(JSON.stringify({ structuredResponse: out, body: bodies[0] }));
    });
  }
});

// 복구로 바뀐 prefix 뒤의 서명만 제외하는 최종 wire 계약. 기존 시험은 사고 이력을 보내지 않았다.
// 가짜 서명은 직렬화 확인용이며 provider의 암호검증을 증명하지 않는다.
test('복구 전 signed reasoning은 유지하고 복구 뒤 signed/redacted만 wire에서 제외', async () => {
  configure('claude-fable-5-2');
  const result = await chat({ messages: [
    { role: 'user', content: 'start' },
    { role: 'assistant', content: [
      { type: 'reasoning', text: 'prefix', providerMetadata: { anthropic: { signature: 'fixture-before' } } },
      { type: 'tool-call', toolCallId: 'first', toolName: 'search', input: {} },
    ] },
    { role: 'assistant', content: [
      { type: 'reasoning', text: 'stale', providerMetadata: { anthropic: { signature: 'fixture-after' } } },
      { type: 'reasoning', text: '', providerMetadata: { anthropic: { redactedData: 'fixture-redacted' } } },
      { type: 'text', text: 'keep text' },
      { type: 'tool-call', toolCallId: 'second', toolName: 'search', input: {} },
    ] },
    { role: 'user', content: [{ type: 'tool-result', toolCallId: 'second', toolName: 'search', output: 'stored' }] },
  ], maxTokens: 64 });
  expect(result.text).toBe('ok');
  const blocks = bodies[0].messages.flatMap((m: any) => m.content);
  expect(blocks.filter((b: any) => b.type === 'thinking')).toEqual([{ type: 'thinking', thinking: 'prefix', signature: 'fixture-before' }]);
  expect(blocks.some((b: any) => b.type === 'redacted_thinking')).toBe(false);
  expect(blocks.filter((b: any) => b.type === 'tool_use').map((b: any) => b.id)).toEqual(['first', 'second']);
  expect(blocks.filter((b: any) => b.type === 'tool_result')).toHaveLength(2);
  expect(blocks).toContainEqual({ type: 'text', text: 'keep text' });
  console.log(JSON.stringify({ response: result.text, body: bodies[0] }));
});
