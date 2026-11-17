/**
 * Pins `repairToolPairing` (the chat()-boundary safety net) and proves, against
 * the REAL AI SDK v6 `generateText`, that the two failure modes this wave fixes
 * are gone:
 *   - an unbalanced tool history (assistant tool-call with no tool-result) is
 *     back-filled so v6 no longer throws AI_MissingToolResultsError, and
 *   - a Date-bearing tool-result (Postgres timestamptz) passes v6's ModelMessage
 *     JSONValue schema after `toModelMessages` ISO-izes it, whereas a raw Date
 *     is still rejected (control).
 *
 * MockLanguageModelV3 = no network / no keys.
 */
import { describe, expect, it } from 'bun:test';
import { generateText } from 'ai';
import { MockLanguageModelV3 } from 'ai/test';
import { repairToolPairing, toModelMessages, type ChatMessage, type ChatBlock } from '../../src/core/ai/gateway.ts';

function mockModel(): MockLanguageModelV3 {
  return new MockLanguageModelV3({
    doGenerate: async () => ({
      content: [{ type: 'text', text: 'ok' }],
      finishReason: 'stop',
      usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
      warnings: [],
    }),
  } as any);
}

describe('repairToolPairing', () => {
  it('is a no-op on a balanced history', () => {
    const msgs: ChatMessage[] = [
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: [{ type: 'tool-call', toolCallId: 'c1', toolName: 'search', input: {} }] },
      { role: 'user', content: [{ type: 'tool-result', toolCallId: 'c1', toolName: 'search', output: { ok: 1 } }] },
    ];
    expect(repairToolPairing(msgs)).toEqual(msgs);
  });

  it('synthesizes a tool-result turn when the assistant tool-call is fully unanswered', () => {
    const msgs: ChatMessage[] = [
      { role: 'assistant', content: [{ type: 'tool-call', toolCallId: 'c1', toolName: 'search', input: {} }] },
    ];
    const out = repairToolPairing(msgs);
    expect(out).toHaveLength(2);
    expect(out[1].role).toBe('user');
    const block = (out[1].content as any[])[0];
    expect(block).toMatchObject({ type: 'tool-result', toolCallId: 'c1', isError: true });
  });

  it('merges stubs into a PARTIALLY-answered turn without duplicating the answered id', () => {
    const msgs: ChatMessage[] = [
      { role: 'assistant', content: [
        { type: 'tool-call', toolCallId: 'a', toolName: 'search', input: {} },
        { type: 'tool-call', toolCallId: 'b', toolName: 'search', input: {} },
      ] },
      { role: 'user', content: [{ type: 'tool-result', toolCallId: 'a', toolName: 'search', output: { ok: 1 } }] },
    ];
    const out = repairToolPairing(msgs);
    expect(out).toHaveLength(2); // merged in place, not appended
    const ids = (out[1].content as any[]).map((x) => x.toolCallId);
    expect(ids).toEqual(['a', 'b']); // 'a' kept once, 'b' back-filled
    expect((out[1].content as any[]).filter((x) => x.toolCallId === 'a')).toHaveLength(1);
  });
});

describe('real AI SDK v6 validation', () => {
  it('an unbalanced history passes generateText after repairToolPairing', async () => {
    const model = mockModel();
    const unbalanced: ChatMessage[] = [
      { role: 'user', content: 'go' },
      { role: 'assistant', content: [{ type: 'tool-call', toolCallId: 'c1', toolName: 'search', input: {} }] },
      // no tool-result turn
    ];
    const result = await generateText({
      model: model as any,
      messages: toModelMessages(repairToolPairing(unbalanced)) as any,
    });
    expect(result.text).toBe('ok');
    const prompt = model.doGenerateCalls[0]!.prompt as any[];
    expect(prompt.some((m) => m.role === 'tool')).toBe(true); // stub promoted to tool role
  });

  it('a Date-bearing tool-result passes after toModelMessages ISO-izes it; a raw Date is rejected', async () => {
    const withDate: ChatMessage[] = [
      { role: 'user', content: 'go' },
      { role: 'assistant', content: [{ type: 'tool-call', toolCallId: 'c1', toolName: 'brain_get_page', input: {} }] },
      { role: 'user', content: [{ type: 'tool-result', toolCallId: 'c1', toolName: 'brain_get_page', output: { updated_at: new Date('2026-06-26T06:56:59.000Z') } }] },
    ];
    // Fixed path: converted history validates.
    await expect(generateText({ model: mockModel() as any, messages: toModelMessages(withDate) as any })).resolves.toBeDefined();

    // Control: a raw Date placed straight into a v6 ModelMessage json value is rejected.
    const rawDateMessages = [
      { role: 'user', content: 'go' },
      { role: 'assistant', content: [{ type: 'tool-call', toolCallId: 'c1', toolName: 'brain_get_page', input: {} }] },
      { role: 'tool', content: [{ type: 'tool-result', toolCallId: 'c1', toolName: 'brain_get_page', output: { type: 'json', value: { updated_at: new Date('2026-06-26T06:56:59.000Z') } } }] },
    ];
    await expect(generateText({ model: mockModel() as any, messages: rawDateMessages as any })).rejects.toThrow();
  });
});

// 복구 위치·서명·빈 assistant·비변이 계약을 보호한다. 기존 pairing 시험에는 reasoning이 없다.
// 순수 함수 원본 대조만 사용하며 provider 암호검증 주장은 하지 않는다.
describe('repairToolPairing의 Anthropic prefix 무효화 경계', () => {
  const signed: ChatBlock = { type: 'reasoning', text: 'signed', providerMetadata: { anthropic: { signature: 'fixture' } } };
  const redacted: ChatBlock = { type: 'reasoning', text: '', providerMetadata: { anthropic: { redactedData: 'opaque' } } };
  const other: ChatBlock = { type: 'reasoning', text: 'other', providerMetadata: { google: { thoughtSignature: 'keep' } } };
  const plain: ChatBlock = { type: 'reasoning', text: 'plain' };
  const call = (id: string): ChatBlock => ({ type: 'tool-call', toolCallId: id, toolName: 'search', input: {} });
  const answer = (id: string): ChatBlock => ({ type: 'tool-result', toolCallId: id, toolName: 'search', output: id });
  for (const partial of [false, true]) {
    it(`${partial ? '부분 병합' : '중간 삽입'} 이후만 제거하고 복수 복구·빈 assistant도 처리`, () => {
      const msgs: ChatMessage[] = [
        { role: 'assistant', content: [signed, call('a'), ...(partial ? [call('b')] : [])] },
        ...(partial ? [{ role: 'user' as const, content: [answer('a')] }] : []),
        { role: 'assistant', content: [signed, redacted] },
        { role: 'assistant', content: [signed, other, plain, { type: 'text', text: 'keep' }, call('c')] },
        { role: 'assistant', content: [redacted, { type: 'text', text: 'later' }] },
      ];
      const original = structuredClone(msgs);
      const out = repairToolPairing(msgs);
      expect(msgs).toEqual(original);
      expect(out[0]).toBe(msgs[0]);
      expect(out).toHaveLength(5);
      expect(out[2].content).toEqual([other, plain, { type: 'text', text: 'keep' }, call('c')]);
      expect(out[4].content).toEqual([{ type: 'text', text: 'later' }]);
      expect(repairToolPairing(out)).toEqual(out);
      expect((out[1].content as ChatBlock[]).filter(b => b.type === 'tool-result')).toHaveLength(partial ? 2 : 1);
    });
  }
  it('balanced 입력과 마지막 결과 append는 과거 signed/redacted reasoning을 보존', () => {
    const msgs: ChatMessage[] = [
      { role: 'assistant', content: [signed, redacted, call('a')] },
      { role: 'user', content: [answer('a')] },
      { role: 'assistant', content: [signed, redacted, call('b')] },
    ];
    expect(repairToolPairing(msgs).slice(0, 3)).toEqual(msgs);
    expect(repairToolPairing(msgs.slice(0, 2))).toEqual(msgs.slice(0, 2));
  });
});
