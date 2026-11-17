import type { ChatMessage } from './gateway.ts';

/** 바뀐 prefix 뒤의 Anthropic 서명 사고만 전송 복사본에서 제외한다. */
export function dropAnthropicReplayReasoning(messages: ChatMessage[], fromIndex = 0): ChatMessage[] {
  return messages.flatMap((message, index) => {
    if (index < fromIndex || message.role !== 'assistant' || typeof message.content === 'string') return [message];
    const content = message.content.filter(block => {
      if (block.type !== 'reasoning') return true;
      const anthropic = block.providerMetadata?.anthropic;
      return anthropic?.signature == null && anthropic?.redactedData == null;
    });
    if (content.length === message.content.length) return [message];
    // 빈 assistant 제거로 prefix가 더 바뀌어도 이후 서명 제거 범위는 유지한다.
    return content.length > 0 ? [{ ...message, content }] : [];
  });
}
