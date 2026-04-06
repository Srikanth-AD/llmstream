/**
 * Unified, normalized event types emitted by streamLLM.
 *
 * Every provider adapter MUST translate its native streaming format into
 * this small set of events. Consumers should be able to write code once
 * against these types and have it work across providers.
 */
export type LLMEvent =
  | { type: 'delta'; text: string }
  | {
      type: 'tool_call'
      id: string
      name: string
      args: Record<string, unknown>
    }
  | { type: 'tool_call_delta'; id: string; argChunk: string }
  | {
      type: 'finish'
      reason: 'stop' | 'tool_calls' | 'length' | 'content_filter' | 'error'
    }
  | {
      type: 'usage'
      inputTokens: number
      outputTokens: number
      totalTokens: number
    }
  | { type: 'error'; message: string; raw?: unknown }

export type Provider = 'openai' | 'anthropic' | 'google' | 'ollama'

export interface StreamOptions {
  provider: Provider
  /**
   * Optional callback invoked for each error event. The error event is also
   * still yielded through the async iterator — this is purely a side-channel
   * for logging or telemetry.
   */
  onError?: (error: Error) => void
}

/**
 * A single SSE event after parseSSEStream has split, stripped, and parsed it.
 *
 * - `event` is the value of any preceding `event:` line (Anthropic uses this)
 * - `data` is the JSON-parsed payload from one or more `data:` lines
 * - `raw` is the original concatenated data string (useful for error reporting)
 */
export interface SSEEvent {
  event?: string
  data: unknown
  raw: string
}

/**
 * Provider adapter signature. Takes the stream of parsed SSE events and
 * yields normalized LLMEvents.
 */
export type ProviderAdapter = (
  source: AsyncIterable<SSEEvent | { type: 'error'; message: string; raw?: unknown }>,
) => AsyncGenerator<LLMEvent>
