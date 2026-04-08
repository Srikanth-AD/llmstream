import type { LLMEvent, SSEEvent } from '../types.js'

type OllamaChunk = {
  model?: string
  created_at?: string
  // /api/generate shape
  response?: string
  // /api/chat shape
  message?: {
    role?: string
    content?: string
    tool_calls?: Array<{
      function?: {
        name?: string
        arguments?: Record<string, unknown> | string
      }
    }>
  }
  done?: boolean
  done_reason?: string
  prompt_eval_count?: number
  eval_count?: number
}

function normalizeFinish(
  reason: string | undefined,
  hadToolCall: boolean,
): (LLMEvent & { type: 'finish' })['reason'] {
  if (hadToolCall) return 'tool_calls'
  switch (reason) {
    case 'stop':
      return 'stop'
    case 'length':
      return 'length'
    default:
      return 'stop'
  }
}

/**
 * Adapter for Ollama's local streaming API.
 *
 * Ollama uses newline-delimited JSON, not SSE — but the parseNDJSONStream
 * helper in `parser.ts` wraps each JSON line into the same SSEEvent shape so
 * this adapter sees a uniform interface.
 *
 * Both /api/generate and /api/chat are supported. Tool calls (chat API only)
 * arrive fully-formed on the final message and are emitted as a single
 * `tool_call` event.
 */
export async function* ollamaAdapter(
  source: AsyncIterable<
    SSEEvent | { type: 'error'; message: string; raw?: unknown }
  >,
): AsyncGenerator<LLMEvent> {
  let toolCounter = 0
  let hadToolCall = false
  let finishReason: string | undefined
  let inputTokens = 0
  let outputTokens = 0
  let usageSeen = false

  for await (const item of source) {
    if ('type' in item && item.type === 'error') {
      yield { type: 'error', message: item.message, raw: item.raw }
      continue
    }

    const sseEvent = item as SSEEvent
    const chunk = sseEvent.data as OllamaChunk | null
    if (!chunk || typeof chunk !== 'object') continue

    if (typeof chunk.response === 'string' && chunk.response.length > 0) {
      yield { type: 'delta', text: chunk.response }
    }

    if (chunk.message) {
      if (
        typeof chunk.message.content === 'string' &&
        chunk.message.content.length > 0
      ) {
        yield { type: 'delta', text: chunk.message.content }
      }
      if (Array.isArray(chunk.message.tool_calls)) {
        for (const tc of chunk.message.tool_calls) {
          const fn = tc.function
          if (!fn?.name) continue
          let args: Record<string, unknown> = {}
          if (fn.arguments && typeof fn.arguments === 'object') {
            args = fn.arguments as Record<string, unknown>
          } else if (typeof fn.arguments === 'string') {
            try {
              const parsed = JSON.parse(fn.arguments)
              if (parsed && typeof parsed === 'object') {
                args = parsed as Record<string, unknown>
              }
            } catch {
              /* fall through with empty args */
            }
          }
          hadToolCall = true
          yield {
            type: 'tool_call',
            id: `ollama-tool-${toolCounter++}`,
            name: fn.name,
            args,
          }
        }
      }
    }

    if (chunk.done === true) {
      finishReason = chunk.done_reason
      if (
        typeof chunk.prompt_eval_count === 'number' ||
        typeof chunk.eval_count === 'number'
      ) {
        inputTokens = chunk.prompt_eval_count ?? 0
        outputTokens = chunk.eval_count ?? 0
        usageSeen = true
      }
    }
  }

  if (usageSeen) {
    yield {
      type: 'usage',
      inputTokens,
      outputTokens,
      totalTokens: inputTokens + outputTokens,
    }
  }

  yield { type: 'finish', reason: normalizeFinish(finishReason, hadToolCall) }
}
