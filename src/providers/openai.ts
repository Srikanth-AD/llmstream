import type { LLMEvent, SSEEvent } from '../types'

type OpenAIToolCallDelta = {
  index: number
  id?: string
  type?: string
  function?: {
    name?: string
    arguments?: string
  }
}

type OpenAIChunk = {
  id?: string
  object?: string
  choices?: Array<{
    index?: number
    delta?: {
      role?: string
      content?: string | null
      tool_calls?: OpenAIToolCallDelta[]
    }
    finish_reason?: string | null
  }>
  usage?: {
    prompt_tokens?: number
    completion_tokens?: number
    total_tokens?: number
  } | null
}

interface AccumulatedToolCall {
  id: string
  name: string
  argChunks: string[]
}

function normalizeFinish(
  reason: string | null | undefined,
): LLMEvent & { type: 'finish' } {
  switch (reason) {
    case 'stop':
    case 'tool_calls':
    case 'length':
    case 'content_filter':
      return { type: 'finish', reason }
    default:
      return { type: 'finish', reason: 'stop' }
  }
}

/**
 * Adapter for OpenAI Chat Completions streaming.
 *
 * Notes:
 *  - Tool call arguments arrive as JSON-string fragments across multiple
 *    chunks. We accumulate by index, emit `tool_call_delta` for each fragment,
 *    then a final `tool_call` once we can parse the full JSON (on
 *    finish_reason or stream end).
 *  - Usage only appears if the caller passed `stream_options.include_usage`.
 */
export async function* openaiAdapter(
  source: AsyncIterable<
    SSEEvent | { type: 'error'; message: string; raw?: unknown }
  >,
): AsyncGenerator<LLMEvent> {
  const toolCalls = new Map<number, AccumulatedToolCall>()
  let finishEvent: (LLMEvent & { type: 'finish' }) | null = null

  for await (const item of source) {
    if ('type' in item && item.type === 'error') {
      yield { type: 'error', message: item.message, raw: item.raw }
      continue
    }

    const sseEvent = item as SSEEvent
    const chunk = sseEvent.data as OpenAIChunk | null
    if (!chunk || typeof chunk !== 'object') continue

    const choice = chunk.choices?.[0]
    if (choice) {
      const delta = choice.delta
      if (delta) {
        if (typeof delta.content === 'string' && delta.content.length > 0) {
          yield { type: 'delta', text: delta.content }
        }

        if (Array.isArray(delta.tool_calls)) {
          for (const tc of delta.tool_calls) {
            const idx = typeof tc.index === 'number' ? tc.index : 0
            let acc = toolCalls.get(idx)
            if (!acc) {
              acc = {
                id: tc.id ?? '',
                name: tc.function?.name ?? '',
                argChunks: [],
              }
              toolCalls.set(idx, acc)
            }
            if (tc.id && !acc.id) acc.id = tc.id
            if (tc.function?.name && !acc.name) acc.name = tc.function.name
            const argFragment = tc.function?.arguments
            if (typeof argFragment === 'string' && argFragment.length > 0) {
              acc.argChunks.push(argFragment)
              yield {
                type: 'tool_call_delta',
                id: acc.id,
                argChunk: argFragment,
              }
            }
          }
        }
      }

      if (choice.finish_reason) {
        finishEvent = normalizeFinish(choice.finish_reason)
      }
    }

    if (chunk.usage) {
      const inputTokens = chunk.usage.prompt_tokens ?? 0
      const outputTokens = chunk.usage.completion_tokens ?? 0
      const totalTokens =
        chunk.usage.total_tokens ?? inputTokens + outputTokens
      yield { type: 'usage', inputTokens, outputTokens, totalTokens }
    }
  }

  // Flush completed tool calls. Parse the joined argument JSON; if it fails
  // emit an error event but still emit the tool_call with empty args so the
  // consumer at least knows the tool was invoked.
  for (const [, acc] of [...toolCalls.entries()].sort((a, b) => a[0] - b[0])) {
    const joined = acc.argChunks.join('')
    let args: Record<string, unknown> = {}
    if (joined.length > 0) {
      try {
        const parsed = JSON.parse(joined)
        if (parsed && typeof parsed === 'object') {
          args = parsed as Record<string, unknown>
        }
      } catch (err) {
        yield {
          type: 'error',
          message: `Failed to parse tool_call arguments as JSON: ${
            err instanceof Error ? err.message : String(err)
          }`,
          raw: joined,
        }
      }
    }
    yield { type: 'tool_call', id: acc.id, name: acc.name, args }
  }

  if (finishEvent) {
    yield finishEvent
  }
}
