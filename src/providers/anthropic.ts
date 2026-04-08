import type { LLMEvent, SSEEvent } from '../types.js'

type AnthropicEvent = {
  type?: string
  index?: number
  message?: {
    usage?: {
      input_tokens?: number
      output_tokens?: number
    }
    stop_reason?: string | null
  }
  content_block?: {
    type?: string
    id?: string
    name?: string
    text?: string
  }
  delta?: {
    type?: string
    text?: string
    partial_json?: string
    stop_reason?: string | null
  }
  usage?: {
    input_tokens?: number
    output_tokens?: number
  }
}

interface ToolBlock {
  id: string
  name: string
  jsonChunks: string[]
}

function normalizeStopReason(
  reason: string | null | undefined,
): (LLMEvent & { type: 'finish' })['reason'] {
  switch (reason) {
    case 'end_turn':
    case 'stop_sequence':
      return 'stop'
    case 'tool_use':
      return 'tool_calls'
    case 'max_tokens':
      return 'length'
    case 'refusal':
      return 'content_filter'
    default:
      return 'stop'
  }
}

/**
 * Adapter for Anthropic Messages streaming API.
 *
 * The Anthropic format uses named SSE events:
 *  - message_start          → carries initial usage (input_tokens)
 *  - content_block_start    → declares a text or tool_use block at an index
 *  - content_block_delta    → text_delta or input_json_delta
 *  - content_block_stop     → block finished
 *  - message_delta          → final stop_reason + cumulative output_tokens
 *  - message_stop           → terminal
 */
export async function* anthropicAdapter(
  source: AsyncIterable<
    SSEEvent | { type: 'error'; message: string; raw?: unknown }
  >,
): AsyncGenerator<LLMEvent> {
  const toolBlocks = new Map<number, ToolBlock>()
  let inputTokens = 0
  let outputTokens = 0
  let finishReason: (LLMEvent & { type: 'finish' })['reason'] | null = null

  for await (const item of source) {
    if ('type' in item && item.type === 'error') {
      yield { type: 'error', message: item.message, raw: item.raw }
      continue
    }

    const sseEvent = item as SSEEvent
    const data = sseEvent.data as AnthropicEvent | null
    if (!data || typeof data !== 'object') continue

    // Anthropic encodes the event name both in the SSE `event:` line AND in
    // the JSON `type` field. We trust the JSON `type` since it's always set.
    const evType = data.type ?? sseEvent.event

    switch (evType) {
      case 'message_start': {
        const usage = data.message?.usage
        if (usage) {
          inputTokens = usage.input_tokens ?? 0
          outputTokens = usage.output_tokens ?? 0
        }
        break
      }
      case 'content_block_start': {
        const block = data.content_block
        const idx = data.index ?? 0
        if (block?.type === 'tool_use') {
          toolBlocks.set(idx, {
            id: block.id ?? '',
            name: block.name ?? '',
            jsonChunks: [],
          })
        } else if (block?.type === 'text' && block.text) {
          // Anthropic occasionally seeds an initial text snippet here.
          yield { type: 'delta', text: block.text }
        }
        break
      }
      case 'content_block_delta': {
        const idx = data.index ?? 0
        const delta = data.delta
        if (!delta) break
        if (delta.type === 'text_delta' && typeof delta.text === 'string') {
          if (delta.text.length > 0) {
            yield { type: 'delta', text: delta.text }
          }
        } else if (
          delta.type === 'input_json_delta' &&
          typeof delta.partial_json === 'string'
        ) {
          const tb = toolBlocks.get(idx)
          if (tb) {
            tb.jsonChunks.push(delta.partial_json)
            yield {
              type: 'tool_call_delta',
              id: tb.id,
              argChunk: delta.partial_json,
            }
          }
        }
        break
      }
      case 'content_block_stop': {
        const idx = data.index ?? 0
        const tb = toolBlocks.get(idx)
        if (tb) {
          const joined = tb.jsonChunks.join('')
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
                message: `Failed to parse tool_use input as JSON: ${
                  err instanceof Error ? err.message : String(err)
                }`,
                raw: joined,
              }
            }
          }
          yield { type: 'tool_call', id: tb.id, name: tb.name, args }
          toolBlocks.delete(idx)
        }
        break
      }
      case 'message_delta': {
        if (data.delta?.stop_reason) {
          finishReason = normalizeStopReason(data.delta.stop_reason)
        }
        if (data.usage?.output_tokens != null) {
          outputTokens = data.usage.output_tokens
        }
        if (data.usage?.input_tokens != null) {
          inputTokens = data.usage.input_tokens
        }
        break
      }
      case 'message_stop': {
        // Terminal — handled after the loop.
        break
      }
      case 'ping':
      case 'error':
      default:
        break
    }
  }

  if (inputTokens > 0 || outputTokens > 0) {
    yield {
      type: 'usage',
      inputTokens,
      outputTokens,
      totalTokens: inputTokens + outputTokens,
    }
  }

  if (finishReason) {
    yield { type: 'finish', reason: finishReason }
  }
}
