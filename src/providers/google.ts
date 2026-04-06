import type { LLMEvent, SSEEvent } from '../types'

type GooglePart = {
  text?: string
  functionCall?: {
    name?: string
    args?: Record<string, unknown>
  }
}

type GoogleChunk = {
  candidates?: Array<{
    content?: {
      parts?: GooglePart[]
      role?: string
    }
    finishReason?: string
    index?: number
  }>
  usageMetadata?: {
    promptTokenCount?: number
    candidatesTokenCount?: number
    totalTokenCount?: number
  }
}

function normalizeFinish(
  reason: string | undefined,
): (LLMEvent & { type: 'finish' })['reason'] {
  switch (reason) {
    case 'STOP':
      return 'stop'
    case 'MAX_TOKENS':
      return 'length'
    case 'SAFETY':
    case 'RECITATION':
    case 'BLOCKLIST':
    case 'PROHIBITED_CONTENT':
    case 'SPII':
      return 'content_filter'
    case 'TOOL_CALLS':
    case 'TOOL_CODE':
      return 'tool_calls'
    default:
      return 'stop'
  }
}

/**
 * Adapter for Google Gemini's streamGenerateContent API.
 *
 * Google emits an array of GenerateContentResponse JSON objects. When using
 * the SSE-style endpoint (`?alt=sse`), each chunk arrives as a `data:` line
 * with one response object. We support that shape here.
 *
 * Tool calls in Gemini are not chunked — `functionCall.args` arrives as a
 * fully-formed object on a single part — so we emit `tool_call` directly.
 * We synthesize a stable id from the function name + index since Gemini does
 * not provide one.
 */
export async function* googleAdapter(
  source: AsyncIterable<
    SSEEvent | { type: 'error'; message: string; raw?: unknown }
  >,
): AsyncGenerator<LLMEvent> {
  let finishReason: (LLMEvent & { type: 'finish' })['reason'] | null = null
  let usageInput = 0
  let usageOutput = 0
  let usageTotal = 0
  let usageSeen = false
  let toolCounter = 0

  for await (const item of source) {
    if ('type' in item && item.type === 'error') {
      yield { type: 'error', message: item.message, raw: item.raw }
      continue
    }

    const sseEvent = item as SSEEvent
    const chunk = sseEvent.data as GoogleChunk | null
    if (!chunk || typeof chunk !== 'object') continue

    const candidate = chunk.candidates?.[0]
    if (candidate) {
      const parts = candidate.content?.parts ?? []
      for (const part of parts) {
        if (typeof part.text === 'string' && part.text.length > 0) {
          yield { type: 'delta', text: part.text }
        }
        if (part.functionCall && part.functionCall.name) {
          const id = `gemini-tool-${toolCounter++}`
          yield {
            type: 'tool_call',
            id,
            name: part.functionCall.name,
            args: part.functionCall.args ?? {},
          }
        }
      }
      if (candidate.finishReason) {
        finishReason = normalizeFinish(candidate.finishReason)
      }
    }

    if (chunk.usageMetadata) {
      usageInput = chunk.usageMetadata.promptTokenCount ?? usageInput
      usageOutput = chunk.usageMetadata.candidatesTokenCount ?? usageOutput
      usageTotal =
        chunk.usageMetadata.totalTokenCount ?? usageInput + usageOutput
      usageSeen = true
    }
  }

  if (usageSeen) {
    yield {
      type: 'usage',
      inputTokens: usageInput,
      outputTokens: usageOutput,
      totalTokens: usageTotal,
    }
  }

  if (finishReason) {
    yield { type: 'finish', reason: finishReason }
  }
}
