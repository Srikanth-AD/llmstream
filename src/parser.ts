import type { SSEEvent } from './types.js'

/**
 * A yielded item from parseSSEStream: either a successfully parsed SSE event
 * or an inline error marker (so adapters can decide whether to surface it).
 */
export type ParsedItem =
  | SSEEvent
  | { type: 'error'; message: string; raw?: unknown }

/**
 * Parse a `Response` body as an SSE (Server-Sent Events) stream.
 *
 * Behavior:
 *  - Reads response.body via getReader()
 *  - Decodes with TextDecoder({ stream: true }) to handle multi-byte chars
 *  - Buffers across chunks; only emits on complete `\n\n` event boundaries
 *  - Strips `data: ` prefix, joins multi-line data fields with `\n`
 *  - Skips the `[DONE]` sentinel
 *  - Captures `event:` lines and attaches them to the next event
 *  - Yields an error item (rather than throwing) on malformed JSON
 *  - Yields an error item (rather than throwing) on network errors, then stops
 */
export async function* parseSSEStream(
  response: Response,
): AsyncGenerator<ParsedItem> {
  if (!response.body) {
    yield {
      type: 'error',
      message: 'Response has no body to stream',
    }
    return
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  try {
    while (true) {
      let chunk: ReadableStreamReadResult<Uint8Array>
      try {
        chunk = await reader.read()
      } catch (err) {
        yield {
          type: 'error',
          message:
            err instanceof Error
              ? `Stream read error: ${err.message}`
              : 'Stream read error',
          raw: err,
        }
        return
      }

      if (chunk.done) {
        // Flush the decoder and any final buffered event.
        buffer += decoder.decode()
        const tail = buffer.trim()
        if (tail.length > 0) {
          yield* emitEvent(tail)
        }
        return
      }

      buffer += decoder.decode(chunk.value, { stream: true })

      // SSE events are separated by a blank line. Accept both \n\n and \r\n\r\n.
      let sepIndex: number
      // Normalize CRLF to LF for splitting simplicity.
      buffer = buffer.replace(/\r\n/g, '\n')
      while ((sepIndex = buffer.indexOf('\n\n')) !== -1) {
        const rawEvent = buffer.slice(0, sepIndex)
        buffer = buffer.slice(sepIndex + 2)
        if (rawEvent.length === 0) continue
        yield* emitEvent(rawEvent)
      }
    }
  } finally {
    try {
      reader.releaseLock()
    } catch {
      /* already released */
    }
  }
}

/**
 * Process one complete SSE event block (the text between blank lines).
 * A block can have multiple `event:` and `data:` lines.
 */
function* emitEvent(block: string): Generator<ParsedItem> {
  const lines = block.split('\n')
  let eventName: string | undefined
  const dataLines: string[] = []

  for (const line of lines) {
    if (line.length === 0) continue
    if (line.startsWith(':')) continue // SSE comment
    if (line.startsWith('event:')) {
      eventName = line.slice(6).trim()
      continue
    }
    if (line.startsWith('data:')) {
      // Per the SSE spec, a single leading space is stripped.
      let value = line.slice(5)
      if (value.startsWith(' ')) value = value.slice(1)
      dataLines.push(value)
      continue
    }
    // Other fields (id:, retry:) are ignored for our purposes.
  }

  if (dataLines.length === 0) {
    // event-only frames (rare) — surface them with empty data
    if (eventName) {
      yield { event: eventName, data: null, raw: '' }
    }
    return
  }

  const raw = dataLines.join('\n')

  // The OpenAI-style "[DONE]" sentinel.
  if (raw === '[DONE]') return

  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch (err) {
    yield {
      type: 'error',
      message: `Failed to parse SSE data as JSON: ${
        err instanceof Error ? err.message : String(err)
      }`,
      raw,
    }
    return
  }

  yield { event: eventName, data: parsed, raw }
}

/**
 * Parse a `Response` body as a newline-delimited JSON stream (Ollama format).
 *
 * Each line is one complete JSON object. We wrap each parsed object into the
 * same SSEEvent shape so provider adapters can be written against a single
 * source type.
 */
export async function* parseNDJSONStream(
  response: Response,
): AsyncGenerator<ParsedItem> {
  if (!response.body) {
    yield { type: 'error', message: 'Response has no body to stream' }
    return
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  try {
    while (true) {
      let chunk: ReadableStreamReadResult<Uint8Array>
      try {
        chunk = await reader.read()
      } catch (err) {
        yield {
          type: 'error',
          message:
            err instanceof Error
              ? `Stream read error: ${err.message}`
              : 'Stream read error',
          raw: err,
        }
        return
      }

      if (chunk.done) {
        buffer += decoder.decode()
        const tail = buffer.trim()
        if (tail.length > 0) {
          for (const line of tail.split('\n')) {
            yield* emitNDJSONLine(line)
          }
        }
        return
      }

      buffer += decoder.decode(chunk.value, { stream: true })
      let nl: number
      while ((nl = buffer.indexOf('\n')) !== -1) {
        const line = buffer.slice(0, nl)
        buffer = buffer.slice(nl + 1)
        yield* emitNDJSONLine(line)
      }
    }
  } finally {
    try {
      reader.releaseLock()
    } catch {
      /* already released */
    }
  }
}

function* emitNDJSONLine(line: string): Generator<ParsedItem> {
  const trimmed = line.trim()
  if (trimmed.length === 0) return
  let parsed: unknown
  try {
    parsed = JSON.parse(trimmed)
  } catch (err) {
    yield {
      type: 'error',
      message: `Failed to parse NDJSON line as JSON: ${
        err instanceof Error ? err.message : String(err)
      }`,
      raw: trimmed,
    }
    return
  }
  yield { data: parsed, raw: trimmed }
}
