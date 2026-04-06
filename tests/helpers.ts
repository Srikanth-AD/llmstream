import type { LLMEvent } from '../src/types'

/**
 * Build a fake `Response` whose body streams the given chunks.
 * Chunks can be strings or Uint8Arrays; strings are encoded as UTF-8.
 * This lets tests exercise the same cross-chunk buffering behavior the
 * parser would see against a real HTTP stream.
 */
export function makeResponse(
  chunks: Array<string | Uint8Array>,
  init: { status?: number; statusText?: string } = {},
): Response {
  const encoder = new TextEncoder()
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const c of chunks) {
        controller.enqueue(typeof c === 'string' ? encoder.encode(c) : c)
      }
      controller.close()
    },
  })
  return new Response(body, {
    status: init.status ?? 200,
    statusText: init.statusText ?? 'OK',
    headers: { 'Content-Type': 'text/event-stream' },
  })
}

export async function collect<T>(it: AsyncIterable<T>): Promise<T[]> {
  const out: T[] = []
  for await (const item of it) out.push(item)
  return out
}

export function deltaText(events: LLMEvent[]): string {
  return events
    .filter((e): e is Extract<LLMEvent, { type: 'delta' }> => e.type === 'delta')
    .map((e) => e.text)
    .join('')
}
