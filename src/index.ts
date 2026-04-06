import { parseSSEStream, parseNDJSONStream } from './parser'
import { openaiAdapter } from './providers/openai'
import { anthropicAdapter } from './providers/anthropic'
import { googleAdapter } from './providers/google'
import { ollamaAdapter } from './providers/ollama'
import type { LLMEvent, StreamOptions, Provider } from './types'

export type { LLMEvent, StreamOptions, Provider } from './types'
export { parseSSEStream, parseNDJSONStream } from './parser'

/**
 * Stream and normalize an LLM HTTP response into a uniform async iterator of
 * LLMEvent. Pass the raw `Response` from `fetch` along with the provider name.
 *
 *   for await (const event of streamLLM(response, { provider: 'openai' })) {
 *     if (event.type === 'delta') process.stdout.write(event.text)
 *   }
 *
 * Errors during parsing are surfaced as `{ type: 'error' }` events rather
 * than thrown, so a single bad chunk won't tear down the whole stream. If
 * `onError` is provided, it is invoked for each error event in addition to
 * yielding it.
 */
export async function* streamLLM(
  response: Response,
  options: StreamOptions,
): AsyncGenerator<LLMEvent> {
  const { provider, onError } = options

  // Surface non-2xx responses up-front. We don't throw — we yield an error
  // event and a synthetic finish so consumers can handle it uniformly.
  if (!response.ok) {
    let body = ''
    try {
      body = await response.text()
    } catch {
      /* ignore */
    }
    const err: LLMEvent = {
      type: 'error',
      message: `HTTP ${response.status} ${response.statusText}${
        body ? `: ${body.slice(0, 500)}` : ''
      }`,
    }
    if (onError) onError(new Error(err.type === 'error' ? err.message : ''))
    yield err
    yield { type: 'finish', reason: 'error' }
    return
  }

  const source =
    provider === 'ollama' ? parseNDJSONStream(response) : parseSSEStream(response)

  const adapter = pickAdapter(provider)

  try {
    for await (const event of adapter(source)) {
      if (event.type === 'error' && onError) {
        onError(new Error(event.message))
      }
      yield event
    }
  } catch (err) {
    const message =
      err instanceof Error ? err.message : 'Unknown streaming error'
    if (onError) onError(err instanceof Error ? err : new Error(message))
    yield { type: 'error', message, raw: err }
    yield { type: 'finish', reason: 'error' }
  }
}

function pickAdapter(provider: Provider) {
  switch (provider) {
    case 'openai':
      return openaiAdapter
    case 'anthropic':
      return anthropicAdapter
    case 'google':
      return googleAdapter
    case 'ollama':
      return ollamaAdapter
    default: {
      const _exhaustive: never = provider
      throw new Error(`Unknown provider: ${String(_exhaustive)}`)
    }
  }
}
