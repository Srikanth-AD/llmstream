# llmstream

A tiny, **zero-dependency** TypeScript library that parses streaming responses
from OpenAI, Anthropic, Google Gemini, and Ollama into a single, clean,
normalized async iterator.

Write your streaming UI / agent loop **once** — swap providers without
touching the consumer.

## Install

```bash
npm install llmstream
```

Requires a runtime with `fetch`, `ReadableStream`, and `TextDecoder`
(Node 18+, Bun, Deno, or any modern browser).

## Usage

```ts
import { streamLLM } from 'llmstream'

const response = await fetch('https://api.openai.com/v1/chat/completions', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
  },
  body: JSON.stringify({
    model: 'gpt-4o',
    messages: [{ role: 'user', content: 'Hello' }],
    stream: true,
  }),
})

for await (const event of streamLLM(response, { provider: 'openai' })) {
  if (event.type === 'delta')     process.stdout.write(event.text)
  if (event.type === 'tool_call') console.log(event.name, event.args)
  if (event.type === 'finish')    console.log('done:', event.reason)
  if (event.type === 'usage')     console.log('tokens:', event.totalTokens)
}
```

The exact same loop works with `provider: 'anthropic'`, `'google'`, or
`'ollama'`.

## Event types

Every provider is normalized to this small, discriminated-union event type:

```ts
type LLMEvent =
  | { type: 'delta';            text: string }
  | { type: 'tool_call';        id: string; name: string; args: Record<string, unknown> }
  | { type: 'tool_call_delta';  id: string; argChunk: string }
  | { type: 'finish';           reason: 'stop' | 'tool_calls' | 'length' | 'content_filter' | 'error' }
  | { type: 'usage';            inputTokens: number; outputTokens: number; totalTokens: number }
  | { type: 'error';            message: string; raw?: unknown }
```

### Notes per provider

- **OpenAI** — pass `stream_options: { include_usage: true }` to get a
  `usage` event. Tool-call argument fragments are emitted as
  `tool_call_delta` while they stream and a single consolidated `tool_call`
  is emitted when the function call is complete.
- **Anthropic** — `stop_reason` is normalized
  (`end_turn`/`stop_sequence` → `stop`, `tool_use` → `tool_calls`,
  `max_tokens` → `length`, `refusal` → `content_filter`).
- **Google Gemini** — use the SSE-style endpoint
  (`...:streamGenerateContent?alt=sse`). `functionCall` parts are emitted as
  a single `tool_call`. Gemini doesn't provide tool-call ids, so a
  synthetic `gemini-tool-N` id is used.
- **Ollama** — uses newline-delimited JSON, not SSE; the library detects
  this automatically. Both `/api/generate` and `/api/chat` are supported.

## Error handling

Parsing errors (malformed JSON, mid-stream disconnects, non-2xx responses)
are surfaced as `{ type: 'error' }` events rather than thrown — a single bad
chunk won't tear down the whole stream. You can also pass an `onError`
callback for logging:

```ts
for await (const event of streamLLM(response, {
  provider: 'openai',
  onError: (err) => console.error('llmstream:', err.message),
})) {
  /* ... */
}
```

## Why

Every LLM provider invents its own streaming format, framing, and
finish-reason vocabulary. `llmstream` collapses all of that into one
iterator so the rest of your code can stay provider-agnostic.

- **Zero dependencies** — just the platform.
- **Tiny** — under ~1KLOC of source, fully tree-shakable.
- **Strict TypeScript** — discriminated unions throughout.
- **Crash-resistant** — errors are events, not exceptions.

## License

MIT
