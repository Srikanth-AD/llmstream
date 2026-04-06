#!/usr/bin/env node
// Tiny CLI demo for llmstream.
//
// Usage:
//   node examples/cli-demo.js openai "Write a haiku about streams"
//   node examples/cli-demo.js anthropic "Explain SSE in one sentence"
//   node examples/cli-demo.js google "What is async iteration?"
//   node examples/cli-demo.js ollama "Tell me a joke"
//
// Required env vars:
//   openai     OPENAI_API_KEY
//   anthropic  ANTHROPIC_API_KEY
//   google     GOOGLE_API_KEY
//   ollama     (none — assumes http://localhost:11434)
//
// Run `npm run build` first so dist/ exists.

import { streamLLM } from '../dist/index.js'

const [, , provider, ...rest] = process.argv
const prompt = rest.join(' ').trim()

if (!provider || !prompt) {
  console.error('Usage: node examples/cli-demo.js <provider> "<prompt>"')
  console.error('       provider = openai | anthropic | google | ollama')
  process.exit(1)
}

const dim = (s) => `\x1b[2m${s}\x1b[0m`
const cyan = (s) => `\x1b[36m${s}\x1b[0m`
const yellow = (s) => `\x1b[33m${s}\x1b[0m`
const red = (s) => `\x1b[31m${s}\x1b[0m`
const green = (s) => `\x1b[32m${s}\x1b[0m`

function buildRequest(provider, prompt) {
  switch (provider) {
    case 'openai':
      return {
        url: 'https://api.openai.com/v1/chat/completions',
        init: {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
          },
          body: JSON.stringify({
            model: 'gpt-4o-mini',
            messages: [{ role: 'user', content: prompt }],
            stream: true,
            stream_options: { include_usage: true },
          }),
        },
      }
    case 'anthropic':
      return {
        url: 'https://api.anthropic.com/v1/messages',
        init: {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'x-api-key': process.env.ANTHROPIC_API_KEY,
            'anthropic-version': '2023-06-01',
          },
          body: JSON.stringify({
            model: 'claude-haiku-4-5-20251001',
            max_tokens: 512,
            stream: true,
            messages: [{ role: 'user', content: prompt }],
          }),
        },
      }
    case 'google': {
      const model = 'gemini-1.5-flash'
      return {
        url: `https://generativelanguage.googleapis.com/v1beta/models/${model}:streamGenerateContent?alt=sse&key=${process.env.GOOGLE_API_KEY}`,
        init: {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            contents: [{ role: 'user', parts: [{ text: prompt }] }],
          }),
        },
      }
    }
    case 'ollama':
      return {
        url: 'http://localhost:11434/api/chat',
        init: {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: process.env.OLLAMA_MODEL ?? 'llama3.2',
            messages: [{ role: 'user', content: prompt }],
            stream: true,
          }),
        },
      }
    default:
      console.error(red(`Unknown provider: ${provider}`))
      process.exit(1)
  }
}

const { url, init } = buildRequest(provider, prompt)

console.log(dim(`→ ${provider}  ${url.split('?')[0]}`))
console.log(dim(`→ prompt: ${prompt}`))
console.log()

const t0 = Date.now()
let response
try {
  response = await fetch(url, init)
} catch (err) {
  console.error(red(`fetch failed: ${err.message}`))
  process.exit(1)
}

let firstByteAt = null
let charCount = 0

for await (const event of streamLLM(response, { provider })) {
  switch (event.type) {
    case 'delta':
      if (firstByteAt === null) firstByteAt = Date.now()
      charCount += event.text.length
      process.stdout.write(event.text)
      break
    case 'tool_call_delta':
      // Skip — we'll show the consolidated tool_call.
      break
    case 'tool_call':
      console.log()
      console.log(cyan(`[tool_call] ${event.name}`), event.args)
      break
    case 'usage':
      console.log()
      console.log(
        yellow(
          `[usage] in=${event.inputTokens}  out=${event.outputTokens}  total=${event.totalTokens}`,
        ),
      )
      break
    case 'finish':
      console.log()
      console.log(green(`[finish] reason=${event.reason}`))
      break
    case 'error':
      console.log()
      console.error(red(`[error] ${event.message}`))
      break
  }
}

const total = Date.now() - t0
const ttfb = firstByteAt ? firstByteAt - t0 : null
console.log(
  dim(
    `\n${charCount} chars in ${total}ms` +
      (ttfb !== null ? `  (ttfb ${ttfb}ms)` : ''),
  ),
)
