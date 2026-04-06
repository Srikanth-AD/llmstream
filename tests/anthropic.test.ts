import { test } from 'node:test'
import assert from 'node:assert/strict'
import { streamLLM } from '../src/index'
import type { LLMEvent } from '../src/types'
import { makeResponse, collect, deltaText } from './helpers'

const sse = (event: string, data: unknown) =>
  `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`

test('anthropic: text deltas, usage, finish', async () => {
  const response = makeResponse([
    sse('message_start', {
      type: 'message_start',
      message: { usage: { input_tokens: 12, output_tokens: 0 } },
    }),
    sse('content_block_start', {
      type: 'content_block_start',
      index: 0,
      content_block: { type: 'text', text: '' },
    }),
    sse('content_block_delta', {
      type: 'content_block_delta',
      index: 0,
      delta: { type: 'text_delta', text: 'Hello' },
    }),
    sse('content_block_delta', {
      type: 'content_block_delta',
      index: 0,
      delta: { type: 'text_delta', text: ' world' },
    }),
    sse('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sse('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: 'end_turn' },
      usage: { output_tokens: 7 },
    }),
    sse('message_stop', { type: 'message_stop' }),
  ])

  const events = await collect(streamLLM(response, { provider: 'anthropic' }))
  assert.equal(deltaText(events), 'Hello world')
  const usage = events.find((e): e is Extract<LLMEvent, { type: 'usage' }> => e.type === 'usage')
  assert.ok(usage)
  assert.equal(usage.inputTokens, 12)
  assert.equal(usage.outputTokens, 7)
  assert.equal(usage.totalTokens, 19)
  const finish = events.find((e): e is Extract<LLMEvent, { type: 'finish' }> => e.type === 'finish')
  assert.equal(finish?.reason, 'stop')
})

test('anthropic: tool_use block accumulates input_json_delta', async () => {
  const response = makeResponse([
    sse('message_start', {
      type: 'message_start',
      message: { usage: { input_tokens: 5, output_tokens: 0 } },
    }),
    sse('content_block_start', {
      type: 'content_block_start',
      index: 0,
      content_block: { type: 'tool_use', id: 'toolu_1', name: 'get_weather' },
    }),
    sse('content_block_delta', {
      type: 'content_block_delta',
      index: 0,
      delta: { type: 'input_json_delta', partial_json: '{"loc' },
    }),
    sse('content_block_delta', {
      type: 'content_block_delta',
      index: 0,
      delta: { type: 'input_json_delta', partial_json: 'ation":"NYC"}' },
    }),
    sse('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sse('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: 'tool_use' },
      usage: { output_tokens: 3 },
    }),
    sse('message_stop', { type: 'message_stop' }),
  ])

  const events = await collect(streamLLM(response, { provider: 'anthropic' }))
  const deltas = events.filter((e) => e.type === 'tool_call_delta')
  assert.equal(deltas.length, 2)
  const call = events.find(
    (e): e is Extract<LLMEvent, { type: 'tool_call' }> => e.type === 'tool_call',
  )
  assert.ok(call)
  assert.equal(call.id, 'toolu_1')
  assert.equal(call.name, 'get_weather')
  assert.deepEqual(call.args, { location: 'NYC' })
  const finish = events.find((e): e is Extract<LLMEvent, { type: 'finish' }> => e.type === 'finish')
  assert.equal(finish?.reason, 'tool_calls')
})

test('anthropic: max_tokens stop_reason maps to length', async () => {
  const response = makeResponse([
    sse('message_start', {
      type: 'message_start',
      message: { usage: { input_tokens: 1, output_tokens: 0 } },
    }),
    sse('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: 'max_tokens' },
      usage: { output_tokens: 1 },
    }),
  ])
  const events = await collect(streamLLM(response, { provider: 'anthropic' }))
  const finish = events.find((e): e is Extract<LLMEvent, { type: 'finish' }> => e.type === 'finish')
  assert.equal(finish?.reason, 'length')
})
