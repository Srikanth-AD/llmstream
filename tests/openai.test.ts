import { test } from 'node:test'
import assert from 'node:assert/strict'
import { streamLLM } from '../src/index'
import type { LLMEvent } from '../src/types'
import { makeResponse, collect, deltaText } from './helpers'

const sse = (obj: unknown) => `data: ${JSON.stringify(obj)}\n\n`

test('openai: emits text deltas and finish', async () => {
  const response = makeResponse([
    sse({
      choices: [{ delta: { role: 'assistant', content: 'Hello' }, finish_reason: null }],
    }),
    sse({ choices: [{ delta: { content: ', ' }, finish_reason: null }] }),
    sse({ choices: [{ delta: { content: 'world!' }, finish_reason: null }] }),
    sse({ choices: [{ delta: {}, finish_reason: 'stop' }] }),
    'data: [DONE]\n\n',
  ])

  const events = await collect(streamLLM(response, { provider: 'openai' }))
  assert.equal(deltaText(events), 'Hello, world!')
  const finish = events.find((e): e is Extract<LLMEvent, { type: 'finish' }> => e.type === 'finish')
  assert.ok(finish)
  assert.equal(finish.reason, 'stop')
})

test('openai: usage event when include_usage', async () => {
  const response = makeResponse([
    sse({ choices: [{ delta: { content: 'hi' }, finish_reason: null }] }),
    sse({ choices: [{ delta: {}, finish_reason: 'stop' }] }),
    sse({ choices: [], usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 } }),
    'data: [DONE]\n\n',
  ])

  const events = await collect(streamLLM(response, { provider: 'openai' }))
  const usage = events.find((e): e is Extract<LLMEvent, { type: 'usage' }> => e.type === 'usage')
  assert.ok(usage)
  assert.equal(usage.inputTokens, 10)
  assert.equal(usage.outputTokens, 5)
  assert.equal(usage.totalTokens, 15)
})

test('openai: accumulates tool call arguments across chunks', async () => {
  const response = makeResponse([
    sse({
      choices: [
        {
          delta: {
            tool_calls: [
              {
                index: 0,
                id: 'call_abc',
                function: { name: 'get_weather', arguments: '{"cit' },
              },
            ],
          },
          finish_reason: null,
        },
      ],
    }),
    sse({
      choices: [
        {
          delta: {
            tool_calls: [{ index: 0, function: { arguments: 'y":"SF"}' } }],
          },
          finish_reason: null,
        },
      ],
    }),
    sse({ choices: [{ delta: {}, finish_reason: 'tool_calls' }] }),
    'data: [DONE]\n\n',
  ])

  const events = await collect(streamLLM(response, { provider: 'openai' }))
  const deltas = events.filter(
    (e): e is Extract<LLMEvent, { type: 'tool_call_delta' }> =>
      e.type === 'tool_call_delta',
  )
  assert.equal(deltas.length, 2)
  const call = events.find(
    (e): e is Extract<LLMEvent, { type: 'tool_call' }> => e.type === 'tool_call',
  )
  assert.ok(call)
  assert.equal(call.id, 'call_abc')
  assert.equal(call.name, 'get_weather')
  assert.deepEqual(call.args, { city: 'SF' })
  const finish = events.find((e): e is Extract<LLMEvent, { type: 'finish' }> => e.type === 'finish')
  assert.equal(finish?.reason, 'tool_calls')
})

test('openai: survives cross-chunk boundaries in SSE framing', async () => {
  // Split a single data: line across two network chunks to make sure the
  // parser's line-buffer does its job.
  const payload = sse({ choices: [{ delta: { content: 'abc' }, finish_reason: null }] })
  const mid = Math.floor(payload.length / 2)
  const response = makeResponse([
    payload.slice(0, mid),
    payload.slice(mid),
    sse({ choices: [{ delta: {}, finish_reason: 'stop' }] }),
  ])
  const events = await collect(streamLLM(response, { provider: 'openai' }))
  assert.equal(deltaText(events), 'abc')
})

test('openai: malformed JSON yields error event without stopping', async () => {
  const response = makeResponse([
    'data: {not json}\n\n',
    sse({ choices: [{ delta: { content: 'ok' }, finish_reason: 'stop' }] }),
  ])
  const events = await collect(streamLLM(response, { provider: 'openai' }))
  assert.ok(events.some((e) => e.type === 'error'))
  assert.equal(deltaText(events), 'ok')
})

test('openai: non-2xx response surfaces error + finish', async () => {
  const response = makeResponse(['{"error":"bad request"}'], {
    status: 400,
    statusText: 'Bad Request',
  })
  const events = await collect(streamLLM(response, { provider: 'openai' }))
  assert.equal(events[0].type, 'error')
  assert.equal(events[1].type, 'finish')
  assert.equal((events[1] as Extract<LLMEvent, { type: 'finish' }>).reason, 'error')
})
