import { test } from 'node:test'
import assert from 'node:assert/strict'
import { streamLLM } from '../src/index'
import type { LLMEvent } from '../src/types'
import { makeResponse, collect, deltaText } from './helpers'

const sse = (obj: unknown) => `data: ${JSON.stringify(obj)}\n\n`

test('google: text deltas, usage, finish', async () => {
  const response = makeResponse([
    sse({
      candidates: [
        { content: { parts: [{ text: 'Hello ' }], role: 'model' }, index: 0 },
      ],
    }),
    sse({
      candidates: [
        {
          content: { parts: [{ text: 'world' }], role: 'model' },
          finishReason: 'STOP',
          index: 0,
        },
      ],
      usageMetadata: {
        promptTokenCount: 4,
        candidatesTokenCount: 2,
        totalTokenCount: 6,
      },
    }),
  ])

  const events = await collect(streamLLM(response, { provider: 'google' }))
  assert.equal(deltaText(events), 'Hello world')
  const usage = events.find((e): e is Extract<LLMEvent, { type: 'usage' }> => e.type === 'usage')
  assert.equal(usage?.totalTokens, 6)
  const finish = events.find((e): e is Extract<LLMEvent, { type: 'finish' }> => e.type === 'finish')
  assert.equal(finish?.reason, 'stop')
})

test('google: functionCall part emits tool_call', async () => {
  const response = makeResponse([
    sse({
      candidates: [
        {
          content: {
            parts: [
              { functionCall: { name: 'get_weather', args: { city: 'LA' } } },
            ],
            role: 'model',
          },
          finishReason: 'STOP',
          index: 0,
        },
      ],
    }),
  ])
  const events = await collect(streamLLM(response, { provider: 'google' }))
  const call = events.find(
    (e): e is Extract<LLMEvent, { type: 'tool_call' }> => e.type === 'tool_call',
  )
  assert.ok(call)
  assert.equal(call.name, 'get_weather')
  assert.deepEqual(call.args, { city: 'LA' })
})

test('google: SAFETY finishReason maps to content_filter', async () => {
  const response = makeResponse([
    sse({
      candidates: [
        { content: { parts: [{ text: 'partial' }] }, finishReason: 'SAFETY', index: 0 },
      ],
    }),
  ])
  const events = await collect(streamLLM(response, { provider: 'google' }))
  const finish = events.find((e): e is Extract<LLMEvent, { type: 'finish' }> => e.type === 'finish')
  assert.equal(finish?.reason, 'content_filter')
})
