import { test } from 'node:test'
import assert from 'node:assert/strict'
import { streamLLM } from '../src/index'
import type { LLMEvent } from '../src/types'
import { makeResponse, collect, deltaText } from './helpers'

const ndjson = (obj: unknown) => JSON.stringify(obj) + '\n'

test('ollama: /api/generate response field', async () => {
  const response = makeResponse([
    ndjson({ model: 'llama3', response: 'Hi ', done: false }),
    ndjson({ model: 'llama3', response: 'there', done: false }),
    ndjson({
      model: 'llama3',
      response: '',
      done: true,
      done_reason: 'stop',
      prompt_eval_count: 8,
      eval_count: 4,
    }),
  ])
  const events = await collect(streamLLM(response, { provider: 'ollama' }))
  assert.equal(deltaText(events), 'Hi there')
  const usage = events.find((e): e is Extract<LLMEvent, { type: 'usage' }> => e.type === 'usage')
  assert.equal(usage?.inputTokens, 8)
  assert.equal(usage?.outputTokens, 4)
  const finish = events.find((e): e is Extract<LLMEvent, { type: 'finish' }> => e.type === 'finish')
  assert.equal(finish?.reason, 'stop')
})

test('ollama: /api/chat message.content', async () => {
  const response = makeResponse([
    ndjson({ message: { role: 'assistant', content: 'Hello' }, done: false }),
    ndjson({ message: { role: 'assistant', content: '!' }, done: false }),
    ndjson({ message: { role: 'assistant', content: '' }, done: true, done_reason: 'stop' }),
  ])
  const events = await collect(streamLLM(response, { provider: 'ollama' }))
  assert.equal(deltaText(events), 'Hello!')
})

test('ollama: tool_calls produce tool_call event and tool_calls finish reason', async () => {
  const response = makeResponse([
    ndjson({
      message: {
        role: 'assistant',
        content: '',
        tool_calls: [
          { function: { name: 'lookup', arguments: { q: 'cats' } } },
        ],
      },
      done: true,
      done_reason: 'stop',
    }),
  ])
  const events = await collect(streamLLM(response, { provider: 'ollama' }))
  const call = events.find(
    (e): e is Extract<LLMEvent, { type: 'tool_call' }> => e.type === 'tool_call',
  )
  assert.ok(call)
  assert.equal(call.name, 'lookup')
  assert.deepEqual(call.args, { q: 'cats' })
  const finish = events.find((e): e is Extract<LLMEvent, { type: 'finish' }> => e.type === 'finish')
  assert.equal(finish?.reason, 'tool_calls')
})

test('ollama: NDJSON survives split-across-chunks line', async () => {
  const line = ndjson({ response: 'split-test', done: false })
  const cut = Math.floor(line.length / 2)
  const response = makeResponse([
    line.slice(0, cut),
    line.slice(cut),
    ndjson({ response: '', done: true, done_reason: 'stop' }),
  ])
  const events = await collect(streamLLM(response, { provider: 'ollama' }))
  assert.equal(deltaText(events), 'split-test')
})
