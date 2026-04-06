// Run with: ANTHROPIC_API_KEY=sk-ant-... node examples/anthropic-example.js
import { streamLLM } from '../dist/index.js'

const response = await fetch('https://api.anthropic.com/v1/messages', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'x-api-key': process.env.ANTHROPIC_API_KEY,
    'anthropic-version': '2023-06-01',
  },
  body: JSON.stringify({
    model: 'claude-opus-4-6',
    max_tokens: 256,
    stream: true,
    messages: [{ role: 'user', content: 'Write a haiku about streams.' }],
  }),
})

for await (const event of streamLLM(response, { provider: 'anthropic' })) {
  switch (event.type) {
    case 'delta':
      process.stdout.write(event.text)
      break
    case 'tool_call':
      console.log('\n[tool_call]', event.name, event.args)
      break
    case 'usage':
      console.log(
        '\n[usage] in:',
        event.inputTokens,
        'out:',
        event.outputTokens,
      )
      break
    case 'finish':
      console.log('\n[finish]', event.reason)
      break
    case 'error':
      console.error('\n[error]', event.message)
      break
  }
}
