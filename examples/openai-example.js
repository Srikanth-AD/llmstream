// Run with: OPENAI_API_KEY=sk-... node examples/openai-example.js
import { streamLLM } from '../dist/index.js'

const response = await fetch('https://api.openai.com/v1/chat/completions', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
  },
  body: JSON.stringify({
    model: 'gpt-4o',
    messages: [{ role: 'user', content: 'Write a haiku about streams.' }],
    stream: true,
    stream_options: { include_usage: true },
  }),
})

for await (const event of streamLLM(response, { provider: 'openai' })) {
  switch (event.type) {
    case 'delta':
      process.stdout.write(event.text)
      break
    case 'tool_call':
      console.log('\n[tool_call]', event.name, event.args)
      break
    case 'usage':
      console.log('\n[usage]', event.totalTokens, 'tokens')
      break
    case 'finish':
      console.log('\n[finish]', event.reason)
      break
    case 'error':
      console.error('\n[error]', event.message)
      break
  }
}
