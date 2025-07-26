from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

completion = client.chat.completions.create(
  extra_body={},
  model="qwen/qwen3-coder:free",
  messages=[
    {
      "role": "user",
      "content": "What is the meaning of life?"
    }
  ]
)
print(completion.choices[0].message.content)