import json
import os

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic()

def calculator(a, op, b):
  try:
    a = float(a)
    b = float(b)
  except (TypeError, ValueError):
    return "Error: a and b must be numbers."

  if op == "+":
    return str(a + b)
  elif op == "-":
    return str(a - b)
  elif op == "*":
    return str(a * b)
  elif op == "/":
    if b == 0:
      return "Error: division by zero."
    return str(a / b)
  else:
    return "Error: op must be one of + - * /."

TOOLS = [
  {
      "name": "calculator",
      "description": "Do arithmetic. Use this for any math. Do not invent the answer.",
      "input_schema": {
        "type": "object",
        "properties": {
          "a": {"type": "number"},
          "op": {"type": "string", "enum": ["+", "-", "*", "/"]},
          "b": {"type": "number"}
        },
        "required": ["a", "op", "b"]
      }
    }
]

def chat(messages, tools):
  response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    system="You are a careful assistant. For any math you MUST call the calculator tool. Never invent a numeric answer. For non-math questions, answer normally and do not call tools.",
    messages=messages,
    tools=tools
  )
  return response

def run_agent(user_text, max_turns=8):
  messages = [{"role": "user", "content": user_text}]

  for turn in range(1, max_turns + 1):
      print(f"\n--- turn {turn} ---")
      response = chat(messages, TOOLS)

      if response.stop_reason != "tool_use":
          text = "".join(b.text for b in response.content if b.type == "text")
          print("final:", text)
          return text

      messages.append({"role": "assistant", "content": response.content})

      results = []
      for block in response.content:
          if block.type != "tool_use":
              continue
          if block.name == "calculator":
              result = calculator(**block.input)
          else:
              result = f"Error: unknown tool {block.name}"
          print(f"tool: {block.name}({block.input})")
          print("result:", result)
          results.append({
              "type": "tool_result",
              "tool_use_id": block.id,
              "content": result,
          })

      messages.append({"role": "user", "content": results})

  print("stopped: hit max turns")
  return None


if __name__ == "__main__":
  question = input("You: ")
  run_agent(question)