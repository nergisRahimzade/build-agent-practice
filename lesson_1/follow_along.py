"""
Lesson 1 reference. Study this, then write your own main.py.
Do not just rename this file and call it done.
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# --- 1. The real function. The model cannot run this. You can. ---

def calculator(a, op, b):
    try:
        a = float(a)
        b = float(b)
    except (TypeError, ValueError):
        return "Error: a and b must be numbers."

    if op == "+":
        return str(a + b)
    if op == "-":
        return str(a - b)
    if op == "*":
        return str(a * b)
    if op == "/":
        if b == 0:
            return "Error: division by zero."
        return str(a / b)
    return "Error: op must be one of + - * /."


# --- 2. The contract the model reads. ---

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Do arithmetic. Use this for any math. Do not invent the answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "op": {"type": "string", "enum": ["+", "-", "*", "/"]},
                    "b": {"type": "number"},
                },
                "required": ["a", "op", "b"],
            },
        },
    }
]


# --- 3. One API call. Returns the assistant message. ---

def chat(messages, tools):
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=tools,
    )
    return response.choices[0].message


# --- 4. The loop. This is the agent. ---

def run_agent(user_text, max_turns=8):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful assistant. "
                "For any math, you MUST call the calculator tool. "
                "Never invent a numeric answer. "
                "For non-math questions, answer normally and do not call tools."
            ),
        },
        {"role": "user", "content": user_text},
    ]

    for turn in range(1, max_turns + 1):
        print(f"\n--- turn {turn} ---")
        message = chat(messages, TOOLS)

        # The API wants this message back as a dict, not the SDK object.
        messages.append(message)

        if not message.tool_calls:
            print("final:", message.content)
            return message.content

        for call in message.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)
            print(f"tool: {name}({args})")

            if name == "calculator":
                result = calculator(**args)
            else:
                result = f"Error: unknown tool {name}"

            print("result:", result)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                }
            )

    print("stopped: hit max turns")
    return None


if __name__ == "__main__":
    question = input("You: ")
    run_agent(question)
