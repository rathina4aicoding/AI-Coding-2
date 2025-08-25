# gradio_openai_tools_demo.py
# Run: pip install -U gradio openai requests
# Set:  export OPENAI_API_KEY=sk-...   (or on Windows: set OPENAI_API_KEY=sk-...)
#
# This fixes the "Invalid 'messages[*].tool_calls': empty array" error by
# ONLY adding the 'tool_calls' field when it is non-empty.

import os
import json
from dotenv import load_dotenv
import time
import sqlite3
import requests
from typing import List, Dict, Any

import gradio as gr
from openai import OpenAI


# ------------------ API Keys retrivals ------------------

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

MODEL = "gpt-4o-mini"  # or "gpt-3.5-turbo", etc.
openai = OpenAI()

# ------------------ Tools (as provided) ------------------

def get_exchange_rate(base: str, target: str) -> dict:
    """
    Returns {"base": ..., "target": ..., "rate": float} or {"error": "..."}.
    Tries Frankfurter first, then open.er-api as a fallback.
    """
    base = (base or "").upper().strip()
    target = (target or "").upper().strip()
    if not base or not target or base == target:
        return {"error": "Provide two different ISO currency codes, e.g., base='USD', target='EUR'."}

    # 1) Primary: Frankfurter (ECB-backed; simple, no key)
    try:
        url = f"https://api.frankfurter.app/latest?from={base}&to={target}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        rate = data.get("rates", {}).get(target)
        if rate is not None:
            return {"base": base, "target": target, "rate": float(rate)}
    except Exception:
        pass  # continue to fallback

    # 2) Fallback: open.er-api (broad coverage; simple, no key)
    try:
        url = f"https://open.er-api.com/v6/latest/{base}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("result") == "success":
            rates = data.get("rates", {})
            if target in rates:
                return {"base": base, "target": target, "rate": float(rates[target])}
            return {"error": f"Target currency '{target}' not found in rates."}
        else:
            return {"error": f"FX API error: {data.get('error-type') or data}"}
    except Exception as e:
        return {"error": f"Network/API error while fetching rates: {e}"}

def count_customers_by_city(city: str) -> dict:
    """
    Simple DB retrieval example using SQLite.
    Creates a tiny demo table if it doesn't exist and counts rows for a city.
    """
    conn = sqlite3.connect("demo.db")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY,
            name TEXT,
            city TEXT
        )
    """)
    # Seed a few rows once (idempotent insert pattern)
    cur.execute("SELECT COUNT(*) FROM customers")
    total = cur.fetchone()[0]
    if total == 0:
        cur.executemany(
            "INSERT INTO customers(name, city) VALUES(?,?)",
            [("Alice","New York"), ("Bob","Boston"), ("Chandra","New York"), ("Diego","Austin")]
        )
        conn.commit()

    cur.execute("SELECT COUNT(*) FROM customers WHERE city = ?", (city,))
    count = cur.fetchone()[0]
    conn.close()
    return {"city": city, "count": count}

# ------------------ Tool schema (OpenAI function calling) ------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": "Get the FX rate from a base currency to a target currency.",
            "parameters": {
                "type": "object",
                "properties": {
                    "base": {"type": "string", "description": "Base currency code, e.g. USD"},
                    "target": {"type": "string", "description": "Target currency code, e.g. EUR"},
                },
                "required": ["base", "target"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_customers_by_city",
            "description": "Count customers in a given city using a local SQLite demo DB.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name to look up."}
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are a helpful assistant. If tools are relevant, call them. "
    "Be concise and return final answers clearly."
)

# ------------------ Core two-call logic (NO empty tool_calls field) ------------------

def two_call_with_tools(messages: List[Dict[str, Any]]) -> str:
    """
    1) First call: let the model decide whether to call tools.
    2) If tool_calls present, execute locally and append results.
    3) Second call: model composes final answer.
    """
    first = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOLS
    )

    assistant_msg = first.choices[0].message

    # Build assistant record WITHOUT an empty tool_calls field
    assistant_record: Dict[str, Any] = {"role": "assistant"}
    # Only include content if present (can be None on tool calls)
    if getattr(assistant_msg, "content", None) is not None:
        assistant_record["content"] = assistant_msg.content
    # Only include tool_calls if non-empty
    tcalls = getattr(assistant_msg, "tool_calls", None)
    if tcalls:
        assistant_record["tool_calls"] = tcalls

    messages.append(assistant_record)

    # If tools were requested, execute and append tool outputs
    if tcalls:
        for tc in tcalls:
            fn_name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")

            if fn_name == "get_exchange_rate":
                result = get_exchange_rate(**args)
            elif fn_name == "count_customers_by_city":
                result = count_customers_by_city(**args)
            else:
                result = {"error": f"Unknown tool {fn_name}"}

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,  # must match assistant tool_call id
                "content": json.dumps(result)
            })

        # Second call to compose final answer
        final = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return final.choices[0].message.content or ""
    else:
        # No tool calls—model answered directly
        return assistant_msg.content or ""

# ------------------ Gradio UI ------------------

def chat_fn(message, history, state_messages):
    """
    Gradio ChatInterface(type="messages") passes:
      - message: {'role': 'user', 'content': '...'}
      - history: list[{'role': ..., 'content': ...}, ...]  (not used for API calls)
      - state_messages: we persist raw OpenAI messages here
    """
    messages = state_messages or [{"role": "system", "content": SYSTEM_PROMPT}]

    # Accept dict or str from Gradio
    user_content = message.get("content", "") if isinstance(message, dict) else str(message)
    messages.append({"role": "user", "content": user_content})

    try:
        assistant_reply = two_call_with_tools(messages)
    except Exception as e:
        assistant_reply = f"Error: {e}"

    # Return assistant reply and updated state
    return assistant_reply, messages

with gr.Blocks(title="OpenAI Tools via Gradio") as demo:
    gr.Markdown("## OpenAI Tools Demo (Two-Call Pattern with Gradio)")
    state = gr.State([])

    chatbot_component = gr.Chatbot(
        type="messages",         # important: match ChatInterface type
        height=420,
        show_copy_button=True
    )

    gr.ChatInterface(
        fn=chat_fn,
        type="messages",
        chatbot=chatbot_component,
        textbox=gr.Textbox(
            placeholder="Ask: 'Convert USD to EUR' or 'How many customers in New York?'",
            container=True,
        ),
        additional_inputs=[state],
        additional_outputs=[state],
    )

if __name__ == "__main__":
    demo.launch()
