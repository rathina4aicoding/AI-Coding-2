"""
Gradio UI — Tools-Only LLM Demo (No RAG)
----------------------------------------
This app shows how an LLM can PLAN which tools to call and a controller EXECUTES those tools:
- Public APIs: Weather (Open-Meteo), Crypto (CoinGecko)
- Web Scraping: Hacker News headlines (requests + BeautifulSoup)
- Database: SQLite (toy ops metrics)
- Final Answer: LLM synthesizes only from tool outputs (no vector DB / RAG)

Great for classroom demos of "LLM + Tools" without retrieval-augmented generation.
"""

import os
import time
import sqlite3
from typing import List, Dict, Any

import requests
import gradio as gr
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# LangChain LLM plumbing (no RAG pieces imported)
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser



# =========================
# 0) Setup & Globals
# =========================
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

# Default model & temperature (conservative → factual)
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMP = 0.0

# Initialize an LLM lazily (we’ll rebuild if the user pastes a key in the UI)
def make_llm(api_key: str | None, model: str, temperature: float) -> ChatOpenAI:
    """
    Build a ChatOpenAI client.
    - If api_key is provided in the UI, use it for this process.
    - Else rely on OPENAI_API_KEY from the environment.
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY. Set env var or paste a key into the UI.")
    return ChatOpenAI(model=model, temperature=temperature)


# =========================
# 1) Tools — External APIs
# =========================

# Small city dictionary for weather lat/lon
CITY_LATLON = {
    "new york": (40.7128, -74.0060),
    "london": (51.5074, -0.1278),
    "chennai": (13.0827, 80.2707),
    "san francisco": (37.7749, -122.4194),
}

def get_weather(city: str = "New York") -> str:
    """
    Fetch current temperature & wind using Open-Meteo (no API key).
    Returns a short tagged string like: "[API/Weather] City: temp, wind ..."
    """
    try:
        latlon = CITY_LATLON.get(city.lower())
        if not latlon:
            return f"[API/Weather] Unknown city '{city}'. Choose: {', '.join(CITY_LATLON.keys())}"
        lat, lon = latlon
        url = (
            "https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m"
        )
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        j = r.json().get("current", {})
        t = j.get("temperature_2m", "NA")
        w = j.get("wind_speed_10m", "NA")
        return f"[API/Weather] {city.title()}: {t}°C, wind {w} m/s (Open-Meteo)"
    except Exception as e:
        return f"[API/Weather] Error: {e}"

def get_crypto_price() -> str:
    """
    Fetch BTC & ETH spot prices from CoinGecko (no key).
    Returns: "[API/Crypto] BTC $..., ETH $..."
    """
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        r = requests.get(url, params={"ids": "bitcoin,ethereum", "vs_currencies": "usd"}, timeout=15)
        r.raise_for_status()
        j = r.json()
        btc = j.get("bitcoin", {}).get("usd")
        eth = j.get("ethereum", {}).get("usd")
        if btc is None or eth is None:
            return "[API/Crypto] Unexpected response"
        return f"[API/Crypto] BTC ${btc:,} | ETH ${eth:,} (CoinGecko)"
    except Exception as e:
        return f"[API/Crypto] Error: {e}"


# =========================
# 2) Tool — Web Scraping
# =========================
def get_top_hn_titles(limit: int = 5) -> str:
    """
    Scrape Hacker News front page titles (lightweight demo).
    Returns: "[Web] Top HN:\n- Title 1\n- Title 2..."
    """
    try:
        url = "https://news.ycombinator.com/"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        titles = [a.get_text(strip=True) for a in soup.select(".titleline a")][:limit]
        if not titles:
            return "[Web] No headlines found."
        return "[Web] Top HN:\n- " + "\n- ".join(titles)
    except Exception as e:
        return f"[Web] Error: {e}"


# =========================
# 3) Tools — SQLite Database
# =========================
def init_db(db_path: str = ":memory:") -> sqlite3.Connection:
    """
    Create & seed a tiny ops metrics database in memory.
    Table: pipeline_runs(id, pipeline, started_at, duration_sec, status)
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id INTEGER PRIMARY KEY,
            pipeline TEXT,
            started_at TEXT,
            duration_sec INTEGER,
            status TEXT
        )
    """)
    rows = [
        ("daily_clearing", "2025-09-10 02:00:05", 420, "success"),
        ("daily_clearing", "2025-09-11 02:00:07", 455, "success"),
        ("daily_clearing", "2025-09-12 02:00:04", 980, "failed"),
        ("aml_screening", "2025-09-12 03:10:00", 310, "success"),
        ("aml_screening", "2025-09-13 03:10:02", 325, "success"),
        ("card_settlement", "2025-09-12 04:30:00", 600, "success"),
        ("card_settlement", "2025-09-13 04:30:03", 615, "success"),
    ]
    c.executemany(
        "INSERT INTO pipeline_runs(pipeline, started_at, duration_sec, status) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return conn

DB = init_db()

def db_query(query: str) -> str:
    """
    Execute a read-only SELECT against the toy DB and return a compact text block.
    For safety, only SELECT is allowed in this demo.
    """
    try:
        q = query.strip().lower()
        if not q.startswith("select"):
            return "[DB] Only SELECT allowed in this demo."
        cur = DB.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        if not rows:
            return "[DB] No rows."
        head = rows[:8]  # show only first 8 rows for brevity
        rendered = "\n".join("- " + ", ".join(str(x) for x in r) for r in head)
        suffix = "" if len(rows) <= 8 else f"\n  ...(+{len(rows)-8} more)"
        return "[DB] Rows:\n" + rendered + suffix
    except Exception as e:
        return f"[DB] Error: {e}"

def db_health_summary() -> str:
    """
    Convenience report: summarize pipeline runs, total duration, and failures by pipeline.
    """
    q = """
    SELECT pipeline,
           COUNT(*) AS runs,
           SUM(duration_sec) AS total_secs,
           SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failures
    FROM pipeline_runs
    GROUP BY pipeline
    ORDER BY failures DESC, total_secs DESC;
    """
    return db_query(q)


# =========================
# 4) Orchestrator — Planner & Synthesizer
# =========================

# (A) PLANNER prompt: ask the LLM to decide which tools to run.
PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a precise planner. Given a user question, decide which tools to call "
         "from this list: WEATHER, CRYPTO, HEADLINES, DB_HEALTH, DB_SQL. "
         "Return a short bullet list like:\n"
         "- WEATHER(city=...)\n- CRYPTO\n- HEADLINES\n- DB_HEALTH\n- DB_SQL(sql=...)\n"
         "Include only relevant tools. If none are relevant, return 'NONE'."),
        ("human", "{q}")
    ]
)
# (B) SYNTH prompt: build the final answer using ONLY tool outputs.
SYNTH_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a careful assistant. Use ONLY the tool outputs provided. "
         "If something isn't present, say you don't know. Keep it concise and tag sources like [API] [Web] [DB]."),
        ("human", "Question:\n{q}\n\nTool outputs:\n{ctx}\n\nAnswer:")
    ]
)
OUT_PARSER = StrOutputParser()

def extract_params(line: str, key: str) -> str:
    """
    Tiny helper to pull parameter values from strings like:
      WEATHER(city=chennai)  -> key="city=" → returns "chennai"
      DB_SQL(sql=select *...) -> key="sql="  → returns "select *..."
    """
    try:
        if key not in line:
            return ""
        start = line.index(key) + len(key)
        # Strip trailing parentheses, commas, and whitespace
        val = line[start:].strip().rstrip("),").strip()
        return val
    except Exception:
        return ""

def run_tools_controller(user_q: str, api_key: str | None, model: str, temperature: float) -> Dict[str, Any]:
    """
    End-to-end controller:
      1) Ask LLM for a tiny tool plan (PLANNER_PROMPT).
      2) Execute selected tools safely (no side effects).
      3) Ask LLM to synthesize a final answer using ONLY tool outputs (SYNTH_PROMPT).
    """
    llm = make_llm(api_key, model, temperature)

    # 1) Plan which tools to call
    plan = (PLANNER_PROMPT | llm | OUT_PARSER).invoke({"q": user_q}).strip()
    plan_lc = plan.lower()

    # 2) Execute tools per plan
    outputs: List[str] = []

    # WEATHER
    if "weather" in plan_lc:
        # Try to read explicit "city=" from plan; else infer from user question; else default
        city = extract_params(plan, "city=")
        if not city:
            for c in CITY_LATLON.keys():
                if c in user_q.lower():
                    city = c
                    break
        if not city:
            city = "new york"
        outputs.append(get_weather(city))

    # CRYPTO
    if "crypto" in plan_lc:
        outputs.append(get_crypto_price())

    # HEADLINES
    if "headlines" in plan_lc or "news" in plan_lc:
        outputs.append(get_top_hn_titles(5))

    # DB_HEALTH
    if "db_health" in plan_lc or "db-health" in plan_lc:
        outputs.append(db_health_summary())

    # DB_SQL(sql=...)
    if "db_sql" in plan_lc or "db-sql" in plan_lc:
        sql = extract_params(plan, "sql=")
        if not sql or not sql.strip().lower().startswith("select"):
            outputs.append("[DB] Skipped DB_SQL: missing/unsafe SQL (only SELECT allowed).")
        else:
            outputs.append(db_query(sql))

    if not outputs:
        outputs.append("(Planner returned NONE or no relevant tools.)")

    # 3) Synthesize a final answer using ONLY tool outputs
    final = (SYNTH_PROMPT | llm | OUT_PARSER).invoke(
        {"q": user_q, "ctx": "\n\n".join(outputs)}
    ).strip()

    return {"plan": plan, "tool_outputs": outputs, "final_answer": final}


# =========================
# 5) Gradio UI
# =========================
with gr.Blocks(title="Tools-Only LLM Demo (No RAG)") as demo:
    # Header / instructions
    gr.Markdown(
        """
        # 🔧 Tools Usage in LLMs - Demo
        This app shows **LLM + tools** (APIs, Web scraping, DB) without any vector DB or RAG.
        
        **How it works**
        1. The LLM proposes a tiny **plan** of which tools to call (weather/crypto/headlines/DB).  
        2. The app **executes** those tools.  
        3. The LLM writes a **final answer** using **only** the tool outputs.
        """
    )

    # Controls: API key & model settings
    with gr.Row():
        api_key_box = gr.Textbox(
            label="OpenAI API Key (optional if set in environment)",
            placeholder="sk-...",
            type="password",
        )
        model_box = gr.Dropdown(
            choices=["gpt-4o-mini"], value="gpt-4o-mini", label="LLM model"
        )
        temp_box = gr.Slider(0.0, 1.0, step=0.1, value=0.0, label="Temperature")

    # Question input and examples
    with gr.Row():
        question_box = gr.Textbox(
            label="Ask a question",
            lines=2,
            placeholder="e.g., What's the weather in Chennai and current BTC/ETH prices? Also show one tech headline.",
        )
    with gr.Row():
        examples = gr.Dropdown(
            label="Example prompts",
            choices=[
                "What’s the weather in Chennai and the current BTC/ETH prices? Also show one tech headline.",
                "Summarize pipeline health from the database, highlighting any failures.",
                "Give me a short market-and-ops snapshot: crypto, one tech news, and any failed pipeline counts from DB.",
                "Run a custom SQL to show all daily_clearing runs ordered by duration desc."
            ],
            value=None,
            interactive=True
        )
        use_example_btn = gr.Button("Use Example")

    # Action button
    run_btn = gr.Button("🚀 Run Tools Plan", variant="primary")

    # Outputs: plan, final answer, tool outputs, and latency
    with gr.Row():
        plan_md = gr.Markdown(label="Plan")
        final_md = gr.Markdown(label="Final Answer")
    tool_outputs_box = gr.Textbox(
        label="Tool Outputs (raw)",
        lines=14,
        interactive=False,
    )
    latency_md = gr.Markdown()

    # Wire example selector to fill the question box
    def use_example(ex):
        return ex or ""
    use_example_btn.click(
        use_example,
        inputs=[examples],
        outputs=[question_box]
    )

    # Main click handler: run the controller and show results
    def on_run(api_key, model, temp, q):
        if not q or not q.strip():
            return ("Please enter a question.",
                    "—",
                    "—",
                    "")

        t0 = time.time()
        try:
            result = run_tools_controller(q.strip(), api_key, model, float(temp))
        except Exception as e:
            return (f"❌ Error: {e}", "—", "—", "")
        t1 = time.time()

        # Pretty-print the tool outputs
        outputs_text = "\n\n".join(result["tool_outputs"])
        latency = f"**Latency:** {t1 - t0:.2f}s"

        return (f"### Plan\n{result['plan']}",
                result["final_answer"],
                outputs_text,
                latency)

    run_btn.click(
        on_run,
        inputs=[api_key_box, model_box, temp_box, question_box],
        outputs=[plan_md, final_md, tool_outputs_box, latency_md],
        api_name="run_tools"
    )

# Entrypoint
if __name__ == "__main__":
    demo.launch()
