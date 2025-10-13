"""
RAG with Caching & Memory (Contextual Recall) — Gradio Demo (LangChain)
Fix: thread-safe SQLite usage for AnswerCache + queued Gradio callbacks.
"""

import os
import time
import hashlib
import sqlite3
import threading
from typing import List, Tuple, Dict, Any, Optional

import gradio as gr
from dotenv import load_dotenv

# ---- LangChain Core
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableMap, RunnablePassthrough

# ---- LLM & Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ---- Vector store & loaders
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# ---- Caching (LLM cache)
import langchain
from langchain.cache import SQLiteCache

# Load environment variables in a file called .env
load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

# =========================
# 0) Setup & Constants
# =========================


DEFAULT_FOLDER = r"C:\RAG\docs"          # <-- change to your local PDFs folder
FAISS_CACHE_ROOT = ".faiss_cache"        # persisted vector indexes
os.makedirs(FAISS_CACHE_ROOT, exist_ok=True)

LLM_MODEL = "gpt-4o-mini"
LLM_TEMP = 0.0
EMBEDDING_MODEL = "text-embedding-3-small"

# Enable a persistent LLM output cache (LangChain's own cache is thread-safe)
langchain.llm_cache = SQLiteCache(database_path=".llm_cache.sqlite")

# =========================
# 1) Utilities — Files/PDF
# =========================
def list_pdf_paths(folder: str, recursive: bool = True) -> List[str]:
    if not folder or not os.path.isdir(folder):
        raise ValueError(f"Invalid folder: {folder}")
    pdfs: List[str] = []
    if recursive:
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(".pdf"):
                    pdfs.append(os.path.join(root, f))
    else:
        for f in os.listdir(folder):
            if f.lower().endswith(".pdf"):
                pdfs.append(os.path.join(folder, f))
    if not pdfs:
        raise ValueError(f"No PDF files found under: {folder}")
    return sorted(pdfs)

def load_pdfs_as_documents(folder: str) -> List[Document]:
    docs: List[Document] = []
    for path in list_pdf_paths(folder):
        loader = PyPDFLoader(path)
        pages = loader.load()
        for d in pages:
            d.metadata["source"] = os.path.basename(path)
        docs.extend(pages)
    return docs

# =========================
# 2) FAISS Index Build/Load (Persistent)
# =========================
def sha256_of_files(paths: List[str]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(os.path.basename(p).encode("utf-8"))
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()

def build_or_load_faiss(folder: str,
                        chunk_size: int = 500,
                        chunk_overlap: int = 80) -> Tuple[FAISS, Dict[str, Any]]:
    pdfs = list_pdf_paths(folder)
    key = f"{sha256_of_files(pdfs)}__{EMBEDDING_MODEL}__{chunk_size}__{chunk_overlap}"
    cache_dir = os.path.join(FAISS_CACHE_ROOT, key)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    if os.path.isdir(cache_dir):
        vectorstore = FAISS.load_local(cache_dir, embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
        return retriever, {"cache_hit": True, "cache_dir": cache_dir, "pdfs": len(pdfs), "pages": "cached"}

    docs = load_pdfs_as_documents(folder)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    os.makedirs(cache_dir, exist_ok=True)
    vectorstore.save_local(cache_dir)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    return retriever, {
        "cache_hit": False,
        "cache_dir": cache_dir,
        "pdfs": len(pdfs),
        "pages": len(docs),
        "chunks": len(chunks)
    }

# =========================
# 3) Answer Cache (SQLite) — thread-safe
# =========================
class AnswerCache:
    """
    Thread-safe, per-call SQLite connections with check_same_thread=False.
    Key = hash(normalized_question + sources_signature)
    """
    def __init__(self, path: str = ".answer_cache.sqlite"):
        self.path = path
        self._lock = threading.Lock()
        # init table using a dedicated connection
        with sqlite3.connect(self.path, check_same_thread=False) as conn:
            cur = conn.cursor()
            cur.execute("""
              CREATE TABLE IF NOT EXISTS answers (
                key TEXT PRIMARY KEY,
                question TEXT,
                sources_sig TEXT,
                answer TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
              )
            """)
            conn.commit()

    def _make_key(self, question: str, sources_sig: str) -> str:
        h = hashlib.sha256()
        h.update(question.strip().lower().encode("utf-8"))
        h.update(b"||")
        h.update(sources_sig.encode("utf-8"))
        return h.hexdigest()

    def get(self, question: str, sources_sig: str) -> Optional[str]:
        key = self._make_key(question, sources_sig)
        with self._lock, sqlite3.connect(self.path, check_same_thread=False) as conn:
            cur = conn.cursor()
            cur.execute("SELECT answer FROM answers WHERE key=?", (key,))
            row = cur.fetchone()
            return row[0] if row else None

    def set(self, question: str, sources_sig: str, answer: str):
        key = self._make_key(question, sources_sig)
        with self._lock, sqlite3.connect(self.path, check_same_thread=False) as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT OR REPLACE INTO answers(key, question, sources_sig, answer) VALUES (?, ?, ?, ?)",
                (key, question, sources_sig, answer),
            )
            conn.commit()

ANSWER_CACHE = AnswerCache()

# =========================
# 4) Conversational Memory
# =========================
class EpisodicMemory:
    """Simple vector memory of past Q/A."""
    def __init__(self):
        self._emb = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self._store = None
        self._lock = threading.Lock()

    def add(self, question: str, answer: str):
        docs = [Document(page_content=f"Q: {question}\nA: {answer}", metadata={"type": "memory"})]
        with self._lock:
            if self._store is None:
                self._store = FAISS.from_documents(docs, embedding=self._emb)
            else:
                self._store.add_documents(docs)

    def recall(self, query: str, k: int = 3) -> List[Document]:
        with self._lock:
            if self._store is None:
                return []
            return self._store.similarity_search(query, k=k)

EPISODIC = EpisodicMemory()

class ShortTermMemory:
    """Rolling window of last N (user, assistant) turns."""
    def __init__(self, window: int = 4):
        self.window = window
        self.turns: List[Tuple[str, str]] = []
        self._lock = threading.Lock()

    def add(self, user: str, assistant: str):
        with self._lock:
            self.turns.append((user, assistant))
            if len(self.turns) > self.window:
                self.turns = self.turns[-self.window:]

    def render(self) -> str:
        with self._lock:
            if not self.turns:
                return ""
            lines = []
            for u, a in self.turns:
                lines.append(f"User: {u}\nAssistant: {a}")
            return "\n\n".join(lines)

SHORT_TERM = ShortTermMemory(window=4)

# =========================
# 5) Prompts & helpers
# =========================
SYSTEM = (
    "You are a factual assistant. Use ONLY the provided context blocks. "
    "Context may include: (A) retrieved document chunks, (B) short-term chat memory, "
    "(C) episodic memory snippets. If required info is missing, say you don't know. "
    "Keep answers concise and cite sources like [filename p.X] when helpful."
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        ("human",
         "Question:\n{question}\n\n"
         "Context — Retrieved Docs:\n{retrieved}\n\n"
         "Context — Short-Term Memory:\n{short_mem}\n\n"
         "Context — Episodic Memory:\n{episodic}\n\n"
         "Answer using ONLY the above context.")
    ]
)

def format_docs(docs: List[Document]) -> str:
    lines = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        p = f" p.{page+1}" if isinstance(page, int) else ""
        lines.append(f"[{src}{p}] {d.page_content}")
    return "\n\n".join(lines) if lines else "(none)"

# =========================
# 6) Core QA Logic (with caching & memory)
# =========================
def answer_question(
    retriever,
    question: str,
    api_key: Optional[str],
    model_name: str = LLM_MODEL,
    temperature: float = LLM_TEMP,
    k: int = 6,
    use_answer_cache: bool = True,
    use_short_memory: bool = True,
    use_episodic_memory: bool = True,
) -> Dict[str, Any]:

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY. Provide it in the textbox or environment.")

    # 1) Retrieve document chunks
    retrieved_docs = retriever.get_relevant_documents(question)
    retrieved_block = format_docs(retrieved_docs)

    # 2) Collect memory blocks
    short_block = SHORT_TERM.render() if use_short_memory else ""
    episodic_docs = EPISODIC.recall(question, k=3) if use_episodic_memory else []
    episodic_block = "\n\n".join(d.page_content for d in episodic_docs) if episodic_docs else ""

    # 3) Build a sources signature for answer cache
    src_sig_items = []
    for d in retrieved_docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        src_sig_items.append(f"{src}:{page}")
    if episodic_docs:
        src_sig_items.append("episodic")
    sources_sig = "|".join(sorted(set(src_sig_items))) or "no-sources"

    # 4) Answer cache
    if use_answer_cache:
        cached = ANSWER_CACHE.get(question, sources_sig)
        if cached:
            return {
                "from_answer_cache": True,
                "answer": cached,
                "retrieved_sources": src_sig_items,
                "retrieved_docs": retrieved_docs,
                "episodic_used": bool(episodic_docs),
                "short_mem_used": bool(short_block),
            }

    # 5) Generate fresh answer (LLM output cache may kick in automatically)
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    chain = (
        RunnableMap({
            "question": RunnablePassthrough(),
            "retrieved": lambda q: retrieved_block,
            "short_mem": lambda q: (short_block if short_block else "(none)"),
            "episodic": lambda q: (episodic_block if episodic_block else "(none)")
        }) | PROMPT | llm | StrOutputParser()
    )
    answer_text = chain.invoke(question).strip()

    # 6) Update caches/memory
    if use_answer_cache:
        ANSWER_CACHE.set(question, sources_sig, answer_text)
    if use_episodic_memory:
        EPISODIC.add(question, answer_text)
    if use_short_memory:
        SHORT_TERM.add(question, answer_text)

    return {
        "from_answer_cache": False,
        "answer": answer_text,
        "retrieved_sources": src_sig_items,
        "retrieved_docs": retrieved_docs,
        "episodic_used": bool(episodic_docs),
        "short_mem_used": bool(short_block),
    }

# =========================
# 7) Gradio UI
# =========================
with gr.Blocks(title="RAG with Caching & Memory (LangChain)") as demo:
    gr.Markdown(
        """
        # 🧠 RAG with Caching & Memory (Contextual Recall)
        - **Document cache**: persisted FAISS index  
        - **LLM output cache**: identical prompts reuse cached generations  
        - **Answer cache**: question+sources signature → skip regeneration  
        - **Memory**: short-term window + episodic vector recall
        """
    )

    with gr.Row():
        api_key_box = gr.Textbox(
            label="OpenAI API Key (optional if set in environment)",
            placeholder="sk-...",
            type="password",
        )
        model_box = gr.Dropdown(choices=[LLM_MODEL], value=LLM_MODEL, label="LLM Model")
        temp_box = gr.Slider(0.0, 1.0, value=LLM_TEMP, step=0.1, label="Temperature")

    with gr.Accordion("Data source & index", open=True):
        folder_box = gr.Textbox(
            label="PDF folder (Windows/local)",
            value=DEFAULT_FOLDER,
            placeholder=r"C:\RAG\docs",
        )
        with gr.Row():
            chunk_size_box = gr.Slider(200, 1200, value=500, step=50, label="Chunk size")
            chunk_overlap_box = gr.Slider(0, 400, value=80, step=10, label="Chunk overlap")
        build_btn = gr.Button("🔧 Build / Load Index", variant="primary")
        index_status = gr.Markdown("")

    with gr.Accordion("Caching & Memory settings", open=True):
        with gr.Row():
            use_ans_cache = gr.Checkbox(value=True, label="Enable Answer Cache")
            use_short_mem = gr.Checkbox(value=True, label="Enable Short-Term Memory (last 4 turns)")
            use_epi_mem = gr.Checkbox(value=True, label="Enable Episodic Memory (vector recall)")

    question_box = gr.Textbox(
        label="Ask a question",
        lines=2,
        placeholder="e.g., What is Amazon Bedrock and how does it support RAG?",
    )
    ask_btn = gr.Button("💬 Ask", variant="primary")

    with gr.Row():
        answer_md = gr.Markdown(label="Answer")
        meta_md = gr.Markdown(label="Diagnostics")
    sources_box = gr.Textbox(label="Retrieved Sources (debug)", lines=6)

    state_retriever = gr.State()

    def on_build(api_key, model, temp, folder, csize, coverlap):
        try:
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            if not os.getenv("OPENAI_API_KEY"):
                return None, "❌ Missing OPENAI_API_KEY. Provide it in the textbox or environment."
            retriever, stats = build_or_load_faiss(folder, int(csize), int(coverlap))
            msg = (
                f"✅ Index ready. PDFs: {stats.get('pdfs')} | Pages: {stats.get('pages')} | "
                f"Cache: {'HIT' if stats.get('cache_hit') else 'MISS'} at `{stats.get('cache_dir')}`"
            )
            return retriever, msg
        except Exception as e:
            return None, f"❌ Error: {e}"

    build_btn.click(
        on_build,
        inputs=[api_key_box, model_box, temp_box, folder_box, chunk_size_box, chunk_overlap_box],
        outputs=[state_retriever, index_status],
        api_name="build_index"
    )

    def on_ask(retriever, api_key, model, temp, q, ans_cache, s_mem, e_mem):
        if retriever is None:
            return "Please build the index first.", "—", ""
        if not q or not q.strip():
            return "Please enter a question.", "—", ""
        t0 = time.time()
        try:
            result = answer_question(
                retriever=retriever,
                question=q.strip(),
                api_key=api_key,
                model_name=model,
                temperature=float(temp),
                use_answer_cache=bool(ans_cache),
                use_short_memory=bool(s_mem),
                use_episodic_memory=bool(e_mem),
            )
        except Exception as e:
            return f"❌ Error: {e}", "—", ""
        t1 = time.time()

        retrieved_docs = result.get("retrieved_docs", [])
        diag = (
            f"**Latency:** {t1 - t0:.2f}s  \n"
            f"**From Answer Cache?** {result.get('from_answer_cache')}  \n"
            f"**Short-Term Memory used?** {result.get('short_mem_used')}  \n"
            f"**Episodic Memory used?** {result.get('episodic_used')}  \n"
            f"**#Retrieved Chunks:** {len(retrieved_docs)}"
        )
        src_lines = []
        for d in retrieved_docs:
            src = d.metadata.get("source", "unknown")
            page = d.metadata.get("page", None)
            src_lines.append(f"{src}: p.{page+1 if isinstance(page, int) else '?'}")
        src_text = "\n".join(src_lines) if src_lines else "(none)"

        return result["answer"], diag, src_text

    ask_btn.click(
        on_ask,
        inputs=[state_retriever, api_key_box, model_box, temp_box, question_box, use_ans_cache, use_short_mem, use_epi_mem],
        outputs=[answer_md, meta_md, sources_box],
        api_name="ask"
    )

# Queue Gradio to serialize/limit worker threads (optional but recommended)
demo.queue(concurrency_count=2)

if __name__ == "__main__":

    try:
    # Gradio 4.x (most current)
       demo.queue(default_concurrency_limit=2)
    except TypeError:
        try:
           # Gradio 3.x fallback
           demo.queue(concurrency_count=2)
        except TypeError:
            # Some builds require no args   
            demo.queue()
    
    demo.launch(inbrowser=True, quiet=True)
