"""
Adaptive / Query-Aware RAG — PDF Folder Edition (LangChain + Gradio)
-------------------------------------------------------------------
What this demo teaches:
- How to do *query-aware* RAG: decide WHEN to retrieve from a KB vs. WHEN to answer from the LLM alone.
- The KB is built from **all PDF files in a local folder** (Windows path supported). No file upload UI.

How it works:
1) Click **Build Index** → we scan the folder, read every *.pdf, split pages to chunks, embed, and index in FAISS.
2) Ask a question:
   - A tiny router LLM predicts: "RETRIEVE" vs "LLM_ONLY" (with a one-line rationale).
   - If "RETRIEVE", we fetch top-N with relevance scores; a **confidence gate** decides if retrieval is strong enough.
   - If the gate fails → **fallback to LLM_ONLY** to avoid retrieval-induced hallucinations.
   - The UI shows: decision, rationale, top score, whether the gate passed, answer, and sources.

Dependencies:
  pip install -U gradio langchain langchain-openai langchain-community \
                 langchain-text-splitters faiss-cpu python-dotenv pypdf

Environment:
  export OPENAI_API_KEY=sk-...   (or paste it in the UI)

Run:
  python adaptive_rag_pdf_folder.py

Sample questions (copy/paste to try):
- What is Amazon Bedrock and how does it support RAG?
- Explain the steps of a typical RAG pipeline.
- Compare read vs write behavior in Apache Iceberg.
- Who won the 2024 FIFA World Cup?
- What are good chunking settings for Vector RAG?
- What is the capital of France?
"""

import os
import glob
from typing import List, Dict, Any, Tuple

import gradio as gr
from dotenv import load_dotenv

# LangChain bits
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables in a file called .env

load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
# =========================
# 0) Setup & Globals
# =========================


# Global state to avoid rebuilding the index on every question
VSTORE: FAISS | None = None
INDEX_STATS: Dict[str, Any] = {}
CURRENT_FOLDER: str | None = None


# =========================
# 1) Load PDFs from a Windows/Local folder
# =========================
def load_pdfs_from_dir(dir_path: str) -> List[Document]:
    """
    Read all *.pdf files under dir_path (non-recursive), return a list of LangChain Documents.
    Each PDF is loaded page-by-page; we attach `source=<filename>` and keep `page` metadata for citations.
    """
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Folder not found: {dir_path}")

    pdf_paths = sorted(glob.glob(os.path.join(dir_path, "*.pdf")))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in: {dir_path}")

    docs: List[Document] = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        pages = loader.load()  # one Document per page
        # ensure consistent 'source' metadata for citations
        for d in pages:
            d.metadata["source"] = os.path.basename(path)
        docs.extend(pages)

    return docs


# =========================
# 2) Build vector index (chunk → embed → FAISS)
# =========================
def build_vector_index_from_folder(
    dir_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    embed_model: str = "text-embedding-3-small",
) -> Tuple[FAISS, Dict[str, Any]]:
    """
    Build a FAISS vector index from all PDFs in dir_path.
    Returns (vectorstore, stats).
    """
    docs = load_pdfs_from_dir(dir_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model=embed_model)
    vstore = FAISS.from_documents(chunks, embedding=embeddings)

    stats = {
        "folder": dir_path,
        "pdf_count": len({d.metadata.get("source") for d in docs}),
        "pages": len(docs),
        "chunks": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embed_model": embed_model,
    }
    return vstore, stats


# =========================
# 3) Router (Query-aware decision): RETRIEVE vs LLM_ONLY
# =========================
ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a routing classifier for a QA system with a small internal knowledge base loaded from PDFs.\n"
         "Decide if the question likely requires DOCUMENT RETRIEVAL from the KB or can be answered from "
         "the model's general knowledge.\n\n"
         "Return JSON with fields:\n"
         '{{ "route": "RETRIEVE" | "LLM_ONLY", "rationale": "<one sentence why>" }}\n\n'
         "Use RETRIEVE if the question is about document-specific topics likely found in the PDFs or "
         "domain-specific details (e.g., Amazon Bedrock, RAG pipeline, chunking, Apache Iceberg specifics).\n"
         "Use LLM_ONLY for general world knowledge (e.g., capitals, generic facts) or if retrieval is unlikely to help."
        ),
        ("human", "Question: {q}")
    ]
)

def route_question(llm: ChatOpenAI, question: str) -> Dict[str, str]:
    """
    Ask a small router LLM which path to take.
    Returns dict like: {"route": "RETRIEVE", "rationale": "..."}.
    """
    raw = (ROUTER_PROMPT | llm | StrOutputParser()).invoke({"q": question})
    # very permissive parsing for demo robustness
    rl = raw.lower()
    if "llm_only" in rl or "llm-only" in rl:
        route = "LLM_ONLY"
    elif "retrieve" in rl:
        route = "RETRIEVE"
    else:
        route = "RETRIEVE"
    return {"route": route, "rationale": raw.strip()}


# =========================
# 4) Confidence gate helpers for retrieval
# =========================
def similarity_search_with_scores(vstore: FAISS, query: str, k: int = 5) -> List[Tuple[Document, float]]:
    """
    Try FAISS.similarity_search_with_relevance_scores; if not available, fall back to similarity_search_with_score.
    Normalizes to a list[(Document, score)] with higher=better in [0,1] if possible.
    """
    if hasattr(vstore, "similarity_search_with_relevance_scores"):
        return vstore.similarity_search_with_relevance_scores(query, k=k)

    # Fallback: similarity_search_with_score returns (Document, distance) with lower=better
    results = vstore.similarity_search_with_score(query, k=k)
    # Convert distances to rough "relevance" scores in (0,1], using 1 / (1 + distance)
    converted: List[Tuple[Document, float]] = []
    for doc, dist in results:
        try:
            score = 1.0 / (1.0 + float(dist))
        except Exception:
            score = 0.0
        converted.append((doc, score))
    # Sort by score descending just in case
    converted.sort(key=lambda x: x[1], reverse=True)
    return converted

def retrieval_confident_enough(results: List[Tuple[Document, float]], min_top_score: float) -> bool:
    """
    Simple gate: if the top hit is below threshold, we doubt retrieval will help.
    """
    if not results:
        return False
    return float(results[0][1]) >= float(min_top_score)


# =========================
# 5) Answer prompts — grounded vs general
# =========================
GROUNDED_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a factual assistant. Answer ONLY using the provided context.\n"
         "If the answer is not present, say you don't know.\n"
         "Be concise and include short citations like [source] when helpful."),
        ("human",
         "Question:\n{q}\n\n"
         "Context:\n{ctx}\n\n"
         "Answer using ONLY the context.")
    ]
)

GENERAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a careful assistant. If you are uncertain, say you don't know."),
        ("human", "{q}")
    ]
)

def format_docs(docs: List[Document]) -> str:
    """Combine docs into a single context string with simple [source] tags."""
    out = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", f"doc_{i}")
        page = d.metadata.get("page", None)
        page_str = f" (page {int(page)+1})" if isinstance(page, int) else ""
        out.append(f"[{src}{page_str}] {d.page_content}")
    return "\n\n".join(out)


# =========================
# 6) Adaptive QA (router → retrieval w/ gate → fallback)
# =========================
def adaptive_answer(
    question: str,
    api_key: str | None,
    model_name: str,
    temperature: float,
    k_candidates: int,
    min_retrieval_score: float,
) -> Dict[str, Any]:
    """
    - Route: RETRIEVE vs LLM_ONLY
    - If RETRIEVE: fetch candidates + gate by top score
    - If gate fails → fallback to LLM_ONLY
    - Return decision, rationale, scores, answer, and sources
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY. Provide it in the textbox or environment.")

    if VSTORE is None:
        raise RuntimeError("Vector index not built yet. Click 'Build Index' first.")

    llm = ChatOpenAI(model=model_name, temperature=temperature)

    # A) Router decision
    route_obj = route_question(llm, question)
    route, rationale = route_obj["route"], route_obj["rationale"]

    # B) Retrieval path with gate
    if route == "RETRIEVE":
        results = similarity_search_with_scores(VSTORE, question, k=k_candidates)
        retrieved_docs = [doc for doc, _ in results]
        top_score = float(results[0][1]) if results else None
        gate_ok = retrieval_confident_enough(results, min_top_score=min_retrieval_score)

        if gate_ok:
            ctx = format_docs(retrieved_docs)
            answer = (GROUNDED_PROMPT | llm | StrOutputParser()).invoke({"q": question, "ctx": ctx}).strip()
            return {
                "used_route": "RETRIEVE",
                "router_rationale": rationale,
                "top_score": top_score,
                "gate_passed": True,
                "answer": answer,
                "sources": [d.metadata.get("source") for d in retrieved_docs],
            }
        else:
            # Fallback to LLM_ONLY if retrieval looked weak
            answer = (GENERAL_PROMPT | llm | StrOutputParser()).invoke({"q": question}).strip()
            return {
                "used_route": "LLM_ONLY (fallback: low retrieval score)",
                "router_rationale": rationale,
                "top_score": top_score,
                "gate_passed": False,
                "answer": answer,
                "sources": [],
            }

    # C) LLM_ONLY route
    answer = (GENERAL_PROMPT | llm | StrOutputParser()).invoke({"q": question}).strip()
    return {
        "used_route": "LLM_ONLY",
        "router_rationale": rationale,
        "top_score": None,
        "gate_passed": None,
        "answer": answer,
        "sources": [],
    }


# =========================
# 7) Gradio UI
# =========================
with gr.Blocks(title="Adaptive / Query-Aware RAG — PDF Folder") as demo:
    gr.Markdown(
        """
        # 🧭 Adaptive / Query-Aware RAG — PDF Folder
        - Reads **all PDFs** from a local folder (Windows path supported).  
        - Decides when to **RETRIEVE** vs. answer from **LLM_ONLY**.  
        - Shows decision, rationale, relevance score, gate status, answer, and sources.
        """
    )

    with gr.Accordion("Build the Vector Index (PDF Folder)", open=True):
        with gr.Row():
            api_key = gr.Textbox(
                label="OpenAI API Key (optional if already set in environment)",
                placeholder="sk-...",
                type="password",
            )
            folder_path = gr.Textbox(
                label="PDF Folder Path (Windows/local)",
                value=r"C:\kb_pdfs",  # change this default to your local folder
                placeholder=r"C:\path\to\your\pdfs",
            )
        with gr.Row():
            chunk_size = gr.Slider(200, 1200, value=500, step=50, label="Chunk size")
            chunk_overlap = gr.Slider(0, 300, value=80, step=10, label="Chunk overlap")
            embed_model = gr.Dropdown(
                choices=["text-embedding-3-small", "text-embedding-3-large"],
                value="text-embedding-3-small",
                label="Embeddings model"
            )
        build_btn = gr.Button("🔧 Build Index", variant="primary")
        index_md = gr.Markdown(visible=False)
        status_md = gr.Markdown()

    gr.Markdown("---")

    with gr.Row():
        model_name = gr.Dropdown(choices=["gpt-4o-mini"], value="gpt-4o-mini", label="LLM")
        temperature = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Temperature")
        k_candidates = gr.Slider(3, 20, value=6, step=1, label="Retriever pool size (top-N)")
        min_score = gr.Slider(0.0, 1.0, value=0.25, step=0.01, label="Min top relevance to accept retrieval")

    question = gr.Textbox(
        label="Question",
        lines=2,
        placeholder="e.g., What is Amazon Bedrock and how does it support RAG?",
    )
    ask_btn = gr.Button("💡 Ask (Adaptive)")

    with gr.Row():
        decision_md = gr.Markdown(label="Decision & Rationale")
        scores_md = gr.Markdown(label="Scores / Gate")
    answer_md = gr.Markdown(label="Answer")
    sources_box = gr.Textbox(label="Sources (if retrieval used)", lines=5)

    # ---- Build index handler ----
    def on_build(api_key_v, folder_v, chunk_sz, chunk_ovl, embed_model_v):
        global VSTORE, INDEX_STATS, CURRENT_FOLDER

        try:
            if api_key_v:
                os.environ["OPENAI_API_KEY"] = api_key_v
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("Missing OPENAI_API_KEY. Provide it here or in the environment.")

            vstore, stats = build_vector_index_from_folder(
                dir_path=folder_v,
                chunk_size=int(chunk_sz),
                chunk_overlap=int(chunk_ovl),
                embed_model=embed_model_v,
            )
            VSTORE = vstore
            INDEX_STATS = stats
            CURRENT_FOLDER = folder_v

            summary = (
                f"**Index ready** for folder: `{stats['folder']}`  \n"
                f"- PDFs: **{stats['pdf_count']}** | Pages: **{stats['pages']}** | Chunks: **{stats['chunks']}**  \n"
                f"- Chunk size/overlap: **{stats['chunk_size']}** / **{stats['chunk_overlap']}**  \n"
                f"- Embeddings: **{stats['embed_model']}**"
            )
            return gr.update(value=summary, visible=True), "✅ Built successfully."
        except Exception as e:
            VSTORE = None
            INDEX_STATS = {}
            CURRENT_FOLDER = None
            return gr.update(value="", visible=False), f"❌ Error: {e}"

    build_btn.click(
        on_build,
        inputs=[api_key, folder_path, chunk_size, chunk_overlap, embed_model],
        outputs=[index_md, status_md],
        api_name="build_index",
    )

    # ---- Ask handler ----
    def on_ask(api_key_v, model_v, temp_v, k_v, min_v, q):
        if not q or not q.strip():
            return ("Please enter a question.", "—", "—", "")
        try:
            result = adaptive_answer(
                question=q.strip(),
                api_key=api_key_v,
                model_name=model_v,
                temperature=float(temp_v),
                k_candidates=int(k_v),
                min_retrieval_score=float(min_v),
            )
        except Exception as e:
            return (f"❌ Error: {e}", "—", "—", "")

        decision = (
            f"**Used route:** {result['used_route']}\n\n"
            f"**Router rationale (raw):**\n\n{result['router_rationale']}"
        )
        scores = (
            f"**Top relevance score:** {result['top_score'] if result['top_score'] is not None else '—'}\n\n"
            f"**Confidence gate passed?** {result['gate_passed'] if result['gate_passed'] is not None else '—'}"
        )
        answer = result["answer"]
        sources = "\n".join(result.get("sources", [])) if result.get("sources") else "None"

        return decision, scores, answer, sources

    ask_btn.click(
        on_ask,
        inputs=[api_key, model_name, temperature, k_candidates, min_score, question],
        outputs=[decision_md, scores_md, answer_md, sources_box],
        api_name="ask_adaptive",
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, quiet=True)
