"""
Hierarchical / Multi-Hop RAG — PDF Question Answering (Gradio UI)
-----------------------------------------------------------------
This file modifies the Vanilla RAG pipeline to support a two-level,
step-by-step retrieval strategy:

LEVEL 1 (Coarse / Hierarchical):
  - Retrieve the most relevant PAGES for the user's question (or
    for each decomposed sub-question) using a page-level vector index.

LEVEL 2 (Fine / Focused):
  - Within the shortlisted pages, retrieve the most relevant CHUNKS
    using a chunk-level vector index and aggregate the best evidence.

Finally:
  - Compose a grounded answer with concise citations.

Why:
  - Multi-hop / hierarchical retrieval helps for complex or compound
    questions by first narrowing the scope (pages) and then focusing
    on the best passages (chunks). This generally improves precision,
    reduces noise, and helps the LLM stay on-topic.

Usage:
  1) Provide OPENAI_API_KEY (textbox or env).
  2) Upload a PDF OR check "Use default" (Amazon Bedrock - User Guide.pdf).
  3) Click "Build Index".
  4) Ask questions. The app will:
      - Decompose into 1–3 sub-questions (when helpful),
      - Select pages for each sub-question,
      - Retrieve/re-rank chunks from those pages,
      - Answer with sources.

Notes:
  - Uses OpenAI embeddings, FAISS, PyPDFLoader, LangChain, Gradio.
  - Console output is quiet; your browser opens automatically.
"""

import os
import tempfile
from typing import List, Tuple, Dict, Any

import gradio as gr
from dotenv import load_dotenv

# LangChain core
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# -----------------------------
# Environment
# -----------------------------
load_dotenv(override=True)

# -----------------------------
# Prompt (grounded, concise)
# -----------------------------
SYSTEM_PROMPT = """You are a factual assistant. Answer ONLY using the provided context.
If the answer is not in the context, say you don't know.
Be concise and include short inline citations like [source] when helpful."""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human",
         "Question:\n{question}\n\n"
         "Context:\n{context}\n\n"
         "Answer with facts grounded in the context.")
    ]
)

# -----------------------------
# Utility: format retrieved docs into context
# -----------------------------
def format_docs(docs: List[Document]) -> str:
    """Join retrieved docs into a single context string with lightweight citations."""
    lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", f"doc_{i}")
        page = d.metadata.get("page", None)
        page_str = f" (page {page+1})" if isinstance(page, int) else ""
        lines.append(f"[{src}{page_str}] {d.page_content}")
    return "\n\n".join(lines)

def describe_sources(docs: List[Document]) -> str:
    """Pretty-print the sources used."""
    if not docs:
        return "No sources retrieved."
    lines = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        start_idx = d.metadata.get("start_index", "?")
        page_str = f"page {page+1}" if isinstance(page, int) else "page ?"
        lines.append(f"- {src} ({page_str}), start_char={start_idx}")
    return "\n".join(lines)

# -----------------------------
# Index building (Hierarchical)
# -----------------------------
def build_hierarchical_indexes(
    pdf_path: str,
    api_key: str,
    embed_model: str = "text-embedding-3-small",
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    k_pages: int = 5,
    k_chunks: int = 24,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Loads a PDF, builds TWO indexes, and returns retrievers + stats.

    1) Page index  : one embedding per PAGE (coarse-level narrowing)
    2) Chunk index : multiple embeddings per page chunk (fine-grained)

    Returns
    -------
    retrievers : dict with keys:
        - "page_retriever"  : Retriever over page-level docs
        - "chunk_retriever" : Retriever over chunk-level docs
    stats : dict with info about the build
    """
    # Key handling
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY. Provide in the textbox or environment.")

    # Load PDF (one Document per page, with 'page' metadata)
    loader = PyPDFLoader(pdf_path)
    page_docs: List[Document] = loader.load()
    for d in page_docs:
        d.metadata["source"] = os.path.basename(pdf_path)  # for pretty citations

    # --- Build page-level index (coarse) ---
    embeddings = OpenAIEmbeddings(model=embed_model)
    page_index = FAISS.from_documents(page_docs, embedding=embeddings)
    page_retriever = page_index.as_retriever(search_kwargs={"k": k_pages})

    # --- Build chunk-level index (fine) ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    chunk_docs: List[Document] = splitter.split_documents(page_docs)
    chunk_index = FAISS.from_documents(chunk_docs, embedding=embeddings)
    chunk_retriever = chunk_index.as_retriever(search_kwargs={"k": k_chunks})

    stats = {
        "pdf_name": os.path.basename(pdf_path),
        "pages": len(page_docs),
        "chunks": len(chunk_docs),
        "k_pages": k_pages,
        "k_chunks": k_chunks,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embed_model": embed_model,
    }
    retrievers = {
        "page_retriever": page_retriever,
        "chunk_retriever": chunk_retriever,
    }
    return retrievers, stats

# -----------------------------
# Multi-hop logic
# -----------------------------
DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You decompose user questions into 1–3 minimal, sequential sub-questions that a product manual can answer. "
     "Return each sub-question on a new line starting with '- '. If decomposition is unnecessary, return exactly one line."),
    ("human", "{question}")
])

def decompose_query(llm: ChatOpenAI, question: str, max_subqs: int = 3) -> List[str]:
    """
    Use an LLM to decompose a complex question into 1–3 sub-questions.
    Falls back to the original question if parsing fails.
    """
    try:
        msg = DECOMPOSE_PROMPT | llm | StrOutputParser()
        text = msg.invoke({"question": question})
        lines = [ln.strip("- ").strip() for ln in text.splitlines() if ln.strip()]
        # Keep up to max_subqs non-empty unique sub-questions
        uniq = []
        for ln in lines:
            if ln and ln not in uniq:
                uniq.append(ln)
        return uniq[:max_subqs] or [question]
    except Exception:
        return [question]

def _unique_key(d: Document) -> Tuple[Any, Any]:
    """Create a stable uniqueness key per chunk to deduplicate (page, start_index)."""
    return (d.metadata.get("page", None), d.metadata.get("start_index", None))

def step_by_step_retrieve(
    llm: ChatOpenAI,
    page_retriever,
    chunk_retriever,
    question: str,
    k_pages_keep: int = 5,
    k_final: int = 8,
) -> Tuple[List[Document], str]:
    """
    Multi-hop retrieval:
      1) Decompose the query (1–3 sub-questions).
      2) For each sub-question, retrieve top pages (coarse).
      3) From those pages, pick the best chunks (fine).
      4) Merge/deduplicate and keep top-k_final by simple relevance mixing.

    Returns:
      - final_docs: List[Document] to feed as context
      - plan_text : Human-readable plan (sub-questions + selected pages)
    """
    subqs = decompose_query(llm, question, max_subqs=3)

    plan_lines = ["### Plan (Multi-Hop)", "**Sub-questions:**"]
    for i, sq in enumerate(subqs, 1):
        plan_lines.append(f"{i}. {sq}")

    candidate_docs: List[Document] = []
    seen = set()

    for i, sq in enumerate(subqs, 1):
        # --- LEVEL 1: Pages ---
        page_hits: List[Document] = page_retriever.get_relevant_documents(sq)
        page_ids = {p.metadata.get("page") for p in page_hits if "page" in p.metadata}
        plan_lines.append(f"- Selected pages for sub-question {i}: {sorted(list(page_ids)) or '[]'}")

        # --- LEVEL 2: Chunks (filter to the pages we shortlisted) ---
        chunk_hits: List[Document] = chunk_retriever.get_relevant_documents(sq)
        filtered = [
            d for d in chunk_hits
            if (not page_ids) or (d.metadata.get("page") in page_ids)  # ✅ fixed syntax
        ]

        for d in filtered:
            key = _unique_key(d)
            if key not in seen:
                seen.add(key)
                candidate_docs.append(d)

    # If filtering was too strict, keep a few general candidates
    if not candidate_docs:
        candidate_docs = chunk_retriever.get_relevant_documents(question)

    # Light heuristic re-scoring: prioritize shorter chunks & lower page numbers slightly
    def _score(doc: Document) -> float:
        page = doc.metadata.get("page", -1)
        length_penalty = len(doc.page_content) ** 0.25  # mild penalty for very long text
        return (1.0 / max(length_penalty, 1e-6)) + (0.001 * (1000 - page if isinstance(page, int) else 0))

    candidate_docs.sort(key=_score, reverse=True)
    final_docs = candidate_docs[:k_final]

    return final_docs, "\n".join(plan_lines)

# -----------------------------
# Answering (LLM generation)
# -----------------------------
def answer_with_context(llm: ChatOpenAI, question: str, docs: List[Document]) -> str:
    """Format context and generate a grounded answer."""
    ctx = format_docs(docs)
    chain = PROMPT | llm | StrOutputParser()
    return chain.invoke({"question": question, "context": ctx}).strip()

# -----------------------------
# Bytes/File helpers
# -----------------------------
def _bytes_to_temp_pdf(file_bytes: bytes, suggested_name: str = "uploaded.pdf") -> str:
    """Write bytes to a temp .pdf and return the path."""
    tmp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(tmp_dir, suggested_name)
    with open(pdf_path, "wb") as f:
        f.write(file_bytes)
    return pdf_path

def _resolve_pdf_path(pdf_input, use_default: bool) -> str:
    """
    Accepts:
      - use_default=True: use local default file
      - pdf_input: either a filepath (str)
    Returns a filesystem path to a readable PDF.
    """
    if use_default:
        default_path = "Amazon Bedrock - User Guide.pdf"
        if not os.path.exists(default_path):
            raise FileNotFoundError("Default PDF not found in project folder.")
        return default_path

    if pdf_input is None:
        raise ValueError("Please upload a PDF or check 'Use default PDF'.")

    if isinstance(pdf_input, str):
        if not os.path.exists(pdf_input):
            raise FileNotFoundError(f"Uploaded path not found: {pdf_input}")
        return pdf_input

    # If the File component was set to give bytes instead of a path, handle here:
    if isinstance(pdf_input, (bytes, bytearray)):
        return _bytes_to_temp_pdf(pdf_input)

    raise TypeError(f"Unsupported file input type: {type(pdf_input)}")

# -----------------------------
# Gradio callbacks
# -----------------------------
def ui_build_index(api_key: str, pdf_file, use_default: bool):
    """
    Builds hierarchical indexes (page + chunk) and stores a Multi-Hop RAG state in memory.
    """
    try:
        pdf_path = _resolve_pdf_path(pdf_file, use_default)
        retrievers, stats = build_hierarchical_indexes(pdf_path, api_key=api_key)

        # Prepare LLM once and keep in state
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

        state = {
            "llm": llm,
            "page_retriever": retrievers["page_retriever"],
            "chunk_retriever": retrievers["chunk_retriever"],
            # Tunables for retrieval
            "k_pages_keep": stats["k_pages"],
            "k_final": 8,
        }

        summary = (
            f"✅ Hierarchical index ready for **{stats['pdf_name']}**\n"
            f"- Pages: {stats['pages']}\n"
            f"- Chunks: {stats['chunks']}\n"
            f"- Page Top-K: {stats['k_pages']}\n"
            f"- Chunk candidate Top-K: {stats['k_chunks']}\n"
            f"- Chunk size/overlap: {stats['chunk_size']}/{stats['chunk_overlap']}\n"
            f"- Embeddings: {stats['embed_model']}"
        )
        return state, gr.update(value=summary, visible=True), "Index built successfully."
    except Exception as e:
        return None, gr.update(value="", visible=True), f"❌ Error: {e}"

def ui_ask_question(state, question: str):
    """
    Step-by-step retrieval + grounded answer:
      - Decompose → Pages → Chunks → Answer (+ plan & sources).
    """
    if state is None:
        return "Please build the index first.", "", ""
    if not question or not question.strip():
        return "Please enter a question.", "", ""

    try:
        llm = state["llm"]
        page_retriever = state["page_retriever"]
        chunk_retriever = state["chunk_retriever"]

        docs, plan = step_by_step_retrieve(
            llm=llm,
            page_retriever=page_retriever,
            chunk_retriever=chunk_retriever,
            question=question.strip(),
            k_pages_keep=state.get("k_pages_keep", 5),
            k_final=state.get("k_final", 8),
        )
        answer = answer_with_context(llm, question.strip(), docs)
        sources = describe_sources(docs)
        return answer, plan, sources
    except Exception as e:
        return f"❌ Error while answering: {e}", "", ""

# -----------------------------
# Build Gradio App (same look & feel)
# -----------------------------
with gr.Blocks(title="Hierarchical / Multi-Hop RAG — PDF QA") as demo:
    gr.Markdown(
        """
        # 🧭 Hierarchical / Multi-Hop RAG — PDF Question Answering
        **How to use**
        1) Provide your **OpenAI API key** (or set `OPENAI_API_KEY` in env).  
        2) Upload a **PDF** *or* use the default `Amazon Bedrock - User Guide.pdf` in this folder.  
        3) Click **Build Index** → Ask questions grounded in the document.

        **What’s new vs. Vanilla RAG**
        - Step-by-step retrieval: **Pages → Chunks**  
        - Optional question decomposition (1–3 sub-questions)  
        - Cleaner context selection for complex queries
        """
    )
    with gr.Row():
        api_key = gr.Textbox(
            label="OpenAI API Key (optional if set in environment)",
            type="password",
            placeholder="sk-...",
        )
    with gr.Row():
        pdf = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
        use_default = gr.Checkbox(value=True, label="Use default: Amazon Bedrock - User Guide.pdf")

    build_btn = gr.Button("🔧 Build Index", variant="primary")

    index_summary = gr.Markdown(visible=False)
    status = gr.Markdown("")

    # Keep hierarchical state (retrievers + llm + params)
    state_obj = gr.State()

    gr.Markdown("---")
    question = gr.Textbox(
        label="Ask a question about the PDF",
        lines=2,
        placeholder="e.g., How does Amazon Bedrock support foundation models and what are the deployment options?"
    )
    ask_btn = gr.Button("💬 Ask")

    answer = gr.Markdown(label="Answer")
    plan = gr.Markdown(label="Plan (Sub-questions & Selected Pages)")
    sources = gr.Textbox(label="Sources (retrieved chunks/pages)", lines=6)

    build_btn.click(
        ui_build_index,
        inputs=[api_key, pdf, use_default],
        outputs=[state_obj, index_summary, status],
        api_name="build_index",
    )

    ask_btn.click(
        ui_ask_question,
        inputs=[state_obj, question],
        outputs=[answer, plan, sources],
        api_name="ask",
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, quiet=True)
