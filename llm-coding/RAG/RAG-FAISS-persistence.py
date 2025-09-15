## FAISS index persistence → caches the vector index to disk (by PDF content hash), so you don’t have to rebuild every run.
## Side-by-side comparison → toggle Vanilla RAG vs RAG + Re-Ranking, and a “Compare” button to show both answers at once.

## How persistence works

## We compute a cache key = SHA-256 hash of the PDF contents + the embedding model + chunking params.
## The FAISS index is saved under .faiss_cache/<cache_key>/.
## Next time you build with the same PDF + params, it loads from that folder (cache HIT) instead of rebuilding (cache MISS).

## Key APIs:
##   Save: vectorstore.save_local(index_dir)
##   Load: FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)

## Note: allow_dangerous_deserialization=True is required by newer LangChain versions when loading persisted indexes.
# ----------------------------------------------------------------------------------------------------------------------

## How the comparison works

## The app builds two chains:
##    vanilla: direct retriever → LLM.
##    rerank: retriever → Cross-Encoder re-ranker → LLM.

## Choose “Compare Both” and ask a question — you’ll get answers for both side-by-side plus each one’s sources.
## Great for teaching recall vs. precision and why re-ranking often produces cleaner answers.


import os
import io
import hashlib
import tempfile
from dataclasses import dataclass
from typing import List, Tuple, Any, Dict

import gradio as gr
from dotenv import load_dotenv

# LangChain
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableMap
from langchain_core.output_parsers import StrOutputParser

# Re-ranking
from sentence_transformers import CrossEncoder
import torch

# Load environment variables in a file called .en
load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

DEFAULT_LOCAL_PDF = "Amazon SageMaker AI-Developer Guide.pdf"
CACHE_ROOT = ".faiss_cache"           # all cached indices stored here
os.makedirs(CACHE_ROOT, exist_ok=True)

SYSTEM_PROMPT = """You are a factual assistant. Answer ONLY using the provided context.
If the answer is not present, say you don't know. Keep answers concise.
Include short inline citations like [source] when helpful."""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human",
         "Question:\n{question}\n\n"
         "Context:\n{context}\n\n"
         "Answer using ONLY the context.")
    ]
)

# ----------------------------
# Helpers
# ----------------------------
def sha256_of_file(path: str) -> str:
    """Content hash for stable cache key."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def bytes_to_temp_pdf(file_bytes: bytes, suggested_name: str = "uploaded.pdf") -> str:
    """Write bytes to a temp .pdf and return the path."""
    tmp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(tmp_dir, suggested_name)
    with open(pdf_path, "wb") as f:
        f.write(file_bytes)
    return pdf_path

def resolve_pdf_path(pdf_input, use_default: bool) -> str:
    """Return a filesystem path to a readable PDF."""
    if use_default:
        if not os.path.exists(DEFAULT_LOCAL_PDF):
            raise FileNotFoundError(
                f"Default PDF not found: {DEFAULT_LOCAL_PDF}. Place it in this folder or upload another file."
            )
        return DEFAULT_LOCAL_PDF

    if pdf_input is None:
        raise ValueError("Please upload a PDF or check 'Use default PDF'.")

    if isinstance(pdf_input, str):  # File(type="filepath")
        if not os.path.exists(pdf_input):
            raise FileNotFoundError(f"Uploaded path not found: {pdf_input}")
        return pdf_input

    if isinstance(pdf_input, (bytes, bytearray)):  # if someone uses type="binary"
        return bytes_to_temp_pdf(pdf_input)

    raise TypeError(f"Unsupported file input type: {type(pdf_input)}")

def format_docs(docs: List[Document]) -> str:
    lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", f"doc_{i}")
        page = d.metadata.get("page", None)
        page_str = f" (page {page+1})" if isinstance(page, int) else ""
        lines.append(f"[{src}{page_str}] {d.page_content}")
    return "\n\n".join(lines)

def show_sources(docs: List[Document]) -> str:
    if not docs:
        return "No sources."
    lines = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        start_idx = d.metadata.get("start_index", "?")
        page_str = f"page {page+1}" if isinstance(page, int) else "page ?"
        lines.append(f"- {src} ({page_str}), start_char={start_idx}")
    return "\n".join(lines)

def load_pdf_as_docs(pdf_path: str) -> List[Document]:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()  # one Document per page
    for d in docs:
        d.metadata["source"] = os.path.basename(pdf_path)
    return docs

# ----------------------------
# Build / Load FAISS with persistence
# ----------------------------
def build_or_load_index(
    pdf_path: str,
    embed_model: str,
    chunk_size: int,
    chunk_overlap: int,
    initial_k: int,
    use_cache: bool = True
) -> Tuple[Any, Dict[str, Any]]:
    """
    If use_cache=True, attempts to load a persisted FAISS index for this PDF+params;
    otherwise, builds from scratch and saves it.
    """
    # Cache key depends on PDF content + key params + embedding model
    key = f"{sha256_of_file(pdf_path)}__{embed_model}__{chunk_size}__{chunk_overlap}"
    index_dir = os.path.join(CACHE_ROOT, key)

    embeddings = OpenAIEmbeddings(model=embed_model)

    if use_cache and os.path.isdir(index_dir):
        # Load persisted FAISS (note: allow_dangerous_deserialization=True is required by LC>=0.2)
        vectorstore = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_kwargs={"k": initial_k})

        # provide rough stats (we don't store pages/chunks; recompute cheaply if needed)
        stats = {
            "pdf_name": os.path.basename(pdf_path),
            "pages": "cached",
            "chunks": "cached",
            "initial_k": initial_k,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embed_model": embed_model,
            "cache_dir": index_dir,
            "cache_hit": True,
        }
        return retriever, stats

    # Build from scratch
    pages = load_pdf_as_docs(pdf_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    chunks = splitter.split_documents(pages)

    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

    # Persist to disk
    os.makedirs(index_dir, exist_ok=True)
    vectorstore.save_local(index_dir)

    retriever = vectorstore.as_retriever(search_kwargs={"k": initial_k})
    stats = {
        "pdf_name": os.path.basename(pdf_path),
        "pages": len(pages),
        "chunks": len(chunks),
        "initial_k": initial_k,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embed_model": embed_model,
        "cache_dir": index_dir,
        "cache_hit": False,
    }
    return retriever, stats

# ----------------------------
# Reranker
# ----------------------------
@dataclass
class RerankerConfig:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 5

class CrossEncoderReranker:
    def __init__(self, cfg: RerankerConfig):
        self.cfg = cfg
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(cfg.model_name, device=device)

    def rerank(self, query: str, docs: List[Document]):
        if not docs:
            return []
        pairs = [(query, d.page_content) for d in docs]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
        return ranked[: self.cfg.top_k]

# ----------------------------
# Chains
# ----------------------------
def make_chain_no_rerank(retriever, model_name="gpt-4o-mini", temperature=0.0):
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    chain = (
        RunnableMap({"question": RunnablePassthrough(), "docs": retriever})
        | RunnableMap({
            "question": lambda x: x["question"],
            "context": lambda x: format_docs(x["docs"]),
            "sources": lambda x: x["docs"],
        })
        | RunnableMap({
            "answer": ChatPromptTemplate.from_messages(
                [("system", SYSTEM_PROMPT),
                 ("human", "Question:\n{question}\n\nContext:\n{context}\n\nAnswer using ONLY the context.")]
            ) | llm | StrOutputParser(),
            "sources": lambda x: x["sources"]
        })
    )
    return chain

def make_chain_with_rerank(retriever, reranker: CrossEncoderReranker, model_name="gpt-4o-mini", temperature=0.0):
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    chain = (
        RunnableMap({"question": RunnablePassthrough(), "candidates": retriever})
        | RunnableMap({
            "question": lambda x: x["question"],
            "reranked": lambda x: [d for d, s in reranker.rerank(x["question"], x["candidates"])],
        })
        | RunnableMap({
            "question": lambda x: x["question"],
            "context": lambda x: format_docs(x["reranked"]),
            "sources": lambda x: x["reranked"],
        })
        | RunnableMap({
            "answer": PROMPT | llm | StrOutputParser(),
            "sources": lambda x: x["sources"]
        })
    )
    return chain

# ----------------------------
# Gradio Callbacks
# ----------------------------
def ui_build_index(api_key: str,
                   pdf_file,
                   use_default: bool,
                   reuse_cache: bool,
                   model_name: str,
                   temperature: float,
                   embed_model: str,
                   chunk_size: int,
                   chunk_overlap: int,
                   pool_k: int,
                   use_rerank: bool,
                   rerank_model: str,
                   rerank_top_k: int):

    try:
        # API key
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Missing OPENAI_API_KEY. Provide it in the textbox or environment.")

        # Resolve PDF path
        pdf_path = resolve_pdf_path(pdf_file, use_default)

        # Build OR load cached index
        retriever, stats = build_or_load_index(
            pdf_path=pdf_path,
            embed_model=embed_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            initial_k=pool_k,
            use_cache=reuse_cache
        )

        # Choose chain
        if use_rerank:
            reranker = CrossEncoderReranker(RerankerConfig(
                model_name=rerank_model, top_k=rerank_top_k
            ))
            chain_rerank = make_chain_with_rerank(retriever, reranker,
                                                  model_name=model_name, temperature=temperature)
            chain_no = make_chain_no_rerank(retriever, model_name=model_name, temperature=temperature)
        else:
            chain_no = make_chain_no_rerank(retriever, model_name=model_name, temperature=temperature)
            chain_rerank = None

        summary = (
            f"✅ Index ready for **{stats['pdf_name']}**\n"
            f"- Cache: {'HIT' if stats['cache_hit'] else 'MISS'} at `{stats['cache_dir']}`\n"
            f"- Pages: {stats['pages']} | Chunks: {stats['chunks']}\n"
            f"- Retriever pool k: {stats['initial_k']}\n"
            f"- Chunk size/overlap: {stats['chunk_size']}/{stats['chunk_overlap']}\n"
            f"- Embeddings: {stats['embed_model']} | LLM: {model_name} (T={temperature})\n"
            f"- Re-rank enabled: {use_rerank} (model={rerank_model if use_rerank else '-'}, top_k={rerank_top_k if use_rerank else '-'})"
        )

        # Keep both chains in state for compare mode
        state = {"vanilla": chain_no, "rerank": chain_rerank}
        return state, gr.update(value=summary, visible=True), "Index built successfully."
    except Exception as e:
        return None, gr.update(value="", visible=True), f"❌ Error: {e}"

def ui_ask(state, question: str, mode: str):
    if state is None:
        return "Please build the index first.", "", ""
    if not question or not question.strip():
        return "Please enter a question.", "", ""

    try:
        if mode == "Vanilla RAG":
            result = state["vanilla"].invoke(question.strip())
            ans = result["answer"].strip()
            src = show_sources(result["sources"])
            return ans, src, ""
        elif mode == "RAG + Re-Ranking":
            if state["rerank"] is None:
                return "Re-ranking chain not built (turn it on in settings and rebuild).", "", ""
            result = state["rerank"].invoke(question.strip())
            ans = result["answer"].strip()
            src = show_sources(result["sources"])
            return ans, src, ""
        else:  # Compare both
            result_no = state["vanilla"].invoke(question.strip())
            ans_no = result_no["answer"].strip()
            src_no = show_sources(result_no["sources"])

            if state["rerank"] is None:
                return ans_no, src_no, "Re-ranking chain not built (turn it on in settings and rebuild)."

            result_rr = state["rerank"].invoke(question.strip())
            ans_rr = result_rr["answer"].strip()
            src_rr = show_sources(result_rr["sources"])

            # Return combined output: left = vanilla, right = rerank
            combined = (
                "### Vanilla RAG\n"
                f"{ans_no}\n\n"
                "---\n\n"
                "### RAG + Re-Ranking\n"
                f"{ans_rr}"
            )
            combined_src = (
                "### Vanilla Sources\n"
                f"{src_no}\n\n"
                "---\n\n"
                "### Re-Ranked Sources\n"
                f"{src_rr}"
            )
            return combined, combined_src, ""
    except Exception as e:
        return f"❌ Error: {e}", "", ""

# ----------------------------
# Gradio UI
# ----------------------------
with gr.Blocks(title="RAG — Persisted FAISS & Compare Re-Ranking") as demo:
    gr.Markdown(
        """
        # 🔎 RAG — Persisted FAISS & Compare Re-Ranking
        1) Provide your **OpenAI API key** (or set `OPENAI_API_KEY` in env).  
        2) Use the default **Amazon Bedrock - User Guide.pdf** (place it in this folder) or upload a PDF.  
        3) Enable **Reuse cached FAISS** to avoid rebuilding each run.  
        4) Turn **Re-Ranking** on/off and compare outputs side-by-side.
        """
    )

    with gr.Row():
        api_key = gr.Textbox(
            label="OpenAI API Key (optional if set in environment)",
            type="password",
            placeholder="sk-...",
        )

    with gr.Accordion("Data source", open=True):
        with gr.Row():
            pdf = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
            use_default = gr.Checkbox(value=True, label=f"Use default: {DEFAULT_LOCAL_PDF}")
            reuse_cache = gr.Checkbox(value=True, label="Reuse cached FAISS (if available)")

    with gr.Accordion("RAG settings", open=True):
        with gr.Row():
            model_name = gr.Dropdown(choices=["gpt-4o-mini"], value="gpt-4o-mini", label="LLM")
            temperature = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Temperature")
        with gr.Row():
            embed_model = gr.Dropdown(
                choices=["text-embedding-3-small", "text-embedding-3-large"],
                value="text-embedding-3-small",
                label="Embeddings model"
            )
            chunk_size = gr.Slider(200, 1200, value=500, step=50, label="Chunk size")
            chunk_overlap = gr.Slider(0, 400, value=80, step=10, label="Chunk overlap")
            pool_k = gr.Slider(5, 50, value=15, step=1, label="Retriever pool size (top-N)")

    with gr.Accordion("Re-ranking settings", open=True):
        with gr.Row():
            use_rerank = gr.Checkbox(value=True, label="Enable Cross-Encoder Re-Ranking")
            rerank_model = gr.Dropdown(
                choices=[
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    "cross-encoder/ms-marco-MiniLM-L-12-v2",
                    "cross-encoder/ms-marco-TinyBERT-L-2-v2"
                ],
                value="cross-encoder/ms-marco-MiniLM-L-6-v2",
                label="Cross-Encoder model"
            )
            rerank_top_k = gr.Slider(1, 15, value=5, step=1, label="Re-rank top-K")

    build_btn = gr.Button("🔧 Build / Load Index", variant="primary")

    index_summary = gr.Markdown(visible=False)
    status = gr.Markdown("")
    state_chains = gr.State()  # holds {'vanilla': chain, 'rerank': chain}

    gr.Markdown("---")
    mode = gr.Radio(choices=["Vanilla RAG", "RAG + Re-Ranking", "Compare Both"], value="Compare Both", label="Mode")
    question = gr.Textbox(
        label="Ask a question about the PDF",
        lines=2,
        placeholder="e.g., What does Amazon Bedrock provide to developers?"
    )
    ask_btn = gr.Button("💬 Ask")
    answer = gr.Markdown(label="Answer")
    sources = gr.Textbox(label="Sources (retrieved chunks/pages)", lines=8)
    right_panel = gr.Markdown(visible=False)  # reserved if you want an extra panel

    build_btn.click(
        ui_build_index,
        inputs=[
            api_key, pdf, use_default, reuse_cache,
            model_name, temperature, embed_model, chunk_size, chunk_overlap,
            pool_k, use_rerank, rerank_model, rerank_top_k
        ],
        outputs=[state_chains, index_summary, status],
        api_name="build_index",
    )

    ask_btn.click(
        ui_ask,
        inputs=[state_chains, question, mode],
        outputs=[answer, sources, status],
        api_name="ask",
    )

if __name__ == "__main__":
    demo.launch()
