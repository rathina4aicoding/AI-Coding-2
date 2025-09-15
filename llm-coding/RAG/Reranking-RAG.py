import os
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

DEFAULT_LOCAL_PDF = "Amazon Bedrock - User Guide.pdf"

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


def _bytes_to_temp_pdf(file_bytes: bytes, suggested_name: str = "uploaded.pdf") -> str:
    """Write bytes to a temp .pdf and return the path."""
    tmp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(tmp_dir, suggested_name)
    with open(pdf_path, "wb") as f:
        f.write(file_bytes)
    return pdf_path


def _resolve_pdf_path(pdf_input, use_default: bool) -> str:
    """
    Returns a filesystem path to a readable PDF.
    - use_default=True -> uses DEFAULT_LOCAL_PDF from current folder.
    - pdf_input can be a filepath (str) or bytes (if input type changed).
    """
    if use_default:
        if not os.path.exists(DEFAULT_LOCAL_PDF):
            raise FileNotFoundError(
                f"Default PDF not found: {DEFAULT_LOCAL_PDF} (place it in this folder or upload another file)."
            )
        return DEFAULT_LOCAL_PDF

    if pdf_input is None:
        raise ValueError("Please upload a PDF or check 'Use default PDF'.")

    if isinstance(pdf_input, str):
        if not os.path.exists(pdf_input):
            raise FileNotFoundError(f"Uploaded path not found: {pdf_input}")
        return pdf_input

    if isinstance(pdf_input, (bytes, bytearray)):
        return _bytes_to_temp_pdf(pdf_input)

    raise TypeError(f"Unsupported file input type: {type(pdf_input)}")


def format_docs(docs: List[Document]) -> str:
    """Formats retrieved docs into a single string with simple citations."""
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


# =========================
# Indexing
# =========================
def load_pdf_as_docs(pdf_path: str) -> List[Document]:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()  # each page is a Document with page metadata
    for d in docs:
        d.metadata["source"] = os.path.basename(pdf_path)
    return docs


def build_vector_index(
    docs: List[Document],
    embed_model: str = "text-embedding-3-small",
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    initial_k: int = 15,
) -> Tuple[Any, Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model=embed_model)
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": initial_k})

    stats = {
        "pages": len(docs),
        "chunks": len(chunks),
        "initial_k": initial_k,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embed_model": embed_model,
    }
    return retriever, stats


# =========================
# Re-Ranker
# =========================
@dataclass
class RerankerConfig:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 5


class CrossEncoderReranker:
    """Re-ranks retrieved chunks using CrossEncoder scoring of (query, chunk)."""
    def __init__(self, cfg: RerankerConfig):
        self.cfg = cfg
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(cfg.model_name, device=device)

    def rerank(self, query: str, docs: List[Document]) -> List[Tuple[Document, float]]:
        if not docs:
            return []
        pairs = [(query, d.page_content) for d in docs]
        scores = self.model.predict(pairs)  # numpy array
        ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
        return ranked[: self.cfg.top_k]


# =========================
# RAG Chains
# =========================
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


def make_chain_with_rerank(retriever, reranker: CrossEncoderReranker,
                           model_name="gpt-4o-mini", temperature=0.0):
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


# =========================
# Gradio Callbacks
# =========================
def ui_build_index(api_key: str,
                   pdf_file,
                   use_default: bool,
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
        pdf_path = _resolve_pdf_path(pdf_file, use_default)

        # Load docs & build index
        pages = load_pdf_as_docs(pdf_path)
        retriever, stats = build_vector_index(
            pages, embed_model=embed_model,
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            initial_k=pool_k
        )

        # Choose chain (with or without reranking)
        if use_rerank:
            reranker = CrossEncoderReranker(RerankerConfig(
                model_name=rerank_model, top_k=rerank_top_k
            ))
            chain = make_chain_with_rerank(retriever, reranker,
                                           model_name=model_name, temperature=temperature)
        else:
            chain = make_chain_no_rerank(retriever,
                                         model_name=model_name, temperature=temperature)

        summary = (
            f"✅ Index ready for **{os.path.basename(pdf_path)}**\n"
            f"- Pages: {stats['pages']} | Chunks: {stats['chunks']}\n"
            f"- Retriever pool k: {stats['initial_k']} | "
            f"Re-rank: {'ON' if use_rerank else 'OFF'} (top_k={rerank_top_k if use_rerank else '-'})\n"
            f"- Chunk size/overlap: {stats['chunk_size']}/{stats['chunk_overlap']}\n"
            f"- Embeddings: {stats['embed_model']} | LLM: {model_name} (T={temperature})"
        )
        return chain, gr.update(value=summary, visible=True), "Index built successfully."
    except Exception as e:
        return None, gr.update(value="", visible=True), f"❌ Error: {e}"


def ui_ask(chain, question: str):
    if chain is None:
        return "Please build the index first.", ""
    if not question or not question.strip():
        return "Please enter a question.", ""
    try:
        result = chain.invoke(question.strip())
        answer = result["answer"]
        docs = result["sources"]
        return answer.strip(), show_sources(docs)
    except Exception as e:
        return f"❌ Error while answering: {e}", ""


# =========================
# Build Gradio UI
# =========================
with gr.Blocks(title="RAG with Re-Ranking — PDF QA") as demo:
    gr.Markdown(
        """
        # 🔎 RAG with Re-Ranking — PDF Question Answering
        1) Provide your **OpenAI API key** (or set `OPENAI_API_KEY` in env).  
        2) Use the default **Amazon Bedrock - User Guide.pdf** (place it in this folder) or upload a PDF.  
        3) Choose retrieval pool size and re-ranking settings, then **Build Index**.  
        4) Ask questions and compare re-rank ON vs OFF for relevance.
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

    build_btn = gr.Button("🔧 Build Index", variant="primary")

    index_summary = gr.Markdown(visible=False)
    status = gr.Markdown("")
    state_chain = gr.State()

    gr.Markdown("---")
    question = gr.Textbox(
        label="Ask a question about the PDF",
        lines=2,
        placeholder="e.g., What does Amazon Bedrock provide to developers?"
    )
    ask_btn = gr.Button("💬 Ask")
    answer = gr.Markdown(label="Answer")
    sources = gr.Textbox(label="Sources (retrieved chunks/pages)", lines=6)

    build_btn.click(
        ui_build_index,
        inputs=[
            api_key, pdf, use_default,
            model_name, temperature, embed_model, chunk_size, chunk_overlap,
            pool_k, use_rerank, rerank_model, rerank_top_k
        ],
        outputs=[state_chain, index_summary, status],
        api_name="build_index",
    )

    ask_btn.click(
        ui_ask,
        inputs=[state_chain, question],
        outputs=[answer, sources],
        api_name="ask",
    )

if __name__ == "__main__":
    demo.launch()
