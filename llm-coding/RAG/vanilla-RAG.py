import os
import tempfile
from typing import List, Tuple, Dict, Any

import gradio as gr
from dotenv import load_dotenv

# LangChain pieces
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough, RunnableMap
from langchain_core.output_parsers import StrOutputParser

# Load environment variables in a file called .env

load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

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

def format_docs(docs: List[Document]) -> str:
    """Join retrieved docs into a single context string with lightweight citations."""
    lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", f"doc_{i}")
        page = d.metadata.get("page", None)
        page_str = f" (page {page+1})" if isinstance(page, int) else ""
        lines.append(f"[{src}{page_str}] {d.page_content}")
    return "\n\n".join(lines)

def build_index(
    pdf_path: str,
    api_key: str,
    embed_model: str = "text-embedding-3-small",
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    k: int = 4,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Loads a PDF, splits into chunks, creates embeddings, and returns a retriever.
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY. Provide in the textbox or environment.")

    # Load PDF (each page becomes a Document with page metadata)
    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()  # List[Document]
    # Add 'source' to each page so citations look nice
    for d in raw_docs:
        d.metadata["source"] = os.path.basename(pdf_path)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    chunks: List[Document] = splitter.split_documents(raw_docs)

    # Vector index
    embeddings = OpenAIEmbeddings(model=embed_model)
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    stats = {
        "pages": len(raw_docs),
        "chunks": len(chunks),
        "pdf_name": os.path.basename(pdf_path),
        "k": k,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embed_model": embed_model,
    }
    return retriever, stats

def make_chain(retriever, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    chain = (
        RunnableMap(
            {"question": RunnablePassthrough(), "docs": retriever}
        )
        | RunnableMap(
            {
                "question": lambda x: x["question"],
                "context": lambda x: format_docs(x["docs"]),
                "docs": lambda x: x["docs"],
            }
        )
        | RunnableMap(
            {
                "answer": PROMPT | llm | StrOutputParser(),
                "docs": lambda x: x["docs"],
            }
        )
    )
    return chain

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
# Helpers to resolve PDF path from Gradio file input
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
      - pdf_input: either a filepath (str) or bytes (if someone changes the File 'type')
    Returns a filesystem path to a readable PDF.
    """
    if use_default:
        default_path = "Amazon Bedrock - User Guide.pdf"
        if not os.path.exists(default_path):
            raise FileNotFoundError("Default PDF not found in project folder.")
        return default_path

    if pdf_input is None:
        raise ValueError("Please upload a PDF or check 'Use default PDF'.")

    # If File(type="filepath"), Gradio gives a string path
    if isinstance(pdf_input, str):
        if not os.path.exists(pdf_input):
            raise FileNotFoundError(f"Uploaded path not found: {pdf_input}")
        return pdf_input

    # If someone kept type="binary", Gradio gives bytes
    if isinstance(pdf_input, (bytes, bytearray)):
        return _bytes_to_temp_pdf(pdf_input)

    # Fallback (rare)
    raise TypeError(f"Unsupported file input type: {type(pdf_input)}")

# -----------------------------
# Gradio Callbacks
# -----------------------------
def ui_build_index(api_key: str, pdf_file, use_default: bool):
    """
    Builds an index from either the uploaded PDF or default local file.
    """
    try:
        pdf_path = _resolve_pdf_path(pdf_file, use_default)
        retriever, stats = build_index(pdf_path, api_key=api_key)
        # Build a default chain so we can reuse it across questions
        chain = make_chain(retriever)
        summary = (
            f"✅ Index ready for **{stats['pdf_name']}**\n"
            f"- Pages: {stats['pages']}\n"
            f"- Chunks: {stats['chunks']}\n"
            f"- Top-K: {stats['k']}\n"
            f"- Chunk size/overlap: {stats['chunk_size']}/{stats['chunk_overlap']}\n"
            f"- Embeddings: {stats['embed_model']}"
        )
        return chain, gr.update(value=summary, visible=True), "Index built successfully."
    except Exception as e:
        return None, gr.update(value="", visible=True), f"❌ Error: {e}"

def ui_ask_question(chain, question: str):
    if chain is None:
        return "Please build the index first.", ""
    if not question or not question.strip():
        return "Please enter a question.", ""
    try:
        result = chain.invoke(question.strip())
        answer = result["answer"]
        docs = result["docs"]
        sources = describe_sources(docs)
        return answer.strip(), sources
    except Exception as e:
        return f"❌ Error while answering: {e}", ""

# -----------------------------
# Build Gradio App
# -----------------------------
with gr.Blocks(title="Vanilla RAG — PDF QA") as demo:
    gr.Markdown(
        """
        # 🔎 Vanilla RAG — PDF Question Answering
        1) Provide your **OpenAI API key** (or set `OPENAI_API_KEY` in env).  
        2) Upload a **PDF** *or* use the default `Amazon Bedrock - User Guide.pdf` in this folder.  
        3) Click **Build Index** → Ask questions grounded in the document.
        """
    )
    with gr.Row():
        api_key = gr.Textbox(
            label="OpenAI API Key (optional if set in environment)",
            type="password",
            placeholder="sk-...",
        )
    with gr.Row():
        # IMPORTANT: use filepath (not binary) to avoid bytes/no .name errors
        pdf = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
        use_default = gr.Checkbox(value=True, label="Use default: Amazon Bedrock - User Guide.pdf")
    build_btn = gr.Button("🔧 Build Index", variant="primary")

    index_summary = gr.Markdown(visible=False)
    status = gr.Markdown("")

    state_chain = gr.State()  # holds the compiled chain

    gr.Markdown("---")
    question = gr.Textbox(label="Ask a question about the PDF", lines=2, placeholder="e.g., What is Amazon Bedrock and what does it provide?")
    ask_btn = gr.Button("💬 Ask")
    answer = gr.Markdown(label="Answer")
    sources = gr.Textbox(label="Sources (retrieved chunks/pages)", lines=6)

    build_btn.click(
        ui_build_index,
        inputs=[api_key, pdf, use_default],
        outputs=[state_chain, index_summary, status],
        api_name="build_index",
    )

    ask_btn.click(
        ui_ask_question,
        inputs=[state_chain, question],
        outputs=[answer, sources],
        api_name="ask",
    )

if __name__ == "__main__":
    demo.launch()
