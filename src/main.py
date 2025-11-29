import argparse
import hashlib
import logging
import os
import sys
import time
from pathlib import Path

# --- INSTALLATION INSTRUCTIONS ---
# If Pylance/Import errors occur, ensure you have the main 'langchain' package installed:
# pip install langchain langchain-community langchain-openai langchain-text-splitters faiss-cpu python-dotenv openai tenacity
# ---------------------------------

from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
    RetryError
)
import openai

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_SYSTEM_PROMPT = os.getenv("OPENAI_SYSTEM_PROMPT", "You are a helpful assistant.")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# --- SILENCE BACKGROUND NOISE ---
logging.getLogger("langsmith").setLevel(logging.CRITICAL)

def sizeof_fmt(num, decimal_places=2):
    for unit in ['B','KB','MB','GB','TB','PB']:
        if num < 1024.0 or unit == 'PB':
            if unit == 'B': return f"{int(num)}{unit}"
            return f"{num:.{decimal_places}f}{unit}"
        num /= 1024.0

def get_hashed_db_path(pdf_path: Path) -> str:
    """
    Generates a unique, deterministic DB path using SHA256.
    Combines absolute path + file size to ensure uniqueness.
    """
    # Resolve absolute path and get file size
    abs_path = pdf_path.resolve()
    file_size = pdf_path.stat().st_size
    
    # Create unique identifier string
    unique_id = f"{abs_path}_{file_size}".encode('utf-8')
    
    # Generate SHA256 hash
    hash_digest = hashlib.sha256(unique_id).hexdigest()
    
    # Return path: ./chroma_db/<hash_digest>
    return os.path.join("./chroma_db", hash_digest)

def build_vector_store(pdf_path):
    """
    Loads a PDF, splits it into chunks, and creates a local FAISS vector store.
    """
    # Create a unique DB path based on the PDF hash (Path + Size)
    db_path = get_hashed_db_path(pdf_path)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Check if DB exists on disk
    if os.path.exists(db_path):
        print(f"Found existing DB for {pdf_path.name} at {db_path}")
        vector_store = Chroma(persist_directory=db_path, embedding_function=embeddings)
        
        # Simple check: If collection has items, return it.
        # (For a more robust check, you might compare file hashes)
        if vector_store._collection.count() > 0:
            print("Loaded existing vector store from disk.")
            return vector_store
        else:
            print("DB directory exists but is empty. Rebuilding...")

    print(f"Loading PDF: {pdf_path.name}...")
    # 1. Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # 2. Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,        # Characters per chunk
        chunk_overlap=CHUNK_OVERLAP   # Overlap between chunks
    )
    chunks = splitter.split_documents(docs)
    print(f"Split document into {len(chunks)} chunks.")

    # 3. Create vector store from chunks
    print("Embedding chunks into Vector Database...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # Create and persist the vector store
    vector_store = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=db_path
    )
    print("Database saved locally.")

    return vector_store

# 4. Create retrieval tool
@tool(response_format="content_and_artifact")
def retrieve_docs(query: str, config: RunnableConfig):
    """Search PDF for relevant passages."""
    pdf_path = config.get("configurable", {}).get("context", {}).get("pdf_path")

    if not pdf_path:
        # Fallback to check nested structure if passed differently
        pdf_path = config.get("context", {}).get("pdf_path")

    if not pdf_path:
        return "Error: pdf_path not provided in context", []

    retrieved = build_vector_store(pdf_path).similarity_search(query, k=3)
    # Format for LLM display
    content = "\n\n".join(
        f"Page {doc.metadata.get('page', '?')}: {doc.page_content}" 
        for doc in retrieved
    )
    return content, retrieved  # content for LLM, docs as artifact

def invoke_agent(pdf_path: str, query: str):
    # 5. Create agent with retrieval tool
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[retrieve_docs],
        system_prompt="""You are a helpful assistant that answers questions about documents.
        Use the retrieve_docs tool to search the PDF when needed.
        Always cite the page number when referencing specific information."""
    )

    # 6. Use the agent
    result = agent.invoke(
        {"messages": [{
            "role": "user",
            "content": query}]
        },
        config={
            "configurable": {
                "context": {"pdf_path": pdf_path}
            }
        }
    )

    return result

def parse_args(argv):
    parser = argparse.ArgumentParser(description="PDF RAG Tool")
    parser.add_argument("pdf_path", help="Path to the PDF manual")
    parser.add_argument("query", help="Question to ask")
    if len(argv) == 0:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args(argv)

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    pdf_path = Path(args.pdf_path).expanduser()

    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    print(invoke_agent(pdf_path, args.query)["messages"][-1].content)

    # Force kill to prevent LangSmith threads from hanging the script
    print("Closing threads...")
    time.sleep(1) # Reduced to 1s
    os._exit(0)