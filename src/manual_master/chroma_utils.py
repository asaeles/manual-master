import os
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .utils_file import *
from .utils_hash import *
from .utils_path import *

CHROMA_PATH = os.getenv("CHROMA_PATH", "~/.local/share/manual-master/chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

def process_documents(source_path: Path) -> tuple[Chroma, str]:
    """
    Orchestrates the ingestion pipeline.

    1. Scans and Hashes content to identify the unique dataset.
    2. Checks if a valid ChromaDB already exists for this content hash.
    3. If not, uses the *already scanned* list to ingest files.
    4. Persists the vector store to disk.

    Args:
        source_path (Path): The input file or directory to ingest.

    Returns:
        Tuple[Chroma, str]: The loaded VectorStore object and the path to the DB.
    """

    # 1. Scan Content & Generate Hash (One-Pass)
    content_hash, files_to_process = scan_content_and_hash(source_path)

    # Construct DB path based on CONTENT hash, not directory path
    chroma_root = resolve_path(CHROMA_PATH)
    db_path = str(chroma_root / content_hash)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # 2. Check for existing DB
    if os.path.exists(db_path) and os.listdir(db_path):
        print(f"Checking for existing database at: {db_path}")
        try:
            vector_store = Chroma(persist_directory=db_path, embedding_function=embeddings)
            if vector_store._collection.count() > 0:
                print("Found valid existing database (Content Match). Skipping processing.")
                return vector_store, db_path
        except Exception as e:
            print(f"Existing DB corrupt, rebuilding. Error: {e}")

    # 3. Ingest (if DB missing)
    print(f"Database not found at: {db_path}")
    if not files_to_process:
        print("\nNo supported files found to process (checked extensions: .pdf, .txt, .html, .xml).")
        sys.exit(0)

    all_docs = []
    print(f"\nIngesting {len(files_to_process)} items based on content signature...\n")
    print(f"{'File Name':<40} | {'Size':<10} | {'Type':<20}")
    print("-" * 80)

    # Define root for relative paths if input is a directory
    root_context = source_path if source_path.is_dir() else None

    # Reuse the list from the hash scan!
    for file_path in sorted(files_to_process):  # Sort for deterministic processing order
        try:
            # Generate safe path string for I/O operations
            safe_path = get_safe_path_str(file_path)

            # Use safe_path for size check
            size_str = sizeof_fmt(os.path.getsize(safe_path))

            # Use original file_path for extension checks, safe_path for opening files
            mime_type, encoding = detect_file_info(file_path, safe_path)

            # We assume files are valid because they passed the hash scan filter,
            # but we run the MIME check again to get the *correct loader*.

            # Simple check for the display
            is_supported = any(ext in mime_type for ext in ['pdf', 'text', 'xml', 'html', 'plain'])
            if 'octet-stream' in mime_type and file_path.suffix.lower() not in ['.pdf', '.txt', '.html', '.xml']:
                is_supported = False

            if is_supported:
                print(f"{file_path.name[:38]:<40} | {size_str:<10} | {mime_type:<20}")
                file_docs = load_file_content(file_path, safe_path, mime_type, root_context, encoding)
                all_docs.extend(file_docs)
            else:
                print(f"{file_path.name[:38]:<40} | {size_str:<10} | {'Unknown/Skip':<20}")

        except Exception as e:
            print(f"Error accessing {file_path.name}: {e}")

    if not all_docs:
        print("\nNo valid documents extracted.")
        sys.exit(0)

    # 4. Build Vector Store
    print(f"\nCreating new Vector Store for {len(all_docs)} documents...")

    # Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(all_docs)
    print(f"Generated {len(chunks)} chunks.")

    # Persist
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=db_path)
    print(f"Database saved locally at: {db_path}")

    return vector_store, db_path
