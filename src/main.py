import argparse
import hashlib
import mimetypes
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Any

import magic
from dotenv import load_dotenv

# LangChain Imports
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Loaders
from langchain_community.document_loaders import PyPDFLoader, TextLoader, BSHTMLLoader


# --- PROJECT STRUCTURE ---
# Robustly determine project root regardless of where script is run from
try:
    # Assuming main.py is inside /src/, we go up two levels to find the root
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    # Fallback for interactive shells
    PROJECT_ROOT = Path(".").resolve()

# --- ENVIRONMENT VARIABLES ---
# Load .env from the explicitly determined project root.
# This ensures it works even if you run the script from a different directory.
load_dotenv(PROJECT_ROOT / ".env")

# --- CONFIGURATION CENTER ---
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHROMA_PATH = os.getenv("CHROMA_PATH", "~/.local/share/manual-master/chroma_db")
SYSTEM_PROMPT_PATH = os.getenv("SYSTEM_PROMPT_PATH", None)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "30"))

# Default prompt used if no file is found
DEFAULT_SYSTEM_PROMPT = """You are a specialized strict assistant for document analysis.

RULES:
1. You must answer the question SOLELY based on the context provided by the 'retrieve_docs' tool.
2. Do NOT use your own outside knowledge, training data, or assumptions.
3. If the answer is not clearly stated in the retrieved documents, you must say: "I cannot find the answer in the provided documents."
4. Do not apologize or be conversational. Just provide the answer and the source.
5. Always cite the 'Source' file name."""
# ----------------------------


def resolve_path(path_str: str) -> Path:
    """
    Resolves a path string into a concrete Path object, handling shell expansions.

    This function expands the user's home directory (~) and environment variables
    (e.g., %LOCALAPPDATA% on Windows or $VAR on Linux/Mac) to ensure the path
    is valid and absolute.

    Args:
        path_str (str): The raw path string containing potential variables or tildes.

    Returns:
        Path: A resolved, absolute pathlib.Path object.
    """
    if not path_str:
        return Path(".")
    # expandvars handles %VAR% on Windows and $VAR on Linux
    # expanduser handles ~
    return Path(os.path.expandvars(os.path.expanduser(path_str))).resolve()


def sizeof_fmt(num: float, decimal_places: int = 2) -> str:
    """
    Converts a byte count into a human-readable string (e.g., '10.5MB').

    Args:
        num (float): The size in bytes.
        decimal_places (int): Number of decimal places to display.

    Returns:
        str: Formatted string representing the size.
    """
    for unit in ['B','KB','MB','GB','TB','PB']:
        if num < 1024.0 or unit == 'PB':
            if unit == 'B': return f"{int(num)}{unit}"
            return f"{num:.{decimal_places}f}{unit}"
        num /= 1024.0
    return f"{num:.{decimal_places}f}PB"


def get_safe_path_str(path: Path) -> str:
    """
    Converts a Path object to a string safe for Windows file operations.

    This handles the Windows MAX_PATH limit (260 chars) by prepending
    the extended path prefix '\\?\' to absolute paths on Windows systems.

    Args:
        path (Path): The pathlib object to convert.

    Returns:
        str: A string path safe for OS I/O operations.
    """
    try:
        abs_path = path.resolve()
        path_str = str(abs_path)
        if os.name == 'nt' and not path_str.startswith("\\\\?\\"):
            return f"\\\\?\\{path_str}"
        return path_str
    except Exception:
        # Fallback if resolve fails
        return str(path)


def detect_file_info(path_obj: Path, safe_path: str) -> Tuple[str, str]:
    """
    Identifies file MIME type and encoding using magic headers or fallback extensions.

    It first attempts to use `python-magic` (libmagic) to read the file binary header.
    If that fails or isn't installed, it falls back to inspecting the first 1024 bytes
    manually or using the file extension.

    Args:
        path_obj (Path): The pathlib object (used for extension checking).
        safe_path (str): The string path (potentially with \\?\) for file opening.

    Returns:
        Tuple[str, str]: A tuple containing (MimeType, Encoding).
                         Defaults to ('application/octet-stream', 'unknown') if failed.
    """
    try:
        mime = magic.Magic(mime=True)
        mime_type = mime.from_file(safe_path)
        # Create a separate magic instance for encoding if needed,
        # or just rely on mime type for logic.
        return mime_type, "unknown"
    except Exception as e:
        print(f"  [Warn] Magic detection failed: {e}")

    # Fallback: Manual Header Inspection & Extension
    try:
        with open(safe_path, 'rb') as f:
            header = f.read(1024)

        # Basic Magic Number checks
        if header.startswith(b'%PDF'):
            return 'application/pdf', 'binary'
        if header.startswith(b'<?xml') or path_obj.suffix.lower() == '.xml':
            return 'text/xml', 'utf-8' # Assumption
        if b'<!DOCTYPE html' in header or b'<html' in header or path_obj.suffix.lower() in ['.html', '.htm']:
            return 'text/html', 'utf-8' # Assumption

        # Fallback to mimetypes library based on extension
        m_type, encoding = mimetypes.guess_type(path_obj)
        return m_type or 'application/octet-stream', encoding or 'unknown'

    except IOError:
        return 'error/unreadable', 'unknown'


def load_file_content(path_obj: Path, safe_path: str, mime_type: str, root_path: Optional[Path] = None) -> List[Document]:
    """
    Loads and parses content from a file based on its identified MIME type.

    Supports PDF, HTML, XML, and plain text. Adds metadata to the resulting documents,
    including the relative path if a root directory context is provided.

    Args:
        path_obj (Path): The original Path object.
        safe_path (str): The safe string path for I/O.
        mime_type (str): The detected MIME type used to select the loader.
        root_path (Optional[Path]): The root directory of the scan, used to calculate relative paths.

    Returns:
        List[Document]: A list of LangChain Document objects containing the text and metadata.
    """
    docs = []
    try:
        if 'pdf' in mime_type:
            loader = PyPDFLoader(safe_path)
            docs = loader.load()
        elif 'html' in mime_type:
            # BSHTMLLoader extracts text from HTML
            loader = BSHTMLLoader(safe_path, open_encoding='utf-8')
            docs = loader.load()
        elif 'xml' in mime_type:
            # Using BSHTMLLoader with xml features or TextLoader as fallback
            try:
                # FIX: Pass features='xml' to silence warning and use lxml parser
                loader = BSHTMLLoader(safe_path, open_encoding='utf-8', bs_kwargs={"features": "xml"})
                docs = loader.load()
            except Exception:
                # Fallback to plain text if BS fails on XML
                loader = TextLoader(safe_path, encoding='utf-8')
                docs = loader.load()
        elif 'text' in mime_type or 'plain' in mime_type:
            loader = TextLoader(safe_path, encoding='utf-8')
            docs = loader.load()

        # Determine source identifier (relative path if root provided, else filename)
        source_identifier = str(path_obj.name)
        if root_path:
            try:
                source_identifier = str(path_obj.relative_to(root_path))
            except ValueError:
                pass  # Fallback to filename if path not relative to root

        # Add metadata source
        for doc in docs:
            doc.metadata['source'] = source_identifier

        return docs
    except Exception as e:
        print(f"  [Error] Failed to load {path_obj.name}: {e}")
        return []


def scan_content_and_hash(source_root: Path) -> Tuple[str, List[Path]]:
    """
    Scans the directory for relevant files and generates a deterministic hash
    based on the file contents (header + tail + size), NOT the filename or directory path.

    This implements "Content-Based Addressing" for the database cache.
    Renaming the folder or files will NOT break the cache/hash as long as content matches.
    
    Args:
        source_root (Path): The root directory or file to scan.
        
    Returns:
        Tuple[str, List[Path]]: A tuple containing the (master_hex_hash, list_of_valid_paths).
    """
    signatures = []
    valid_files = []
    
    # Extensions to consider for the hash. We filter here for speed during the hash phase.
    # We purposefully exclude .exe, .dll, etc. to avoid hashing massive binaries unnecessarily.
    relevant_extensions = {'.pdf', '.txt', '.html', '.htm', '.xml', '.md'}
    
    # Normalize input
    items_to_scan = []
    if source_root.is_file():
        items_to_scan = [source_root]
    elif source_root.is_dir():
        for root, dirs, files in os.walk(source_root):
            for file in files:
                items_to_scan.append(Path(root) / file)
    else:
        return hashlib.sha256(b'empty').hexdigest(), []

    print(f"Scanning content signature for {len(items_to_scan)} items...")

    for path in items_to_scan:
        # Fast extension check
        if path.suffix.lower() not in relevant_extensions:
            continue

        try:
            # 1. Get Size
            size = path.stat().st_size
            
            # 2. Read Head (and Tail) safely as binary
            # Reading tail ensures we distinguish files that share a common header (like PDF templates)
            with open(path, "rb") as f:
                header = f.read(1024)
                
                # Logic to read tail if file is large enough
                tail = b''
                if size > 2048:
                    f.seek(-1024, 2) # Seek to 1024 bytes before end
                    tail = f.read(1024)
            
            # 3. Create component hash (Size + Header + Tail)
            # We encode size to ascii bytes to safely mix with binary data
            # Filename is EXCLUDED from the hash to allow renaming
            file_data = str(size).encode('ascii') + header + tail
            file_sig = hashlib.sha256(file_data).hexdigest()
            
            signatures.append(file_sig)
            valid_files.append(path)
            
        except (IOError, OSError) as e:
            # If unreadable, we skip it for the hash (and thus for ingestion)
            # OR we could add an error signature. Skipping is safer for stability.
            continue

    # 4. Sort signatures to ensure directory traversal order doesn't matter (Determinism)
    signatures.sort()
    
    # 5. Master Hash
    # We hash the concatenated sorted signatures
    if not signatures:
        return hashlib.sha256(b'empty').hexdigest(), []
        
    master_hash = hashlib.sha256("".join(signatures).encode('ascii')).hexdigest()
    
    return master_hash, valid_files


def process_documents(source_path: Path):
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
    db_path = os.path.join(CHROMA_PATH, content_hash)
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
    for file_path in files_to_process:
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
                file_docs = load_file_content(file_path, safe_path, mime_type, root_context)
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
    print("Database saved locally.")

    return vector_store, db_path


@tool(response_format="content_and_artifact")
def retrieve_docs(query: str, config: RunnableConfig):
    """
    LangChain Tool: Searches the Knowledge Base for relevant context.

    This function performs a similarity search against the ChromaDB vector store
    associated with the current session.

    Args:
        query (str): The natural language question or search term.
        config (RunnableConfig): LangChain config object containing the 'db_path'
                                 in the configurable context.

    Returns:
        Tuple[str, List[Document]]: A tuple containing the formatted context string
                                    and the raw list of retrieved artifacts.
    """
    # We retrieve the db_path passed via configuration
    db_path = config.get("configurable", {}).get("context", {}).get("db_path")

    if not db_path:
        return "Error: Database path not configured.", []

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(persist_directory=db_path, embedding_function=embeddings)

    # Use configured K value
    retrieved = vector_store.similarity_search(query, k=RETRIEVAL_K)

    # --- ADDED PRINT STATEMENT HERE ---
    sources = [doc.metadata.get('source', 'Unknown') for doc in retrieved]
    print(f"\nRetrieved {len(retrieved)} documents: {', '.join(sources)}")
    # ----------------------------------

    content = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}"
        for doc in retrieved
    )
    return content, retrieved


def invoke_agent(db_path: str, query: str) -> Any:
    """
    Initializes and invokes the ReAct agent to answer a question.

    Configures the ChatOpenAI model independently to be able to modify
    the temperature and configures the system prompt,
    then executes the agent graph using the provided database context.

    Args:
        db_path (str): The path to the ChromaDB folder to use for retrieval.
        query (str): The user's question.

    Returns:
        Any: The result object from the agent execution (contains messages).
    """

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=TEMPERATURE)
    
    # Logic to select system prompt: 
    # 1. Env Var (SYSTEM_PROMPT_PATH)
    # 2. File in default prompts dir (PROJECT_ROOT/prompts/system_prompt.txt)
    # 3. Hardcoded Default (DEFAULT_SYSTEM_PROMPT)
    
    system_prompt = DEFAULT_SYSTEM_PROMPT
    
    # Determine which path to use (Env var override OR default convention)
    target_prompt_path = None
    
    if SYSTEM_PROMPT_PATH:
        target_prompt_path = resolve_path(SYSTEM_PROMPT_PATH)
    else:
        # Check for convention: <PROJECT_ROOT>/prompts/system_prompt.txt
        target_prompt_path = PROJECT_ROOT / "prompts" / "system_prompt.txt"

    # If a path was determined, try to load it
    if target_prompt_path:
        if target_prompt_path.exists() and target_prompt_path.is_file():
            try:
                system_prompt = target_prompt_path.read_text(encoding='utf-8')
                # Optional: Print confirmation only on startup/first use if you prefer
                # print(f"Loaded system prompt from: {target_prompt_path}") 
            except Exception as e:
                print(f"Warning: Failed to read system prompt file: {e}. Using default.")
        elif SYSTEM_PROMPT_PATH:
             # Only warn if the user explicitly set a path that doesn't exist
             print(f"Warning: Configured system prompt file not found at: {target_prompt_path}. Using default.")

    agent = create_agent(
        model=llm,
        tools=[retrieve_docs],
        system_prompt=system_prompt
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={
            "configurable": {
                "context": {"db_path": db_path}
            }
        }
    )

    return result


def parse_args(argv):
    """
    Parses command-line arguments.

    Args:
        argv (list): List of command line arguments (usually sys.argv[1:]).

    Returns:
        argparse.Namespace: The parsed arguments object containing 'path'.
    """
    parser = argparse.ArgumentParser(description="Multi-Format RAG Tool")
    parser.add_argument("path", help="Path to a file or directory")
    # Query argument removed here

    if len(argv) == 0:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args(argv)


if __name__ == "__main__":
    CHROMA_PATH = resolve_path(CHROMA_PATH)

    # Ensure the CHROMA_PATH directory exists (create recursively if needed)
    try:
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create CHROMA_PATH '{CHROMA_PATH}': {e}")

    args = parse_args(sys.argv[1:])
    # Apply path resolution (handling ~, %VAR%) to the user input argument
    input_path = resolve_path(args.path)

    if not input_path.exists():
        print(f"Error: Path not found: {input_path}")
        sys.exit(1)

    # Process files and get DB (runs once)
    _, db_path = process_documents(input_path)

    print("\nSystem ready! Type 'exit' or 'quit' to stop.")
    print("-" * 40)

    # Chat Loop
    while True:
        try:
            user_query = input("\nYour Question: ").strip()

            if user_query.lower() in ['exit', 'quit', 'q']:
                break

            if not user_query:
                continue

            print("\n--- AI Response ---")
            response = invoke_agent(db_path, user_query)
            print(response["messages"][-1].content)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")