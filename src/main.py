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


def get_hashed_db_path(source_path: Path) -> str:
    """
    Generates a deterministic filesystem path for the vector database based on the source path.

    This ensures that the same source directory always maps to the same ChromaDB collection folder.
    It handles path normalization (stripping trailing slashes, lowercasing on Windows) to prevent
    duplicate caches for the same location.

    Args:
        source_path (Path): The input file or directory being processed.

    Returns:
        str: The absolute path to the specific ChromaDB folder for this source.
    """
    # Resolve absolute path (pathlib handles normalizations like ../ and ./)
    abs_path_obj = source_path.resolve()

    # Convert to string
    path_str = str(abs_path_obj)

    # NORMALIZATION:
    # 1. Strip trailing separator (redundant for pathlib but safe for string manipulations)
    path_str = path_str.rstrip(os.sep)

    # 2. On Windows, normalize case to prevent "MyFolder" vs "myfolder" hash mismatch
    if os.name == 'nt':
        path_str = path_str.lower()

    path_bytes = path_str.encode('utf-8')
    hash_digest = hashlib.sha256(path_bytes).hexdigest()

    # Join with the resolved global CHROMA_PATH
    return os.path.join(CHROMA_PATH, hash_digest)


def process_documents(source_path: Path):
    """
    Orchestrates the ingestion pipeline.

    1. Checks if a valid ChromaDB already exists for this path.
    2. If not, scans the directory for supported files.
    3. Loads content, splits text into chunks, and generates embeddings.
    4. Persists the vector store to disk.

    Args:
        source_path (Path): The input file or directory to ingest.

    Returns:
        Tuple[Chroma, str]: The loaded VectorStore object and the path to the DB.
    """
    # 1. Check for existing DB FIRST before scanning files
    db_path = get_hashed_db_path(source_path)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    if os.path.exists(db_path) and os.listdir(db_path):
        print(f"Checking for existing database at: {db_path}")
        try:
            vector_store = Chroma(persist_directory=db_path, embedding_function=embeddings)
            if vector_store._collection.count() > 0:
                print("Found valid existing database. Skipping file processing.")
                return vector_store, db_path
        except Exception as e:
            print(f"Existing DB corrupt or unreadable, rebuilding. Error: {e}")

    # 2. Identify Files (Only if DB missing)
    files_to_process = []

    if source_path.is_file():
        files_to_process.append(source_path)
    elif source_path.is_dir():
        for root, dirs, files in os.walk(source_path):
            for file in files:
                files_to_process.append(Path(root) / file)
    else:
        raise ValueError("Invalid path provided.")

    all_docs = []
    print(f"Scanning {len(files_to_process)} items in '{source_path.name}'...\n")
    print(f"{'File Name':<40} | {'Size':<10} | {'Type':<20}")
    print("-" * 80)

    # 3. Process Files
    valid_extensions = ['.pdf', '.txt', '.html', '.xml']

    # Define root for relative paths if input is a directory
    root_context = source_path if source_path.is_dir() else None

    for file_path in files_to_process:
        try:
            # Generate safe path string for I/O operations
            safe_path = get_safe_path_str(file_path)

            # Use safe_path for size check
            size_str = sizeof_fmt(os.path.getsize(safe_path))

            # Use original file_path for extension checks, safe_path for opening files
            mime_type, encoding = detect_file_info(file_path, safe_path)

            # Check if supported
            is_supported = False
            if 'pdf' in mime_type: is_supported = True
            elif 'text' in mime_type: is_supported = True
            elif 'xml' in mime_type: is_supported = True
            elif 'html' in mime_type: is_supported = True

            # Double check against explicit unsupported binary types if magic failed
            if 'octet-stream' in mime_type and file_path.suffix.lower() not in valid_extensions:
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
        print("\nNo valid documents found to process.")
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