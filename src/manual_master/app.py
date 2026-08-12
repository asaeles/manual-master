import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import ToolRuntime
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .chroma_utils import *
from .utils_path import *

# --- PROJECT STRUCTURE ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
except NameError:
    PROJECT_ROOT = Path(".").resolve()

# --- ENVIRONMENT VARIABLES ---
load_dotenv(PROJECT_ROOT / ".env")

# --- CONFIGURATION CENTER ---
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
CHROMA_PATH = os.getenv("CHROMA_PATH", "~/.local/share/manual-master/chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
SYSTEM_PROMPT_PATH = os.getenv("SYSTEM_PROMPT_PATH", None)
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "30"))

DEFAULT_SYSTEM_PROMPT = """You are a specialized strict assistant for document analysis.

RULES:
1. You must answer the question SOLELY based on the context provided by the 'retrieve_docs' tool.
2. Do NOT use your own outside knowledge, training data, or assumptions.
3. If the answer is not clearly stated in the retrieved documents, you must say: "I cannot find the answer in the provided documents."
4. Do not apologize or be conversational. Just provide the answer and the source.
5. Always cite the 'Source' file name."""
# ----------------------------


@dataclass
class RAGContext:
    """
    Doc-Block: Runtime context schema injected into the agent and its tools.

    Attributes:
        db_path (str): Filesystem path to the ChromaDB persistence directory
            used for the current session's retrieval calls.
    """
    db_path: str


@tool(response_format="content_and_artifact")
def retrieve_docs(query: str, runtime: ToolRuntime[RAGContext]) -> tuple[str, list[Document]]:
    """
    Doc-Block: LangChain tool that performs a similarity search against the
    session's ChromaDB vector store.

    Args:
        query (str): Natural language question or search term.
        runtime (ToolRuntime[RAGContext]): Injected runtime object carrying
            the RAGContext, which supplies db_path. Excluded from the tool's
            schema, so the model never sees or supplies this argument.

    Returns:
        tuple[str, list[Document]]: Formatted context string paired with the
            raw retrieved Document artifacts.

    Raises:
        None: Retrieval failures are caught internally and surfaced as an
            error string in the returned content.
    """
    db_path = runtime.context.db_path if runtime and runtime.context else None

    if not db_path or not os.path.exists(db_path):
        return f"Error: Database path not configured at {db_path}", []

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    try:
        vector_store = Chroma(persist_directory=db_path, embedding_function=embeddings)
        retrieved_docs = vector_store.similarity_search(query, k=RETRIEVAL_K)
        sources = [doc.metadata.get('source', 'Unknown') for doc in retrieved_docs]
        print(f"\nRetrieved {len(retrieved_docs)} documents: {', '.join(sources[:3])}...")

        content = "\n\n".join(
            f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}"
            for doc in retrieved_docs
        )

        return content, retrieved_docs

    except Exception as e:
        print(f"[Error] Retrieval failed: {e}")
        return f"Retrieval error: {str(e)}", []


def invoke_agent(db_path: str, messages: list) -> dict[str, Any]:
    """
    Doc-Block: Initializes and invokes the agent with the full conversation
    history to answer the latest question.

    Args:
        db_path (str): Path to the ChromaDB folder to use for retrieval,
            passed to the agent via RAGContext.
        messages (list): The accumulated conversation history, including
            the latest user message, as LangChain message objects.

    Returns:
        dict[str, Any]: The agent's execution result, containing 'messages'.

    Raises:
        Exception: Re-raised if agent construction or invocation fails.
    """
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=TEMPERATURE)

    system_prompt = DEFAULT_SYSTEM_PROMPT
    target_prompt_path = (
        resolve_path(SYSTEM_PROMPT_PATH)
        if SYSTEM_PROMPT_PATH
        else PROJECT_ROOT / "prompts" / "system_prompt.txt"
    )
    if target_prompt_path and target_prompt_path.exists():
        try:
            system_prompt = target_prompt_path.read_text(encoding='utf-8')
        except Exception:
            print("Warning: Failed to read system prompt file. Using default.")

    try:
        agent = create_agent(
            model=llm,
            tools=[retrieve_docs],
            system_prompt=system_prompt if system_prompt else None,
            context_schema=RAGContext,
        )

        result = agent.invoke(
            {"messages": messages},
            context=RAGContext(db_path=db_path),
        )

        return result

    except Exception as e:
        print(f"[Error] Agent invocation failed: {e}")
        raise

def parse_args(argv: list[str]) -> argparse.Namespace:
    """
    Doc-Block: Parses command-line arguments.

    Args:
        argv (list[str]): List of command line arguments (usually sys.argv[1:]).

    Returns:
        argparse.Namespace: The parsed arguments object containing 'path'.

    Raises:
        SystemExit: Raised via argparse if no arguments are provided or
            parsing fails.
    """

    parser = argparse.ArgumentParser(prog="manual-master", description="Multi-Format RAG Tool")
    parser.add_argument("path", help="Path to a file or directory")

    if len(argv) == 0:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args(argv)


def main() -> None:
    """
    Doc-Block: Entry point that builds the vector store and runs the
    interactive chat loop.

    Args:
        None

    Returns:
        None

    Raises:
        SystemExit: Raised if the input path is invalid or document
            processing fails.
    """
    global CHROMA_PATH

    try:
        CHROMA_PATH = resolve_path(CHROMA_PATH)
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create CHROMA_PATH '{CHROMA_PATH}': {e}")

    args = parse_args(sys.argv[1:])
    input_path = resolve_path(args.path)

    if not input_path.exists():
        print(f"Error: Path not found: {input_path}")
        sys.exit(1)

    try:
        _, db_path = process_documents(input_path)

        if not os.path.exists(db_path):
            print("\nNo database was created. Exiting.")
            sys.exit(0)

    except Exception as e:
        print(f"\nError processing documents:\n  {e}\n")
        sys.exit(1)

    print("\n" + "="*60)
    print("System Ready! Type 'exit' or Ctrl+C to stop.")
    print("="*60 + "\n")

    conversation_history = []

    try:
        while True:
            user_query = input("\nYour Question: ").strip()

            if not user_query:
                continue

            if any(user_query.lower().startswith(cmd)
                   for cmd in ['exit', 'quit']):
                print("Goodbye!")
                break

            try:
                print("\n--- AI Response ---\n")

                conversation_history.append(HumanMessage(content=user_query))
                response = invoke_agent(db_path, conversation_history)

                if isinstance(response, dict):
                    messages = response.get('messages', [])
                    if messages:
                        conversation_history = messages  # includes tool calls + AI reply
                        last_message = messages[-1]
                        print(last_message.content)

            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Exiting...")
                break

    except Exception as e:
        print(f"\nAn unexpected error occurred:\n{e}")

if __name__ == "__main__":
    main()