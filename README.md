# Manual Master

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-v0.2-green?logo=chainlink&logoColor=white)
![OpenAI](https://img.shields.io/badge/Provider-OpenAI-orange?logo=openai&logoColor=white)
![ChromaDB](https://img.shields.io/badge/Vector%20DB-Chroma-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A robust, CLI-based **Retrieval-Augmented Generation (RAG)** tool that turns your local documents into an interactive knowledge base. 

Point it at any file or directory, and it will ingest the content, store embeddings locally, and spawn an AI agent that answers questions solely based on your data.

## Key Features

* **Recursive Directory Scanning:** Ingests entire folders or single files.
* **Multi-Format Support:** Automatically detects and parses:
    * PDFs (`.pdf`)
    * HTML (`.html`, `.htm`)
    * XML (`.xml`)
    * Plain Text (`.txt`, code files, etc.)
* **Persistent Vector Storage:** Uses [ChromaDB](https://www.trychroma.com/) to save embeddings. You only index the data once; subsequent runs are instant.
* **Robust Path Handling:** Includes specific logic for Windows Long Paths (`\\?\`) and environment variable expansion (e.g., `%APPDATA%` or `~/docs`).
* **ReAct Agent:** Uses a LangChain Agent that "thinks" before answering, strictly citing sources from the retrieved context.
* **Highly Configurable:** Customize chunk sizes, models, and system prompts via environment variables.

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/asaeles/manual-master.git
cd manual-master
```

### 2. System Dependencies (Critical)
This tool uses `python-magic` for file type detection. You must have the underlying C-library installed.

* **MacOS (Homebrew):**
    ```bash
    brew install libmagic
    ```
* **Linux (Debian/Ubuntu):**
    ```bash
    sudo apt-get install libmagic1
    ```
* **Windows:**
    You usually need `python-magic-bin` instead of `python-magic`.
    ```bash
    pip install python-magic-bin
    ```

### 3. Python Dependencies
Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## Configuration

1.  Create a `.env` file in the project root.
2.  Add your OpenAI API Key (Required).
3.  Customize other settings (Optional).

**Example `.env`:**

```ini
OPENAI_API_KEY=sk-proj-12345...
OPENAI_MODEL=gpt-4o-mini
CHROMA_PATH=~/my_cache/chroma_db

# Optional: LangSmith Tracing
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=lsv2_pt_...
```

### Environment Variables Guide

| Variable | Default | Description |
| :--- | :--- | :--- |
| `OPENAI_API_KEY` | **Required** | Your OpenAI API key. |
| `OPENAI_MODEL` | `gpt-5-mini` | The Chat Model to use (e.g., `gpt-4o`, `gpt-3.5-turbo`). |
| `TEMPERATURE` | `0.0` | Controls randomness of the output (0.0 is deterministic, 1.0 is creative). |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model used for vectorizing text. |
| `CHROMA_PATH` | `~/.local/share/manual-master/chroma_db` | Storage location. Adheres to the **XDG Base Directory specification** for user data. |
| `SYSTEM_PROMPT_PATH`| `None` | Path to a custom text file containing the AI persona rules. |
| `CHUNK_SIZE` | `1000` | Number of characters per text chunk. |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks to maintain context. |
| `RETRIEVAL_K` | `30` | Number of document chunks to retrieve per query. |
| `LANGCHAIN_TRACING_V2` | `false` | Set to `true` to enable LangSmith tracing. |
| `LANGSMITH_API_KEY` | `None` | Your LangSmith API Key (for LangSmith tracing). |

---

## Usage

Run the script by providing a path to a file or a folder.

### Basic Usage
```bash
python ./src/main.py ~/documents/finance_reports
```

### How it works
1.  **Scanning:** The script scans the path for supported files.
2.  **Indexing:** If this is the first time running on this folder, it creates a unique hash of the path and builds a Vector Database in `CHROMA_PATH`.
3.  **Chat:** It enters an interactive loop where you can ask questions.

### Interactive Session Example
```text
Scanning 5 items in 'finance_reports'...
report_2023.pdf                          | 2.50MB     | application/pdf     
notes.txt                                | 12.00KB    | text/plain          

Creating new Vector Store for 2 documents...
Database saved locally.

System ready! Type 'exit' or 'quit' to stop.
----------------------------------------

Your Question: What was the total revenue in Q3?

--- AI Response ---
According to 'report_2023.pdf', the total revenue in Q3 was $15.4 million.
```

---

## Customizing the AI Personality

By default, the AI is strict and only answers from documents. You can change this behavior by creating a custom system prompt.

1.  Create a folder named `prompts` in the project root.
2.  Create a file `prompts/system_prompt.txt`.
3.  Add your instructions.

**Example `system_prompt.txt`:**
```text
You are a helpful and pirate-themed assistant. 
Always answer questions based on the retrieved docs, but end every sentence with 'Arrr!'.
```

The script automatically detects this file. Alternatively, point to a specific file using the `SYSTEM_PROMPT_PATH` environment variable.

---

## Project Structure

```text
.
├── src/
│   └── main.py              # The entry point script
├── prompts/
│   └── system_prompt.txt    # (Optional) Custom system instructions
├── .env                     # Environment variables (API Keys)
├── LICENSE                  # License file
├── README.md                # This file
└── requirements.txt         # Python dependencies
```

---

## Troubleshooting

* **`ImportError: failed to find libmagic`**: 
    * See the **System Dependencies** section above. Windows users ensure you installed `python-magic-bin`.
* **`RateLimitError`**: 
    * Check your OpenAI quota. You may need to switch `EMBEDDING_MODEL` to an older version or add credits.
* **Database Path Issues**: 
    * The script uses a hash of the *absolute path* of your input folder to determine which DB to load. If you move the source folder, the script will treat it as new data and rebuild the index.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
