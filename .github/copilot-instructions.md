# Copilot Instructions for manual-master

## Project Overview

**Purpose:** CLI-based **Retrieval-Augmented Generation (RAG)** tool that ingests local documents and provides an interactive knowledge base with strict citation sourcing.

**Entry Point:** `src/main.py` (574 lines)

**Core Workflow:** Scan → Content-Based Fingerprint → Index → Interactive Query Loop

---

## Critical Architecture Patterns

### 1. Content-Based Fingerprinting (Not Path-Based)

The project uses **content hashing** instead of file paths to cache databases. This enables a unique feature: **you can rename or move source folders without breaking the cache**.

**How it works** (`scan_content_and_hash()` in `src/main.py`):
```python
# Hash = Header (1KB) + Tail (1KB) + File Size
# Filename and directory path are EXCLUDED from hash
file_data = str(size).encode('ascii') + header + tail
file_sig = hashlib.sha256(file_data).hexdigest()
```

**Benefit:** Database IDs persist even if source directories are renamed/moved.

**Relevant extensions:** `.pdf`, `.txt`, `.html`, `.htm`, `.xml`, `.md`

---

### 2. Path Handling (Windows-Aware)

The project includes robust cross-platform path utilities:

**`resolve_path(path_str: str) -> Path`**
- Expands `~` (home directory)
- Expands environment variables: `%VAR%` on Windows, `$VAR` on Linux/macOS
- Returns absolute `pathlib.Path`

**`get_safe_path_str(path: Path) -> str`**
- Handles Windows MAX_PATH limit (260 chars)
- Prepends `\\?\` prefix for paths > 260 chars on Windows
- Always use this for file I/O operations

**When to use:**
- Input path from user or environment → `resolve_path()`
- Before any file I/O operation → `get_safe_path_str()`

---

### 3. Multi-Format File Support with MIME Detection

**Detection Priority:**
1. Try `python-magic` (libmagic) for binary header inspection
2. Fallback: Manual magic number checks (PDF, XML, HTML)
3. Last resort: Extension-based guessing with `mimetypes` library

**Supported Formats:**
| Format | MIME Type | Loader |
|--------|-----------|--------|
| PDF | `application/pdf` | `PyPDFLoader` |
| HTML | `text/html` | `BSHTMLLoader` |
| XML | `text/xml` | `BSHTMLLoader` (with XML features) |
| Plain Text | `text/plain` | `TextLoader` |

**Important Notes:**
- **Windows:** Requires `python-magic-bin` (NOT `python-magic`)
- **Linux/macOS:** Requires `libmagic1` system package
- Fallback detection works without libmagic for basic formats

---

### 4. Configuration Center (Environment Variables)

All configuration is **centralized in `.env`** and read at startup (lines 29-43):

```python
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHROMA_PATH = os.getenv("CHROMA_PATH", "~/.local/share/manual-master/chroma_db")
SYSTEM_PROMPT_PATH = os.getenv("SYSTEM_PROMPT_PATH", None)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "30"))
```

**Required:** `OPENAI_API_KEY` in `.env`

**Optional but Recommended:**
- `SYSTEM_PROMPT_PATH`: Point to `prompts/system_prompt.txt` or custom file
- `LANGCHAIN_TRACING_V2=true` + `LANGSMITH_API_KEY`: Enable LangSmith observability

---

### 5. LangChain ReAct Agent Pattern

The AI is a **ReAct agent** with a single tool: `retrieve_docs`.

**Key Behavior:**
- Agent thinks about the question before answering
- Uses `retrieve_docs(query)` to fetch `RETRIEVAL_K` document chunks
- Strictly cites sources from retrieved context
- Falls back to "I cannot find this in the documents" if no match

**Default System Prompt** (lines 45-54):
```
You must answer SOLELY based on context provided by 'retrieve_docs'.
Do NOT use outside knowledge.
Always cite the 'Source' file name.
```

**Customize:** Create `prompts/system_prompt.txt` or set `SYSTEM_PROMPT_PATH` env var.

---

### 6. Chunking Strategy

**Chunking Parameters:**
```python
chunker = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,      # Default: 1000 chars
    chunk_overlap=CHUNK_OVERLAP  # Default: 200 chars
)
```

**Tuning Guide:**
- **Smaller chunks** (500-800): Better for code/structured data
- **Larger chunks** (1500-2000): Better for narrative/book content
- **Overlap:** Maintains context across chunk boundaries

---

## Project Structure

```
manual-master/
├── src/
│   └── main.py                    # Entry point (574 lines)
├── prompts/
│   └── system_prompt.txt          # (Optional) Custom AI instructions
├── .env                           # Configuration (REQUIRED: OPENAI_API_KEY)
├── requirements.txt               # Dependencies
├── .github/
│   └── copilot-instructions.md    # This file
└── README.md
```

---

## Development Workflows

### First-Time Setup

1. Install system dependencies:
   - **Windows:** `pip install python-magic-bin`
   - **Linux:** `sudo apt-get install libmagic1`
   - **macOS:** `brew install libmagic`

2. Install Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Create `.env` in project root:
   ```ini
   OPENAI_API_KEY=sk-proj-xxxxx
   OPENAI_MODEL=gpt-4o-mini
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   ```

### Running the Agent

```bash
python src/main.py ~/documents/my_folder
```

**Workflow:**
1. Script scans folder and generates content hash
2. Checks if database exists (in `CHROMA_PATH`) for this hash
3. If yes: Loads instantly. If no: Indexes all documents (one-time operation)
4. Enters interactive loop for questions

### Debugging Checklist

| Issue | Solution |
|-------|----------|
| `ImportError: failed to find libmagic` | Install system package (see above) |
| `RateLimitError` | Check OpenAI quota; consider switching `EMBEDDING_MODEL` |
| Files not detected | Check `detect_file_info()` fallback detection |
| Path errors on Windows | Ensure `get_safe_path_str()` is used for I/O |
| Cache not updating | Content-based hash is stable; modify file content to trigger re-index |
| LLM talking from training data | Ensure system prompt is strict (or customize `prompts/system_prompt.txt`) |

---

## Common Modifications

### Customize AI Personality

Create `prompts/system_prompt.txt`:
```
You are a specialized assistant for financial documents.
Answer questions ONLY from retrieved documents.
Use a professional tone and always cite sources.
```

### Adjust Chunking for Code

```bash
CHUNK_SIZE=500 CHUNK_OVERLAP=100 python src/main.py /code/repo
```

### Add New File Format

In `detect_file_info()` and `load_file_content()`:
- Add extension to `relevant_extensions`
- Add MIME type check in `detect_file_info()`
- Add corresponding LangChain loader in `load_file_content()`

### Enable LangSmith Tracing

```ini
# .env
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=lsv2_pt_xxxxx
```

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `langchain` | ReAct agent framework |
| `langchain-openai` | ChatOpenAI + OpenAIEmbeddings |
| `langchain-chroma` | Vector database integration |
| `pypdf` | PDF parsing |
| `beautifulsoup4` | HTML/XML parsing |
| `python-magic-bin` | Windows file type detection |

---

## Tips for AI Agents

1. **Always use path utilities:** Never work with raw path strings; use `resolve_path()` and `get_safe_path_str()`
2. **Content hash is stable:** Renaming folders is safe; modifying file content changes the hash
3. **MIME detection has fallback:** Even without libmagic, the system detects PDF, HTML, XML via headers
4. **Chunk size impacts retrieval:** Smaller chunks = more targeted results; larger chunks = more context
5. **LLM citation mode is enforced:** The system prompt is strict; customization requires `prompts/system_prompt.txt`
6. **Extensions matter:** Only `.pdf`, `.txt`, `.html`, `.htm`, `.xml`, `.md` are indexed in hash calculation
