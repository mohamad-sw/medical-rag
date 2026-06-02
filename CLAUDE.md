# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- Python 3.12, managed with **Pipenv**
- Virtual env: `/Users/neonlearn/.local/share/virtualenvs/code-OusFg6gK/`
- Python interpreter (for running scripts): `/Users/neonlearn/.local/share/virtualenvs/code-OusFg6gK/bin/python3`

## Commands

```bash
# Install dependencies
pipenv install

# Run a script
pipenv run python project1/app.py
# or using the venv python directly
/Users/neonlearn/.local/share/virtualenvs/code-OusFg6gK/bin/python3 project1/app.py
```

## Architecture

This is a **LangChain RAG (Retrieval-Augmented Generation)** project. The single project (`project1/app.py`) follows this pipeline:

1. **Load & split** — PDFs from `project1/docs/` are loaded with `PyPDFLoader` and chunked via `RecursiveCharacterTextSplitter` (chunk_size=500, overlap=50).
2. **Embed & store** — Chunks are embedded and persisted to `chroma_db/` using ChromaDB. Two embedding options exist: Ollama (`nomic-embed-text` at `localhost:11434`) and HuggingFace (`all-MiniLM-L6-v2`).
3. **LLM** — Two LLM options: local Ollama (`gemma3:1b`) or cloud Groq (`llama-3.1-8b-instant`). The active choice is toggled by commenting/uncommenting.
4. **RAG chain** — LangChain LCEL chain: `retriever | format_docs` → `ChatPromptTemplate` → `LLM` → `StrOutputParser`.

### First run vs. subsequent runs

The vector store ingestion code (loading PDFs, splitting, `Chroma.from_documents`) is **commented out** after the first run. On re-runs, ChromaDB is loaded from the persisted `chroma_db/` directory. When adding new documents or changing the embedding model, uncomment the ingestion block and re-run.

### Services required

- **Ollama** (if using local LLM/embeddings): must be running at `http://localhost:11434` with the relevant model pulled (`ollama pull nomic-embed-text`, `ollama pull gemma3:1b`)
- **Groq API key**: currently hardcoded in `app.py:56` — should be moved to `.env` and loaded with `python-dotenv` (the package is already installed)
- **Pinecone** credentials are in `project1/.env` but not yet wired up in `app.py`
