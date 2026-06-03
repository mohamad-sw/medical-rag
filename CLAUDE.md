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
pipenv run python app.py
# or using the venv python directly
/Users/neonlearn/.local/share/virtualenvs/code-OusFg6gK/bin/python3 app.py
```

## Architecture

This is a **LangChain RAG (Retrieval-Augmented Generation)** project. `app.py` lives at the repo root and follows this pipeline:

1. **Load & split** — PDFs from `docs/` are loaded with `pypdf` and chunked via `RecursiveCharacterTextSplitter` (chunk_size=500, overlap=50).
2. **Embed & store** — Chunks are embedded and persisted to `chroma_db/` using ChromaDB with HuggingFace embeddings (`all-MiniLM-L6-v2`).
3. **LLM** — Groq (`llama-3.1-8b-instant`), key loaded from `.env` via `python-dotenv`.
4. **RAG chain** — LangChain LCEL chain: `retriever | format_docs` → `ChatPromptTemplate` → `LLM` → `StrOutputParser`.
5. **Cleanup** — `chroma_db/` is deleted after every run.

### Services required

- **Groq API key**: set `GROQ_API_KEY` in `.env`
- **Pinecone** credentials are in `.env` but not yet wired up in `app.py`
