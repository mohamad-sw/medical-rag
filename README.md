# MedQuery — Clinical Document Intelligence

A Retrieval-Augmented Generation (RAG) app that answers clinical questions from medical PDF documents — grounded strictly in source content, with built-in response quality evaluation.

**Live demo:** [diseases-rag.streamlit.app](https://diseases-rag.streamlit.app)

---

## What it does

- Ask questions about medical conditions, symptoms, and clinical findings
- Pre-loaded with a 100-disease medical reference PDF
- Upload any PDF of your own (clinical guidelines, research papers, lab reports)
- Evaluate response quality on demand with RAGAS scores

## How it works

1. **Parse & Chunk** — PDF pages are split into overlapping 500-token segments
2. **Embed** — Each chunk is embedded locally with HuggingFace `all-MiniLM-L6-v2` (no external embedding API)
3. **Retrieve** — Top 3 semantically relevant chunks are fetched from ChromaDB
4. **Answer** — LLaMA 3.1 8B (via Groq) answers using only the retrieved context
5. **Evaluate** — RAGAS measures Faithfulness and Answer Relevancy against the source

## Stack

| Layer | Tool |
|---|---|
| Framework | LangChain (LCEL) |
| LLM | LLaMA 3.1 8B via Groq |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector store | ChromaDB (ephemeral) |
| Evaluation | RAGAS |
| UI | Streamlit |

## Setup

**Prerequisites:** Python 3.12, Pipenv, a Groq API key

```bash
git clone <repo-url>
cd <repo>
pipenv install
```

Create a `.env` file:

```
GROQ_API_KEY=your_key_here
```

Run the app:

```bash
pipenv run streamlit run app.py
```

## Deployment (Streamlit Cloud)

Add your `GROQ_API_KEY` to the app's **Secrets** in the Streamlit Cloud dashboard. The app reads from `st.secrets` automatically when `.env` is not present.
