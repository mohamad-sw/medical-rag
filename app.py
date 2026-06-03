# ragas imports ChatVertexAI from a module removed in langchain-community 0.3+; stub it out
import sys as _sys
from types import ModuleType as _M
_vc = _M("langchain_community.chat_models.vertexai")
_vc.ChatVertexAI = type("ChatVertexAI", (), {})  # type: ignore[attr-defined]
_sys.modules["langchain_community.chat_models.vertexai"] = _vc

import io
import os

import chromadb
import pypdf
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, Faithfulness

load_dotenv()

# On Streamlit Cloud there is no .env; inject st.secrets into os.environ
for _k, _v in st.secrets.items():
    if _k not in os.environ:
        os.environ[_k] = str(_v)

st.set_page_config(page_title="PDF Q&A", layout="wide")

st.markdown("""
<style>
    html, body, [class*="css"] { font-size: 20px !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    return ChatGroq(model="llama-3.1-8b-instant")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

st.title("RAG-Powered Document Intelligence")
st.markdown("""
Ask questions about any PDF using a fully local **Retrieval-Augmented Generation (RAG)** pipeline.

**How it works:**
- Uploaded PDFs are parsed and split into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`
- Each chunk is embedded into a high-dimensional vector using **HuggingFace's `all-MiniLM-L6-v2`** model, running entirely on-device — no external embedding API
- Embeddings are indexed in an **in-memory ChromaDB** vector store for fast semantic search
- At query time, the top-3 most relevant chunks are retrieved and injected into a prompt sent to **LLaMA 3.1 8B** via the Groq inference API
- The LLM answers strictly from the retrieved context, preventing hallucination on out-of-scope questions
- Response quality is measured on-demand with **RAGAS** — two reference-free metrics (**Faithfulness** and **Answer Relevancy**) score how grounded and relevant each answer is

**Stack:** LangChain · ChromaDB · HuggingFace Transformers · Streamlit · RAGAS
""")

st.divider()

DEFAULT_PDF = "medical_diseases_symptoms.pdf"
DEFAULT_PDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), DEFAULT_PDF)

SUGGESTIONS = [
    "What is the ABCDE criteria used to evaluate in skin cancer?",
    "What is Raynaud's phenomenon and which connective tissue disease is it an early sign of?",
    "What is a myasthenic crisis and which muscles does it affect?",
    "What is fibro fog as described in fibromyalgia?",
]

source = st.radio(
    "PDF source",
    ["Upload your file", "Use default file (100 diseases)"],
    horizontal=True,
)

using_default = source == "Use default file (100 diseases)"

# Clear cached state on source switch
if st.session_state.get("pdf_source") != source:
    st.session_state.pdf_source = source
    if "vectorstore" in st.session_state:
        st.session_state.vectorstore.delete_collection()
    for key in ("vectorstore", "retriever", "current_file", "last_question",
                "last_chunks", "last_response", "ragas_scores"):
        st.session_state.pop(key, None)

if using_default:
    active_name = DEFAULT_PDF
    needs_indexing = st.session_state.get("current_file") != active_name
    if needs_indexing:
        with open(DEFAULT_PDF_PATH, "rb") as f:
            active_file = io.BytesIO(f.read())
    else:
        active_file = None
else:
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    active_file = uploaded_file
    active_name = uploaded_file.name if uploaded_file else None
    needs_indexing = bool(active_file and st.session_state.get("current_file") != active_name)

if needs_indexing and active_file:
    if "vectorstore" in st.session_state:
        st.session_state.vectorstore.delete_collection()

    bar = st.progress(0, text="Reading PDF...")

    reader = pypdf.PdfReader(active_file)
    documents = [Document(page_content=page.extract_text()) for page in reader.pages]
    bar.progress(25, text="Splitting into chunks...")

    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(documents)
    bar.progress(50, text="Loading embedding model...")

    embeddings = load_embeddings()
    bar.progress(75, text="Building vector store...")

    vectorstore = Chroma.from_documents(chunks, embeddings, client=chromadb.EphemeralClient(), collection_name="data")
    bar.progress(100, text="Done!")

    st.session_state.vectorstore = vectorstore
    st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    st.session_state.current_file = active_name
    st.session_state.pop("last_question", None)
    bar.empty()

if "retriever" in st.session_state:
    st.success(f"Ready — {st.session_state.current_file}")

    if using_default:
        st.caption("Suggested questions:")
        cols = st.columns(2)
        for i, s in enumerate(SUGGESTIONS):
            if cols[i % 2].button(s, use_container_width=True):
                st.session_state["question_input"] = s

    question = st.text_input("Ask a question", key="question_input")

    if question:
        prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context below.
If you don't know, say "I don't have enough information."

Context:
{context}

Question: {question}
""")

        rag_chain = (
            {"context": st.session_state.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | load_llm()
            | StrOutputParser()
        )

        # Cache results so RAGAS button click doesn't re-run the RAG chain
        if st.session_state.get("last_question") != question:
            with st.spinner("Generating response..."):
                retrieved = st.session_state.retriever.invoke(question)
                response = rag_chain.invoke(question)
            st.session_state.last_question = question
            st.session_state.last_chunks = retrieved
            st.session_state.last_response = response
            st.session_state.pop("ragas_scores", None)

        retrieved = st.session_state.last_chunks
        response = st.session_state.last_response

        st.write(response)

        with st.expander(f"Retrieved chunks ({len(retrieved)})"):
            for i, chunk in enumerate(retrieved):
                st.markdown(f"**Chunk {i + 1}**")
                st.caption(chunk.page_content)
                st.divider()

        st.divider()
        st.subheader("RAGAS Evaluation")
        st.caption("Measures response quality using two reference-free metrics powered by the same LLM.")

        if st.button("Evaluate with RAGAS"):
            with st.spinner("Running evaluation — this makes additional LLM calls and takes ~30 seconds..."):
                try:
                    evaluator_llm = LangchainLLMWrapper(load_llm())
                    evaluator_emb = LangchainEmbeddingsWrapper(load_embeddings())

                    sample = SingleTurnSample(
                        user_input=question,
                        retrieved_contexts=[c.page_content for c in retrieved],
                        response=response,
                    )
                    dataset = EvaluationDataset(samples=[sample])

                    results = evaluate(
                        dataset=dataset,
                        metrics=[Faithfulness(), AnswerRelevancy()],
                        llm=evaluator_llm,
                        embeddings=evaluator_emb,
                    )

                    st.session_state.ragas_scores = dict(results.scores[0])
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")

        if st.session_state.get("ragas_scores") and st.session_state.get("last_question") == question:
            scores = st.session_state.ragas_scores
            col1, col2 = st.columns(2)

            faithfulness = scores.get("faithfulness", 0)
            relevancy = scores.get("answer_relevancy", 0)

            col1.metric(
                "Faithfulness",
                f"{faithfulness:.2f} / 1.00",
                help="Are all claims in the answer supported by the retrieved context? Low score = hallucination risk.",
            )
            col2.metric(
                "Answer Relevancy",
                f"{relevancy:.2f} / 1.00",
                help="Does the answer actually address what was asked? Low score = off-topic or evasive response.",
            )

            overall = (faithfulness + relevancy) / 2
            color = "#2ecc71" if overall >= 0.7 else "#e67e22" if overall >= 0.4 else "#e74c3c"
            label = "Good" if overall >= 0.7 else "Moderate" if overall >= 0.4 else "Poor"
            st.markdown(
                f"<div style='margin-top:0.5rem; padding:0.6rem 1rem; border-radius:6px; "
                f"background:{color}22; border-left:4px solid {color}; font-size:1rem;'>"
                f"Overall: <strong>{label}</strong> ({overall:.2f})</div>",
                unsafe_allow_html=True,
            )
