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

load_dotenv()

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    return ChatGroq(model="llama-3.1-8b-instant")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

st.title("PDF Q&A")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and ("current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name):
    if "vectorstore" in st.session_state:
        st.session_state.vectorstore.delete_collection()

    bar = st.progress(0, text="Reading PDF...")

    reader = pypdf.PdfReader(uploaded_file)
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
    st.session_state.current_file = uploaded_file.name
    bar.empty()

if "retriever" in st.session_state:
    st.success(f"Ready — {st.session_state.current_file}")

    question = st.text_input("Ask a question about your PDF")

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

        with st.spinner("Generating response..."):
            response = rag_chain.invoke(question)

        st.write(response)
