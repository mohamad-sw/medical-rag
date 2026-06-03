import warnings

from langchain_huggingface import HuggingFaceEmbeddings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import shutil
from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


# --- 1. Load & split ---
# frist run
loader = PyPDFLoader("./project1/docs/data.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)


# --- 2. Embed & store ---
# custom embedding
# embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
# huggingface embedding
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# first run
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db", collection_name="data")
# other runs
# vectorstore = Chroma(
#     persist_directory="./chroma_db",
#     embedding_function=embeddings,
#     collection_name="data"
# )

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print(retriever.invoke("why do bees outperform their rural counterparts?"))

# Test retriever
# results = retriever.invoke("burger is delicious")

# for i, doc in enumerate(results):
#     print(f"--- Chunk {i+1} ---")
#     print(doc.page_content)
#     print()

# --- 3. LLM ---
# llm = ChatOllama(
#     model="gemma3:1b",
#     temperature=0.7,
#     base_url="http://localhost:11434",
#     num_predict=512,
# )

llm = ChatGroq(model="llama-3.1-8b-instant")

# # --- 4. RAG prompt ---
prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context below. 
If you don't know, say "I don't have enough information."

Context:
{context}

Question: {question}
""")

def format_docs(docs):
    context =  "\n\n".join(doc.page_content for doc in docs)
    return context

# --- 5. Chain ---
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 6. Ask ---
response = rag_chain.invoke("why do bees outperform their rural counterparts?")
print(response)

# --- 7. Cleanup ---
vectorstore.delete_collection()
shutil.rmtree("./chroma_db", ignore_errors=True)