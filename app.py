import shutil

import pypdf
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

# --- 1. Load & split ---
reader = pypdf.PdfReader("./docs/data.pdf")
documents = [
    Document(page_content=page.extract_text(), metadata={"source": "./docs/data.pdf", "page": i})
    for i, page in enumerate(reader.pages)
]
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(documents)

# --- 2. Embed & store ---
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db", collection_name="data")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- 3. LLM ---
llm = ChatGroq(model="llama-3.1-8b-instant")

# --- 4. Prompt ---
prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context below.
If you don't know, say "I don't have enough information."

Context:
{context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

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
