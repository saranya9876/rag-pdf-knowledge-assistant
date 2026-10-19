import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

DATA_PATH = "data"
DB_PATH = "chroma_db"

# -------------------------
# Load PDFs
# -------------------------
def load_pdfs():
    docs = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            path = os.path.join(DATA_PATH, file)
            loader = PyMuPDFLoader(path)
            docs.extend(loader.load())

    return docs


# -------------------------
# Chunking
# -------------------------
def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    return splitter.split_documents(docs)


# -------------------------
# Store in Chroma DB
# -------------------------
def store_embeddings(chunks):
    embeddings = FastEmbedEmbeddings()

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    return db


# -------------------------
# MAIN PIPELINE
# -------------------------
if __name__ == "__main__":
    print("Loading PDFs...")
    docs = load_pdfs()

    print("Chunking documents...")
    chunks = chunk_docs(docs)

    print(f"Total chunks: {len(chunks)}")

    print("Storing embeddings...")
    store_embeddings(chunks)

    print("Ingestion completed successfully ✔")