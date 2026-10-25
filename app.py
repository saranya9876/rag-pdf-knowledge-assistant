
import os
import subprocess
import sys
import streamlit as st

st.set_page_config(page_title="RAG Chatbot", layout="wide")

DB_PATH = "chroma_db"
DATA_PATH = "data"


def get_pdf_files():
    if not os.path.isdir(DATA_PATH):
        return []
    return [
        f for f in os.listdir(DATA_PATH)
        if f.lower().endswith(".pdf")
    ]


def get_document_count():
    if not os.path.exists(os.path.join(DB_PATH, "chroma.sqlite3")):
        return 0

    from langchain_chroma import Chroma
    from langchain_community.embeddings import FastEmbedEmbeddings

    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=FastEmbedEmbeddings()
    )
    return db._collection.count()


pdf_files = get_pdf_files()

if not pdf_files:
    st.error("No PDF files found in the data/ folder.")
    st.stop()

try:
    count = get_document_count()

    if count == 0:
        with st.spinner(
            "Preparing your PDF knowledge base. This may take several minutes..."
        ):
            result = subprocess.run(
                [sys.executable, "ingest.py"],
                capture_output=True,
                text=True,
                timeout=900
            )

        if result.returncode != 0:
            st.error("PDF ingestion failed.")
            st.code((result.stdout + "\n" + result.stderr)[-4000:])
            st.stop()

        count = get_document_count()

    st.caption(f"Knowledge base status: {count} document chunks indexed.")

    if count == 0:
        st.error("Ingestion completed, but the database contains no chunks.")
        st.stop()

except Exception as e:
    st.error(f"Knowledge base setup failed: {e}")
    st.stop()


try:
    from retriever import retrieve
    from generator import generate_answer
except Exception as e:
    st.error("Failed to load the retrieval or generation modules.")
    st.exception(e)
    st.stop()

st.title("RAG Chatbot (PDF Knowledge Assistant)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask something from your documents...")

if query:
    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Retrieving relevant chunks..."):
        docs = retrieve(query)

    with st.expander("Sources & Retrieved Chunks", expanded=True):
        if not docs:
            st.warning("No relevant document chunks were retrieved.")
        else:
            for i, doc in enumerate(docs, start=1):
                metadata = doc.metadata or {}
                source = metadata.get("source", "Unknown PDF")
                page = metadata.get("page", "Unknown")

                st.markdown(f"**Source {i}:** `{source}`")
                st.caption(f"Page: {page}")
                st.write(doc.page_content[:500])
                st.divider()

    with st.spinner("Generating answer..."):
        answer = generate_answer(query, docs)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    with st.chat_message("assistant"):
        st.markdown(answer)
