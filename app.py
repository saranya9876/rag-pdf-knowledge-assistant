
import streamlit as st
from retriever import retrieve
from generator import generate_answer

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("RAG Chatbot (PDF Knowledge Assistant)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask something from your documents...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Retrieving relevant chunks..."):
        docs = retrieve(query)

    with st.spinner("Generating answer..."):
        answer = generate_answer(query, docs)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)

    with st.expander("Sources & Retrieved Chunks"):
        if not docs:
            st.info("No relevant document chunks were retrieved.")
        else:
            for i, doc in enumerate(docs, start=1):
                metadata = doc.metadata or {}
                source = metadata.get("source", "Unknown PDF")
                page = metadata.get("page", "Unknown")

                st.markdown(f"**Source {i}:** `{source}`")
                st.caption(f"Page: {page}")
                st.write(doc.page_content[:500])
                st.divider()
