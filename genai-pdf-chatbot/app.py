import os
import streamlit as st
from dotenv import load_dotenv
from chatbot import build_vector_store, answer_question

load_dotenv()

st.set_page_config(
    page_title="GenAI PDF Chatbot",
    page_icon="📄",
    layout="wide",
)

st.title("📄 GenAI PDF Chatbot")
st.caption("Ask questions about your PDF documents using Retrieval-Augmented Generation (RAG).")

api_key = os.getenv("OPENAI_API_KEY", "").strip()
if not api_key:
    st.error("OPENAI_API_KEY is missing. Add it to your .env file and restart the app.")
    st.stop()

with st.sidebar:
    st.header("Document upload")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )
    chunk_size = st.slider("Chunk size", min_value=500, max_value=2000, value=1000, step=100)
    chunk_overlap = st.slider("Chunk overlap", min_value=50, max_value=400, value=200, step=50)
    top_k = st.slider("Retrieved chunks", min_value=1, max_value=8, value=4)

    if st.button("Process documents", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Upload at least one PDF before processing.")
        else:
            try:
                with st.spinner("Extracting text, creating embeddings, and building the FAISS index..."):
                    vector_store, document_count = build_vector_store(
                        uploaded_files,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                st.session_state.vector_store = vector_store
                st.session_state.document_count = document_count
                st.session_state.messages = []
                st.success(f"Processed {len(uploaded_files)} PDF(s) into {document_count} text chunks.")
            except Exception as exc:
                st.error(f"Could not process the PDFs: {exc}")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.info("Upload PDF files in the sidebar and select **Process documents** to get started.")
    st.stop()

st.success(
    f"Knowledge base ready: {st.session_state.get('document_count', 0)} text chunks indexed."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.write(f"**{source['file']}** — page {source['page']}")

question = st.chat_input("Ask a question about your uploaded documents...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents and generating an answer..."):
                result = answer_question(
                    vector_store=st.session_state.vector_store,
                    question=question,
                    top_k=top_k,
                )
            st.markdown(result["answer"])
            if result["sources"]:
                with st.expander("Sources"):
                    for source in result["sources"]:
                        st.write(f"**{source['file']}** — page {source['page']}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            })
        except Exception as exc:
            error_message = f"An error occurred while generating the answer: {exc}"
            st.error(error_message)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
            })
