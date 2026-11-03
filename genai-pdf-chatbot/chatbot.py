from __future__ import annotations

from typing import Any
import pdfplumber
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


def extract_pdf_documents(uploaded_files: list[Any]) -> list[Document]:
    """Extract text page-by-page and retain source filename and page metadata."""
    documents: list[Document] = []

    for uploaded_file in uploaded_files:
        uploaded_file.seek(0)
        with pdfplumber.open(uploaded_file) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": uploaded_file.name,
                                "page": page_number,
                            },
                        )
                    )

    if not documents:
        raise ValueError(
            "No extractable text was found. The PDFs may be scanned images; "
            "OCR is not included in this version."
        )

    return documents


def build_vector_store(
    uploaded_files: list[Any],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> tuple[FAISS, int]:
    """Create text chunks, embed them with OpenAI, and index them in FAISS."""
    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    pages = extract_pdf_documents(uploaded_files)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    if not chunks:
        raise ValueError("The uploaded PDFs did not produce any text chunks.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store, len(chunks)


def _format_documents(documents: list[Document]) -> str:
    """Format retrieved chunks with source metadata for the LLM context."""
    formatted = []
    for doc in documents:
        source = doc.metadata.get("source", "Unknown file")
        page = doc.metadata.get("page", "Unknown")
        formatted.append(
            f"[Source: {source}, Page: {page}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)


def answer_question(
    vector_store: FAISS,
    question: str,
    top_k: int = 4,
) -> dict[str, Any]:
    """Retrieve relevant PDF chunks and generate a grounded answer."""
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant that answers questions using the
provided document context.

Rules:
- Answer using only the supplied context.
- If the context does not contain the answer, clearly say that the documents
  do not provide enough information.
- Do not invent facts.
- Keep the answer clear and concise.
- Refer to source filenames and page numbers when useful.

Document context:
{context}""",
            ),
            ("human", "{question}"),
        ]
    )

    chain = (
        {
            "context": retriever | _format_documents,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(question)
    retrieved_docs = retriever.invoke(question)

    seen = set()
    sources = []
    for doc in retrieved_docs:
        key = (doc.metadata.get("source", "Unknown file"), doc.metadata.get("page", "Unknown"))
        if key not in seen:
            seen.add(key)
            sources.append({"file": key[0], "page": key[1]})

    return {"answer": answer, "sources": sources}
