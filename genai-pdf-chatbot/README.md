# GenAI PDF Chatbot (RAG)

A Streamlit application that lets users upload PDF documents and ask natural-language questions about their content. The application uses a Retrieval-Augmented Generation (RAG) pipeline to retrieve relevant document passages and generate context-grounded answers with an OpenAI chat model.

## Features

- Upload and process multiple PDF documents.
- Extract text page-by-page using `pdfplumber`.
- Split extracted text into overlapping chunks.
- Generate embeddings using OpenAI embeddings.
- Store and search document vectors using FAISS.
- Retrieve relevant chunks for a user question.
- Generate answers using an OpenAI chat model and LangChain.
- Display source filenames and page numbers.
- Configure chunk size, overlap, and number of retrieved chunks from the UI.

## Technology stack

- Python
- Streamlit
- LangChain
- OpenAI API
- FAISS
- pdfplumber

## RAG workflow

1. **Document ingestion:** Users upload PDF files through the Streamlit interface.
2. **Text extraction:** `pdfplumber` extracts text from each PDF page and records source metadata.
3. **Chunking:** `RecursiveCharacterTextSplitter` divides page text into smaller overlapping chunks.
4. **Embedding:** OpenAI's `text-embedding-3-small` model converts chunks into vector embeddings.
5. **Vector indexing:** FAISS stores the embeddings for similarity-based retrieval.
6. **Retrieval:** Relevant chunks are retrieved when a user submits a question.
7. **Generation:** LangChain passes the retrieved context and question to `gpt-4o-mini`.
8. **Response:** The generated answer and source page references are displayed in Streamlit.

## Project structure

```text
genai-pdf-chatbot/
├── app.py
├── chatbot.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Prerequisites

- Python 3.10 or 3.11
- An OpenAI API key with API access and billing configured
- Internet access for OpenAI API requests

## Setup and run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/genai-pdf-chatbot.git
cd genai-pdf-chatbot
```

### 2. Create and activate a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure the API key

Copy `.env.example` to `.env` and replace the placeholder with your own key:

```env
OPENAI_API_KEY=your_actual_openai_api_key
```

Do not commit `.env` or expose your API key in source code, screenshots, logs, or GitHub.

### 5. Start the application

```bash
streamlit run app.py
```

Streamlit will display a local URL, usually `http://localhost:8501`.

## Usage

1. Open the application in your browser.
2. Upload one or more text-based PDF documents.
3. Adjust chunk size, overlap, and retrieval count if required.
4. Select **Process documents**.
5. Enter a question in the chat input.
6. Review the generated answer and expand **Sources** to see the referenced PDF pages.

## Configuration

The sidebar exposes the following retrieval settings:

- **Chunk size:** Maximum target size of each text chunk.
- **Chunk overlap:** Shared text between adjacent chunks to preserve context.
- **Retrieved chunks (`top_k`):** Number of similar chunks passed to the answer-generation chain.

The embedding model and chat model are configured in `chatbot.py`.

## Limitations

- This version extracts selectable text from PDFs. Scanned/image-only PDFs require an OCR step, which is not included.
- The FAISS index is created in memory when documents are processed; it is not persisted between application restarts.
- OpenAI API usage may incur charges.
- Answers depend on the quality and content of the uploaded documents and retrieved passages.

## Security

- Keep API keys in `.env` or a secure secrets manager.
- `.env` is excluded through `.gitignore`.
- Do not upload confidential or sensitive documents unless you are authorized to process them with the configured service.

## Possible enhancements

- Add OCR support for scanned PDFs.
- Persist and reload FAISS indexes.
- Add chat history-aware retrieval.
- Support additional document formats.
- Add automated tests and deployment configuration.

## License

Choose and add a license before making the repository public. If you do not intend to grant reuse rights, do not add an open-source license by default.
