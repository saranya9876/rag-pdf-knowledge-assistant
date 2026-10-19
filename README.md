# RAG PDF Knowledge Assistant

A local PDF question-answering chatbot built with Retrieval-Augmented Generation (RAG). It retrieves relevant passages from indexed PDF documents and uses a locally hosted language model to generate answers with supporting evidence.

## Features

- PDF document ingestion and text chunking
- Vector search using Chroma
- Embeddings using FastEmbed
- Streamlit chat interface
- Local LLM inference through LM Studio's OpenAI-compatible API
- Retrieved source chunks and page metadata shown in the interface
- Basic handling of questions that are not supported by the retrieved context

## Technology stack

- Python
- Streamlit
- LangChain
- ChromaDB
- FastEmbed
- LM Studio
- OpenAI-compatible API client

## Project files

- `app.py` — Streamlit chat interface
- `ingest.py` — ingests PDF files and builds the vector database
- `retriever.py` — retrieves relevant document chunks
- `generator.py` — prepares answers from retrieved evidence
- `config.py` — project configuration
- `data/` — local PDF input folder (not included in this repository)
- `chroma_db/` — generated vector database (not included in this repository)

## Requirements

- Python 3.10 or a compatible Python version for the installed dependencies
- LM Studio installed locally
- A language model downloaded and loaded in LM Studio
- LM Studio's local server enabled (the project configuration currently uses `http://127.0.0.1:1234/v1`)

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/saranya9876/rag-pdf-knowledge-assistant.git
cd rag-pdf-knowledge-assistant
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add your PDF documents

Create a `data` folder in the project root and place the PDFs you are allowed to use inside it. The PDFs and generated vector database are excluded from this repository.

### 5. Start LM Studio

1. Download and load a model in LM Studio.
2. Start the local server using LM Studio's developer/server interface.
3. Confirm that the server URL and model identifier match the settings in `config.py` and the code in `generator.py`.

### 6. Build the vector database

Run:

```bash
python ingest.py
```

Run ingestion again when you change the documents or need to rebuild the index.

### 7. Start the chatbot

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`, that you can open in your browser.

## Notes and limitations

- This project currently expects LM Studio to be running on the same computer as the chatbot. A cloud deployment cannot access your computer's `127.0.0.1` endpoint; deployment requires a separately accessible inference service or a different model setup.
- The `data/` and `chroma_db/` folders are intentionally excluded. You must add your own PDFs and create the index locally.
- Answers depend on the quality of retrieval and the source documents. Verify important answers against the cited passages.
- This repository is a learning/demo project, not a production-hardened service.

## License

No license has been added yet. Add a license file if you decide how you want others to use, modify, and distribute this project.
