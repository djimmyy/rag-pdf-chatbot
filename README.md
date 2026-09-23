# RAG PDF Chatbot

A conversational assistant that answers questions about PDF documents in natural language, grounding every response in the document's actual content to limit hallucinations.

## Overview

The app lets a user upload one or several PDFs and ask questions about them in a chat interface. Instead of relying on the LLM's own knowledge, the answer is generated from passages retrieved directly from the uploaded documents, and those source passages are shown alongside the answer so the user can verify where the information came from.

## How It Works

1. **Text extraction** — each uploaded PDF is parsed page by page and cleaned (whitespace normalization, stray character removal).
2. **Chunking** — the cleaned text is split into overlapping chunks (`RecursiveCharacterTextSplitter`), with chunk size and overlap adjustable live from the sidebar.
3. **Embedding** — each chunk is embedded with a multilingual sentence-transformer model (`paraphrase-multilingual-MiniLM-L12-v2`), so the app handles documents and questions in multiple languages.
4. **Persistent vector storage** — embeddings are stored in a local ChromaDB collection, keyed by an MD5 hash of the uploaded files' content. If the same set of files is uploaded again, the existing collection is reused instead of reprocessing everything from scratch; if the files change, the app detects the new hash and reprocesses automatically.
5. **Retrieval** — on each question, the most relevant chunks are retrieved from ChromaDB (number of chunks adjustable from the sidebar) based on embedding similarity.
6. **Generation** — the retrieved chunks are inserted into a prompt that instructs the model to answer only from the provided context, and to explicitly say when the answer isn't in the document, rather than guessing. The response is generated locally via Ollama (Mistral), with the full conversation history passed along for context.
7. **Source display** — each answer is shown with an expandable panel listing the actual passages used to generate it, so the user can check the answer against the source material directly.

## Tech Stack

Python, Streamlit, LangChain (text splitting), sentence-transformers, ChromaDB, Ollama (Mistral), pypdf.

## Features

- Multi-PDF support, processed together into a single queryable collection
- Automatic detection of new or changed files, with reprocessing only when needed
- Adjustable RAG parameters (chunk size, chunk overlap, number of retrieved passages) directly from the UI
- Persistent storage — previously processed documents don't need to be re-embedded on the next run
- Full conversation history maintained across turns
- Source passages shown for every answer, for verifiability

## Limitations

- Runs against a local LLM (Mistral via Ollama), so response quality and speed depend on the local model and hardware rather than a larger hosted model.
- Retrieval is based on chunk-level embedding similarity only; there is no re-ranking or hybrid (keyword + semantic) search step.
- The MD5-based collection cache is keyed on exact file content, so any change to the uploaded files, however small, triggers a full reprocessing pass.

## Repository Contents

- `main.py` — the complete Streamlit application: PDF processing, embedding, ChromaDB storage and retrieval, prompt construction, and chat UI.
