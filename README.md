# rag-publications-bot
RAG-powered assistant built for ReadyTensor Project 1

# RAG Publications Bot

This project is my submission for **ReadyTensor Project 1** in the Agentic AI Developer Certification Program.

 Overview
This is a **Retrieval-Augmented Generation (RAG) assistant** that can:
- Ingest custom documents (`project_1_publications.json`)
- Chunk and embed them using HuggingFace models
- Store embeddings in a FAISS vector database
- Retrieve relevant documents for user queries
- Generate answers using an open-source LLM (`flan-t5-base`)

##  Tech Stack
- Python
- LangChain
- FAISS
- Sentence-Transformers
- HuggingFace Transformers

##  Usage
1. Clone this repo
2. Install dependencies:
   ```bash
   python "my Rag boat.py"

   pip install -r requirements.txt

