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
1. Clone Repository
git clone https://github.com/samialrae/RAG-Project-ReadyTensor.git
cd RAG-Project-ReadyTensor

2. Install Dependencies
pip install -r requirements.txt

3. Set API Key

Create a .env file:

OPENAI_API_KEY=your_api_key_here

4. Run Demo
python rag_demo.py


Or launch UI:

streamlit run app.py
   python "my Rag boat.py"

   pip install -r requirements.txt

