import os
import re
import json

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# Load environment & setup

load_dotenv()

DATA_FILE = "project_1_publications.json"

# Query Preprocessing

def preprocess_query(query: str) -> str:
    """Clean query for better retrieval."""
    query = query.lower().strip()
    query = re.sub(r"[^a-zA-Z0-9\s]", "", query)
    return query


# Load and preprocess docs

def load_documents(file_path: str):
    with open(file_path, "r") as f:
        data = json.load(f)

    docs = []
    for record in data:
        text = f"Title: {record.get('title', '')}\n\nContent: {record.get('content', '')}"
        docs.append(Document(page_content=text, metadata={"id": record.get("id", None)}))
    return docs

docs = load_documents(DATA_FILE)
print("Loaded:", len(docs), "docs")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
chunks = splitter.split_documents(docs)
print("Chunks:", len(chunks))

# Build vectorstore & retriever

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# LLM setup (Flan-T5 base)

gen_pipeline = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256,
    temperature=0,
)

llm = HuggingFacePipeline(pipeline=gen_pipeline)

# Prompt template

prompt = PromptTemplate.from_template(
    """You are a helpful assistant that answers ONLY using the provided context.
If the answer is not in the context, say: "I don't know based on the provided documents."

Question:
{question}

Context:
{context}

Answer:"""
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


# RAG Chain

rag_chain = (
    {
        "question": RunnablePassthrough(),
        "context": retriever | format_docs,
    }
    | prompt
    | llm
    | StrOutputParser()
)


# Main run

if __name__ == "__main__":
    while True:
        user_q = input("\nEnter your question (or 'exit' to quit): ")
        if user_q.lower() in ["exit", "quit"]:
            break

        clean_q = preprocess_query(user_q)
        print("\n Query after preprocessing:", clean_q)

        answer = rag_chain.invoke(clean_q)
        print("\n Answer:", answer)


