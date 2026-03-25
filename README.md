# 📄 RAG Chatbot with Groq, Streamlit and LLM Judge as evaluator

A **Retrieval-Augmented Generation (RAG)** chatbot built with Streamlit, LangChain, and Groq.  
It answers questions based on a PDF document (like a resume) and provides automatic evaluation of answer quality.

## ✨ Features

- Upload a PDF file or specify a local path
- Ask questions about the document content
- Retrieve relevant chunks using FAISS vector search
- Generate answers with Groq's `llama-3.1-8b-instant` model
- Evaluate answers on **faithfulness**, **relevance**, and **correctness** (1–5 scale)

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo

