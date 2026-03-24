# 📄 RAG Chatbot with Groq & Streamlit

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

## Demo Screenshot:

<img width="1252" height="835" alt="image" src="https://github.com/user-attachments/assets/0c82095b-b54a-4098-ac59-2a15f45025b7" />
