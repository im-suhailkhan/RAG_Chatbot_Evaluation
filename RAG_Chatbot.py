import os
import warnings
import logging
import hashlib
import tempfile
import json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

# Load environment variables (e.g., GROQ_API_KEY)
load_dotenv()

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.title("Ask Chatbot!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# Sidebar for PDF selection
st.sidebar.header("RAG Document")
pdf_path_input = st.sidebar.text_input("PDF path", value="./resume.pdf")
uploaded_pdf = st.sidebar.file_uploader("Or upload a PDF", type=["pdf"])

pdf_hash: str
pdf_path: str

if uploaded_pdf is not None:
    pdf_bytes = uploaded_pdf.getvalue()
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

    cached_hash = st.session_state.get("pdf_hash")
    cached_path = st.session_state.get("uploaded_pdf_path")
    if cached_hash != pdf_hash or not cached_path or not Path(cached_path).exists():
        tmp_path = os.path.join(tempfile.gettempdir(), f"rag_chatbot_{pdf_hash}.pdf")
        Path(tmp_path).write_bytes(pdf_bytes)
        st.session_state["pdf_hash"] = pdf_hash
        st.session_state["uploaded_pdf_path"] = tmp_path

    pdf_path = st.session_state["uploaded_pdf_path"]
else:
    pdf_path = pdf_path_input
    pdf_hash = f"path:{pdf_path}"

# Cache the vectorstore (avoids reloading on every rerun)
@st.cache_resource
def get_vectorstore(_pdf_hash: str, _pdf_path: str):
    pdf_file = Path(_pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(
            f"PDF not found at `{_pdf_path}`. Please provide a valid PDF file."
        )

    loader = PyPDFLoader(str(pdf_file))
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore

vectorstore = get_vectorstore(pdf_hash, pdf_path)

# --- Set up the RAG chain and judge chain ---
# Get API key from environment (set GROQ_API_KEY in .env or in Streamlit secrets)
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found. Please set it in your environment.")
    st.stop()

model = "llama-3.1-8b-instant"
groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)

# Retriever for context retrieval
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt_template = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.

Context:
{context}

Question:
{question}
""")

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt_template
    | groq_chat
    | StrOutputParser()
)

judge_prompt = ChatPromptTemplate.from_template("""
You are an expert RAG evaluator. Score the following answer on a scale of 1-5 (5 = perfect).

Question: {question}
Context (resume chunks): {context}
Generated Answer: {answer}

Evaluate ONLY these 3 metrics:
1. Faithfulness (1-5): Does the answer use ONLY information from the context? No hallucinations.
2. Answer Relevance (1-5): Does it directly answer the question asked?
3. Correctness (1-5): Is the information factually accurate according to the resume?

Return ONLY a valid JSON object:
{{
  "faithfulness": X,
  "relevance": X,
  "correctness": X,
  "explanation": "one short sentence explaining the scores"
}}
""")
judge_chain = judge_prompt | groq_chat

# --- Main chat input ---
prompt = st.chat_input("Pass your prompt here")

if prompt:
    # 1. Add user message to history and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate response using RAG chain
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Retrieve the context documents for later evaluation
            retrieved_docs = retriever.invoke(prompt)
            context = format_docs(retrieved_docs)

            # Generate answer
            response = rag_chain.invoke(prompt)
            st.markdown(response)

    # 3. Save assistant response and the context used
    st.session_state.messages.append({"role": "assistant", "content": response})
    # Store the latest Q/A and context for the evaluation button
    st.session_state.last_prompt = prompt
    st.session_state.last_response = response
    st.session_state.last_context = context

# --- Evaluation button (appears after a response has been generated) ---
if st.session_state.get("last_response") is not None:
    if st.button("🔍 Evaluate this answer", key="eval_button"):
        with st.spinner("Evaluating with Groq..."):
            try:
                eval_input = {
                    "question": st.session_state.last_prompt,
                    "context": st.session_state.last_context,
                    "answer": st.session_state.last_response
                }
                eval_result = judge_chain.invoke(eval_input)
                # Parse JSON safely
                scores = json.loads(eval_result.content.strip())
                st.success("✅ Evaluation Complete")
                st.json(scores)  # nice formatted JSON

                # Optional: store evaluation history
                st.session_state.setdefault("evaluations", []).append({
                    "question": st.session_state.last_prompt,
                    "answer": st.session_state.last_response,
                    "scores": scores
                })
            except Exception as e:
                st.error(f"Evaluation failed: {e}")