import os
import warnings
import logging
import hashlib
import tempfile
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

# Load .env (optional) so GROQ_API_KEY can be set locally.
load_dotenv()

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.title("Ask Chatbot!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

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
        # Cache uploads across reruns by hashing bytes to a deterministic temp file.
        tmp_path = os.path.join(tempfile.gettempdir(), f"rag_chatbot_{pdf_hash}.pdf")
        Path(tmp_path).write_bytes(pdf_bytes)
        st.session_state["pdf_hash"] = pdf_hash
        st.session_state["uploaded_pdf_path"] = tmp_path

    pdf_path = st.session_state["uploaded_pdf_path"]
else:
    pdf_path = pdf_path_input
    pdf_hash = f"path:{pdf_path}"


@st.cache_resource
def get_vectorstore(_pdf_hash: str, _pdf_path: str):
    pdf_file = Path(_pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(
            f"PDF not found at `{_pdf_path}`. Put `reflexion.pdf` in the project folder or upload a PDF in the sidebar."
        )

    # Load PDF
    loader = PyPDFLoader(str(pdf_file))
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")

    # Create vector store (replaces VectorstoreIndexCreator)
    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore

vectorstore = get_vectorstore(pdf_hash, pdf_path)

prompt = st.chat_input("Pass your prompt here")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        st.error("Missing `GROQ_API_KEY`. Set it in your environment or create a local `.env` file.")
        st.stop()

    model = "llama-3.1-8b-instant"
    groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)

    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        prompt_template = ChatPromptTemplate.from_template(
            """Answer the question based only on the context below.

        Context:
        {context}

        Question:
        {question}
        """
        )

        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt_template
            | groq_chat
            | StrOutputParser()
        )

        response = rag_chain.invoke(prompt)
    except Exception as e:
        st.error(f"Error: {str(e)}")


