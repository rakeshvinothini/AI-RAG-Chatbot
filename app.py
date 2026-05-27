# ==========================================
# 🤖 Professional RAG Chatbot
# ==========================================

import os
import streamlit as st
from dotenv import load_dotenv

# PDF Loader
from langchain_community.document_loaders import PyPDFLoader

# Text Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Vector Store
from langchain_community.vectorstores import FAISS

# Gemini SDK
import google.generativeai as genai

# ==========================================
# LOAD ENV
# ==========================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.stTextInput input {
    border-radius: 10px;
    padding: 12px;
}

.answer-box {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 10px;
    color: white;
    margin-top: 20px;
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# TITLE
# ==========================================

st.title("🤖 AI RAG Chatbot")

st.write("Ask questions from your PDF knowledge base")

# ==========================================
# LOAD DOCUMENTS
# ==========================================

@st.cache_resource
def load_documents():

    documents = []

    folder_path = r"D:\New RAG\data"

    for file in os.listdir(folder_path):

        if file.endswith(".pdf"):

            pdf_path = os.path.join(folder_path, file)

            loader = PyPDFLoader(pdf_path)

            documents.extend(loader.load())

    return documents

# ==========================================
# CREATE VECTOR STORE
# ==========================================

@st.cache_resource
def create_vector_store():

    documents = load_documents()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    split_docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(
        split_docs,
        embeddings
    )

    return vector_store

# ==========================================
# LOAD GEMINI MODEL
# ==========================================

model = genai.GenerativeModel(
    "models/gemini-flash-lite-latest"
)

# ==========================================
# USER QUESTION
# ==========================================

question = st.text_input("Ask Your Question")

# ==========================================
# ANSWER GENERATION
# ==========================================

if question:

    with st.spinner("Searching Answer..."):

        vector_store = create_vector_store()

        docs = vector_store.similarity_search(
            question,
            k=4
        )

        # Combine Context
        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        # Prompt
        prompt = f"""
        Answer the question using ONLY the context below.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """

        # Generate Response
        response = model.generate_content(prompt)

        answer = response.text

    # ======================================
    # DISPLAY ANSWER
    # ======================================

    st.subheader("📌 Answer")

    st.markdown(
        f"""
        <div class="answer-box">
        {answer}
        </div>
        """,
        unsafe_allow_html=True
    )

    # ======================================
    # SHOW SOURCE CHUNKS
    # ======================================

    with st.expander("📄 Source Chunks"):

        for i, doc in enumerate(docs):

            st.write(f"Chunk {i+1}")

            st.write(doc.page_content[:500])

            st.divider()