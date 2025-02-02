import streamlit as st
import PyPDF2
import faiss
import numpy as np
import openai
import os
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---------------------- CONFIGURATIONS ----------------------

# Securely Fetch OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Client
client = openai.OpenAI(api_key=api_key)

# Load Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------- PDF PROCESSING ----------------------

def extract_text_from_pdfs(uploaded_files):
    """Extract text from multiple PDFs."""
    all_texts = []
    for uploaded_file in uploaded_files:
        text = ""
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            all_texts.append(text.strip())
        except Exception as e:
            st.error(f"⚠️ Error reading PDF: {uploaded_file.name} - {e}")
    return all_texts

# ---------------------- TEXT PROCESSING ----------------------

def preprocess_texts(texts, chunk_size=500, chunk_overlap=50):
    """Splits and preprocesses extracted text into chunks."""
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    for text in texts:
        chunks = text_splitter.split_text(text)
        all_chunks.extend(chunks)
    return all_chunks

# ---------------------- FAISS VECTOR SEARCH ----------------------

def create_vector_index(chunks):
    """Creates FAISS index from text chunks using embeddings."""
    try:
        embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(embeddings)
        return faiss_index, embeddings, chunks
    except Exception as e:
        st.error(f"⚠️ Error creating vector index: {e}")
        return None, None, None

def retrieve_relevant_chunks(query, index, chunks, embeddings, top_k=5):
    """Retrieves top-K relevant sections using FAISS vector search."""
    try:
        query_embedding = embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = index.search(query_embedding, top_k)
        retrieved_texts = [chunks[i] for i in indices[0] if i < len(chunks)]
        return retrieved_texts
    except Exception as e:
        st.error(f"⚠️ Error retrieving relevant sections: {e}")
        return []

# ---------------------- AI-POWERED ANSWER GENERATION ----------------------

def generate_llm_response(query, retrieved_texts):
    """Generates AI response using OpenAI's GPT-4."""
    context = "\n".join(retrieved_texts)
    prompt = f"""
    You are an expert research assistant analyzing multiple academic papers. 
    Based on the following extracted sections, provide a detailed, well-structured answer.

    Context: {context}

    Query: {query}

    Answer:
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert research assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except openai.OpenAIError as e:
        st.error(f"⚠️ OpenAI API Error: {e}")
        return "❌ Error: Could not generate a response."

# ---------------------- STREAMLIT UI ----------------------

st.set_page_config(page_title="📖 DeepDive AI - Research Paper Insights", layout="wide")
st.title("📖 DeepDive AI - Multi-Paper Insight")

# File Upload Section
uploaded_files = st.file_uploader("📂 Upload AI Research Papers (PDF)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("🔍 Extracting and Processing Text..."):
        raw_texts = extract_text_from_pdfs(uploaded_files)
        if not raw_texts:
            st.error("⚠️ No text extracted. Ensure PDFs contain selectable text.")
        else:
            chunks = preprocess_texts(raw_texts)
            faiss_index, embeddings, stored_chunks = create_vector_index(chunks)
            if faiss_index:
                st.success(f"✅ {len(uploaded_files)} PDF(s) processed successfully! {len(chunks)} text chunks created.")

    # Query Input
    query = st.text_input("🔎 Enter your research query:")
    
    if query and faiss_index:
        with st.spinner("🧐 Searching for relevant sections..."):
            retrieved_chunks = retrieve_relevant_chunks(query, faiss_index, stored_chunks, embeddings)
        
        if retrieved_chunks:
            st.subheader("📌 Relevant Sections from the Papers:")
            for i, chunk in enumerate(retrieved_chunks, 1):
                st.markdown(f"**🔹 Section {i}:** {chunk}")

            # AI-Generated Answer
            with st.spinner("🤖 Generating AI Answer..."):
                answer = generate_llm_response(query, retrieved_chunks)

            st.subheader("📝 AI-Generated Answer:")
            st.write(answer)

st.info("🔹 This application provides intelligent insights from multiple research papers using Retrieval-Augmented Generation (RAG) and LLMs.")
