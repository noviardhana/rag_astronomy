# app.py
import streamlit as st
import chromadb
from rag_utils import generate_answer_with_rag, model, client

# ========================
# 1️⃣ Initialize ChromaDB Collection
# ========================
client_chroma = chromadb.PersistentClient(
    path=r"D:\Sanbercode\otomasi\final_project\chroma_db"
)
collection = client_chroma.get_or_create_collection(name="astro_paper")

# ========================
# 2️⃣ Streamlit UI
# ========================
st.set_page_config(page_title="RAG Astronomy Assistant", layout="wide")
st.title("RAG Astronomy Assistant")
st.write(
    "Ask a question about astronomy, astrophysics, or cosmology. "
    "The assistant answers based on scientific papers retrieved from NASA ADS."
)

query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if query.strip() == "":
        st.warning("Please enter a question before submitting.")
    else:
        with st.spinner("Processing your question..."):
            # Run RAG pipeline
            answer = generate_answer_with_rag(query, collection, n_results=5)

        st.success("Answer generated successfully!")
        st.subheader("Answer:")
        st.write(answer)
