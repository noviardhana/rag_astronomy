# rag_utils.py
"""
RAG Utilities for Astronomy Paper Assistant

Contains:
- retrieve_and_augment_prompt: retrieve relevant abstracts and build prompt
- generate_answer_with_rag: full RAG pipeline including LLM query and sources
"""

from sentence_transformers import SentenceTransformer
from openai import OpenAI
import chromadb

# =========================
# Global Model & Client
# =========================
# Embedding model (all-MiniLM-L6-v2)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# OpenRouter / OpenAI client
open_ai_token = "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"
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=open_ai_token,
)

# =========================
# RAG Functions
# =========================
def retrieve_and_augment_prompt(query_text, collection, n_results=5):
    """
    Retrieve relevant astronomy paper abstracts from a ChromaDB collection,
    then construct a context-enriched prompt for an astronomy-focused RAG system.

    Args:
        query_text (str): The user's natural language question.
        collection (ChromaDB Collection): Vector DB collection containing paper abstracts.
        n_results (int): Number of most relevant chunks to retrieve.

    Returns:
        tuple:
            - str: Constructed, context-enriched prompt ready for LLM.
            - list: List of tuples (bibcode, title, pub) for retrieved papers.
    """
    print(f"\nStep 1: Received user query -> '{query_text}'")

    # Embed the user query
    query_embedding = model.encode([query_text]).tolist()

    # Query ChromaDB for relevant abstracts
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )

    retrieved_chunks = results["documents"][0]
    metadata = results["metadatas"][0]
    print(f"Step 2: Retrieved {len(retrieved_chunks)} relevant abstracts.")

    # Build scientific context string
    context_string = ""
    paper_info = []

    for i, (doc, meta) in enumerate(zip(retrieved_chunks, metadata), 1):
        bibcode = meta.get("bibcode", "N/A")
        title = meta.get("title", "Unknown Title")
        pub = meta.get("pub", "Unknown Journal")
        paper_info.append((bibcode, title, pub))
        context_string += (
            f"Paper {i}:\n"
            f"Title: {title}\n"
            f"Year: {meta.get('year', 'Unknown')}\n"
            f"Source: {bibcode}\n"
            f"Abstract: {doc}\n\n---\n\n"
        )

    # Construct the LLM prompt
    prompt_template = f"""
You are AstroRAG — an advanced AI specializing in astronomy, astrophysics, and cosmology.
Answer the user's question accurately and scientifically, based *only* on the retrieved paper abstracts.
If the answer is not in the context, respond:
"I could not find information related to that in the available paper abstracts."

Guidelines:
- Use clear, concise scientific explanations.
- Explain key concepts when relevant (e.g., dark matter, exoplanets, CMB, gravitational waves).
- Avoid speculation.
- Structure responses logically: define → explain → conclude.

Relevant Scientific Context:
{context_string}

---

User Question:
{query_text}

Your Answer:
    """

    print("Step 3: Context-enriched prompt successfully generated.")
    return prompt_template.strip(), paper_info


def generate_answer_with_rag(query_text, collection, n_results=5, model_name="openai/gpt-4o"):
    """
    Full RAG pipeline: retrieve -> augment -> generate answer with sources.

    Args:
        query_text (str): User question.
        collection (ChromaDB Collection): Collection of vectorized abstracts.
        n_results (int): Number of top documents to retrieve.
        model_name (str): LLM model name on OpenRouter.

    Returns:
        str: LLM answer with appended list of retrieved papers.
    """
    # Retrieve prompt and paper metadata
    prompt, paper_info = retrieve_and_augment_prompt(query_text, collection, n_results=n_results)

    # Send prompt to LLM
    print("\nStep 4: Sending augmented prompt to LLM...\n")
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a scientific AI specializing in astronomy, astrophysics, and cosmology. "
                    "Maintain precise, academic tone and use only provided context to answer."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        extra_headers={
            "HTTP-Referer": "https://astroresearchhub.org/",
            "X-Title": "astro_paper_RAG_expert",
        },
    )

    # Extract LLM answer
    answer = completion.choices[0].message.content.strip()
    print("Step 5: LLM response received.\n")

    # Append retrieved paper titles at the end
    sources_text = "\n".join([f"Paper {i}: {bibcode} - {title} - {pub}"
                              for i, (bibcode, title, pub) in enumerate(paper_info, 1)])

    final_output = f"{answer}\n\n=== SOURCES USED ===\n{sources_text}"
    return final_output
