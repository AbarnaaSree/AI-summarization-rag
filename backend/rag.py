from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chunks_store = []
index = None


def chunk_text(text, size=500):
    return [text[i:i+size] for i in range(0, len(text), size)]


def build_index(text):
    global chunks_store, index

    chunks_store = chunk_text(text)

    embeddings = embedding_model.encode(chunks_store)

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))


def retrieve(query, top_k=3):
    global index, chunks_store

    query_embedding = embedding_model.encode([query])

    distances, indices = index.search(np.array(query_embedding), top_k)

    return [chunks_store[i] for i in indices[0] if i < len(chunks_store)]