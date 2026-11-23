"""Split policy documents into sections and store them in a ChromaDB vector database.

Run:  python -m src.ingest
"""
import re

import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config

COLLECTION = "policies"


def load_sections():
    """Return one chunk per '## ' section of every policy file."""
    chunks = []
    for path in sorted(config.POLICY_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip()
        parts = re.split(r"^## ", text, flags=re.MULTILINE)[1:]  # skip the title block
        for part in parts:
            heading, _, body = part.partition("\n")
            chunks.append({
                "id": f"{path.stem}::{heading.strip()}",
                "policy": title,
                "file": path.stem,
                "section": heading.strip(),
                # Put the policy name in the text so a question naming the product finds it
                "text": f"{title} - {heading.strip()}\n{body.strip()}",
            })
    return chunks


class TfidfEmbedding:
    """Offline embedding function for ChromaDB (no model download)."""

    def __init__(self, corpus):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.vectorizer.fit(corpus)

    def __call__(self, input):
        return self.vectorizer.transform(input).toarray().tolist()

    def name(self):  # ChromaDB asks embedding functions for a name
        return "tfidf"

    def embed_query(self, input):
        return self(input)


def embedding_function(chunks):
    if config.EMBEDDING_BACKEND == "tfidf":
        return TfidfEmbedding([c["text"] for c in chunks])
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    return SentenceTransformerEmbeddingFunction(model_name=config.MINILM_MODEL)


def build_collection(persist=True):
    """Embed every section and (re)create the collection. Returns (collection, chunks)."""
    chunks = load_sections()
    ef = embedding_function(chunks)
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR)) if persist else chromadb.EphemeralClient()
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(
        COLLECTION, embedding_function=ef, metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{k: c[k] for k in ("policy", "file", "section")} for c in chunks],
    )
    return collection, chunks


if __name__ == "__main__":
    collection, chunks = build_collection()
    print(f"Indexed {len(chunks)} sections from {len({c['file'] for c in chunks})} policies "
          f"with '{config.EMBEDDING_BACKEND}' embeddings.")
