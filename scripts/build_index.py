"""
Run this once to embed all chunks and build the FAISS index.
Saves the index to data/indexes/ — do not need to run again unless chunks change.

Usage:
    python -m scripts.build_index
"""

from src.retrieval.vector_store import build

if __name__ == "__main__":
    print("Building FAISS index from shared chunks...")
    build()
    print("Done.")
