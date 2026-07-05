import json
import time
from pathlib import Path

import faiss
import numpy as np

from src.retrieval.embeddings import embed_query, embed_text

CHUNKS_PATH = Path("data/processed/shared_corpus/shared_chunks_v1.jsonl") # points to final shared copus (219 records, 298 chunks)
INDEX_PATH = Path("data/indexes/faiss.index")        # the FAISS vector index is saved here
METADATA_PATH = Path("data/indexes/chunks_metadata.json")  # saves the original chunk dictionaries.
#It is the lookup table that lets you turn a FAISS result back into a readable Europeana chunk.


def build():
    # load all 298 chunks from disk
    chunks = []
    with CHUNKS_PATH.open(encoding="utf-8") as f: # opens the JSONL file containing the chunks
        for line in f:
            if line.strip():
                chunks.append(json.loads(line)) # takes each line ands turn it into a dictionary, appends to chunk

    # --- Step 2: embed each chunk using Gemini ---
    # sends each chunk's embedding_text to Gemini, gets back a 768-number vector
    vectors = [] # this will hold numerical values of the chunks
    for i, chunk in enumerate(chunks): # loops through every chunk while also keeping track of its position.
        vector = embed_text(chunk["embedding_text"]) # takes the text of the chunk, sends it to gemini and gets a vector reprenstation of it.
        vectors.append(vector) # appends the vector to the list of vectors
        print(f"  Embedded {i + 1}/{len(chunks)}") # prints progress
        time.sleep(1)  # pause between requests to avoid hitting the rate limit

    # --- Step 3: normalise and build the FAISS index ---
    # normalise so all vectors have length 1 — makes similarity about meaning, not magnitude
    matrix = np.array(vectors, dtype=np.float32) # Before this line, vectors is a Python list of 298 separate lists. After this line, it becomes a NumPy matrix.
    faiss.normalize_L2(matrix) # nromalise so each vector has length 1 (imortant for cosine similarity)

    # inner product (IP) = cosine similarity (tells you how similaur two vectors are).
    # Flat = compare against every stored vector exactly
    # index = searcghable index store
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix) # puts all 298 vectores into the index so we can search them later. FAISS will assign them positions so we can look up the original chunk later.

    # --- Step 4: save index and metadata to disk ---
    # FAISS stores vectors only — we save the chunk text separately so we can return it
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True) # make sure dir exists, if not creates it 
    faiss.write_index(index, str(INDEX_PATH)) # saves the 298 vectors and the index structure.
    METADATA_PATH.write_text(
        json.dumps(chunks, ensure_ascii=False), encoding="utf-8"
    ) # saves the complete list of original chunks as one JSON file. 
    print(f"Index built: {len(chunks)} chunks saved.")


def search(query: str, k: int = 5): # return the five most semantically similar chunks (not final decision yet for diss)
    # load the saved index and chunk metadata from disk
    index = faiss.read_index(str(INDEX_PATH)) # reopens FAISS index from disk 
    chunks = json.loads(METADATA_PATH.read_text(encoding="utf-8")) # reopens the original chunk metadata from disk

    # so the function now has FAISS index → vectors and similarity search
    # chunks list → the original readable historical evidence

    # embed the query the same way as the documents, then normalise
    query_vector = np.array([embed_query(query)], dtype=np.float32) # embed question
    faiss.normalize_L2(query_vector) # normalise query to length 1 

    # Compare this question vector with all 298 chunk vectors and return the top k closest ones.
    _, indices = index.search(query_vector, k) # ignore scores for now

    # map FAISS integer indices back to the actual chunk dicts
    # -1 means FAISS found fewer than k results, so we skip those
    return [chunks[i] for i in indices[0] if i != -1]
