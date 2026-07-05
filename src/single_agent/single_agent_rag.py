import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from src.retrieval.vector_store import search

RESULTS_DIR = Path("experiments/single_agent_results")

load_dotenv()
GENERATION_MODEL = "gemini-2.5-flash"
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# function receives a question and a list of retrieved chunks, and returns one text prompt.
def build_prompt(question, chunks):
    evidence = ""
    for i, chunk in enumerate(chunks, start=1):
        evidence += f"Record {i}: {chunk['title']}\n"
        evidence += f"{chunk['chunk_text']}\n\n"

    return f"""You are an assistant helping researchers explore WWI personal archive records from Europeana.

Use only the evidence below to answer the question. 
If the evidence does not contain enough information, say so.

Evidence:
{evidence}
Question: {question}

Answer:"""


def save_result(result):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # one file per run, named by timestamp so nothing gets overwritten
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"result_{timestamp}.json"

    # strip full chunk dicts down to just what's needed for evaluation
    result_to_save = {
        "question": result["question"],
        "answer": result["answer"],
        "model": GENERATION_MODEL,
        "k": len(result["chunks_used"]),
        "source_chunk_ids": [c["chunk_id"] for c in result["chunks_used"]],
        "contexts": [c["chunk_text"] for c in result["chunks_used"]],
        "timestamp": timestamp,
    }

    output_path.write_text(
        json.dumps(result_to_save, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Result saved to {output_path}")


def answer(question: str, k: int = 5):
    # step 1: retrieve the top k most relevant chunks using FAISS
    chunks = search(question, k=k)

    # step 2: format chunks + question into a structured prompt
    prompt = build_prompt(question, chunks)

    # step 3: send prompt to Gemini and get a response
    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )

    result = {
        "question": question,
        "answer": response.text,
        "chunks_used": chunks,
    }

    save_result(result)
    return result


