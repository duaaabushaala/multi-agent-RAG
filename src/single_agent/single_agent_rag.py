import os
from dotenv import load_dotenv
from google import genai
from src.retrieval.vector_store import search  # search lives in vector_store, not embeddings

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

    # return question, answer, and source chunks
    # chunks_used is needed later for RAGAS evaluation
    return {
        "question": question,
        "answer": response.text,
        "chunks_used": chunks,
    }


