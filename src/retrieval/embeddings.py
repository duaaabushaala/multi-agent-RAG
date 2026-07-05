import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def embed_text(text: str) -> list[float]:
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )
    return response.embeddings[0].values


def embed_query(text: str) -> list[float]:
    return embed_text(text)
