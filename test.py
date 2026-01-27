import os

from dotenv import load_dotenv
from mistralai import Mistral

from app.rag import patient_store

load_dotenv()


client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

response = client.chat.complete(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ],
    model="mistral-small",
)
