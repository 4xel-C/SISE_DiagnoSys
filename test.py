import os

from dotenv import load_dotenv
from ecologits import EcoLogits
from mistralai import Mistral


load_dotenv()


EcoLogits.init(providers=["mistralai"], electricity_mix_zone="FRA")


client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

response = client.chat.complete(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ],
    model="mistral-small-latest",
)

result = response.impacts

print(response.usage)
