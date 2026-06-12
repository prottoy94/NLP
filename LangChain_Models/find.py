from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("=" * 60)
print("CURRENTLY ACTIVE GROQ MODELS")
print("=" * 60)

models = client.models.list()
for model in models.data:
    print(f"✅ {model.id}")