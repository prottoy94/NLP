from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    print("❌ GROQ_API_KEY not found in .env file")
    print("Get one from: https://console.groq.com/keys")
    exit(1)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    groq_api_key=groq_api_key
)

response = llm.invoke("What is the capital of France?")
print("Answer:", response.content)