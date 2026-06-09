from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    print("❌ Please add HUGGINGFACEHUB_API_TOKEN to your .env file")
    print("Get one from: https://huggingface.co/settings/tokens")
    exit(1)

llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-large",
    temperature=0.7,
    huggingfacehub_api_token=hf_token,
    max_new_tokens=512
)

response = llm.invoke("What is the capital of France?")
print(response)