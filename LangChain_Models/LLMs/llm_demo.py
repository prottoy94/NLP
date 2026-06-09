from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Use the full absolute path
env_path = r"D:\NLP\LangChain_Models\.env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print(f"ERROR: API key not found in {env_path}")
else:
    print("✓ API key loaded!")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9)
result = llm.invoke("What is the capital of France?")
print(result.content)