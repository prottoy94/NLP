from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

#genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# List all available models
#print("Available models for your API key:")
#print("-" * 40)

#for model in genai.list_models():
#    if 'generateContent' in model.supported_generation_methods:
#        print(f"✓ {model.name}")
#    else:
#        print(f"✗ {model.name} (no generateContent support)")
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ GOOGLE_API_KEY not found in .env file")
    print("Please add: GOOGLE_API_KEY=your-key-here")
    exit(1)

try:
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",  
        temperature=0.9,
        max_output_tokens=2048,    
        google_api_key=api_key
    )
    
    result = llm.invoke("What is the capital of France?")
    print(f"Answer: {result.content}")
    
except Exception as e:
    print(f"Error: {e}")