from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ GOOGLE_API_KEY not found in .env file")
    print("Please add: GOOGLE_API_KEY=your-key-here")
    exit(1)

try:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",  # Changed: use "embedding-001" instead of "gemini-embedding-001"
        google_api_key=api_key
    )
    
    # Changed: Multiple texts instead of single text
    texts = [
        "What is the capital of France?",
        "Tell me about machine learning",
        "Explain LangChain framework"
    ]
    
    # Changed: embed_documents for multiple texts
    docs = embeddings.embed_documents(texts)
    
    # Changed: Using first document to show dimension
    result = embeddings.embed_query(texts[0])  # Query with first text
    print(f"Number of documents: {len(docs)}")
    print(f"Embedding dimension: {len(result)}")
    print(f"First document first 10 values: {docs}")
    
except Exception as e:
    print(f"Error: {e}")