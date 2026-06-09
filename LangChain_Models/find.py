import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ GOOGLE_API_KEY not found in .env file")
    print("Please add: GOOGLE_API_KEY=your-key-here")
    exit(1)

# Configure the API
genai.configure(api_key=api_key)

print("=" * 60)
print("🔍 AVAILABLE GOOGLE MODELS")
print("=" * 60)

# List all available models
print("\n📋 ALL MODELS:")
print("-" * 40)

for model in genai.list_models():
    print(f"\n📌 Model: {model.name}")
    print(f"   Display Name: {model.display_name}")
    print(f"   Supported Methods: {model.supported_generation_methods}")
    
    # Check if it supports embeddings
    if 'embedContent' in model.supported_generation_methods:
        print(f"   ✅ SUPPORTS EMBEDDINGS")
        print(f"   Input Token Limit: {model.input_token_limit}")
        print(f"   Output Token Limit: {model.output_token_limit}")
    else:
        print(f"   ❌ Does NOT support embeddings")

print("\n" + "=" * 60)
print("🔤 EMBEDDING MODELS ONLY:")
print("=" * 60)

# Filter for embedding models only
embedding_models = []
for model in genai.list_models():
    if 'embedContent' in model.supported_generation_methods:
        embedding_models.append(model)
        print(f"\n✅ {model.name}")
        print(f"   Display Name: {model.display_name}")
        print(f"   Input Token Limit: {model.input_token_limit}")

if not embedding_models:
    print("\n❌ No embedding models found for your API key")
else:
    print(f"\n📊 Total embedding models found: {len(embedding_models)}")
    print("\n💡 Recommended embedding model:")
    print("   models/embedding-001 (most stable and widely available)")