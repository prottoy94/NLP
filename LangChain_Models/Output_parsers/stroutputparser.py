from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from dotenv import load_dotenv
import os
import torch

load_dotenv()

# Get token from .env file
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Create pipeline with token for faster download
pipe = pipeline(
    "text-generation",
    model="google/flan-t5-large",
    max_new_tokens=512,
    device=-1,  # Use CPU for now (change to 0 if you have GPU)
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # Fixed deprecated parameter
    token=hf_token  # Add token for faster download
)

llm = HuggingFacePipeline(pipeline=pipe)
response = llm.invoke("What is the capital of France?")
print(response)