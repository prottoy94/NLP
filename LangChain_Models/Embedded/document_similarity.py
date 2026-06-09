from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "Who is the god of cricket?"
doc_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

print(cosine_similarity([query_embedding], doc_embeddings))
print(cosine_similarity([query_embedding], doc_embeddings)[0])

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

print(list(enumerate(scores)))
print(sorted(list(enumerate(scores))))
print(sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)) 


index, score = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[0]
print(query) 
print(documents[index]) 