from langchain_text_splitters import TextSplitter, CharacterTextSplitter, RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()

# Step 1: Your source documents
documents = [
    # --- Existing 4 ---
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
    
    # --- 20 New Documents ---
    Document(page_content="Prompt templates allow developers to dynamically format inputs for LLMs."),
    Document(page_content="Vector stores enable semantic search by comparing mathematical representations of text."),
    Document(page_content="Retrievers are interfaces that return relevant documents based on an unstructured query."),
    Document(page_content="Document loaders facilitate extracting data from various sources like PDFs and websites."),
    Document(page_content="Text splitters break large documents into smaller chunks to fit within LLM context windows."),
    Document(page_content="LLM chains allow multiple components to be linked together to solve complex tasks."),
    Document(page_content="Memory modules in LangChain help LLMs remember previous interactions in a conversation."),
    Document(page_content="FAISS is a highly efficient library for similarity search and clustering of dense vectors."),
    Document(page_content="Pinecone is a fully managed cloud-native vector database designed for high scalability."),
    Document(page_content="Hugging Face provides open-source embedding models that can be run locally."),
    Document(page_content="Semantic search focuses on the meaning of a query rather than just keyword matching."),
    Document(page_content="RAG combines retrieval systems with generative models to produce accurate, context-aware answers."),
    Document(page_content="Cosine similarity is a common metric used to measure the distance between two vectors."),
    Document(page_content="Metadata filtering allows users to narrow down vector search results based on specific attributes."),
    Document(page_content="LangChain agents use LLMs to determine which actions to take and in what order."),
    Document(page_content="Output parsers structure the raw text responses from LLMs into usable formats like JSON."),
    Document(page_content="Tokenization is the process of breaking down text into smaller units called tokens."),
    Document(page_content="Context windows represent the maximum amount of text an LLM can process at one time."),
    Document(page_content="Fine-tuning involves further training a pre-trained LLM on a specific, specialized dataset."),
    Document(page_content="Chunk overlap is a technique used during text splitting to preserve context between adjacent chunks.")
]

# Step 2: Initialize embedding model
embedding_model =embedding_function=GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2"
    )


# Step 3: Create FAISS vector store in memory
vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=embedding_model,
collection_name="my_collection"
)

# Step 4: Convert vectorstore into a retriever
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 2, "lambda_mult": 0.7}) # Return 2 most similar documents
query = "What is LangChain is used for?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"Content:\n{doc.page_content}...")