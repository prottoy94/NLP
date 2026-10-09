from langchain_text_splitters import TextSplitter, CharacterTextSplitter, RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever
)
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from dotenv import load_dotenv
import os

load_dotenv()

from langchain_core.documents import Document

from langchain_core.documents import Document

all_docs = [
    # --- Extracted from the image ---
    Document(page_content="""
    The Grand Canyon is one of the most visited natural wonders in the world.
    Photosynthesis is the process by which green plants convert sunlight into energy.
    Millions of tourists travel to see it every year. The rocks date back millions of years.""", 
    metadata={"source": "Doc1"}),

    Document(page_content="""
    In medieval Europe, castles were built primarily for defense.
    The chlorophyll in plant cells captures sunlight during photosynthesis.
    Knights wore armor made of metal. Siege weapons were often used to breach castle walls.""", 
    metadata={"source": "Doc2"}),

    Document(page_content="""
    Basketball was invented by Dr. James Naismith in the late 19th century.
    It was originally played with a soccer ball and peach baskets. NBA is now a global league.""", 
    metadata={"source": "Doc3"}),

    Document(page_content="""
    The history of cinema began in the late 1800s. Silent films were the earliest form.
    Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.""", 
    metadata={"source": "Doc4"}),

    # --- 10 Newly Generated Documents ---
    Document(page_content="""
    The Amazon Rainforest produces about 20% of the world's oxygen.
    Volcanoes are openings in the Earth's crust that allow lava and ash to escape.
    It is home to millions of species of plants and animals. Deforestation is a major threat to its survival.""", 
    metadata={"source": "Doc5"}),

    Document(page_content="""
    The invention of the printing press revolutionized the spread of information.
    Jupiter is the largest planet in our solar system and has a Great Red Spot.
    Johannes Gutenberg is credited with creating the first movable type printing press in Europe.""", 
    metadata={"source": "Doc6"}),

    Document(page_content="""
    The human heart pumps blood through a complex network of arteries and veins.
    The Sahara Desert is the largest hot desert in the world, covering much of North Africa.
    A healthy diet and regular exercise are crucial for maintaining cardiovascular health.""", 
    metadata={"source": "Doc7"}),

    Document(page_content="""
    The Great Wall of China was built over many centuries by various dynasties.
    Quantum mechanics describes the behavior of matter and energy at atomic scales.
    It stretches thousands of miles and was primarily constructed for defense against invasions.""", 
    metadata={"source": "Doc8"}),

    Document(page_content="""
    Artificial Intelligence is transforming industries like healthcare and finance.
    The Pacific Ocean is the largest and deepest of Earth's oceanic divisions.
    Machine learning models require massive amounts of data to improve their accuracy.""", 
    metadata={"source": "Doc9"}),

    Document(page_content="""
    The French Revolution began in 1789 and drastically changed the political landscape of Europe.
    Photosynthesis requires carbon dioxide, water, and sunlight to produce glucose.
    The storming of the Bastille is considered a pivotal moment in the revolution.""", 
    metadata={"source": "Doc10"}),

    Document(page_content="""
    The Internet was originally developed as a military communication project called ARPANET.
    Sharks have been on Earth longer than trees and predate the dinosaurs.
    Tim Berners-Lee invented the World Wide Web in 1989, changing how we share information.""", 
    metadata={"source": "Doc11"}),

    Document(page_content="""
    Vincent van Gogh painted over 2,000 artworks during his lifetime, though he only sold a few.
    The Earth's magnetic field protects the planet from harmful solar radiation.
    His unique style and use of color heavily influenced modern art movements.""", 
    metadata={"source": "Doc12"}),

    Document(page_content="""
    The theory of relativity was proposed by Albert Einstein in the early 20th century.
    Honey never spoils because it has a very low water content and high acidity.
    It fundamentally changed our understanding of space, time, and gravity.""", 
    metadata={"source": "Doc13"}),

    Document(page_content="""
    Mount Everest is the highest mountain above sea level, located in the Himalayas.
    The invention of the telephone is credited to Alexander Graham Bell in 1876.
    Climbing Everest requires extensive physical training and acclimatization to high altitudes.""", 
    metadata={"source": "Doc14"})
]

embedding_model =embedding_function=GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2"
    )


vectorstore = FAISS.from_documents(
    documents=all_docs,
    embedding=embedding_model,
#collection_name="my_collection"
)

base_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)
compressor = LLMChainExtractor.from_llm(llm) # Create a compressor using the LLMChainExtractor with the specified LLM

compressor_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=base_retriever
)

similarity_retriever = vectorstore.as_retriever(search_type="similarity", 
                                                search_kwargs={"k": 2}
                                                ) # Return 2 most similar documents
query = "What is mount everest?"
results = compressor_retriever.invoke(query)
results1 = similarity_retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"Content:\n{doc.page_content}...")
    
for i, doc in enumerate(results1):
    print(f"\n--- Similarity Result {i+1} ---")
    print(f"Content:\n{doc.page_content}...")