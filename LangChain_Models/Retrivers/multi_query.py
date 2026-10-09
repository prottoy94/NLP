from langchain_text_splitters import TextSplitter, CharacterTextSplitter, RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv
import os

load_dotenv()

from langchain_core.documents import Document

all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),

    Document(page_content="Omega-3 fatty acids found in fish oil are essential for brain health and reducing inflammation.", metadata={"source": "H6"}),
    Document(page_content="High-intensity interval training improves cardiovascular endurance more efficiently than steady-state cardio.", metadata={"source": "H7"}),
    Document(page_content="Maintaining a consistent sleep schedule regulates the body's internal circadian rhythm.", metadata={"source": "H8"}),
    Document(page_content="Vitamin D synthesis occurs when the skin is exposed to ultraviolet B rays from sunlight.", metadata={"source": "H9"}),
    Document(page_content="Practicing gratitude journaling is linked to increased overall life satisfaction and reduced stress.", metadata={"source": "H10"}),
    Document(page_content="CRISPR-Cas9 is a revolutionary gene-editing technology that allows for precise modifications to DNA.", metadata={"source": "I6"}),
    Document(page_content="Quantum computing leverages superposition and entanglement to process information in ways classical computers cannot.", metadata={"source": "I7"}),
    Document(page_content="The Industrial Revolution marked a major turning point in history, shifting economies from agriculture to manufacturing.", metadata={"source": "I8"}),
    Document(page_content="Machine learning models require large datasets to accurately identify patterns and make predictions.", metadata={"source": "I9"}),
    Document(page_content="Blockchain technology provides a decentralized and immutable ledger for recording transactions.", metadata={"source": "I10"}),
    Document(page_content="The Great Barrier Reef is the world's largest coral reef system, supporting a vast diversity of marine life.", metadata={"source": "I11"}),
    Document(page_content="The Internet of Things connects everyday physical objects to the network, enabling remote monitoring and control.", metadata={"source": "I12"}),
    Document(page_content="Mount Everest's height continues to increase slightly each year due to tectonic plate movement.", metadata={"source": "I13"}),
    Document(page_content="The human brain contains approximately 86 billion neurons, forming complex networks for processing information.", metadata={"source": "H11"}),
    Document(page_content="Renewable energy sources like wind and solar are critical for reducing global carbon emissions.", metadata={"source": "I14"}),
    Document(page_content="A Mediterranean diet rich in olive oil, nuts, and fish is associated with a lower risk of heart disease.", metadata={"source": "H12"}),
    Document(page_content="Augmented reality overlays digital information onto the real world, enhancing user perception and interaction.", metadata={"source": "I15"}),
    Document(page_content="Stretching before exercise can improve flexibility and reduce the risk of muscle injuries.", metadata={"source": "H13"}),
    Document(page_content="The Turing Test, proposed by Alan Turing, evaluates a machine's ability to exhibit intelligent behavior equivalent to a human.", metadata={"source": "I16"}),
    Document(page_content="Ocean acidification, caused by absorbing excess atmospheric carbon dioxide, threatens calcifying marine organisms.", metadata={"source": "I17"})
]
embedding_model =embedding_function=GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2"
    )


vectorstore = FAISS.from_documents(
    documents=all_docs,
    embedding=embedding_model,
#collection_name="my_collection"
)

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
)
similarity_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2}) # Return 2 most similar documents
query = "How to balance energy and resources?"
results = multiquery_retriever.invoke(query)
results1 = similarity_retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"Content:\n{doc.page_content}...")
    
for i, doc in enumerate(results1):
    print(f"\n--- Similarity Result {i+1} ---")
    print(f"Content:\n{doc.page_content}...")