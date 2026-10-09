from langchain_text_splitters import TextSplitter, CharacterTextSplitter, RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()


doc1 = Document(
    page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and longevity with Royal Challengers Bangalore.",
    metadata={"team": "Royal Challengers Bangalore"}
)

doc2 = Document(
    page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings.",
    metadata={"team": "Mumbai Indians"}
)

doc3 = Document(
    page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills and wicket-keeping are legendary.",
    metadata={"team": "Chennai Super Kings"}
)

doc4 = Document(
    page_content="David Warner is a prolific opening batsman in the IPL, known for his consistency and aggressive stroke play. He has captained Sunrisers Hyderabad to a title victory.",
    metadata={"team": "Delhi Capitals"}
)

doc5 = Document(
    page_content="AB de Villiers, also known as Mr. 360, is renowned for his innovative shot-making and ability to hit boundaries all around the wicket for Royal Challengers Bangalore.",
    metadata={"team": "Royal Challengers Bangalore"}
)

doc6 = Document(
    page_content="KL Rahul is a stylish opening batsman and a reliable wicket-keeper. He has been a consistent run-scorer for Punjab Kings and Lucknow Super Giants.",
    metadata={"team": "Lucknow Super Giants"}
)

doc7 = Document(
    page_content="Jasprit Bumrah is one of the best death-overs bowlers in IPL history. His unique bowling action and pinpoint yorkers make him a key asset for Mumbai Indians.",
    metadata={"team": "Mumbai Indians"}
)

doc8 = Document(
    page_content="Rashid Khan is a world-class leg-spinner known for his quick googlies and economical bowling. He has been a match-winner for Gujarat Titans and Sunrisers Hyderabad.",
    metadata={"team": "Gujarat Titans"}
)

doc9 = Document(
    page_content="Sanju Samson is a talented wicket-keeper batsman known for his elegant stroke play. He captains Rajasthan Royals and is known for hitting massive sixes.",
    metadata={"team": "Rajasthan Royals"}
)

doc10 = Document(
    page_content="Andre Russell is a powerful all-rounder from West Indies, known for his explosive batting and handy fast bowling. He is a key player for Kolkata Knight Riders.",
    metadata={"team": "Kolkata Knight Riders"}
)

doc11 = Document(
    page_content="Shikhar Dhawan is a veteran opening batsman known for his consistency and charisma. He has scored heavily for Delhi Capitals and Punjab Kings throughout his career.",
    metadata={"team": "Punjab Kings"}
)
doc12 = Document(
    page_content="Hardik Pandya is an all-rounder who has played for Mumbai Indians and contributed with both explosive batting and fast-medium bowling. He returned to lead Mumbai Indians as captain in 2024.",
    metadata={"team": "Mumbai Indians"}
)

doc13 = Document(
    page_content="Suryakumar Yadav is one of the most innovative T20 batters. Representing Mumbai Indians, he is known for his 360-degree stroke play and ability to score quickly in the middle overs.",
    metadata={"team": "Mumbai Indians"}
)

doc14 = Document(
    page_content="Ishan Kishan is a left-handed wicket-keeper batter who has represented Mumbai Indians in the IPL. He is known for aggressive starts and attacking stroke play in the powerplay.",
    metadata={"team": "Mumbai Indians"}
)

doc15 = Document(
    page_content="Kieron Pollard was a powerful all-rounder for Mumbai Indians. His finishing ability, big hitting, and useful medium-pace bowling helped the franchise win multiple IPL titles.",
    metadata={"team": "Mumbai Indians"}
)

doc16 = Document(
    page_content="Lasith Malinga was a legendary fast bowler for Mumbai Indians. Famous for his accurate yorkers and unusual bowling action, he played a major role in the team's IPL success.",
    metadata={"team": "Mumbai Indians"}
)

doc17 = Document(
    page_content="Tilak Varma is a left-handed middle-order batter who has represented Mumbai Indians. He is known for handling pressure, rotating the strike, and accelerating the scoring rate.",
    metadata={"team": "Mumbai Indians"}
)

doc18 = Document(
    page_content="Mumbai Indians won their first IPL title in 2013 under captain Rohit Sharma. The franchise subsequently won IPL championships in 2015, 2017, 2019, and 2020.",
    metadata={"team": "Mumbai Indians"}
)

doc19 = Document(
    page_content="Mumbai Indians is an IPL franchise owned by Indiawin Sports. The team plays its home matches at Wankhede Stadium in Mumbai and has historically been known for its strong batting and bowling lineups.",
    metadata={"team": "Mumbai Indians"}
)
vectorstore = Chroma(
    embedding_function=GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2"
    ),
    collection_name="ipl_players",
    persist_directory="vectorstore_data"
)

vectorstore.add_documents([doc1, doc2, doc3, doc4, doc5, doc6, doc7, doc8, doc9, doc10, doc11, doc12, doc13, doc14, doc15, doc16, doc17, doc18, doc19])

result = vectorstore.get(include=["embeddings", "documents", "metadatas"])

for i in range(len(result["ids"])):
    print(f"\nDocument {i + 1}")
    print("ID:", result["ids"][i])
    print("Embedding:", result["embeddings"][i])
    print("Document:", result["documents"][i])
    print("Metadata:", result["metadatas"][i])
    
result = vectorstore.similarity_search("Who is the best captain in IPL history?", k=3)
score = vectorstore.similarity_search_with_score("Who is the best captain in IPL history?", k=3)
print("\nTop 3 similar documents:")
for i, doc in enumerate(result):
    print(f"\nDocument {i + 1}")
    print("Content:", doc.page_content)
    print("Metadata:", doc.metadata)
    print("Score:", score[i][1])

filtered_result = vectorstore.similarity_search(
    "Mumbai Indians players",
    k=3,
    filter={"team": "Mumbai Indians"}
)
print("\nMatching documents for Mumbai Indians:")

for i, doc in enumerate(filtered_result, start=1):
    print(f"\nDocument {i}")
    print("Content:", doc.page_content)
    print("Metadata:", doc.metadata)