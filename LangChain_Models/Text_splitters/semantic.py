from langchain_text_splitters import TextSplitter, CharacterTextSplitter, RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

loader = PyPDFLoader('Text_splitters/dl-curriculum.pdf')
docs=loader.load()

text = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass.
The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.

Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""

splitter = SemanticChunker( GoogleGenerativeAIEmbeddings(model="gemini-embedding-2"), breakpoint_threshold_type='percentile',breakpoint_threshold_amount=71)
#splitter = TextSplitter(chunk_size=200, chunk_overlap=50)

result = splitter.create_documents([text])

print(result[0])
print(len(result))

