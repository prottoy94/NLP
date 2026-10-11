
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

#indexing
video_id= "DzpHeXVSC5I"
try:
    transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
    transcript_text = " ".join([entry.text for entry in transcript_list])
    #print(transcript_text)
except TranscriptsDisabled:
    print("Transcripts are disabled for this video.")
    
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript_text])

embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
vectorstore = FAISS.from_documents(chunks, embedding=embedding_model)

#print("Indexing completed. Number of chunks:", len(chunks))
retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k": 4}) # Return 2 most similar documents
print(retriever)