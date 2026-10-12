
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

#retriever
#print("Indexing completed. Number of chunks:", len(chunks))
retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k": 3}) # Return 2 most similar documents
#print(retriever)

#results = retriever.invoke("What is the main topic of the video?")
#context_text="\n\n".join([doc.page_content for doc in results])

#print(context_text)


# Prompt
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

prompt_template = PromptTemplate.from_template("""You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

{context}
Question: {question}""")

question = "is the topic of the Deep Learning was discussed in this video?"
retriver_docs=retriever.invoke(question)
context_text="\n\n".join([doc.page_content for doc in retriver_docs])
#print("Context:\n", context_text)
final_prompt = prompt_template.format(context=context_text, question=question)
response = llm.invoke(final_prompt)
print("Response:\n", response.content)