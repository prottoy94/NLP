
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


def format_docs(docs):
    context_text = "\n\n".join([doc.page_content for doc in docs])
    return context_text


# Indexing
video_id = "DzpHeXVSC5I"

try:
    transcript_list = YouTubeTranscriptApi().fetch(
        video_id, languages=["en"]
    )
    transcript_text = " ".join([entry.text for entry in transcript_list])

except TranscriptsDisabled:
    print("Transcripts are disabled for this video.")
    raise SystemExit

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.create_documents([transcript_text])

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)

vectorstore = FAISS.from_documents(
    chunks, embedding=embedding_model
)

# Retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Prompt
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

prompt_template = PromptTemplate.from_template("""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

Context:
{context}

Question: {question}
""")

# Parallel chain
parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

# Complete RAG chain
parser = StrOutputParser()

main_chain = parallel_chain | prompt_template | llm | parser

# Question
question = "Was deep learning discussed in this video?"

# Run the chain
response = main_chain.invoke(question)

print("Response:\n", response)
