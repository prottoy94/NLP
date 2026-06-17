from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.9
)

prompt = PromptTemplate(
    template="Write a summary for the following poem -{poem}.",
    input_variables=["poem"]
)

loader = TextLoader("Document_loaders/cricket.txt", encoding="utf-8")
docs = loader.load()
parser = StrOutputParser()

print(docs)
print(type(docs))
print(len(docs))
print(docs[0])

chain = prompt | model | parser
print(chain.invoke({"poem": docs[0].page_content}))