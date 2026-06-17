from langchain_community.document_loaders import  WebBaseLoader
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
    template="Answer the following questions \n {question} from the following text: \n {context}",
    input_variables=["question", "context"]
)
parser = StrOutputParser()

urls = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Deep_learning",
]

loader = WebBaseLoader(web_path=urls)
chain = prompt | model | parser

docs = loader.load()

max_chars = 4000  
truncated_content = docs[0].page_content[:max_chars]

print(len(docs))
print(truncated_content)

print(chain.invoke({
    "question": "What is Artificial Intelligence?give me a summary from the text.",
    "context": truncated_content
}))
