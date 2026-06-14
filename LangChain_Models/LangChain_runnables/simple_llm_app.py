from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.9
)

prompt = PromptTemplate(
    input_variables=["product"],
    template="Suggest a good blog title about {product}."
)

chain = prompt | llm

topic = input("Enter a product: ")

output = chain.invoke({"product": topic})

print(f"Blog title: {output.content}")