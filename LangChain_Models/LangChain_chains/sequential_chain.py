import os
from langchain_google_genai import ChatGoogleGenerativeAI   
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",  
        temperature=0.9,
        max_output_tokens=2048,    
        google_api_key=api_key
    )


prompt1=PromptTemplate(
    template="Generate a detailed report on {topic}.",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="Generate a 5 pointer summary from the following text \n {text}",
    input_variables=["text"]
)

parser=StrOutputParser()

chain= prompt1 | model | parser | prompt2 | model | parser

chain_result=chain.invoke({"topic":"Unemployment in the Bangladesh"})

print(chain_result)

chain.get_graph().print_ascii()