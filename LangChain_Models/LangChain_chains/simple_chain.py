#from itertools import chain
import os
from langchain_google_genai import ChatGoogleGenerativeAI   
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser



load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
prompt=PromptTemplate(
    template="Generate 5 interesting facts about {topic}.",
    input_variables=["topic"]
)

model = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",  
        temperature=0.9,
        max_output_tokens=2048,    
        google_api_key=api_key
    )

parser=StrOutputParser()

chain = prompt | model | parser
chain_result=chain.invoke({"topic":"space exploration"})
print(chain_result)

chain.get_graph().print_ascii()