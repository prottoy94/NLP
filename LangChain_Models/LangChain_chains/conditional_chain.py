import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq   
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnableBranch
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, EmailStr, Field
from typing import Literal

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",  
        temperature=0.9,
        max_output_tokens=2048,    
        google_api_key=api_key
    )

model1 = ChatGroq(
    model="llama-3.1-8b-instant",  # Active model
    temperature=0.9,
    max_tokens=2048,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

parser=StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"]= Field(..., description="The overall sentiment of the feedback, either positive or negative.")
parser2 = PydanticOutputParser(pydantic_object=Feedback)

promt1=PromptTemplate(
    template ="Classify the sentiment of the following feedback text into a positive or negative sentiment. Return only the sentiment label without any additional text or explanation.\n {feedback} \n {format_instructions}",
    input_variables=["feedback"],
    partial_variables={"format_instructions": parser2.get_format_instructions()}
)

classifier_chain= promt1 | model | parser2

promt2=PromptTemplate(
    template="Generate a response to the following positive feedback: {feedback}",
    input_variables=["feedback"]
)
promt3=PromptTemplate(
    template="Generate a response to the following negative feedback: {feedback}",
    input_variables=["feedback"]
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", promt2 | model1 | parser), 
    (lambda x: x.sentiment == "negative", promt3 | model1 | parser), 
    RunnableLambda(lambda x: "could not find sentiment")
)

chain= classifier_chain | branch_chain
result=chain.invoke({"feedback":"The smartphone is terrible and doesn't work properly!"})

print(result)

chain.get_graph().print_ascii()