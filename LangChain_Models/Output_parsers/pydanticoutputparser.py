from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, EmailStr, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
import os
import json  

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    model="llama-3.1-8b-instant",  # Active model
    temperature=0.9,
    max_tokens=2048,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

#schema
class Review(BaseModel):
    key_themes: list[str]= Field(..., description="A list of the main themes or topics discussed in the review in a list")
    summary: str= Field(..., description="A brief summary of the review, capturing the main points in one or two sentences.")
    sentiment: Literal["positive", "negative"]= Field(..., description="The overall sentiment of the review, either positive, negative, or neutral.")
    pros: Optional[list[str]]= Field(None, description="A list of the positive aspects mentioned in the review, if any. If there are no pros, this should be an empty list or null.")
    cons: Optional[list[str]]= Field(None, description="A list of the negative aspects mentioned in the review, if any. If there are no cons, this should be an empty list or null.")
    reviewer_name: Optional[str]= Field(None, description="Write the name of the reviewer who wrote the review. If the name is not mentioned in the review, this should be null.")

class Person(BaseModel):
    name: str= Field(..., description="The name of the person")
    age: int= Field(..., gt=18, description="The age of the person")
    city: str= Field(..., description="The city where the person lives")
    
parser = PydanticOutputParser(pydantic_object=Person)

# FIX 1: Changed the template to be more direct
template3 = PromptTemplate(
    template="Generate a fictional {place} person. Return ONLY valid JSON with name, age, and city. Do not include any code, explanations, or markdown.\n{format_instructions}",
    input_variables=["place"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

prompt = template3.invoke({"place": "Bangladeshi"})

# FIX 2: Lower the temperature for more consistent output
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,  # Changed from 0.9 to 0.3
    max_tokens=512,   # Reduced from 2048
    groq_api_key=os.getenv("GROQ_API_KEY")
)

model_response = model.invoke(prompt)
final_result = parser.parse(model_response.content)
print(final_result)