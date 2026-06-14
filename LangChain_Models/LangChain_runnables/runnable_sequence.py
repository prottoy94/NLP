from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence 

load_dotenv()

prompt1 = PromptTemplate(
    template="Write a joke about {topic}.",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Explain the following joke {text}.",
    input_variables=["topic"]
)

model = ChatGroq( 
    model="llama-3.3-70b-versatile",
    temperature=0.9
)
parser = StrOutputParser()
chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)
#The pipe operator | is just syntactic sugar for RunnableSequence. Behind the scenes, prompt1 | model | parser creates a RunnableSequence object.

print(chain.invoke({"topic": "Chicken Dinner"}))
