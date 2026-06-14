from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch

load_dotenv()

def count_words(text):
    return len(text.split())

prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}.",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Summarize the following report \n {text}.",
    input_variables=["topic"]
)

model = ChatGroq( 
    model="llama-3.3-70b-versatile",
    temperature=0.9
)
parser = StrOutputParser()
report_gen_chain= RunnableSequence(prompt1, model, parser)
branch_chain = RunnableBranch(
    (lambda x: len(x.split())>300, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)
final_chain = RunnableSequence(report_gen_chain, branch_chain)

print(final_chain.invoke({"topic": "Football World Cup"}))
