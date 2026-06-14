from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv()

def count_words(text):
    return len(text.split())

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
joke_gen_chain= RunnableSequence(prompt1, model, parser)
parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "word_count": RunnableLambda(count_words)
})
final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

print(final_chain.invoke({"topic": "Football"}))
