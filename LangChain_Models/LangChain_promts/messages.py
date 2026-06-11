from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
model = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    google_api_key=api_key
)

messages=[
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?")
]
result=model.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages)