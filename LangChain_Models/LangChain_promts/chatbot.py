from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

chat_history = [
    SystemMessage(content="You are a helpful and Professional assistant.")
]

while True:
    user_input = input("You: ")
    
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chatbot. Goodbye!")
        break
    
    chat_history.append(HumanMessage(content=user_input))
    
    result = model.invoke(chat_history)
    answer = result.content
    
    chat_history.append(AIMessage(content=answer))
    
    print(f"Chatbot: {answer}")

print("\nChat History:")
for message in chat_history:
    print(f"{message.type}: {message.content}")