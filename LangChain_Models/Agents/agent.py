import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

search_tool = DuckDuckGoSearchRun()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

agent = create_agent(
    model=llm,
    tools=[search_tool],
    system_prompt=(
        "You are a helpful research assistant. "
        "Use the search tool for current information."
    ),
)

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Give a detailed preview of today's IPL match "
                    "between CSK and PBKS."
                ),
            }
        ]
    }
)

print(response["messages"][-1].content)
