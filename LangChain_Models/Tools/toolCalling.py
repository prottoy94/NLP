import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers and return their product."""
    return a * b


def main() -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY was not found in the environment.")

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-3.5-flash-lite",
        temperature=0,
        google_api_key=api_key,
    )
    llm_with_tools = llm.bind_tools([multiply])

    response = llm_with_tools.invoke(
        "Hi"
    )
    print(response)

    for tool_call in response.tool_calls:
        if tool_call["name"] == multiply.name:
            result = multiply.invoke(tool_call["args"])
            print(f"Tool result: {result}")


if __name__ == "__main__":
    main()