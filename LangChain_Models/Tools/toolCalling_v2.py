import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
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
        model="models/gemini-2.5-flash",
        temperature=0,
        google_api_key=api_key,
    )
    llm_with_tools = llm.bind_tools([multiply])

    query = HumanMessage(content="Can you multiply 3 with 10?")
    messages = [query]
    print(query.content)
    response = llm_with_tools.invoke(messages)
    print(response)
    messages.append(response)

    for tool_call in response.tool_calls:
        if tool_call["name"] == multiply.name:
            tool_result = multiply.invoke(tool_call)
            messages.append(tool_result)

    final_response = llm_with_tools.invoke(messages)
    print(final_response)


if __name__ == "__main__":
    main()