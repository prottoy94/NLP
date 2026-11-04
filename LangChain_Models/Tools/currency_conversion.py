import os
import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """Fetch the current conversion rate between two currencies."""
    api_key = os.getenv("EXCHANGE_RATE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "EXCHANGE_RATE_API_KEY was not found in the environment."
        )

    url = (
        f"https://v6.exchangerate-api.com/v6/{api_key}/pair/"
        f"{base_currency.upper()}/{target_currency.upper()}"
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()

    if data.get("result") != "success":
        raise RuntimeError(
            f"Currency conversion API request failed: {data.get('error-type', 'unknown error')}"
        )

    return float(data["conversion_rate"])


@tool
def convert(base_currency_value: float, conversion_rate: float) -> float:
    """Convert an amount using a currency conversion rate."""
    return base_currency_value * conversion_rate


def main() -> None:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY was not found in the environment.")

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=0,
        google_api_key=api_key,
    )
    llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

    query = HumanMessage(
        content="Convert 100 USD to INR."
    )
    messages = [query]
    print(query.content)
    response = llm_with_tools.invoke(messages)

    while response.tool_calls:
        messages.append(response)
        for tool_call in response.tool_calls:
            if tool_call["name"] == get_conversion_factor.name:
                tool_result = get_conversion_factor.invoke(tool_call)
                print(f"Conversion factor: {tool_result}")
                messages.append(tool_result)
            elif tool_call["name"] == convert.name:
                tool_result = convert.invoke(tool_call)
                print(f"Converted amount: {tool_result}")
                messages.append(tool_result)
            else:
                raise RuntimeError(
                    f"Unknown tool requested: {tool_call['name']}"
                )
        print(f"Invoking LLM with updated messages...: {messages}")
        response = llm_with_tools.invoke(messages)

    print(response.content)


if __name__ == "__main__":
    main()