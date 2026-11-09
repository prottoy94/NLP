import os

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


class LLMError(Exception):
    pass


PROVIDERS = ["gemini", "groq"]
DEFAULT_PROVIDER = "gemini"


def check_api_key(key_name):
    if not os.getenv(key_name):
        raise LLMError(f"{key_name} is missing. Add it to your .env file.")


def get_llm(provider=DEFAULT_PROVIDER, temperature=0):
    provider = (provider or DEFAULT_PROVIDER).lower()

    if provider == "gemini":
        check_api_key("GOOGLE_API_KEY")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    elif provider == "groq":
        check_api_key("GROQ_API_KEY")
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=temperature,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
    else:
        raise LLMError(
            f"Unknown provider '{provider}'. Use one of: {', '.join(PROVIDERS)}"
        )

    return llm


def get_embeddings():
    check_api_key("GOOGLE_API_KEY")

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    return embedding_model
