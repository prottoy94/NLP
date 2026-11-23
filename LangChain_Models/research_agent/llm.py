import os

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


class LLMError(Exception):
    pass


PROVIDERS = ["gemini", "groq", "openrouter"]
DEFAULT_PROVIDER = "gemini"

OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"


def check_api_key(key_name):
    if not os.getenv(key_name):
        raise LLMError(f"{key_name} is missing. Add it to your .env file.")


def get_llm(provider=DEFAULT_PROVIDER, temperature=0, max_tokens=None):
    provider = (provider or DEFAULT_PROVIDER).lower()

    if provider == "gemini":
        check_api_key("GOOGLE_API_KEY")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=temperature,
            max_tokens=max_tokens,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    elif provider == "groq":
        check_api_key("GROQ_API_KEY")
        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=temperature,
            max_tokens=max_tokens,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )
    elif provider == "openrouter":
        check_api_key("OPENROUTER_API_KEY")
        llm = ChatOpenAI(
            model=OPENROUTER_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
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
