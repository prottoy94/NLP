"""
rag_pipeline.py
---------------
All the LangChain / RAG logic lives here.
This is the SAME pipeline as using_chain.py, just split into small functions
so the Flask web app (app.py) can call each step when it needs it.

    Step 1: extract_video_id()    -> get the video ID out of a YouTube URL
    Step 2: fetch_transcript()    -> download the transcript text
    Step 3: build_retriever()     -> split + embed + FAISS + retriever
    Step 4: build_rag_chain()     -> RunnableParallel | prompt | llm | parser
    Step 5: generate_summary()    -> summary from the full transcript
"""

import os
import re
import json
import logging
import urllib.request
import urllib.parse

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    InvalidVideoId,
    RequestBlocked,
    CouldNotRetrieveTranscript,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# load_dotenv() searches the current folder and then the parent folders,
# so it finds the .env file in D:\NLP\LangChain_Models
load_dotenv()

logger = logging.getLogger(__name__)


# Our own error type. When something goes wrong we raise this with a
# short, friendly message, and app.py shows that message to the user.
class RAGError(Exception):
    pass


# ---------------------------------------------------------------------------
# Step 1: Extract and validate the video ID
# ---------------------------------------------------------------------------
# A YouTube video ID is always 11 characters: letters, digits, "-" or "_".
VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")

# This regex handles all of these:
#   https://www.youtube.com/watch?v=VIDEO_ID
#   https://youtu.be/VIDEO_ID
#   https://www.youtube.com/embed/VIDEO_ID
#   https://www.youtube.com/shorts/VIDEO_ID
#   (also m.youtube.com, music.youtube.com, no "https://", extra ?t=30 etc.)
YOUTUBE_URL_PATTERN = re.compile(
    r"^(?:https?://)?"  # optional http:// or https://
    r"(?:www\.|m\.|music\.)?"  # optional subdomain
    r"(?:"
    r"youtube\.com/watch\?(?:.*&)?v=(?P<id1>[A-Za-z0-9_-]{11})"  # watch?v=ID
    r"|youtube\.com/(?:embed|shorts|live|v)/(?P<id2>[A-Za-z0-9_-]{11})"  # /embed/ID, /shorts/ID
    r"|youtu\.be/(?P<id3>[A-Za-z0-9_-]{11})"  # youtu.be/ID
    r")"
    r"(?:[?&#/].*)?$"  # anything after the ID (?t=30, &list=...)
)


def extract_video_id(url):
    """Return the 11-character video ID from a YouTube URL, or raise RAGError."""
    url = (url or "").strip()
    if not url:
        raise RAGError("Please paste a YouTube video URL.")

    match = YOUTUBE_URL_PATTERN.match(url)
    if not match:
        raise RAGError(
            "That doesn't look like a valid YouTube video URL. "
            "Example: https://www.youtube.com/watch?v=VIDEO_ID"
        )

    # Only one of the three groups will have a value
    video_id = match.group("id1") or match.group("id2") or match.group("id3")

    # Double check (the regex already ensures this, but it makes the rule clear)
    if not VIDEO_ID_PATTERN.match(video_id):
        raise RAGError("The video ID in this URL is not valid.")

    return video_id


def get_video_title(video_id):
    """
    Get the video title using YouTube's public oEmbed endpoint (no API key needed).
    The title is only "nice to have", so if anything fails we just return None.
    """
    try:
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        oembed_url = (
            "https://www.youtube.com/oembed?format=json&url="
            + urllib.parse.quote(video_url)
        )
        with urllib.request.urlopen(oembed_url, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data.get("title")
    except Exception as error:
        logger.info("Could not fetch title for %s: %s", video_id, error)
        return None


# ---------------------------------------------------------------------------
# Step 2: Fetch the transcript
# ---------------------------------------------------------------------------
def fetch_transcript(video_id):
    """Download the English transcript and return it as one long string."""
    try:
        transcript_list = YouTubeTranscriptApi().fetch(
            video_id, languages=["en", "en-US", "en-GB"]
        )
        transcript_text = " ".join([entry.text for entry in transcript_list])

    # Each "except" turns a technical error into a friendly message.
    # Because we RAISE here, processing stops - we never continue with an
    # undefined transcript_text (this was the bug in the old main.py).
    except TranscriptsDisabled:
        raise RAGError("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        raise RAGError(
            "This video has no English transcript, so it can't be processed."
        )
    except VideoUnavailable:
        raise RAGError("This video is unavailable (it may be private or deleted).")
    except InvalidVideoId:
        raise RAGError("The video ID is not valid.")
    except RequestBlocked:
        raise RAGError(
            "YouTube is blocking transcript requests from your network right now. Try again later."
        )
    except CouldNotRetrieveTranscript as error:
        logger.warning("Transcript error for %s: %s", video_id, type(error).__name__)
        raise RAGError("Could not get the transcript for this video.")
    except Exception as error:
        # Network problems, timeouts, etc.
        logger.exception("Unexpected error while fetching transcript for %s", video_id)
        raise RAGError(
            "Network error while fetching the transcript. Check your internet connection."
        ) from error

    if not transcript_text.strip():
        raise RAGError("The transcript for this video is empty.")

    return transcript_text


# ---------------------------------------------------------------------------
# Helper: turn Google API errors into friendly messages
# ---------------------------------------------------------------------------
def friendly_google_error(error, action):
    """
    Look at the error text from Google and return a short message.
    `action` is a word like "creating embeddings" or "generating the answer".
    """
    text = str(error).lower()
    if (
        "api key" in text
        or "api_key" in text
        or "permission_denied" in text
        or "401" in text
        or "403" in text
    ):
        return (
            "Google API authentication failed. Check GOOGLE_API_KEY in your .env file."
        )
    if (
        "429" in text
        or "quota" in text
        or "resource_exhausted" in text
        or "rate" in text
    ):
        return "Google API quota or rate limit reached. Please wait a minute and try again."
    return f"Something went wrong with Google Gemini while {action}. Please try again."


def check_api_key():
    if not os.getenv("GOOGLE_API_KEY"):
        raise RAGError("GOOGLE_API_KEY is missing. Add it to your .env file.")


# ---------------------------------------------------------------------------
# Step 3: Split -> Embed -> FAISS -> Retriever (same settings as using_chain.py)
# ---------------------------------------------------------------------------
def build_retriever(transcript_text):
    """Create the FAISS vector store and return (retriever, number_of_chunks)."""
    check_api_key()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript_text])

    embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

    try:
        vectorstore = FAISS.from_documents(chunks, embedding=embedding_model)
    except Exception as error:
        logger.exception("Indexing (embeddings / FAISS) failed")
        raise RAGError(friendly_google_error(error, "creating embeddings")) from error

    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )
    return retriever, len(chunks)


# ---------------------------------------------------------------------------
# Step 4: The RAG chain (same design as using_chain.py)
# ---------------------------------------------------------------------------
def format_docs(docs):
    context_text = "\n\n".join([doc.page_content for doc in docs])
    return context_text


# The LLM can be created once and reused for every video.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

parser = StrOutputParser()

# Small change from using_chain.py: the rules are a bit stricter so the
# assistant clearly says when the transcript doesn't contain the answer.
prompt_template = PromptTemplate.from_template("""
You are a helpful assistant that answers questions about a YouTube video.
Answer ONLY from the provided transcript context below.
Do NOT use outside knowledge and do NOT invent anything the speaker did not say.
If the context does not contain enough information to answer, reply:
"I can't determine that from the available transcript context."

Context:
{context}

Question: {question}
""")


def build_rag_chain(retriever):
    """Build the chain for ONE video's retriever."""
    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )

    main_chain = parallel_chain | prompt_template | llm | parser
    return main_chain


def answer_question(rag_chain, question):
    """Run the RAG chain with the user's question (from the chat box)."""
    check_api_key()
    try:
        return rag_chain.invoke(question)
    except Exception as error:
        logger.exception("RAG chain failed")
        raise RAGError(friendly_google_error(error, "generating the answer")) from error


# ---------------------------------------------------------------------------
# Step 5: Summary
# ---------------------------------------------------------------------------
# For the summary we want the WHOLE video, not just 3 chunks, so we send the
# full transcript. (Chat questions still use the retriever.)
# Gemini 2.5 Flash has a very large context window, but we still put a limit
# on the size so a huge video can't create a massive request.
MAX_SUMMARY_CHARS = 400_000

summary_prompt = PromptTemplate.from_template("""
You are a helpful assistant. Summarize the YouTube video using ONLY the transcript below.
Do not add facts that are not in the transcript.

Write:
1. A short overview (2-3 sentences).
2. The main points as a bullet list.
3. A one-line conclusion.

Transcript:
{transcript}
""")

summary_chain = summary_prompt | llm | parser


def generate_summary(transcript_text):
    check_api_key()
    transcript_for_summary = transcript_text[:MAX_SUMMARY_CHARS]
    try:
        return summary_chain.invoke({"transcript": transcript_for_summary})
    except Exception as error:
        logger.exception("Summary generation failed")
        raise RAGError(
            friendly_google_error(error, "generating the summary")
        ) from error
