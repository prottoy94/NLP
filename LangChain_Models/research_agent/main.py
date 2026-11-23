import json
import logging
import threading
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

import store
from agent import build_agent, ask_agent
from llm import DEFAULT_PROVIDER, PROVIDERS
from tools import is_rate_limit, is_daily_limit, add_uploaded_paper

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

TEMPLATES_DIR = store.BASE_DIR / "templates"
STATIC_DIR = store.BASE_DIR / "static"

app = FastAPI(title="Research Paper Agent")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

cached_count = store.load_cache()
logger.info("Loaded %s papers from the disk cache", cached_count)


# ---------------------------------------------------------------------------
# Step 1: One agent per provider, all sharing the same memory
# ---------------------------------------------------------------------------
memory = InMemorySaver()
agents = {}
agents_lock = threading.Lock()


def get_agent(provider):
    with agents_lock:
        if provider not in agents:
            logger.info("Creating agent for provider %s", provider)
            agents[provider] = build_agent(provider, memory)

        agent = agents[provider]
    return agent


# ---------------------------------------------------------------------------
# Step 2: Request and response shapes
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    provider: str = DEFAULT_PROVIDER
    thread_id: str | None = None


class ChatResponse(BaseModel):
    thread_id: str
    provider: str
    answer: str
    tool_steps: list[str]


# ---------------------------------------------------------------------------
# Step 3: Helper functions
# ---------------------------------------------------------------------------
def check_chat_request(chat_request):
    message = chat_request.message.strip()
    provider = chat_request.provider.lower()
    thread_id = chat_request.thread_id

    if not message:
        raise HTTPException(status_code=400, detail="Please type a message.")

    if provider not in PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown provider '{provider}'. Use one of: {', '.join(PROVIDERS)}",
        )

    if not thread_id:
        thread_id = str(uuid.uuid4())

    return message, provider, thread_id


def count_messages(agent, thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    state = agent.get_state(config)
    messages = state.values.get("messages", [])
    return len(messages)


def get_tool_steps(messages):
    tool_steps = []

    for message in messages:
        if message.type == "ai" and message.tool_calls:
            for tool_call in message.tool_calls:
                args_text = str(tool_call["args"])
                if len(args_text) > 100:
                    args_text = args_text[:100] + "..."
                tool_steps.append(f"{tool_call['name']} {args_text}")

    return tool_steps


def friendly_agent_error(error):
    if is_daily_limit(error):
        message = "The free-tier daily limit of this provider is used up. Switch provider (for example groq) or try again tomorrow."
    elif "tool_use_failed" in str(error) or "tool call validation" in str(error).lower():
        message = "The model called a tool the wrong way (free models do this sometimes). Type 'continue' to let it try again."
    elif is_rate_limit(error):
        message = "The free-tier rate limit was reached. Wait a while or switch provider."
    else:
        message = f"Agent error: {str(error)[:300]}"
    return message


def make_event(event_type, data):
    event = {"type": event_type}
    event.update(data)
    return f"data: {json.dumps(event)}\n\n"


def stream_agent_events(agent, message, thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    user_message = {"messages": [{"role": "user", "content": message}]}

    yield make_event("start", {"thread_id": thread_id})

    try:
        for update in agent.stream(user_message, config=config, stream_mode="updates"):
            for node_output in update.values():
                if not node_output:
                    continue

                for new_message in node_output.get("messages", []):
                    if new_message.type == "ai" and new_message.tool_calls:
                        for tool_call in new_message.tool_calls:
                            yield make_event(
                                "tool_call",
                                {"name": tool_call["name"], "args": tool_call["args"]},
                            )

                    elif new_message.type == "tool":
                        yield make_event(
                            "tool_result",
                            {"name": new_message.name, "preview": new_message.text[:300]},
                        )

                    elif new_message.type == "ai":
                        yield make_event("final", {"answer": new_message.text})

    except Exception as error:
        logger.exception("Error while streaming the agent")
        yield make_event("error", {"message": friendly_agent_error(error)})

    yield make_event("done", {})


# ---------------------------------------------------------------------------
# Step 4: Routes
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(TEMPLATES_DIR / "index.html")


@app.get("/api/providers")
def list_providers():
    return {"providers": PROVIDERS, "default": DEFAULT_PROVIDER}


@app.post("/api/chat", response_model=ChatResponse)
def chat(chat_request: ChatRequest):
    message, provider, thread_id = check_chat_request(chat_request)
    agent = get_agent(provider)

    try:
        old_count = count_messages(agent, thread_id)
        messages = ask_agent(agent, message, thread_id)
    except Exception as error:
        logger.exception("Error in /api/chat")
        raise HTTPException(status_code=500, detail=friendly_agent_error(error))

    new_messages = messages[old_count:]

    return ChatResponse(
        thread_id=thread_id,
        provider=provider,
        answer=messages[-1].text,
        tool_steps=get_tool_steps(new_messages),
    )


@app.post("/api/chat/stream")
def chat_stream(chat_request: ChatRequest):
    message, provider, thread_id = check_chat_request(chat_request)
    agent = get_agent(provider)

    events = stream_agent_events(agent, message, thread_id)
    return StreamingResponse(events, media_type="text/event-stream")


@app.get("/api/papers")
def list_papers():
    papers = []

    for paper_id, entry in list(store.papers.items()):
        meta = entry["meta"]
        if meta is None:
            continue

        papers.append(
            {
                "paper_id": meta["paper_id"],
                "title": meta["title"],
                "published": meta["published"],
                "uploaded": store.is_upload(paper_id),
                "loaded": store.pdf_path(paper_id).exists(),
                "summarized": entry["summary"] is not None,
            }
        )

    return {"papers": papers}


@app.post("/api/upload")
def upload_paper(file: UploadFile):
    file_bytes = file.file.read()

    try:
        entry = add_uploaded_paper(file.filename, file_bytes)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        logger.exception("Error in /api/upload")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(error)[:300]}")

    meta = entry["meta"]
    return {
        "paper_id": meta["paper_id"],
        "title": meta["title"],
        "pages": entry["pages"],
        "characters": len(entry["text"]),
    }


@app.get("/api/notes")
def list_notes():
    notes = []

    for note_path in sorted(store.NOTES_DIR.glob("*.md"), reverse=True):
        notes.append(
            {
                "name": note_path.name,
                "size_kb": round(note_path.stat().st_size / 1024, 1),
            }
        )

    return {"notes": notes}


@app.get("/api/notes/{note_name}")
def download_note(note_name: str):
    note_path = (store.NOTES_DIR / note_name).resolve()

    if note_path.parent != store.NOTES_DIR.resolve() or note_path.suffix != ".md":
        raise HTTPException(status_code=400, detail="Invalid note name.")

    if not note_path.exists():
        raise HTTPException(status_code=404, detail="Note not found.")

    return FileResponse(note_path, media_type="text/markdown", filename=note_path.name)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
