import json
import logging
import os
import re
import sqlite3
import threading
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.sqlite import SqliteSaver
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
DB_PATH = store.BASE_DIR / "chat_history.db"

TITLE_MAX_CHARS = 50
PAPER_LINE_PATTERN = re.compile(r"^\[Selected paper: [^\]]*\]\n")

app = FastAPI(title="Research Paper Agent")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

cached_count = store.load_cache()
logger.info("Loaded %s papers from the disk cache", cached_count)

renamed_count = store.rename_old_pdfs()
logger.info("Renamed %s old PDF files to the new name format", renamed_count)


# ---------------------------------------------------------------------------
# Step 1: One agent per provider, all sharing the same SQLite memory
# ---------------------------------------------------------------------------
conn = sqlite3.connect(database=str(DB_PATH), check_same_thread=False)
memory = SqliteSaver(conn=conn)
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
    paper_id: str | None = None


class ChatResponse(BaseModel):
    thread_id: str
    provider: str
    answer: str
    tool_steps: list[str]


class FolderRequest(BaseModel):
    path: str


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

    if chat_request.paper_id:
        message = make_paper_line(chat_request.paper_id) + message

    return message, provider, thread_id


def make_paper_line(paper_id):
    entry = store.get(paper_id)

    if entry and entry["meta"]:
        title = entry["meta"]["title"]
        line = f"[Selected paper: {paper_id} - {title}]\n"
    else:
        line = f"[Selected paper: {paper_id}]\n"
    return line


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


def retrive_all_thread_ids():
    all_threads = []

    for checkpoint in memory.list(None):
        thread_id = checkpoint.config["configurable"]["thread_id"]
        if thread_id not in all_threads:
            all_threads.append(thread_id)

    return all_threads


def load_conversation(thread_id):
    agent = get_agent(DEFAULT_PROVIDER)
    config = {"configurable": {"thread_id": thread_id}}
    return agent.get_state(config=config).values.get("messages", [])


def make_chat_history(messages):
    chat_history = []

    for message in messages:
        if message.type == "human":
            role = "user"
        elif message.type == "ai" and message.text.strip():
            role = "assistant"
        else:
            continue

        chat_history.append({"role": role, "content": message.text})

    return chat_history


def get_thread_title(messages):
    for message in messages:
        if message.type == "human":
            text = PAPER_LINE_PATTERN.sub("", message.text).strip()
            text = " ".join(text.split())

            if len(text) > TITLE_MAX_CHARS:
                text = text[:TITLE_MAX_CHARS] + "..."
            return text

    return "Empty chat"


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
                "loaded": store.find_pdf(paper_id) is not None,
                "summarized": entry["summary"] is not None,
            }
        )

    return {"papers": papers}


@app.delete("/api/papers/{paper_id}")
def delete_paper(paper_id: str):
    try:
        paper_id = store.normalize_id(paper_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    if store.get(paper_id) is None and store.find_pdf(paper_id) is None:
        raise HTTPException(status_code=404, detail="Paper not found.")

    try:
        store.delete_paper(paper_id)
    except Exception as error:
        logger.exception("Error while deleting %s", paper_id)
        raise HTTPException(status_code=500, detail=f"Could not delete the paper: {str(error)[:300]}")

    logger.info("Deleted paper %s", paper_id)
    return {"deleted": paper_id}


@app.get("/api/threads")
def list_threads():
    threads = []

    for thread_id in retrive_all_thread_ids():
        messages = load_conversation(thread_id)
        threads.append(
            {
                "thread_id": thread_id,
                "title": get_thread_title(messages),
            }
        )

    return {"threads": threads}


@app.get("/api/threads/{thread_id}")
def get_thread(thread_id: str):
    messages = load_conversation(thread_id)

    if not messages:
        raise HTTPException(status_code=404, detail="Chat not found.")

    return {"thread_id": thread_id, "messages": make_chat_history(messages)}


@app.get("/api/settings")
def get_settings():
    return {"download_folder": str(store.PAPERS_DIR)}


@app.post("/api/settings/download-folder")
def change_download_folder(folder_request: FolderRequest):
    try:
        path = store.set_papers_dir(folder_request.path)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        logger.exception("Error while changing the download folder")
        raise HTTPException(status_code=400, detail=f"Could not use this folder: {str(error)[:300]}")

    logger.info("Download folder changed to %s", path)
    return {"download_folder": str(path)}


@app.post("/api/settings/open-folder")
def open_download_folder():
    if not hasattr(os, "startfile"):
        raise HTTPException(status_code=400, detail="Opening folders only works on Windows.")

    os.startfile(store.PAPERS_DIR)
    return {"opened": str(store.PAPERS_DIR)}


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
