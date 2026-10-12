"""
app.py
------
The Flask web server. It:
  - serves the web page (templates/index.html)
  - provides 3 JSON API endpoints that call functions from rag_pipeline.py

Run it with:   python app.py
Then open:     http://127.0.0.1:5000
"""

import os
import uuid
import logging
import threading

from flask import Flask, render_template, request, jsonify, session

import rag_pipeline
from rag_pipeline import RAGError

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Flask needs a secret key to sign the session cookie.
# A random key each time the server starts is fine for this project,
# because our video data is kept in memory and is also reset on restart.
app.secret_key = os.urandom(24)


# ---------------------------------------------------------------------------
# In-memory storage for each browser session
# ---------------------------------------------------------------------------
# Every browser gets its own random "sid" (session id) in a cookie.
# video_sessions[sid] holds THAT user's video, transcript and RAG chain,
# so two people using the app at the same time never mix their videos.
#
# video_sessions = {
#     "some-sid": {
#         "video_id": "...",
#         "title": "...",
#         "transcript_text": "...",
#         "rag_chain": <chain>,
#         "summary": None or "...",
#     }
# }
video_sessions = {}
sessions_lock = threading.Lock()  # protects the dict when requests run at the same time
MAX_SESSIONS = 20  # keep memory usage small


def get_session_id():
    """Return this browser's session id (create one if it doesn't exist yet)."""
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex
    return session["sid"]


def get_current_video():
    """Return the loaded video data for this browser, or None."""
    with sessions_lock:
        return video_sessions.get(get_session_id())


def error_response(message, status_code=400):
    return jsonify({"success": False, "error": message}), status_code


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/load-video", methods=["POST"])
def load_video():
    data = request.get_json(silent=True) or {}
    url = data.get("url", "")

    try:
        # 1. URL -> video ID
        video_id = rag_pipeline.extract_video_id(url)
        logger.info("Loading video %s", video_id)

        # 2. Transcript
        transcript_text = rag_pipeline.fetch_transcript(video_id)

        # 3. Split + embed + FAISS + retriever  (done ONCE per video)
        retriever, number_of_chunks = rag_pipeline.build_retriever(transcript_text)

        # 4. Build the RAG chain for this video's retriever
        rag_chain = rag_pipeline.build_rag_chain(retriever)

        # Title is optional
        title = rag_pipeline.get_video_title(video_id)

    except RAGError as error:
        return error_response(str(error))
    except Exception:
        logger.exception("Unexpected error in /api/load-video")
        return error_response("Unexpected server error while loading the video.", 500)

    # 5. Save (or replace) this browser's active video
    sid = get_session_id()
    with sessions_lock:
        # If too many sessions are stored, remove the oldest one
        if sid not in video_sessions and len(video_sessions) >= MAX_SESSIONS:
            oldest_sid = next(iter(video_sessions))
            del video_sessions[oldest_sid]

        video_sessions[sid] = {
            "video_id": video_id,
            "title": title,
            "transcript_text": transcript_text,
            "rag_chain": rag_chain,
            "summary": None,
        }

    logger.info("Video %s indexed: %d chunks", video_id, number_of_chunks)
    return jsonify(
        {
            "success": True,
            "video_id": video_id,
            "title": title,
            "embed_url": f"https://www.youtube.com/embed/{video_id}",
            "num_chunks": number_of_chunks,
            "transcript_chars": len(transcript_text),
        }
    )


@app.route("/api/summarize", methods=["POST"])
def summarize():
    video = get_current_video()
    if video is None:
        return error_response("Please load a video first.")

    # Re-use the summary if we already made one for this video
    if video["summary"]:
        return jsonify({"success": True, "summary": video["summary"]})

    try:
        summary = rag_pipeline.generate_summary(video["transcript_text"])
    except RAGError as error:
        return error_response(str(error))
    except Exception:
        logger.exception("Unexpected error in /api/summarize")
        return error_response(
            "Unexpected server error while generating the summary.", 500
        )

    video["summary"] = summary
    return jsonify({"success": True, "summary": summary})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return error_response("Please type a question.")

    video = get_current_video()
    if video is None:
        return error_response("Please load a video before asking questions.")

    try:
        # The question comes from the user's chat box - nothing is hardcoded.
        answer = rag_pipeline.answer_question(video["rag_chain"], question)
    except RAGError as error:
        return error_response(str(error))
    except Exception:
        logger.exception("Unexpected error in /api/chat")
        return error_response("Unexpected server error while answering.", 500)

    return jsonify({"success": True, "answer": answer})


if __name__ == "__main__":
    # debug=True auto-reloads when you edit code. Only use it on your own computer.
    app.run(host="127.0.0.1", port=5000, debug=True)
