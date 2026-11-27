# YT_Summarizer

Chat with any YouTube video. Paste a link and the app downloads the video's transcript. You can then ask questions about the video or get a short summary, and every answer comes only from what was said in it.

The app is built with **LangChain** (a RAG pipeline) and **Google Gemini** (the LLM and embeddings), and it runs on a small **Flask** web server.

## Features

- **Load any YouTube video.** Works with `watch?v=`, `youtu.be/`, `/shorts/`, `/embed/` and `/live/` links.
- **Chat with the video.** Questions are answered from the transcript only. If the transcript doesn't contain the answer, the assistant says so instead of guessing.
- **One-click summary.** You get a short overview, the main points as bullets and a one-line conclusion.
- **Embedded player.** The video plays on the same page as the chat.
- **Separate sessions.** Each browser gets its own video and chat, so several people can use the app at once.
- **Clear error messages.** The app explains common failures such as disabled transcripts, private videos, a missing API key or a rate limit.

## How it works

```
YouTube URL
   │
   ▼
1. extract_video_id()   → get the 11-character video ID
2. fetch_transcript()   → download the English transcript
3. build_retriever()    → split into chunks (1000 chars, 200 overlap)
                          → embed with gemini-embedding-2
                          → store in FAISS, retrieve top 3 chunks
4. build_rag_chain()    → RunnableParallel(context, question)
                          | prompt | gemini-2.5-flash | StrOutputParser
5. generate_summary()   → sends the full transcript to Gemini for a summary
```

Chat questions go through the retriever, so only the 3 most relevant chunks reach the LLM. The summary sends the whole transcript, capped at 400,000 characters, because a summary needs the full video.

## Tech stack

| Part | Tool |
|---|---|
| RAG pipeline | LangChain (`langchain-core`, `langchain-community`, `langchain-text-splitters`) |
| LLM | Google Gemini `gemini-2.5-flash` |
| Embeddings | Google `gemini-embedding-2` |
| Vector store | FAISS (`faiss-cpu`) |
| Transcripts | `youtube-transcript-api` |
| Web server | Flask |
| Frontend | HTML, CSS, JavaScript |

## Project structure

```
yt_rag/
├── app.py              # Flask server: serves the page and the JSON API
├── rag_pipeline.py     # All LangChain / RAG logic, split into steps
├── using_chain.py      # Command-line version of the same RAG chain
├── main.py             # First version: step-by-step RAG without chains
├── templates/
│   └── index.html      # Web page
└── static/
    ├── css/style.css
    └── js/app.js
```

## Setup

1. **Install the dependencies** from the parent folder (`LangChain_Models`):

   ```bash
   pip install -r requirements.txt
   ```

2. **Add your Google API key.** Create a `.env` file in `LangChain_Models/` (the parent folder):

   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

   You can get a key from [Google AI Studio](https://aistudio.google.com/app/apikey).

3. **Run the app:**

   ```bash
   cd yt_rag
   python app.py
   ```

4. Open **http://127.0.0.1:5000** in your browser.

## API endpoints

| Method | Endpoint | Body | What it does |
|---|---|---|---|
| `POST` | `/api/load-video` | `{"url": "..."}` | Fetches the transcript and builds the index for the video |
| `POST` | `/api/summarize` | none | Returns a summary of the loaded video. It is generated once and then reused |
| `POST` | `/api/chat` | `{"question": "..."}` | Answers a question about the loaded video |

Every response is JSON with `"success": true` or `"success": false` plus an `"error"` message.

## Limitations

- The video needs an **English transcript**, either manual or auto-generated.
- Videos and indexes are kept **in memory**, so they are lost when the server restarts. Up to 20 sessions are stored at a time.
- YouTube sometimes blocks transcript requests from some networks or cloud servers.
- `debug=True` is turned on in `app.py`, so use it only on your own computer.
