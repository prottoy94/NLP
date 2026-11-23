# 📚 Research Paper Agent

An AI agent that searches arXiv, reads papers, writes structured summaries, answers questions,
compares papers and saves Markdown notes. You can also **upload your own PDF papers** and use
every tool on them.

Built with LangChain `create_agent`, FastAPI and a plain HTML/CSS/JS page.

## What it can do
| Tool | What it does |
|---|---|
| `search_arxiv` | Finds papers on arXiv (by relevance or newest first) |
| `list_uploaded_papers` | Lists the PDFs you uploaded |
| `load_paper` | Downloads an arXiv PDF (or reads your uploaded PDF) and extracts the text |
| `summarize_paper` | Structured summary: problem, method, datasets, results, limitations, equations |
| `ask_paper` | Answers a question using only that paper's text (RAG with FAISS) |
| `compare_papers` | Markdown comparison table for 2+ papers (arXiv and uploaded can be mixed) |
| `save_notes` | Saves the result to `notes/<date>-<title>.md` |

## Setup
1. Use the shared virtual environment in `LangChain_Models/venv` (or create one) and install:
   ```bash
   pip install -r requirements.txt
   ```
2. Put your keys in `LangChain_Models/.env`:
   ```
   GOOGLE_API_KEY=...       # Gemini chat + embeddings (embeddings are always Gemini)
   GROQ_API_KEY=...         # optional
   OPENROUTER_API_KEY=...   # optional
   ```

## Run
**Web app** (recommended):
```bash
python main.py
```
Then open http://127.0.0.1:8000. The API docs are at http://127.0.0.1:8000/docs.

**Command line chat:**
```bash
python agent.py            # Gemini
python agent.py groq       # or openrouter
```

## Using the web page
- Pick a **provider** at the top (gemini, groq, openrouter). You can switch in the middle of a chat.
- **Upload a paper** in the sidebar (PDF, up to 20 MB). It gets an id like `upload-my-paper`.
  Then ask things like *"Summarize the uploaded paper upload-my-paper"* or
  *"Compare my uploaded paper with 2 recent arXiv papers on AI text detection"*.
- Watch the **live steps** while the agent works.
- Download saved notes from the sidebar.

## Folders
| Folder | What is inside |
|---|---|
| `papers/` | Downloaded arXiv PDFs (cache) |
| `uploads/` | Your uploaded PDFs |
| `cache/` | Saved summaries (`.json`) and FAISS indexes, so a restart does not redo work |
| `notes/` | Saved Markdown notes |

## Good to know
- **Gemini free tier = 20 chat requests per day.** When it is used up, switch to groq or openrouter.
- **Groq** only sees the start and end of long papers, so its summaries can miss details.
- Asking the first question about a paper takes about 1-2 minutes (embedding has a rate limit).
  After that the FAISS index is saved in `cache/`, so later questions take seconds.
- Scanned PDFs (images only, no text) cannot be read.
