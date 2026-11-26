import json
import logging
import re
import time
from datetime import date
from pathlib import Path

import arxiv
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

import store
from llm import get_llm, get_embeddings
from schemas import PaperSummary, PaperComparison

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

GEMINI_MAX_CHARS = 150_000
GROQ_START_CHARS = 12_000
GROQ_END_CHARS = 4_000

EMBED_BATCH_SIZE = 30
EMBED_WAIT_SECONDS = 62

MAX_UPLOAD_MB = 20
COMPARE_MAX_TOKENS = 8000

arxiv_client = arxiv.Client(page_size=50, delay_seconds=3.0, num_retries=3)

HTTP_HEADERS = {"User-Agent": "research-paper-agent/0.1 (learning project)"}


# ---------------------------------------------------------------------------
# Step 1: Error helpers
# ---------------------------------------------------------------------------
def is_rate_limit(error):
    text = str(error).lower()
    if "429" in text or "resource_exhausted" in text or "rate limit" in text:
        return True
    return False


def is_daily_limit(error):
    text = str(error).lower()
    if "per day" in text or "perday" in text:
        return True
    return False


def friendly_error(action, error):
    logger.warning("Tool failed while %s: %s", action, str(error)[:300])

    if is_daily_limit(error):
        return (
            f"ERROR while {action}: the provider's free-tier DAILY limit is used up. "
            "Switch to another provider (for example groq) or try again tomorrow."
        )

    if is_rate_limit(error):
        return (
            f"ERROR while {action}: the LLM provider's free-tier rate limit was reached. "
            "Wait about a minute and try again, or switch provider."
        )

    if isinstance(error, requests.exceptions.ConnectionError):
        return f"ERROR while {action}: could not connect to the internet / arXiv. Check your connection."

    if isinstance(error, requests.exceptions.Timeout):
        return f"ERROR while {action}: the download took too long. Try again later."

    if isinstance(error, requests.exceptions.HTTPError):
        return f"ERROR while {action}: arXiv returned an error ({error.response.status_code}). Try again later."

    message = " ".join(str(error).split())
    return f"ERROR while {action}: {message[:300]}"


# ---------------------------------------------------------------------------
# Step 2: arXiv search
# ---------------------------------------------------------------------------
def make_meta(result):
    authors = []
    for author in result.authors:
        authors.append(author.name)

    meta = {
        "paper_id": result.get_short_id(),
        "title": " ".join(result.title.split()),
        "authors": authors,
        "published": result.published.date().isoformat(),
        "abstract": " ".join(result.summary.split()),
        "pdf_url": result.pdf_url,
        "source": "arxiv",
    }
    return meta


def build_arxiv_query(query):
    if re.search(r'\w+:|\bAND\b|\bOR\b|\bANDNOT\b|"', query):
        return query

    words = []
    for word in re.split(r"[^A-Za-z0-9]+", query):
        if len(word) > 1:
            words.append(f"all:{word}")

    if not words:
        return query
    return " AND ".join(words)


def search_papers(query, max_results=5, sort="relevance"):
    max_results = int(max_results)
    if max_results < 1:
        max_results = 1
    if max_results > 20:
        max_results = 20

    query = build_arxiv_query(query)

    if sort == "recent":
        search = arxiv.Search(query=query, max_results=max(25, max_results * 5))
    else:
        search = arxiv.Search(query=query, max_results=max_results)

    results = []
    for result in arxiv_client.results(search):
        results.append(make_meta(result))

    if sort == "recent":
        results.sort(key=lambda meta: meta["published"], reverse=True)
        results = results[:max_results]

    for meta in results:
        store.upsert(meta["paper_id"], meta=meta)
    return results


def get_meta(paper_id):
    entry = store.get(paper_id)
    if entry and entry["meta"]:
        return entry["meta"]

    if store.is_upload(paper_id):
        raise ValueError(
            f"No uploaded paper with id {paper_id}. Use list_uploaded_papers to see the uploaded papers."
        )

    search = arxiv.Search(id_list=[paper_id], max_results=1)
    results = list(arxiv_client.results(search))
    if not results:
        raise ValueError(f"No arXiv paper found with id {paper_id}.")

    meta = make_meta(results[0])
    store.upsert(paper_id, meta=meta)
    return meta


# ---------------------------------------------------------------------------
# Step 3: Load the paper's text (arXiv download or uploaded PDF)
# ---------------------------------------------------------------------------
def download_pdf(meta, path):
    response = requests.get(meta["pdf_url"], headers=HTTP_HEADERS, timeout=60)
    response.raise_for_status()

    if not response.content.startswith(b"%PDF"):
        raise ValueError(f"arXiv did not return a PDF for {meta['paper_id']}.")

    path.write_bytes(response.content)


def read_pdf_text(path, paper_id):
    try:
        loader = PyPDFLoader(str(path))
        pages = loader.load()
    except Exception as error:
        raise ValueError(f"Could not read the PDF for {paper_id}. The file may be broken. ({error})")

    text = "\n".join([page.page_content for page in pages]).strip()

    if len(text) < 500:
        raise ValueError(
            f"Could not read text from {paper_id} (it may be a scanned PDF with images only)."
        )
    return text, len(pages)


def load_paper_text(paper_id):
    paper_id = store.normalize_id(paper_id)

    entry = store.get(paper_id)
    if entry and entry["text"]:
        return entry

    meta = get_meta(paper_id)

    if not store.is_upload(paper_id):
        store.rename_old_pdf(meta)

    path = store.find_pdf(paper_id)

    if path is None:
        if store.is_upload(paper_id):
            raise ValueError(f"The uploaded file for {paper_id} is missing. Please upload it again.")
        path = store.pdf_path(meta)
        download_pdf(meta, path)

    text, page_count = read_pdf_text(path, paper_id)

    entry = store.upsert(paper_id, meta=meta, text=text, pages=page_count)
    store.save_cache(paper_id)
    return entry


# ---------------------------------------------------------------------------
# Step 4: Uploaded papers
# ---------------------------------------------------------------------------
def make_upload_id(file_name):
    name = make_file_name(Path(file_name).stem, max_length=50)
    upload_id = store.UPLOAD_PREFIX + name
    return upload_id


def add_uploaded_paper(file_name, file_bytes):
    if not file_name.lower().endswith(".pdf"):
        raise ValueError("Only PDF files can be uploaded.")

    if not file_bytes.startswith(b"%PDF"):
        raise ValueError("This file is not a real PDF.")

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise ValueError(f"The file is {size_mb:.1f} MB. The limit is {MAX_UPLOAD_MB} MB.")

    paper_id = make_upload_id(file_name)
    title = Path(file_name).stem.replace("_", " ").replace("-", " ").strip()

    meta = {
        "paper_id": paper_id,
        "title": title,
        "authors": [],
        "published": date.today().isoformat(),
        "abstract": "",
        "pdf_url": None,
        "source": "upload",
    }

    store.remove(paper_id)
    store.delete_cache(paper_id)

    path = store.pdf_path(meta)
    path.write_bytes(file_bytes)
    store.upsert(paper_id, meta=meta)

    try:
        entry = load_paper_text(paper_id)
    except Exception:
        path.unlink()
        store.remove(paper_id)
        store.delete_cache(paper_id)
        raise

    return entry


def list_uploads():
    uploads = []

    for paper_id, entry in list(store.papers.items()):
        if store.is_upload(paper_id) and entry["meta"] is not None:
            uploads.append(entry["meta"])

    return uploads


# ---------------------------------------------------------------------------
# Step 5: Summaries
# ---------------------------------------------------------------------------
def cut_text_for_provider(text, provider):
    if provider == "groq":
        if len(text) <= GROQ_START_CHARS + GROQ_END_CHARS:
            return text
        start = text[:GROQ_START_CHARS]
        end = text[-GROQ_END_CHARS:]
        return start + "\n\n[... middle of the paper omitted to fit the model's limits ...]\n\n" + end

    return text[:GEMINI_MAX_CHARS]


def load_summary_prompt():
    file_path = PROMPTS_DIR / "paper_summary_v2.json"
    data = json.loads(file_path.read_text(encoding="utf-8"))

    prompt = PromptTemplate(
        template=data["template"],
        input_variables=data["input_variables"],
    )
    return prompt


def summarize(paper_id, provider, style="Technical", length="Medium"):
    entry = load_paper_text(paper_id)
    if entry["summary"] is not None:
        return entry["summary"]

    llm = get_llm(provider).with_structured_output(PaperSummary)
    chain = load_summary_prompt() | llm

    summary = chain.invoke(
        {
            "paper_text": cut_text_for_provider(entry["text"], provider),
            "style": style,
            "length": length,
        }
    )
    if summary is None:
        raise ValueError("The model did not return a valid summary. Try again or switch provider.")

    store.upsert(paper_id, summary=summary)
    store.save_cache(paper_id)
    return summary


def make_bullets(items):
    if not items:
        return "- Insufficient information available"

    lines = []
    for item in items:
        lines.append(f"- {item}")
    return "\n".join(lines)


def paper_link(paper_id, text=None):
    if text is None:
        text = paper_id

    if store.is_upload(paper_id):
        return f"{text} (uploaded PDF)"
    return f"[{text}](https://arxiv.org/abs/{paper_id})"


def summary_to_markdown(paper_id, summary):
    entry = store.get(paper_id)
    meta = {}
    if entry and entry["meta"]:
        meta = entry["meta"]

    all_authors = meta.get("authors", [])
    authors = ", ".join(all_authors[:4])
    if len(all_authors) > 4:
        authors += " et al."

    if store.is_upload(paper_id):
        info_line = f"**Paper:** {paper_link(paper_id)}"
        if meta.get("published"):
            info_line += f" · **Uploaded:** {meta['published']}"
    else:
        info_line = f"**arXiv:** {paper_link(paper_id)}"
        if meta.get("published"):
            info_line += f" · **Published:** {meta['published']}"

    if authors:
        info_line += f" · **Authors:** {authors}"

    markdown = f"### {summary.title}\n"
    markdown += info_line + "\n\n"
    markdown += f"**Problem:** {summary.problem}\n\n"
    markdown += f"**Method:** {summary.method}\n\n"
    markdown += f"**Datasets:**\n{make_bullets(summary.datasets)}\n\n"
    markdown += f"**Key results:**\n{make_bullets(summary.key_results)}\n\n"
    markdown += f"**Limitations:**\n{make_bullets(summary.limitations)}\n\n"
    if summary.key_equations:
        markdown += f"**Key equations:**\n{make_bullets(summary.key_equations)}\n\n"
    markdown += f"**Takeaway:** {summary.one_line_takeaway}\n"
    return markdown


# ---------------------------------------------------------------------------
# Step 6: Questions about one paper (RAG, the FAISS index is saved to disk)
# ---------------------------------------------------------------------------
def build_vectorstore(chunks):
    vectorstore = None

    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[start : start + EMBED_BATCH_SIZE]

        if start > 0:
            time.sleep(EMBED_WAIT_SECONDS)

        for attempt in range(3):
            try:
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(batch, embedding=get_embeddings())
                else:
                    vectorstore.add_documents(batch)
                break
            except Exception as error:
                if attempt == 2:
                    raise

                if is_rate_limit(error):
                    wait_seconds = EMBED_WAIT_SECONDS
                else:
                    wait_seconds = 5

                logger.warning("Embedding failed (%s), waiting %ss", error, wait_seconds)
                time.sleep(wait_seconds)

    return vectorstore


def load_or_build_vectorstore(paper_id, text):
    folder = store.index_path(paper_id)

    if folder.exists():
        logger.info("Loading saved FAISS index for %s", paper_id)
        vectorstore = FAISS.load_local(
            str(folder), get_embeddings(), allow_dangerous_deserialization=True
        )
        return vectorstore

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=250)
    chunks = splitter.create_documents([text])

    vectorstore = build_vectorstore(chunks)
    vectorstore.save_local(str(folder))
    return vectorstore


def get_retriever(paper_id):
    entry = load_paper_text(paper_id)
    if entry["retriever"] is not None:
        return entry["retriever"]

    vectorstore = load_or_build_vectorstore(paper_id, entry["text"])
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 6}
    )

    store.upsert(paper_id, retriever=retriever)
    return retriever


def format_docs(docs):
    context_text = "\n\n".join([doc.page_content for doc in docs])
    return context_text


parser = StrOutputParser()

ask_prompt = PromptTemplate.from_template("""
You are a helpful assistant that answers questions about ONE research paper.
Answer ONLY from the paper excerpts below. Do NOT use outside knowledge.
If the excerpts do not contain the answer, reply exactly:
"I can't determine that from the paper's text."

Paper excerpts:
{context}

Question: {question}
""")


def ask(paper_id, question, provider):
    retriever = get_retriever(paper_id)

    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )

    main_chain = parallel_chain | ask_prompt | get_llm(provider) | parser

    try:
        answer = main_chain.invoke(question)
    except Exception as error:
        if not is_rate_limit(error):
            raise
        logger.warning("Rate limit hit while asking, waiting %ss", EMBED_WAIT_SECONDS)
        time.sleep(EMBED_WAIT_SECONDS)
        answer = main_chain.invoke(question)

    return answer


# ---------------------------------------------------------------------------
# Step 7: Compare papers
# ---------------------------------------------------------------------------
compare_prompt = PromptTemplate.from_template("""
Compare the research papers below using ONLY the structured summaries given.
Pick 4-5 useful comparison dimensions (for example Method, Datasets, Key result, Limitations).
Keep every cell short (under 25 words). Use "Insufficient information available" when a value is missing.
Use the paper ids exactly as they are written below.
{focus}

Summaries (JSON):
{summaries}
""")


def clean_cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def summary_for_comparison(summary):
    data = {
        "title": summary.title,
        "problem": summary.problem,
        "method": summary.method,
        "datasets": summary.datasets,
        "key_results": summary.key_results,
        "limitations": summary.limitations,
    }
    return data


def llm_comparison_table(summaries, provider, focus):
    if focus:
        focus_text = f"Focus especially on: {focus}"
    else:
        focus_text = ""

    llm = get_llm(provider, max_tokens=COMPARE_MAX_TOKENS).with_structured_output(
        PaperComparison, method="function_calling"
    )
    chain = compare_prompt | llm
    result = chain.invoke(
        {"summaries": json.dumps(summaries, indent=1), "focus": focus_text}
    )
    if result is None:
        raise ValueError("The model did not return a valid comparison.")

    columns = []
    for dimension in result.dimensions:
        columns.append(clean_cell(dimension))
    number_of_columns = len(columns)

    lines = []
    lines.append("| Paper | " + " | ".join(columns) + " |")
    lines.append("|" + "---|" * (number_of_columns + 1))

    for row in result.rows:
        cells = []
        for cell in row.cells:
            cells.append(clean_cell(cell))

        while len(cells) < number_of_columns:
            cells.append("")
        cells = cells[:number_of_columns]

        title = clean_cell(row.title)
        lines.append("| " + paper_link(row.paper_id, title) + " | " + " | ".join(cells) + " |")

    table = "\n".join(lines)
    return table + f"\n\n**Overall insight:** {result.overall_insight}\n"


def first_item(items):
    if not items:
        return "Insufficient information available"
    return items[0]


def simple_comparison_table(summaries):
    lines = []
    lines.append("| Paper | Method | Datasets | Key result | Limitation |")
    lines.append("|---|---|---|---|---|")

    for paper_id, summary in summaries.items():
        title = clean_cell(summary["title"])
        method = clean_cell(summary["method"])
        datasets = clean_cell(", ".join(summary["datasets"]))
        key_result = clean_cell(first_item(summary["key_results"]))
        limitation = clean_cell(first_item(summary["limitations"]))

        lines.append(
            f"| {paper_link(paper_id, title)} | {method} | {datasets} | {key_result} | {limitation} |"
        )

    table = "\n".join(lines)
    return table + "\n\n*(Simple table built directly from the saved summaries, because the model's comparison failed.)*\n"


def compare(paper_ids, provider, focus=""):
    clean_ids = []
    for paper_id in paper_ids:
        clean_ids.append(store.normalize_id(paper_id))

    if len(clean_ids) < 2:
        raise ValueError("Give at least 2 paper ids to compare.")

    summaries = {}
    for paper_id in clean_ids:
        summary = summarize(paper_id, provider)
        summaries[paper_id] = summary_for_comparison(summary)

    try:
        table = llm_comparison_table(summaries, provider, focus)
    except Exception as error:
        if is_rate_limit(error):
            raise
        logger.warning("LLM comparison failed (%s), using the simple table", str(error)[:200])
        table = simple_comparison_table(summaries)

    return table


# ---------------------------------------------------------------------------
# Step 8: Save notes
# ---------------------------------------------------------------------------
def make_file_name(title, max_length=60):
    name = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    name = name[:max_length].rstrip("-")
    if not name:
        name = "notes"
    return name


def save_markdown(title, markdown):
    base_name = f"{date.today().isoformat()}-{make_file_name(title)}"
    path = store.NOTES_DIR / f"{base_name}.md"

    counter = 2
    while path.exists():
        path = store.NOTES_DIR / f"{base_name}-{counter}.md"
        counter += 1

    if markdown.lstrip().startswith("#"):
        content = markdown
    else:
        content = f"# {title}\n\n{markdown}"

    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Step 9: The tools the agent can use
# ---------------------------------------------------------------------------
def build_tools(provider="gemini"):
    @tool
    def search_arxiv(query: str, max_results: int = 5, sort: str = "relevance") -> str:
        """Search arXiv for research papers.

        Args:
            query: keywords, e.g. "AI-generated text detection".
            max_results: how many papers to return (1-20).
            sort: "relevance" (best match) or "recent" (relevant papers, newest first).
        """
        if sort != "recent":
            sort = "relevance"

        try:
            results = search_papers(query, max_results, sort)
        except Exception as error:
            return friendly_error("searching arXiv", error)

        if not results:
            return f"No arXiv papers found for '{query}'. Try broader or different keywords."

        lines = []
        for number, meta in enumerate(results, 1):
            authors = ", ".join(meta["authors"][:3])
            if len(meta["authors"]) > 3:
                authors += " et al."

            lines.append(
                f"{number}. [{meta['paper_id']}] {meta['title']} ({meta['published']}) - {authors}\n"
                f"   Abstract: {meta['abstract'][:300]}..."
            )
        return "\n".join(lines)

    @tool
    def list_uploaded_papers() -> str:
        """List the PDF papers the user uploaded. Their ids start with "upload-"
        and work with load_paper, summarize_paper, ask_paper and compare_papers."""
        uploads = list_uploads()

        if not uploads:
            return "The user has not uploaded any papers yet."

        lines = []
        for number, meta in enumerate(uploads, 1):
            lines.append(f"{number}. [{meta['paper_id']}] {meta['title']} (uploaded {meta['published']})")
        return "\n".join(lines)

    @tool
    def load_paper(paper_id: str) -> str:
        """Load a paper's text so it can be summarized or questioned.
        arXiv papers are downloaded, uploaded papers are read from the uploaded PDF.

        Args:
            paper_id: an arXiv id like "2401.12345" or an uploaded paper id like "upload-my-paper".
        """
        try:
            entry = load_paper_text(paper_id)
        except Exception as error:
            return friendly_error(f"loading paper {paper_id}", error)

        meta = entry["meta"]
        return (
            f"Loaded [{meta['paper_id']}] '{meta['title']}': {entry['pages']} pages, "
            f"{len(entry['text']):,} characters of text."
        )

    @tool
    def summarize_paper(paper_id: str, style: str = "Technical", length: str = "Medium") -> str:
        """Write a structured summary (problem, method, datasets, results, limitations, equations)
        of one paper, using only the paper's own text. Loads the paper automatically if needed.

        Args:
            paper_id: an arXiv id like "2401.12345" or an uploaded paper id like "upload-my-paper".
            style: "Beginner-Friendly", "Technical", "Code-Oriented" or "Mathematical".
            length: "Short", "Medium" or "Long".
        """
        try:
            paper_id = store.normalize_id(paper_id)
            summary = summarize(paper_id, provider, style, length)
        except Exception as error:
            return friendly_error(f"summarizing paper {paper_id}", error)

        return summary_to_markdown(paper_id, summary)

    @tool
    def ask_paper(paper_id: str, question: str) -> str:
        """Answer a specific question about one paper, grounded only in that paper's text.

        Args:
            paper_id: an arXiv id like "2401.12345" or an uploaded paper id like "upload-my-paper".
            question: the question to answer, e.g. "Which datasets were used?".
        """
        try:
            paper_id = store.normalize_id(paper_id)
            answer = ask(paper_id, question, provider)
        except Exception as error:
            return friendly_error(f"answering a question about {paper_id}", error)

        return answer

    @tool
    def compare_papers(paper_ids: list[str], focus: str = "") -> str:
        """Compare 2 or more papers side by side and return a Markdown table plus an overall insight.
        Summarizes any paper that has not been summarized yet. Uploaded and arXiv papers can be mixed.

        Args:
            paper_ids: list of paper ids, e.g. ["2401.12345", "upload-my-paper"].
            focus: optional aspect to focus on, e.g. "detection method and robustness".
        """
        try:
            comparison = compare(paper_ids, provider, focus)
        except Exception as error:
            return friendly_error("comparing papers", error)

        return comparison

    @tool
    def save_notes(title: str, markdown: str) -> str:
        """Save Markdown research notes (summaries, comparison tables) to a file in the notes/ folder.

        Args:
            title: short title for the notes, e.g. "AI text detection literature review".
            markdown: the full Markdown content to save.
        """
        try:
            path = save_markdown(title, markdown)
        except Exception as error:
            return friendly_error("saving notes", error)

        return f"Saved notes to notes/{path.name}"

    tools = [
        search_arxiv,
        list_uploaded_papers,
        load_paper,
        summarize_paper,
        ask_paper,
        compare_papers,
        save_notes,
    ]
    return tools
