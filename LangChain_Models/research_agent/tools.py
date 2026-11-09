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

EMBED_BATCH_SIZE = 80
EMBED_WAIT_SECONDS = 62

arxiv_client = arxiv.Client(page_size=50, delay_seconds=3.0, num_retries=3)

HTTP_HEADERS = {"User-Agent": "research-paper-agent/0.1 (learning project)"}


def is_rate_limit(error):
    text = str(error).lower()
    if "429" in text or "resource_exhausted" in text or "rate limit" in text:
        return True
    return False


def friendly_error(action, error):
    logger.warning("Tool failed while %s: %s", action, error)

    if is_rate_limit(error):
        return (
            f"ERROR while {action}: the LLM provider's free-tier rate limit was reached. "
            "Wait about a minute and try again, or switch provider."
        )

    message = " ".join(str(error).split())
    return f"ERROR while {action}: {message[:300]}"


def make_meta(result):
    authors = []
    for author in result.authors:
        authors.append(author.name)

    meta = {
        "arxiv_id": result.get_short_id(),
        "title": " ".join(result.title.split()),
        "authors": authors,
        "published": result.published.date().isoformat(),
        "abstract": " ".join(result.summary.split()),
        "pdf_url": result.pdf_url,
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
        store.upsert(meta["arxiv_id"], meta=meta)
    return results


def get_meta(arxiv_id):
    entry = store.get(arxiv_id)
    if entry and entry["meta"]:
        return entry["meta"]

    search = arxiv.Search(id_list=[arxiv_id], max_results=1)
    results = list(arxiv_client.results(search))
    if not results:
        raise ValueError(f"No arXiv paper found with id {arxiv_id}.")

    meta = make_meta(results[0])
    store.upsert(arxiv_id, meta=meta)
    return meta


def load_paper_text(arxiv_id):
    arxiv_id = store.normalize_id(arxiv_id)

    entry = store.get(arxiv_id)
    if entry and entry["text"]:
        return entry

    meta = get_meta(arxiv_id)
    path = store.pdf_path(arxiv_id)

    if not path.exists():
        response = requests.get(meta["pdf_url"], headers=HTTP_HEADERS, timeout=60)
        response.raise_for_status()
        if not response.content.startswith(b"%PDF"):
            raise ValueError(f"arXiv did not return a PDF for {arxiv_id}.")
        path.write_bytes(response.content)

    loader = PyPDFLoader(str(path))
    pages = loader.load()
    text = "\n".join([page.page_content for page in pages]).strip()

    if len(text) < 500:
        raise ValueError(
            f"Could not read text from {arxiv_id} (it may be a scanned PDF)."
        )

    entry = store.upsert(arxiv_id, meta=meta, text=text, pages=len(pages))
    return entry


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


def summarize(arxiv_id, provider, style="Technical", length="Medium"):
    entry = load_paper_text(arxiv_id)
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

    store.upsert(arxiv_id, summary=summary)
    return summary


def make_bullets(items):
    if not items:
        return "- Insufficient information available"

    lines = []
    for item in items:
        lines.append(f"- {item}")
    return "\n".join(lines)


def summary_to_markdown(arxiv_id, summary):
    entry = store.get(arxiv_id)
    meta = {}
    if entry and entry["meta"]:
        meta = entry["meta"]

    all_authors = meta.get("authors", [])
    authors = ", ".join(all_authors[:4])
    if len(all_authors) > 4:
        authors += " et al."

    info_line = f"**arXiv:** [{arxiv_id}](https://arxiv.org/abs/{arxiv_id})"
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
                if not is_rate_limit(error) or attempt == 2:
                    raise
                logger.warning("Embedding rate limit hit, waiting %ss", EMBED_WAIT_SECONDS)
                time.sleep(EMBED_WAIT_SECONDS)

    return vectorstore


def get_retriever(arxiv_id):
    entry = load_paper_text(arxiv_id)
    if entry["retriever"] is not None:
        return entry["retriever"]

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=250)
    chunks = splitter.create_documents([entry["text"]])

    vectorstore = build_vectorstore(chunks)
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )

    store.upsert(arxiv_id, retriever=retriever)
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


def ask(arxiv_id, question, provider):
    retriever = get_retriever(arxiv_id)

    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )

    main_chain = parallel_chain | ask_prompt | get_llm(provider) | parser
    return main_chain.invoke(question)


compare_prompt = PromptTemplate.from_template("""
Compare the research papers below using ONLY the structured summaries given.
Pick 4-5 useful comparison dimensions (for example Method, Datasets, Key result, Limitations).
Keep every cell short (under 25 words). Use "Insufficient information available" when a value is missing.
{focus}

Summaries (JSON):
{summaries}
""")


def clean_cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def compare(arxiv_ids, provider, focus=""):
    paper_ids = []
    for arxiv_id in arxiv_ids:
        paper_ids.append(store.normalize_id(arxiv_id))

    if len(paper_ids) < 2:
        raise ValueError("Give at least 2 arXiv ids to compare.")

    summaries = {}
    for arxiv_id in paper_ids:
        summary = summarize(arxiv_id, provider)
        summaries[arxiv_id] = summary.model_dump()

    if focus:
        focus_text = f"Focus especially on: {focus}"
    else:
        focus_text = ""

    llm = get_llm(provider).with_structured_output(PaperComparison)
    chain = compare_prompt | llm
    result = chain.invoke(
        {"summaries": json.dumps(summaries, indent=1), "focus": focus_text}
    )
    if result is None:
        raise ValueError("The model did not return a valid comparison. Try again or switch provider.")

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
        paper_link = f"[{title}](https://arxiv.org/abs/{row.arxiv_id})"
        lines.append("| " + paper_link + " | " + " | ".join(cells) + " |")

    table = "\n".join(lines)
    return table + f"\n\n**Overall insight:** {result.overall_insight}\n"


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
            return f"No arXiv papers found for '{query}'. Try broader keywords."

        lines = []
        for number, meta in enumerate(results, 1):
            authors = ", ".join(meta["authors"][:3])
            if len(meta["authors"]) > 3:
                authors += " et al."

            lines.append(
                f"{number}. [{meta['arxiv_id']}] {meta['title']} ({meta['published']}) - {authors}\n"
                f"   Abstract: {meta['abstract'][:300]}..."
            )
        return "\n".join(lines)

    @tool
    def load_paper(arxiv_id: str) -> str:
        """Download a paper's PDF from arXiv and extract its text so it can be summarized or questioned.

        Args:
            arxiv_id: the arXiv id, e.g. "2401.12345".
        """
        try:
            entry = load_paper_text(arxiv_id)
        except Exception as error:
            return friendly_error(f"loading paper {arxiv_id}", error)

        meta = entry["meta"]
        return (
            f"Loaded [{meta['arxiv_id']}] '{meta['title']}': {entry['pages']} pages, "
            f"{len(entry['text']):,} characters of text."
        )

    @tool
    def summarize_paper(arxiv_id: str, style: str = "Technical", length: str = "Medium") -> str:
        """Write a structured summary (problem, method, datasets, results, limitations, equations)
        of one paper, using only the paper's own text. Loads the paper automatically if needed.

        Args:
            arxiv_id: the arXiv id, e.g. "2401.12345".
            style: "Beginner-Friendly", "Technical", "Code-Oriented" or "Mathematical".
            length: "Short", "Medium" or "Long".
        """
        try:
            arxiv_id = store.normalize_id(arxiv_id)
            summary = summarize(arxiv_id, provider, style, length)
        except Exception as error:
            return friendly_error(f"summarizing paper {arxiv_id}", error)

        return summary_to_markdown(arxiv_id, summary)

    @tool
    def ask_paper(arxiv_id: str, question: str) -> str:
        """Answer a specific question about one paper, grounded only in that paper's text.

        Args:
            arxiv_id: the arXiv id, e.g. "2401.12345".
            question: the question to answer, e.g. "Which datasets were used?".
        """
        try:
            arxiv_id = store.normalize_id(arxiv_id)
            answer = ask(arxiv_id, question, provider)
        except Exception as error:
            return friendly_error(f"answering a question about {arxiv_id}", error)

        return answer

    @tool
    def compare_papers(arxiv_ids: list[str], focus: str = "") -> str:
        """Compare 2 or more papers side by side and return a Markdown table plus an overall insight.
        Summarizes any paper that has not been summarized yet.

        Args:
            arxiv_ids: list of arXiv ids, e.g. ["2401.12345", "2312.00001"].
            focus: optional aspect to focus on, e.g. "detection method and robustness".
        """
        try:
            comparison = compare(arxiv_ids, provider, focus)
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

    tools = [search_arxiv, load_paper, summarize_paper, ask_paper, compare_papers, save_notes]
    return tools
