import re
import threading
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PAPERS_DIR = BASE_DIR / "papers"
NOTES_DIR = BASE_DIR / "notes"

PAPERS_DIR.mkdir(exist_ok=True)
NOTES_DIR.mkdir(exist_ok=True)


papers = {}
papers_lock = threading.Lock()


ARXIV_ID_PATTERN = re.compile(
    r"(\d{4}\.\d{4,5}(?:v\d+)?"
    r"|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)"
)


def normalize_id(raw_id):
    match = ARXIV_ID_PATTERN.search(str(raw_id).strip())
    if not match:
        raise ValueError(
            f"'{raw_id}' does not look like an arXiv id. Example: 2401.12345"
        )
    return match.group(1)


def remove_version(arxiv_id):
    return re.sub(r"v\d+$", "", arxiv_id)


def get(arxiv_id):
    key = remove_version(arxiv_id)
    return papers.get(key)


def upsert(arxiv_id, **fields):
    key = remove_version(arxiv_id)

    with papers_lock:
        if key not in papers:
            papers[key] = {
                "meta": None,
                "text": None,
                "pages": 0,
                "retriever": None,
                "summary": None,
            }

        entry = papers[key]
        entry.update(fields)
        return entry


def pdf_path(arxiv_id):
    file_name = remove_version(arxiv_id).replace("/", "_") + ".pdf"
    return PAPERS_DIR / file_name
