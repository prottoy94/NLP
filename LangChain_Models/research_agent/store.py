import json
import re
import shutil
import threading
from pathlib import Path

from schemas import PaperSummary


BASE_DIR = Path(__file__).resolve().parent
PAPERS_DIR = BASE_DIR / "papers"
UPLOADS_DIR = BASE_DIR / "uploads"
NOTES_DIR = BASE_DIR / "notes"
CACHE_DIR = BASE_DIR / "cache"

PAPERS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)
NOTES_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

UPLOAD_PREFIX = "upload-"


papers = {}
papers_lock = threading.Lock()


ARXIV_ID_PATTERN = re.compile(
    r"(\d{4}\.\d{4,5}(?:v\d+)?"
    r"|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)"
)


# ---------------------------------------------------------------------------
# Step 1: Paper ids (arXiv ids like 2401.12345, or uploaded ids like upload-my-paper)
# ---------------------------------------------------------------------------
def is_upload(paper_id):
    return str(paper_id).strip().startswith(UPLOAD_PREFIX)


def normalize_id(raw_id):
    text = str(raw_id).strip()

    if is_upload(text):
        return text.removesuffix(".pdf")

    match = ARXIV_ID_PATTERN.search(text)
    if not match:
        raise ValueError(
            f"'{raw_id}' does not look like a paper id. "
            "Use an arXiv id (example: 2401.12345) or an uploaded paper id (example: upload-my-paper)."
        )
    return match.group(1)


def remove_version(paper_id):
    if is_upload(paper_id):
        return paper_id
    return re.sub(r"v\d+$", "", paper_id)


def safe_name(paper_id):
    return remove_version(paper_id).replace("/", "_")


# ---------------------------------------------------------------------------
# Step 2: The in-memory papers dictionary
# ---------------------------------------------------------------------------
def get(paper_id):
    key = remove_version(paper_id)
    return papers.get(key)


def upsert(paper_id, **fields):
    key = remove_version(paper_id)

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


def remove(paper_id):
    key = remove_version(paper_id)

    with papers_lock:
        if key in papers:
            del papers[key]


# ---------------------------------------------------------------------------
# Step 3: File paths
# ---------------------------------------------------------------------------
def pdf_path(paper_id):
    file_name = safe_name(paper_id) + ".pdf"

    if is_upload(paper_id):
        return UPLOADS_DIR / file_name
    return PAPERS_DIR / file_name


def cache_path(paper_id):
    return CACHE_DIR / f"{safe_name(paper_id)}.json"


def index_path(paper_id):
    return CACHE_DIR / f"{safe_name(paper_id)}_index"


# ---------------------------------------------------------------------------
# Step 4: Disk cache (meta + summary as JSON, so a restart does not lose work)
# ---------------------------------------------------------------------------
def save_cache(paper_id):
    entry = get(paper_id)
    if entry is None or entry["meta"] is None:
        return

    data = {"meta": entry["meta"], "summary": None}
    if entry["summary"] is not None:
        data["summary"] = entry["summary"].model_dump()

    cache_path(paper_id).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_cache():
    count = 0

    for path in CACHE_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            meta = data["meta"]

            summary = None
            if data["summary"] is not None:
                summary = PaperSummary(**data["summary"])

            upsert(meta["paper_id"], meta=meta, summary=summary)
            count += 1
        except Exception as error:
            print(f"Skipping broken cache file {path.name}: {error}")

    return count


def delete_cache(paper_id):
    path = cache_path(paper_id)
    if path.exists():
        path.unlink()

    folder = index_path(paper_id)
    if folder.exists():
        shutil.rmtree(folder)
