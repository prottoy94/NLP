import json
import re
import shutil
import threading
from pathlib import Path

from schemas import PaperSummary


BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
NOTES_DIR = BASE_DIR / "notes"
CACHE_DIR = BASE_DIR / "cache"
SETTINGS_PATH = BASE_DIR / "settings.json"

UPLOADS_DIR.mkdir(exist_ok=True)
NOTES_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

UPLOAD_PREFIX = "upload-"
MAX_TITLE_CHARS = 80


# ---------------------------------------------------------------------------
# Step 0: Settings (the download folder for arXiv PDFs)
# ---------------------------------------------------------------------------
def load_settings():
    if not SETTINGS_PATH.exists():
        return {}

    try:
        settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception as error:
        print(f"Could not read settings.json: {error}")
        settings = {}
    return settings


def save_settings(settings):
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def set_papers_dir(folder):
    global PAPERS_DIR

    path = Path(str(folder).strip()).expanduser()
    if not path.is_absolute():
        raise ValueError("Please give a full folder path, for example D:\\Papers")

    path.mkdir(parents=True, exist_ok=True)
    PAPERS_DIR = path

    settings = load_settings()
    settings["download_folder"] = str(path)
    save_settings(settings)
    return path


PAPERS_DIR = Path(load_settings().get("download_folder", BASE_DIR / "papers"))
PAPERS_DIR.mkdir(parents=True, exist_ok=True)


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
# Step 3: File paths (arXiv PDFs are named [Paper no]-[Year]-[Title].pdf)
# ---------------------------------------------------------------------------
def make_pdf_name(meta):
    paper_id = safe_name(meta["paper_id"])

    if is_upload(meta["paper_id"]):
        return paper_id + ".pdf"

    year = meta["published"][:4]

    title = re.sub(r'[\\/:*?"<>|]', "", meta["title"])
    title = " ".join(title.split())
    title = title[:MAX_TITLE_CHARS].strip(" .")

    return f"{paper_id}-{year}-{title}.pdf"


def pdf_path(meta):
    file_name = make_pdf_name(meta)

    if is_upload(meta["paper_id"]):
        return UPLOADS_DIR / file_name
    return PAPERS_DIR / file_name


def find_pdf(paper_id):
    name = safe_name(paper_id)

    if is_upload(paper_id):
        path = UPLOADS_DIR / f"{name}.pdf"
        if path.exists():
            return path
        return None

    for path in PAPERS_DIR.glob(f"{name}-*.pdf"):
        return path

    old_path = PAPERS_DIR / f"{name}.pdf"
    if old_path.exists():
        return old_path
    return None


def rename_old_pdf(meta):
    old_path = PAPERS_DIR / f"{safe_name(meta['paper_id'])}.pdf"
    new_path = pdf_path(meta)

    if old_path.exists() and not new_path.exists():
        old_path.rename(new_path)


def rename_old_pdfs():
    count = 0

    for paper_id, entry in list(papers.items()):
        meta = entry["meta"]
        if meta is None or is_upload(paper_id):
            continue

        old_path = PAPERS_DIR / f"{safe_name(paper_id)}.pdf"
        if old_path.exists():
            rename_old_pdf(meta)
            count += 1

    return count


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


# ---------------------------------------------------------------------------
# Step 5: Delete a paper (PDF + cache + memory)
# ---------------------------------------------------------------------------
def delete_paper(paper_id):
    path = find_pdf(paper_id)
    if path is not None:
        path.unlink()

    delete_cache(paper_id)
    remove(paper_id)
