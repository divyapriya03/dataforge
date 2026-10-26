"""Central configuration: storage locations and operational limits.

Everything is resolved from this file's location (never the current working
directory) and can be overridden with environment variables. Values are read at
call time so tests can point the app at a temporary directory.

Environment variables:
    DATAFORGE_DATA_DIR            root for uploads/, models/, synthetic/ (default: backend/data)
    DATAFORGE_DB_PATH             SQLite file (default: backend/dataforge.db)
    DATAFORGE_MAX_UPLOAD_MB       upload size limit (default: 25)
    DATAFORGE_MAX_GENERATE_ROWS   max rows per generation request (default: 50000)
    DATAFORGE_CORS_ORIGINS        comma-separated browser origins allowed to call the API
                                  (default: the Vite dev/preview servers on localhost; "*" allows every origin)
"""
import ntpath
import os
import re
from typing import List, Optional

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MIN_TRAIN_ROWS = 10          # minimum rows needed to fit a model
MAX_CATEGORIES = 50          # one-hot width guard per categorical column
KINDS = ("uploads", "models", "synthetic")


def get_data_dir() -> str:
    return os.path.abspath(os.environ.get("DATAFORGE_DATA_DIR") or os.path.join(BACKEND_DIR, "data"))


def get_db_path() -> str:
    return os.path.abspath(os.environ.get("DATAFORGE_DB_PATH") or os.path.join(BACKEND_DIR, "dataforge.db"))


def get_dir(kind: str) -> str:
    """Absolute path of the uploads/models/synthetic directory (created if missing)."""
    if kind not in KINDS:
        raise ValueError(f"unknown storage kind: {kind}")
    path = os.path.join(get_data_dir(), kind)
    os.makedirs(path, exist_ok=True)
    return path


def _int_env(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, default))
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


def max_upload_bytes() -> int:
    return _int_env("DATAFORGE_MAX_UPLOAD_MB", 25) * 1024 * 1024


def max_generate_rows() -> int:
    return _int_env("DATAFORGE_MAX_GENERATE_ROWS", 50000)


DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173", "http://127.0.0.1:5173",   # vite dev server
    "http://localhost:4173", "http://127.0.0.1:4173",   # vite preview
    "http://localhost:3000", "http://127.0.0.1:3000",
)


def cors_origins() -> List[str]:
    raw = os.environ.get("DATAFORGE_CORS_ORIGINS", "").strip()
    if not raw:
        return list(DEFAULT_CORS_ORIGINS)
    return [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]


_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str, default: str = "dataset.csv", max_len: int = 80) -> str:
    """A file name that is valid on every OS: only letters, digits, '.', '_' and '-', no leading dots, limited length.
    The extension (if any) is kept. Used for every name that is written to disk."""
    name = _UNSAFE.sub("_", name or "").strip("._")
    if not name:
        return default
    if len(name) > max_len:
        stem, dot, ext = name.rpartition(".")
        if dot and 0 < len(ext) <= 8:
            name = stem[: max_len - len(ext) - 1].rstrip("._") + "." + ext
        else:
            name = name[:max_len]
    return name or default


def safe_stem(name: str, max_len: int = 60) -> str:
    """safe_filename without the extension, for building derived file names."""
    base = name.rsplit(".", 1)[0] if "." in name else name
    return safe_filename(base, default="data", max_len=max_len)


def to_stored_path(abs_path: str) -> str:
    """Path to persist in SQLite: relative to the data dir (POSIX separators) when inside it."""
    abs_path = os.path.abspath(abs_path)
    data_dir = get_data_dir()
    try:
        if os.path.commonpath([abs_path, data_dir]) == data_dir:
            return os.path.relpath(abs_path, data_dir).replace(os.sep, "/")
    except ValueError:  # different drives on Windows
        pass
    return abs_path


def resolve_stored_path(stored: Optional[str], kind: str) -> Optional[str]:
    """Turn a path stored in SQLite into a usable absolute path.

    * relative paths are resolved against the data dir (current convention);
    * absolute paths are used as-is when the file exists;
    * otherwise (legacy rows written on another machine) the file name is looked
      up in the matching ``<data dir>/<kind>/`` folder.
    The returned path may not exist; callers must check.
    """
    if not stored:
        return None
    if not (os.path.isabs(stored) or ntpath.isabs(stored)):
        return os.path.join(get_data_dir(), *stored.replace("\\", "/").split("/"))
    if os.path.exists(stored):
        return stored
    return os.path.join(get_dir(kind), ntpath.basename(stored))
