"""Read public Google Drive folders and pull video files out of them.

Two backends, tried in order:

1. **Drive API v3** — used when GOOGLE_API_KEY is set. Stable, paginated,
   returns real metadata. This is the one you want.
2. **embeddedfolderview** — Drive's public, key-free folder listing page.
   Works today, but it is an undocumented HTML endpoint, so treat it as a
   fallback rather than the foundation.
"""
from __future__ import annotations

import html
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import requests

from ..config import GOOGLE_API_KEY, USER_AGENT

DRIVE_API = "https://www.googleapis.com/drive/v3/files"
EMBED_VIEW = "https://drive.google.com/embeddedfolderview"
FOLDER_MIME = "application/vnd.google-apps.folder"

VIDEO_EXTS = (".mp4", ".mov", ".webm", ".m4v", ".avi", ".mkv")

_ENTRY_RE = re.compile(
    r'<div class="flip-entry"[^>]*id="entry-([A-Za-z0-9_-]+)".*?'
    r'<div class="flip-entry-title">(.*?)</div>',
    re.DOTALL,
)
_FOLDER_HINT_RE = re.compile(r'flip-entry-info[^>]*>\s*<img[^>]*folder', re.I)


@dataclass
class DriveFile:
    file_id: str
    name: str
    is_folder: bool = False
    size: Optional[int] = None

    @property
    def is_video(self) -> bool:
        return self.name.lower().endswith(VIDEO_EXTS)

    @property
    def stream_url(self) -> str:
        return f"https://drive.google.com/uc?export=download&id={self.file_id}"

    @property
    def preview_url(self) -> str:
        return f"https://drive.google.com/file/d/{self.file_id}/preview"


class DriveError(RuntimeError):
    pass


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


# ----------------------------------------------------------------------
# Backend 1: Drive API v3
# ----------------------------------------------------------------------
def _list_via_api(folder_id: str, api_key: str, timeout: int = 30) -> List[DriveFile]:
    out: List[DriveFile] = []
    page_token = None
    sess = _session()

    while True:
        params = {
            "q": f"'{folder_id}' in parents and trashed = false",
            "key": api_key,
            "fields": "nextPageToken, files(id, name, mimeType, size)",
            "pageSize": 1000,
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
            "orderBy": "name",
        }
        if page_token:
            params["pageToken"] = page_token

        resp = sess.get(DRIVE_API, params=params, timeout=timeout)
        if resp.status_code == 403:
            raise DriveError(
                "Drive API rejected the key (403). Enable the Drive API in your "
                "Google Cloud project, or clear GOOGLE_API_KEY to use the "
                "key-free fallback."
            )
        resp.raise_for_status()
        payload = resp.json()

        for f in payload.get("files", []):
            size = f.get("size")
            out.append(
                DriveFile(
                    file_id=f["id"],
                    name=f.get("name", ""),
                    is_folder=f.get("mimeType") == FOLDER_MIME,
                    size=int(size) if size else None,
                )
            )

        page_token = payload.get("nextPageToken")
        if not page_token:
            break
    return out


# ----------------------------------------------------------------------
# Backend 2: public embedded folder view (no key)
# ----------------------------------------------------------------------
def _list_via_embed(folder_id: str, timeout: int = 30) -> List[DriveFile]:
    sess = _session()
    resp = sess.get(
        EMBED_VIEW, params={"id": folder_id, "pageSize": 1000}, timeout=timeout
    )
    resp.raise_for_status()
    body = resp.text

    out: List[DriveFile] = []
    for chunk in body.split('<div class="flip-entry"')[1:]:
        chunk = '<div class="flip-entry"' + chunk
        m = _ENTRY_RE.search(chunk)
        if not m:
            continue
        file_id, raw_name = m.group(1), html.unescape(m.group(2)).strip()
        looks_like_folder = bool(_FOLDER_HINT_RE.search(chunk)) or (
            "." not in raw_name[-6:]
        )
        out.append(
            DriveFile(file_id=file_id, name=raw_name, is_folder=looks_like_folder)
        )

    if not out:
        raise DriveError(
            f"Could not read folder {folder_id} without a key. Set GOOGLE_API_KEY."
        )
    return out


def list_folder(folder_id: str, api_key: str | None = None) -> List[DriveFile]:
    key = api_key if api_key is not None else GOOGLE_API_KEY
    if key:
        return _list_via_api(folder_id, key)
    return _list_via_embed(folder_id)


def walk_folder(
    folder_id: str,
    api_key: str | None = None,
    max_depth: int = 3,
    _depth: int = 0,
    _seen: set | None = None,
) -> Iterator[DriveFile]:
    """Yield every video file under a folder, descending into subfolders."""
    _seen = _seen if _seen is not None else set()
    if folder_id in _seen or _depth > max_depth:
        return
    _seen.add(folder_id)

    try:
        entries = list_folder(folder_id, api_key)
    except Exception as exc:  # one bad folder should not kill the crawl
        print(f"  ! skipped folder {folder_id}: {exc}")
        return

    for entry in entries:
        if entry.is_folder:
            time.sleep(0.15)
            yield from walk_folder(entry.file_id, api_key, max_depth, _depth + 1, _seen)
        elif entry.is_video:
            yield entry


# ----------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------
_CONFIRM_RE = re.compile(r'name="confirm"\s+value="([^"]+)"')


def download(file_id: str, dest: Path, timeout: int = 120) -> Path:
    """Fetch a Drive file to disk, clearing the virus-scan interstitial."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1024:
        return dest

    sess = _session()
    url = "https://drive.google.com/uc?export=download"
    resp = sess.get(url, params={"id": file_id}, stream=True, timeout=timeout)

    if "text/html" in resp.headers.get("Content-Type", ""):
        token = None
        for k, v in sess.cookies.items():
            if k.startswith("download_warning"):
                token = v
        if not token:
            m = _CONFIRM_RE.search(resp.text[:60000])
            token = m.group(1) if m else "t"
        resp = sess.get(
            url,
            params={"id": file_id, "confirm": token},
            stream=True,
            timeout=timeout,
        )

    resp.raise_for_status()
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as fh:
        for block in resp.iter_content(chunk_size=1 << 16):
            if block:
                fh.write(block)

    if tmp.stat().st_size < 1024:
        tmp.unlink(missing_ok=True)
        raise DriveError(f"Download for {file_id} came back empty.")

    tmp.replace(dest)
    return dest
