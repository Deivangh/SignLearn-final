"""Client for the data.gov.in Indian Sign Language Dictionary resource.

The resource does not hand out video URLs directly. It hands out 27 Google
Drive folder links: one per letter A-Z plus an "All Dictionary Videos" root.
This module gets those links; ``drive.py`` turns them into a word list.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import List

import requests

from ..config import (
    DATAGOV_API_KEY,
    DATAGOV_BASE,
    DATAGOV_RESOURCE_ID,
    MANIFEST_FILE,
    USER_AGENT,
)

FOLDER_ID_RE = re.compile(r"/folders/([A-Za-z0-9_-]{10,})")


@dataclass
class Folder:
    """One letter-folder of the official dictionary."""

    sr_no: int
    name: str
    link: str
    folder_id: str

    @property
    def is_root(self) -> bool:
        return "all dictionary" in self.name.lower()


class DataGovError(RuntimeError):
    pass


def fetch_folders(api_key: str | None = None, timeout: int = 30) -> List[Folder]:
    """Pull every record from the dictionary resource."""
    key = api_key or DATAGOV_API_KEY
    if not key:
        raise DataGovError(
            "No data.gov.in API key. Put DATAGOV_API_KEY in your .env file."
        )

    url = f"{DATAGOV_BASE}/{DATAGOV_RESOURCE_ID}"
    params = {"api-key": key, "format": "json", "limit": "100", "offset": "0"}
    resp = requests.get(
        url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT}
    )
    resp.raise_for_status()
    payload = resp.json()

    if payload.get("status") != "ok":
        raise DataGovError(f"data.gov.in returned: {payload.get('message')}")

    folders: List[Folder] = []
    for rec in payload.get("records", []):
        link = (rec.get("link_") or "").strip()
        match = FOLDER_ID_RE.search(link)
        if not match:
            continue
        folders.append(
            Folder(
                sr_no=int(rec.get("_sr_no_", 0)),
                name=(rec.get("folder_name") or "").strip(),
                link=link,
                folder_id=match.group(1),
            )
        )

    if not folders:
        raise DataGovError("The resource returned no usable Drive folder links.")
    return folders


def save_manifest(folders: List[Folder]) -> None:
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.write_text(
        json.dumps([asdict(f) for f in folders], indent=2), encoding="utf-8"
    )


def load_manifest() -> List[Folder]:
    if not MANIFEST_FILE.exists():
        return []
    raw = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    return [Folder(**item) for item in raw]
