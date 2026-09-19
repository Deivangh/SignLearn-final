#!/usr/bin/env python
"""Build the local ISL dictionary index. Run this once, first.

    python scripts/build_index.py
    python scripts/build_index.py --letters A B C   # partial, for a quick test
    python scripts/build_index.py --refresh         # re-annotate, no crawling

It asks data.gov.in for the dictionary's Drive folder links, crawls each
letter folder plus the extra collections that sit only under the root
("New 2500 ISL Dictionary Videos", "NCERT", "MHSL"), and writes
``data/isl_index.json`` — a word -> Drive file id map. No video is
downloaded here; that happens on demand, or via warm_cache.py.

Synonym groups ("House_Home.mp4" serves both words) are worked out with
WordNet, so building needs ``pip install nltk`` and
``python -c "import nltk; nltk.download('wordnet')"``. The app itself does not.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signbridge.config import GOOGLE_API_KEY, INDEX_FILE  # noqa: E402
from signbridge.isl import datagov, drive, synonyms  # noqa: E402
from signbridge.isl.index import SignEntry, SignIndex, index_aliases, normalise  # noqa: E402


def _annotate(entries: list[SignEntry]) -> None:
    if not synonyms.available():
        print("  ! WordNet not available — synonym groups skipped.")
        print("    pip install nltk && python -c \"import nltk; nltk.download('wordnet')\"")
        return
    print("→ Finding synonym groups…", flush=True)
    found = synonyms.annotate(entries)
    print(f"  {found} multi-word names are synonym groups")


def _refresh() -> int:
    """Recompute aliases and groups on the existing index, without crawling."""
    index = SignIndex.load()
    entries = [e for e in index.entries if not e.path]
    if not entries:
        print("✗ No index to refresh. Run without --refresh first.")
        return 1
    for e in entries:
        e.aliases = index_aliases(e.word, e.source_name, e.letter)
    _annotate(entries)
    SignIndex.save(entries)
    print(f"✓ {len(entries):,} signs refreshed in {INDEX_FILE}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the ISL dictionary index.")
    ap.add_argument("--letters", nargs="*", help="Only crawl these folders, e.g. A B C")
    ap.add_argument("--no-extras", action="store_true",
                    help="Skip the collections that live only under the root folder.")
    ap.add_argument("--refresh", action="store_true",
                    help="Recompute aliases and synonym groups on the existing index.")
    ap.add_argument("--max-depth", type=int, default=4)
    args = ap.parse_args()

    if args.refresh:
        return _refresh()

    print("→ Asking data.gov.in for the dictionary layout…")
    try:
        folders = datagov.fetch_folders()
    except Exception as exc:
        print(f"✗ {exc}")
        return 1

    datagov.save_manifest(folders)
    print(f"  found {len(folders)} folders")

    root = next((f for f in folders if f.is_root), None)
    folders = [f for f in folders if not f.is_root]
    if args.letters:
        wanted = {l.upper() for l in args.letters}
        folders = [f for f in folders if f.name.upper() in wanted]

    # The root also holds whole collections that no letter folder links to.
    # Letter folders go first so the original signs keep their keys.
    crawl = [(f.name, f.folder_id) for f in folders]
    if root and not args.letters and not args.no_extras:
        known = {f.name.strip().upper() for f in folders}
        try:
            for sub in drive.list_folder(root.folder_id):
                if sub.is_folder and sub.name.strip().upper() not in known:
                    crawl.append((sub.name.strip(), sub.file_id))
        except Exception as exc:
            print(f"  ! could not list the root folder: {exc}")

    if not GOOGLE_API_KEY:
        print("  no GOOGLE_API_KEY set — using Drive's public folder view.")
        print("  If folders come back empty, add a key with the Drive API enabled.")

    entries: list[SignEntry] = []
    taken: set[str] = set()
    tally = Counter()

    for name, folder_id in crawl:
        print(f"→ {name or folder_id}", flush=True)
        count = 0
        for f in drive.walk_folder(folder_id, max_depth=args.max_depth):
            key = normalise(f.name)
            if not key or key in taken:        # Hindi-only names normalise to ""
                continue
            taken.add(key)
            entries.append(
                SignEntry(
                    word=key,
                    file_id=f.file_id,
                    source_name=f.name,
                    letter=name,
                    aliases=index_aliases(key, f.name, name),
                )
            )
            count += 1
        tally[name] = count
        print(f"  {count} signs")

    if not entries:
        print("\n✗ Nothing indexed. Most likely the folder listing was blocked.")
        print("  Set GOOGLE_API_KEY in .env (Google Cloud → enable Drive API) and retry.")
        return 1

    _annotate(entries)
    SignIndex.save(entries)
    print(f"\n✓ {len(entries):,} signs written to {INDEX_FILE}")
    thin = [n for n, c in tally.items() if c == 0]
    if thin:
        print(f"  empty folders (worth re-running): {', '.join(thin)}")
    print("\nNext:  streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
