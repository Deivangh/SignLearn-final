#!/usr/bin/env python
"""Pre-download clips and pre-extract avatar motion.

The app does both on demand, but the first read of a page is much nicer if
the common words are already sitting on disk.

    python scripts/warm_cache.py --common 500        # download 500 frequent words
    python scripts/warm_cache.py --poses             # extract motion for what is cached
    python scripts/warm_cache.py --words apple book school
    python scripts/warm_cache.py --all --poses       # everything (slow, large)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signbridge.avatar.framing import load_crops, save_crop, scan_clip  # noqa: E402
from signbridge.avatar.pose_extract import extract_pose, has_pose  # noqa: E402
from signbridge.isl import drive  # noqa: E402
from signbridge.isl.index import SignIndex  # noqa: E402

# Rough frequency order for school-level English reading. Downloading these
# first covers the bulk of any ordinary page.
COMMON = """
me you we he she they my your our this that here there now today tomorrow
yesterday good bad big small new old happy sad angry hungry thirsty tired sick
hot cold fast slow easy hard right wrong yes no not more less many few all some
go come eat drink sleep wake sit stand walk run stop start open close give take
make do work play read write learn teach study help ask answer tell say talk
listen hear see look watch know understand think remember forget want need like
love hate feel try find get put keep hold buy sell pay meet call wait live die
man woman boy girl child baby friend family mother father brother sister
teacher student doctor police farmer worker name age year month week day
morning afternoon evening night time hour minute home house school college
hospital shop market office road city village country india water food rice
bread milk tea fruit apple banana mango vegetable money book pen paper bag
phone car bus train cycle tree flower animal dog cat bird cow sun moon star sky
rain wind fire earth head eye ear nose mouth hand finger leg foot heart body
one two three four five six seven eight nine ten hundred thousand
""".split()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--common", type=int, metavar="N",
                    help="Download the N most common school words.")
    ap.add_argument("--words", nargs="*", help="Download these specific words.")
    ap.add_argument("--all", action="store_true", help="Download the whole dictionary.")
    ap.add_argument("--poses", action="store_true",
                    help="Extract avatar motion from every cached clip.")
    ap.add_argument("--crops", action="store_true",
                    help="Find the signer in every cached clip so the video "
                         "player can zoom to their hands. Fast; do this first.")
    ap.add_argument("--limit", type=int, default=0, help="Cap the work.")
    args = ap.parse_args()

    index = SignIndex.load()
    if not len(index):
        print("✗ No index. Run scripts/build_index.py first.")
        return 1

    targets = []
    if args.all:
        targets = list(index.entries)
    else:
        words = list(args.words or [])
        if args.common:
            words += COMMON[: args.common]
        seen = set()
        for w in words:
            res = index.get(w)
            if res.entry and res.entry.word not in seen:
                seen.add(res.entry.word)
                targets.append(res.entry)

    if args.limit:
        targets = targets[: args.limit]

    if targets:
        print(f"→ Downloading {len(targets)} clips…")
        ok = skip = fail = 0
        for i, entry in enumerate(targets, 1):
            if entry.is_cached:
                skip += 1
                continue
            try:
                drive.download(entry.file_id, entry.local_path)
                ok += 1
                print(f"  [{i}/{len(targets)}] {entry.word}")
            except Exception as exc:
                fail += 1
                print(f"  [{i}/{len(targets)}] {entry.word} — failed: {exc}")
        print(f"✓ {ok} downloaded, {skip} already there, {fail} failed")

    if args.crops:
        known = load_crops()
        todo = [e for e in index.entries if e.is_cached and e.word not in known]
        if args.limit:
            todo = todo[: args.limit]
        print(f"\n-> Finding the signer in {len(todo)} clips...")
        done = 0
        for i, entry in enumerate(todo, 1):
            try:
                box = scan_clip(entry.local_path)
                if box:
                    save_crop(entry.word, box)
                    done += 1
                    print(f"  [{i}/{len(todo)}] {entry.word}")
                else:
                    print(f"  [{i}/{len(todo)}] {entry.word} - no signer found")
            except Exception as exc:
                print(f"  [{i}/{len(todo)}] {entry.word} - {exc}")
        print(f"OK framed {done} clips")

    if args.poses:
        cached = [e for e in index.entries if e.is_cached and not has_pose(e.word)]
        if args.limit:
            cached = cached[: args.limit]
        print(f"\n→ Extracting motion for {len(cached)} clips…")
        done = 0
        for i, entry in enumerate(cached, 1):
            try:
                if extract_pose(entry.local_path, entry.word):
                    done += 1
                    print(f"  [{i}/{len(cached)}] {entry.word}")
                else:
                    print(f"  [{i}/{len(cached)}] {entry.word} — no signer detected")
            except Exception as exc:
                print(f"  [{i}/{len(cached)}] {entry.word} — {exc}")
        print(f"✓ motion extracted for {done} signs")

    if not targets and not args.poses and not args.crops:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
