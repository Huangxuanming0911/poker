"""Fetch Byron Knoll's public-domain SVG playing-card deck to static/img/cards/.

Source: https://github.com/notpeter/Vector-Playing-Cards (cards-svg/ directory)
License: Public domain.

Run once after cloning:
    python scripts/download_cards.py

The repo uses short filenames like "AS.svg" (ace of spades), "10D.svg" (ten of
diamonds), "JC.svg" (jack of clubs). We mirror that naming so the frontend's
card.js can reference files directly without any lookup table.

The card back is not included in the upstream repo, so the frontend always
renders the back as a CSS pattern.
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request

BASE = "https://raw.githubusercontent.com/notpeter/Vector-Playing-Cards/master/cards-svg/"
RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
SUITS = ["S", "H", "D", "C"]


def main() -> int:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dest = os.path.normpath(os.path.join(script_dir, "..", "static", "img", "cards"))
    os.makedirs(dest, exist_ok=True)
    print(f"Downloading SVG cards into {dest} ...")
    failed = []
    for rank in RANKS:
        for suit in SUITS:
            name = f"{rank}{suit}.svg"
            path = os.path.join(dest, name)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                continue
            url = BASE + name
            try:
                urllib.request.urlretrieve(url, path)
                print(f"  {name} ✓")
            except (urllib.error.URLError, urllib.error.HTTPError) as exc:
                failed.append((name, str(exc)))
                print(f"  {name} ✗ ({exc})")
    if failed:
        print(f"\n{len(failed)} downloads failed; the SVG mode will fall back to CSS for those cards.")
        return 1
    print(f"\nAll {len(RANKS) * len(SUITS)} cards present at {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
