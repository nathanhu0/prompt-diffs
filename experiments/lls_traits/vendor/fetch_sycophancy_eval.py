"""Fetch + checksum-verify the SycophancyEval source data.

The prepared probe set (experiments/lls_traits/data/are_you_sure.json) was built
from this file, but the source itself was never kept, so the prep step could not
be re-run. This restores that: it downloads to the mirror directory and refuses
to proceed on a checksum mismatch, so a silent upstream edit can't slip in.

  PYTHONPATH=. uv run python \
    experiments/lls_traits/vendor/fetch_sycophancy_eval.py [--force]
"""
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))          # repo root

from experiments.lls_traits.vendor.sycophancy_eval import (
    ARE_YOU_SURE_SHA256, RAW_URL)

MIRROR = Path("/nlp/scr/nathu/latent_rewrite/vendor/sycophancy_eval")
DEST = MIRROR / "are_you_sure.jsonl"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="re-download even if the mirror already verifies")
    args = ap.parse_args()

    MIRROR.mkdir(parents=True, exist_ok=True)
    if DEST.exists() and not args.force:
        got = sha256(DEST)
        if got == ARE_YOU_SURE_SHA256:
            print(f"{DEST} already present and verified ({got[:12]}...)")
            return
        print(f"checksum mismatch on the existing mirror:\n  want {ARE_YOU_SURE_SHA256}"
              f"\n  got  {got}\nre-downloading", file=sys.stderr)

    print(f"downloading {RAW_URL}")
    with urllib.request.urlopen(RAW_URL, timeout=120) as r:
        DEST.write_bytes(r.read())
    got = sha256(DEST)
    if got != ARE_YOU_SURE_SHA256:
        raise SystemExit(
            f"REFUSING: checksum mismatch after download\n"
            f"  want {ARE_YOU_SURE_SHA256}\n  got  {got}\n"
            f"Upstream changed. Inspect the diff before pinning a new hash — the "
            f"prepared probe set at experiments/lls_traits/data/ was built from "
            f"the old file and would no longer correspond to it.")
    n = sum(1 for _ in DEST.open())
    print(f"wrote {DEST}  ({n} records, sha256 verified)")


if __name__ == "__main__":
    main()
