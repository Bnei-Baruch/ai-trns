"""
The source text of each lesson of a set (what the reader reads), for the evaluation (TASK-paragraphs.md, 2026-11-05):
the editor's transcript has no "קריין:" before most readings, so without the source they count as the Rav's words.

Per lesson (kmedia backend, public): the unit's `sources` (SOURCE units: Hebrew .docx, else the .pdf) and its derived
LIKUTIM units (Hebrew .docx); every Hebrew text file found, joined, -> <dir>/sources/<id>.txt, and
<dir>/sources/index.json (which files, per lesson). A lesson with none -> no file.
The LIKUTIM of a lesson is sometimes another part's: evaluate.source_passages only marks passages that are in
the transcript too, so a wrong source marks (almost) nothing -- `coverage` in methods.py shows it.

Finding and reading the files: ai_trns.sources (as the API does for /diarize/* `uid`). PDF (old TES parts have no
.docx): needs pypdf (requirements-api.txt); split into words by the vocabulary of the lesson's transcript and of all
the set's sources. Extracted texts are kept in <dir>/sources/files/<file id>.txt.

  python3 scripts/paragraphs_eval/fetch_sources.py --dir D --set test   # + --set tune; ids as in methods.py
  (a lesson in the set file may be a kmedia file folder name: --alias YSTvODeO=DidKWBE3 maps it to its unit)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from ai_trns import sources  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--set", action="append", required=True)
    ap.add_argument("--alias", action="append", default=[], help="<id in the set>=<kmedia unit id>")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    from methods import read_set
    alias = dict(a.split("=", 1) for a in args.alias)
    ids = [i for s in args.set for i in read_set(os.path.join(here, s + ".txt"))]
    dl, out_dir = os.path.join(args.dir, "sources", "files"), os.path.join(args.dir, "sources")
    os.makedirs(dl, exist_ok=True)
    found, pdfs = {}, {}
    for lid in ids:
        files = sources.text_files(alias.get(lid, lid), timeout=60)
        found[lid] = [{"id": f["id"], "name": f["name"]} for f in files]
        print(lid, [f["name"] for f in files], flush=True)
    cache = sources.Cache(dl, max_mb=None, max_days=None)
    texts = {}
    for lid, files in found.items():
        parts = []
        for f in files:
            text = sources.file_text(f, cache, timeout=120)
            if sources.is_pdf(f["name"]):
                pdfs.setdefault(lid, []).append(text)
            else:
                parts.append(text)
        texts[lid] = parts
    if pdfs:
        docs = [open(p, encoding="utf-8").read() for p in (os.path.join(args.dir, "docs", lid + ".txt") for lid in pdfs)
                if os.path.exists(p)]
        vocab = sources.vocabulary([p for parts in texts.values() for p in parts] + docs)
        for lid, letters in pdfs.items():
            texts[lid] += [sources.segment_text(t, vocab) for t in letters]
    for lid, parts in texts.items():
        if parts:
            with open(os.path.join(out_dir, lid + ".txt"), "w", encoding="utf-8") as f:
                f.write("\n\n".join(parts))
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(found, f, ensure_ascii=False, indent=1)
    print(f"{sum(1 for p in texts.values() if p)} of {len(ids)} lessons have a source text")


if __name__ == "__main__":
    main()
