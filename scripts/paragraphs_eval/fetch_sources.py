"""
The source text of each lesson of a set (what the reader reads), for the evaluation (TASK-paragraphs.md, 2026-11-05):
the editor's transcript has no "קריין:" before most readings, so without the source they count as the Rav's words.

Per lesson (kmedia backend, public): the unit's `sources` (SOURCE units: Hebrew .docx, else the .pdf) and its derived
LIKUTIM units (Hebrew .docx); every Hebrew text file found, joined, -> <dir>/sources/<id>.txt, and
<dir>/sources/index.json (which files, per lesson). A lesson with none -> no file.
The LIKUTIM of a lesson is sometimes another part's: evaluate.source_passages only marks passages that are in
the transcript too, so a wrong source marks (almost) nothing -- `coverage` in methods.py shows it.

PDF (old TES parts have no .docx): needs pypdf (not in requirements; run with a venv that has it). Their glyphs are
ISO-8859-8 codes shown as /Cnnn or StandardEncoding names and the text has no spaces: decoded here and split into
words by the vocabulary of the lesson's transcript and of the other sources (dynamic programming, unknown letters cost
more), which is enough for the n-gram match.

  python3 scripts/paragraphs_eval/fetch_sources.py --dir D --set test   # + --set tune; ids as in methods.py
  (a lesson in the set file may be a kmedia file folder name: --alias YSTvODeO=DidKWBE3 maps it to its unit)
"""
import argparse
import io
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from ai_trns.paragraphs import build  # noqa: E402

BACKEND = "https://kabbalahmedia.info/backend"
CDN = "https://cdn.kabbalahmedia.info"
STD = {"Æ": 0xE1, "ª": 0xE3, "Ł": 0xE8, "Ø": 0xE9, "Œ": 0xEA, "º": 0xEB, "æ": 0xF1, "ı": 0xF5, "ł": 0xF8, "ø": 0xF9,
       "œ": 0xFA, "ß": 0xFB}


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def unit_files(uid):
    units = get_json(f"{BACKEND}/content_units?id={uid}&with_files=true")["content_units"]
    return units[0].get("files", []) if units else []


def text_files(uid):
    """[(file id, name)]: the Hebrew text files of a lesson's sources and LIKUTIM, .docx preferred over .pdf per unit"""
    unit = get_json(f"{BACKEND}/content_units/{uid}")
    out = []
    groups = [unit_files(s) for s in unit.get("sources") or []]
    groups += [d.get("files", []) for d in (unit.get("derived_units") or {}).values() if d.get("content_type") == "LIKUTIM"]
    groups.append([f for f in unit.get("files", []) if f.get("insert_type") == "likutim"])
    for files in groups:
        he = [f for f in files if f.get("language") == "he" and f.get("type") == "text"
              and f.get("insert_type") != "source-scan"]
        docx = [f for f in he if f["name"].lower().endswith(".docx")]
        pdf = [f for f in he if f["name"].lower().endswith(".pdf")]
        out += [(f["id"], f["name"]) for f in (docx or pdf[:1])]
    return out


def download(fid, path):
    if not os.path.exists(path):
        with urllib.request.urlopen(f"{CDN}/{fid}", timeout=120) as r, open(path, "wb") as f:
            f.write(r.read())
    return path


def docx_text(path):
    import docx  # as ai_trns.api.diarize.source_text
    return "\n".join(p.text for p in docx.Document(path).paragraphs)


def pdf_letters(path):
    from pypdf import PdfReader
    text = "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
    heb = lambda c: bytes([c]).decode("iso-8859-8") if 0xE0 <= c <= 0xFA else ""
    text = re.sub(r"/C(\d{3})", lambda m: heb(int(m.group(1))), text)
    return "".join(heb(STD[c]) if c in STD else c for c in text)


def segment(line, vocab, longest):
    """Letters without spaces -> words of vocab (an unknown letter costs 3, a word 1); keeps the letters' order"""
    s = build.norm(line)
    n = len(s)
    best = [0.0] + [float("inf")] * n
    back = [0] * (n + 1)
    for i in range(1, n + 1):
        best[i], back[i] = best[i - 1] + 3, i - 1
        for j in range(max(0, i - longest), i):
            if s[j:i] in vocab and best[j] + 1 < best[i]:
                best[i], back[i] = best[j] + 1, j
    words, i = [], n
    while i > 0:
        words.append(s[back[i]:i])
        i = back[i]
    return " ".join(reversed(words))


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
        files = text_files(alias.get(lid, lid))
        found[lid] = [{"id": fid, "name": name} for fid, name in files]
        print(lid, [name for _, name in files], flush=True)
    texts = {}
    for lid, files in found.items():
        parts = []
        for f in files:
            ext = f["name"].rsplit(".", 1)[-1].lower()
            path = download(f["id"], os.path.join(dl, f"{f['id']}.{ext}"))
            if ext == "docx":
                parts.append(docx_text(path))
            else:
                pdfs.setdefault(lid, []).append(pdf_letters(path))
        texts[lid] = parts
    if pdfs:
        vocab = set()
        for parts in texts.values():
            for p in parts:
                vocab.update(build.norm(w) for w in p.split())
        for lid in pdfs:
            doc = os.path.join(args.dir, "docs", lid + ".txt")
            if os.path.exists(doc):
                vocab.update(build.norm(w) for w in open(doc, encoding="utf-8").read().split())
        vocab.discard("")
        longest = max(map(len, vocab))
        for lid, letters in pdfs.items():
            texts[lid] += ["\n".join(segment(line, vocab, longest) for line in t.split("\n")) for t in letters]
    for lid, parts in texts.items():
        if parts:
            with open(os.path.join(out_dir, lid + ".txt"), "w", encoding="utf-8") as f:
                f.write("\n\n".join(parts))
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(found, f, ensure_ascii=False, indent=1)
    print(f"{sum(1 for p in texts.values() if p)} of {len(ids)} lessons have a source text")


if __name__ == "__main__":
    main()
