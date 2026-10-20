"""
Word-level diff of a reference text against heard words (moved from stt-api tadiff/diff.py).
Pure logic, stdlib only.

Both sides are normalized the same way (niqqud, quotes, punctuation stripped) before
comparison; indices in the result always refer to the ORIGINAL text's whitespace-split
tokens, so tokens that normalize to empty (bare punctuation) are skipped for comparison
but never shift indices.

A global SequenceMatcher pass finds exact-equal anchors; each non-equal `replace` block is
then re-aligned with a small edit-distance DP in which words at or above `fuzzy_threshold`
character similarity count as equal (so spelling variants like ktiv male/haser don't show up
as changes). Every maximal run of non-equal ops between two equal words becomes one range:
text-only -> removed, heard-only -> added, both -> changed.

    {"removed": [{"text_idx": [a, b], "text", "at"}],
     "added":   [{"start", "end", "heard", "after_text_idx"}],
     "changed": [{"text_idx": [a, b], "text", "start", "end", "heard"}],
     "stats":   {"text_words", "audio_words"}}
"""
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

FUZZY_THRESHOLD = 0.8
# Above this many cells, a replace block skips the fuzzy DP and stays one change
_MAX_DP_CELLS = 1_000_000

_NIQQUD_RE = re.compile(r"[֑-ׇ]")
_QUOTE_RE = re.compile(r"[\"'׳״‘’“”]")
_PUNCT_RE = re.compile(r"[,.;:!?()]")


@dataclass
class Word:
    text: str
    start: float
    end: float


def words_of(doc: dict) -> list[Word]:
    """Heard words of a words document (ai_trns.words), with the engine's own times"""
    return [Word(w["word"], w["start_s"], w["end_s"]) for w in doc["words"]]


def load_words(data) -> list[Word]:
    """JSON array of {"word" (or "text"), "start", "end"} in heard order -> words; ValueError if malformed"""
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in timestamps file: {e}") from e
    if not isinstance(parsed, list):
        raise ValueError("Timestamps file must be a JSON array of {\"word\", \"start\", \"end\"} objects")

    words = []
    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"Timestamps entry {i} is not an object")
        text = item.get("word", item.get("text"))
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Timestamps entry {i} is missing a non-empty \"word\"")
        try:
            start, end = float(item["start"]), float(item["end"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Timestamps entry {i} has invalid \"start\"/\"end\": {e}") from e
        if start > end:
            raise ValueError(f"Timestamps entry {i} has start ({start}) after end ({end})")
        words.append(Word(text, start, end))
    return words


def _norm(token: str) -> str:
    """Niqqud, quotes and punctuation stripped; a token with no letters/digits left (dash, maqaf...) is not a word"""
    norm = _PUNCT_RE.sub("", _QUOTE_RE.sub("", _NIQQUD_RE.sub("", token)))
    return norm if any(c.isalnum() for c in norm) else ""


def _similar(a: str, b: str, threshold: float) -> bool:
    return a == b or SequenceMatcher(None, a, b).ratio() >= threshold


def _fuzzy_ops(ref: list[str], hyp: list[str], i0: int, j0: int, threshold: float) -> list[tuple]:
    """Edit-distance alignment of one replace block -> ops with absolute indices"""
    n, m = len(ref), len(hyp)
    if n * m > _MAX_DP_CELLS:
        return [("del", i0 + i) for i in range(n)] + [("ins", j0 + j) for j in range(m)]

    sim = [[_similar(ref[i], hyp[j], threshold) for j in range(m)] for i in range(n)]
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i
    for j in range(1, m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(
                cost[i - 1][j - 1] + (0 if sim[i - 1][j - 1] else 1),
                cost[i - 1][j] + 1,
                cost[i][j - 1] + 1,
            )

    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + (0 if sim[i - 1][j - 1] else 1):
            ops.append(("eq" if sim[i - 1][j - 1] else "sub", i0 + i - 1, j0 + j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            ops.append(("del", i0 + i - 1))
            i -= 1
        else:
            ops.append(("ins", j0 + j - 1))
            j -= 1
    ops.reverse()
    return ops


def _word_ops(ref: list[str], hyp: list[str], threshold: float) -> list[tuple]:
    ops = []
    matcher = SequenceMatcher(None, ref, hyp, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            ops.extend(("eq", i1 + k, j1 + k) for k in range(i2 - i1))
        elif tag == "delete":
            ops.extend(("del", i) for i in range(i1, i2))
        elif tag == "insert":
            ops.extend(("ins", j) for j in range(j1, j2))
        else:
            ops.extend(_fuzzy_ops(ref[i1:i2], hyp[j1:j2], i1, j1, threshold))
    return ops


def compare(text: str, words: list[Word], fuzzy_threshold: float = FUZZY_THRESHOLD) -> dict:
    tokens = text.split()
    ref_orig_idx, ref = [], []
    for idx, tok in enumerate(tokens):
        norm = _norm(tok)
        if norm:
            ref_orig_idx.append(idx)
            ref.append(norm)

    heard, hyp = [], []
    for w in words:
        norm = _norm(w.text)
        if norm:
            heard.append(w)
            hyp.append(norm)

    result = {"removed": [], "added": [], "changed": []}
    last_ref, last_hyp = -1, -1  # last equal word seen, for positioning removed/added ranges
    run_ref: list[int] = []
    run_hyp: list[int] = []

    def flush():
        if not run_ref and not run_hyp:
            return
        entry = {}
        if run_ref:
            a, b = ref_orig_idx[run_ref[0]], ref_orig_idx[run_ref[-1]]
            entry["text_idx"] = [a, b]
            entry["text"] = " ".join(tokens[a:b + 1])
        if run_hyp:
            entry["start"] = round(heard[run_hyp[0]].start, 3)
            entry["end"] = round(heard[run_hyp[-1]].end, 3)
            entry["heard"] = " ".join(heard[j].text for j in run_hyp)
        if run_ref and run_hyp:
            result["changed"].append(entry)
        elif run_ref:
            # where in the audio the missing words should have been
            entry["at"] = round(heard[last_hyp].end, 3) if last_hyp >= 0 else 0.0
            result["removed"].append(entry)
        else:
            # which input text word the extra audio follows (-1 = before the first)
            entry["after_text_idx"] = ref_orig_idx[last_ref] if last_ref >= 0 else -1
            result["added"].append(entry)
        run_ref.clear()
        run_hyp.clear()

    for op in _word_ops(ref, hyp, fuzzy_threshold):
        kind = op[0]
        if kind == "eq":
            flush()
            last_ref, last_hyp = op[1], op[2]
        elif kind == "del":
            run_ref.append(op[1])
        elif kind == "ins":
            run_hyp.append(op[1])
        else:
            run_ref.append(op[1])
            run_hyp.append(op[2])
    flush()

    result["stats"] = {"text_words": len(ref), "audio_words": len(hyp)}
    return result
