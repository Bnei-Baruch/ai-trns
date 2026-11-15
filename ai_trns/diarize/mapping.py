"""Model-free glue between diarization turns and SRT cues.

turns: [(start, end, cluster), ...] -- pyannote's (exclusive) speaker turns,
cluster being its raw label ("SPEAKER_00", ...).
"""
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


def overlaps(start: float, end: float, turns: list[tuple]) -> dict:
    """Seconds of [start, end] covered by each cluster."""
    result = {}
    for t_start, t_end, cluster in turns:
        d = min(end, t_end) - max(start, t_start)
        if d > 0:
            result[cluster] = result.get(cluster, 0.0) + d
    return result


def nearest_cluster(start: float, end: float, turns: list[tuple]):
    """Cluster of the turn closest in time to [start, end] (None if no turns)."""
    best, best_gap = None, None
    for t_start, t_end, cluster in turns:
        gap = max(t_start - end, start - t_end, 0.0)
        if best_gap is None or gap < best_gap:
            best, best_gap = cluster, gap
    return best


def span_cluster(start: float, end: float, turns: list[tuple]):
    """Cluster with the most overlap with [start, end], else the nearest one."""
    ov = overlaps(start, end, turns)
    if ov:
        return max(ov, key=ov.get)
    return nearest_cluster(start, end, turns)


def dominant_cluster(start: float, end: float, turns: list[tuple], dominant_ratio: float):
    """(cluster, is_mixed) for a cue: is_mixed when no single cluster covers
    at least dominant_ratio of the cue's diarized speech."""
    ov = overlaps(start, end, turns)
    if not ov:
        return nearest_cluster(start, end, turns), False
    top = max(ov, key=ov.get)
    return top, ov[top] / sum(ov.values()) < dominant_ratio


def smooth_runs(words: list[dict], clusters: list, min_piece_words: int, min_piece_s: float) -> list[list[int]]:
    """Groups consecutive same-cluster words into runs (lists of word indices),
    then repeatedly folds the shortest run that is below min_piece_words
    words or min_piece_s seconds into its longer neighbour."""
    runs = []
    for i, c in enumerate(clusters):
        if runs and clusters[runs[-1][0]] == c:
            runs[-1].append(i)
        else:
            runs.append([i])

    def duration(run):
        return words[run[-1]]["end"] - words[run[0]]["start"]

    def too_short(run):
        return len(run) < min_piece_words or duration(run) < min_piece_s

    while len(runs) > 1:
        short = [k for k, r in enumerate(runs) if too_short(r)]
        if not short:
            break
        k = min(short, key=lambda k: (len(runs[k]), duration(runs[k])))
        if k == 0:
            target = 1
        elif k == len(runs) - 1:
            target = k - 1
        else:
            target = k - 1 if len(runs[k - 1]) >= len(runs[k + 1]) else k + 1
        lo, hi = min(k, target), max(k, target)
        runs[lo:hi + 1] = [runs[lo] + runs[hi]]
        # merging may have joined two runs of the same cluster around it
        merged = []
        for r in runs:
            if merged and clusters[merged[-1][0]] == clusters[r[0]]:
                merged[-1] = merged[-1] + r
            else:
                merged.append(r)
        runs = merged
    return runs


def word_runs(words: list[dict], turns: list[tuple], min_piece_words: int, min_piece_s: float) -> list[tuple[list[int], object]]:
    """Speaker runs of a mixed cue's words ([{"word", "start", "end"}] in
    absolute time): each word gets the cluster covering it, short runs are
    folded into neighbours (smooth_runs). Returns [(word indices, cluster)];
    a run's cluster is the one most of its words got (ties -> first seen)."""
    clusters = [span_cluster(w["start"], w["end"], turns) for w in words]
    result = []
    for run in smooth_runs(words, clusters, min_piece_words, min_piece_s):
        votes = {}
        for i in run:
            votes[clusters[i]] = votes.get(clusters[i], 0) + 1
        result.append((run, max(votes, key=votes.get)))
    return result


def split_cue(cue: dict, words: list[dict], turns: list[tuple], min_piece_words: int, min_piece_s: float) -> list[dict]:
    """Splits a mixed cue at speaker changes (see word_runs). Returns
    [{"start", "end", "text", "cluster"}]."""
    if not words:
        cluster = span_cluster(cue["start"], cue["end"], turns)
        return [{**cue, "cluster": cluster}]

    runs = word_runs(words, turns, min_piece_words, min_piece_s)
    pieces = []
    for n, (run, cluster) in enumerate(runs):
        start = cue["start"] if n == 0 else max(cue["start"], words[run[0]]["start"])
        end = cue["end"] if n == len(runs) - 1 else min(cue["end"], words[run[-1]]["end"])
        pieces.append({
            "start": start,
            "end": end,
            "text": " ".join(words[i]["word"] for i in run),
            "cluster": cluster,
        })
    return pieces


def match_samples(similarity: np.ndarray, threshold: float,
                  extra_threshold: Optional[float] = None) -> dict[int, int]:
    """similarity: [n_samples, n_clusters] cosine similarities. One-to-one
    assignment maximizing total similarity; pairs below threshold dropped.
    extra_threshold (None = off): then every cluster left unassigned goes to
    its most similar sample too, if that sample got a cluster of its own and
    the similarity is at least extra_threshold -- one voice that diarization
    split into several clusters gets all of them.
    Returns {cluster_idx: sample_idx}."""
    if similarity.size == 0:
        return {}
    rows, cols = linear_sum_assignment(-similarity)
    matched = {int(c): int(r) for r, c in zip(rows, cols) if similarity[r, c] >= threshold}
    if extra_threshold is None:
        return matched
    named = set(matched.values())
    for c in range(similarity.shape[1]):
        if c in matched:
            continue
        r = int(np.argmax(similarity[:, c]))
        if r in named and similarity[r, c] >= extra_threshold:
            matched[c] = r
    return matched


def voice_report(speakers: list[str], similarity: np.ndarray, matched: dict[int, int], threshold: float,
                 margin: float = 0.1) -> list[dict]:
    """How sure the naming is, per speaker with voice samples: [{"speaker", "similarity": of its cluster (the
    closest one if none was named), "named", "next": the closest cluster of nobody else, "uncertain"}].
    Uncertain: named or not by less than margin, or another cluster would pass the threshold too (one voice
    that diarization split in two: the other half stays anonymous). matched: match_samples' answer"""
    report = []
    for s, name in enumerate(speakers):
        row = similarity[s]
        if not row.size:
            continue
        own = [c for c, r in matched.items() if r == s] or [int(np.argmax(row))]
        named = any(matched.get(c) == s for c in own)
        best = float(max(row[c] for c in own))
        free = [float(row[c]) for c in range(row.size) if c not in own and c not in matched]
        nxt = max(free) if free else None
        uncertain = (best < threshold + margin) if named else (best >= threshold - margin)
        report.append({"speaker": name, "similarity": round(best, 3), "named": named,
                       "next": None if nxt is None else round(nxt, 3),
                       "uncertain": bool(uncertain or (named and nxt is not None and nxt >= threshold))})
    return report


def best_per_speaker(names: list[str], similarity: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Collapses several samples of the same speaker into one row: a
    speaker's similarity to a cluster is that of its closest sample.
    names: one per similarity row. Returns (unique names in first-seen
    order, [n_speakers, n_clusters])."""
    speakers = list(dict.fromkeys(names))
    if not speakers:
        return [], similarity[:0]
    rows = [similarity[[i for i, n in enumerate(names) if n == s]].max(axis=0) for s in speakers]
    return speakers, np.stack(rows)


def speaker_names(clusters_in_order: list, named: dict) -> dict:
    """named: {cluster: name} for matched clusters. Every other cluster gets
    SPEAKER_NN, numbered by first appearance."""
    names, n = {}, 0
    for c in clusters_in_order:
        if c in names:
            continue
        if c in named:
            names[c] = named[c]
        else:
            names[c] = f"SPEAKER_{n:02d}"
            n += 1
    return names


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """[n, d] x [m, d] -> [n, m]; rows with nan or zero norm give -1."""
    a = np.nan_to_num(np.asarray(a, dtype=np.float64))
    b = np.nan_to_num(np.asarray(b, dtype=np.float64))
    na = np.linalg.norm(a, axis=1, keepdims=True)
    nb = np.linalg.norm(b, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = (a / na) @ (b / nb).T
    return np.where(np.isfinite(sim), sim, -1.0)
