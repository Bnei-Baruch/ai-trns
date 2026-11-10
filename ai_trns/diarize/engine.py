"""
Worker engine "diarize" (moved from stt-api diarize/service.py, diarize/vad_align.py,
tadiff/transcribe/ctc_align.py): audio + cues -> diarization document (ai_trns.diarize).

1. pyannote on the whole file: speaker turns, each in an anonymous cluster.
2. Voice samples of known speakers (built-in ones in SPEAKERS_DIR/<Name>/*, plus any sent with
   the job) are embedded with pyannote's own embedding model and matched one-to-one to the
   clusters by cosine similarity (optionally a speaker also takes the other clusters close enough to it,
   AI_TRNS_DIARIZE_EXTRA_THRESHOLD); the rest stay SPEAKER_NN.
3. Every word of a cue is force-aligned with CTC (Meta MMS_FA via torchaudio, text romanized
   by uroman); if that fails, the cue's words are spread over its VAD speech regions.
4. A cue one speaker dominates gives all its words that speaker; a mixed one labels its words
   by speaker run (mapping.word_runs).

Needs requirements-diarize.txt and a HuggingFace token with access to the pyannote pipeline.
"""
import logging
import os
import time
from typing import Callable, Optional

import numpy as np

from .. import audio
from . import KIND, VERSION, mapping

log = logging.getLogger("ai-trns.diarize")

SR = 16000
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac"}
PIPELINE = "pyannote/speaker-diarization-community-1"
# min cosine similarity sample <-> cluster to name the cluster
MATCH_THRESHOLD = 0.5
# a speaker already matched also names every other cluster most similar to it at or above this
# (mapping.match_samples); None = one cluster per speaker
MATCH_EXTRA_THRESHOLD = None
# share of a cue one speaker must cover to label it whole
DOMINANT_RATIO = 0.85
# smaller speaker runs inside a cue are merged into a neighbour
MIN_PIECE_WORDS = 2
MIN_PIECE_S = 0.6
# samples are embedded in chunks of this length and averaged
SAMPLE_CHUNK_S = 10
# VAD fallback
MIN_SILENCE_DURATION_MS = 100
# uroman language of a job language
UROMAN_LANGS = {"he": "heb", "ru": "rus", "en": "eng"}
# share of the engine's progress taken by pyannote; the rest is aligning cues
DIARIZE_SHARE = 0.5
PYANNOTE_STEPS = ("segmentation", "speaker_counting", "embeddings", "discrete_diarization")


def builtin_samples(speakers_dir: str) -> list[tuple[str, str]]:
    """[(name, path)] for the voice samples in speakers_dir: a subdirectory per speaker (its name =
    the label, every audio file inside = one sample of that speaker), or a single audio file
    directly in the dir (its name without extension = the label)"""
    if not os.path.isdir(speakers_dir):
        return []

    def is_audio(path):
        return os.path.isfile(path) and os.path.splitext(path)[1].lower() in AUDIO_EXTS

    result = []
    for entry in os.listdir(speakers_dir):
        path = os.path.join(speakers_dir, entry)
        if os.path.isdir(path):
            result.extend((entry, os.path.join(path, f)) for f in os.listdir(path) if is_audio(os.path.join(path, f)))
        elif is_audio(path):
            result.append((os.path.splitext(entry)[0], path))
    return sorted(result)


# --- VAD fallback: no boundary per word, only where speech is. Each speech region gets a share of
# the words proportional to its duration, spread evenly over it, so no word straddles a silence.

def spread_evenly(words: list[str], start: float, end: float) -> list[dict]:
    step = (end - start) / len(words)
    return [{"word": w, "start": start + i * step, "end": start + (i + 1) * step} for i, w in enumerate(words)]


def apportion(total: int, weights: list[float]) -> list[int]:
    """total split into len(weights) non-negative integers proportional to weights, summing to
    exactly total (largest-remainder method)"""
    weight_sum = sum(weights)
    raw = [total * w / weight_sum for w in weights]
    counts = [int(r) for r in raw]
    order = sorted(range(len(weights)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in order[:total - sum(counts)]:
        counts[i] += 1
    return counts


def distribute_over_regions(words: list[str], regions: list[tuple[float, float]],
                            start: float, end: float) -> list[dict]:
    durations = [r_end - r_start for r_start, r_end in regions]
    if not durations or sum(durations) <= 0:
        return spread_evenly(words, start, end)
    result, idx = [], 0
    for (r_start, r_end), k in zip(regions, apportion(len(words), durations)):
        if k:
            result.extend(spread_evenly(words[idx:idx + k], r_start, r_end))
            idx += k
    return result


def _vad_words(window: np.ndarray, text: str) -> list[dict]:
    """Word times in the window from VAD speech regions (faster-whisper's Silero)"""
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    words = text.split()
    duration = len(window) / SR
    if not len(window):
        return spread_evenly(words, 0.0, duration)
    options = VadOptions(min_silence_duration_ms=MIN_SILENCE_DURATION_MS, min_speech_duration_ms=250,
                         speech_pad_ms=30)
    regions = [(r["start"] / SR, r["end"] / SR) for r in get_speech_timestamps(window, options, SR)]
    return distribute_over_regions(words, regions, 0.0, duration)


class Diarizer:
    """pyannote pipeline, MMS_FA aligner and uroman, loaded once; parameters are passed per call"""

    def __init__(self, speakers_dir: str, device: str = "cuda", pipeline: str = PIPELINE,
                 hf_token: Optional[str] = None, extra_threshold: Optional[float] = MATCH_EXTRA_THRESHOLD,
                 step: Optional[float] = None):
        import torch
        import torchaudio
        import uroman
        from pyannote.audio import Pipeline

        self.torch = torch
        self.speakers_dir = speakers_dir
        self.extra_threshold = extra_threshold
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        try:
            self.pipeline = Pipeline.from_pretrained(pipeline, token=hf_token)  # pyannote >= 4
        except TypeError:
            self.pipeline = Pipeline.from_pretrained(pipeline, use_auth_token=hf_token)  # pyannote 3.x
        if self.pipeline is None:
            raise RuntimeError(f"Could not load {pipeline}: check HF_TOKEN and that its license is accepted on HF")
        self.pipeline.to(self.device)
        self.pipeline_name = pipeline
        if step:
            self._set_step(step)

        bundle = torchaudio.pipelines.MMS_FA
        self.mms = bundle.get_model(with_star=False).to(self.device)
        self.tokenizer = bundle.get_tokenizer()
        self.aligner = bundle.get_aligner()
        # MMS_FA's dictionary; index 0 is the CTC blank, spelled "-"
        self.mms_chars = set(bundle.get_dict()) - {"-"}
        self.uroman = uroman.Uroman()
        self.builtin_embeddings = {}

    # --- speakers

    def _set_step(self, step: float):
        """pyannote slides a window over the audio and embeds every speaker in every position: nearly all of its
        time. step: the slide as a share of the window (the pipeline's own: 0.1); twice the step = half the time"""
        inference = getattr(self.pipeline, "_segmentation", None)
        if inference is None or not hasattr(inference, "step") or not hasattr(inference, "duration"):
            raise RuntimeError("this pyannote pipeline has no segmentation step to set (AI_TRNS_DIARIZE_STEP)")
        if not 0 < step <= 0.5:
            raise ValueError("AI_TRNS_DIARIZE_STEP: a share of the window, above 0 and up to 0.5")
        inference.step = step * inference.duration
        log.info("segmentation step %.2f of the window (%.1f s)", step, inference.step)

    def _diarize(self, waveform, speaker_kwargs: dict, hook):
        """-> (turns, cluster_labels, centroids) where centroids[i] belongs to cluster_labels[i]"""
        import pyannote.audio
        file = {"waveform": waveform[None], "sample_rate": SR}
        if int(pyannote.audio.__version__.split(".")[0]) >= 4:
            out = self.pipeline(file, hook=hook, **speaker_kwargs)
            annotation, exclusive, centroids = out.speaker_diarization, out.exclusive_speaker_diarization, out.speaker_embeddings
        else:
            annotation, centroids = self.pipeline(file, return_embeddings=True, hook=hook, **speaker_kwargs)
            exclusive = annotation
        turns = sorted((seg.start, seg.end, label) for seg, _, label in exclusive.itertracks(yield_label=True))
        return turns, list(annotation.labels()), centroids

    def _embed_sample(self, path: str) -> np.ndarray:
        """Mean speaker embedding of a voice sample, embedded in SAMPLE_CHUNK_S pieces"""
        waveform = self.torch.from_numpy(audio.load_pcm(path).copy())
        chunk = int(SAMPLE_CHUNK_S * SR)
        pieces = [waveform[i:i + chunk] for i in range(0, waveform.shape[0], chunk)]
        # a short tail gives a noisy embedding: drop it unless it's all we have
        pieces = [p for p in pieces if p.shape[0] >= 2 * SR] or pieces[:1]
        embs = [self.pipeline._embedding(p[None, None])[0] for p in pieces]
        embs = [e for e in embs if np.all(np.isfinite(e))]
        if not embs:
            raise ValueError(f"voice sample {os.path.basename(path)} is too short or silent to embed")
        return np.mean(embs, axis=0)

    def _sample_embeddings(self, samples: list[tuple[str, str]], builtin_paths: set):
        """-> (names, [n, d] embeddings). Built-in samples are embedded once (keyed by path + mtime);
        one that can't be embedded is skipped with a warning instead of failing every job"""
        names, embs = [], []
        for name, path in samples:
            if path in builtin_paths:
                key = (path, os.path.getmtime(path))
                if key not in self.builtin_embeddings:
                    try:
                        self.builtin_embeddings[key] = self._embed_sample(path)
                    except ValueError as e:
                        log.warning("built-in speaker sample %s skipped: %s", path, e)
                        continue
                emb = self.builtin_embeddings[key]
            else:
                emb = self._embed_sample(path)
            names.append(name)
            embs.append(emb)
        return names, np.stack(embs) if embs else np.zeros((0, 0))

    def _name_clusters(self, labels: list, centroids, samples: list[tuple[str, str]]) -> tuple[dict, list]:
        """({cluster: name} for the clusters matched to a voice sample, how sure it is: mapping.voice_report)"""
        builtin = builtin_samples(self.speakers_dir)
        requested = {name for name, _ in samples}
        all_samples = [s for s in builtin if s[0] not in requested] + list(samples)
        if not all_samples or not labels:
            return {}, []
        if centroids is None:
            raise ValueError("this diarization pipeline returns no speaker embeddings; samples unsupported")
        names, sample_embs = self._sample_embeddings(all_samples, {p for _, p in builtin})
        sim = mapping.cosine_similarity(sample_embs, centroids) if names else np.zeros((0, len(labels)))
        speakers, sim = mapping.best_per_speaker(names, sim)
        log.info("speaker/cluster similarity (%s): %s", speakers, np.round(sim, 3).tolist())
        matched = mapping.match_samples(sim, MATCH_THRESHOLD, self.extra_threshold)
        return {labels[c]: speakers[s] for c, s in matched.items()}, \
            mapping.voice_report(speakers, sim, matched, MATCH_THRESHOLD)

    # --- word timings

    def _sanitize_roman(self, word: str) -> str:
        """A romanized word reduced to characters MMS_FA's tokenizer accepts: uroman gives "-" for
        the maqaf (the CTC blank, which forced_align rejects) and a curly apostrophe for the geresh
        (KeyError in the tokenizer); both would crash the whole window"""
        word = word.lower().replace("’", "'").replace("‘", "'")
        return "".join(c for c in word if c in self.mms_chars)

    def _ctc_words(self, window, text: str, lang_code: str) -> list[dict]:
        """CTC forced alignment of the cue's own text to its audio -> [{"word", "start", "end"}] in
        window time. A word with nothing to align (only maqaf / punctuation) gets a zero-length span
        at the previous word's end, so the words stay 1:1 with text.split()"""
        orig_words = text.split()
        roman_words = self.uroman.romanize_string(text, lcode=lang_code).split()
        if len(roman_words) != len(orig_words):
            raise ValueError(f"romanization changed the word count ({len(orig_words)} -> {len(roman_words)})")
        sanitized = [self._sanitize_roman(w) for w in roman_words]
        align_idx = [i for i, w in enumerate(sanitized) if w]
        if not align_idx:
            raise ValueError("nothing to align after romanization")

        with self.torch.inference_mode():
            emission, _ = self.mms(window[None].to(self.device))
        token_spans = self.aligner(emission[0], self.tokenizer([sanitized[i] for i in align_idx]))
        seconds_per_frame = window.shape[0] / emission.size(1) / SR
        aligned = {align_idx[j]: (spans[0].start * seconds_per_frame, spans[-1].end * seconds_per_frame)
                   for j, spans in enumerate(token_spans)}

        results, last_end = [], 0.0
        for i, word in enumerate(orig_words):
            start, end = aligned.get(i, (last_end, last_end))
            last_end = end
            results.append({"word": word, "start": float(start), "end": float(end)})
        return results

    def _cue_words(self, waveform, cue: dict, lang_code: str) -> tuple[list[dict], bool]:
        """-> (the cue's words in absolute time, whether CTC worked; on failure they are VAD-spread)"""
        window = waveform[int(cue["start"] * SR):int(cue["end"] * SR)]
        try:
            aligned, ok = self._ctc_words(window, cue["text"], lang_code), True
        except Exception:
            log.debug("ctc failed on cue %.2f-%.2f, using VAD", cue["start"], cue["end"], exc_info=True)
            aligned, ok = _vad_words(window.numpy(), cue["text"]), False
        words = [{"word": w["word"], "start": min(cue["end"], cue["start"] + w["start"]),
                  "end": min(cue["end"], cue["start"] + w["end"])} for w in aligned]
        return words, ok

    # --- job

    def diarize(self, wav_file: str, cues: list[dict], samples: list[tuple[str, str]] = (),
                lang: str = "he", speakers: Optional[dict] = None,
                on_progress: Optional[Callable[[float], None]] = None,
                should_cancel: Optional[Callable[[], bool]] = None) -> dict:
        """cues: [{"start", "end", "text"}] (ai_trns.diarize.srt.parse_srt); samples: [(name, path)]
        on top of the built-in ones, a same name replaces them; speakers: num_speakers /
        min_speakers / max_speakers hints. Returns the diarization document"""
        from ..transcriber import Cancelled
        on_progress = on_progress or (lambda p: None)
        should_cancel = should_cancel or (lambda: False)

        def hook(step, artifact, file=None, total=None, completed=None):
            if should_cancel():
                raise Cancelled()
            i = PYANNOTE_STEPS.index(step) if step in PYANNOTE_STEPS else 0
            # pyannote ends every step (and reports a step without a total) with completed=None
            part = completed / total if total and completed is not None else 1.0
            on_progress(DIARIZE_SHARE * min(1.0, (i + part) / len(PYANNOTE_STEPS)))

        # seconds of each stage, into the document's stats: where a job's time goes
        seconds, mark = {}, time.time()

        def lap(stage: str):
            nonlocal mark
            seconds[stage], mark = round(time.time() - mark, 1), time.time()

        pcm = audio.load_pcm(wav_file)
        waveform = self.torch.from_numpy(pcm.copy())
        lap("audio")
        speaker_kwargs = {k: v for k, v in (speakers or {}).items() if v is not None}
        turns, labels, centroids = self._diarize(waveform, speaker_kwargs, hook)
        log.info("diarized: %d turns, %d clusters", len(turns), len(labels))
        lap("speakers")
        named, voices = self._name_clusters(labels, centroids, samples)
        lap("samples")

        lang_code = UROMAN_LANGS.get(lang, "heb")
        labeled = []
        mixed_cues = fallback = 0
        for cue_idx, cue in enumerate(cues):
            if should_cancel():
                raise Cancelled()
            on_progress(DIARIZE_SHARE + (1 - DIARIZE_SHARE) * cue_idx / max(1, len(cues)))
            if not cue["text"].split():
                continue
            words, ok = self._cue_words(waveform, cue, lang_code)
            fallback += not ok
            cluster, mixed = mapping.dominant_cluster(cue["start"], cue["end"], turns, DOMINANT_RATIO)
            if mixed and len(words) > 1:
                mixed_cues += 1
                for run_idx, run_cluster in mapping.word_runs(words, turns, MIN_PIECE_WORDS, MIN_PIECE_S):
                    labeled.extend((words[i], run_cluster, cue_idx) for i in run_idx)
            else:
                labeled.extend((w, cluster, cue_idx) for w in words)
        log.info("words: %d, ctc failed on %d of %d cues", len(labeled), fallback, len(cues))
        lap("align")

        names = mapping.speaker_names([c for _, c, _ in labeled if c is not None], named)
        return {
            "version": VERSION,
            "kind": KIND,
            "duration_s": len(pcm) / SR,
            "language": lang,
            "model": self.pipeline_name,
            "words": [{"word": w["word"], "start": round(w["start"], 3), "end": round(w["end"], 3),
                       "speaker": names.get(c), "cue": cue_idx} for w, c, cue_idx in labeled],
            "stats": {"speakers": len(set(names.values())), "named": len(named),
                      "mixed_cues": mixed_cues, "vad_fallback_cues": fallback, "seconds": seconds,
                      "voices": voices, "uncertain_voices": sum(v["uncertain"] for v in voices)},
        }
