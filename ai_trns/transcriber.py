import copy
import json
import os
from typing import Callable, Optional

from faster_whisper import BatchedInferencePipeline, WhisperModel

from .audio import load_pcm
from .words import build_document

PROFILES = {
    "fast": dict(
        vad_filter=True,
        beam_size=3,
        temperature=[0.0],
        no_speech_threshold=0.5
    ),
    "stable": dict(
        vad_filter=False,
        beam_size=8,
        temperature=[0.0, 0.2],
        no_speech_threshold=0.7,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4
    ),
    # whisper_trainer's served decode (model card of orendar/whisper-large-v3-turbo-hebrew-kabbalah-ct2):
    # greedy, no previous-text prompt, VAD windows decoded in batches
    "served": dict(
        batched=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=1,
        condition_on_previous_text=False,
    ),
}

# Optional file in a model directory: {"profile": "served", "boundary_correction_s": {"start", "end"}, "name": ...}
MODEL_META = "ai_trns_model.json"


class Cancelled(Exception):
    """Raised when a transcription is cancelled between segments"""


def model_path_for(lang: str, models_dir: str = "./models") -> str:
    # For auto language detection, use English model as base multilingual model
    return os.path.join(models_dir, "en" if lang == "auto" else lang)


class Transcriber:
    """One loaded Whisper model; all job parameters are passed per call"""

    def __init__(self, lang: str, models_dir: str = "./models",
                 device: str = "cuda", compute_type: str = "float16"):
        self.lang = lang
        path = model_path_for(lang, models_dir)
        meta_path = os.path.join(path, MODEL_META)
        self.meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                self.meta = json.load(f)
        self.name = self.meta.get("name", os.path.basename(os.path.realpath(path)))
        self.default_profile = self.meta.get("profile", "stable")
        self.model = WhisperModel(path, device=device, compute_type=compute_type)
        self.batched = BatchedInferencePipeline(self.model)

    def transcribe(self, audio, profile: Optional[str] = None, batch_size: int = 8,
                   on_progress: Optional[Callable[[float], None]] = None,
                   should_cancel: Optional[Callable[[], bool]] = None) -> dict:
        """Audio (path or 16 kHz array) -> words document (ai_trns.words)"""
        if isinstance(audio, str):
            audio = load_pcm(audio)
        kwargs = copy.deepcopy(PROFILES[profile or self.default_profile])
        batched = kwargs.pop("batched", False)
        kwargs["language"] = None if self.lang == "auto" else self.lang
        kwargs["word_timestamps"] = True
        if batched:
            segments, info = self.batched.transcribe(audio, batch_size=batch_size, **kwargs)
        else:
            segments, info = self.model.transcribe(audio, **kwargs)
        # segments is a lazy generator: progress and cancellation work between segments
        result = []
        for segment in segments:
            if should_cancel and should_cancel():
                raise Cancelled()
            result.append(segment)
            if on_progress and info.duration:
                on_progress(min(1.0, segment.end / info.duration))
        return build_document(result, info.duration, info.language, self.name,
                              self.meta.get("boundary_correction_s"))
