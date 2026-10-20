import copy
import os
from typing import Callable, Optional

from faster_whisper import WhisperModel

from .subtitles import ADVANCED_SRT_AVAILABLE, render

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
        logprob_threshold=-1.0,
        compression_ratio_threshold=2.4
    )
}


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
        self.model = WhisperModel(
            model_path_for(lang, models_dir),
            device=device,
            compute_type=compute_type
        )

    def segments(self, audio, profile: str = "stable", word_timestamps: bool = False,
                 on_progress: Optional[Callable[[float], None]] = None,
                 should_cancel: Optional[Callable[[], bool]] = None) -> list:
        kwargs = copy.deepcopy(PROFILES[profile])
        kwargs["language"] = None if self.lang == "auto" else self.lang
        kwargs["word_timestamps"] = word_timestamps
        segments, info = self.model.transcribe(audio, **kwargs)
        # segments is a lazy generator: progress and cancellation work between segments
        result = []
        for segment in segments:
            if should_cancel and should_cancel():
                raise Cancelled()
            result.append(segment)
            if on_progress and info.duration:
                on_progress(min(1.0, segment.end / info.duration))
        return result

    def transcribe(self, audio, formats=("srt",), profile: str = "stable",
                   impl: int = 2, fix_rtl: bool = False,
                   on_progress: Optional[Callable[[float], None]] = None,
                   should_cancel: Optional[Callable[[], bool]] = None) -> dict:
        """One Whisper pass for all requested formats -> {format: content}"""
        words = "srt" in formats and impl == 2 and ADVANCED_SRT_AVAILABLE
        segments = self.segments(audio, profile, words, on_progress, should_cancel)
        return {fmt: render(segments, fmt, self.lang, impl, fix_rtl) for fmt in formats}
