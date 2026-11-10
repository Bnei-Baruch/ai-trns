"""
Speaker labels for an existing transcript (moved from stt-api /diarize): audio + its SRT in,
the same text marked by speaker out. Text and cue timings of the SRT are trusted; only who
speaks is detected.

GPU part (worker engine "diarize", engine.py): pyannote diarization of the whole file, voice
samples of known speakers matched to its clusters, every word of every cue timed by CTC
forced alignment, each word given a speaker. It sends back the diarization document:

    {
      "version": 1,
      "kind": "diarize",
      "duration_s": float,
      "language": "he" | ...,
      "model": str,                              # pyannote pipeline
      "words": [{"word", "start", "end", "speaker", "cue"}],
      "stats": {"speakers", "named", "mixed_cues", "vad_fallback_cues",
                "seconds": {"audio", "speakers", "samples", "align", "paragraphs"},   # of each stage
                "voices": [{"speaker", "similarity", "named", "next", "uncertain"}],     # mapping.voice_report
                "uncertain_voices": int}   # > 0: the speakers' names are worth a look
    }

speaker: a known speaker's name ("Rav") or SPEAKER_NN, None only when no speech was found;
cue: 0-based index of the word's cue in the input SRT.

CPU part (API, labels.py / resegment.py): the input SRT labeled and split at speaker changes,
or a new SRT built from the words. Model-free modules: srt, mapping, resegment, labels.
"""
VERSION = 1
KIND = "diarize"
