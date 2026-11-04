"""
Live transcription: an audio stream -> final and partial words within seconds, and at the end a
words document (ai_trns.words, timing source "live"). See TASK-live-transcription.md.

    stream.py  LocalAgreement over Whisper (StreamProcessor), words document from final words
    source.py  ffmpeg: .sdp (RTP from Janus) / URL / file -> 16 kHz PCM, read in a thread
    engine.py  source -> processor -> events (final / partial)
    hub.py     events for any number of listeners (stdlib only: the API server imports it)
    demo.py    a page with a player and the text, without the API (python -m ai_trns.live --serve)
"""
