"""
Live transcription from the command line, to try it out before the API and the worker know about it.

    python -m ai_trns.live --input lesson.mp3 --realtime --lang he --models-dir ./models --out live
    python -m ai_trns.live --input janus.sdp ...            # RTP from Janus rtp_forward

Events go to stdout as JSON lines (final / partial); --out writes <out>.json (words document) and
<out>.srt at the end. Ctrl+C stops the stream and still writes the result.

    python -m ai_trns.live --input lesson.mp3 --serve 8770 ...   # page with a player and the text

With --serve the stream starts on the first Play on the page (a file is then read at its natural pace).
"""
import argparse
import json
import logging
import signal
import sys
import threading

from ..subtitles import render
from .engine import MIN_CHUNK_S, run
from .source import AudioSource
from .stream import MAX_BUFFER_S, TRIM_S, StreamProcessor, WhisperAsr, build_live_document


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help=".sdp, http(s) URL or file")
    p.add_argument("--realtime", action="store_true", help="read a file at its natural pace")
    p.add_argument("--lang", default="he")
    p.add_argument("--models-dir", default="./models")
    p.add_argument("--device", default="cuda")
    p.add_argument("--compute-type", default="float16")
    p.add_argument("--min-chunk", type=float, default=MIN_CHUNK_S, help="seconds of new audio per decode")
    p.add_argument("--trim", type=float, default=TRIM_S)
    p.add_argument("--max-buffer", type=float, default=MAX_BUFFER_S)
    p.add_argument("--prompt-chars", type=int, default=0,
                   help="previous final text as prompt (the Hebrew model is served without one)")
    p.add_argument("--partial", action="store_true", help="print partial events too")
    p.add_argument("--out", help="write <out>.json and <out>.srt at the end")
    p.add_argument("--serve", type=int, metavar="PORT", help="serve the demo page instead of printing events")
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()

    from ..transcriber import Transcriber
    transcriber = Transcriber(args.lang, args.models_dir, args.device, args.compute_type)

    def new_processor():
        return StreamProcessor(WhisperAsr(transcriber), args.prompt_chars, args.trim, args.max_buffer)

    def document(processor, duration):
        return build_live_document(processor.final, duration, args.lang, transcriber.name,
                                   transcriber.meta.get("boundary_correction_s"))

    if args.serve is not None:
        from .demo import Session, serve
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        server = serve(Session(args.input, new_processor, document, args.min_chunk, args.out),
                       args.host, args.serve, args.lang)
        print(f"http://{args.host}:{server.server_port}/", file=sys.stderr, flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        return

    processor = new_processor()

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    def on_event(e):
        if e["type"] == "final" or args.partial:
            print(json.dumps(e, ensure_ascii=False), flush=True)

    with AudioSource(args.input, args.realtime) as source:
        duration = run(source, processor, on_event, args.min_chunk, stop.is_set)
    doc = document(processor, duration)
    print(f"{duration:.1f} s, {len(doc['words'])} words", file=sys.stderr)
    if args.out:
        with open(args.out + ".json", "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)
        with open(args.out + ".srt", "w", encoding="utf-8") as f:
            f.write(render(doc, "srt"))


if __name__ == "__main__":
    main()
