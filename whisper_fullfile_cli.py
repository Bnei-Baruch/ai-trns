"""
CLI for exec-api: one file per run, the model is loaded on every start.
Output names (output.txt / output.srt / report_python.log) are what exec-api and ai-tools expect.
"""
import argparse
import os
import shutil
import signal
import sys
import time

print("[DEBUG] Starting from directory:", os.getcwd())
print("[DEBUG] Launch arguments:", sys.argv)

from ai_trns import audio
from ai_trns.subtitles import render
from ai_trns.transcriber import Transcriber

report_file = "report_python.log"


def report(message: str, end: str = "\n"):
    """Print and overwrite the exec-api report file (ai-tools reads `NN%` from it)"""
    print(message, end=end, flush=True)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(message.replace("\r", ""))


def bar(percent: int) -> str:
    return ('#' * (percent // 2)).ljust(50)


def fmt_time(sec: float) -> str:
    return f"{int(sec // 60)} min {int(sec % 60)} sec"


def copy_to_nginx(src: str, dest_dir: str):
    if not dest_dir:
        return
    if os.path.exists(src):
        try:
            shutil.copy(src, dest_dir)
            report(f"[✓] {src} copied to {dest_dir}")
        except Exception as e:
            report(f"[✗] Error copying {src}: {e}")
    else:
        report(f"[!] File {src} not found, copying skipped.")


def run(args):
    formats = [fmt for fmt in ("txt", "srt", "json") if getattr(args, fmt)]
    if not formats:
        raise RuntimeError("nothing to do: pass --txt, --srt and/or --json")

    # Load the required model by language
    transcriber = Transcriber(args.lang, args.models_dir)

    duration = audio.get_duration(args.input)
    if duration == 0:
        raise RuntimeError("could not determine audio duration")

    wav_file = os.path.join(args.out_dir, "output.wav")
    report(f"[i] Converting {args.input} → {wav_file} ...")
    try:
        def ffmpeg_progress(p: float):
            percent = int(p * 100)
            report(f"\r[FFmpeg] [{bar(percent)}] {percent}%", end="")

        audio.convert_to_wav(args.input, wav_file, duration, ffmpeg_progress)
        print()

        expected_sec = 4.5 * (duration / args.rtf)
        report(f"[~] Expected transcription time: {fmt_time(expected_sec)}")
        if "srt" in formats and args.fix_rtl and args.lang in ("he", "auto"):
            report("[i] SRT with RTL fixes for Hebrew")
        report("[~] Transcribing... (this may take several minutes)")

        start = time.time()

        def whisper_progress(p: float):
            percent = int(p * 100)
            report(f"\r[Whisper] [{bar(percent)}] {percent}% - {int(time.time() - start)} sec", end="")

        doc = transcriber.transcribe(wav_file, args.profile, args.batch_size, on_progress=whisper_progress)
        actual_time = time.time() - start
        report(f"\r[Whisper] [{bar(100)}] 100% - {int(actual_time)} sec")
    finally:
        if not args.keep_wav:
            try:
                os.remove(wav_file)
            except FileNotFoundError:
                pass

    for fmt in formats:
        out_file = os.path.join(args.out_dir, f"output.{fmt}")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(render(doc, fmt, args.fix_rtl))
        report(f"[✓] Saved to {out_file} (actual time: {fmt_time(actual_time)})")
        copy_to_nginx(out_file, args.nginx_dir)


def main():
    global report_file
    parser = argparse.ArgumentParser(description="Full file transcription (or URL) via ffmpeg + Whisper")
    parser.add_argument("--input", required=True, help="Path to audio file or URL (mp3, m4a, flac, wav, ...)")
    parser.add_argument("--txt", action="store_true", help="Save result to output.txt")
    parser.add_argument("--srt", action="store_true", help="Save result to output.srt")
    parser.add_argument("--json", action="store_true", help="Save words with timecodes to output.json")
    # Old SRT implementations were removed; still accepted so existing exec-api configs keep working
    parser.add_argument("--impl", type=int, choices=[1, 2], help=argparse.SUPPRESS)
    parser.add_argument("--fix-rtl", action="store_true", help="Add RTL marks to Hebrew subtitles")
    parser.add_argument("--profile", choices=["stable", "fast", "served"], help="Decoding profile (default: the model's own, else stable)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for the served profile")
    parser.add_argument("--keep-wav", action="store_true", help="Don't delete temporary output.wav after transcription")
    parser.add_argument("--rtf", type=float, default=40.0, help="Expected processing speed (x real-time), only for the time estimate")
    parser.add_argument("--lang", choices=["en", "he", "ru", "auto"], default="he", help="Model language: en, he, ru, auto")
    parser.add_argument("--out-dir", default=".", help="Directory for output.* files (default: current)")
    parser.add_argument("--report-file", default="report_python.log", help="Progress file, report_<service ID>.log for exec-api")
    parser.add_argument("--models-dir", default="./models", help="Directory with models/<lang>")
    parser.add_argument("--nginx-dir", default="/usr/share/nginx/html/ai/", help="Copy results here; empty to disable")
    args = parser.parse_args()

    report_file = args.report_file
    os.makedirs(args.out_dir, exist_ok=True)
    # exec-api `stop` sends SIGTERM: exit through finally so output.wav is removed
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    try:
        run(args)
    except SystemExit:
        report("[✗] ERROR: stopped")
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        report(f"[✗] ERROR: {e}")
        sys.exit(1)
    report("[✓] DONE")


if __name__ == "__main__":
    main()
