import argparse
import subprocess
import sys
import os
import re
import time
import threading
import shutil

print("[DEBUG] Starting from directory:", os.getcwd())
print("[DEBUG] Launch arguments:", sys.argv)

from transcriber import (
    transcribe_file,
    transcribe_file_to_srt,
    transcribe_file_to_advanced_srt,
    PROFILES,
    transcribe_kwargs,
    load_model,
)

def write_progress(message: str):
    message = message.replace('\r', '')  # remove \r
    with open("report_python.log", "w", encoding="utf-8") as f:
        f.write(message)

class TeeStdout:
    def __init__(self, log_path, mode="w", encoding="utf-8"):
        self.log_path = log_path
        self.stdout = sys.__stdout__

    def write(self, msg):
        self.stdout.write(msg)
        self.stdout.flush()

    def flush(self):
        self.stdout.flush()

def get_audio_duration_sec(input_path: str) -> float:
    cmd = [
        "ffprobe", "-i", input_path,
        "-show_entries", "format=duration",
        "-v", "quiet", "-of", "csv=p=0"
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    output = result.stdout.decode().strip()
    try:
        return float(output)
    except ValueError:
        print("[!] Could not determine audio length.")
        write_progress("[!] Could not determine audio length.")
        return 0.0

def ffmpeg_progress_monitor(stderr_pipe, total_duration):
    time_pattern = re.compile(r"time=(\d+):(\d+):(\d+\.\d+)")
    while True:
        line = stderr_pipe.readline()
        if not line:
            break
        match = time_pattern.search(line)
        if match:
            h, m, s = map(float, match.groups())
            current_sec = h * 3600 + m * 60 + s
            percent = min(100, int(current_sec / total_duration * 100))
            bar = ('#' * (percent // 2)).ljust(50)
            progress = f"[FFmpeg] [{bar}] {percent}%"
            print(f"\r{progress}", end='')
            write_progress(progress)
    print()

def convert_to_wav(input_path, wav_file, total_duration):
    cmd = [
        "ffmpeg", "-i", input_path,
        "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-f", "wav", wav_file, "-y", "-loglevel", "info"
    ]
    process = subprocess.Popen(cmd, stderr=subprocess.PIPE, universal_newlines=True)
    ffmpeg_progress_monitor(process.stderr, total_duration)
    process.wait()
    if process.returncode != 0:
        print("[✗] FFmpeg conversion failed with error.")
        write_progress("[✗] FFmpeg conversion failed with error.")
        raise RuntimeError("FFmpeg error")

def show_expected_time_and_progress(expected_sec, task):
    print(f"[~] Expected transcription time: {int(expected_sec // 60)} min {int(expected_sec % 60)} sec")
    write_progress(f"[~] Expected transcription time: {int(expected_sec // 60)} min {int(expected_sec % 60)} sec")
    start = time.time()
    finished = [False]

    def timer_progress():
        while not finished[0]:
            elapsed = time.time() - start
            percent = min(100, int(100 * elapsed / expected_sec))
            bar = ('#' * (percent // 2)).ljust(50)
            progress = f"[Whisper] [{bar}] {percent}% - {int(elapsed)} sec"
            print(f"\r{progress}", end='')
            write_progress(progress)
            time.sleep(1)
        elapsed = time.time() - start
        final = f"[Whisper] [{'#'*50}] 100% - {int(elapsed)} sec"
        print(f"\r{final}")
        write_progress(final)

    t = threading.Thread(target=timer_progress, daemon=True)
    t.start()
    result = task()
    finished[0] = True
    t.join()
    return result, time.time() - start

def copy_to_nginx(src, dest_dir="/usr/share/nginx/html/ai/"):
    if os.path.exists(src):
        try:
            shutil.copy(src, dest_dir)
            print(f"[✓] {src} copied to {dest_dir}")
            write_progress(f"[✓] {src} copied to {dest_dir}")
        except Exception as e:
            print(f"[✗] Error copying {src}: {e}")
            write_progress(f"[✗] Error copying {src}: {e}")
    else:
        print(f"[!] File {src} not found, copying skipped.")
        write_progress(f"[!] File {src} not found, copying skipped.")

def main():
    def excepthook(exctype, value, tb):
        import traceback
        print("".join(traceback.format_exception(exctype, value, tb)))
    sys.excepthook = excepthook

    parser = argparse.ArgumentParser(description="Full file transcription (or URL) via ffmpeg + Whisper")
    parser.add_argument("--input", required=True, help="Path to audio file or URL (mp3, m4a, flac, wav, ...)")
    parser.add_argument("--txt", action="store_true", help="Save result to output.txt")
    parser.add_argument("--srt", action="store_true", help="Save result to output.srt")
    parser.add_argument("--impl", type=int, choices=[1, 2], default=2, help="SRT implementation: 1=standard, 2=advanced with pause and punctuation analysis")
    parser.add_argument("--fix-rtl", action="store_true", help="Fix RTL punctuation issues for Hebrew (only for --impl 2)")
    parser.add_argument("--profile", choices=["stable", "fast"], default="stable", help="Quality profile: stable or fast")
    parser.add_argument("--keep-wav", action="store_true", help="Don't delete temporary output.wav after transcription")
    parser.add_argument("--rtf", type=float, default=40.0, help="Expected processing speed (x real-time), e.g. 40 for RTX A5000")
    parser.add_argument("--lang", choices=["en", "he", "he_old", "he_bb", "ru", "auto"], default="he", help="Model language: en, he, he_old, he_bb, ru, auto")
    args = parser.parse_args()

    # Load the required model by language
    load_model(args.lang)

    transcribe_kwargs.clear()
    transcribe_kwargs.update(PROFILES[args.profile])

    duration = get_audio_duration_sec(args.input)
    if duration == 0:
        print("[✗] Error: could not determine audio duration.")
        write_progress("[✗] Error: could not determine audio duration.")
        sys.exit(1)

    wav_file = "output.wav"
    print(f"[i] Converting {args.input} → {wav_file} ...")
    write_progress(f"[i] Converting {args.input} → {wav_file} ...")
    convert_to_wav(args.input, wav_file, duration)

    expected_speed = args.rtf
    expected_sec = 4.5 * (duration / expected_speed)

    print("\n[~] Transcribing... (this may take several minutes)\n")
    write_progress("[~] Transcribing... (this may take several minutes)")

    if args.txt:
        def transcribe_task():
            return transcribe_file(wav_file)
        result, actual_time = show_expected_time_and_progress(expected_sec, transcribe_task)
        with open("output.txt", "w", encoding="utf-8") as f:
            f.write(result)
        final = f"[✓] Saved to output.txt (actual time: {int(actual_time // 60)} min {int(actual_time % 60)} sec)"
        print("\n" + final)
        write_progress(final)
        copy_to_nginx("output.txt")

    if args.srt:
        if args.impl == 1:
            # Standard implementation
            def transcribe_task():
                return transcribe_file_to_srt(wav_file)
            impl_name = "standard"
        else:
            # Advanced implementation
            def transcribe_task():
                return transcribe_file_to_advanced_srt(wav_file, args.lang, args.fix_rtl)
            impl_name = "advanced"
            if args.fix_rtl and (args.lang == "he" or args.lang == "he_bb" or args.lang == "auto"):
                impl_name += " with RTL fixes"
        
        print(f"[i] Using {impl_name} SRT implementation")
        write_progress(f"[i] Using {impl_name} SRT implementation")
        
        result, actual_time = show_expected_time_and_progress(expected_sec, transcribe_task)
        with open("output.srt", "w", encoding="utf-8") as f:
            f.write(result)
        final = f"[✓] Saved to output.srt ({impl_name}) (actual time: {int(actual_time // 60)} min {int(actual_time % 60)} sec)"
        print("\n" + final)
        write_progress(final)
        copy_to_nginx("output.srt")

    if not args.keep_wav:
        try:
            os.remove(wav_file)
        except Exception:
            pass

if __name__ == "__main__":
    main()
