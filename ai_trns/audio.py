import re
import subprocess
from typing import Callable, Optional
from urllib.parse import urlparse

_url_re = re.compile(r"^https?://", re.IGNORECASE)
_time_re = re.compile(r"time=(\d+):(\d+):(\d+\.\d+)")


def is_url(source: str) -> bool:
    return bool(_url_re.match(source))


def check_url(url: str, allowed_hosts: list):
    """Empty allowed_hosts = any host"""
    parsed = urlparse(url)
    if parsed.scheme.lower() not in ("http", "https"):
        raise ValueError("only http(s) URLs are allowed")
    if allowed_hosts and parsed.hostname not in allowed_hosts:
        raise ValueError(f"host {parsed.hostname!r} is not allowed")


def _protocols(source: str) -> list:
    # Only what is needed to read a local file or an http(s) URL: no concat:, data:, subfile:, etc.,
    # and a URL input cannot make ffmpeg read local files (e.g. via a crafted playlist)
    return ["-protocol_whitelist", "http,https,tcp,tls" if is_url(source) else "file"]


def get_duration(source: str) -> float:
    cmd = ["ffprobe", *_protocols(source), "-i", source,
           "-show_entries", "format=duration", "-v", "quiet", "-of", "csv=p=0"]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, timeout=300)
    try:
        return float(result.stdout.decode().strip())
    except ValueError:
        return 0.0


def convert_to_wav(source: str, wav_file: str, duration: float,
                   on_progress: Optional[Callable[[float], None]] = None,
                   should_cancel: Optional[Callable[[], bool]] = None) -> bool:
    """16 kHz mono wav. Returns False if cancelled; raises RuntimeError on ffmpeg failure"""
    cmd = ["ffmpeg", "-nostdin", *_protocols(source), "-i", source,
           "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-f", "wav", wav_file, "-y", "-loglevel", "info"]
    process = subprocess.Popen(cmd, stderr=subprocess.PIPE, universal_newlines=True, errors="replace")
    tail = []
    try:
        for line in process.stderr:
            if should_cancel and should_cancel():
                process.terminate()
                return False
            tail = (tail + [line.strip()])[-5:]
            match = _time_re.search(line)
            if match and duration and on_progress:
                h, m, s = map(float, match.groups())
                on_progress(min(1.0, (h * 3600 + m * 60 + s) / duration))
        process.wait()
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
    if process.returncode != 0:
        raise RuntimeError("ffmpeg failed: " + " | ".join(tail))
    return True
