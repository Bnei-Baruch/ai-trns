"""
Live audio in: ffmpeg decodes the source to 16 kHz mono float32, a thread reads it so ffmpeg is
never blocked while the GPU decodes (an RTP input would drop packets).

Sources: a live stream (HLS .m3u8 or DASH .mpd over http(s), an .sdp file with RTP from Janus
rtp_forward), a media URL or a local file (with realtime=True a recording is read at its natural
pace, to simulate a live stream).
"""
from __future__ import annotations

import queue
import subprocess
import threading
from typing import Optional

import numpy as np

from ..audio import is_live  # noqa: F401  (the live engine's sources are told apart here)
from .stream import SAMPLE_RATE

BYTES_PER_SAMPLE = 4
READ_S = 0.1


def _protocols(source: str) -> list:
    if source.endswith(".sdp"):
        return ["-protocol_whitelist", "file,udp,rtp"]
    if source.lower().startswith(("http://", "https://")):
        # crypto: AES-encrypted HLS segments
        return ["-protocol_whitelist", "http,https,tcp,tls,crypto"]
    return ["-protocol_whitelist", "file"]


class AudioSource:
    """Use as a context manager; read() returns whatever audio arrived since the last call"""

    def __init__(self, source: str, realtime: bool = False):
        self.cmd = ["ffmpeg", "-nostdin", "-v", "error", *_protocols(source), *(["-re"] if realtime else []),
                    "-i", source, "-map", "0:a:0",
                    # gaps in the input timestamps (lost packets, a publisher reconnecting) become silence,
                    # so word times stay stream times. No first_pts: a live stream's timestamps start
                    # far from 0, that would be hours of padding
                    "-af", "aresample=async=1",
                    "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
        self.chunks: queue.Queue = queue.Queue()
        self.process: Optional[subprocess.Popen] = None
        self.ended = threading.Event()
        self.error = ""

    def __enter__(self):
        self.process = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        threading.Thread(target=self._read, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.close()

    def _read(self):
        size = int(SAMPLE_RATE * READ_S) * BYTES_PER_SAMPLE
        rest = b""
        while True:
            data = self.process.stdout.read1(size)
            if not data:
                break
            data = rest + data
            whole = len(data) - len(data) % BYTES_PER_SAMPLE
            rest = data[whole:]
            self.chunks.put(np.frombuffer(data[:whole], np.float32))
        self.process.wait()
        if self.process.returncode > 0:
            self.error = self.process.stderr.read().decode(errors="replace").strip()[-300:]
        self.ended.set()

    def read(self, wait_s: float) -> Optional[np.ndarray]:
        """Audio received so far, waiting up to wait_s for the first piece; None once the source ended"""
        parts = []
        try:
            parts.append(self.chunks.get(timeout=wait_s))
            while True:
                parts.append(self.chunks.get_nowait())
        except queue.Empty:
            pass
        if parts:
            return np.concatenate(parts)
        if self.ended.is_set() and self.chunks.empty():
            if self.error:
                raise RuntimeError("ffmpeg failed: " + self.error)
            return None
        return np.zeros(0, dtype=np.float32)

    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
