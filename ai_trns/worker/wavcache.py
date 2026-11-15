"""
The converted audio (16 kHz mono wav) of a media URL, kept for a while in a directory the workers of one host
share (AI_TRNS_AUDIO_CACHE_DIR): the diarize job that follows a transcription of the same URL does not read and
convert it again. Whole media only (a clip is cheap). A file here is never opened by a job: a job gets its own
hard link (a copy across file systems), so removing an old file cannot break a running job. Stdlib only.
"""
import hashlib
import os
import shutil
import time


def _path(cache_dir: str, url: str) -> str:
    return os.path.join(cache_dir, hashlib.sha256(url.encode("utf-8")).hexdigest()[:32] + ".wav")


def _place(src: str, dest: str):
    try:
        os.link(src, dest)
    except OSError:
        shutil.copyfile(src, dest)


def take(cache_dir: str, url: str, dest: str, max_age_s: float) -> bool:
    """The URL's wav into dest if it is here and not older than max_age_s"""
    path = _path(cache_dir, url)
    try:
        if time.time() - os.path.getmtime(path) > max_age_s:
            return False
        _place(path, dest)
        return True
    except OSError:
        return False


def put(cache_dir: str, url: str, wav_file: str, max_age_s: float):
    """wav_file (a job's own) is kept as the URL's wav; older files are removed. Never fails a job"""
    try:
        os.makedirs(cache_dir, exist_ok=True)
        now = time.time()
        for name in os.listdir(cache_dir):
            old = os.path.join(cache_dir, name)
            if now - os.path.getmtime(old) > max_age_s:
                os.remove(old)
        path = _path(cache_dir, url)
        # another worker never sees a half-written file
        tmp = f"{path}.{os.getpid()}.tmp"
        _place(wav_file, tmp)
        os.replace(tmp, path)
        os.utime(path)
    except OSError:
        pass
