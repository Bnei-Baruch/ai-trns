"""HTTP client of the paragraphing service (service.py on `stt`). stdlib only: runs in the worker's env"""
import json
import urllib.error
import urllib.request

KEY_HEADER = "X-API-Key"


def split(url: str, text: str, key: str = "", timeout: float = 1800) -> dict:
    """POST /paragraphs {"text"} -> {"sentences": [[start, end], ...], "breaks": [sentence numbers], ...}.
    Raises RuntimeError with the service's message"""
    request = urllib.request.Request(url.rstrip("/") + "/paragraphs", method="POST",
                                     data=json.dumps({"text": text}).encode("utf-8"),
                                     headers={"Content-Type": "application/json", KEY_HEADER: key})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"paragraphs service: {e.code} {e.read().decode('utf-8', 'replace')[:300]}") from None
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(f"paragraphs service: {getattr(e, 'reason', e)}") from None
