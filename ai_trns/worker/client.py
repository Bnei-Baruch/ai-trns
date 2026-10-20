"""HTTP client for the worker side of ai_trns.protocol. Stdlib only."""
import json
import shutil
import urllib.error
import urllib.request
from typing import Optional
from urllib.parse import quote

from .. import protocol


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"API {status}: {message}")
        self.status = status


class Client:
    def __init__(self, api_url: str, key: str, worker: str, timeout: float = 60):
        self.api_url = api_url
        self.worker = worker
        self.timeout = timeout
        self.headers = {protocol.KEY_HEADER: key, protocol.VERSION_HEADER: str(protocol.VERSION)}

    def _request(self, method: str, path: str, body=None, timeout: Optional[float] = None):
        data = None
        headers = dict(self.headers)
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.api_url + path, data=data, headers=headers, method=method)
        try:
            return urllib.request.urlopen(request, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as e:
            raise ApiError(e.code, e.read().decode("utf-8", errors="replace")[:500]) from None

    def _json(self, method: str, path: str, body=None):
        with self._request(method, path, body) as response:
            if response.status == 204:
                return None
            return json.loads(response.read())

    def _q(self) -> str:
        return f"?worker={quote(self.worker)}"

    def lease(self, langs: list, engines: list) -> Optional[dict]:
        return self._json("POST", "/worker/lease", {"worker": self.worker, "langs": langs, "engines": engines})

    def download(self, path: str, dest: str):
        """The job's uploaded audio (its source's "upload" path) or another input (file_path)"""
        with self._request("GET", path + self._q(), timeout=600) as response, open(dest, "wb") as out:
            shutil.copyfileobj(response, out, 1024 * 1024)

    @staticmethod
    def file_path(job_id: str, name: str) -> str:
        return f"/worker/jobs/{job_id}/files/{quote(name)}"

    def heartbeat(self, job_id: str, stage: str, progress: float) -> bool:
        """True if the job was cancelled (or is no longer this worker's)"""
        reply = self._json("POST", f"/worker/jobs/{job_id}/heartbeat",
                           {"worker": self.worker, "stage": stage, "progress": round(progress, 4)})
        return bool(reply.get("cancel"))

    def result(self, job_id: str, doc: dict):
        self._json("PUT", f"/worker/jobs/{job_id}/result{self._q()}", doc)

    def fail(self, job_id: str, error: str):
        self._json("POST", f"/worker/jobs/{job_id}/fail", {"worker": self.worker, "error": error[:2000]})
