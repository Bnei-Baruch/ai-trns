"""
API <-> GPU worker protocol. Workers pull: they call the API, the API never calls them,
so a worker only needs outbound HTTP to the API.

    POST /worker/lease                     {"worker", "langs": [...], "engines": [...]} -> 200 job | 204 nothing to do
    GET  /worker/jobs/{id}/audio           the uploaded file of a job with an upload source
    GET  /worker/jobs/{id}/files/{name}    another input: a diarize job's cues.json and voice samples
    POST /worker/jobs/{id}/heartbeat       {"worker", "stage", "progress"} -> {"cancel": bool}
    PUT  /worker/jobs/{id}/result          words document (ai_trns.words) or diarization document (ai_trns.diarize)
    POST /worker/jobs/{id}/fail            {"worker", "error"}

A worker gets transcriptions in its langs and jobs of its other engines ("engines" absent:
["transcribe"]); params["kind"] == "diarize" is the diarize engine's.
A leased job is the worker's until the lease runs out; every heartbeat extends it. A lost
worker's job goes back to the queue. Requests carry the worker key and the protocol version
in headers. Stdlib only: the worker may run inside another project's locked environment.
"""

VERSION = 1

KEY_HEADER = "X-Worker-Key"
VERSION_HEADER = "X-Worker-Protocol"

# Seconds a lease lasts without a heartbeat; workers send heartbeats several times per lease
LEASE_S = 120
HEARTBEAT_S = 5
