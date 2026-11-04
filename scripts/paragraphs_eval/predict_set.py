"""Adapter answers for a set of Rav texts, the way the paragraphs service gives them (service.Model.split).
Runs on the GPU host in mdb-lora's env, with ai_trns on PYTHONPATH:
  PARAGRAPHS_ADAPTER=paragraphing_asr python predict_set.py texts.m0.jsonl preds/paragraphing_asr.m0.jsonl
Input lines {"id", "text"} (methods.py texts); output lines {"id", "sentences", "breaks", "chunks", "seconds", ...}.
Already answered ids in the output are skipped."""
import json
import os
import sys

os.environ.setdefault("PARAGRAPHS_KEY", "unused")
from ai_trns.paragraphs import service  # noqa: E402


def main(src, dst):
    done = set()
    if os.path.exists(dst):
        done = {json.loads(l)["id"] for l in open(dst, encoding="utf-8")}
    with open(dst, "a", encoding="utf-8") as out:
        for line in open(src, encoding="utf-8"):
            row = json.loads(line)
            if row["id"] in done:
                continue
            res = service.MODEL.split(row["text"])
            out.write(json.dumps({"id": row["id"], **res}, ensure_ascii=False) + "\n")
            out.flush()
            print(row["id"], len(res["breaks"]), res["seconds"], flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
