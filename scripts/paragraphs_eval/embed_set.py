"""bge-m3 embeddings (ollama /api/embed) of the Rav text's sentences, for TextTiling in methods.py.
  python3 embed_set.py <dir> texts.<set>.m<merge>.jsonl [ollama url, default http://10.66.24.55:11434]
-> <dir>/emb/<id>.m<merge>.json (a list of vectors, one per sentence). Only reads embeddings."""
import json
import os
import re
import sys
import urllib.request


def embed(url, texts):
    req = urllib.request.Request(url.rstrip("/") + "/api/embed", method="POST",
                                 data=json.dumps({"model": "bge-m3", "input": texts}).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["embeddings"]


def main(d, src, url="http://10.66.24.55:11434"):
    merge = re.search(r"\.m(\w+)\.jsonl$", src).group(1)
    os.makedirs(os.path.join(d, "emb"), exist_ok=True)
    for line in open(os.path.join(d, src), encoding="utf-8"):
        row = json.loads(line)
        path = os.path.join(d, "emb", f"{row['id']}.m{merge}.json")
        if os.path.exists(path):
            continue
        sents, vecs = row["sentences"], []
        for i in range(0, len(sents), 64):
            vecs += [[round(x, 5) for x in v] for v in embed(url, sents[i:i + 64])]
        json.dump(vecs, open(path, "w"))
        print(row["id"], len(vecs), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
