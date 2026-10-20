import os

import uvicorn

if __name__ == "__main__":
    # One process only: the worker and the model cache live in memory
    uvicorn.run("ai_trns.server.app:app", workers=1,
                host=os.environ.get("AI_TRNS_HOST", "127.0.0.1"),
                port=int(os.environ.get("AI_TRNS_PORT", "8000")))
