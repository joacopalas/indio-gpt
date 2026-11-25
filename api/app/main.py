import os
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import StreamingResponse, FileResponse

from .schemas import GenerateRequest
from .serving import (
    MODEL_INDEX,
    get_bundle,
    generate_events,
    generate_full,
)

app = FastAPI(title="Indio-GPT")

api = APIRouter(prefix="/api")


@api.get("/health")
def health():
    return {
        "status": "ok",
        "models": list(MODEL_INDEX.keys()),
    }


@api.get("/models")
def models():
    items = sorted(
        MODEL_INDEX.values(),
        key=lambda x: x["target"],
    )

    return {
        "models": items,
        "default": items[0]["id"] if items else None,
    }


@api.post("/generate")
def generate(req: GenerateRequest):
    try:
        bundle = get_bundle(req.candidate)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not req.stream:
        text, hit_end = generate_full(bundle, req)

        return {
            "candidate": req.candidate,
            "text": text,
            "hit_end": hit_end,
            "prompt": req.prompt,
        }

    def event_stream():
        for event in generate_events(bundle, req):
            payload = json.dumps(event, ensure_ascii=False)
            yield f"data: {payload}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

app.include_router(api)

STATIC_DIR = Path(os.environ.get("STATIC_DIR", "static"))

if STATIC_DIR.exists():

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")

        file_path = (STATIC_DIR / full_path).resolve()

        if file_path.is_file():
            return FileResponse(file_path)

        return FileResponse(STATIC_DIR / "index.html")