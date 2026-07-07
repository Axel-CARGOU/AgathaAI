import json, time
from pathlib import Path

from fastapi import APIRouter, FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


WEB_DIR = Path(__file__).resolve().parent
INDEX_FILE = WEB_DIR / "index.html"
GRAPH_DIR = WEB_DIR.parent / "logs" / "graphs"
TWITCH_CALLBACK_FILE = WEB_DIR.parent / "src" / "streaming" / "twitch" / "twitch_oauth_callback.json"

router = APIRouter()


@router.get("/", include_in_schema=False)
async def index(
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
    error_description: str | None = Query(default=None),
):
    if code is not None or error is not None:
        return _persist_twitch_callback(code, state, error, error_description)

    return FileResponse(INDEX_FILE)


@router.get("/health")
async def health():
    return {"ok": True, "service": "agathaai-web"}


@router.get("/metrics/graphs/{filename}", include_in_schema=False)
async def metrics_graph(filename: str):
    safe_name = Path(filename).name
    if safe_name != filename or Path(safe_name).suffix.lower() not in {".png", ".csv"}:
        return JSONResponse(status_code=404, content={"ok": False, "error": "file not found"})

    path = GRAPH_DIR / safe_name
    if not path.exists() or not path.is_file():
        return JSONResponse(status_code=404, content={"ok": False, "error": "file not found"})

    return FileResponse(path)


@router.get("/twitch_callback", include_in_schema=False)
async def twitch_callback(
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
    error_description: str | None = Query(default=None),
):
    return _persist_twitch_callback(code, state, error, error_description)


def _persist_twitch_callback(
    code: str | None,
    state: str | None,
    error: str | None,
    error_description: str | None,
):
    if error:
        _write_twitch_callback(
            {
                "provider": "twitch",
                "ok": False,
                "error": error,
                "error_description": error_description,
                "state": state,
                "received_at": time.time(),
            }
        )

        return JSONResponse(
            status_code=400,
            content={
                "ok": False,
                "provider": "twitch",
                "error": error,
                "error_description": error_description,
            },
        )

    _write_twitch_callback(
        {
            "provider": "twitch",
            "ok": True,
            "code": code,
            "state": state,
            "received_at": time.time(),
        }
    )

    return {
        "ok": True,
        "provider": "twitch",
        "code_received": code is not None,
        "state_received": state is not None,
        "message": "Twitch OAuth callback received. You can close this tab.",
    }


def _write_twitch_callback(payload: dict) -> None:
    TWITCH_CALLBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = TWITCH_CALLBACK_FILE.with_suffix(".tmp")

    with tmp_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    tmp_file.replace(TWITCH_CALLBACK_FILE)


def create_app() -> FastAPI:
    app = FastAPI(title="AgathaAI Web")
    app.include_router(router)
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web_static")
    return app


app = create_app()
