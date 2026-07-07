from __future__ import annotations

import time
import threading
from copy import deepcopy
from typing import Any


_STATE_LOCK = threading.RLock()
_STREAM_BUFFER_CONDITION = threading.Condition()
_STREAM_APPLY_LOCK = threading.RLock()
_STREAM_BUFFER: list[tuple[int, str]] = []
_STREAM_WORKER_THREAD: threading.Thread | None = None
_STREAM_FLUSH_INTERVAL_S = 0.15
_STREAM_PREVIEW_MAX_CHARS = 8000
_STREAM_GENERATION_ID = 0

_STATE: dict[str, Any] = {
    "modules_active": {
        "bot": False,
        "llm": False,
        "tts": False,
        "stt": False,
        "vtuber": False,
        "rag": False,
    },
    "controls": {
        "botDisabled": False,
        "llmDisabled": False,
        "ttsDisabled": False,
        "muted": False,
        "vtuberDisabled": False,
    },
    "signals": {
        "aiThinking": False,
        "aiSpeaking": False,
        "humanSpeaking": False,
    },
    "current_message": {
        "source": "Idle",
        "content": "",
        "streamMode": "Waiting",
        "streamPreview": "",
    },
    "vision_enabled": False,
    "vision_last_path": None,
    "vision_error": None,
    "temporary_context": "",
    "abort_generation_requested": False,
    "stop_audio_requested": False,
    "updated_at": time.time(),
}


def snapshot() -> dict[str, Any]:
    with _STATE_LOCK:
        return deepcopy(_STATE)


def touch() -> None:
    _STATE["updated_at"] = time.time()


def set_module_active(name: str, value: bool) -> None:
    with _STATE_LOCK:
        _STATE["modules_active"][name] = bool(value)
        touch()


def set_control(name: str, value: bool) -> None:
    with _STATE_LOCK:
        _STATE["controls"][name] = bool(value)

        if name == "botDisabled":
            _STATE["modules_active"]["bot"] = not bool(value)
        elif name == "llmDisabled":
            _STATE["modules_active"]["llm"] = not bool(value)
        elif name == "ttsDisabled":
            _STATE["modules_active"]["tts"] = not bool(value)
        elif name == "muted":
            _STATE["modules_active"]["stt"] = not bool(value)
        elif name == "vtuberDisabled":
            _STATE["modules_active"]["vtuber"] = not bool(value)

        touch()


def toggle_control(name: str) -> bool:
    with _STATE_LOCK:
        next_value = not bool(_STATE["controls"].get(name))
    set_control(name, next_value)
    return next_value


def set_signal(name: str, value: bool) -> None:
    with _STATE_LOCK:
        if name in _STATE["signals"]:
            _STATE["signals"][name] = bool(value)
            touch()


def set_current_message(*, content: str, source: str = "discord") -> None:
    global _STREAM_GENERATION_ID

    _ensure_stream_worker()
    _clear_stream_buffer()

    with _STREAM_APPLY_LOCK:
        with _STATE_LOCK:
            _STREAM_GENERATION_ID += 1
            _STATE["current_message"].update(
                {
                    "source": source or "discord",
                    "content": str(content or ""),
                    "streamMode": "Thinking",
                    "streamPreview": "",
                }
            )
            _STATE["signals"]["aiThinking"] = True
            _STATE["signals"]["aiSpeaking"] = False
            _STATE["abort_generation_requested"] = False
            _STATE["stop_audio_requested"] = False
            touch()


def clear_current_message() -> None:
    global _STREAM_GENERATION_ID

    _clear_stream_buffer()
    with _STREAM_APPLY_LOCK:
        with _STATE_LOCK:
            _STREAM_GENERATION_ID += 1
            _STATE["current_message"].update(
                {
                    "source": "Idle",
                    "content": "",
                    "streamMode": "Waiting",
                    "streamPreview": "",
                }
            )
            _STATE["signals"]["aiThinking"] = False
            _STATE["signals"]["aiSpeaking"] = False
            touch()


def append_stream_preview(text: str) -> None:
    if not text:
        return

    _ensure_stream_worker()
    with _STATE_LOCK:
        generation_id = _STREAM_GENERATION_ID
        if _STATE["current_message"].get("streamMode") != "Streaming":
            _STATE["current_message"]["streamMode"] = "Streaming"
        _STATE["signals"]["aiThinking"] = False
        _STATE["signals"]["aiSpeaking"] = True
        touch()

    with _STREAM_BUFFER_CONDITION:
        _STREAM_BUFFER.append((generation_id, str(text)))
        _STREAM_BUFFER_CONDITION.notify()


def finish_generation() -> None:
    flush_stream_preview()
    with _STREAM_APPLY_LOCK:
        with _STATE_LOCK:
            _STATE["current_message"]["streamMode"] = "Done"
            _STATE["signals"]["aiThinking"] = False
            _STATE["signals"]["aiSpeaking"] = False
            touch()


def fail_generation(error: str | None = None) -> None:
    if error:
        append_stream_preview(f"\n[error] {error}")
    flush_stream_preview()
    with _STREAM_APPLY_LOCK:
        with _STATE_LOCK:
            _STATE["current_message"]["streamMode"] = "Error"
            _STATE["signals"]["aiThinking"] = False
            _STATE["signals"]["aiSpeaking"] = False
            touch()


def _ensure_stream_worker() -> None:
    global _STREAM_WORKER_THREAD

    if _STREAM_WORKER_THREAD is not None and _STREAM_WORKER_THREAD.is_alive():
        return

    with _STREAM_BUFFER_CONDITION:
        if _STREAM_WORKER_THREAD is not None and _STREAM_WORKER_THREAD.is_alive():
            return

        _STREAM_WORKER_THREAD = threading.Thread(
            target=_stream_preview_worker,
            name="control-panel-stream-preview",
            daemon=True,
        )
        _STREAM_WORKER_THREAD.start()


def _stream_preview_worker() -> None:
    while True:
        with _STREAM_BUFFER_CONDITION:
            while not _STREAM_BUFFER:
                _STREAM_BUFFER_CONDITION.wait()
            _STREAM_BUFFER_CONDITION.wait(timeout=_STREAM_FLUSH_INTERVAL_S)

        flush_stream_preview()


def _clear_stream_buffer() -> None:
    with _STREAM_BUFFER_CONDITION:
        _STREAM_BUFFER.clear()


def flush_stream_preview() -> None:
    with _STREAM_BUFFER_CONDITION:
        if not _STREAM_BUFFER:
            return

        chunks = list(_STREAM_BUFFER)
        _STREAM_BUFFER.clear()

    chunks_by_generation: dict[int, list[str]] = {}
    for generation_id, text in chunks:
        chunks_by_generation.setdefault(generation_id, []).append(text)

    with _STREAM_APPLY_LOCK:
        for generation_id, texts in chunks_by_generation.items():
            _apply_stream_preview_text(generation_id, "".join(texts))


def _apply_stream_preview_text(generation_id: int, text: str) -> None:
    if not text:
        return

    with _STATE_LOCK:
        if generation_id != _STREAM_GENERATION_ID:
            return

        preview = _STATE["current_message"].get("streamPreview") or ""
        preview = f"{preview}{text}"
        if len(preview) > _STREAM_PREVIEW_MAX_CHARS:
            preview = preview[-_STREAM_PREVIEW_MAX_CHARS:]

        _STATE["current_message"]["streamPreview"] = preview
        _STATE["current_message"]["streamMode"] = "Streaming"
        _STATE["signals"]["aiThinking"] = False
        _STATE["signals"]["aiSpeaking"] = True
        touch()


def set_vision_enabled(value: bool) -> None:
    with _STATE_LOCK:
        _STATE["vision_enabled"] = bool(value)
        touch()


def set_vision_result(path: str | None, error: str | None = None) -> None:
    with _STATE_LOCK:
        _STATE["vision_last_path"] = path
        _STATE["vision_error"] = error
        touch()


def set_temporary_context(content: str) -> None:
    with _STATE_LOCK:
        _STATE["temporary_context"] = str(content or "")
        touch()


def temporary_context() -> str:
    with _STATE_LOCK:
        return str(_STATE.get("temporary_context") or "")


def request_abort() -> None:
    with _STATE_LOCK:
        _STATE["abort_generation_requested"] = True
        _STATE["stop_audio_requested"] = True
        touch()


def abort_generation_requested() -> bool:
    with _STATE_LOCK:
        return bool(_STATE["abort_generation_requested"])


def clear_abort_generation() -> None:
    with _STATE_LOCK:
        _STATE["abort_generation_requested"] = False
        touch()


def stop_audio_requested() -> bool:
    with _STATE_LOCK:
        return bool(_STATE["stop_audio_requested"])


def clear_stop_audio() -> None:
    with _STATE_LOCK:
        _STATE["stop_audio_requested"] = False
        touch()


def tts_disabled() -> bool:
    with _STATE_LOCK:
        return bool(_STATE["controls"].get("ttsDisabled"))


def llm_disabled() -> bool:
    with _STATE_LOCK:
        return bool(_STATE["controls"].get("llmDisabled"))


def bot_disabled() -> bool:
    with _STATE_LOCK:
        return bool(_STATE["controls"].get("botDisabled"))


def stt_muted() -> bool:
    with _STATE_LOCK:
        return bool(_STATE["controls"].get("muted"))


def vtuber_disabled() -> bool:
    with _STATE_LOCK:
        return bool(_STATE["controls"].get("vtuberDisabled"))
