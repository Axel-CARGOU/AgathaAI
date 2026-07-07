import asyncio, subprocess, logging, signal, platform, websockets, sys, threading, aiohttp, json, argparse, time
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI
from src.config.config import settings, save_shutdown_time, load_last_shutdown, cfg_edit
from src.control_panel import runtime as cp_runtime
from src.control_panel.moderation import (
    add_blacklist_word,
    delete_blacklist_word,
    load_blacklist,
    reload_blacklist,
    save_blacklist,
)

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--debug', action='store_true', help="Active le DEV_MODE au lancement.")
parser.add_argument(
    '-t', 
    '--test', 
    type=str, 
    help="Test une sous-fonction en particulier sans charger tout le pipeline.\nDisponibles : VTS, CHESS, TWITCH"
    )
parser.add_argument('-nr', '--no_rag', action='store_true', help="Désactive le RAG au lancement.")
args, _ = parser.parse_known_args()

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

backend = FastAPI()
ws_clients = {
    "discord": set(),
    "tts": set(),
    "dummy_audio": set(),
    "vision": set(),
    "control_panel": set(),
    "unknown": set(),
}

ws, discord_bot = None, None
llm_model, tts_model = None, None
load_llm, load_tts = None, None
unload_llm, unload_tts = None, None
non_stream_output, stream_output = None, None
non_stream_audio, stream_audio = None, None
control_panel_broadcast_task = None
control_panel_frontend_process = None
metrics_auto_renderer_process = None
twitch_chat_reader_task = None
game_tasks = {}
vision_capture_requests = {}
VISION_CAPTURE_TIMEOUT_S = 5.0
TWITCH_CHAT_READ_INTERVAL_S = 10.0
last_discord_prompt_at = time.monotonic()

CONTROL_PANEL_SETTING_KEYS = [
    "LANGUAGE",
    "DEV_MODE",
    "VTUBING",
    "VTS_PORT",
    "RAG_ENABLED",
    "RAG_QDRANT_PATH",
    "RAG_COLLECTION_NAME",
    "RAG_SHORT_TERM_LIMIT",
    "RAG_LONG_TERM_CONTEXT_LIMIT",
    "RAG_EMBEDDING_MODEL",
    "RAG_HASH_EMBEDDING_SIZE",
    "RAG_MAX_CONTEXT_CHARS",
    "RAG_MAX_ENTRY_CHARS",
    "RAG_MAX_STORED_CHARS",
    "LLM_MODEL_PATH",
    "LLM_MAX_NEW_TOKENS",
    "LLM_TEMPERATURE",
    "LLM_TOP_P",
    "LLM_MIN_P",
    "LLM_MAX_SEQ_LEN",
    "LLM_REPETITION_PENALTY",
    "LLM_PRESENCE_PENALTY",
    "KV_Q4",
    "LLM_STARTUP_TIMEOUT",
    "LLM_REQUEST_TIMEOUT",
    "LLM_GPU_MAX_USE",
    "EDIT_FILE",
    "FILES_SEARCH",
    "WEB_SEARCH",
    "TWITCH_CHAT",
    "TWITCH_POLL",
    "TWITCH_BAN",
    "TWITCH_TIMEOUT",
    "VOICE_SAMPLE",
    "SAMPLE_RATE",
    "STT_MODEL",
]

CONTROL_PANEL_TOGGLE_SETTINGS = {
    "devMode": "DEV_MODE",
    "webSearch": "WEB_SEARCH",
    "fileSearch": "FILES_SEARCH",
    "editFile": "EDIT_FILE",
    "twitchChat": "TWITCH_CHAT",
    "twitchPoll": "TWITCH_POLL",
    "twitchBan": "TWITCH_BAN",
    "twitchTimeout": "TWITCH_TIMEOUT",
}

CONTROL_PANEL_CONTROL_KEYS = {
    "bot": "botDisabled",
    "llm": "llmDisabled",
    "tts": "ttsDisabled",
    "mute": "muted",
    "vtuber": "vtuberDisabled",
}

CONTROL_PANEL_INTERNAL_COMMAND_TYPES = {
    "stream_intro",
    "stream_outro",
    "tweet_shitpost",
    "tweet_stream_schedule",
}

CONTROL_PANEL_SETTING_LOCKS = {
    "VTS_PORT": "vtuber",
    "LLM_MODEL_PATH": "llm",
    "LLM_MAX_SEQ_LEN": "llm",
    "KV_Q4": "llm",
    "LLM_GPU_MAX_USE": "llm",
    "RAG_QDRANT_PATH": "rag",
    "RAG_COLLECTION_NAME": "rag",
    "RAG_EMBEDDING_MODEL": "rag",
    "RAG_HASH_EMBEDDING_SIZE": "rag",
    "SAMPLE_RATE": "stt",
}

CONTEXT_DIR = Path(__file__).resolve().parent / "src" / "AI" / "LLM" / "contexts"
MODERATION_DIR = Path(__file__).resolve().parent / "web" / "moderation"
BLACKLIST_FILE = MODERATION_DIR / "blacklist.json"
STATES_PATH = Path(__file__).resolve().parent / "src" / "config" / "states.json"

CONTEXT_FILE_MAP = {
    "main": "main_context.json",
    "format": "format_context.json",
    "emotes": "emotes_context.json",
    "tools": "tool_context.json",
    "twitch": "twitch_context.json",
}
_context_cache_lock = threading.Lock()
_json_context_cache = {}
_prompt_contexts_cache = {"signature": None, "value": None}
_seen_by_llm_cache = {"signature": None, "value": ""}
_metrics_snapshot_cache = {"expires_at": 0.0, "value": None}

@backend.get("/")
async def root():
    return {"ok": True}


def _json_response(message_type, payload=None, request_id=None, ok=True):
    body = {
        "type": message_type,
        "ok": ok,
        "payload": payload or {},
    }

    if request_id is not None:
        body["request_id"] = request_id

    return json.dumps(body)


def _setting_snapshot():
    return {
        key: getattr(settings, key)
        for key in CONTROL_PANEL_SETTING_KEYS
    }


def _context_path_for_name(name):
    file_name = CONTEXT_FILE_MAP.get(name)
    if file_name:
        return CONTEXT_DIR / file_name
    return CONTEXT_DIR / f"{name}_context.json"


def _path_signature(path):
    try:
        stat = Path(path).stat()
        return (str(path), stat.st_size, stat.st_mtime_ns)
    except OSError:
        return (str(path), None, None)


def _context_files_signature():
    return tuple(
        _path_signature(_context_path_for_name(name))
        for name in ("main", "format", "emotes", "tools", "twitch")
    )


def _load_json_context(name):
    path = _context_path_for_name("tools" if name == "tool" else name)
    signature = _path_signature(path)

    with _context_cache_lock:
        cached = _json_context_cache.get(str(path))
        if cached and cached["signature"] == signature:
            return cached["value"]

    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)

    with _context_cache_lock:
        _json_context_cache[str(path)] = {
            "signature": signature,
            "value": value,
        }

    return value


def _load_prompt_contexts():
    signature = _context_files_signature()
    with _context_cache_lock:
        if _prompt_contexts_cache["signature"] == signature and _prompt_contexts_cache["value"] is not None:
            return dict(_prompt_contexts_cache["value"])

    contexts = {}
    for name in ("main", "format", "emotes", "tool", "twitch"):
        try:
            contexts["tools" if name == "tool" else name] = json.dumps(
                _load_json_context(name),
                indent=2,
                ensure_ascii=False,
            )
        except Exception as exc:
            contexts["tools" if name == "tool" else name] = f"[unable to load {name}_context.json] {exc}"

    with _context_cache_lock:
        _prompt_contexts_cache["signature"] = signature
        _prompt_contexts_cache["value"] = dict(contexts)

    return contexts


def _save_prompt_context(name, content):
    if name not in CONTEXT_FILE_MAP:
        raise KeyError(f"Unknown context: {name}")

    parsed = json.loads(str(content or "{}"))
    path = CONTEXT_DIR / CONTEXT_FILE_MAP[name]
    tmp_path = path.with_suffix(".tmp")

    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)
        f.write("\n")

    tmp_path.replace(path)


def _load_moderation_blacklist():
    return load_blacklist()


def _save_moderation_blacklist(words):
    return save_blacklist(words)


def _add_blacklist_word(word):
    words = add_blacklist_word(word)
    reload_blacklist()
    logger.info("\nModeration blacklist updated: added %r (%s words)", word, len(words))
    return words


def _delete_blacklist_word(word):
    words = delete_blacklist_word(word)
    reload_blacklist()
    logger.info("\nModeration blacklist updated: deleted %r (%s words)", word, len(words))
    return words


def _memory_snapshot():
    try:
        from src.AI.RAG.rag import rag_memory

        return [entry.to_payload() for entry in rag_memory.short_term_snapshot()]
    except Exception:
        return []


def _metrics_snapshot():
    now = time.monotonic()
    cached = _metrics_snapshot_cache["value"]
    if cached is not None and now < _metrics_snapshot_cache["expires_at"]:
        return cached

    try:
        from src.metrics.metrics import current_metrics_snapshot

        snapshot = current_metrics_snapshot()
    except Exception as exc:
        snapshot = {"error": str(exc), "files": []}

    process = metrics_auto_renderer_process
    if process is None:
        status = "Stopped"
        pid = None
    elif process.returncode is None:
        status = "Running"
        pid = process.pid
    else:
        status = f"Exited {process.returncode}"
        pid = process.pid

    snapshot["autoRenderer"] = {
        "status": status,
        "pid": pid,
        "intervalSeconds": 5,
    }

    _metrics_snapshot_cache["value"] = snapshot
    _metrics_snapshot_cache["expires_at"] = now + 2.0
    return snapshot


def _load_states():
    try:
        with STATES_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    return data if isinstance(data, dict) else {}


def _save_states(data):
    STATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = STATES_PATH.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp_path.replace(STATES_PATH)


def _update_states(**updates):
    data = _load_states()
    data.update(updates)
    _save_states(data)
    return data


def _mark_discord_prompt_received():
    global last_discord_prompt_at
    last_discord_prompt_at = time.monotonic()


def _bot_is_in_voice_channel():
    return _load_states().get("is_in_vc") is True


def _control_panel_discord_state():
    return {
        "inVoice": _bot_is_in_voice_channel(),
    }


def _format_internal_command(payload):
    command_type = str(payload.get("type") or "").strip().lower()
    if command_type not in CONTROL_PANEL_INTERNAL_COMMAND_TYPES:
        allowed = ", ".join(sorted(CONTROL_PANEL_INTERNAL_COMMAND_TYPES))
        raise ValueError(f"Unknown internal command type: {command_type or '<empty>'}. Allowed: {allowed}")

    language = str(payload.get("language") or settings.LANGUAGE or "FR").strip().upper()
    if language not in {"FR", "EN"}:
        raise ValueError("Internal command language must be FR or EN")

    context = str(payload.get("context") or "").strip()
    instruction = str(payload.get("instruction") or "").strip()
    if not instruction:
        raise ValueError("Internal command instruction is required")

    return "\n".join([
        "[INTERNAL_COMMAND]",
        f"type: {command_type}",
        f"language: {language}",
        f"context: {context}",
        f"instruction: {instruction}",
    ])


def _ensure_internal_command_ready():
    if not _bot_is_in_voice_channel():
        raise RuntimeError("Agatha must be connected to a Discord voice channel first")

    if cp_runtime.bot_disabled() or cp_runtime.llm_disabled() or cp_runtime.tts_disabled():
        raise RuntimeError("Bot, LLM and TTS must be enabled before sending an internal command")

    modules_active = cp_runtime.snapshot()["modules_active"]
    if not modules_active.get("bot") or not modules_active.get("llm") or not modules_active.get("tts"):
        raise RuntimeError("Bot, LLM and TTS modules must be active before sending an internal command")

    if stream_output is None:
        raise RuntimeError("LLM stream pipeline is not ready")


async def _run_internal_command(content):
    _mark_discord_prompt_received()
    cp_runtime.set_current_message(
        content=content,
        source="control_panel_command",
    )
    await _broadcast_control_panel_state(full=False)

    logger.info("\nControl panel internal command queued:\n%s", content)
    try:
        await stream_output(
            {
                "content": content,
                "user_name": "Control Panel",
                "user_id": "control_panel",
                "image_url": None,
                "channel_id": None,
                "message_id": None,
                "guild_id": None,
                "is_dm": False,
                "attachments": [],
                "source": "control_panel_command",
                "preformatted": True,
                "speaker_turns": None,
            },
            "control_panel_command",
        )
    except Exception as exc:
        cp_runtime.fail_generation(str(exc))
        logger.exception("\nControl panel internal command failed: %s", exc)
    finally:
        await _broadcast_control_panel_state(full=False)


async def _handle_internal_command(payload):
    content = _format_internal_command(payload)
    _ensure_internal_command_ready()
    asyncio.create_task(_run_internal_command(content))


def _format_twitch_chat_message(message):
    user_name = (
        message.get("user_name")
        or message.get("user_login")
        or "Twitch chat"
    )
    content = str(message.get("content") or "").strip()
    return user_name, content


def _format_twitch_poll_result_message(result):
    title = str(result.get("title") or "Sondage Twitch").strip()
    status = str(result.get("status") or "completed").strip()
    choices = result.get("choices") or []
    winner = result.get("winner") or {}
    winner_title = str(winner.get("title") or "").strip()

    choice_lines = []
    for choice in choices:
        choice_title = str(choice.get("title") or "Choix sans nom").strip()
        votes = int(choice.get("votes") or 0)
        choice_lines.append(f"- {choice_title}: {votes} vote{'s' if votes != 1 else ''}")

    if not choice_lines:
        choice_lines.append("- Aucun vote exploitable")

    winner_line = f"Gagnant: {winner_title}" if winner_title else "Gagnant: aucun"
    content = "\n".join(
        [
            "[TWITCH_POLL_RESULT]",
            f"Titre: {title}",
            f"Statut: {status}",
            "Choix:",
            *choice_lines,
            winner_line,
            "",
            "Réagis naturellement en stream. Ne fais pas un rapport statistique froid.",
        ]
    )
    return "Résultat du sondage Twitch", content


async def _twitch_chat_reader_loop():
    global last_discord_prompt_at

    last_seen_prompt_at = last_discord_prompt_at

    while True:
        await asyncio.sleep(TWITCH_CHAT_READ_INTERVAL_S)

        if last_discord_prompt_at > last_seen_prompt_at:
            last_seen_prompt_at = last_discord_prompt_at
            logger.debug(
                "\n[TWITCH] Auto chat warning: skipped because a text/voice prompt arrived in the last %.0fs.",
                TWITCH_CHAT_READ_INTERVAL_S,
            )
            continue

        is_in_vc = _bot_is_in_voice_channel()
        twitch_reader_enabled = settings.TWITCH_CHAT or settings.TWITCH_POLL
        if not twitch_reader_enabled or not is_in_vc:
            logger.debug(
                "\n[TWITCH] Auto reader warning: skipped because TWITCH_CHAT=%s, TWITCH_POLL=%s and is_in_vc=%s.",
                bool(settings.TWITCH_CHAT),
                bool(settings.TWITCH_POLL),
                is_in_vc,
            )
            continue

        if cp_runtime.bot_disabled() or cp_runtime.llm_disabled() or cp_runtime.tts_disabled():
            logger.debug(
                "\n[TWITCH] Auto chat warning: skipped because a required module is disabled "
                "(bot=%s, llm=%s, tts=%s).",
                not cp_runtime.bot_disabled(),
                not cp_runtime.llm_disabled(),
                not cp_runtime.tts_disabled(),
            )
            continue

        try:
            from src.streaming.twitch.tw_plugin import _read_chat_msg, _read_poll_result

            poll_result = await _read_poll_result() if settings.TWITCH_POLL else None
            message = None if poll_result else await _read_chat_msg() if settings.TWITCH_CHAT else None
        except Exception as exc:
            logger.debug(
                "\n[TWITCH] Auto reader warning: unable to read Twitch event: %s",
                exc,
            )
            continue

        if poll_result:
            user_name, content = _format_twitch_poll_result_message(poll_result)
            prompt_source = "twitch_poll"
            prompt_user_id = None
            prompt_message_id = poll_result.get("poll_id")
            preformatted = True
        elif message:
            user_name, content = _format_twitch_chat_message(message)
            prompt_source = "twitch_chat"
            prompt_user_id = message.get("user_id")
            prompt_message_id = message.get("message_id")
            preformatted = False
        else:
            logger.debug(
                "\n[TWITCH] Auto reader warning: no chat message or poll result found, retrying in %.0fs.",
                TWITCH_CHAT_READ_INTERVAL_S,
            )
            continue

        if not content:
            logger.debug(
                "\n[TWITCH] Auto reader warning: empty Twitch event found, retrying in %.0fs.",
                TWITCH_CHAT_READ_INTERVAL_S,
            )
            continue

        if stream_output is None:
            logger.debug("\n[TWITCH] Auto chat warning: stream_output is not ready yet.")
            continue

        logger.info("\n[TWITCH] Auto prompt from %s: %s", user_name, content)
        cp_runtime.set_current_message(
            content=content if preformatted else f"{user_name} : {content}",
            source=prompt_source,
        )
        await _broadcast_control_panel_state(full=False)

        try:
            await stream_output(
                {
                    "content": content,
                    "user_name": user_name,
                    "user_id": prompt_user_id,
                    "image_url": None,
                    "channel_id": None,
                    "message_id": prompt_message_id,
                    "guild_id": None,
                    "is_dm": False,
                    "attachments": [],
                    "source": prompt_source,
                    "preformatted": preformatted,
                },
                prompt_source,
            )
        finally:
            await _broadcast_control_panel_state(full=False)


def _games_snapshot():
    states = _load_states()
    lichess_task = game_tasks.get("lichess")
    task_running = lichess_task is not None and not lichess_task.done()
    lichess_running = bool(states.get("is_playing_chess")) or task_running

    return {
        "lichess": {
            "status": "Running" if lichess_running else "Stopped",
        },
    }


def _prompt_language():
    return "fr" if str(settings.LANGUAGE).lower().startswith("fr") else "en"


def _build_seen_by_llm_context(temporary_context=None):
    try:
        language = _prompt_language()
        temporary_context = (
            cp_runtime.temporary_context()
            if temporary_context is None
            else str(temporary_context or "")
        )
        signature = (
            language,
            bool(settings.WEB_SEARCH),
            bool(settings.EDIT_FILE),
            bool(settings.FILES_SEARCH),
            bool(settings.TWITCH_CHAT),
            bool(settings.TWITCH_POLL),
            bool(settings.TWITCH_BAN),
            bool(settings.TWITCH_TIMEOUT),
            temporary_context,
            _context_files_signature(),
        )

        with _context_cache_lock:
            if _seen_by_llm_cache["signature"] == signature:
                return _seen_by_llm_cache["value"]

        format_ctx = _load_json_context("format")
        main_ctx = _load_json_context("main")
        tool_ctx = _load_json_context("tool")
        twitch_ctx = _load_json_context("twitch")
        emotes_ctx = _load_json_context("emotes")

        tool_mode = settings.WEB_SEARCH or settings.EDIT_FILE or settings.FILES_SEARCH
        twitch_mode = settings.TWITCH_CHAT or settings.TWITCH_POLL or settings.TWITCH_BAN or settings.TWITCH_TIMEOUT

        if tool_mode:
            system_full = format_ctx.get(f"tool_format_{language}", "")
        elif twitch_mode:
            system_full = format_ctx.get(f"twitch_format_{language}", "")
        else:
            system_full = format_ctx.get(f"normal_format_{language}", "")

        system_full = f"{system_full}\n{main_ctx.get(language, '')}"

        if tool_mode:
            system_full = (
                f"{system_full}\n"
                f"{emotes_ctx.get(f'emotes_tool_{language}', '')}\n"
                f"{tool_ctx.get(f'wrap_begin_{language}', '')}"
                f"{tool_ctx.get(f'read_file_{language}', '') if settings.FILES_SEARCH else ''}"
                f"{tool_ctx.get(f'edit_file_{language}', '') if settings.EDIT_FILE else ''}"
                f"{tool_ctx.get(f'websearch_{language}', '') if settings.WEB_SEARCH else ''}"
                f"{tool_ctx.get(f'wrap_end_{language}', '')}"
            )
        elif twitch_mode:
            system_full = (
                f"{system_full}\n"
                f"{emotes_ctx.get(f'emotes_twitch_{language}', '')}\n"
                f"{twitch_ctx.get(f'wrap_begin_{language}', '')}"
                f"{twitch_ctx.get(f'poll_{language}', '') if settings.TWITCH_POLL else ''}"
                f"{twitch_ctx.get(f'ban_{language}', '') if settings.TWITCH_BAN else ''}"
                f"{twitch_ctx.get(f'timeout_{language}', '') if settings.TWITCH_TIMEOUT else ''}"
                f"{twitch_ctx.get(f'wrap_end_{language}', '')}"
            )
        else:
            system_full = f"{system_full}\n{emotes_ctx.get(f'emotes_normal_{language}', '')}"

        temporary_context = temporary_context.strip()
        if temporary_context:
            system_full = f"{temporary_context}\n{system_full}"

        with _context_cache_lock:
            _seen_by_llm_cache["signature"] = signature
            _seen_by_llm_cache["value"] = system_full

        return system_full
    except Exception as exc:
        logger.exception("\nUnable to build control panel LLM context preview: %s", exc)
        return f"[unable to build Seen by LLM preview] {exc}"


def _twitch_messages_snapshot(limit=50):
    try:
        from src.streaming.twitch import tw_plugin

        if tw_plugin.plugin is None:
            return []

        messages = list(tw_plugin.plugin._chat_messages)
        return messages[-limit:] if limit else messages
    except Exception:
        return []


def _control_panel_runtime_state(*, scope="realtime"):
    runtime = cp_runtime.snapshot()
    return {
        "scope": scope,
        "runtime": {
            "modulesActive": runtime["modules_active"],
            "signals": runtime["signals"],
            "controls": runtime["controls"],
        },
        "currentMessage": runtime["current_message"],
        "discord": _control_panel_discord_state(),
        "vision": {
            "enabled": runtime["vision_enabled"],
            "lastPath": runtime["vision_last_path"],
            "error": runtime["vision_error"],
        },
        "twitch": {
            "messages": _twitch_messages_snapshot(),
        },
    }


def _control_panel_state(*, memories=None):
    runtime = cp_runtime.snapshot()
    state = _control_panel_runtime_state(scope="full")
    state.update({
        "toggles": {
            "devMode": settings.DEV_MODE,
            "vision": runtime["vision_enabled"],
            "webSearch": settings.WEB_SEARCH,
            "fileSearch": settings.FILES_SEARCH,
            "editFile": settings.EDIT_FILE,
            "twitchChat": settings.TWITCH_CHAT,
            "twitchPoll": settings.TWITCH_POLL,
            "twitchBan": settings.TWITCH_BAN,
            "twitchTimeout": settings.TWITCH_TIMEOUT,
        },
        "settingsValues": _setting_snapshot(),
        "contexts": _load_prompt_contexts(),
        "seenByLlm": _build_seen_by_llm_context(runtime.get("temporary_context", "")),
        "temporaryContext": runtime.get("temporary_context", ""),
        "memories": memories if memories is not None else _memory_snapshot(),
        "moderation": {
            "blacklist": _load_moderation_blacklist(),
        },
        "games": _games_snapshot(),
        "metrics": _metrics_snapshot(),
        "twitch": {
            "messages": _twitch_messages_snapshot(),
        },
    })
    return state


async def _send_json_safe(websocket, message):
    try:
        await websocket.send(message)
        return True
    except websockets.exceptions.ConnectionClosed:
        return False
    except Exception as exc:
        logger.warning("\nFailed to send WebSocket message: %s", exc)
        return False


async def _broadcast_to_role(role, message):
    clients = list(ws_clients.get(role, set()))
    if not clients:
        return

    results = await asyncio.gather(
        *(_send_json_safe(client, message) for client in clients),
        return_exceptions=True,
    )

    for client, result in zip(clients, results):
        if result is False or isinstance(result, Exception):
            ws_clients[role].discard(client)


async def _broadcast_control_panel_state(*, full=True, state=None):
    if not ws_clients["control_panel"]:
        return

    if state is None:
        state = _control_panel_state() if full else _control_panel_runtime_state()
    await _broadcast_to_role(
        "control_panel",
        _json_response("control_panel.state", state),
    )


async def _control_panel_state_publisher():
    while True:
        await asyncio.sleep(0.75)
        await _broadcast_control_panel_state(full=False)


async def _send_discord_control_state():
    controls = cp_runtime.snapshot()["controls"]
    await _broadcast_to_role(
        "discord",
        json.dumps({
            "type": "discord.control",
            "payload": {
                "botDisabled": controls["botDisabled"],
                "muted": controls["muted"],
                "ttsDisabled": controls["ttsDisabled"],
            },
        }),
    )


def _coerce_setting_value(name, value):
    current = getattr(settings, name)

    if isinstance(current, bool):
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)

    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)

    if isinstance(current, float):
        return float(value)

    return str(value)


def _ensure_setting_editable(key):
    if key not in CONTROL_PANEL_SETTING_KEYS:
        raise KeyError(f"Unknown setting: {key}")

    module_name = CONTROL_PANEL_SETTING_LOCKS.get(key)
    if not module_name:
        return

    if cp_runtime.snapshot()["modules_active"].get(module_name):
        raise RuntimeError(f"{key} is locked while {module_name.upper()} is active")


def _vision_output_path():
    output_dir = Path("web") / "runtime" / "vision"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return output_dir / f"screen_{stamp}.png"


def _windows_path_for(path):
    resolved = Path(path).resolve()
    if platform.system().lower() == "windows":
        return str(resolved)

    try:
        return subprocess.check_output(
            ["wslpath", "-w", str(resolved)],
            text=True,
            encoding="utf-8",
            errors="replace",
        ).strip()
    except Exception:
        return str(resolved)


def _cleanup_old_vision_screens(output_dir, keep=12):
    try:
        screenshots = sorted(
            output_dir.glob("screen_*.png"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        for screenshot in screenshots[keep:]:
            screenshot.unlink(missing_ok=True)
    except Exception:
        pass


def _grab_screen_with_pil(output_path):
    from PIL import ImageGrab

    image = ImageGrab.grab(all_screens=False)
    image.save(output_path)


def _grab_screen_with_mss(output_path):
    import mss
    from PIL import Image

    with mss.mss() as screenshotter:
        monitor = screenshotter.monitors[1]
        screenshot = screenshotter.grab(monitor)
        image = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        image.save(output_path)


def _grab_screen_with_powershell(output_path):
    windows_path = subprocess.check_output(
        ["wslpath", "-w", str(output_path.resolve())],
        text=True,
        encoding="utf-8",
        errors="replace",
    ).strip()
    script = r"""
param([string]$Path)
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$bitmap = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size)
$bitmap.Save($Path, [System.Drawing.Imaging.ImageFormat]::Png)
$graphics.Dispose()
$bitmap.Dispose()
"""
    subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script, windows_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _grab_primary_screen():
    output_path = _vision_output_path()
    errors = []

    for grabber in (_grab_screen_with_pil, _grab_screen_with_mss, _grab_screen_with_powershell):
        try:
            grabber(output_path)
            _cleanup_old_vision_screens(output_path.parent)
            return str(output_path.resolve())
        except Exception as exc:
            errors.append(f"{grabber.__name__}: {exc}")

    raise RuntimeError("; ".join(errors))


async def _grab_primary_screen_from_client():
    clients = list(ws_clients.get("vision", set()))
    if not clients:
        raise RuntimeError("Windows vision client is not connected")

    output_path = _vision_output_path()
    request_id = f"vision-{time.time_ns()}"
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    vision_capture_requests[request_id] = future

    message = _json_response(
        "vision.capture_request",
        {
            "outputPath": str(output_path.resolve()),
            "outputPathWindows": _windows_path_for(output_path),
            "display": "primary",
        },
        request_id=request_id,
    )

    try:
        sent = False
        for client in clients:
            if await _send_json_safe(client, message):
                sent = True
                break

        if not sent:
            raise RuntimeError("Windows vision client is disconnected")

        result = await asyncio.wait_for(future, timeout=VISION_CAPTURE_TIMEOUT_S)
        if not result.get("ok", False):
            raise RuntimeError(result.get("error") or "Windows vision client failed")

        if not output_path.exists():
            raise RuntimeError(f"Windows vision client did not create {output_path}")

        _cleanup_old_vision_screens(output_path.parent)
        logger.info("\nWindows vision client screenshot received: %s", output_path.resolve())
        return str(output_path.resolve())
    finally:
        vision_capture_requests.pop(request_id, None)


async def _maybe_attach_vision(llm_input):
    if llm_input.get("image_url") or not cp_runtime.snapshot()["vision_enabled"]:
        return llm_input

    try:
        try:
            image_path = await _grab_primary_screen_from_client()
        except Exception as client_exc:
            if ws_clients.get("vision"):
                logger.warning("\nWindows vision client capture failed, falling back locally: %s", client_exc)
            image_path = await asyncio.to_thread(_grab_primary_screen)
        cp_runtime.set_vision_result(image_path)
        next_input = dict(llm_input)
        content = str(next_input.get("content") or "")
        marker = "[Capture écran actuelle jointe au prompt.]"
        if marker not in content:
            next_input["content"] = f"{content}\n\n{marker}".strip()
        next_input["image_url"] = image_path
        next_input["attachments"] = [
            *(next_input.get("attachments") or []),
            {
                "name": Path(image_path).name,
                "url": image_path,
                "contentType": "image/png",
                "source": "control_panel_vision",
            },
        ]
        logger.info("\nControl panel vision attached screenshot to LLM prompt: %s", image_path)
        return next_input
    except Exception as exc:
        cp_runtime.set_vision_result(None, str(exc))
        logger.warning("\nControl panel vision screenshot failed: %s", exc)
        return llm_input

def stream_logs(pipe, log_func, prefix=""):
    try:
        with pipe:
            for line in iter(pipe.readline, ''):
                line = line.rstrip()
                if line:
                    log_func(f"\n{prefix}: {line}")
    except Exception as e:
        logger.error(f"\nError while reading subprocess pipe: {e}\n")


async def stream_async_logs(stream, log_func, prefix=""):
    if stream is None:
        return
    try:
        while True:
            line = await stream.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            if text:
                log_func(f"\n{prefix}: {text}")
    except Exception as e:
        logger.error(f"\nError while reading async subprocess pipe: {e}\n")


async def check_internet():
    test_urls = [
        "https://clients3.google.com/generate_204",
        "https://www.gstatic.com/generate_204",
    ]

    timeout = aiohttp.ClientTimeout(total=1)

    for url in test_urls:
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status in (200, 204):
                        logger.info("\nInternet connection check passed.")
                        return True
        except Exception:
            pass

    logger.error("\nInternet connection check failed. Internet access is required to run this app.")
    await shutdown()
    return False

async def mainframe():
    global ws, llm_model, discord_bot, load_llm, unload_llm, non_stream_output, stream_output
    global tts_model, load_tts, unload_tts, non_stream_audio, stream_audio
    global control_panel_broadcast_task, control_panel_frontend_process, metrics_auto_renderer_process
    global twitch_chat_reader_task
    
                                                 
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()
    
    def handler():
        loop.call_soon_threadsafe(shutdown_event.set)

    signal.signal(signal.SIGINT, lambda *_: handler())
    signal.signal(signal.SIGTERM, lambda *_: handler())
    
                 
    t = datetime.now().strftime("Y%Y.M%m.D%d-%Hh.%Mm.%Ss")
    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s : %(message)s', 
        filename=f'logs/logs-{t}.log', 
        level=logging.INFO
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    
    logger.info("\nLogging Started\n")
    
    if args.debug:
        logger.info("\n\nDEV mode activated\n\n")
        logger.setLevel(logging.DEBUG)
        console_handler.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
        cfg_edit("DEV_MODE", True)
    
    logger.info("\nPreparing boot sequence...")
    _update_states(is_in_vc=False)
    logger.info(
    f"\nPython version: {platform.python_version()}\n"
    f"NodeJS version: {subprocess.check_output(['node', '-v'], text=True).strip()}\n"
    f"LLM model : {settings.LLM_MODEL_PATH}, max context = {settings.LLM_MAX_SEQ_LEN}, max tokens = {settings.LLM_MAX_NEW_TOKENS}\n")
    
                               
    await check_internet()
    
                                  
    logger.info("\nLaunching WebSocket server on bind.example:8765...")
    try:
        ws = await websockets.serve(ws_serv, "bind.example", 8765)
        control_panel_broadcast_task = asyncio.create_task(_control_panel_state_publisher())
        logger.info("\nWebsocket server launched")
    except Exception as E:
        logger.exception(f"\nUnable to launch WebSocket server : {E}")
        await shutdown()
    
                                                                    
    
    logger.info("\nControl Panel back end is served by the main WebSocket on bind.example:8765")
    
                                  
    logger.info("\nInitializing Control Panel (front end)...")
    try:
        control_panel_frontend_process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "uvicorn", "web.routes:app",
            "--host", "local.example",
            "--port", "8080"
        )
        logger.info("\nControl Panel (front end) initialized at http://local.example:8080")
    except Exception as E:
        logger.exception(f"\nUnable to start Control Panel (front end) : {E}")
        await shutdown()

    logger.info("\nInitializing metrics auto-renderer...")
    try:
        from src.metrics.metrics import auto_renderer_command

        metrics_auto_renderer_process = await asyncio.create_subprocess_exec(
            *auto_renderer_command(interval=5.0),
            cwd=str(Path(__file__).resolve().parent),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        asyncio.create_task(
            stream_async_logs(metrics_auto_renderer_process.stdout, logger.info, "[Metrics]")
        )
        asyncio.create_task(
            stream_async_logs(metrics_auto_renderer_process.stderr, logger.error, "[Metrics]")
        )
        logger.info("\nMetrics auto-renderer initialized with a 5s refresh interval")
    except Exception as E:
        logger.exception(f"\nUnable to start metrics auto-renderer : {E}")

                         
    logger.info("\nStarting Twitch plugin...")
    try:
        from src.streaming.twitch.tw_plugin import start_twitch_plugin

        await start_twitch_plugin()
        logger.info("\nTwitch plugin boot task started")
    except Exception as E:
        logger.exception(f"\nFailed to start Twitch plugin : {E}")
    
                     
    if args.no_rag:
        logger.info("\nRAG disabled, skipping initialization...")
        cfg_edit("RAG_ENABLED", False)
    else:
        logger.info("\nInitializing RAG memory...")
        try:
            from src.AI.RAG.rag import init_rag_memory

            if await init_rag_memory():
                cp_runtime.set_module_active("rag", True)
                logger.info("\nRAG long-term memory initialized")
            else:
                cp_runtime.set_module_active("rag", False)
                logger.warning("\nRAG long-term memory is unavailable; short-term memory remains active")
        except Exception as E:
            cp_runtime.set_module_active("rag", False)
            logger.exception(f"\nUnable to initialize RAG memory : {E}")
    
                        
    logger.info("\nStarting Discord bot...")
    try:
        discord_bot = subprocess.Popen(
            ["node", "--env-file=.env", "src/discord/bot.js"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1
        )
        
        threading.Thread(
            target=stream_logs,
            args=(discord_bot.stdout, logger.info, "[Discord Bot]"),
            daemon=True
        ).start()
        
        threading.Thread(
            target=stream_logs,
            args=(discord_bot.stderr, logger.error, "[Discord Bot]"),
            daemon=True
        ).start()
        
        cp_runtime.set_module_active("bot", True)
        cp_runtime.set_module_active("stt", True)
        logger.info("Discord bot started.")
    except Exception as E:
        logger.exception(f"Unable to start Discord bot : {E}")
        await shutdown()
    
               
    from src.AI.LLM.llm_agathaai_vision_vllm import (load_llm, unload_llm, non_stream_output, stream_output)
    logger.info("\nLoading LLM...")
    try:
        logger.info("\nLoading AgathaAI Qwen2.5-VL vision & text model (vLLM)")
        llm_model = await load_llm()
        cp_runtime.set_module_active("llm", True)
        logger.info("\nLLM loaded")
    except Exception as E:
        logger.exception(f"\nFailed to load LLM : {E}")
        await shutdown()
        
             
    logger.info("\nStarting STT...")
    logger.info("\nCurrent STT is Google STT API, the Discord bot script will also handle the STT")
    
                
    from src.AI.TTS.tts import (load_tts, unload_tts, non_stream_audio, stream_audio)
    logger.info("\nLoading TTS...")
    try:
        tts_model = await load_tts()
        cp_runtime.set_module_active("tts", True)
        logger.info("\nTTS loaded")
    except Exception as E:
        logger.exception(f"\nFailed to load TTS : {E}")
        await shutdown()
    
    if settings.VTUBING:
        logger.info("\nStarting VTuber plugin...")
        from src.vtuber.vts_plugin import start_vts_plugin
        try:
            await start_vts_plugin()
            cp_runtime.set_module_active("vtuber", True)
            logger.info("\nVTuber plugin loaded in background")
        except Exception as E:
            logger.exception(f"\nFailed to load VTuber plugin : {E}")
            await shutdown()
    else:
        cp_runtime.set_module_active("vtuber", False)
        logger.info("\nVTubing disabled in settings, skipping VTuber plugin.")

    twitch_chat_reader_task = asyncio.create_task(_twitch_chat_reader_loop())
    logger.info("\nTwitch auto chat reader started")
        
    logger.info("\n\nBoot sequence completed")
    
               
    await shutdown_event.wait()
    await shutdown()
    return


def all_ws_clients():
    return [
        client
        for clients in ws_clients.values()
        for client in clients
    ]


async def _set_llm_disabled(disabled):
    global llm_model

    cp_runtime.set_control("llmDisabled", disabled)

    if disabled:
        cp_runtime.request_abort()
        if llm_model and unload_llm:
            await unload_llm()
            llm_model = None
        cp_runtime.set_module_active("llm", False)
        return

    if load_llm and llm_model is None:
        llm_model = await load_llm()
    cp_runtime.set_module_active("llm", llm_model is not None)


async def _set_tts_disabled(disabled):
    global tts_model

    cp_runtime.set_control("ttsDisabled", disabled)

    try:
        from src.AI.TTS.tts import stop_audio_queue

        await stop_audio_queue()
    except Exception as exc:
        logger.warning("\nUnable to stop TTS queue from control panel: %s", exc)

    if disabled:
        if tts_model and unload_tts:
            await unload_tts()
            tts_model = None
        cp_runtime.set_module_active("tts", False)
        return

    if load_tts and tts_model is None:
        tts_model = await load_tts()
    cp_runtime.set_module_active("tts", tts_model is not None)


async def _set_rag_enabled(enabled, *, persist=True):
    if persist:
        cfg_edit("RAG_ENABLED", bool(enabled))

    if not enabled:
        try:
            from src.AI.RAG.rag import reset_rag_memory

            await reset_rag_memory()
        except Exception as exc:
            logger.warning("\nUnable to reset RAG from control panel: %s", exc)
        cp_runtime.set_module_active("rag", False)
        return

    try:
        from src.AI.RAG.rag import init_rag_memory, reset_rag_memory

        if not cp_runtime.snapshot()["modules_active"].get("rag"):
            await reset_rag_memory()

        ready = await init_rag_memory()
        cp_runtime.set_module_active("rag", ready)
        if not ready:
            logger.warning("\nRAG long-term memory is unavailable; short-term memory remains active")
    except Exception as exc:
        cp_runtime.set_module_active("rag", False)
        logger.warning("\nUnable to initialize RAG from control panel: %s", exc)


async def _set_vtuber_disabled(disabled, *, persist=True):
    cp_runtime.set_control("vtuberDisabled", disabled)

    if disabled:
        if persist:
            cfg_edit("VTUBING", False)
        try:
            from src.vtuber import vts_plugin

            await vts_plugin.stop_vts_plugin()
        except Exception as exc:
            logger.warning("\nUnable to close VTuber plugin from control panel: %s", exc)
        cp_runtime.set_module_active("vtuber", False)
        return

    if persist:
        cfg_edit("VTUBING", True)
    try:
        from src.vtuber.vts_plugin import start_vts_plugin

        await start_vts_plugin()
        cp_runtime.set_module_active("vtuber", True)
    except Exception as exc:
        cp_runtime.set_module_active("vtuber", False)
        logger.warning("\nUnable to start VTuber plugin from control panel: %s", exc)


async def _apply_setting_side_effect(key, value):
    if key == "DEV_MODE":
        logging.getLogger().setLevel(logging.DEBUG if bool(value) else logging.INFO)
        return

    if key == "VTUBING":
        await _set_vtuber_disabled(not bool(value), persist=False)
        return

    if key == "RAG_ENABLED":
        await _set_rag_enabled(bool(value), persist=False)
        return


async def _handle_control_panel_control(action):
    if action not in CONTROL_PANEL_CONTROL_KEYS:
        raise KeyError(f"Unknown control action: {action}")

    control_key = CONTROL_PANEL_CONTROL_KEYS[action]
    next_value = not cp_runtime.snapshot()["controls"].get(control_key, False)

    if action == "llm":
        await _set_llm_disabled(next_value)
    elif action == "tts":
        await _set_tts_disabled(next_value)
    elif action == "vtuber":
        await _set_vtuber_disabled(next_value)
    elif action == "bot":
        cp_runtime.set_control(control_key, next_value)
        await _send_discord_control_state()
    elif action == "mute":
        cp_runtime.set_control(control_key, next_value)
        await _send_discord_control_state()


def _run_lichess_plugin_blocking():
    try:
        from src.AI.game_ai.chess.Lichess import boot_lichess, stream_events

        _update_states(is_gaming=True, is_playing_chess=True)
        boot_lichess(model_path="src/AI/game_ai/chess/agathaai_chess_ai.pth")
        stream_events()
    finally:
        _update_states(is_gaming=False, is_playing_chess=False)


async def _handle_game_action(action):
    if action == "lichess-start":
        task = game_tasks.get("lichess")
        if task is not None and not task.done():
            return

        _update_states(is_gaming=True, is_playing_chess=True)
        game_tasks["lichess"] = asyncio.create_task(asyncio.to_thread(_run_lichess_plugin_blocking))
        return

    if action == "lichess-stop":
        _update_states(is_gaming=False, is_playing_chess=False)
        try:
            from src.AI.game_ai.chess.Lichess import request_stop

            await asyncio.to_thread(request_stop)
        except Exception as exc:
            logger.warning("\nUnable to request Lichess plugin stop: %s", exc)
        task = game_tasks.get("lichess")
        if task is not None and not task.done():
            task.cancel()
        return

    raise KeyError(f"Unknown game action: {action}")


async def _handle_control_panel_message(websocket, data):
    request_id = data.get("request_id")
    message_type = data.get("type")
    payload = data.get("payload") or {}
    memories_override = None

    try:
        if message_type == "control_panel.get_state":
            pass
        elif message_type == "control_panel.set_toggle":
            key = payload["key"]
            value = bool(payload.get("value"))
            if key == "vision":
                cp_runtime.set_vision_enabled(value)
            elif key in CONTROL_PANEL_TOGGLE_SETTINGS:
                setting_key = CONTROL_PANEL_TOGGLE_SETTINGS[key]
                if getattr(settings, setting_key) != value:
                    cfg_edit(setting_key, value)
                await _apply_setting_side_effect(setting_key, value)
            else:
                raise KeyError(f"Unknown toggle: {key}")
        elif message_type == "control_panel.set_setting":
            key = str(payload["key"]).upper()
            _ensure_setting_editable(key)
            value = _coerce_setting_value(key, payload.get("value"))
            if getattr(settings, key) != value:
                cfg_edit(key, value)
            await _apply_setting_side_effect(key, value)
        elif message_type == "control_panel.control":
            await _handle_control_panel_control(payload["action"])
        elif message_type == "control_panel.abort":
            cp_runtime.request_abort()
            try:
                from src.AI.TTS.tts import stop_audio_queue

                await stop_audio_queue()
            except Exception as exc:
                logger.warning("\nUnable to stop TTS queue from abort button: %s", exc)
        elif message_type == "control_panel.clear_current":
            cp_runtime.clear_current_message()
        elif message_type == "control_panel.signal":
            cp_runtime.set_signal(payload["name"], bool(payload.get("value")))
        elif message_type == "control_panel.set_temporary_context":
            content = payload.get("content") or ""
            cp_runtime.set_temporary_context(content)
            logger.info("\nTemporary Lobotomy context updated (%s chars)", len(content))
        elif message_type == "control_panel.set_context":
            _save_prompt_context(payload["name"], payload.get("content") or "{}")
        elif message_type == "control_panel.internal_command":
            await _handle_internal_command(payload)
        elif message_type == "control_panel.search_memories":
            from src.AI.RAG.rag import list_memory_entries

            memories_override = await list_memory_entries(
                query=payload.get("query") or "",
                limit=int(payload.get("limit") or 100),
            )
        elif message_type == "control_panel.save_memory":
            from src.AI.RAG.rag import add_manual_memory, list_memory_entries, update_memory_entry

            memory_id = payload.get("id")
            if memory_id:
                await update_memory_entry(memory_id, payload)
            else:
                await add_manual_memory(
                    payload.get("user_prompt") or "",
                    payload.get("llm_response") or "",
                    user_name=payload.get("user_name") or "Control Panel",
                )

            memories_override = await list_memory_entries(limit=100)
        elif message_type == "control_panel.delete_memory":
            from src.AI.RAG.rag import delete_memory_entry, list_memory_entries

            await delete_memory_entry(payload["id"])
            memories_override = await list_memory_entries(limit=100)
        elif message_type == "control_panel.clear_short_memories":
            from src.AI.RAG.rag import clear_short_term_entries, list_memory_entries

            await clear_short_term_entries()
            memories_override = await list_memory_entries(limit=100)
        elif message_type == "control_panel.import_memories":
            from src.AI.RAG.rag import add_manual_memory, list_memory_entries

            entries = payload.get("entries") or []
            if not isinstance(entries, list):
                raise ValueError("entries must be a list")

            for entry in entries:
                if not isinstance(entry, dict):
                    continue

                await add_manual_memory(
                    entry.get("user_prompt") or entry.get("prompt") or "",
                    entry.get("llm_response") or entry.get("response") or "",
                    user_name=entry.get("user_name") or "Imported",
                )

            memories_override = await list_memory_entries(limit=100)
        elif message_type == "control_panel.add_blacklist_word":
            _add_blacklist_word(payload.get("word"))
        elif message_type == "control_panel.delete_blacklist_word":
            _delete_blacklist_word(payload.get("word"))
        elif message_type == "control_panel.game":
            await _handle_game_action(payload["action"])
        elif message_type == "control_panel.refresh_metrics":
            from src.metrics.metrics import refresh_metrics_graphs

            await refresh_metrics_graphs()
        else:
            raise KeyError(f"Unknown control panel message: {message_type}")

        state = _control_panel_state(memories=memories_override)
        if request_id:
            await websocket.send(_json_response("control_panel.result", {"state": state}, request_id=request_id))
        await _broadcast_control_panel_state(state=state)

    except Exception as exc:
        logger.exception("\nControl panel command failed: %s", exc)
        if request_id:
            await websocket.send(_json_response(
                "control_panel.result",
                {"error": str(exc), "state": _control_panel_state()},
                request_id=request_id,
                ok=False,
            ))

async def shutdown():
    global ws, llm_model, tts_model, control_panel_broadcast_task, control_panel_frontend_process
    global metrics_auto_renderer_process, twitch_chat_reader_task
    logger.info("\nShutting down...")
    try:
        if twitch_chat_reader_task is not None:
            twitch_chat_reader_task.cancel()
            try:
                await twitch_chat_reader_task
            except asyncio.CancelledError:
                pass
            twitch_chat_reader_task = None

        if control_panel_broadcast_task is not None:
            control_panel_broadcast_task.cancel()
            control_panel_broadcast_task = None

        save_shutdown_time()
        try:
            from src.AI.RAG.rag import flush_short_term_memory

            logger.info("\nFlushing RAG short-term memory...")
            flushed_count = await flush_short_term_memory()
            logger.info(f"\nRAG short-term memory flushed to long-term: {flushed_count} entries")
        except Exception as E:
            logger.exception(f"\nUnable to flush RAG short-term memory : {E}")

        clients = all_ws_clients()
        
                      
        if clients:
            logger.info("\nBroadcasting WebSocket shutdown signal to all connected clients...")

            send_results = await asyncio.gather(
                *(client.send("shutdown") for client in clients),
                return_exceptions=True
            )

            for i, result in enumerate(send_results):
                if isinstance(result, Exception):
                    logger.warning(f"\nFailed to send WebSocket shutdown to client #{i + 1}: {result}")

            close_results = await asyncio.gather(
                *(client.close() for client in clients),
                return_exceptions=True
            )

            for i, result in enumerate(close_results):
                if isinstance(result, Exception):
                    logger.warning(f"\nFailed to close WebSocket client #{i + 1}: {result}")

            logger.info("\nWebSocket shutdown signal broadcast completed.")

        if control_panel_frontend_process is not None:
            logger.info("\nStopping Control Panel front end...")
            try:
                control_panel_frontend_process.terminate()
                await asyncio.wait_for(control_panel_frontend_process.wait(), timeout=10)
                logger.info("\nControl Panel front end stopped")
            except asyncio.TimeoutError:
                logger.warning("\nControl Panel front end did not stop after SIGTERM, killing it...")
                control_panel_frontend_process.kill()
                await control_panel_frontend_process.wait()
            except ProcessLookupError:
                pass
            except Exception as E:
                logger.exception(f"\nUnable to stop Control Panel front end : {E}")
            finally:
                control_panel_frontend_process = None
        
        if llm_model:
            logger.info("\nUnloading LLM...")
            try:
                await unload_llm()
                cp_runtime.set_module_active("llm", False)
                logger.info("\nLLM unloaded")
            except Exception as E:
                logger.exception(f"\nUnable to unload LLM : {E}")
        
             
        if tts_model:
            logger.info("\nUnloading TTS...")
            try:
                await unload_tts()
                cp_runtime.set_module_active("tts", False)
                logger.info("\nTTS unloaded")
            except Exception as E:
                logger.exception(f"\nUnable to unload TTS : {E}")

                       
        try:
            from src.streaming.twitch.tw_plugin import stop_twitch_plugin

            logger.info("\nStopping Twitch plugin...")
            await stop_twitch_plugin()
            logger.info("\nTwitch plugin stopped")
        except Exception as E:
            logger.exception(f"\nUnable to stop Twitch plugin : {E}")

                       
        try:
            from src.vtuber.vts_plugin import stop_vts_plugin

            logger.info("\nStopping VTuber plugin...")
            await stop_vts_plugin()
            logger.info("\nVTuber plugin stopped")
        except Exception as E:
            logger.exception(f"\nUnable to stop VTuber plugin : {E}")

                      
        if game_tasks.get("lichess") is not None:
            try:
                from src.AI.game_ai.chess.Lichess import request_stop

                await asyncio.to_thread(request_stop)
            except Exception as E:
                logger.warning("\nUnable to request Lichess stop during shutdown: %s", E)
        
        for module_name in ("bot", "stt", "vtuber", "rag"):
            cp_runtime.set_module_active(module_name, False)

        pending_game_tasks = [
            task
            for task in list(game_tasks.values())
            if task is not None and not task.done()
        ]
        for task in pending_game_tasks:
            task.cancel()
        if pending_game_tasks:
            await asyncio.wait(pending_game_tasks, timeout=3)
        for task in list(game_tasks.values()):
            if task is not None and not task.done():
                task.cancel()
        game_tasks.clear()
        _update_states(is_in_vc=False, is_gaming=False, is_playing_chess=False)
        
                           
        if ws:
            logger.info("\nClosing WebSocket server...")
            try:
                for clients_set in ws_clients.values():
                    clients_set.clear()
                ws.close()
                await ws.wait_closed()
                logger.info("\nWebSocket server closed")
            except Exception as E:
                logger.exception(f"\nUnable to close WebSocket server : {E}")

        if metrics_auto_renderer_process is not None:
            logger.info("\nStopping metrics auto-renderer...")
            try:
                metrics_auto_renderer_process.terminate()
                await asyncio.wait_for(metrics_auto_renderer_process.wait(), timeout=5)
                logger.info("\nMetrics auto-renderer stopped")
            except asyncio.TimeoutError:
                logger.warning("\nMetrics auto-renderer did not stop after SIGTERM, killing it...")
                metrics_auto_renderer_process.kill()
                await metrics_auto_renderer_process.wait()
            except ProcessLookupError:
                pass
            except Exception as E:
                logger.exception(f"\nUnable to stop metrics auto-renderer : {E}")
            finally:
                metrics_auto_renderer_process = None
        
                       
        from src.metrics.metrics import make_vllm_graph_on_shutdown
        await make_vllm_graph_on_shutdown()
                
                        
        if settings.DEV_MODE:
            logger.info("\n\nDEV mode now deactivated\n\n")
            cfg_edit("DEV_MODE", False)
            
        if settings.RAG_ENABLED is False:
            logger.info("\n\nRAG re-enabled\n\n")
            cfg_edit("RAG_ENABLED", True)
                
        logger.info(f"\n\nShutdown complete at : {load_last_shutdown()}")
        sys.exit(1)
    except Exception as E:
        logger.exception(f"Unable to shutdown properly : {E}")
        sys.exit(1)
    return
    

async def ws_serv(websocket):
    role = "unknown"
    ws_clients[role].add(websocket)
    
    try:
        async for message in websocket:
                                                      
            data = json.loads(message)
            message_type = data.get("type")

            if message_type == "ws.identify":
                new_role = data.get("role", "unknown")

                if new_role not in ws_clients:
                    new_role = "unknown"

                ws_clients[role].discard(websocket)
                role = new_role
                ws_clients[role].add(websocket)

                logger.info(f"WS client identified as: {role}")
                if role == "control_panel":
                    await websocket.send(_json_response("control_panel.state", _control_panel_state()))
                continue

            if message_type == "vision.capture_response":
                request_id = data.get("request_id")
                future = vision_capture_requests.get(request_id)
                if future is not None and not future.done():
                    future.set_result(data.get("payload") or {})
                continue

            if message_type == "discord.voice_state":
                payload = data.get("payload") or {}
                is_in_vc = bool(payload.get("is_in_vc"))
                _update_states(is_in_vc=is_in_vc)
                logger.debug("\nDiscord voice state updated: is_in_vc=%s", is_in_vc)
                continue

            if message_type and message_type.startswith("control_panel."):
                await _handle_control_panel_message(websocket, data)
                continue

            if message_type == "discord.llm_prompt":
                pipeline_start = time.perf_counter()
                _mark_discord_prompt_received()

                if cp_runtime.bot_disabled() or cp_runtime.llm_disabled():
                    logger.info("\nDiscord text prompt ignored because Bot or LLM is disabled from control panel.")
                    continue

                payload = data["payload"]
                request_id = data.get("request_id") or f"discord-{time.time_ns()}"
                client_sent_at_ms = payload.get("client_sent_at_ms") or data.get("client_sent_at_ms")
                parse_s = 0.0
                pre_panel_s = 0.0
                vision_s = 0.0
                llm_s = 0.0
                post_panel_s = 0.0
                response_send_s = 0.0
                try:
                    logger.info(f"WS Discord message : {message}")
                    stage_start = time.perf_counter()
                    full_msg = {
                        "content": payload["content"],
                        "image_url": payload.get("image_url"),
                        "user_id": payload["author_id"],
                        "user_name": payload["author_name"],
                        "channel_id": payload["channel_id"],
                        "message_id": payload["message_id"],
                        "guild_id": payload["guild_id"],
                        "is_dm": payload["is_dm"],
                        "attachments": payload.get("attachments") or [],
                        "source": payload.get("source"),
                        "preformatted": payload.get("preformatted", False),
                        "speaker_turns": payload.get("speaker_turns"),
                    }
                    parse_s = time.perf_counter() - stage_start

                    stage_start = time.perf_counter()
                    cp_runtime.set_current_message(
                        content=full_msg["content"],
                        source=full_msg["source"] or "discord",
                    )
                    await _broadcast_control_panel_state(full=False)
                    pre_panel_s = time.perf_counter() - stage_start

                    llm_input = {
                        "user_name": full_msg["user_name"],
                        "user_id": full_msg["user_id"],
                        "content": full_msg["content"],
                        "image_url": full_msg["image_url"],
                        "channel_id": full_msg["channel_id"],
                        "message_id": full_msg["message_id"],
                        "guild_id": full_msg["guild_id"],
                        "is_dm": full_msg["is_dm"],
                        "attachments": full_msg["attachments"],
                        "source": full_msg["source"],
                        "preformatted": full_msg["preformatted"],
                        "speaker_turns": full_msg["speaker_turns"],
                    }

                    stage_start = time.perf_counter()
                    llm_input = await _maybe_attach_vision(llm_input)
                    vision_s = time.perf_counter() - stage_start

                    stage_start = time.perf_counter()
                    llm_reply = await non_stream_output(llm_input, "discord")
                    llm_s = time.perf_counter() - stage_start

                    stage_start = time.perf_counter()
                    cp_runtime.finish_generation()
                    await _broadcast_control_panel_state(full=False)
                    post_panel_s = time.perf_counter() - stage_start

                    timings = {
                        "main_total_s": round(time.perf_counter() - pipeline_start, 4),
                        "main_parse_s": round(parse_s, 4),
                        "main_pre_panel_broadcast_s": round(pre_panel_s, 4),
                        "main_vision_s": round(vision_s, 4),
                        "main_llm_total_s": round(llm_s, 4),
                        "main_post_panel_broadcast_s": round(post_panel_s, 4),
                    }
                    if client_sent_at_ms is not None:
                        try:
                            timings["client_to_main_s"] = round(
                                (time.time() * 1000 - float(client_sent_at_ms)) / 1000,
                                4,
                            )
                        except (TypeError, ValueError):
                            pass

                    stage_start = time.perf_counter()
                    await websocket.send(json.dumps({
                        "type": "discord.llm_response",
                        "request_id": request_id,
                        "content": llm_reply,
                        "channel_id": full_msg["channel_id"],
                        "reply_to_message_id": full_msg["message_id"],
                        "is_dm": full_msg["is_dm"],
                        "guild_id": full_msg["guild_id"],
                        "author_id": full_msg["user_id"],
                        "timings": timings,
                        "client_sent_at_ms": client_sent_at_ms,
                    }))
                    response_send_s = time.perf_counter() - stage_start
                    timings["main_response_ws_send_s"] = round(response_send_s, 4)
                    timings["main_total_s"] = round(time.perf_counter() - pipeline_start, 4)
                    logger.info(
                        "\n[PIPELINE:MAIN_TEXT] request_id=%s timings=%s response_chars=%s",
                        request_id,
                        json.dumps(timings, ensure_ascii=False, sort_keys=True),
                        len(llm_reply or ""),
                    )
                except Exception as E:
                    cp_runtime.fail_generation(str(E))
                    await _broadcast_control_panel_state(full=False)
                    logger.exception(f"\nError in WS server while trying to handle Discord data : {E}")
                    
            if message_type == "discord.llm_prompt_vc":
                _mark_discord_prompt_received()

                if cp_runtime.bot_disabled() or cp_runtime.llm_disabled() or cp_runtime.stt_muted():
                    logger.info("\nDiscord voice prompt ignored because Bot, LLM or STT is disabled from control panel.")
                    continue

                payload = data["payload"]
                try:
                    logger.info(f"WS Discord voice message : {message}")
                    full_msg = {
                        "content": payload["content"],
                        "image_url": payload.get("image_url"),
                        "user_id": payload["author_id"],
                        "user_name": payload["author_name"],
                        "channel_id": payload["channel_id"],
                        "message_id": payload["message_id"],
                        "guild_id": payload["guild_id"],
                        "is_dm": payload["is_dm"],
                        "attachments": payload.get("attachments") or [],
                        "source": payload.get("source"),
                        "preformatted": payload.get("preformatted", False),
                        "speaker_turns": payload.get("speaker_turns"),
                    }
                    cp_runtime.set_current_message(
                        content=full_msg["content"],
                        source=full_msg["source"] or "voice_conversation",
                    )
                    await _broadcast_control_panel_state(full=False)

                    llm_input = {
                        "user_name": full_msg["user_name"],
                        "user_id": full_msg["user_id"],
                        "content": full_msg["content"],
                        "image_url": full_msg["image_url"],
                        "channel_id": full_msg["channel_id"],
                        "message_id": full_msg["message_id"],
                        "guild_id": full_msg["guild_id"],
                        "is_dm": full_msg["is_dm"],
                        "attachments": full_msg["attachments"],
                        "source": full_msg["source"],
                        "preformatted": full_msg["preformatted"],
                        "speaker_turns": full_msg["speaker_turns"],
                    }
                    llm_input = await _maybe_attach_vision(llm_input)
                    await stream_output(llm_input, "discord")                                                                  
                    await _broadcast_control_panel_state(full=False)

                except Exception as E:
                    cp_runtime.fail_generation(str(E))
                    await _broadcast_control_panel_state(full=False)
                    logger.exception(f"\nError in WS server while trying to handle Discord VC data : {E}")
                    
            if message_type == "tts_output":
                try:
                    payload = data["payload"]
            
                    discord_message = json.dumps({
                        "type": "discord.llm_vocal_response",
                        "payload": payload,
                    })
            
                    audio_message = json.dumps({
                        "type": "llm_vocal_response",
                        "payload": payload,
                    })
            
                    async def broadcast(role, message):
                        dead_clients = []
            
                        for client in ws_clients[role]:
                            try:
                                await client.send(message)
                            except websockets.exceptions.ConnectionClosed:
                                dead_clients.append(client)
            
                        for client in dead_clients:
                            ws_clients[role].discard(client)
            
                    await asyncio.gather(
                        broadcast("discord", discord_message),
                        broadcast("dummy_audio", audio_message),
                    )
            
                except Exception as E:
                    logger.exception(f"\nError while sending TTS output: {E}")
            
    except websockets.exceptions.ConnectionClosed:
        logger.info("WebSocket Client connection closed.")
    except Exception as E:
        logger.exception(f"WebSocket server error : {E}")
    finally:
        ws_clients[role].discard(websocket)

async def _test_vts():
    from src.vtuber.vts_plugin import start_vts_plugin
    print("Executing VTS test run...")
    await start_vts_plugin(test=True)
        
async def _test_chess():
    t = datetime.now().strftime("%Y-%m-%d-%Hh.%Mm.%Ss")
    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s : %(message)s', 
        filename=f'logs/TEST_LICHESS_logs-{t}.log', 
        level=logging.INFO
    )
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)
    
    from src.AI.game_ai.chess.Lichess import boot_lichess, stream_events
    from src.AI.LLM.llm_agathaai_vision_vllm import load_llm, unload_llm
    await load_llm()
    
    boot_lichess(model_path="src/AI/game_ai/chess/agathaai_chess_ai.pth")
    stream_events()
    
    await unload_llm()


async def _test_twitch():
    from src.streaming.twitch.tw_plugin import _test_twitch as run_twitch_test

    await run_twitch_test()
    
if __name__ == '__main__':
    if not args.test:
        asyncio.run(mainframe())
    elif "vts" in args.test.lower():
        asyncio.run(_test_vts())        
    elif "twitch" in args.test.lower():
        asyncio.run(_test_twitch())
    elif "chess" in args.test.lower():
        asyncio.run(_test_chess())
