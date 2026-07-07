import asyncio
import io
import json
import logging
import mimetypes
import re
import shutil
import subprocess
import unicodedata
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import aiohttp

logger = logging.getLogger("app")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AGATHA_ENV_ROOT = PROJECT_ROOT / "agathaai_env"

READ_TOKEN_LIMIT = 2500
READ_LINE_LIMIT = 255
WEB_RESULT_LIMIT = 5
WEB_TEXT_TOKEN_LIMIT = 1800
WEB_DOWNLOAD_LIMIT_BYTES = 4 * 1024 * 1024
WEB_TOTAL_TIMEOUT_SECONDS = 4.0
WEB_SEARCH_TIMEOUT_SECONDS = 8
WEB_PAGE_TIMEOUT_SECONDS = 10
WEB_PARALLEL_PAGE_FETCHES = 3
DOWNLOAD_LIMIT_BYTES = 25 * 1024 * 1024

MEDIA_MIME_PREFIXES = ("audio/", "image/", "video/")
MEDIA_EXTENSIONS = {
    ".3gp", ".aac", ".aiff", ".ape", ".avi", ".bmp", ".flac", ".gif",
    ".heic", ".ico", ".jpeg", ".jpg", ".m4a", ".mkv", ".mov", ".mp3",
    ".mp4", ".mpeg", ".mpg", ".ogg", ".opus", ".png", ".tif", ".tiff",
    ".wav", ".webm", ".webp", ".wmv",
}
BINARY_EXECUTABLE_EXTENSIONS = {
    ".apk", ".app", ".bin", ".class", ".com", ".dll", ".dmg", ".dylib",
    ".elf", ".exe", ".jar", ".msi", ".o", ".obj", ".scr", ".so",
}
web_search_tool = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web or read a direct URL. Use this when current or external "
            "information is needed. With a text query, the tool finds 5 pages and "
            "scrapes one useful page."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text, or a direct http(s) URL.",
                },
                "result_index": {
                    "type": "integer",
                    "description": "Optional 1-based search result index to read.",
                },
                "url": {
                    "type": "string",
                    "description": "Optional direct http(s) URL. If set, it is read directly.",
                },
            },
            "required": ["query"],
        },
    },
}

read_file_tool = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "Read a text/JSON/PDF file located inside AgathaAI_Portfolio/agathaai_env/, or a "
            "non-media Discord attachment by name. Use only a filename or a path relative "
            "to agathaai_env; never invent an absolute path. Images/audio/video and "
            "executable binaries are refused. At most about 2500 tokens or 255 lines "
            "are returned."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "description": (
                        "Filename or path relative to agathaai_env, for example Test.txt "
                        "or notes/Test.txt. It may also be a Discord attachment name."
                    ),
                },
                "start_line": {
                    "type": "integer",
                    "description": "1-based first line to read. Defaults to 1.",
                },
                "lines": {
                    "type": "integer",
                    "description": "Number of lines to read, capped at 255.",
                },
                "token_limit": {
                    "type": "integer",
                    "description": "Approximate token limit, capped at 2500.",
                },
            },
            "required": ["file"],
        },
    },
}

edit_file_tool = {
    "type": "function",
    "function": {
        "name": "edit_file",
        "description": (
            "Create or modify a file only inside AgathaAI_Portfolio/agathaai_env/. Any extension "
            "is allowed. Use append=true to add to an existing file, or encoding=base64 "
            "to write binary content."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "description": (
                        "Target filename or path relative to agathaai_env, for example "
                        "notes/file.txt. Never use an absolute path."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": "File content, or base64 data when encoding is base64.",
                },
                "append": {
                    "type": "boolean",
                    "description": "Append to the file instead of replacing it.",
                },
                "encoding": {
                    "type": "string",
                    "enum": ["utf-8", "base64"],
                    "description": "Use utf-8 for text or base64 for binary bytes.",
                },
            },
            "required": ["file", "content"],
        },
    },
}

twitch_create_poll_tool = {
    "type": "function",
    "function": {
        "name": "create_twitch_poll",
        "description": (
            "Create a Twitch poll in the current channel. Use only when the user asks "
            "to create or launch a poll."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Poll title or question.",
                },
                "choices": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Between 2 and 5 poll choices.",
                    "minItems": 2,
                    "maxItems": 5,
                },
                "duration": {
                    "type": "integer",
                    "description": "Poll duration in seconds. Defaults to 60.",
                },
                "channel_points_voting_enabled": {
                    "type": "boolean",
                    "description": "Whether Twitch channel-point voting is enabled.",
                },
                "channel_points_per_vote": {
                    "type": "integer",
                    "description": "Optional channel points cost per extra vote.",
                },
            },
            "required": ["title", "choices"],
        },
    },
}

twitch_ban_user_tool = {
    "type": "function",
    "function": {
        "name": "ban_twitch_user",
        "description": (
            "Permanently ban a Twitch user from the current channel. Use only when the "
            "user explicitly asks for a permanent ban."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "username": {
                    "type": "string",
                    "description": "Twitch login or user id to ban.",
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for the ban.",
                },
            },
            "required": ["username"],
        },
    },
}

twitch_timeout_user_tool = {
    "type": "function",
    "function": {
        "name": "timeout_twitch_user",
        "description": (
            "Temporarily timeout a Twitch user in the current channel. Use only when "
            "the user explicitly asks for a timeout."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "username": {
                    "type": "string",
                    "description": "Twitch login or user id to timeout.",
                },
                "duration": {
                    "type": "integer",
                    "description": "Timeout duration in seconds. Defaults to 60.",
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for the timeout.",
                },
            },
            "required": ["username"],
        },
    },
}


@dataclass
class ExtractedText:
    text: str
    source_type: str


@dataclass
class LimitedText:
    text: str
    start_line: int
    end_line: int
    total_lines: int
    token_count: int
    truncated_by_tokens: bool
    truncated_by_lines: bool


def parse_tool_json(output):
    if isinstance(output, dict):
        return output

    if not isinstance(output, str):
        return None

    text = output.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    decoder = json.JSONDecoder()

    start = text.find("{")
    if start == -1:
        return None

    try:
        obj, _ = decoder.raw_decode(text[start:])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


async def _detect_tool_call(data, context=None):
    logger.info(f"\n\n\n[DEBUG] DATA = {data}\n\n\n")
    data = parse_tool_json(data)
    if not isinstance(data, dict):
        return None

    tool = data.get("tool")
    tool = None if tool in [None, "", "null", "None"] else str(tool).lower().strip()
    query = data.get("query")

    if tool and "read_file" in tool:
        return await _run_and_log_tool("read_file", query, _read_file(query, context=context))
    if tool and ("edit_file" in tool or "write_file" in tool):
        return await _run_and_log_tool("write_file", query, _write_file(query))
    if tool and ("web_search" in tool or "websearch" in tool):
        return await _run_and_log_tool("web_search", query, _web_search(query))
    if tool is None:
        return None

    result = (
        "Format outil invalide. Utilise seulement read_file, edit_file/write_file "
        "ou web_search, ou mets tool/query a null."
    )
    logger.info(
        "\n[TOOL INVALID]\nTool: %s\nQuery: %s\nResult:\n%s\n[/TOOL INVALID]\n",
        tool,
        _format_log_value(query),
        result,
    )
    return result


async def _run_and_log_tool(tool_name, query, coroutine):
    logger.info(
        "\n[TOOL CALL]\nTool: %s\nQuery: %s\n[/TOOL CALL]\n",
        tool_name,
        _format_log_value(query),
    )
    result = await coroutine
    logger.info(
        "\n[TOOL RESULT]\nTool: %s\nResult:\n%s\n[/TOOL RESULT]\n",
        tool_name,
        result,
    )
    return result


def _format_log_value(value):
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False, indent=2)
        except Exception:
            pass

    return str(value)


def _format_json_tool_result(tool_name: str, result: Any) -> str:
    try:
        content = json.dumps(result, ensure_ascii=False, default=str, indent=2)
    except Exception:
        content = str(result)
    return f"[tool:{tool_name}]\n{content}"


async def execute_native_tool_call(tool_call, context=None):
    name = _native_tool_call_name(tool_call)
    arguments = _native_tool_call_arguments(tool_call)

    if name == "read_file":
        return await _run_and_log_tool(name, arguments, _read_file(arguments, context=context))
    if name in {"edit_file", "write_file"}:
        return await _run_and_log_tool(name, arguments, _write_file(arguments))
    if name in {"web_search", "websearch"}:
        return await _run_and_log_tool(name, arguments, _web_search(arguments))
    if name == "create_twitch_poll":
        return await _run_and_log_tool(name, arguments, _create_twitch_poll(arguments))
    if name == "ban_twitch_user":
        return await _run_and_log_tool(name, arguments, _ban_twitch_user(arguments))
    if name == "timeout_twitch_user":
        return await _run_and_log_tool(name, arguments, _timeout_twitch_user(arguments))

    result = f"Outil inconnu: {name}"
    logger.info(
        "\n[TOOL INVALID]\nTool: %s\nQuery: %s\nResult:\n%s\n[/TOOL INVALID]\n",
        name,
        _format_log_value(arguments),
        result,
    )
    return result


def _native_tool_call_name(tool_call) -> str:
    function = _tool_call_value(tool_call, "function") or {}
    return str(_tool_call_value(function, "name") or "").strip()


def _native_tool_call_arguments(tool_call) -> dict[str, Any]:
    function = _tool_call_value(tool_call, "function") or {}
    raw_args = _tool_call_value(function, "arguments")

    if isinstance(raw_args, dict):
        return dict(raw_args)
    if raw_args in (None, ""):
        return {}

    try:
        parsed = json.loads(str(raw_args))
        return parsed if isinstance(parsed, dict) else {"query": parsed}
    except json.JSONDecodeError:
        return _query_to_dict(str(raw_args), default_key="query")


def _tool_call_value(obj, key):
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


async def _read_file(query, context=None):
    logger.debug("READ FILE DETECTED")
    request = _query_to_dict(query, default_key="file")
    start_line = _positive_int(
        request.get("start_line")
        or request.get("line")
        or request.get("from_line")
        or request.get("offset")
        or 1,
        default=1,
    )
    requested_lines = _positive_int(
        request.get("lines")
        or request.get("line_count")
        or request.get("limit_lines")
        or READ_LINE_LIMIT,
        default=READ_LINE_LIMIT,
    )
    line_limit = min(requested_lines, READ_LINE_LIMIT)
    token_limit = min(
        _positive_int(request.get("token_limit") or request.get("tokens") or READ_TOKEN_LIMIT, READ_TOKEN_LIMIT),
        READ_TOKEN_LIMIT,
    )

    try:
        source = await _resolve_read_source(request, context or {})
        if source["kind"] == "attachment":
            extracted = await _read_attachment(source["attachment"])
            display_name = _attachment_name(source["attachment"])
        else:
            path = source["path"]
            extracted = await asyncio.to_thread(_extract_local_file_text, path)
            display_name = str(path.relative_to(PROJECT_ROOT)) if _is_relative_to(path, PROJECT_ROOT) else str(path)

        limited = _limit_text(extracted.text, start_line=start_line, line_limit=line_limit, token_limit=token_limit)
        return _format_read_file_result(
            display_name,
            extracted.source_type,
            limited,
        )
    except Exception as exc:
        logger.exception("\nread_file failed: %s", exc)
        return f"[tool:read_file]\nErreur: {exc}"


async def _write_file(query):
    logger.debug("WRITE FILE DETECTED")
    request = _query_to_dict(query, default_key="content")

    try:
        raw_path = (
            request.get("file")
            or request.get("path")
            or request.get("filename")
            or request.get("name")
        )
        if not raw_path:
            return (
                "[tool:write_file]\nErreur: aucun fichier fourni. "
                "Utilise {\"file\":\"notes/fichier.txt\",\"content\":\"...\"}."
            )

        target = _safe_env_path(str(raw_path))
        mode = str(request.get("mode") or "").lower().strip()
        append = bool(request.get("append")) or mode == "append"
        encoding = str(request.get("encoding") or "utf-8").lower().strip()
        content = request.get("content")
        if content is None:
            content = request.get("text")
        if content is None:
            content = ""

        target.parent.mkdir(parents=True, exist_ok=True)
        if encoding == "base64":
            import base64

            data = base64.b64decode(str(content), validate=True)
            write_mode = "ab" if append else "wb"
            with target.open(write_mode) as f:
                f.write(data)
            byte_count = len(data)
        else:
            text = str(content)
            write_mode = "a" if append else "w"
            with target.open(write_mode, encoding="utf-8", newline="") as f:
                f.write(text)
            byte_count = len(text.encode("utf-8"))

        rel_path = target.relative_to(AGATHA_ENV_ROOT)
        return (
            "[tool:write_file]\n"
            f"OK: fichier {'mis a jour' if append else 'ecrit'} dans agathaai_env/{rel_path}\n"
            f"Taille ecrite: {byte_count} octets"
        )
    except Exception as exc:
        logger.exception("\nwrite_file failed: %s", exc)
        return f"[tool:write_file]\nErreur: {exc}"


async def _web_search(query):
    logger.debug("WEB SEARCH DETECTED")
    request = _query_to_dict(query, default_key="query")
    raw_query = str(request.get("url") or request.get("query") or request.get("q") or "").strip()
    if not raw_query:
        return "[tool:web_search]\nErreur: query vide."

    deadline = asyncio.get_running_loop().time() + WEB_TOTAL_TIMEOUT_SECONDS
    try:
        return await _web_search_with_budget(request, raw_query, deadline)
    except asyncio.TimeoutError:
        logger.warning("\nweb_search reached %.1fs total budget for: %s", WEB_TOTAL_TIMEOUT_SECONDS, raw_query)
        return (
            "[tool:web_search]\n"
            f"Budget de recherche web depasse ({WEB_TOTAL_TIMEOUT_SECONDS:.0f}s) pour: {raw_query}"
        )
    except Exception as exc:
        logger.exception("\nweb_search failed: %s", exc)
        return f"[tool:web_search]\nErreur: {exc}"


async def _web_search_with_budget(request: dict[str, Any], raw_query: str, deadline: float) -> str:
    if _looks_like_url(raw_query):
        url = _normalize_url(raw_query)
        result = await _await_with_web_budget(_fetch_and_extract_url(url), deadline)
        limited = _limit_text(
            _prioritize_web_text(result.text, raw_query),
            token_limit=WEB_TEXT_TOKEN_LIMIT,
        )
        return _format_web_result(url, [], url, result.source_type, limited)

    results = await _await_with_web_budget(_search_web(raw_query), deadline)
    if not results:
        return f"[tool:web_search]\nAucun resultat web trouve pour: {raw_query}"

    selected_url = _select_search_result_url(results, request)
    candidates = _candidate_result_urls(selected_url, results)[:WEB_PARALLEL_PAGE_FETCHES]
    page_result = await _fetch_first_useful_web_result(raw_query, results, candidates, deadline)
    if page_result["limited"] is not None:
        return _format_web_result(
            raw_query,
            results,
            page_result["url"],
            page_result["source_type"],
            page_result["limited"],
        )

    return _format_web_results_only(
        raw_query,
        results,
        page_result["errors"],
        timed_out=page_result["timed_out"],
    )


async def _await_with_web_budget(awaitable, deadline: float):
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        raise asyncio.TimeoutError()
    return await asyncio.wait_for(awaitable, timeout=remaining)


async def _fetch_first_useful_web_result(
    query: str,
    results: list[dict[str, str]],
    candidates: list[str],
    deadline: float,
) -> dict[str, Any]:
    tasks = {
        asyncio.create_task(_fetch_web_candidate(query, candidate_url)): candidate_url
        for candidate_url in candidates
    }
    errors = []
    low_content_fallback = None
    timed_out = False

    try:
        while tasks:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                timed_out = True
                break

            done, _ = await asyncio.wait(
                tasks,
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                timed_out = True
                break

            for task in done:
                candidate_url = tasks.pop(task)
                try:
                    result = task.result()
                except Exception as exc:
                    logger.warning("\nweb_search could not scrape %s: %s", candidate_url, exc)
                    errors.append(f"{candidate_url}: {exc}")
                    continue

                if _is_low_value_web_text(result["limited"].text) and len(results) > 1:
                    if low_content_fallback is None:
                        low_content_fallback = result
                    errors.append(f"{candidate_url}: contenu trop court")
                    continue

                return {**result, "errors": errors, "timed_out": False}

        if low_content_fallback is not None:
            return {**low_content_fallback, "errors": errors, "timed_out": timed_out}

        return {
            "url": "",
            "source_type": "",
            "limited": None,
            "errors": errors,
            "timed_out": timed_out,
        }
    finally:
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


async def _fetch_web_candidate(query: str, candidate_url: str) -> dict[str, Any]:
    result = await _fetch_and_extract_url(candidate_url)
    limited = _limit_text(
        _prioritize_web_text(result.text, query),
        token_limit=WEB_TEXT_TOKEN_LIMIT,
    )
    return {
        "url": candidate_url,
        "source_type": result.source_type,
        "limited": limited,
    }


async def _create_twitch_poll(query):
    request = _query_to_dict(query, default_key="title")
    title = str(request.get("title") or request.get("question") or "").strip()
    choices = _string_list(request.get("choices") or request.get("options"))
    duration = _positive_int(request.get("duration") or request.get("seconds") or 60, 60)
    channel_points_enabled = _bool_value(request.get("channel_points_voting_enabled"), default=False)
    channel_points_per_vote = request.get("channel_points_per_vote")
    channel_points_per_vote = (
        _positive_int(channel_points_per_vote, 0)
        if channel_points_per_vote not in (None, "")
        else None
    )

    if not title:
        return "[tool:create_twitch_poll]\nErreur: titre de sondage manquant."
    if len(choices) < 2 or len(choices) > 5:
        return "[tool:create_twitch_poll]\nErreur: un sondage Twitch demande entre 2 et 5 choix."

    from src.streaming.twitch.tw_plugin import _create_poll

    result = await _create_poll(
        title,
        choices,
        duration=duration,
        channel_points_voting_enabled=channel_points_enabled,
        channel_points_per_vote=channel_points_per_vote,
    )
    return _format_json_tool_result("create_twitch_poll", result)


async def _ban_twitch_user(query):
    request = _query_to_dict(query, default_key="username")
    username = str(request.get("username") or request.get("user") or request.get("login") or "").strip()
    reason = str(request.get("reason") or "Banned by AgathaAI.").strip()

    if not username:
        return "[tool:ban_twitch_user]\nErreur: utilisateur Twitch manquant."

    from src.streaming.twitch.tw_plugin import _ban_user

    result = await _ban_user(username, reason=reason)
    return _format_json_tool_result("ban_twitch_user", result)


async def _timeout_twitch_user(query):
    request = _query_to_dict(query, default_key="username")
    username = str(request.get("username") or request.get("user") or request.get("login") or "").strip()
    reason = str(request.get("reason") or "Timed out by AgathaAI.").strip()
    duration = _positive_int(request.get("duration") or request.get("seconds") or 60, 60)

    if not username:
        return "[tool:timeout_twitch_user]\nErreur: utilisateur Twitch manquant."

    from src.streaming.twitch.tw_plugin import _ban_user

    result = await _ban_user(username, reason=reason, duration=duration)
    return _format_json_tool_result("timeout_twitch_user", result)


async def _resolve_read_source(request: dict[str, Any], context: dict[str, Any]):
    raw_path = (
        request.get("file")
        or request.get("path")
        or request.get("filename")
        or request.get("name")
        or request.get("attachment")
        or request.get("query")
        or ""
    )
    raw_path = str(raw_path).strip()
    attachments = context.get("attachments") or []

    attachment = _find_attachment(raw_path, attachments)
    if attachment:
        return {"kind": "attachment", "attachment": attachment}

    if _looks_like_explicit_http_url(raw_path):
        return {
            "kind": "attachment",
            "attachment": {
                "name": Path(urlparse(raw_path).path).name or "attachment",
                "url": raw_path,
                "contentType": mimetypes.guess_type(raw_path)[0],
                "size": None,
            },
        }

    path = _resolve_local_read_path(raw_path)
    return {"kind": "local", "path": path}


def _resolve_local_read_path(raw_path: str) -> Path:
    if not raw_path:
        raise ValueError("aucun fichier fourni")

    relative = _normalize_env_file_reference(raw_path)
    direct = (AGATHA_ENV_ROOT / relative).resolve()
    if not _is_relative_to(direct, AGATHA_ENV_ROOT):
        raise ValueError("lecture autorisee seulement dans AgathaAI_Portfolio/agathaai_env")
    if direct.is_file():
        return direct

    matches = _find_env_files(relative)
    if not matches:
        raise FileNotFoundError(f"fichier introuvable dans agathaai_env: {relative.as_posix()}")
    if len(matches) > 1:
        preview = "\n".join(
            f"- {path.relative_to(AGATHA_ENV_ROOT)}"
            for path in matches[:10]
        )
        raise ValueError(f"plusieurs fichiers correspondent dans agathaai_env:\n{preview}")

    return matches[0]


def _find_env_files(relative: Path) -> list[Path]:
    filename = relative.name.lower()
    if not filename:
        return []

    matches = []
    for path in AGATHA_ENV_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.name.lower() == filename:
            matches.append(path.resolve())
        if len(matches) >= 25:
            break

    return matches


async def _read_attachment(attachment: dict[str, Any]) -> ExtractedText:
    url = attachment.get("url")
    if not url:
        raise ValueError("piece jointe sans URL")

    name = _attachment_name(attachment)
    content_type = attachment.get("contentType") or attachment.get("content_type")
    _validate_non_media_non_executable(name, content_type)

    data, response_content_type = await _download_bytes(url)
    content_type = content_type or response_content_type
    _validate_non_media_non_executable(name, content_type, data)

    return _extract_bytes_text(data, name, content_type)


def _extract_local_file_text(path: Path) -> ExtractedText:
    mime_type = mimetypes.guess_type(path.name)[0]
    _validate_non_media_non_executable(path.name, mime_type)
    data = path.read_bytes()
    _validate_non_media_non_executable(path.name, mime_type, data)
    return _extract_bytes_text(data, path.name, mime_type)


def _extract_bytes_text(data: bytes, name: str, content_type: str | None = None) -> ExtractedText:
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf" or content_type == "application/pdf":
        return ExtractedText(_extract_pdf_text(data), "pdf")
    if suffix == ".json" or content_type in {"application/json", "text/json"}:
        text = _decode_text(data)
        try:
            parsed = json.loads(text)
            text = json.dumps(parsed, indent=2, ensure_ascii=False)
        except Exception:
            pass
        return ExtractedText(text, "json")

    return ExtractedText(_decode_text(data), "text")


def _extract_pdf_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        pages = []
        for index, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(f"[page {index}]\n{page_text.strip()}")
        return "\n\n".join(pages).strip()
    except Exception as exc:
        logger.warning("\npypdf extraction failed: %s", exc)

    pdftotext = shutil.which("pdftotext")
    if pdftotext:
        try:
            process = subprocess.run(
                [pdftotext, "-layout", "-", "-"],
                input=data,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=20,
            )
            return process.stdout.decode("utf-8", errors="replace").strip()
        except Exception as exc:
            logger.warning("\npdftotext extraction failed: %s", exc)

    raise ValueError("impossible de lire ce PDF: aucun extracteur PDF disponible")


def _decode_text(data: bytes) -> str:
    if not data:
        return ""

    try:
        from charset_normalizer import from_bytes

        best = from_bytes(data).best()
        if best is not None:
            return str(best)
    except Exception:
        pass

    for encoding in ("utf-8-sig", "utf-8", "utf-16", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue

    return data.decode("utf-8", errors="replace")


def _validate_non_media_non_executable(
    name: str,
    content_type: str | None,
    data: bytes | None = None,
) -> None:
    suffix = Path(str(name or "")).suffix.lower()
    mime = str(content_type or "").lower()

    if suffix in MEDIA_EXTENSIONS or mime.startswith(MEDIA_MIME_PREFIXES):
        raise ValueError("piece jointe media ignoree")

    if suffix in BINARY_EXECUTABLE_EXTENSIONS:
        raise ValueError("fichier executable binaire refuse")

    if data is None:
        return

    if _is_binary_bytes(data):
        if suffix == ".pdf" or mime == "application/pdf":
            return
        raise ValueError("fichier binaire non lisible refuse")


def _is_binary_bytes(data: bytes) -> bool:
    sample = data[:4096]
    if b"\x00" in sample:
        return True
    if not sample:
        return False

    control = sum(byte < 9 or (13 < byte < 32) for byte in sample)
    return control / len(sample) > 0.12


async def _download_bytes(
    url: str,
    *,
    timeout_seconds: int = 30,
    byte_limit: int = DOWNLOAD_LIMIT_BYTES,
    reject_media: bool = False,
) -> tuple[bytes, str | None]:
    timeout = aiohttp.ClientTimeout(
        total=timeout_seconds,
        connect=min(4, timeout_seconds),
        sock_connect=min(4, timeout_seconds),
        sock_read=timeout_seconds,
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
        )
    }
    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        async with session.get(url, allow_redirects=True) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").split(";")[0].strip()
            if reject_media and content_type.lower().startswith(MEDIA_MIME_PREFIXES):
                raise ValueError(f"contenu media ignore: {content_type}")
            content_length = response.content_length
            if content_length and content_length > byte_limit:
                raise ValueError("fichier trop volumineux")
            chunks = []
            total_bytes = 0
            async for chunk in response.content.iter_chunked(64 * 1024):
                total_bytes += len(chunk)
                if total_bytes > byte_limit:
                    raise ValueError("fichier trop volumineux")
                chunks.append(chunk)
            data = b"".join(chunks)
            return data, content_type or None


async def _search_web(query: str) -> list[dict[str, str]]:
    search_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
    html, _ = await _fetch_text_url(search_url, timeout_seconds=WEB_SEARCH_TIMEOUT_SECONDS)
    return _parse_duckduckgo_results(html)[:WEB_RESULT_LIMIT]


async def _fetch_and_extract_url(url: str) -> ExtractedText:
    data, content_type = await _download_bytes(
        url,
        timeout_seconds=WEB_PAGE_TIMEOUT_SECONDS,
        byte_limit=WEB_DOWNLOAD_LIMIT_BYTES,
        reject_media=True,
    )
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix == ".pdf" or content_type == "application/pdf":
        return _extract_bytes_text(data, Path(urlparse(url).path).name or "page.pdf", content_type)

    text = _decode_text(data)
    if content_type and "html" not in content_type and suffix not in {".htm", ".html", ""}:
        return ExtractedText(text, "text")

    title, extracted = _html_to_text(text)
    if title:
        extracted = f"{title}\n\n{extracted}".strip()
    return ExtractedText(extracted, "html")


async def _fetch_text_url(url: str, *, timeout_seconds: int = 30) -> tuple[str, str | None]:
    data, content_type = await _download_bytes(url, timeout_seconds=timeout_seconds)
    return _decode_text(data), content_type


def _parse_duckduckgo_results(html: str) -> list[dict[str, str]]:
    results = []
    pattern = re.compile(
        r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(html):
        url = _clean_search_result_url(match.group(1))
        title = _strip_html(match.group(2))
        if (
            _is_usable_search_result_url(url)
            and title
            and not any(item["url"] == url for item in results)
        ):
            results.append({"title": title, "url": url})
        if len(results) >= WEB_RESULT_LIMIT:
            break

    if results:
        return results

    fallback = re.compile(r'href="(https?://[^"]+)"[^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
    for match in fallback.finditer(html):
        url = _clean_search_result_url(match.group(1))
        title = _strip_html(match.group(2))
        if (
            _is_usable_search_result_url(url)
            and title
            and not any(item["url"] == url for item in results)
        ):
            results.append({"title": title, "url": url})
        if len(results) >= WEB_RESULT_LIMIT:
            break
    return results


def _clean_search_result_url(raw_url: str) -> str:
    url = unescape(str(raw_url or "")).strip()
    if url.startswith("//"):
        url = "https:" + url
    elif url.startswith("/"):
        url = "https://duckduckgo.com" + url

    parsed = urlparse(url)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        qs = parse_qs(parsed.query)
        url = unquote(qs.get("uddg", [url])[0])

    return url.strip()


def _is_usable_search_result_url(url: str) -> bool:
    if not _looks_like_url(url):
        return False

    parsed = urlparse(_normalize_url(url))
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    query = parsed.query.lower()

    if "duckduckgo.com" in host:
        return False
    if "ad_domain=" in query or "ad_provider=" in query:
        return False
    if path.endswith("/y.js") or "/y.js" in path:
        return False

    blocked_hosts = (
        "ad.doubleclick.net",
        "googleadservices.com",
        "googlesyndication.com",
        "bing.com",
        "bat.bing.com",
    )
    if any(host == blocked or host.endswith("." + blocked) for blocked in blocked_hosts):
        return False

    blocked_path_fragments = ("/aclick", "/pagead/", "/ads/")
    if any(fragment in path for fragment in blocked_path_fragments):
        return False

    return True


def _select_search_result_url(results: list[dict[str, str]], request: dict[str, Any]) -> str:
    if request.get("result_url") and _looks_like_url(str(request["result_url"])):
        return _normalize_url(str(request["result_url"]))

    index = request.get("result_index")
    if index is None:
        index = request.get("index")
    if index is not None:
        selected_index = max(1, min(_positive_int(index, 1), len(results))) - 1
        return results[selected_index]["url"]

    return results[0]["url"]


def _candidate_result_urls(selected_url: str, results: list[dict[str, str]]) -> list[str]:
    urls = [selected_url]
    urls.extend(item["url"] for item in results)

    deduped = []
    for url in urls:
        if url not in deduped:
            deduped.append(url)
    return deduped


def _is_low_value_web_text(text: str) -> bool:
    cleaned = _normalize_space(text)
    return len(cleaned) < 240 or _count_tokens(cleaned) < 45


def _html_to_text(html: str) -> tuple[str, str]:
    title = _extract_html_title(html)
    html = _select_readable_html_fragment(html)
    html = re.sub(
        r"(?is)<(script|style|noscript|svg|canvas|iframe|nav|footer|aside|form|button|select)[^>]*>.*?</\1>",
        " ",
        html,
    )
    parser = _ReadableHTMLParser()
    parser.feed(html)
    parser.close()

    lines = []
    seen = set()
    for line in parser.lines:
        line = _normalize_space(line)
        if len(line) < 2:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(line)

    return (parser.title.strip() or title), "\n".join(lines).strip()


def _extract_html_title(html: str) -> str:
    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", html or "")
    return _strip_html(match.group(1)) if match else ""


def _select_readable_html_fragment(html: str) -> str:
    text = html or ""

    for pattern in (
        r"(?is)<main\b[^>]*>.*?</main>",
        r"(?is)<article\b[^>]*>.*?</article>",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(0)

    content_pattern = re.compile(
        r"(?is)<[^>]+(?:id|class)=[\"'][^\"']*(?:mw-content-text|article-body|entry-content|post-content|page-content|main-content|content-body)[^\"']*[\"'][^>]*>"
    )
    content_match = None
    for match in content_pattern.finditer(text):
        tag = match.group(0).lower()
        if any(hint in tag for hint in ("toc-", "vector-toc", "sidebar", "navbox")):
            continue
        content_match = match
        break

    if content_match:
        end_match = re.search(
            r"(?is)<[^>]+(?:id|class)=[\"'][^\"']*(?:catlinks|comments|related|footer|site-footer)[^\"']*[\"']",
            text[content_match.end():],
        )
        end = content_match.end() + end_match.start() if end_match else len(text)
        return text[content_match.start():end]

    body_match = re.search(r"(?is)<body\b[^>]*>.*?</body>", text)
    if body_match:
        return body_match.group(0)

    return text


class _ReadableHTMLParser(HTMLParser):
    BLOCK_TAGS = {
        "article", "blockquote", "br", "div", "h1", "h2", "h3", "h4",
        "li", "main", "p", "pre", "section", "td", "th", "title",
        "tr",
    }
    SKIP_TAGS = {
        "aside", "button", "canvas", "dialog", "fieldset", "footer", "form",
        "iframe", "input", "label", "nav", "noscript", "option", "script",
        "select", "style", "svg",
    }
    SKIP_ATTR_HINTS = (
        "ad-", "ads", "advert", "banner", "breadcrumb", "cookie", "dialog",
        "footer", "header", "menu", "modal", "nav", "newsletter", "popup",
        "promo", "related", "search", "share", "sidebar", "social",
        "sponsor", "subscribe", "toolbar", "toc",
    )
    VOID_TAGS = {
        "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
        "meta", "param", "source", "track", "wbr",
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._skip_stack = []
        self._current = []
        self.lines = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if self._skip_depth:
            if (
                tag not in self.VOID_TAGS
                and (tag in self.SKIP_TAGS or (self._skip_stack and tag == self._skip_stack[-1]))
            ):
                self._skip_depth += 1
                self._skip_stack.append(tag)
            return
        if self._should_skip(tag, attrs):
            self._flush()
            if tag not in self.VOID_TAGS:
                self._skip_depth += 1
                self._skip_stack.append(tag)
            return
        if tag == "title":
            self._in_title = True
        if tag in self.BLOCK_TAGS:
            self._flush()

    def handle_endtag(self, tag):
        tag = tag.lower()
        if self._skip_stack and tag == self._skip_stack[-1]:
            self._skip_stack.pop()
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = False
        if tag in self.BLOCK_TAGS:
            self._flush()

    def handle_data(self, data):
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data
            return
        if data.strip():
            self._current.append(data)

    def close(self):
        self._flush()
        super().close()

    def _flush(self):
        text = _normalize_space(" ".join(self._current))
        self._current = []
        if text:
            self.lines.append(text)

    def _should_skip(self, tag: str, attrs: list[tuple[str, str | None]]) -> bool:
        if tag in self.SKIP_TAGS:
            return True

        attr_text = " ".join(
            str(value or "").lower()
            for name, value in attrs
            if name.lower() in {"id", "class", "role", "aria-label"}
        )
        return any(hint in attr_text for hint in self.SKIP_ATTR_HINTS)


def _prioritize_web_text(text: str, query: str) -> str:
    terms = _query_terms(query)
    if not terms:
        return text

    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return text

    normalized_lines = []
    scored = []
    for index, line in enumerate(lines):
        normalized = _normalize_for_search(line)
        normalized_lines.append(normalized)
        score = sum(1 for term in terms if term in normalized)
        if score:
            scored.append((score, index))

    if not scored:
        return text

    selected_indexes = set()
    for term in terms:
        matches_for_term = 0
        for index, normalized in enumerate(normalized_lines):
            if term not in normalized:
                continue
            for nearby in range(max(0, index - 1), min(len(lines), index + 2)):
                selected_indexes.add(nearby)
            matches_for_term += 1
            if matches_for_term >= 3:
                break

    for _, index in sorted(scored, key=lambda item: (-item[0], item[1]))[:12]:
        for nearby in range(max(0, index - 1), min(len(lines), index + 2)):
            selected_indexes.add(nearby)

    focused = [lines[index] for index in sorted(selected_indexes)]
    if not focused:
        return text

    return (
        "Extraits pertinents:\n"
        + "\n".join(focused)
        + "\n\nTexte de la page:\n"
        + str(text or "")
    )


def _query_terms(query: str) -> list[str]:
    if _looks_like_url(query):
        return []

    normalized = _normalize_for_search(query)
    stopwords = {
        "avec", "dans", "des", "est", "les", "pour", "que", "qui", "sur",
        "the", "and", "for", "from", "what", "who", "with",
    }
    expansions = {
        "auteur": ["author", "writer", "written", "created", "creator", "screenwriter", "scenariste"],
        "author": ["auteur", "writer", "written", "created", "creator", "screenwriter"],
        "ecrit": ["written", "writer", "author", "auteur", "scenariste"],
        "written": ["writer", "author", "auteur", "scenariste"],
        "writer": ["written", "author", "auteur", "scenariste"],
        "createur": ["creator", "created", "created by", "author", "auteur"],
        "creator": ["createur", "created", "created by", "author"],
        "scenariste": ["screenwriter", "written", "writer", "author", "auteur"],
        "screenwriter": ["scenariste", "written", "writer", "author"],
        "realisateur": ["director", "directed", "directed by"],
        "director": ["realisateur", "directed", "directed by"],
    }
    terms = []
    for term in re.findall(r"[a-z0-9]{3,}", normalized):
        if term in stopwords:
            continue
        if term not in terms:
            terms.append(term)
        for expanded in expansions.get(term, []):
            if expanded not in terms:
                terms.append(expanded)
    return terms[:16]


def _normalize_for_search(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or "").lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _limit_text(
    text: str,
    *,
    start_line: int = 1,
    line_limit: int = READ_LINE_LIMIT,
    token_limit: int = READ_TOKEN_LIMIT,
) -> LimitedText:
    lines = str(text or "").splitlines()
    total_lines = len(lines)
    start_line = max(1, start_line)
    line_limit = max(1, min(line_limit, READ_LINE_LIMIT))

    start_index = start_line - 1
    end_index = min(start_index + line_limit, total_lines)
    selected_lines = lines[start_index:end_index]
    selected_text = "\n".join(selected_lines).strip()
    truncated_by_lines = end_index < total_lines

    token_count = _count_tokens(selected_text)
    truncated_by_tokens = False
    if token_count > token_limit:
        selected_text = _truncate_to_tokens(selected_text, token_limit)
        token_count = _count_tokens(selected_text)
        truncated_by_tokens = True

    actual_line_count = len(selected_text.splitlines()) if selected_text else 0
    end_line = start_line + max(actual_line_count - 1, 0)
    if actual_line_count == 0:
        end_line = start_line - 1

    return LimitedText(
        text=selected_text,
        start_line=start_line,
        end_line=end_line,
        total_lines=total_lines,
        token_count=token_count,
        truncated_by_tokens=truncated_by_tokens,
        truncated_by_lines=truncated_by_lines,
    )


def _count_tokens(text: str) -> int:
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text or ""))
    except Exception:
        return max(1, len(text or "") // 4)


def _truncate_to_tokens(text: str, token_limit: int) -> str:
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text or "")
        return encoding.decode(tokens[:token_limit]).rstrip()
    except Exception:
        return (text or "")[: token_limit * 4].rstrip()


def _format_read_file_result(name: str, source_type: str, limited: LimitedText) -> str:
    more_hint = ""
    if limited.truncated_by_lines or limited.truncated_by_tokens:
        more_hint = f"\nSuite disponible avec start_line={limited.end_line + 1}."

    return (
        "[tool:read_file]\n"
        f"Source: {name}\n"
        f"Type: {source_type}\n"
        f"Lignes: {limited.start_line}-{limited.end_line} / {limited.total_lines}\n"
        f"Tokens approx: {limited.token_count} / {READ_TOKEN_LIMIT}\n"
        f"Tronque par lignes: {limited.truncated_by_lines}\n"
        f"Tronque par tokens: {limited.truncated_by_tokens}"
        f"{more_hint}\n\n"
        f"{limited.text}"
    )


def _format_web_result(
    query: str,
    results: list[dict[str, str]],
    selected_url: str,
    source_type: str,
    limited: LimitedText,
) -> str:
    result_lines = "\n".join(
        f"{idx}. {item['title']} - {item['url']}"
        for idx, item in enumerate(results, start=1)
    )
    if not result_lines:
        result_lines = "Recherche directe par URL."

    return (
        "[tool:web_search]\n"
        f"Query/URL: {query}\n"
        f"Resultats ({len(results)}):\n{result_lines}\n"
        f"Page selectionnee: {selected_url}\n"
        f"Type: {source_type}\n"
        f"Tokens approx: {limited.token_count} / {WEB_TEXT_TOKEN_LIMIT}\n"
        f"Tronque: {limited.truncated_by_tokens or limited.truncated_by_lines}\n\n"
        f"{limited.text}"
    )


def _format_web_results_only(
    query: str,
    results: list[dict[str, str]],
    errors: list[str],
    *,
    timed_out: bool,
) -> str:
    result_lines = "\n".join(
        f"{idx}. {item['title']} - {item['url']}"
        for idx, item in enumerate(results, start=1)
    )
    detail = (
        f"Budget de {WEB_TOTAL_TIMEOUT_SECONDS:.0f}s atteint avant de lire une page exploitable."
        if timed_out
        else "Aucune page exploitable trouvee dans le budget web_search."
    )
    error_lines = "\n".join(f"- {error}" for error in errors[:WEB_RESULT_LIMIT])

    return (
        "[tool:web_search]\n"
        f"Query: {query}\n"
        f"Resultats ({len(results)}):\n{result_lines}\n"
        f"{detail}\n"
        "Utilise les titres et URLs ci-dessus, et precise si la page complete n'a pas pu etre lue."
        + (f"\nErreurs:\n{error_lines}" if error_lines else "")
    )


def _query_to_dict(query, *, default_key: str) -> dict[str, Any]:
    if isinstance(query, dict):
        request = dict(query)
        unwrapped = _unwrap_nested_query_request(request, default_key=default_key)
        if unwrapped is not None:
            return unwrapped
        return request

    if query is None:
        return {}

    text = str(query).strip()
    parsed = parse_tool_json(text)
    if isinstance(parsed, dict):
        return _query_to_dict(parsed, default_key=default_key)

    loose = _parse_loose_tool_object(text)
    if loose:
        return loose

    if _looks_like_url(text):
        return {default_key: text}

    fields = _parse_key_value_block(text)
    recognized_keys = {
        "append", "attachment", "content", "encoding", "file", "filename",
        "from_line", "index", "line", "line_count", "limit_lines", "lines",
        "mode", "name", "path", "query", "q", "result_index",
        "result_url", "start_line", "text", "token_limit", "tokens", "url",
    }
    if fields and set(fields).intersection(recognized_keys):
        return fields

    return {default_key: text}


def _unwrap_nested_query_request(request: dict[str, Any], *, default_key: str) -> dict[str, Any] | None:
    if default_key == "query":
        return None
    if not isinstance(request.get("query"), str):
        return None

    structural_keys = {
        "append", "attachment", "content", "encoding", "file", "filename",
        "from_line", "line", "line_count", "limit_lines", "lines", "mode",
        "name", "path", "start_line", "text", "token_limit", "tokens", "url",
    }
    if set(request).intersection(structural_keys):
        return None

    nested = _query_to_dict(request["query"], default_key=default_key)
    if not nested:
        return None

    passthrough_keys = {"append", "encoding", "lines", "start_line", "token_limit"}
    for key in passthrough_keys:
        if key in request and key not in nested:
            nested[key] = request[key]
    return nested


def _parse_loose_tool_object(text: str) -> dict[str, Any]:
    stripped = str(text or "").strip()
    if not stripped.startswith("{"):
        return {}

    fields: dict[str, Any] = {}
    for key in ("file", "path", "filename", "name", "encoding", "mode"):
        value = _extract_loose_string_field(stripped, key)
        if value is not None:
            fields[key] = value

    content = _extract_loose_content_field(stripped)
    if content is not None:
        fields["content"] = content

    for key in ("append",):
        value = _extract_loose_bool_field(stripped, key)
        if value is not None:
            fields[key] = value

    for key in ("lines", "start_line", "token_limit", "duration", "result_index"):
        value = _extract_loose_int_field(stripped, key)
        if value is not None:
            fields[key] = value

    return fields


def _extract_loose_string_field(text: str, key: str) -> str | None:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"', text, flags=re.DOTALL)
    if not match:
        return None
    return _decode_jsonish_string(match.group(1)).strip()


def _extract_loose_content_field(text: str) -> str | None:
    match = re.search(r'"content"\s*:\s*', text)
    if not match:
        return None

    value = text[match.end():].strip()
    if value.endswith("}"):
        value = value[:-1].rstrip()
    if value.endswith(","):
        value = value[:-1].rstrip()

    try:
        decoded, end = json.JSONDecoder().raw_decode(value)
        if isinstance(decoded, str) and not value[end:].strip().strip(",}"):
            return decoded
    except json.JSONDecodeError:
        pass

    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    elif value.startswith('"'):
        value = value[1:]

    return _decode_jsonish_string(value).strip()


def _extract_loose_bool_field(text: str, key: str) -> bool | None:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*(true|false)', text, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower() == "true"


def _extract_loose_int_field(text: str, key: str) -> int | None:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*(-?\d+)', text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _decode_jsonish_string(value: str) -> str:
    try:
        return json.loads(f'"{value}"')
    except json.JSONDecodeError:
        return (
            value
            .replace("\\n", "\n")
            .replace("\\r", "\r")
            .replace("\\t", "\t")
            .replace('\\"', '"')
            .replace("\\\\", "\\")
        )


def _parse_key_value_block(text: str) -> dict[str, str]:
    fields = {}
    content_lines = []
    in_content = False
    for line in text.splitlines():
        if in_content:
            content_lines.append(line)
            continue

        match = re.match(r"^\s*([A-Za-z_][\w-]*)\s*:\s*(.*)$", line)
        if not match:
            continue

        key = match.group(1).lower().replace("-", "_")
        value = match.group(2)
        if key in {"content", "text"}:
            in_content = True
            content_lines.append(value)
        else:
            fields[key] = value.strip()

    if content_lines:
        fields["content"] = "\n".join(content_lines).strip("\n")
    return fields


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if value in (None, ""):
        return []

    text = str(value).strip()
    if not text:
        return []

    parsed = parse_tool_json(text)
    if isinstance(parsed, dict):
        for key in ("choices", "options"):
            if key in parsed:
                return _string_list(parsed[key])

    return [
        item.strip(" \t\r\n\"'")
        for item in re.split(r"\s*(?:\||,|\n)\s*", text)
        if item.strip(" \t\r\n\"'")
    ]


def _bool_value(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "oui", "on"}:
        return True
    if text in {"0", "false", "no", "n", "non", "off"}:
        return False
    return default


def _safe_env_path(raw_path: str) -> Path:
    relative = _normalize_env_file_reference(raw_path)
    resolved = (AGATHA_ENV_ROOT / relative).resolve()
    env_root = AGATHA_ENV_ROOT.resolve()

    if not _is_relative_to(resolved, env_root):
        raise ValueError("ecriture autorisee seulement dans AgathaAI_Portfolio/agathaai_env")
    if resolved.exists() and resolved.is_dir():
        raise ValueError("le chemin cible est un dossier")
    return resolved


def _normalize_env_file_reference(raw_path: str) -> Path:
    text = str(raw_path or "").strip().strip("\"'")
    if not text:
        raise ValueError("aucun fichier fourni")
    if "\x00" in text:
        raise ValueError("chemin invalide")

    normalized = text.replace("\\", "/")
    env_root_text = AGATHA_ENV_ROOT.resolve().as_posix()
    lowered = normalized.lower()

    if lowered == env_root_text.lower():
        raise ValueError("le chemin cible est un dossier")
    if lowered.startswith(env_root_text.lower() + "/"):
        normalized = normalized[len(env_root_text) + 1:]
    else:
        marker = "/agathaai_env/"
        marker_index = lowered.rfind(marker)
        if marker_index >= 0:
            normalized = normalized[marker_index + len(marker):]
        elif lowered.startswith("agathaai_env/"):
            normalized = normalized[len("agathaai_env/"):]
        elif lowered.startswith("agathaai_portfolio/agathaai_env/"):
            normalized = normalized[len("agathaai_portfolio/agathaai_env/"):]
        elif normalized.startswith("/") or re.match(r"^[A-Za-z]:/", normalized) or normalized.startswith("~/"):
            normalized = normalized.rstrip("/").rsplit("/", 1)[-1]

    while normalized.startswith("./"):
        normalized = normalized[2:]
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    if not parts:
        raise ValueError("aucun fichier fourni")
    if any(part == ".." for part in parts):
        raise ValueError("les chemins relatifs avec '..' sont refuses")

    relative = Path(*parts)
    if relative.is_absolute():
        raise ValueError("chemin de fichier invalide")
    return relative


def _find_attachment(query: str, attachments: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not attachments:
        return None

    cleaned = _normalize_attachment_query(query)
    cleaned_basename = cleaned.replace("\\", "/").rsplit("/", 1)[-1] if cleaned else ""
    if cleaned:
        for attachment in attachments:
            name = _attachment_name(attachment)
            name_lower = name.lower()
            stem_lower = Path(name).stem.lower()
            if cleaned in {name_lower, stem_lower} or cleaned_basename in {name_lower, stem_lower}:
                return attachment
            if cleaned and cleaned in name_lower:
                return attachment

    if len(attachments) == 1 and (
        not cleaned
        or cleaned in {"attachment", "piece jointe", "pièce jointe", "fichier joint", "file"}
    ):
        return attachments[0]

    return None


def _normalize_attachment_query(query: str) -> str:
    cleaned = str(query or "").strip().lower()
    cleaned = re.sub(r"^(attachment|piece jointe|pièce jointe|fichier joint|file)\s*[:#-]?\s*", "", cleaned)
    cleaned = cleaned.strip("\"' ")
    return cleaned


def _attachment_name(attachment: dict[str, Any]) -> str:
    return (
        attachment.get("name")
        or attachment.get("filename")
        or Path(urlparse(str(attachment.get("url") or "")).path).name
        or "attachment"
    )


def _looks_like_url(value: str) -> bool:
    value = str(value or "").strip()
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return True
    return bool(re.match(r"^[\w.-]+\.[A-Za-z]{2,}(/.*)?$", value))


def _looks_like_explicit_http_url(value: str) -> bool:
    parsed = urlparse(str(value or "").strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _normalize_url(value: str) -> str:
    value = str(value or "").strip()
    if not urlparse(value).scheme:
        value = "https://" + value
    return value


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", unescape(str(value or ""))).strip()


def _strip_html(value: str) -> str:
    value = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", value)
    value = re.sub(r"(?s)<[^>]+>", " ", value)
    return _normalize_space(value)


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False
