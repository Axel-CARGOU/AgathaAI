import asyncio, base64, gc, json, logging, mimetypes, os, re, signal, time, urllib.error, urllib.request, platform, regex, uuid
from datetime import datetime
from pathlib import Path
from openai import AsyncOpenAI
from langdetect import detect
from src.config.config import settings
from src.control_panel import runtime as cp_runtime
from src.control_panel.moderation import redact_blacklisted_words
from src.AI.RAG.rag import build_rag_context, remember_interaction
from src.AI.LLM.tools import (
    AGATHA_ENV_ROOT,
    edit_file_tool,
    execute_native_tool_call,
    parse_tool_json,
    read_file_tool,
    twitch_ban_user_tool,
    twitch_create_poll_tool,
    twitch_timeout_user_tool,
    web_search_tool,
)
from src.AI.TTS.tts import get_input
from src.metrics.metrics import (
    record_llm_request_metric,
    record_vllm_engine_metric_line,
)
from src.vtuber.vts_plugin import move_by_sentence

logger = logging.getLogger("app")

VLLM_HOST = "bind.example"
VLLM_CLIENT_HOST = "local.example"
VLLM_PORT = 8008
VLLM_BASE_URL = f"http://{VLLM_CLIENT_HOST}:{VLLM_PORT}/v1"
VLLM_HEALTH_URL = f"http://{VLLM_CLIENT_HOST}:{VLLM_PORT}/health"
VLLM_API_KEY = "<redacted>"

proc, vllm_log_task, vllm_client = None, None, None
model_path, user_id, warmup = None, None, None
language = "en"
stream_usage_supported = True

VIP_USERS = {
    "000000000000000000": ("Axel", " (ton créateur)"),
    "000000000000000000": ("VIPUser1", " (ta petite sœur)"),
    "000000000000000000": ("VIPUser2", " (un ami de ton créateur)"),
    "000000000000000000": ("VIPUser3", " (un ami de ton créateur)"),
}

async def load_llm():
    global proc, vllm_log_task, vllm_client, model_path, warmup

    if proc is not None and proc.returncode is None:
        logger.info("\nvLLM backend is already running")
        if vllm_client is None:
            vllm_client = _make_vllm_client()
        return proc

    model_path = _resolve_model_path(settings.LLM_MODEL_PATH)
    vllm_client = _make_vllm_client()

    logger.info(f"\nMODEL PATH = {repr(model_path)}")
    logger.info(f"\nSERVED MODEL NAME = {settings.LLM_MODEL_PATH}")
    logger.info("\nStarting vLLM backend...")

    process_kwargs = {}
    if hasattr(os, "setsid"):
        process_kwargs["preexec_fn"] = os.setsid

    vllm_command = [
        "vllm", "serve", model_path,
        "--host", VLLM_HOST,
        "--port", str(VLLM_PORT),
        "--max-model-len", str(settings.LLM_MAX_SEQ_LEN),
        "--gpu-memory-utilization", str(0.7) if ("microsoft" in platform.uname().release.lower() or platform.system().lower() == "windows") else str(settings.LLM_GPU_MAX_USE),
        "--trust-remote-code",
        "--served-model-name", settings.LLM_MODEL_PATH,
        "--limit-mm-per-prompt", '{"image":1}',
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
    ]

    chat_template_path = Path(model_path) / "chat_template.jinja"
    if chat_template_path.exists():
        vllm_command.extend(["--chat-template", str(chat_template_path)])
        logger.info(f"\nUsing vLLM chat template: {chat_template_path}")

    proc = await asyncio.create_subprocess_exec(
        *vllm_command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        **process_kwargs,
    )

    vllm_log_task = asyncio.create_task(_stream_vllm_logs(proc))

    await _wait_for_vllm(proc)
    
             
    logger.info("\nvLLM warming-up...")
    try:
        warmup = True
        await _create_chat_completion(
            [{"role": "user", "content": " "}],
            0,
            1,
            False,
            source="warmup",
        )
        logger.info("\nvLLM warm-up complete")
        warmup = False
    except Exception as E:
        logger.error(f"\nUnable to do vLLM warm-up : {E}")

    logger.info("\nvLLM backend ready")
    cp_runtime.set_module_active("llm", True)
    return proc

async def _ensure_loaded():
    if proc is None and vllm_client is None:
        logger.info("LLM not loaded")
        logger.info("LLM loading ...")
        await load_llm()
        logger.info("LLM loaded !")

async def unload_llm():
    global proc, vllm_log_task, vllm_client, model_path

    if proc is not None and proc.returncode is None:
        logger.info("\nStopping vLLM backend...")
        try:
            if os.name == "nt":
                proc.terminate()
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            return

        try:
            await asyncio.wait_for(proc.wait(), timeout=30)
        except asyncio.TimeoutError:
            logger.warning("\nvLLM did not stop after SIGTERM, terminating it...")
            if os.name == "nt":
                proc.kill()
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            await proc.wait()

    if vllm_client is not None:
        await vllm_client.close()

    proc = None
    vllm_log_task = None
    vllm_client = None
    model_path = None

    gc.collect()
    cp_runtime.set_module_active("llm", False)
    return


async def build_prompt(prompt, username, img_path, preformatted=False, metadata=None):
    global user_id, language

    profile_start = time.perf_counter()
    image_s = 0.0
    context_load_s = 0.0
    langdetect_s = 0.0
    game_context_s = 0.0
    rag_s = 0.0
    prompt_text = str(prompt or "").strip()
    if img_path:
        stage_start = time.perf_counter()
        img = _image_to_data_url(img_path)
        image_s = time.perf_counter() - stage_start
    else:
        img = None

    stage_start = time.perf_counter()
    states_path = Path(__file__).resolve().parents[2] / "config" / "states.json"
    with states_path.open("r", encoding="utf-8") as f:
        states = json.load(f)

    contexts_dir = Path(__file__).resolve().parent / "contexts"
    with (contexts_dir / "format_context.json").open("r", encoding="utf-8") as f:
        format_ctx = json.load(f)
    with (contexts_dir / "main_context.json").open("r", encoding="utf-8") as f:
        main_ctx = json.load(f)
    with (contexts_dir / "sub_context.json").open("r", encoding="utf-8") as f:
        sub_ctx = json.load(f)
    with (contexts_dir / "tool_context.json").open("r", encoding="utf-8") as f:
        tool_ctx = json.load(f)
    with (contexts_dir / "twitch_context.json").open("r", encoding="utf-8") as f:
        twitch_ctx = json.load(f)
    with (contexts_dir / "emotes_context.json").open("r", encoding="utf-8") as f:
        emotes_ctx = json.load(f)
    context_load_s = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    language = _internal_command_language(prompt_text) or _detect_prompt_language(prompt_text)
    langdetect_s = time.perf_counter() - stage_start
    
    native_tool_mode = _native_tools_enabled()
    twitch_mode = _twitch_context_enabled()

    system_full = f"{format_ctx.get('normal_format_fr') if language == 'fr' else format_ctx.get('normal_format_en')}"
    system_full = f"{system_full}\n{main_ctx.get('fr') if language == 'fr' else main_ctx.get('en')}"

    system_full = (
        f"{system_full}\n"
        f"{emotes_ctx.get('emotes_normal_fr') if language == 'fr' else emotes_ctx.get('emotes_normal_en')}\n"
    )

    if native_tool_mode:
        system_full = (
            f"{system_full}"
            f"{tool_ctx.get('tools_fr') if language == 'fr' else tool_ctx.get('tools_en')}\n"
        )
        file_context = _format_agatha_env_file_context(language)
        if file_context:
            system_full = f"{system_full}{file_context}\n"

    if twitch_mode:
        system_full = f"{system_full}{_twitch_context_note(twitch_ctx, language)}"

    if states.get("is_gaming"):
        stage_start = time.perf_counter()
        final = ""
        logger.debug("GET is_gaming -> True")
        playing_ctx = sub_ctx.get("game_mode", "") if language == "fr" else sub_ctx.get("game_mode_en", "")
        if states.get("is_playing_chess"):
            logger.debug("GAME = CHESS")
            game = "Échecs"
            try:
                from src.AI.game_ai.chess.Lichess import get_current_game_snapshot
                snapshot = get_current_game_snapshot()
            except Exception:
                logger.exception("Failed to get chess snapshot")
                snapshot = None
        
            if snapshot:
                fen = snapshot.get("fen", "")
                turn = snapshot.get("turn", "")
                last_me = snapshot.get("last_agatha_move", "")
                last_opp = snapshot.get("last_opponent_move", "")
                board_ascii = snapshot.get("board_ascii", "")
                moves_tail = snapshot.get("moves_uci_tail", [])
        
                chess_game_state = (
                    "État actuel de la partie d'Échecs :\n"
                    f"FEN: {fen}\n"
                    f"Tour: {turn}\n"
                    f"Ton dernier coup : {last_me}\n"
                    f"Dernier coup de ton adversaire: {last_opp}\n"
                    f"Historique récent (UCI): {moves_tail}\n"
                    f"Plateau:\n{board_ascii}\n"
                )
            else:
                chess_game_state = "Impossible de récupérer l'état actuel de la partie d'Échecs.\n"
                
            final = chess_game_state
        elif states.get("is_playing_starcraft2"):
            game = "StarCraft II"
            logger.debug("GAME = StarCraft II")
        else:
            game = "ERREUR : Jeu inconnu"
            logger.warning("UNABLE TO FIND GAME")
        system_full = f"{system_full}\n\n{playing_ctx} {game} avec {username}.\n{final}"
        game_context_s = time.perf_counter() - stage_start
        
    memory_query = prompt_text if preformatted else f"{username}: {prompt_text}".strip()
    stage_start = time.perf_counter()
    memory_context = await build_rag_context(memory_query, language=language)
    rag_s = time.perf_counter() - stage_start
    if memory_context:
        system_full = f"{system_full}\n\n{memory_context}"

    temporary_context = cp_runtime.temporary_context().strip()
    if temporary_context:
        system_full = f"{temporary_context}\n{system_full}"

    if preformatted:
        user_text = prompt_text
    else:
        vip_name, vip_text = _vip_display_name(user_id, username)
        user_text = f"{vip_name}{vip_text} : {prompt_text}"

    attachment_context = ""
    if settings.FILES_SEARCH:
        attachment_context = _format_attachment_context((metadata or {}).get("attachments") or [])
    if attachment_context:
        user_text = f"{user_text}\n\n{attachment_context}"

    if img:
        messages = [
            {"role": "system", "content": system_full},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": img}},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
    else:
        messages = [
            {"role": "system", "content": system_full},
            {"role": "user", "content": user_text},
        ]

    logger.debug(f"MESSAGES : {messages}")
    logger.info(
        "\n[PIPELINE:PROMPT] source=%s total=%.4fs image=%.4fs context_load=%.4fs "
        "langdetect=%.4fs game=%.4fs rag=%.4fs prompt_chars=%s image_attached=%s",
        (metadata or {}).get("source") or "",
        time.perf_counter() - profile_start,
        image_s,
        context_load_s,
        langdetect_s,
        game_context_s,
        rag_s,
        sum(len(str(message.get("content", ""))) for message in messages if isinstance(message, dict)),
        bool(img),
    )

    return messages


async def non_stream_output(prompt, source):
    global user_id
    profile_start = time.perf_counter()
    extract_s = 0.0
    build_prompt_s = 0.0
    completion_s = 0.0
    postprocess_s = 0.0
    remember_s = 0.0
    vts_s = 0.0
    used_tools = False
    try:
        stage_start = time.perf_counter()
        content, user_name, user_id, img_path, metadata = _extract_prompt_data(prompt, source)
        content, user_name, preformatted = _prepare_prompt_content(
            content,
            user_name,
            user_id,
            metadata,
        )
        extract_s = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        messages = await build_prompt(
            content,
            user_name,
            img_path,
            preformatted=preformatted,
            metadata=metadata,
        )
        build_prompt_s = time.perf_counter() - stage_start

        native_tools = _enabled_native_tools()
        if native_tools:
            used_tools = True
            stage_start = time.perf_counter()
            assistant_message = await _create_chat_completion(
                messages,
                settings.LLM_TEMPERATURE,
                settings.LLM_TOP_P,
                False,
                source=source,
                tools=native_tools,
                tool_choice="auto",
                return_message=True,
            )
            decoded_output = await _complete_native_tool_followups(
                messages,
                assistant_message,
                metadata,
                source,
                native_tools,
            )
            completion_s = time.perf_counter() - stage_start
        else:
            stage_start = time.perf_counter()
            decoded_output = await _create_chat_completion(
                messages,
                settings.LLM_TEMPERATURE,
                settings.LLM_TOP_P,
                False,
                source=source,
            )
            completion_s = time.perf_counter() - stage_start
        
        if settings.VTUBING and settings.DEV_MODE:                            
            stage_start = time.perf_counter()
            await move_by_sentence(decoded_output)
            vts_s = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        final = clean_model_output(decoded_output)
        final = _redact_moderated_text(final, context="LLM non-stream response")
        cp_runtime.append_stream_preview(final)
        cp_runtime.finish_generation()
        postprocess_s = time.perf_counter() - stage_start

        logger.info(f"\nAgathaAI :\n{final}")
        stage_start = time.perf_counter()
        await _remember_rag_interaction(
            content,
            final,
            source=source,
            user_name=user_name,
            user_id=user_id if source == "discord" else None,
            img_path=img_path,
            metadata=metadata if source == "discord" else None,
        )
        remember_s = time.perf_counter() - stage_start
        logger.info(
            "\n[PIPELINE:LLM_TEXT] source=%s total=%.4fs extract=%.4fs build_prompt=%.4fs "
            "completion=%.4fs vts=%.4fs postprocess=%.4fs remember=%.4fs used_tools=%s response_chars=%s",
            source,
            time.perf_counter() - profile_start,
            extract_s,
            build_prompt_s,
            completion_s,
            vts_s,
            postprocess_s,
            remember_s,
            used_tools,
            len(final),
        )
        return final

    except Exception as E:
        cp_runtime.fail_generation(str(E))
        logger.exception(f"\nUnable to generate prompt : {E}")
        return "An error occured, please tell Axel." if language == "en" else "Une erreur est survenue, veuillez prévenir Axel."


def _redact_moderated_text(text, *, context):
    redacted, censored_words = redact_blacklisted_words(text)
    if censored_words:
        logger.warning(
            "\n%s censored by moderation blacklist: %s",
            context,
            ", ".join(censored_words),
        )
    return redacted


async def _emit_stream_text(text, *, detected_lang=None, context, log_label="Sentence"):
    text = clean_model_output(text, True)
    if not text:
        return detected_lang, ""

    text = _redact_moderated_text(text, context=context)
    cp_runtime.append_stream_preview(f"{text} ")

    if detected_lang is None:
        detected_lang = _detect_prompt_language(text)

    logger.info(f"{log_label} : {text}")
    await get_input(text, True, detected_lang)
    return detected_lang, text


async def _stream_completion_to_outputs(response_stream, *, detected_lang=None):
    buffer = ""
    full_response = ""
    saw_tool_call_delta = False

    async for chunk in response_stream:
        if _chunk_tool_calls_delta(chunk):
            saw_tool_call_delta = True

        delta = _chunk_content(chunk)
        if not delta:
            continue

        buffer += delta
        full_response += delta

        sentences, buffer = pop_by_sentences(buffer)

        for sentence in sentences:
            detected_lang, _ = await _emit_stream_text(
                sentence,
                detected_lang=detected_lang,
                context="LLM streaming sentence",
            )

    remaining = buffer.strip()
    if remaining:
        detected_lang, _ = await _emit_stream_text(
            remaining,
            detected_lang=detected_lang,
            context="LLM streaming remainder",
        )

    full_response = _redact_moderated_text(full_response, context="LLM full streamed response")
    return full_response, detected_lang, saw_tool_call_delta


async def _stream_plain_text_retry(messages, *, source, detected_lang=None, reason=""):
    retry_note = (
        "\n\n=== RETRY SORTIE VOCALE ===\n"
        "La tentative precedente a produit un appel outil ou une sortie vide. "
        "Reponds maintenant uniquement avec l'objet JSON attendu, sans appel outil, "
        "sans balise tool_call, avec une cle response non vide et actions vide si aucune action n'est utile."
    )
    retry_messages = []
    note_inserted = False
    for message in messages:
        if (
            not note_inserted
            and isinstance(message, dict)
            and message.get("role") == "system"
            and isinstance(message.get("content"), str)
        ):
            message = dict(message)
            message["content"] = message["content"] + retry_note
            note_inserted = True
        retry_messages.append(message)

    if not note_inserted:
        retry_messages.insert(0, {"role": "system", "content": retry_note.strip()})

    logger.warning(
        "\nRetrying empty streamed response as plain text. source=%s reason=%s",
        source,
        reason,
    )
    response_stream = await _create_chat_completion(
        retry_messages,
        settings.LLM_TEMPERATURE,
        settings.LLM_TOP_P,
        stream=True,
        source=source,
    )
    return await _stream_completion_to_outputs(
        response_stream,
        detected_lang=detected_lang,
    )


async def stream_output(prompt, source):
    global user_id

    pipeline_start = time.perf_counter()
    timings = {}
    retried_empty_stream = False
    saw_tool_call_delta = False
    used_tools = False

    try:
        extract_start = time.perf_counter()
        content, user_name, user_id, img_path, metadata = _extract_prompt_data(prompt, source)
        content, user_name, preformatted = _prepare_prompt_content(
            content,
            user_name,
            user_id,
            metadata,
        )
        timings["extract_s"] = round(time.perf_counter() - extract_start, 4)

        build_start = time.perf_counter()
        messages = await build_prompt(
            content,
            user_name,
            img_path,
            preformatted=preformatted,
            metadata=metadata,
        )
        timings["build_prompt_s"] = round(time.perf_counter() - build_start, 4)

        native_tools = _enabled_native_tools()
        if native_tools:
            used_tools = True
            initial_start = time.perf_counter()
            assistant_message = await _create_chat_completion(
                messages,
                settings.LLM_TEMPERATURE,
                settings.LLM_TOP_P,
                False,
                source=source,
                tools=native_tools,
                tool_choice="auto",
                return_message=True,
            )
            timings["initial_completion_s"] = round(time.perf_counter() - initial_start, 4)
            detected_lang = None
            if _message_tool_calls(assistant_message):
                preface_start = time.perf_counter()
                detected_lang, _ = await _emit_stream_text(
                    _message_content(assistant_message),
                    detected_lang=detected_lang,
                    context="LLM native tool preface",
                    log_label="Tool preface",
                )
                timings["tool_preface_emit_s"] = round(time.perf_counter() - preface_start, 4)

            followup_start = time.perf_counter()
            full_response, detected_lang, saw_tool_call_delta = await _stream_native_tool_followup(
                messages,
                assistant_message,
                metadata,
                source,
                native_tools,
                detected_lang=detected_lang,
            )
            timings["tool_followup_stream_emit_s"] = round(time.perf_counter() - followup_start, 4)

            if not full_response.strip() and saw_tool_call_delta:
                retry_start = time.perf_counter()
                full_response, detected_lang, _ = await _stream_plain_text_retry(
                    messages,
                    source=source,
                    detected_lang=detected_lang,
                    reason="native_tool_followup_tool_call_delta",
                )
                retried_empty_stream = True
                timings["empty_stream_retry_s"] = round(time.perf_counter() - retry_start, 4)

            post_start = time.perf_counter()
            final = clean_model_output(full_response, True)
            final = _redact_moderated_text(final, context="LLM native tool streamed response")
            timings["postprocess_s"] = round(time.perf_counter() - post_start, 4)

            if settings.VTUBING:
                vts_start = time.perf_counter()
                await move_by_sentence(final)
                timings["vts_s"] = round(time.perf_counter() - vts_start, 4)

            cp_runtime.finish_generation()
            logger.info(f"\nAgathaAI streamed:\n{final}")
            remember_start = time.perf_counter()
            await _remember_rag_interaction(
                content,
                final,
                source=source,
                user_name=user_name,
                user_id=user_id,
                img_path=img_path,
                metadata=metadata,
            )
            timings["remember_s"] = round(time.perf_counter() - remember_start, 4)
            logger.info(
                "\n[PIPELINE:LLM_STREAM] source=%s total=%.4fs timings=%s "
                "used_tools=%s retried_empty_stream=%s saw_tool_call_delta=%s response_chars=%s",
                source,
                time.perf_counter() - pipeline_start,
                json.dumps(timings, sort_keys=True),
                used_tools,
                retried_empty_stream,
                saw_tool_call_delta,
                len(final),
            )
            return final

        stream_start = time.perf_counter()
        response_stream = await _create_chat_completion(
            messages,
            settings.LLM_TEMPERATURE,
            settings.LLM_TOP_P,
            stream=True,
            source=source,
        )

        full_response, detected_lang, saw_tool_call_delta = await _stream_completion_to_outputs(response_stream)
        timings["stream_emit_s"] = round(time.perf_counter() - stream_start, 4)
        if not full_response.strip() and saw_tool_call_delta:
            retry_start = time.perf_counter()
            full_response, detected_lang, _ = await _stream_plain_text_retry(
                messages,
                source=source,
                detected_lang=detected_lang,
                reason="tool_call_delta_without_enabled_tools",
            )
            retried_empty_stream = True
            timings["empty_stream_retry_s"] = round(time.perf_counter() - retry_start, 4)

        if settings.VTUBING:
            vts_start = time.perf_counter()
            await move_by_sentence(full_response)
            timings["vts_s"] = round(time.perf_counter() - vts_start, 4)
            
        post_start = time.perf_counter()
        final = clean_model_output(full_response, True)
        timings["postprocess_s"] = round(time.perf_counter() - post_start, 4)
        cp_runtime.finish_generation()
        logger.info(f"\nAgathaAI streamed:\n{final}")
        remember_start = time.perf_counter()
        await _remember_rag_interaction(
            content,
            final,
            source=source,
            user_name=user_name,
            user_id=user_id,
            img_path=img_path,
            metadata=metadata,
        )
        timings["remember_s"] = round(time.perf_counter() - remember_start, 4)
        logger.info(
            "\n[PIPELINE:LLM_STREAM] source=%s total=%.4fs timings=%s "
            "used_tools=%s retried_empty_stream=%s saw_tool_call_delta=%s response_chars=%s",
            source,
            time.perf_counter() - pipeline_start,
            json.dumps(timings, sort_keys=True),
            used_tools,
            retried_empty_stream,
            saw_tool_call_delta,
            len(final),
        )
        return final

    except Exception as E:
        cp_runtime.fail_generation(str(E))
        logger.exception(f"\nUnable to stream prompt: {E}")
        return "An error occured, please tell Axel." if language == "en" else "Une erreur est survenue, veuillez prévenir Axel."


def _extract_prompt_data(prompt, source):
    if source == "discord":
        content = prompt["content"]
        user_name = prompt["user_name"]
        prompt_user_id = prompt["user_id"]
        img_path = prompt.get("img_path") or prompt.get("image_path") or prompt.get("image_url")
        metadata = _extract_prompt_metadata(prompt)
    elif source == "lichess":
        content = prompt["content"]
        user_name = prompt["user_name"]
        prompt_user_id = None
        img_path = None
        metadata = {}
    elif isinstance(prompt, dict):
        content = prompt.get("content", "")
        user_name = prompt.get("user_name", "User")
        prompt_user_id = prompt.get("user_id")
        img_path = prompt.get("img_path") or prompt.get("image_path") or prompt.get("image_url")
        metadata = _extract_prompt_metadata(prompt)
    else:
        content = prompt
        user_name = "User"
        prompt_user_id = None
        img_path = None
        metadata = {}

    return content or "", user_name, prompt_user_id, img_path, metadata


def _extract_prompt_metadata(prompt):
    return {
        "channel_id": prompt.get("channel_id"),
        "guild_id": prompt.get("guild_id"),
        "message_id": prompt.get("message_id"),
        "is_dm": prompt.get("is_dm"),
        "source": prompt.get("source"),
        "preformatted": prompt.get("preformatted", False),
        "speaker_turns": prompt.get("speaker_turns"),
        "attachments": prompt.get("attachments") or [],
    }


def _prepare_prompt_content(content, user_name, prompt_user_id, metadata):
    metadata = metadata or {}
    preformatted = bool(metadata.get("preformatted"))
    if metadata.get("source") != "voice_conversation":
        return content, user_name, preformatted

    normalized = _format_voice_speaker_turns(
        metadata.get("speaker_turns"),
        prompt_user_id,
        user_name,
    )
    if not normalized:
        return content, user_name, preformatted

    logger.info(
        "\nNormalized voice prompt from speaker_turns: turns=%s chars=%s",
        len(metadata.get("speaker_turns") or []),
        len(normalized),
    )
    return normalized, user_name, True


def _format_voice_speaker_turns(speaker_turns, fallback_user_id, fallback_name):
    if not isinstance(speaker_turns, list):
        return ""

    lines = []
    for turn in speaker_turns:
        if not isinstance(turn, dict):
            continue

        text = str(turn.get("content") or "").strip()
        if not text:
            continue

        turn_user_id = (
            turn.get("userId")
            or turn.get("user_id")
            or fallback_user_id
        )
        turn_name = (
            turn.get("userName")
            or turn.get("user_name")
            or fallback_name
        )
        display_name, display_suffix = _vip_display_name(turn_user_id, turn_name)
        lines.append(f"{display_name}{display_suffix} : {text}")

    return "\n".join(lines)


def _vip_display_name(prompt_user_id, fallback_name):
    return VIP_USERS.get(
        str(prompt_user_id or ""),
        (_sanitize_user_display_name(fallback_name), ""),
    )


def _sanitize_user_display_name(name):
    text = str(name or "User").strip()
    text = "".join(
        grapheme
        for grapheme in regex.findall(r"\X", text)
        if not regex.search(r"\p{Extended_Pictographic}", grapheme)
    )
    text = regex.sub(r"\p{M}+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or "User"


def _format_attachment_context(attachments):
    readable_attachments = [
        attachment
        for attachment in attachments
        if _is_readable_attachment_metadata(attachment)
    ]
    if not readable_attachments:
        return ""

    lines = ["Pieces jointes disponibles pour read_file:"]
    for index, attachment in enumerate(readable_attachments, start=1):
        name = attachment.get("name") or attachment.get("filename") or f"attachment_{index}"
        content_type = attachment.get("contentType") or attachment.get("content_type") or "type inconnu"
        size = attachment.get("size")
        size_text = f", {size} octets" if size is not None else ""
        lines.append(f"{index}. {name} ({content_type}{size_text})")

    lines.append('Pour lire une piece jointe non-media, appelle read_file avec {"file":"son_nom"}.')
    return "\n".join(lines)


def _format_agatha_env_file_context(prompt_language):
    if not (settings.FILES_SEARCH or settings.EDIT_FILE):
        return ""

    files, truncated = _list_agatha_env_files()
    if prompt_language == "fr":
        lines = ["=== FICHIERS DISPONIBLES : agathaai_env/ ===", ""]
        if files:
            lines.append("Fichiers visibles pour read_file/edit_file :")
            lines.extend(f"- {item}" for item in files)
            if truncated:
                lines.append("- ...")
            lines.append("")
            lines.append("Utilise ces noms tels quels quand tu lis ou modifies un fichier existant.")
        else:
            lines.append("Aucun fichier n'est actuellement présent dans agathaai_env/.")
            lines.append("Si l'utilisateur demande une lecture, n'invente pas de fichier existant.")
            lines.append("Si l'utilisateur demande explicitement une création, edit_file peut créer le fichier demandé.")
    else:
        lines = ["=== AVAILABLE FILES: agathaai_env/ ===", ""]
        if files:
            lines.append("Files visible to read_file/edit_file:")
            lines.extend(f"- {item}" for item in files)
            if truncated:
                lines.append("- ...")
            lines.append("")
            lines.append("Use these names exactly when reading or modifying an existing file.")
        else:
            lines.append("No file is currently present in agathaai_env/.")
            lines.append("If the user asks to read a file, do not invent an existing file.")
            lines.append("If the user explicitly asks to create a file, edit_file may create the requested file.")

    return "\n".join(lines)


def _list_agatha_env_files(max_files=40):
    root = AGATHA_ENV_ROOT
    if not root.exists() or not root.is_dir():
        return [], False

    entries = []
    for path in sorted(root.rglob("*")):
        if len(entries) >= max_files:
            return entries, True
        if not path.is_file():
            continue
        try:
            relative = path.relative_to(root).as_posix()
            size = path.stat().st_size
        except OSError:
            continue
        content_type = mimetypes.guess_type(relative)[0] or "type inconnu"
        entries.append(f"{relative} ({content_type}, {size} octets)")

    return entries, False


def _is_readable_attachment_metadata(attachment):
    name = str(attachment.get("name") or attachment.get("filename") or attachment.get("url") or "")
    content_type = str(attachment.get("contentType") or attachment.get("content_type") or "")
    lower_name = name.lower()
    lower_type = content_type.lower()

    if lower_type.startswith(("audio/", "image/", "video/")):
        return False
    if re.search(r"\.(3gp|aac|aiff|ape|avi|bmp|flac|gif|heic|ico|jpe?g|m4a|mkv|mov|mp3|mp4|mpe?g|ogg|opus|png|tiff?|wav|webm|webp|wmv)(?:$|\?)", lower_name):
        return False
    if re.search(r"\.(apk|app|bin|class|com|dll|dmg|dylib|elf|exe|jar|msi|o|obj|scr|so)(?:$|\?)", lower_name):
        return False

    return True


def _native_tools_enabled():
    return (
        settings.WEB_SEARCH
        or settings.FILES_SEARCH
        or settings.EDIT_FILE
        or _twitch_action_tools_enabled()
    )


def _twitch_action_tools_enabled():
    return settings.TWITCH_POLL or settings.TWITCH_BAN or settings.TWITCH_TIMEOUT


def _twitch_context_enabled():
    return (
        settings.TWITCH_CHAT
        or settings.TWITCH_POLL
        or settings.TWITCH_BAN
        or settings.TWITCH_TIMEOUT
    )


def _twitch_context_note(twitch_ctx, prompt_language):
    if _twitch_action_tools_enabled():
        return (twitch_ctx.get("native_fr") if prompt_language == "fr" else twitch_ctx.get("native_en")) or ""
    return (twitch_ctx.get("chat_fr") if prompt_language == "fr" else twitch_ctx.get("chat_en")) or ""


def _enabled_native_tools():
    tools = []
    if settings.WEB_SEARCH:
        tools.append(web_search_tool)
    if settings.FILES_SEARCH:
        tools.append(read_file_tool)
    if settings.EDIT_FILE:
        tools.append(edit_file_tool)
    if settings.TWITCH_POLL:
        tools.append(twitch_create_poll_tool)
    if settings.TWITCH_BAN:
        tools.append(twitch_ban_user_tool)
    if settings.TWITCH_TIMEOUT:
        tools.append(twitch_timeout_user_tool)
    return tools


async def _complete_native_tool_followups(
    messages,
    initial_message,
    metadata,
    source,
    tools,
    max_rounds=2,
):
    current_messages = list(messages)
    assistant_message = initial_message

    for _ in range(max_rounds):
        tool_calls = _message_tool_calls(assistant_message)
        recovered_inline_call = False
        if not tool_calls:
            tool_calls = _recover_inline_tool_calls(assistant_message, tools)
            recovered_inline_call = bool(tool_calls)
            if recovered_inline_call:
                logger.warning(
                    "\nRecovered %s inline tool call(s) from assistant content.",
                    len(tool_calls),
                )

        if not tool_calls:
            return _filter_tool_call_payload(_message_content(assistant_message), tools)

        serialized_tool_calls = [_serialize_tool_call(tool_call) for tool_call in tool_calls]
        current_messages.append(
            _assistant_tool_call_message(
                assistant_message,
                serialized_tool_calls,
                include_content=not recovered_inline_call,
            )
        )

        for tool_call, serialized_tool_call in zip(tool_calls, serialized_tool_calls):
            tool_name = serialized_tool_call["function"]["name"]
            tool_call_id = serialized_tool_call["id"]
            tool_result = await execute_native_tool_call(tool_call, context=metadata)

            current_messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": tool_result,
            })

        assistant_message = await _create_chat_completion(
            current_messages,
            settings.LLM_TEMPERATURE,
            settings.LLM_TOP_P,
            False,
            source=source,
            return_message=True,
        )

    logger.warning("\nNative tool call loop reached max_rounds=%s", max_rounds)
    return _filter_tool_call_payload(_message_content(assistant_message), tools)


async def _stream_native_tool_followup(
    messages,
    initial_message,
    metadata,
    source,
    tools,
    *,
    detected_lang=None,
):
    current_messages = list(messages)
    tool_calls = _message_tool_calls(initial_message)
    recovered_inline_call = False
    if not tool_calls:
        tool_calls = _recover_inline_tool_calls(initial_message, tools)
        recovered_inline_call = bool(tool_calls)
        if recovered_inline_call:
            logger.warning(
                "\nRecovered %s inline tool call(s) from assistant content.",
                len(tool_calls),
            )

    if not tool_calls:
        content = _filter_tool_call_payload(_message_content(initial_message), tools)
        await _emit_stream_text(
            content,
            detected_lang=detected_lang,
            context="LLM native tool direct response",
        )
        return content, detected_lang, False

    serialized_tool_calls = [_serialize_tool_call(tool_call) for tool_call in tool_calls]
    current_messages.append(
        _assistant_tool_call_message(
            initial_message,
            serialized_tool_calls,
            include_content=not recovered_inline_call,
        )
    )

    for tool_call, serialized_tool_call in zip(tool_calls, serialized_tool_calls):
        tool_name = serialized_tool_call["function"]["name"]
        tool_call_id = serialized_tool_call["id"]
        tool_result = await execute_native_tool_call(tool_call, context=metadata)

        current_messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": tool_result,
        })

    response_stream = await _create_chat_completion(
        current_messages,
        settings.LLM_TEMPERATURE,
        settings.LLM_TOP_P,
        stream=True,
        source=source,
    )
    full_response, detected_lang, saw_tool_call_delta = await _stream_completion_to_outputs(
        response_stream,
        detected_lang=detected_lang,
    )
    return full_response, detected_lang, saw_tool_call_delta


def _assistant_tool_call_message(assistant_message, serialized_tool_calls, *, include_content=True):
    content = _message_content(assistant_message)
    message = {
        "role": "assistant",
        "tool_calls": serialized_tool_calls,
    }
    if include_content and content:
        message["content"] = content
    return message


def _serialize_tool_call(tool_call):
    return {
        "id": _tool_call_id(tool_call),
        "type": _tool_call_type(tool_call),
        "function": {
            "name": _tool_call_name(tool_call),
            "arguments": _tool_call_arguments(tool_call),
        },
    }


def _message_content(message):
    return _object_value(message, "content") or ""


def _message_tool_calls(message):
    return _object_value(message, "tool_calls") or []


def _recover_inline_tool_calls(message, tools):
    payload = _parse_inline_tool_call_payload(_message_content(message))
    if not payload:
        return []

    name = str(payload.get("name") or "").strip()
    allowed_names = {
        str(tool.get("function", {}).get("name") or "").strip()
        for tool in tools or []
    }
    if not name or name not in allowed_names:
        return []

    arguments = payload.get("arguments")
    if not isinstance(arguments, (dict, str)):
        return []

    if isinstance(arguments, dict):
        arguments = json.dumps(arguments, ensure_ascii=False)

    return [{
        "id": f"recovered_{uuid.uuid4().hex}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }]


def _parse_inline_tool_call_payload(content):
    text = str(content or "").strip()
    if not text:
        return None

    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    match = re.fullmatch(r"\s*<tool_call>\s*(.*?)\s*</tool_call>\s*", text, flags=re.DOTALL)
    if match:
        text = match.group(1).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None
    if set(payload).difference({"name", "arguments"}):
        return None
    if "name" not in payload or "arguments" not in payload:
        return None
    return payload


def _filter_tool_call_payload(content, tools):
    if _recover_inline_tool_calls({"content": content}, tools):
        logger.warning("\nSuppressed inline tool-call JSON from final model output.")
        return ""
    return str(content or "")


def _tool_call_id(tool_call):
    return str(_object_value(tool_call, "id") or uuid.uuid4().hex)


def _tool_call_type(tool_call):
    return str(_object_value(tool_call, "type") or "function")


def _tool_call_name(tool_call):
    function = _object_value(tool_call, "function") or {}
    return str(_object_value(function, "name") or "")


def _tool_call_arguments(tool_call):
    function = _object_value(tool_call, "function") or {}
    arguments = _object_value(function, "arguments")
    if isinstance(arguments, dict):
        return json.dumps(arguments, ensure_ascii=False)
    return str(arguments or "{}")


def _object_value(obj, key):
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


async def _remember_rag_interaction(
    content,
    final,
    *,
    source,
    user_name,
    user_id,
    img_path,
    metadata,
):
    if not final:
        return

    try:
        await remember_interaction(
            content,
            final,
            source=source,
            user_name=user_name,
            user_id=user_id,
            has_image=bool(img_path),
            metadata=metadata,
        )
    except Exception as exc:
        logger.warning("\nUnable to save RAG memory entry: %s", exc)


def _internal_command_language(prompt):
    prompt = str(prompt or "")
    if not prompt.lstrip().startswith("[INTERNAL_COMMAND]"):
        return None

    match = re.search(r"(?im)^\s*language\s*:\s*(fr|en)\s*$", prompt)
    return match.group(1).lower() if match else None


def _detect_prompt_language(prompt):
    try:
        return "fr" if detect(str(prompt or "")) == "fr" else "en"
    except Exception:
        return "fr" if settings.LANGUAGE.lower() == "fr" else "en"


def clean_model_output(input, vc=False):
    emotes = ["emote_frown", "emote_smile", "emote_scary_face"]
    tools = [
        "web_search",
        "edit_file",
        "read_file",
        "create_twitch_poll",
        "ban_twitch_user",
        "timeout_twitch_user",
    ]
    pattern = r"(" + "|".join(emotes) + ")"
    
    logger.info(f"\nMODEL OUTPUT BEFORE CLEANING : \n{input}\n")        

    if _parse_inline_tool_call_payload(input):
        logger.warning("\nSuppressed inline tool-call JSON before output/TTS.")
        return ""
    
    tool_mode = _native_tools_enabled()
    parser = parse_tool_json if tool_mode else parse_json

    result = None
    try:
        parsed = parser(input)
        if isinstance(parsed, dict):
            result = parsed.get("response")
    except Exception as E:
        logger.error(f"\n\nModel output JSON parser failed : {E}\n\n")

    if result is None:
        result = extract_json_response_fragment(input)

    if result is None:
        result = str(input or "")
    else:
        result = str(result)

    cleaned_output = re.sub(pattern, "", result, flags=re.IGNORECASE).strip()
    cleaned_output = cleaned_output.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()

    first_line = cleaned_output.split("\n")[0].strip().lower()
    
    if vc:
        cleaned_output = re.sub(r"\.{3,}", ".", cleaned_output)
        cleaned_output = cleaned_output.replace("…", ".")

        cleaned_output = ''.join(
            x for x in regex.findall(r'\X', cleaned_output)
            if not regex.match(r'\p{Extended_Pictographic}', x)
        ).strip()
        
    for tool in tools:
        if first_line.startswith(tool + " "):
            return first_line[len(tool):].strip()

    return cleaned_output


def _resolve_model_path(raw_path):
    path = Path(raw_path).expanduser()
    llm_dir = Path(__file__).resolve().parent
    project_root = llm_dir.parents[2]

    candidates = [
        path,
        llm_dir / path,
        project_root / path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    return raw_path

def _make_vllm_client():
    return AsyncOpenAI(
        api_key=VLLM_API_KEY,
        base_url=VLLM_BASE_URL,
        timeout=settings.LLM_REQUEST_TIMEOUT,
    )


def _image_to_data_url(img_path):
    img = str(img_path)
    if img.startswith(("http://", "https://", "data:")):
        return img

    path = Path(img).expanduser().resolve()
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"

    with path.open("rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")

    return f"data:{mime_type};base64,{encoded}"


async def _stream_vllm_logs(proc):
    if proc.stdout is None:
        return

    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        text = line.decode("utf-8", errors="replace").rstrip()
        record_vllm_engine_metric_line(text)
        logger.info("\n[vLLM] %s", text)


async def _wait_for_vllm(proc):
    deadline = asyncio.get_running_loop().time() + settings.LLM_STARTUP_TIMEOUT

    while asyncio.get_running_loop().time() < deadline:
        if proc.returncode is not None:
            raise RuntimeError(f"vLLM stopped during startup with exit code {proc.returncode}")

        if await asyncio.to_thread(_is_vllm_healthy):
            return

        await asyncio.sleep(1.5)

    raise TimeoutError(f"vLLM did not become ready after {settings.LLM_STARTUP_TIMEOUT}s")


def _is_vllm_healthy():
    try:
        with urllib.request.urlopen(VLLM_HEALTH_URL, timeout=5) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


async def _create_chat_completion(
    messages,
    temperature,
    top_p,
    stream,
    source=None,
    tools=None,
    tool_choice=None,
    return_message=False,
):
    global stream_usage_supported
    
    await _ensure_loaded()

    if not warmup and source != "warmup" and cp_runtime.llm_disabled():
        raise RuntimeError("LLM generation blocked by control panel")
    
    if vllm_client is None:
        raise RuntimeError("vLLM OpenAI client is not initialized")
    
    extra_body = {                          
        "min_p": settings.LLM_MIN_P,
        "repetition_penalty": settings.LLM_REPETITION_PENALTY,
    }

    max_tokens = 1 if warmup else settings.LLM_MAX_NEW_TOKENS
    request_id = uuid.uuid4().hex[:12]
    started_at = datetime.now().isoformat(timespec="milliseconds")
    start_perf = time.perf_counter()
    request_stats = _message_stats(messages)
    if request_stats.get("image_count"):
        logger.info(
            "\nLLM request includes %s image(s), prompt_chars=%s, source=%s",
            request_stats["image_count"],
            request_stats["prompt_chars"],
            source,
        )

    use_stream_endpoint = stream or (not warmup and stream_usage_supported and not return_message)
    create_kwargs = {
        "model": settings.LLM_MODEL_PATH,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": settings.LLM_PRESENCE_PENALTY,
        "extra_body": extra_body,
        "timeout": settings.LLM_REQUEST_TIMEOUT,
        "stream": use_stream_endpoint,
    }

    if tools and not warmup:
        create_kwargs["tools"] = tools
        create_kwargs["tool_choice"] = tool_choice or "auto"
    elif not warmup:
        create_kwargs["tool_choice"] = "none"

    if use_stream_endpoint and stream_usage_supported:
        create_kwargs["stream_options"] = {"include_usage": True}

    try:
        try:
            response = await vllm_client.chat.completions.create(**create_kwargs)
        except Exception as exc:
            if use_stream_endpoint and stream_usage_supported and _should_retry_stream_without_usage(exc):
                stream_usage_supported = False
                create_kwargs.pop("stream_options", None)
                if stream:
                    logger.warning("\nvLLM stream token usage is unavailable; retrying without stream_options.")
                else:
                    logger.warning("\nvLLM stream token usage is unavailable; falling back to non-stream metrics.")
                    create_kwargs["stream"] = False
                    use_stream_endpoint = False
                response = await vllm_client.chat.completions.create(**create_kwargs)
            else:
                raise
    except Exception as exc:
        _record_llm_metric(
            request_id=request_id,
            started_at=started_at,
            start_perf=start_perf,
            source=source,
            mode="stream" if stream else "non_stream",
            success=False,
            request_stats=request_stats,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            error=exc,
        )
        raise
    
    if use_stream_endpoint and stream:
        return _measured_completion_stream(
            response,
            request_id=request_id,
            started_at=started_at,
            start_perf=start_perf,
            source=source,
            mode="stream",
            request_stats=request_stats,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    if use_stream_endpoint:
        return await _collect_measured_completion_stream(
            response,
            request_id=request_id,
            started_at=started_at,
            start_perf=start_perf,
            source=source,
            request_stats=request_stats,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    choice = response.choices[0]
    message = _choice_value(choice, "message")
    decoded_output = _choice_value(message, "content") or ""

    finish_reason = _choice_value(choice, "finish_reason")
    if return_message and _choice_value(message, "tool_calls"):
        finish_reason = finish_reason or "tool_calls"

    _record_llm_metric(
        request_id=request_id,
        started_at=started_at,
        start_perf=start_perf,
        source=source,
        mode="non_stream",
        success=True,
        request_stats=request_stats,
        usage=_usage_to_dict(getattr(response, "usage", None)),
        completion_chars=len(decoded_output),
        finish_reason=finish_reason,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    if return_message:
        return message

    return decoded_output


async def _measured_completion_stream(
    response,
    *,
    request_id,
    started_at,
    start_perf,
    source,
    mode,
    request_stats,
    temperature,
    top_p,
    max_tokens,
):
    usage = {}
    first_token_perf = None
    completion_chars = 0
    chunk_count = 0
    content_chunk_count = 0
    finish_reason = ""
    success = False
    error = None
    panel_visible = not warmup and source != "warmup"

    try:
        async for chunk in response:
            if panel_visible and cp_runtime.abort_generation_requested():
                cp_runtime.clear_abort_generation()
                logger.warning("\nLLM generation aborted by control panel")
                break

            chunk_count += 1
            chunk_usage = _usage_to_dict(getattr(chunk, "usage", None))
            if chunk_usage:
                usage = chunk_usage

            if getattr(chunk, "choices", None):
                choice = chunk.choices[0]
                finish_reason = _choice_value(choice, "finish_reason") or finish_reason
                delta = _choice_value(choice, "delta")
                content = _choice_value(delta, "content") if delta is not None else None

                if content:
                    if first_token_perf is None:
                        first_token_perf = time.perf_counter()
                    content_chunk_count += 1
                    completion_chars += len(content)
            yield chunk

        success = True
    except Exception as exc:
        error = exc
        raise
    finally:
        time_to_first_token_s = (
            first_token_perf - start_perf if first_token_perf is not None else None
        )
        _record_llm_metric(
            request_id=request_id,
            started_at=started_at,
            start_perf=start_perf,
            source=source,
            mode=mode,
            success=success,
            request_stats=request_stats,
            usage=usage,
            completion_chars=completion_chars,
            time_to_first_token_s=time_to_first_token_s,
            chunk_count=chunk_count,
            content_chunk_count=content_chunk_count,
            finish_reason=finish_reason,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            error=error,
        )


async def _collect_measured_completion_stream(
    response,
    *,
    request_id,
    started_at,
    start_perf,
    source,
    request_stats,
    temperature,
    top_p,
    max_tokens,
):
    parts = []

    async for chunk in _measured_completion_stream(
        response,
        request_id=request_id,
        started_at=started_at,
        start_perf=start_perf,
        source=source,
        mode="non_stream",
        request_stats=request_stats,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    ):
        content = _chunk_content(chunk)
        if content:
            parts.append(content)

    return "".join(parts)


def _record_llm_metric(
    *,
    request_id,
    started_at,
    start_perf,
    source,
    mode,
    success,
    request_stats,
    temperature,
    top_p,
    max_tokens,
    usage=None,
    completion_chars=0,
    time_to_first_token_s=None,
    chunk_count=0,
    content_chunk_count=0,
    finish_reason="",
    error=None,
):
    duration_s = max(time.perf_counter() - start_perf, 0.0)
    usage = usage or {}
    prompt_tokens = _int_or_none(usage.get("prompt_tokens"))
    completion_tokens = _int_or_none(usage.get("completion_tokens"))
    total_tokens = _int_or_none(usage.get("total_tokens"))

    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    generation_duration_s = None
    if mode == "stream" and time_to_first_token_s is not None:
        generation_duration_s = max(duration_s - time_to_first_token_s, 0.0)
    elif mode == "non_stream" and time_to_first_token_s is not None:
        generation_duration_s = max(duration_s - time_to_first_token_s, 0.0)
    elif mode == "non_stream":
        generation_duration_s = duration_s

    queue_prefill_s = time_to_first_token_s
    decode_s = generation_duration_s
    ttft_share = _safe_div(queue_prefill_s, duration_s)
    decode_share = _safe_div(decode_s, duration_s)
    prompt_tok_s = _safe_div(prompt_tokens, time_to_first_token_s or duration_s)
    prompt_prefill_tok_s = _safe_div(prompt_tokens, time_to_first_token_s)
    completion_tok_s = _safe_div(completion_tokens, generation_duration_s)
    total_tok_s = _safe_div(total_tokens, duration_s)
    completion_chars_s = _safe_div(completion_chars, generation_duration_s or duration_s)
    prompt_completion_ratio = _safe_div(prompt_tokens, completion_tokens)
    completion_chars_per_token = _safe_div(completion_chars, completion_tokens)

    record_llm_request_metric({
        "request_id": request_id,
        "timestamp": started_at,
        "source": source or "",
        "mode": mode,
        "model": settings.LLM_MODEL_PATH,
        "success": success,
        "is_warmup": bool(warmup),
        "message_count": request_stats["message_count"],
        "image_count": request_stats["image_count"],
        "prompt_chars": request_stats["prompt_chars"],
        "completion_chars": completion_chars,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "duration_s": _round_metric(duration_s),
        "time_to_first_token_s": _round_metric(time_to_first_token_s),
        "generation_duration_s": _round_metric(generation_duration_s),
        "queue_prefill_s": _round_metric(queue_prefill_s),
        "decode_s": _round_metric(decode_s),
        "ttft_share": _round_metric(ttft_share),
        "decode_share": _round_metric(decode_share),
        "prompt_tok_s": _round_metric(prompt_tok_s),
        "prompt_prefill_tok_s": _round_metric(prompt_prefill_tok_s),
        "completion_tok_s": _round_metric(completion_tok_s),
        "total_tok_s": _round_metric(total_tok_s),
        "completion_chars_s": _round_metric(completion_chars_s),
        "prompt_completion_ratio": _round_metric(prompt_completion_ratio),
        "completion_chars_per_token": _round_metric(completion_chars_per_token),
        "chunk_count": chunk_count,
        "content_chunk_count": content_chunk_count,
        "finish_reason": finish_reason or "",
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "error": _format_metric_error(error),
    })


def _message_stats(messages):
    stats = {
        "message_count": len(messages or []),
        "image_count": 0,
        "prompt_chars": 0,
    }

    for message in messages or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            stats["prompt_chars"] += len(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    stats["prompt_chars"] += len(str(part.get("text") or ""))
                elif part.get("type") == "image_url":
                    stats["image_count"] += 1

    return stats


def _usage_to_dict(usage):
    if usage is None:
        return {}

    if isinstance(usage, dict):
        return {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }

    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _choice_value(obj, attr):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(attr)
    return getattr(obj, attr, None)


def _chunk_content(chunk):
    if not getattr(chunk, "choices", None):
        return ""

    choice = chunk.choices[0]
    delta = _choice_value(choice, "delta")
    if delta is None:
        return ""

    return _choice_value(delta, "content") or ""


def _chunk_tool_calls_delta(chunk):
    if not getattr(chunk, "choices", None):
        return None

    choice = chunk.choices[0]
    delta = _choice_value(choice, "delta")
    if delta is None:
        return None

    return _choice_value(delta, "tool_calls")


def _safe_div(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    try:
        return float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _int_or_none(value):
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _round_metric(value):
    if value is None:
        return None
    return round(float(value), 4)


def _format_metric_error(error):
    if error is None:
        return ""
    return f"{type(error).__name__}: {error}"[:500]


def _should_retry_stream_without_usage(error):
    text = str(error).lower()
    return "stream_options" in text or "include_usage" in text

def pop_by_sentences(buffer: str):
    SENTENCE_END = re.compile(r"(.+?[.!?。！？…]+)(\s+|$)", re.DOTALL)
                                                                                            
    sentences = []

    while True:
        match = SENTENCE_END.match(buffer)
        if not match:
            break

        sentence = match.group(1).strip()
        if sentence:
            sentences.append(sentence)

        buffer = buffer[match.end():]

    return sentences, buffer

def parse_json(output):
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
        obj, end = decoder.raw_decode(text[start:])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def extract_json_response_fragment(output):
    if not isinstance(output, str):
        return None

    text = output.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    match = re.search(r'"response"\s*:', text)
    if match:
        value = text[match.end():].lstrip()
        if not value:
            return ""

        if value.startswith('"'):
            return _read_json_string_fragment(value)

        end = re.search(r'\s*,\s*"(?:actions|emote|tool|query)"\s*:', value)
        if end:
            return value[:end.start()].strip()

        return value.rstrip("}").strip()

    tail = _strip_json_tail_fields(text)
    return tail if tail != text else None


def _read_json_string_fragment(value):
    chars = []
    escaped = False
    index = 1

    while index < len(value):
        char = value[index]

        if escaped:
            if char == "n":
                chars.append("\n")
            elif char == "r":
                chars.append("\r")
            elif char == "t":
                chars.append("\t")
            elif char == "b":
                chars.append("\b")
            elif char == "f":
                chars.append("\f")
            elif char == "u" and index + 4 < len(value):
                hex_value = value[index + 1:index + 5]
                try:
                    chars.append(chr(int(hex_value, 16)))
                    index += 4
                except ValueError:
                    chars.append("\\u")
            else:
                chars.append(char)

            escaped = False
            index += 1
            continue

        if char == "\\":
            escaped = True
            index += 1
            continue

        if char == '"':
            return "".join(chars)

        chars.append(char)
        index += 1

    return "".join(chars)


def _strip_json_tail_fields(text):
    tail_match = re.search(
        r'"\s*,\s*"(?:actions|emote|tool|query)"\s*:',
        text,
        flags=re.DOTALL,
    )
    if tail_match:
        return text[:tail_match.start()].strip()

    return text
