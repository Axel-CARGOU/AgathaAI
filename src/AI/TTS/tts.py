import torch, logging, gc, datetime, asyncio, base64, json, math, time
from concurrent.futures import ThreadPoolExecutor
from types import MethodType
from websockets.asyncio.client import connect
from TTS.api import TTS
from pathlib import Path
from scipy.signal import resample_poly
import numpy as np

from src.control_panel import runtime as cp_runtime

logger = logging.getLogger("app")

tts, tts_voice, tts_ws, playing_audio = None, None, None, False
tts_ws_lock = asyncio.Lock()
audio_queue = asyncio.Queue()
worker_task = None
tts_executor = None

WS_URI = "ws://local.example:8765"
OUTPUT_DIR = Path("src/AI/TTS/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE_OUT = 48000
FRAME_MS = 20
SAMPLES_PER_FRAME = int(SAMPLE_RATE_OUT * FRAME_MS / 1000)
VOICE_ID = "AgathaAI"
XTTS_STREAM_CHUNK_SIZE = 20
XTTS_STREAM_OVERLAP_WAV_LEN = 1024
STREAM_DONE = object()


def _prepare_tts_text(text, lang):
    text = str(text or "").replace("[REDACTED]", "REDACTED")
    if lang == "fr":
        text = text.replace("Agatha", "Émilie").replace("Axel", "Némanthame")
        text = text.replace("VIPUser1", "Rozaline").replace("VIPUser2", "Hyène-Seingue")
        text = text.replace("VIPUser3", "Skaïe-nomme").replace("Vedal", "Védull")
    else:
        text = text.replace("Émilie", "Agatha").replace("Axel", "Nementhame")
        text = text.replace("Rozaline", "VIPUser1").replace("VIPUser2", "Yen-seng")
        text = text.replace("VIPUser3", "Sky-nom").replace("Vedal", "Veedull")
    return text

def _get_speaker(lang):                                   
    if lang == "fr":
        return "AgathaAI"
    else:
        return "AgathaAI_ENG"                                          

def xtts_get_initial_cache_position(self, cur_len, device, model_kwargs):
    self._xtts_first_stream_step = True
    return model_kwargs

def xtts_prepare_inputs_for_generation(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    **kwargs,
):
    is_first_stream_step = getattr(self, "_xtts_first_stream_step", False)
    if is_first_stream_step:
        self._xtts_first_stream_step = False

    if past_key_values is not None and not is_first_stream_step:
        input_ids = input_ids[:, -1:]
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, -1:]

    if attention_mask is not None and position_ids is None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids[:, -input_ids.shape[1]:]

    model_inputs = {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "use_cache": kwargs.get("use_cache"),
        "output_hidden_states": True,
    }

    if attention_mask is not None:
        model_inputs["attention_mask"] = attention_mask
    if token_type_ids is not None:
        model_inputs["token_type_ids"] = token_type_ids
    if position_ids is not None:
        model_inputs["position_ids"] = position_ids

    return model_inputs

def get_tts_sample_rate(default=24000):
    try:
        return int(tts.synthesizer.output_sample_rate)
    except Exception:
        return default

async def get_tts_ws():
    global tts_ws

    async with tts_ws_lock:
        if tts_ws is not None:
            try:
                await tts_ws.ping()
                return tts_ws
            except Exception:
                try:
                    await tts_ws.close()
                except Exception:
                    pass
                tts_ws = None

        tts_ws = await connect(WS_URI)

        await tts_ws.send(json.dumps({
            "type": "ws.identify",
            "role": "tts",
        }))

        logger.info("(TTS-DEBUG) persistent WS connected")
        return tts_ws

def to_pcm16(wav):
    wav = np.asarray(wav)

    if wav.ndim > 1:
        wav = wav[:, 0]

    if wav.dtype == np.int16:
        return wav

    if np.issubdtype(wav.dtype, np.floating):
        max_abs = float(np.max(np.abs(wav))) if wav.size else 0.0

        logger.info(
            f"TTS wav debug: dtype={wav.dtype}, shape={wav.shape}, "
            f"min={float(np.min(wav)) if wav.size else 0.0}, "
            f"max={float(np.max(wav)) if wav.size else 0.0}, "
            f"max_abs={max_abs}"
        )

        if max_abs > 1.5:
            wav = wav / max_abs

        wav = np.clip(wav, -1.0, 1.0)
        return (wav * 32767).astype(np.int16)

    return wav.astype(np.int16)

async def send_via_ws_file(path):
    websocket = await get_tts_ws()

    await websocket.send(json.dumps({
        "type": "tts_output",
        "payload": {
            "mode": "file",
            "path": path,
        },
    }))

def get_tts_executor():
    global tts_executor

    if tts_executor is None:
        tts_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")

    return tts_executor

def wav_to_pcm16(wav, sample_rate=None, normalize=False):
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().float().cpu().numpy()
    else:
        wav = np.asarray(wav, dtype=np.float32)

    if wav.ndim > 1:
        wav = wav[:, 0]

    wav = np.nan_to_num(wav, nan=0.0, posinf=1.0, neginf=-1.0)
    input_sr = sample_rate or get_tts_sample_rate()

    if input_sr != SAMPLE_RATE_OUT:
        gcd = math.gcd(SAMPLE_RATE_OUT, input_sr)
        up = SAMPLE_RATE_OUT // gcd
        down = input_sr // gcd
        wav = resample_poly(wav, up=up, down=down)

    peak = np.max(np.abs(wav)) if wav.size else 0.0
    if normalize and peak > 0:
        wav = 0.98 * (wav / peak)

    pcm16 = np.clip(wav, -1.0, 1.0)
    return (pcm16 * 32767).astype(np.int16)

def wav_debug_stats(wav):
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().float().cpu()
        return (
            tuple(wav.shape),
            float(wav.min().item()) if wav.numel() else 0.0,
            float(wav.max().item()) if wav.numel() else 0.0,
        )

    wav = np.asarray(wav)
    return (
        wav.shape,
        float(np.min(wav)) if wav.size else 0.0,
        float(np.max(wav)) if wav.size else 0.0,
    )

async def send_stream_start(websocket):
    await websocket.send(json.dumps({
        "type": "tts_output",
        "payload": {
            "mode": "stream_start",
            "sample_rate": SAMPLE_RATE_OUT,
            "channels": 1,
            "format": "pcm_s16le",
        },
    }))

async def send_stream_chunk(websocket, frame):
    await websocket.send(json.dumps({
        "type": "tts_output",
        "payload": {
            "mode": "stream_chunk",
            "audio": base64.b64encode(frame.tobytes()).decode("ascii"),
        },
    }))

async def send_stream_end(websocket):
    await websocket.send(json.dumps({
        "type": "tts_output",
        "payload": {
            "mode": "stream_end",
        },
    }))

async def send_pcm16_frames(websocket, pcm16, pad_final=False):
    idx = 0
    total = len(pcm16)

    while total - idx >= SAMPLES_PER_FRAME:
        frame = pcm16[idx:idx + SAMPLES_PER_FRAME]
        idx += SAMPLES_PER_FRAME
        await send_stream_chunk(websocket, frame)

    remaining = pcm16[idx:]

    if pad_final and len(remaining):
        frame = np.pad(remaining, (0, SAMPLES_PER_FRAME - len(remaining)))
        await send_stream_chunk(websocket, frame)
        return np.empty(0, dtype=np.int16)

    return remaining

async def send_via_ws_audio(wav, sample_rate=None):
    input_sr = sample_rate or get_tts_sample_rate()
    shape, min_value, max_value = wav_debug_stats(wav)

    logger.info(
        f"TTS raw debug: input_sr={input_sr}, "
        f"shape={shape}, min={min_value}, max={max_value}"
    )

    pcm16 = wav_to_pcm16(wav, input_sr, normalize=True)

    websocket = await get_tts_ws()
    await send_stream_start(websocket)
    await send_pcm16_frames(websocket, pcm16, pad_final=True)
    await send_stream_end(websocket)

def cache_tts_voice():
    global tts_voice

    model = tts.synthesizer.tts_model
    voice = model.clone_voice(
        speaker_wav=None,
        speaker_id=VOICE_ID,
        voice_dir=tts.synthesizer.voice_dir,
    )

    tts_voice = {
        "gpt_conditioning_latents": voice["gpt_conditioning_latents"].to(model.device),
        "speaker_embedding": voice["speaker_embedding"].to(model.device),
    }

    logger.info(f"\nTTS voice cached: {VOICE_ID}")

def patch_xtts_streaming_compat():
    try:
        gpt_inference = tts.synthesizer.tts_model.gpt.gpt_inference
        gpt_inference._get_initial_cache_position = MethodType(
            xtts_get_initial_cache_position,
            gpt_inference,
        )
        gpt_inference.prepare_inputs_for_generation = MethodType(
            xtts_prepare_inputs_for_generation,
            gpt_inference,
        )
        logger.info("\nXTTS streaming compatibility patch applied")
    except Exception as E:
        logger.exception(f"\nUnable to apply XTTS streaming compatibility patch : {E}")

def build_xtts_stream(text, lang):
    if tts is None:
        raise RuntimeError("TTS is not loaded")

    if tts_voice is None:
        cache_tts_voice()

    model = tts.synthesizer.tts_model
    config = model.config

    return model.inference_stream(
        text=text,
        language=lang,
        gpt_cond_latent=tts_voice["gpt_conditioning_latents"],
        speaker_embedding=tts_voice["speaker_embedding"],
        stream_chunk_size=XTTS_STREAM_CHUNK_SIZE,
        overlap_wav_len=XTTS_STREAM_OVERLAP_WAV_LEN,
        temperature=config.temperature,
        length_penalty=config.length_penalty,
        repetition_penalty=config.repetition_penalty,
        top_k=config.top_k,
        top_p=config.top_p,
        enable_text_splitting=False,
    )

def next_xtts_chunk(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return STREAM_DONE

async def get_next_xtts_chunk(iterator):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(get_tts_executor(), next_xtts_chunk, iterator)

async def load_tts():
    global tts
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"\nTTS will be loaded on : {device}")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    patch_xtts_streaming_compat()
    cache_tts_voice()
    cp_runtime.set_module_active("tts", True)
    return tts
    
async def unload_tts():
    global tts, tts_voice, tts_executor
    
    logger.info("\nStopping TTS...")
    
    try:
        tts = None
        tts_voice = None
        if tts_executor is not None:
            tts_executor.shutdown(wait=False, cancel_futures=True)
            tts_executor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        cp_runtime.set_module_active("tts", False)
    except Exception as E:
        logger.exception(f"\nFailed to unload TTS : {E}")
        
async def get_input(input, streaming, lang):
    if cp_runtime.tts_disabled() or cp_runtime.stop_audio_requested():
        return

    text = _prepare_tts_text(input, lang)
    if not text:
        return

    await audio_queue.put({
        "text": text,
        "streaming": streaming,
        "lang": lang,
    })

    global worker_task
    if worker_task is None or worker_task.done():
        worker_task = asyncio.create_task(audio_worker())

def non_stream_audio(input, lang):
    if cp_runtime.tts_disabled() or cp_runtime.stop_audio_requested():
        return None

    input = _prepare_tts_text(input, lang)
    if not input:
        return None

    logger.info("\nGenerating audio to file")

    t = datetime.datetime.now()
    path = OUTPUT_DIR / (
        f"agatha_audio_generated-"
        f"Y{t.year}.M{t.month}.D{t.day}-"
        f"{t.hour}h.{t.minute}m.{t.second}s.{t.microsecond}.wav"
    )

    tts.tts_to_file(
        text=input,
        speaker=VOICE_ID,
        language=lang,
        file_path=str(path),
    )

    logger.info(f"\nGenerated file: {path}")
    return str(path)

async def stream_audio(input, lang):
    total_start = time.perf_counter()
    if cp_runtime.tts_disabled() or cp_runtime.stop_audio_requested():
        return

    input = _prepare_tts_text(input, lang)
    if not input:
        return

    websocket = None
    stream_started = False

    try:
        ws_start = time.perf_counter()
        websocket = await get_tts_ws()
        ws_s = time.perf_counter() - ws_start
        start_send_start = time.perf_counter()
        await send_stream_start(websocket)
        stream_started = True
        stream_start_s = time.perf_counter() - start_send_start

        input_sr = get_tts_sample_rate()
        build_start = time.perf_counter()
        stream = build_xtts_stream(input, lang)
        build_s = time.perf_counter() - build_start
        pending = np.empty(0, dtype=np.int16)
        chunk_count = 0
        first_chunk_s = None
        first_frame_s = None

        while True:
            if cp_runtime.tts_disabled() or cp_runtime.stop_audio_requested():
                logger.info("TTS stream stopped by control panel")
                break

            wav_chunk = await get_next_xtts_chunk(stream)
            if wav_chunk is STREAM_DONE:
                break
            if first_chunk_s is None:
                first_chunk_s = time.perf_counter() - total_start

            pcm16 = wav_to_pcm16(wav_chunk, input_sr, normalize=False)

            if len(pending):
                pcm16 = np.concatenate((pending, pcm16))

            pending = await send_pcm16_frames(websocket, pcm16, pad_final=False)
            if first_frame_s is None:
                first_frame_s = time.perf_counter() - total_start
            chunk_count += 1

        if len(pending):
            await send_pcm16_frames(websocket, pending, pad_final=True)
            if first_frame_s is None:
                first_frame_s = time.perf_counter() - total_start

        logger.info(f"TTS stream completed: {chunk_count} XTTS chunks")
        logger.info(
            "\n[PIPELINE:TTS_STREAM] total=%.4fs ws=%.4fs stream_start=%.4fs "
            "build=%.4fs first_chunk=%s first_frame=%s chunks=%s text_chars=%s lang=%s",
            time.perf_counter() - total_start,
            ws_s,
            stream_start_s,
            build_s,
            "null" if first_chunk_s is None else f"{first_chunk_s:.4f}s",
            "null" if first_frame_s is None else f"{first_frame_s:.4f}s",
            chunk_count,
            len(input),
            lang,
        )
    except Exception as E:
        logger.exception(f"\nError while trying to send audio via WS : {E}")
    finally:
        if stream_started:
            try:
                await send_stream_end(websocket)
            except Exception as E:
                logger.exception(f"\nError while trying to end TTS stream via WS : {E}")

async def audio_worker():
    global playing_audio

    playing_audio = True

    try:
        while not audio_queue.empty():
            item = await audio_queue.get()

            text = item["text"]
            streaming = item["streaming"]
            lang = item["lang"]

            try:
                if cp_runtime.tts_disabled() or cp_runtime.stop_audio_requested():
                    continue

                if streaming:
                    logger.info("TTS will stream audio directly")
                    await stream_audio(text, lang)
                else:
                    logger.info("TTS will write audio file then send path")
                    path = await asyncio.to_thread(non_stream_audio, text, lang)
                    if path:
                        await send_via_ws_file(path)

            except Exception as e:
                logger.exception(f"TTS worker error: {e}")

            finally:
                audio_queue.task_done()

    finally:
        playing_audio = False
        if cp_runtime.stop_audio_requested():
            cp_runtime.clear_stop_audio()


async def stop_audio_queue():
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
            audio_queue.task_done()
        except asyncio.QueueEmpty:
            break

    if not playing_audio:
        cp_runtime.clear_stop_audio()
