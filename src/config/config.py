                    
import json
import threading
from pathlib import Path
from datetime import datetime
from pydantic_settings import BaseSettings
from typing import Any

          
config_path = Path(__file__).resolve().parent / "settings.json"
_SETTINGS_CACHE_LOCK = threading.RLock()
_SETTINGS_CACHE: "Settings | None" = None
_SETTINGS_CACHE_SIGNATURE: tuple[int, int] | None = None

class Settings(BaseSettings):
    HOST: str = "bind.example"
    PORT: int = 8258

                       
    DISCORD_KEY: str | None = None
    OPENAI_KEY: str | None = None
    GOOGLE_APPLICATION_CREDENTIALS: str | None = None
    ASSEMBLYAI_KEY: str | None = None
    ELEVENLABS_KEY: str | None = None
    VOICEMOD_KEY: str | None = None
    TWITCH_ID: str | None = None
    TWITCH_SECRET: str | None = None
    CONTROL_PANEL_ID: str | None = None
    AGATHA_DISCORD_ID: str | None = None
    OPENAI_AGATHA_MODEL: str | None = None
    OWNER_ID: str | None = None

         
    LLM_MODEL_PATH: str = "agathaai_vision_v7_7b_qwen2.5-vl_gptq_int4"
    LLM_MAX_NEW_TOKENS: int = 512
    LLM_TEMPERATURE: float = 0.7
    LLM_TOP_P: float = 1.0
    LLM_MIN_P: float = 0.1
    LLM_MAX_SEQ_LEN: int = 10240
    LLM_REPETITION_PENALTY: float = 1.2
    LLM_PRESENCE_PENALTY: float = 0.3
    KV_Q4: bool = True
    LLM_STARTUP_TIMEOUT: int = 900
    LLM_REQUEST_TIMEOUT: int = 300
    LLM_GPU_MAX_USE: float = 0.9
    EDIT_FILE: bool = False
    FILES_SEARCH: bool = False
    WEB_SEARCH: bool = False
    TWITCH_CHAT: bool = False
    TWITCH_POLL: bool = False
    TWITCH_BAN: bool = False
    TWITCH_TIMEOUT: bool = False

                  
    RAG_ENABLED: bool = True
    RAG_QDRANT_PATH: str = "qdrant_data"
    RAG_COLLECTION_NAME: str = "agathaai_long_term_memory"
    RAG_SHORT_TERM_LIMIT: int = 10
    RAG_LONG_TERM_CONTEXT_LIMIT: int = 5
    RAG_EMBEDDING_MODEL: str = "intfloat/multilingual-e5-small"
    RAG_HASH_EMBEDDING_SIZE: int = 384
    RAG_MAX_CONTEXT_CHARS: int = 4000
    RAG_MAX_ENTRY_CHARS: int = 650
    RAG_MAX_STORED_CHARS: int = 3000
    
         
    VOICE_SAMPLE: str = "agatha.wav"
    
         
    SAMPLE_RATE: int = 48000
    STT_MODEL: str = "Google STT"

                      
    LANGUAGE: str = "fr-FR"
    
             
    LANGUAGE: str = "fr"
    DEV_MODE: bool = False
    VTUBING: bool = True
    VTS_PORT: int = 7801

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra="ignore"


def load_settings() -> Settings:
    settings = Settings()

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        llm_conf = data.get("LLM", {})
        rag_conf = data.get("RAG", {})
        tts_conf = data.get("TTS", {})
        stt_conf = data.get("STT", {})
        general_conf = data.get("GENERAL", {})

             
        settings.LLM_MODEL_PATH = llm_conf.get("MODEL_PATH", settings.LLM_MODEL_PATH)
        settings.LLM_MAX_NEW_TOKENS = llm_conf.get("MAX_TOKENS", settings.LLM_MAX_NEW_TOKENS)
        settings.LLM_TEMPERATURE = llm_conf.get("TEMPERATURE", settings.LLM_TEMPERATURE)
        settings.LLM_TOP_P = llm_conf.get("TOP_P", settings.LLM_TOP_P)
        settings.LLM_MIN_P = llm_conf.get("MIN_P", settings.LLM_MIN_P)
        settings.LLM_MAX_SEQ_LEN = llm_conf.get("MAX_SEQ_LEN", settings.LLM_MAX_SEQ_LEN)
        settings.LLM_REPETITION_PENALTY = llm_conf.get("REPETITION_PENALTY", settings.LLM_REPETITION_PENALTY)
        settings.LLM_PRESENCE_PENALTY = llm_conf.get("PRESENCE_PENALTY", settings.LLM_PRESENCE_PENALTY)
        settings.KV_Q4 = llm_conf.get("KV_Q4", settings.KV_Q4)
        settings.LLM_STARTUP_TIMEOUT = llm_conf.get("VLLM_STARTUP_TIMEOUT", settings.LLM_STARTUP_TIMEOUT)
        settings.LLM_REQUEST_TIMEOUT = llm_conf.get("VLLM_REQUEST_TIMEOUT", settings.LLM_REQUEST_TIMEOUT)
        settings.LLM_GPU_MAX_USE = llm_conf.get("VLLM_GPU_ALLOWED_USE", settings.LLM_GPU_MAX_USE)
        settings.EDIT_FILE = llm_conf.get("EDIT_FILE", settings.EDIT_FILE)
        settings.FILES_SEARCH = llm_conf.get("FILES_SEARCH", settings.FILES_SEARCH)
        settings.WEB_SEARCH = llm_conf.get("WEB_SEARCH", settings.WEB_SEARCH)
        settings.TWITCH_CHAT = llm_conf.get("TWITCH_CHAT", settings.TWITCH_CHAT)
        settings.TWITCH_POLL = llm_conf.get("TWITCH_POLL", settings.TWITCH_POLL)
        settings.TWITCH_BAN = llm_conf.get("TWITCH_BAN", settings.TWITCH_BAN)
        settings.TWITCH_TIMEOUT = llm_conf.get("TWITCH_TIMEOUT", settings.TWITCH_TIMEOUT)

             
        settings.RAG_ENABLED = rag_conf.get("ENABLED", settings.RAG_ENABLED)
        settings.RAG_QDRANT_PATH = rag_conf.get("QDRANT_PATH", settings.RAG_QDRANT_PATH)
        settings.RAG_COLLECTION_NAME = rag_conf.get("COLLECTION_NAME", settings.RAG_COLLECTION_NAME)
        settings.RAG_SHORT_TERM_LIMIT = rag_conf.get("SHORT_TERM_LIMIT", settings.RAG_SHORT_TERM_LIMIT)
        settings.RAG_LONG_TERM_CONTEXT_LIMIT = rag_conf.get("LONG_TERM_CONTEXT_LIMIT", settings.RAG_LONG_TERM_CONTEXT_LIMIT)
        settings.RAG_EMBEDDING_MODEL = rag_conf.get("EMBEDDING_MODEL", settings.RAG_EMBEDDING_MODEL)
        settings.RAG_HASH_EMBEDDING_SIZE = rag_conf.get("HASH_EMBEDDING_SIZE", settings.RAG_HASH_EMBEDDING_SIZE)
        settings.RAG_MAX_CONTEXT_CHARS = rag_conf.get("MAX_CONTEXT_CHARS", settings.RAG_MAX_CONTEXT_CHARS)
        settings.RAG_MAX_ENTRY_CHARS = rag_conf.get("MAX_ENTRY_CHARS", settings.RAG_MAX_ENTRY_CHARS)
        settings.RAG_MAX_STORED_CHARS = rag_conf.get("MAX_STORED_CHARS", settings.RAG_MAX_STORED_CHARS)

             
        voice_name = tts_conf.get("VOICE_SAMPLE", settings.VOICE_SAMPLE)
        voices_dir = Path.home() / "AgathaAI/app/services/voices"
        settings.VOICE_SAMPLE = str((voices_dir / voice_name).resolve())
        
             
        settings.SAMPLE_RATE = stt_conf.get("SAMPLE_RATE", settings.SAMPLE_RATE)
        settings.STT_MODEL = stt_conf.get("MODEL", settings.STT_MODEL)
        
                 
        settings.LANGUAGE = general_conf.get("LANGUAGE", settings.LANGUAGE)
        settings.DEV_MODE = general_conf.get("DEV_MODE", settings.DEV_MODE)
        settings.VTUBING = general_conf.get("VTUBING", settings.VTUBING)
        settings.VTS_PORT = general_conf.get("VTS_PORT", settings.VTS_PORT)

    else: 
        print("[CONFIG] settings.json not found, using default settings ...")
        
    return settings


def _settings_file_signature() -> tuple[int, int] | None:
    try:
        stat = config_path.stat()
        return stat.st_size, stat.st_mtime_ns
    except OSError:
        return None


def get_cached_settings() -> Settings:
    global _SETTINGS_CACHE, _SETTINGS_CACHE_SIGNATURE

    signature = _settings_file_signature()
    with _SETTINGS_CACHE_LOCK:
        if _SETTINGS_CACHE is not None and _SETTINGS_CACHE_SIGNATURE == signature:
            return _SETTINGS_CACHE

        fresh_settings = load_settings()
        _SETTINGS_CACHE = fresh_settings
        _SETTINGS_CACHE_SIGNATURE = signature
        return fresh_settings


def invalidate_settings_cache() -> None:
    global _SETTINGS_CACHE, _SETTINGS_CACHE_SIGNATURE

    with _SETTINGS_CACHE_LOCK:
        _SETTINGS_CACHE = None
        _SETTINGS_CACHE_SIGNATURE = None


class LiveSettings:
    def __getattr__(self, name: str):
        fresh_settings = get_cached_settings()
        return getattr(fresh_settings, name)

    def __setattr__(self, name: str, value):
        raise AttributeError(
            "settings est en lecture dynamique. "
            "Utilise cfg_edit(name, value) pour modifier settings.json."
        )

      

OTHERS_PATH = Path(__file__).resolve().parent / "others.json"

def save_shutdown_time():
    try:
        with open(OTHERS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}

    data["last_shutdown"] = datetime.now().isoformat(timespec="seconds")

    with open(OTHERS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_last_shutdown():
    try:
        with open(OTHERS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("last_shutdown")
    except Exception:
        return None
    
    
_SETTINGS_JSON_KEYS: dict[str, tuple[str, str]] = {
    "LLM_MODEL_PATH": ("LLM", "MODEL_PATH"),
    "LLM_MAX_NEW_TOKENS": ("LLM", "MAX_TOKENS"),
    "LLM_TEMPERATURE": ("LLM", "TEMPERATURE"),
    "LLM_TOP_P": ("LLM", "TOP_P"),
    "LLM_MIN_P": ("LLM", "MIN_P"),
    "LLM_MAX_SEQ_LEN": ("LLM", "MAX_SEQ_LEN"),
    "LLM_REPETITION_PENALTY": ("LLM", "REPETITION_PENALTY"),
    "LLM_PRESENCE_PENALTY": ("LLM", "PRESENCE_PENALTY"),
    "KV_Q4": ("LLM", "KV_Q4"),
    "LLM_STARTUP_TIMEOUT": ("LLM", "VLLM_STARTUP_TIMEOUT"),
    "LLM_REQUEST_TIMEOUT": ("LLM", "VLLM_REQUEST_TIMEOUT"),
    "LLM_GPU_MAX_USE": ("LLM", "VLLM_GPU_ALLOWED_USE"),
    "EDIT_FILE": ("LLM", "EDIT_FILE"),
    "FILES_SEARCH": ("LLM", "FILES_SEARCH"),
    "WEB_SEARCH": ("LLM", "WEB_SEARCH"),
    "TWITCH_CHAT": ("LLM", "TWITCH_CHAT"),
    "TWITCH_POLL": ("LLM", "TWITCH_POLL"),
    "TWITCH_BAN": ("LLM", "TWITCH_BAN"),
    "TWITCH_TIMEOUT": ("LLM", "TWITCH_TIMEOUT"),

    "RAG_ENABLED": ("RAG", "ENABLED"),
    "RAG_QDRANT_PATH": ("RAG", "QDRANT_PATH"),
    "RAG_COLLECTION_NAME": ("RAG", "COLLECTION_NAME"),
    "RAG_SHORT_TERM_LIMIT": ("RAG", "SHORT_TERM_LIMIT"),
    "RAG_LONG_TERM_CONTEXT_LIMIT": ("RAG", "LONG_TERM_CONTEXT_LIMIT"),
    "RAG_EMBEDDING_MODEL": ("RAG", "EMBEDDING_MODEL"),
    "RAG_HASH_EMBEDDING_SIZE": ("RAG", "HASH_EMBEDDING_SIZE"),
    "RAG_MAX_CONTEXT_CHARS": ("RAG", "MAX_CONTEXT_CHARS"),
    "RAG_MAX_ENTRY_CHARS": ("RAG", "MAX_ENTRY_CHARS"),
    "RAG_MAX_STORED_CHARS": ("RAG", "MAX_STORED_CHARS"),

    "VOICE_SAMPLE": ("TTS", "VOICE_SAMPLE"),
    "SAMPLE_RATE": ("STT", "SAMPLE_RATE"),
    "STT_MODEL": ("STT", "MODEL"),

    "LANGUAGE": ("GENERAL", "LANGUAGE"),
    "DEV_MODE": ("GENERAL", "DEV_MODE"),
    "VTUBING": ("GENERAL", "VTUBING"),
    "VTS_PORT": ("GENERAL", "VTS_PORT"),
}


def cfg_edit(name: str, value: Any) -> Any:
    if not isinstance(name, str):
        raise TypeError(
            "cfg_edit attend le nom du setting en string, exemple: "
            "cfg_edit('DEV_MODE', True)"
        )

    name = name.upper()

    if name not in _SETTINGS_JSON_KEYS:
        raise KeyError(f"Setting inconnu: {name!r}")

    section, json_key = _SETTINGS_JSON_KEYS[name]

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"settings.json est invalide: {exc}") from exc

    if not isinstance(data.get(section), dict):
        data[section] = {}

    if data[section].get(json_key) == value:
        return value

    data[section][json_key] = value

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    invalidate_settings_cache()

    print(f"[CONFIG] {name} updated to {value!r}")
    return value
    
settings = LiveSettings()
