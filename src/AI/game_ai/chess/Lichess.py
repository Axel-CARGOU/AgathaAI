import argparse
import chess, torch, json, threading, atexit, time, os, sys, random, asyncio, re, logging, datetime
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

try:
    import berserk
except ModuleNotFoundError:
    berserk = None

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.AI.game_ai.chess.fight_chess import mcts
from src.AI.game_ai.chess.chess_model import ChessResNet
from src.AI.game_ai.chess.chess_model_v2 import ChessActionValueTransformer, choose_move_v2, load_v2_checkpoint

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(*_args, **_kwargs):
        return False

logger = logging.getLogger("app")

BASE_DIR = SCRIPT_PATH.parents[3]
STATES_PATH = BASE_DIR / "config" / "states.json"
CHESS_DIR = SCRIPT_PATH.parent
DEFAULT_V2_MODEL_PATH = CHESS_DIR / "checkpoints" / "agathaai_chess_ai_v2.2.pth"

CURRENT_GAME_ID: Optional[str] = None
STOP_REQUESTED = threading.Event()

client: Optional[Any] = None
model: Optional[Any] = None
device: Optional[torch.device] = None
BOT_USERNAME: Optional[str] = None
DEFAULT_SIMULATIONS: int = 256
MODEL_KIND: str = "v1"
V2_SEARCH_TIME_LIMIT_SECONDS: float = 10.0

@dataclass
class GameSnapshot:
    game_id: str
    fen: str
    moves_uci: List[str]
    agatha_color: str
    last_agatha_move: Optional[str]
    last_opponent_move: Optional[str]

GAME_SNAPSHOTS: Dict[str, GameSnapshot] = {}

_ENABLE_LLM_CHAT = True
_AUTO_CHAT_MIN_TURNS = 5
_AUTO_CHAT_MAX_TURNS = 10
_CHAT_MAX_LEN = 180
_CHAT_MAX_SENTENCES = 2

_CHAT_EXEC = ThreadPoolExecutor(max_workers=1)

_GAME_OPPONENT = {}
_GAME_NEXT_CHAT_PLY = {}
_GAME_LAST_PLY = {}
_GAME_HANDLED_DRAW_OFFERS = {}

_DRAW_DECISION_TIMEOUT_SECONDS = 20

_SENT_SPLIT = re.compile(r"(?<=[\.\!\?])\s+")


def _resolve_chess_model_path(model_path: str) -> Path:
    path = Path(model_path).expanduser()
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return CHESS_DIR / path


def _is_v2_model_path(model_path: Path) -> bool:
    return "v2" in model_path.stem.lower()


def _configure_standalone_logging() -> None:
    t = datetime.datetime.now()
    log_path = (
        f"LICHESS_STANDALONE-logs-Y{t.year}.M{t.month}.D{t.day}-"
        f"{t.hour}h.{t.minute}m.{t.second}s.log"
    )
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s : %(message)s",
        filename=log_path,
        level=logging.INFO,
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s : %(message)s"))
    logger.addHandler(console_handler)


def _clock_limited_search_seconds(state: Optional[dict], bot_color) -> float:
    max_seconds = float(V2_SEARCH_TIME_LIMIT_SECONDS)
    if max_seconds <= 0:
        return 0.0
    if not isinstance(state, dict):
        return max_seconds

    time_key = "wtime" if bot_color == chess.WHITE else "btime"
    inc_key = "winc" if bot_color == chess.WHITE else "binc"
    remaining_ms = state.get(time_key)
    if not isinstance(remaining_ms, (int, float)):
        return max_seconds

    increment_ms = state.get(inc_key, 0)
    if not isinstance(increment_ms, (int, float)):
        increment_ms = 0

    remaining_s = max(0.0, float(remaining_ms) / 1000.0)
    increment_s = max(0.0, float(increment_ms) / 1000.0)
    if remaining_s <= 0.0:
        return min(max_seconds, 0.1)

    buffer_s = 1.0 if remaining_s >= 15.0 else 0.25
    clock_cap = max(0.1, (remaining_s - buffer_s) * 0.08 + increment_s * 0.5)
    return min(max_seconds, clock_cap)


def _clamp_sentences(text: str, max_sentences: int = _CHAT_MAX_SENTENCES) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    parts = _SENT_SPLIT.split(t)
    t = " ".join(parts[:max_sentences]).strip()
    return t


def _clamp_len(text: str, max_len: int = _CHAT_MAX_LEN) -> str:
    t = (text or "").strip()
    if len(t) <= max_len:
        return t
    t = t[:max_len].rstrip()
    return t

def _detect_if_in_vc():
    with open(STATES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("is_in_vc") is True

def _schedule_next_auto_chat(game_id: str, current_ply: int):
    interval_turns = random.randint(_AUTO_CHAT_MIN_TURNS, _AUTO_CHAT_MAX_TURNS)
    _GAME_NEXT_CHAT_PLY[game_id] = current_ply + interval_turns * 2


def _submit_async(coro):
    if STOP_REQUESTED.is_set():
        try:
            coro.close()
        except Exception:
            pass
        return

    def runner():
        try:
            asyncio.run(coro)
        except Exception as e:
            logger.error(f"[Lichess] async coroutine error: {e}")
    _CHAT_EXEC.submit(runner)


def post_lichess_message(text: str, *, spectator: bool = False, game_id: str | None = None) -> bool:
    global client, CURRENT_GAME_ID

    gid = game_id or CURRENT_GAME_ID
    if not gid or client is None:
        return False

    msg = _clamp_len(_clamp_sentences(text))
    if not msg:
        return False

    try:
        return bool(client.bots.post_message(gid, msg, spectator=spectator))
    except Exception as e:
        logger.error(f"[Lichess] post_message failed: {e}")
        return False

                                                                  
async def _llm_game_start_greeting(opponent_username: str) -> str:
    if not _ENABLE_LLM_CHAT:
        return ""
    if _detect_if_in_vc() is False:
    
        from src.AI.LLM.llm_agathaai_vision_vllm import non_stream_output

        command_text = (
            f"La partie d'Échecs commence contre {opponent_username}. "
            "Salue ton opponnent avec assurance et une pointe d'arrogance. "
            "Ta réponse doit faire au maximum 140 caractères (espaces et caractères spéciaux compris)."
        )
        
        prompt = {
            "user_name": "System",
            "content": command_text,
        }

        response = await non_stream_output(prompt, "lichess")

        return response
    else:
        return


async def _llm_game_win_reaction(opponent_username: str) -> str:
    if not _ENABLE_LLM_CHAT:
        return ""
    if _detect_if_in_vc() is False:
        
        from src.AI.LLM.llm_agathaai_vision_vllm import non_stream_output

        command_text = (
            f"Tu viens de gagner ta partie d'Échecs contre {opponent_username}. "
            "Réagis avec fierté, ironie et de manière légèrement insultante. "
            "Ta réponse doit faire au maximum 140 caractères (espaces et caractères spéciaux compris)."
        )

        prompt = {
            "user_name": "System",
            "content": command_text,
        }
        response = await non_stream_output(prompt, "lichess")

        return response
    else:
        return


async def _llm_game_loss_reaction(opponent_username: str) -> str:
    if not _ENABLE_LLM_CHAT:
        return ""
    if _detect_if_in_vc() is False:
        
        from src.AI.LLM.llm_agathaai_vision_vllm import non_stream_output

        command_text = (
            f"Tu viens de perdre ta partie d'Échecs contre {opponent_username}. "
            "Réagis avec froideur, dédain ou amusement, "
            "Ta réponse doit faire au maximum 140 caractères (espaces et caractères spéciaux compris)."
        )

        prompt = {
            "user_name": "System",
            "content": command_text,
        }

        response = await non_stream_output(prompt, "lichess")

        return response
    else:
        return


async def _llm_reply_to_chat(username: str, user_msg: str) -> str:
    if not _ENABLE_LLM_CHAT:
        return ""
    
    if _detect_if_in_vc() is False:

        from src.AI.LLM.llm_agathaai_vision_vllm import non_stream_output, _remember_rag_interaction

        prompt = {
            "user_name": username,
            "content": user_msg,
        }

        response = await non_stream_output(prompt, "lichess")

        try:
            await _remember_rag_interaction(user_msg, response)
        except Exception:
            pass

        return response
    else:
        return


async def _llm_auto_comment(opponent_username: str) -> str:
    if not _ENABLE_LLM_CHAT:
        return ""
    
    if _detect_if_in_vc() is False:
    
        from src.AI.LLM.llm_agathaai_vision_vllm import non_stream_output

        command_text = (
            f"Tu es en train de jouer aux Échecs contre {opponent_username}. "
            "Commente la partie et moque-toi s'il joue mal ou s'il est en train de perdre.\n"
            "Ta réponse doit faire au maximum 140 caractères (espaces et caractères spéciaux compris).\n\n"
        )

        prompt = {
            "user_name": "System",
            "content": command_text,
        }

        response = await non_stream_output(prompt, "lichess")

        return response
    else:
        return


def _parse_draw_offer_decision(text: str) -> Optional[bool]:
    first_word = re.split(r"[\s,.;:!?]+", (text or "").strip().lower(), maxsplit=1)[0]

    if first_word in ("accepter", "accepte", "accept", "accepted", "yes", "oui"):
        return True
    if first_word in ("refuser", "refuse", "decline", "declined", "reject", "no", "non"):
        return False

    return None


async def _llm_should_accept_draw_offer(opponent_username: str) -> Optional[bool]:
    if not _ENABLE_LLM_CHAT:
        return None
    if _detect_if_in_vc() is True:
        return None

    from src.AI.LLM.llm_agathaai_vision_vllm import non_stream_output

    command_text = (
        f"{opponent_username} te propose un match nul dans ta partie d'Échecs actuelle. "
        "Décide si tu veux accepter cette proposition ou continuer la partie. "
        "Base ta décision sur l'état actuel du plateau donné dans le contexte. "
        "Réponds uniquement par ACCEPTER ou REFUSER, sans autre mot."
    )

    prompt = {
        "user_name": "System",
        "content": command_text,
    }

    response = await non_stream_output(prompt, "lichess")
    return _parse_draw_offer_decision(response)


def _ask_llm_draw_offer_decision(opponent_username: str) -> bool:
    if not _ENABLE_LLM_CHAT:
        logger.info("[Lichess] draw offer declined because LLM chat is disabled")
        return False

    try:
        decision = asyncio.run(
            asyncio.wait_for(
                _llm_should_accept_draw_offer(opponent_username),
                timeout=_DRAW_DECISION_TIMEOUT_SECONDS,
            )
        )
    except asyncio.TimeoutError:
        logger.warning("[Lichess] draw offer decision timed out; declining by default")
        return False
    except Exception as e:
        logger.error(f"[Lichess] draw offer decision failed: {e}")
        return False

    if decision is None:
        logger.warning("[Lichess] draw offer decision was unclear; declining by default")
        return False

    return decision


def _respond_to_draw_offer(game_id: str, accept: bool) -> bool:
    if client is None:
        return False

    accept_value = "yes" if accept else "no"
    try:
        client.bots._r.post(f"/api/bot/game/{game_id}/draw/{accept_value}")
        logger.info(f"[Lichess] draw offer {'accepted' if accept else 'declined'}")
        return True
    except Exception as e:
        logger.error(f"[Lichess] failed to respond to draw offer: {e}")
        return False


async def _handle_incoming_chat_line(game_id: str, room: str, username: str, text: str):
    if STOP_REQUESTED.is_set():
        return
    if not _ENABLE_LLM_CHAT:
        return
    if room == "spectator":
        return

    try:
        if BOT_USERNAME and username and username.lower() == BOT_USERNAME.lower():              
            return
    except Exception:
        pass

    reply = await _llm_reply_to_chat(username, text)
    reply = _clamp_len(_clamp_sentences(reply))
    if reply:
        post_lichess_message(reply, spectator=False, game_id=game_id)


async def _maybe_auto_chat(game_id: str):
    if not _ENABLE_LLM_CHAT:
        return

    ply = int(_GAME_LAST_PLY.get(game_id, 0))
    nxt = int(_GAME_NEXT_CHAT_PLY.get(game_id, 10**9))
    if ply < nxt:
        return

    opponent = _GAME_OPPONENT.get(game_id, "opponnent")
    msg = await _llm_auto_comment(opponent)
    msg = _clamp_len(_clamp_sentences(msg))
    if msg:
        post_lichess_message(msg, spectator=False, game_id=game_id)

    _schedule_next_auto_chat(game_id, ply)

_states_lock = threading.Lock()

def _update_chess_state(is_playing: bool):
    with _states_lock:
        try:
            if STATES_PATH.exists():
                with open(STATES_PATH, "r", encoding="utf-8") as f:
                    states = json.load(f)
            else:
                states = {}

            states["is_playing_chess"] = bool(is_playing)

            with open(STATES_PATH, "w", encoding="utf-8") as f:
                json.dump(states, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to update states.json: {e}")

def _player_name(p) -> Optional[str]:
    if not isinstance(p, dict):
        return None

    for k in ("name", "id", "username"):
        if k in p and isinstance(p[k], str):
            return p[k]

    u = p.get("user")
    if isinstance(u, dict):
        for k in ("name", "id", "username"):
            if k in u and isinstance(u[k], str):
                return u[k]

    return None


def _compute_fen_from_moves(moves_str: str) -> str:
    b = chess.Board()
    if moves_str:
        for u in moves_str.split():
            try:
                b.push_uci(u)
            except Exception:
                break
    return b.fen()


def _board_from_fen_or_moves(fen: str, moves_uci: Optional[List[str]] = None) -> chess.Board:
    if moves_uci:
        board = chess.Board()
        try:
            for move_uci in moves_uci:
                board.push_uci(move_uci)
            return board
        except Exception:
            pass
    return chess.Board(fen)


def _last_moves_from_list(moves: List[str], agatha_is_white: Optional[bool]) -> Tuple[Optional[str], Optional[str]]:
    if not moves or agatha_is_white is None:
        return None, None

    white_moves = moves[0::2]
    black_moves = moves[1::2]

    last_white = white_moves[-1] if white_moves else None
    last_black = black_moves[-1] if black_moves else None

    if agatha_is_white:
        return last_white, last_black
    else:
        return last_black, last_white


def _agatha_is_white_from_gamefull(event_gamefull: dict) -> Optional[bool]:
    if BOT_USERNAME is None:
        return None

    white_name = _player_name(event_gamefull.get("white"))
    black_name = _player_name(event_gamefull.get("black"))

    if white_name and white_name.lower() == BOT_USERNAME.lower():
        return True
    if black_name and black_name.lower() == BOT_USERNAME.lower():
        return False

    return None


def _draw_flag_enabled(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes")
    return value is True or value == 1


def _bot_color_from_state(agatha_is_white: Optional[bool], fallback_color) -> bool:
    if agatha_is_white is None:
        return fallback_color
    return chess.WHITE if agatha_is_white else chess.BLACK


def _opponent_draw_offer_color(state: dict, agatha_is_white: Optional[bool], fallback_color) -> Optional[bool]:
    bot_color = _bot_color_from_state(agatha_is_white, fallback_color)

    if bot_color == chess.WHITE and _draw_flag_enabled(state.get("bdraw")):
        return chess.BLACK
    if bot_color == chess.BLACK and _draw_flag_enabled(state.get("wdraw")):
        return chess.WHITE

    return None


def _maybe_handle_draw_offer(
    game_id: str,
    state: dict,
    moves_str: str,
    agatha_is_white: Optional[bool],
    fallback_color,
) -> bool:
    offer_color = _opponent_draw_offer_color(state, agatha_is_white, fallback_color)
    if offer_color is None:
        _GAME_HANDLED_DRAW_OFFERS.pop(game_id, None)
        return False

    offer_signature = (offer_color, moves_str.strip())
    if _GAME_HANDLED_DRAW_OFFERS.get(game_id) == offer_signature:
        return False

    _GAME_HANDLED_DRAW_OFFERS[game_id] = offer_signature

    opponent = _GAME_OPPONENT.get(game_id, "opponnent")
    logger.info(f"[Lichess] draw offer received from {opponent}")

    accept = _ask_llm_draw_offer_decision(opponent)
    if _respond_to_draw_offer(game_id, accept):
        return accept

    _GAME_HANDLED_DRAW_OFFERS.pop(game_id, None)
    return False


def _update_snapshot(game_id: str, fen: str, moves_str: str, agatha_is_white: Optional[bool]):
    moves = moves_str.split() if moves_str else []
    last_agatha, last_opp = _last_moves_from_list(moves, agatha_is_white)

    if agatha_is_white is None:
        agatha_color = "unknown"
    else:
        agatha_color = "white" if agatha_is_white else "black"

    GAME_SNAPSHOTS[game_id] = GameSnapshot(
        game_id=game_id,
        fen=fen,
        moves_uci=moves,
        agatha_color=agatha_color,
        last_agatha_move=last_agatha,
        last_opponent_move=last_opp,
    )


def get_board_fen(game_id: str) -> Optional[str]:
    s = GAME_SNAPSHOTS.get(game_id)
    return s.fen if s else None


def get_board_ascii(game_id: str) -> Optional[str]:
    s = GAME_SNAPSHOTS.get(game_id)
    if not s:
        return None
    b = chess.Board(s.fen)
    return b.unicode(empty_square="·")


def get_agatha_color(game_id: str) -> Optional[str]:
    s = GAME_SNAPSHOTS.get(game_id)
    return s.agatha_color if s else None


def get_last_moves(game_id: str) -> Tuple[Optional[str], Optional[str]]:
    s = GAME_SNAPSHOTS.get(game_id)
    if not s:
        return None, None
    return s.last_agatha_move, s.last_opponent_move

def get_current_game_snapshot() -> Optional[dict]:
    if CURRENT_GAME_ID is None:
        return None
    return get_game_snapshot(CURRENT_GAME_ID)

def get_game_snapshot(game_id: str) -> Optional[dict]:
    s = GAME_SNAPSHOTS.get(game_id)
    if not s:
        return None

    board = chess.Board(s.fen)

    return {
        "game_id": s.game_id,
        "fen": s.fen,
        "turn": "white" if board.turn == chess.WHITE else "black",
        "board_ascii": board.unicode(empty_square="·"),
        "moves_uci": s.moves_uci,
        "moves_uci_tail": s.moves_uci[-6:],
        "agatha_color": s.agatha_color,
        "last_agatha_move": s.last_agatha_move,
        "last_opponent_move": s.last_opponent_move,
    }

def boot_lichess(
    model_path: str = str(DEFAULT_V2_MODEL_PATH),
    token_env: str = "TOKEN_LICHESS",
    simulations_default: int = 256,
    v2_search_time_limit: float = 10.0,
    enable_llm_chat: Optional[bool] = None,
    dotenv: bool = True,
):
    global client, model, device, BOT_USERNAME, DEFAULT_SIMULATIONS, MODEL_KIND, V2_SEARCH_TIME_LIMIT_SECONDS, _ENABLE_LLM_CHAT

    STOP_REQUESTED.clear()
    V2_SEARCH_TIME_LIMIT_SECONDS = max(0.0, float(v2_search_time_limit))
    if enable_llm_chat is not None:
        _ENABLE_LLM_CHAT = bool(enable_llm_chat)

    if dotenv:
        load_dotenv()

    if berserk is None:
        raise RuntimeError("[CHESS] berserk is not installed; install it or run Lichess.py --test")

    token = os.getenv(token_env)
    if token is None:
        raise ValueError(f"[CHESS] Env not defined")

    session = berserk.TokenSession(token)
    client = berserk.Client(session=session)

    try:
        me = client.account.get()
        BOT_USERNAME = me.get("username") or me.get("id") or me.get("name")
    except Exception:
        BOT_USERNAME = None
        
    logger.info("[CHESS] Successfully connected to Lichess API")

    DEFAULT_SIMULATIONS = int(simulations_default)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved_model_path = _resolve_chess_model_path(model_path)
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"[CHESS] Model not found: {resolved_model_path}")

    if _is_v2_model_path(resolved_model_path):
        model = load_v2_checkpoint(resolved_model_path, device=device)
        MODEL_KIND = "v2"
    else:
        model = ChessResNet().to(device)
        model.load_state_dict(torch.load(resolved_model_path, map_location=device))
        model.eval()
        MODEL_KIND = "v1"

    logger.info(f"Client and {MODEL_KIND} model ready: {resolved_model_path}")
    if MODEL_KIND == "v2":
        logger.info(f"V2 search time limit: {V2_SEARCH_TIME_LIMIT_SECONDS:.2f}s")
    if BOT_USERNAME:
        logger.info(f"Bot username: {BOT_USERNAME}")
    
    _update_chess_state(True)
    atexit.register(_update_chess_state, False)

    return client, model


def request_stop():
    STOP_REQUESTED.set()
    _update_chess_state(False)
    logger.info("Lichess stop requested")

    try:
        session = getattr(client, "session", None)
        close = getattr(session, "close", None)
        if callable(close):
            close()
    except Exception as e:
        logger.error(f"[Lichess] failed to close session: {e}")


def get_bot_color_from_file():
    try:
        with open("botcolor.json", "r") as f:                                        
            data = json.load(f)
            return chess.WHITE if data.get("bot_color") == "white" else chess.BLACK
    except Exception:
        return chess.BLACK


def _write_bot_color_file(is_white: bool):
    try:
        with open("botcolor.json", "w") as f:                                         
            json.dump({"bot_color": "white" if is_white else "black"}, f)
    except Exception:
        pass


def choose_move(
    fen: str,
    simulations: Optional[int] = None,
    time_limit_s: Optional[float] = None,
    moves_uci: Optional[List[str]] = None,
) -> str:
    if model is None:
        raise RuntimeError("Model not initialised")
    sims = DEFAULT_SIMULATIONS if simulations is None else int(simulations)

    board = _board_from_fen_or_moves(fen, moves_uci)
    if MODEL_KIND == "v2":
        search_time = V2_SEARCH_TIME_LIMIT_SECONDS if time_limit_s is None else float(time_limit_s)
        best_move = choose_move_v2(
            board,
            model,
            device=device,
            time_limit_s=search_time if search_time > 0 else None,
        )
    else:
        best_move = mcts(board, model, sims)
    return best_move.uci()


def stream_events():
    try:
        if client is None:
            raise RuntimeError("Client not initialised")

        for event in client.bots.stream_incoming_events():
            if STOP_REQUESTED.is_set():
                logger.info("Lichess stop requested")
                break

            if event["type"] == "challenge":
                challenge = event["challenge"]
                challenge_id = challenge["id"]
                requested_color = challenge.get("color", "random")

                logger.info(f"Challenge received {challenge_id}. Accepting...")
                client.bots.accept_challenge(challenge_id)
                
                if requested_color == "white":
                    _write_bot_color_file(True)
                elif requested_color == "black":
                    _write_bot_color_file(False)

            elif event["type"] == "gameStart":
                game = event["game"]
                game_id = game["id"]
                global CURRENT_GAME_ID
                CURRENT_GAME_ID = game_id
                logger.info(f"New game started : {game_id}")
                play_game(game_id)
    except KeyboardInterrupt:
        logger.info("Lichess interrupted by user")
    except Exception as e:
        logger.error(f"Lichess error: {e}")
    finally:
        _update_chess_state(False)


def play_game(game_id: str, simulations: Optional[int] = None):
    global CURRENT_GAME_ID

    try:
        if client is None:
            raise RuntimeError("Client not initialised.")

        CURRENT_GAME_ID = game_id

        stream = client.bots.stream_game_state(game_id)
        logger.info("Waiting for game state updates...")

        agatha_is_white: Optional[bool] = None
        fallback_color = get_bot_color_from_file()

        for event in stream:
            if STOP_REQUESTED.is_set():
                logger.info("Lichess game stop requested")
                break

            state = None
            
            if event["type"] == "gameFull":
                state = event.get("state", {})
                moves_str = state.get("moves", "")
                fen = state.get("fen") or _compute_fen_from_moves(moves_str)

                agatha_is_white = _agatha_is_white_from_gamefull(event)
                if agatha_is_white is not None:
                    _write_bot_color_file(agatha_is_white)

                white_name = event["white"].get("name") or event["white"].get("id", "")
                black_name = event["black"].get("name") or event["black"].get("id", "")

                opponent = black_name if agatha_is_white else white_name
                _GAME_OPPONENT[game_id] = opponent
                
                if _ENABLE_LLM_CHAT:
                    _submit_async(_llm_game_start_greeting(opponent))

                ply = 0 if not moves_str.strip() else len(moves_str.split())
                _GAME_LAST_PLY[game_id] = ply
                _schedule_next_auto_chat(game_id, ply)

                _update_snapshot(game_id, fen, moves_str, agatha_is_white)

            elif event["type"] == "gameState":
                state = event
                moves_str = event.get("moves", "") or ""
                fen = event.get("fen") or _compute_fen_from_moves(moves_str)

                _GAME_LAST_PLY[game_id] = 0 if not moves_str.strip() else len(moves_str.split())

                _update_snapshot(game_id, fen, moves_str, agatha_is_white)

            elif event["type"] == "chatLine":
                room = event.get("room", "player")
                username = event.get("username", "")
                text = event.get("text", "")

                if username and text:
                    _submit_async(
                        _handle_incoming_chat_line(
                            game_id,
                            room,
                            username,
                            text
                        )
                    )
                continue

            else:
                continue

            board = chess.Board(GAME_SNAPSHOTS[game_id].fen)

            if state and _maybe_handle_draw_offer(game_id, state, moves_str, agatha_is_white, fallback_color):
                CURRENT_GAME_ID = None
                break

            if board.is_game_over(claim_draw=True):
                outcome = board.outcome(claim_draw=True)
                CURRENT_GAME_ID = None
                logger.info("Game finished")

                if outcome and outcome.winner is not None:
                    if (outcome.winner == chess.WHITE and agatha_is_white) or\
                       (outcome.winner == chess.BLACK and not agatha_is_white):
                        if _ENABLE_LLM_CHAT:
                            _submit_async(_llm_game_win_reaction(_GAME_OPPONENT.get(game_id, "opponnent")))
                    else:
                        if _ENABLE_LLM_CHAT:
                            _submit_async(_llm_game_loss_reaction(_GAME_OPPONENT.get(game_id, "opponnent")))

                break


            if agatha_is_white is None:
                bot_color = fallback_color
            else:
                bot_color = chess.WHITE if agatha_is_white else chess.BLACK

            if board.turn != bot_color:
                continue

            move_time_limit = _clock_limited_search_seconds(state, bot_color)
            best_move = choose_move(
                board.fen(),
                simulations=simulations,
                time_limit_s=move_time_limit,
                moves_uci=moves_str.split() if moves_str else None,
            )

            try:
                time.sleep(random.uniform(0.4, 1.2))
                client.bots.make_move(game_id, best_move)
            except Exception as e:
                logger.error("Error while making move :", e)
                continue

            _GAME_LAST_PLY[game_id] = int(_GAME_LAST_PLY.get(game_id, 0)) + 1
            _submit_async(_maybe_auto_chat(game_id))

    except KeyboardInterrupt:
        logger.info("Lichess interrupted by user")
    except Exception as e:
        logger.error(f"Error while interrupting Lichess : {e}")
    finally:
        CURRENT_GAME_ID = None
        _GAME_HANDLED_DRAW_OFFERS.pop(game_id, None)
        _update_chess_state(False)


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AgathaAI Lichess chess bot or a local chess-model test.")
    parser.add_argument("-t", "--test", action="store_true", help="local model-only test; no Lichess API and no LLM chat")
    parser.add_argument("--model", default=str(DEFAULT_V2_MODEL_PATH), help="checkpoint path; defaults to v2.2")
    parser.add_argument("--fen", default=chess.STARTING_FEN, help="FEN used by --test")
    parser.add_argument("--device", default="", help="cuda, cpu, or empty for auto")
    parser.add_argument("--time-limit", type=float, default=10.0, help="max V2 search seconds per move")
    parser.add_argument("--simulations", type=int, default=256, help="V1 MCTS simulations")
    parser.add_argument("--token-env", default="TOKEN_LICHESS", help="Lichess token environment variable")
    parser.add_argument("--no-llm-chat", action="store_true", help="disable LLM chat/reactions in live Lichess mode")
    parser.add_argument("--no-search", action="store_true", help="only for --test: direct V2 policy without search")
    return parser.parse_args()


def _load_local_model(model_path: str, selected_device: torch.device) -> tuple[Any, str, Path]:
    resolved_model_path = _resolve_chess_model_path(model_path)
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"[CHESS] Model not found: {resolved_model_path}")

    if _is_v2_model_path(resolved_model_path):
        return load_v2_checkpoint(resolved_model_path, device=selected_device), "v2", resolved_model_path

    loaded = ChessResNet().to(selected_device)
    loaded.load_state_dict(torch.load(resolved_model_path, map_location=selected_device))
    loaded.eval()
    return loaded, "v1", resolved_model_path


def _run_local_test(args: argparse.Namespace) -> None:
    selected_device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    loaded_model, model_kind, resolved_model_path = _load_local_model(args.model, selected_device)

    board = chess.Board(args.fen)
    started = time.perf_counter()
    if model_kind == "v2":
        move = choose_move_v2(
            board,
            loaded_model,
            device=selected_device,
            search=not args.no_search,
            time_limit_s=args.time_limit if not args.no_search else None,
        )
    else:
        move = mcts(board, loaded_model, simulations=args.simulations)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    print(f"model={model_kind} path={resolved_model_path}")
    print(f"device={selected_device} search={'off' if args.no_search else 'on'} time_limit={args.time_limit:.2f}s")
    print(f"fen={board.fen()}")
    print(f"move={move.uci()} legal={move in board.legal_moves} elapsed_ms={elapsed_ms:.1f}")
    board.push(move)
    print(f"after={board.fen()}")


if __name__ == "__main__":
    args = _parse_cli_args()
    _configure_standalone_logging()

    if args.test:
        _run_local_test(args)
    else:
        boot_lichess(
            model_path=args.model,
            token_env=args.token_env,
            simulations_default=args.simulations,
            v2_search_time_limit=args.time_limit,
            enable_llm_chat=not args.no_llm_chat,
        )
        logger.info("\nLichess starting in stand-alone mode...\n")
        stream_events()
