from __future__ import annotations

import asyncio, logging, pyvts, platform, os, subprocess, re, random, pygame
from pathlib import Path
from typing import Any

from src.config.config import settings
from src.vtuber.setup_params import create_all_tracking_parameters

logger = logging.getLogger("app")
pygame.mixer.init()

host = None
plugin = None
active_emote = None

_hold_task: asyncio.Task | None = None
_HOLD_INTERVAL = 0.1     
    
def get_windows_host() -> str:
    """
    Return Windows host IP reachable from WSL.
    """

    try:
        output = subprocess.check_output(
            ["ipconfig.exe"],
            text=True,
            encoding="utf-8",
            errors="ignore",
        )

        matches = re.findall(
            r"IPv4[^\:]*:\s*([0-9.]+)",
            output
        )

        for ip in matches:
            if ip.startswith("172."):
                return ip

        for ip in matches:
            if ip.startswith("private-lan."):
                return ip

    except Exception:
        pass

    return "local.example"

if platform.system().lower() == "windows" or "WSL_DISTRO_NAME" in os.environ:
    host = get_windows_host()
else:
    host = "example.local"                                                                               

class AgathaVTSPlugin:
    def __init__(
        self,
        *,
        vtubing: bool = True,
        plugin_name: str = "AgathaAI Plugin",
        developer: str = "Axel",
        token_path: str | Path = "./vts_token.txt",
        host: str = host,
        port: int = settings.VTS_PORT,
        retry_delay: float = 1.0,
        create_parameters: bool = True,
    ) -> None:
        self.vtubing = vtubing
        self.authenticated = False
        self.create_parameters = create_parameters
        self.retry_delay = retry_delay
        self.token_path = Path(token_path)

        self._lock = asyncio.Lock()
        self._connect_task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        self.vts = pyvts.vts(
            plugin_info={
                "plugin_name": plugin_name,
                "developer": developer,
                "authentication_token_path": str(token_path),
            },
            vts_api_info={
                "name": "VTubeStudioPublicAPI",
                "version": "1.0",
                "host": host,
                "port": port,
            },
        )
        logger.info(f"[VTUBER] Connecting to VTube Studio at ws://{host}:{port}")

    async def start(self) -> "AgathaVTSPlugin":
        if not self.vtubing:
            return self

        self._loop = asyncio.get_running_loop()

        if self._connect_task is None or self._connect_task.done():
            self._connect_task = asyncio.create_task(self._connection_loop())

        return self

    async def _connection_loop(self) -> None:
        while self.vtubing:
            if self.authenticated:
                await asyncio.sleep(self.retry_delay)
                continue

            try:
                await self.vts.connect()
                
                try:
                    await self.vts.request_authenticate()
                except Exception:
                    logger.warning("[VTUBER] Auth failed, regenerating token...")

                    if self.token_path.exists():
                        self.token_path.unlink()
                
                    await self.vts.request_authenticate_token()
                    await self.vts.request_authenticate()

                self.authenticated = True
                logger.info("[VTUBER] Authentication to VTube Studio successful.")

                if self.create_parameters:
                    await create_all_tracking_parameters(self.vts)

                logger.info("[VTUBER] VTube Studio plugin ready.")

            except Exception as e:
                self.authenticated = False
                                                                                          

                try:
                    await self.vts.close()
                except Exception:
                    pass

                await asyncio.sleep(self.retry_delay)

    async def close(self) -> None:
        self.vtubing = False
        self.authenticated = False
        await _stop_parameter_hold_async()

        if self._connect_task:
            self._connect_task.cancel()
            try:
                await self._connect_task
            except asyncio.CancelledError:
                pass
            self._connect_task = None

        async with self._lock:
            try:
                await self.vts.close()
            except Exception:
                pass

    async def update_parameter(                   
        self,
        parameter_name: str,
        value: float,
        *,
        weight: float | None = None,
        mode: str = "set",
    ) -> dict[str, Any] | None:
        if not self.vtubing or not self.authenticated:
            return None

        parameter_value: dict[str, Any] = {
            "id": parameter_name,
            "value": float(value),
        }

        if weight is not None:
            parameter_value["weight"] = float(weight)

        request = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": f"update_param_{parameter_name}",
            "messageType": "InjectParameterDataRequest",
            "data": {
                "mode": mode,
                "parameterValues": [parameter_value],
            },
        }

        async with self._lock:
            try:
                return await self.vts.request(request)
            except Exception as e:
                logger.error(f"[VTUBER] Parameter update failed: {e}")
                self.authenticated = False
                return None

    async def update_parameters(                      
        self,
        values: dict[str, float],
        *,
        weight: float | None = None,
        mode: str = "set",
    ) -> dict[str, Any] | None:
        if not self.vtubing or not self.authenticated:
            return None

        parameter_values: list[dict[str, Any]] = []

        for name, value in values.items():
            item: dict[str, Any] = {
                "id": name,
                "value": float(value),
            }

            if weight is not None:
                item["weight"] = float(weight)

            parameter_values.append(item)

        request = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "update_multiple_params",
            "messageType": "InjectParameterDataRequest",
            "data": {
                "mode": mode,
                "parameterValues": parameter_values,
            },
        }

        async with self._lock:
            try:
                return await self.vts.request(request)
            except Exception as e:
                logger.error(f"[VTUBER] Parameters update failed: {e}")
                self.authenticated = False
                return None

    async def trigger_hotkey(self, hotkey_id: str) -> dict[str, Any] | None:
        if not self.vtubing or not self.authenticated:
            return None

        async with self._lock:
            try:
                request = self.vts.vts_request.requestTriggerHotKey(hotkey_id)
                return await self.vts.request(request)
            except Exception as e:
                logger.error(f"[VTUBER] Hotkey trigger failed: {e}")
                self.authenticated = False
                return None


def _stop_parameter_hold() -> None:
    global _hold_task

    if _hold_task is not None and not _hold_task.done():
        _hold_task.cancel()

    _hold_task = None


async def _stop_parameter_hold_async() -> None:
    global _hold_task

    task = _hold_task
    _hold_task = None

    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def _start_parameter_hold(values: dict[str, float], timer: int | None = None) -> None:
    global _hold_task

    _stop_parameter_hold()
    _hold_task = asyncio.create_task(
        _parameter_hold_loop(dict(values), timer)
    )


async def _parameter_hold_loop(
    values: dict[str, float],
    timer: int | None = None,
) -> None:
    loop = asyncio.get_running_loop()
    end_time = None if timer is None else loop.time() + (timer / 1000)

    try:
        while True:
            now = loop.time()

            if end_time is not None and now >= end_time:
                break

            current_plugin = plugin
            if current_plugin is not None and values:
                await current_plugin.update_parameters(values)

            sleep_time = _HOLD_INTERVAL

            if end_time is not None:
                sleep_time = min(sleep_time, max(0.0, end_time - loop.time()))

            await asyncio.sleep(sleep_time)

    except asyncio.CancelledError:
        pass
    

async def move_by_sentence(input):
    current_plugin = plugin
    target_loop = getattr(current_plugin, "_loop", None)

    if target_loop is not None and target_loop.is_running():
        current_loop = asyncio.get_running_loop()
        if current_loop is not target_loop:
            future = asyncio.run_coroutine_threadsafe(
                _move_by_sentence_on_vts_loop(input),
                target_loop,
            )
            return await asyncio.wrap_future(future)

    return await _move_by_sentence_on_vts_loop(input)


async def _move_by_sentence_on_vts_loop(input):
    input = str(input).lower()
    
            
    if "scary face" in input or "scary_face" in input:
        logger.debug("[VTS] SCARY_FACE DETECTED")
        _emote_scary_face()
    elif "smile" in input:
        logger.debug("[VTS] SMILE EMOTE DETECTED")
        await _emote_smile()
    elif "frown" in input:
        logger.debug("[VTS] FROWN EMOTE DETECTED")
        await _emote_frown()
    else:
        logger.debug("[VTS] NO EMOTE DETECTED, PARAMETER HOLD STOPPED")
        _stop_parameter_hold()
    
                 
    if active_emote is None or active_emote == "scary_face":
        if "?" in input:
            logger.debug("[VTS] QUESTION MARK DETECTED")
            await _question_in_sentence()
        elif "!" in input:
            logger.debug("[VTS] EXCLAMATION MARK DETECTED")
            await _exclamation_in_sentence()
        elif "." in input:
            logger.debug("[VTS] SIMPLE DOT DETECTED")
            _stop_parameter_hold()
    
    """
    Also add sounds when Body X and AngleK (where K = x, y, z) moves at 90% or more of max param value
    """
    
           
    if active_emote is None:
        if re.search(r"\b(?:oui|yes)\b", input, re.IGNORECASE):
            logger.debug("[VTS] \"YES\" WORD DETECTED")
            await _yes_in_sentence()
        elif re.search(r"\b(?:non|no)\b", input, re.IGNORECASE):
            logger.debug("[VTS] \"NO\" WORD DETECTED")
            await _no_in_sentence()
        elif "regarde" in input or "look" in input:
            logger.debug("[VTS] \"LOOK\" WORD DETECTED")
            await _look_in_sentence()
        elif "cherche" in input or "search" in input:
            logger.debug("[VTS] \"SEARCH\" WORD DETECTED")
            await _search_in_sentence()
        elif "attaque" in input or "attack" in input:
            logger.debug("[VTS] \"ATTACK\" WORD DETECTED")
            await _attack_in_sentence()
        elif "détrui" in input or "destroy" in input:
            logger.debug("[VTS] \"DESTROY\" WORD DETECTED")
            await _destroy_in_sentence()
        elif "manipule" in input or "manipulate" in input:
            logger.debug("[VTS] \"MANIPULATE\" WORD DETECTED")
            await _manipulate_in_sentence()
        
async def start_vts_plugin(*, create_parameters: bool = True, test: bool = False,) -> AgathaVTSPlugin:
    global plugin
    if plugin is not None:
        await stop_vts_plugin()

    plugin = AgathaVTSPlugin(create_parameters=create_parameters)
    await plugin.start()
    
    if test:
        await asyncio.sleep(1)
        print("VTS PLUGIN STARTED")
        await test_run()
        
    return plugin


async def stop_vts_plugin() -> None:
    global plugin, active_emote

    current_plugin = plugin
    active_emote = None
    await _stop_parameter_hold_async()

    if current_plugin is None:
        return

    await current_plugin.close()

    if plugin is current_plugin:
        plugin = None


def _emote_scary_face(test_mode=False):
    if plugin is None:
        return
    
    global active_emote
    active_emote = "scary_face"
    
    asyncio.create_task(_emote_scary_face_task(test_mode))
    
    active_emote = None


async def _emote_scary_face_task(test_mode=False):
    current_plugin = plugin
    if current_plugin is None:
        return

    try:
        await current_plugin.update_parameter("ToggleScaryFace", 1.0)
        await asyncio.sleep(0.01)       
        if test_mode:
            values = {"ToggleScaryFace": 1.0,}
            _start_parameter_hold(values, 4999)
            await asyncio.sleep(4.99)
    finally:
        await current_plugin.update_parameter("ToggleScaryFace", 0.0)

async def _emote_smile():
    current_plugin = plugin
    if current_plugin is None:
        return
    
    global active_emote
    active_emote = "smile"
    
    values = {"VoiceFrequencyPlusMouthSmile": 1.0,}
    _start_parameter_hold(values, 5000)
    
    await current_plugin.update_parameter("VoiceFrequencyPlusMouthSmile", 1.0)
    
    active_emote = None

async def _emote_frown():
    current_plugin = plugin
    if current_plugin is None:
        return
    
    global active_emote
    active_emote = "frown"
    
    values = {
        "VoiceFrequencyPlusMouthSmile": -1.0,
        "MoveBrowLY": -1.0,
        "MoveBrowRY": -1.0,
        "FormBrowLY": -1.0,
        "FormBrowRY": -1.0,
    }
    _start_parameter_hold(values)
    
    await current_plugin.update_parameters({
        "VoiceFrequencyPlusMouthSmile": -1.0, 
        "MoveBrowLY": -1.0,
        "MoveBrowRY": -1.0,
        "FormBrowLY": -1.0,
        "FormBrowRY": -1.0,
    })
    
    active_emote = None

async def _question_in_sentence():
    current_plugin = plugin
    if current_plugin is None:
        return
    
    brow = f"FormBrow{random.choice(['R', 'L'])}Y"
    angle_z = random.choice([-30.0, 30.0])
    
    values = {
        brow: 1.0,
        "MoveAngleZ": angle_z,
    }
    _start_parameter_hold(values, 2500)
    
    await current_plugin.update_parameters({brow: 1.0, "MoveAngleZ": angle_z})
    
async def _exclamation_in_sentence():
    current_plugin = plugin
    if current_plugin is None:
        return
    
    values = {
        "FormBrowRY": -1.0,
        "FormBrowLY": -1.0,
    }
    _start_parameter_hold(values, 2500)
    
    await current_plugin.update_parameters({
        "FormBrowRY": -1.0,
        "FormBrowLY": -1.0,
    })

async def _yes_in_sentence():
    current_plugin = plugin
    if current_plugin is None:
        return
    
    x = random.choice([1, -1])
    
                     
    for i in range(3):
        await current_plugin.update_parameter("MoveAngleY", -7.5 * x)
        await asyncio.sleep(0.3)
        await current_plugin.update_parameter("MoveAngleY", 7.5 * x)
        await asyncio.sleep(0.3)

async def _no_in_sentence():
    current_plugin = plugin
    if current_plugin is None:
        return
    
    x = random.choice([1, -1])
    
                       
    for i in range(3):
        await current_plugin.update_parameter("MoveAngleX", -15 * x)
        await asyncio.sleep(0.3)
        await current_plugin.update_parameter("MoveAngleX", 15 * x)
        await asyncio.sleep(0.3)
        
async def _look_in_sentence():
    current_plugin = plugin
    if current_plugin is None:
        return
    
    if random.randint(1, 10) == 10:
        sound = get_random_doll_sound(0.2)
        play_audio(sound["path"], volume=sound["volume"])
    
    x = random.choice([1, -1])
    
    values = {
        "MoveAngleX": -30.0 * x,
        "MoveAngleY": -7.0 * x,
        "MoveAngleZ": -2.0 * x,
        "MoveEyeBallsY": 0.1,
        "MoveEyeBallsX": 1.0 * x,
        "MoveBodyX": -10.0 * x,
    }
    _start_parameter_hold(values, 1500)
    
    await current_plugin.update_parameters({
        "MoveAngleX": -30.0 * x,
        "MoveAngleY": -7.0 * x,
        "MoveAngleZ": -2.0 * x,
        "MoveEyeBallsY": 0.1,
        "MoveEyeBallsX": -1.0 * x,
        "MoveBodyX": -10.0 * x,
    })

async def _search_in_sentence():
    current_plugin = plugin
    if current_plugin is None:
        return
    
    x = random.choice([1, -1])
    
    values = {
        "MoveAngleX": -5.0 * x,
        "MoveAngleY": -6.0,
        "MoveEyeBallsY": -0.25,
        "MoveEyeBallsX": -0.15 * x,
        "MoveBodyX": -2.0 * x,
    }
    _start_parameter_hold(values, 4000)
    
    await current_plugin.update_parameters({
        "MoveAngleX": -4.5 * x,
        "MoveAngleY": -4.0,
        "MoveEyeBallsY": -0.1,
        "MoveEyeBallsX": -0.15 * x,
        "MoveBodyX": -2.0 * x,
    })
    
async def _attack_in_sentence():
    current_plugin = plugin
    if current_plugin is None:
        return
    
    values = {
        "MoveAngleY": -6.0,
        "VoiceFrequencyPlusMouthSmile": 1.0,
    }
    _start_parameter_hold(values, 2000)
    
    await current_plugin.update_parameters({
        "MoveAngleY": -6.0,
        "VoiceFrequencyPlusMouthSmile": 1.0,
    })

async def _destroy_in_sentence():
    current_plugin = plugin
    if current_plugin is None:
        return
    
    values = {
        "MoveAngleY": -6.0,
        "VoiceFrequencyPlusMouthSmile": 1.0,
        "VoiceVolumePlusMouthOpen": 0.3,
    }
    _start_parameter_hold(values, 2250)
    
    await current_plugin.update_parameters({
        "MoveAngleY": -6.0,
        "VoiceFrequencyPlusMouthSmile": 1.0,
        "VoiceVolumePlusMouthOpen": 0.3,
    })

async def _manipulate_in_sentence():
    current_plugin = plugin
    if current_plugin is None:
        return
    
    x = random.choice([1, -1])
    
    values = {
        "MoveAngleY": -15.0,
        "MoveAngleZ": -6.0 * x,
        "MoveEyeBallsY": 0.4,
        "VoiceFrequencyPlusMouthSmile": 1.0,
    }
    _start_parameter_hold(values, 3000)
    
    await current_plugin.update_parameters({
        "MoveAngleY": -15.0,
        "MoveAngleZ": -6.0 * x,
        "MoveEyeBallsY": 0.4,
        "VoiceFrequencyPlusMouthSmile": 1.0,
    })

def get_random_doll_sound(volume: float = 1.0) -> dict:
    
    base_path = Path("src/vtuber/res")

    return {
        "path": base_path / random.choice([
            f"doll{i}.mp3"
            for i in range(1, 6)
        ]),
        "volume": volume
    }


def play_audio(path, volume=1.0):
    sound = pygame.mixer.Sound(str(path))
    sound.set_volume(volume)
    sound.play()

async def test_run():
    print("TEST RUN BEGIN")
    await asyncio.sleep(5)
    _emote_scary_face(True)
    print("SCARY FACE EMOTE TESTED")
    await asyncio.sleep(10)
    await _emote_frown()
    print("FROWN EMOTE TESTED")
    await asyncio.sleep(10)
    await _emote_smile()
    print("SMILE EMOTE TESTED")
    await asyncio.sleep(10)
    _stop_parameter_hold()
    print("PARAMETER HOLD STOPPED 1/3")
    await asyncio.sleep(5)
    await move_by_sentence("?")
    print("\"?\" IN SENTENCE TESTED")
    await asyncio.sleep(10)
    await move_by_sentence("!")
    print("\"!\" IN SENTENCE TESTED")
    await asyncio.sleep(10)
    _stop_parameter_hold()
    print("PARAMETER HOLD STOPPED 2/3")
    await asyncio.sleep(5)
    await move_by_sentence("yes")
    print("WORD \"YES\" IN SENTENCE TESTED")
    await asyncio.sleep(10)
    await move_by_sentence("no")
    print("WORD \"NO\" IN SENTENCE TESTED")
    await asyncio.sleep(10)
    await move_by_sentence("look")
    print("WORD \"LOOK\" IN SENTENCE TESTED")
    await asyncio.sleep(10)
    await move_by_sentence("search")
    print("WORD \"SEARCH\" IN SENTENCE TESTED")
    await asyncio.sleep(10)
    await move_by_sentence("attack")
    print("WORD \"ATTACK\" IN SENTENCE TESTED")
    await asyncio.sleep(10)
    await move_by_sentence("destroy")
    print("WORD \"DESTROY\" IN SENTENCE TESTED")
    await asyncio.sleep(10)
    await move_by_sentence("manipulate")
    print("WORD \"MANIPULATE\" IN SENTENCE TESTED")
    await asyncio.sleep(10)
    _stop_parameter_hold()
    print("PARAMETER HOLD STOPPED 3/3")
    await asyncio.sleep(1)
    print("TEST RUN COMPLETED, EXITING...")
    exit(0)
