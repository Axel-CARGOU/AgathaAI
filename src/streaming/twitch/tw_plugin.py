from __future__ import annotations

import asyncio, json, logging, random, secrets, subprocess, sys, time, webbrowser

from collections import deque
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import aiohttp
import twitchio
from twitchio import eventsub
from twitchio.authentication import OAuth, Scopes
from twitchAPI.twitch import Twitch
from twitchAPI.type import AuthScope, TwitchAPIException

from src.config.config import settings
from src.control_panel.moderation import (
    load_twitch_moderation_model,
    moderate_twitch_message,
)


logger = logging.getLogger("app")

PLUGIN_DIR = Path(__file__).resolve().parent
TOKEN_FILE = PLUGIN_DIR / "twitch_tokens.json"
CALLBACK_FILE = PLUGIN_DIR / "twitch_oauth_callback.json"
TWITCH_REDIRECT_URI = "http://local.example:8080"

TWITCH_API_SCOPES = [
    AuthScope.USER_READ_CHAT,
    AuthScope.CHANNEL_MANAGE_POLLS,
    AuthScope.MODERATOR_MANAGE_BANNED_USERS,
]
REQUIRED_SCOPE_NAMES = {scope.value for scope in TWITCH_API_SCOPES}

plugin: AgathaTwitchPlugin | None = None

known_bots = {
        "nightbot",
        "streamelements",
        "moobot",
        "fossabot",
        "wizebot",
        "sery_bot",
        "commanderroot",
}


def _object_to_dict(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()

    if hasattr(value, "__dict__"):
        return dict(value.__dict__)

    return value


def _twitch_action_error(action: str, exc: Exception, **context: Any) -> dict[str, Any]:
    logger.warning(f"[TWITCH] {action} failed without stopping the caller: {exc}")
    return {
        "ok": False,
        "provider": "twitch",
        "action": action,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "context": context,
    }


class AgathaTwitchIOClient(twitchio.Client):
    def __init__(self, owner: AgathaTwitchPlugin, *, bot_id: str) -> None:
        super().__init__(
            client_id=owner.client_id,
            client_secret=owner.client_secret,
            bot_id=bot_id,
            fetch_client_user=False,
        )
        self.owner = owner

    async def setup_hook(self) -> None:
        payload = eventsub.ChatMessageSubscription(
            broadcaster_user_id=self.owner.broadcaster_id,
            user_id=self.owner.user_id,
        )
        await self.subscribe_websocket(payload, token_for=self.owner.user_id)
        logger.info("[TWITCH] Chat EventSub websocket subscription ready.")

        if settings.TWITCH_POLL:
            poll_end_payload = eventsub.ChannelPollEndSubscription(
                broadcaster_user_id=self.owner.broadcaster_id,
            )
            await self.subscribe_websocket(poll_end_payload, token_for=self.owner.user_id)
            logger.info("[TWITCH] Poll end EventSub websocket subscription ready.")

    async def event_message(self, payload: twitchio.ChatMessage) -> None:
        self.owner.store_chat_message(payload)

    async def event_poll_end(self, payload: twitchio.ChannelPollEnd) -> None:
        self.owner.store_poll_result(payload)

    async def event_token_refreshed(self, payload: twitchio.TokenRefreshedPayload) -> None:
        await self.owner.handle_twitchio_token_refresh(payload)


class AgathaTwitchPlugin:
    def __init__(
        self,
        *,
        redirect_uri: str = TWITCH_REDIRECT_URI,
        chat_buffer_size: int = 100,
        retry_delay: float = 5.0,
    ) -> None:
        self.client_id = settings.TWITCH_ID
        self.client_secret = settings.TWITCH_SECRET
        self.redirect_uri = redirect_uri
        self.retry_delay = retry_delay
        self.enabled = bool(self.client_id and self.client_secret)

        self.user_id: str | None = None
        self.user_login: str | None = None
        self.broadcaster_id: str | None = None
        self.moderator_id: str | None = None
        self.token_data: dict[str, Any] | None = None
        self.ready = False

        self.twitch_api: Twitch | None = None
        self.twitchio_client: AgathaTwitchIOClient | None = None
        self._chat_messages: deque[dict[str, Any]] = deque(maxlen=chat_buffer_size)
        self._poll_results: deque[dict[str, Any]] = deque(maxlen=20)
        self._runner_task: asyncio.Task | None = None
        self._twitchio_task: asyncio.Task | None = None
        self._closing = False

        self.oauth = OAuth(
            client_id=self.client_id or "",
            client_secret=self.client_secret or "",
            redirect_uri=self.redirect_uri,
            scopes=self._twitchio_scopes(),
        )

        try:
            load_twitch_moderation_model()
        except Exception as exc:
            logger.warning(f"[MODERATION] OpenAI Twitch moderator unavailable: {exc}")

    def _twitchio_scopes(self) -> Scopes:
        return Scopes(
            user_read_chat=True,
            channel_manage_polls=True,
            moderator_manage_banned_users=True,
        )

    async def start(self) -> AgathaTwitchPlugin:
        if not self.enabled:
            logger.warning("[TWITCH] TWITCH_ID or TWITCH_SECRET missing; Twitch plugin disabled.")
            return self

        if self._runner_task is None or self._runner_task.done():
            self._closing = False
            self._runner_task = asyncio.create_task(self._run())

        return self

    async def _run(self) -> None:
        while not self._closing:
            try:
                self.ready = False
                self.token_data = await self._load_or_authorize_token()
                self._apply_token_identity(self.token_data)
                await self._start_twitch_api(self.token_data)
                await self._start_chat_client(self.token_data)
                self.ready = True

                while not self._closing and self._twitchio_task and not self._twitchio_task.done():
                    await asyncio.sleep(30)

                if self._closing:
                    break

                if self._twitchio_task and self._twitchio_task.done():
                    error = self._twitchio_task.exception()
                    if error:
                        raise error

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.exception(f"[TWITCH] Plugin loop failed, retrying in {self.retry_delay}s: {exc}")
                self.ready = False
                await self._stop_clients()
                await asyncio.sleep(self.retry_delay)

    async def _load_or_authorize_token(self) -> dict[str, Any]:
        token_data = self._read_token_file()

        if token_data:
            refreshed = await self._validate_or_refresh_token(token_data)
            if refreshed:
                logger.info("[TWITCH] OAuth token loaded.")
                return refreshed

        return await self._authorize_with_callback()

    def _read_token_file(self) -> dict[str, Any] | None:
        try:
            with TOKEN_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return None
        except Exception as exc:
            logger.warning(f"[TWITCH] Unable to read token file: {exc}")
            return None

        if not isinstance(data, dict):
            return None

        if not data.get("access_token") or not data.get("refresh_token"):
            return None

        return data

    async def _validate_or_refresh_token(self, token_data: dict[str, Any]) -> dict[str, Any] | None:
        try:
            validated = await self.oauth.validate_token(token_data["access_token"])
        except Exception:
            validated = None

        if validated is not None and self._has_required_scopes(validated.scopes):
            token_data.update(
                {
                    "user_id": validated.user_id,
                    "user_login": validated.login,
                    "scopes": validated.scopes,
                    "expires_at": time.time() + validated.expires_in,
                }
            )
            self._save_token_data(token_data)
            return token_data

        if token_data.get("refresh_token"):
            try:
                return await self._refresh_token_data(token_data["refresh_token"])
            except Exception as exc:
                logger.warning(f"[TWITCH] Stored token refresh failed, OAuth is required: {exc}")

        return None

    async def _refresh_token_data(self, refresh_token: str) -> dict[str, Any]:
        refreshed = await self.oauth.refresh_token(refresh_token)
        validated = await self.oauth.validate_token(refreshed.access_token)

        if not self._has_required_scopes(validated.scopes):
            raise RuntimeError("Twitch token is missing required scopes.")

        token_data = {
            "access_token": refreshed.access_token,
            "refresh_token": refreshed.refresh_token,
            "expires_at": time.time() + refreshed.expires_in,
            "scopes": validated.scopes,
            "user_id": validated.user_id,
            "user_login": validated.login,
            "redirect_uri": self.redirect_uri,
        }
        self._save_token_data(token_data)
        return token_data

    async def _authorize_with_callback(self) -> dict[str, Any]:
        state = secrets.token_urlsafe(32)
        self._delete_callback_file()
        await self._wait_for_callback_server()

        auth_payload = self.oauth.get_authorization_url(
            scopes=self._twitchio_scopes(),
            state=state,
            redirect_uri=self.redirect_uri,
        )

        logger.info(f"[TWITCH] OAuth required. Open this URL if your browser did not: {auth_payload.url}")

        try:
            webbrowser.open(auth_payload.url)
        except Exception as exc:
            logger.warning(f"[TWITCH] Unable to open browser automatically: {exc}")

        while not self._closing:
            callback = self._read_callback_file()

            if callback is None:
                await asyncio.sleep(1.0)
                continue

            if callback.get("state") != state:
                logger.warning("[TWITCH] Ignoring OAuth callback with invalid state.")
                self._delete_callback_file()
                await asyncio.sleep(1.0)
                continue

            if callback.get("error"):
                error = callback.get("error_description") or callback["error"]
                self._delete_callback_file()
                raise RuntimeError(f"Twitch OAuth failed: {error}")

            code = callback.get("code")
            if not code:
                self._delete_callback_file()
                raise RuntimeError("Twitch OAuth callback did not include a code.")

            user_token = await self.oauth.user_access_token(code, redirect_uri=self.redirect_uri)
            validated = await self.oauth.validate_token(user_token.access_token)

            if not self._has_required_scopes(validated.scopes):
                self._delete_callback_file()
                raise RuntimeError("Twitch OAuth token is missing required scopes.")

            token_data = {
                "access_token": user_token.access_token,
                "refresh_token": user_token.refresh_token,
                "expires_at": time.time() + user_token.expires_in,
                "scopes": validated.scopes,
                "user_id": validated.user_id,
                "user_login": validated.login,
                "redirect_uri": self.redirect_uri,
            }
            self._save_token_data(token_data)
            self._delete_callback_file()
            logger.info(f"[TWITCH] OAuth complete for {validated.login or validated.user_id}.")
            return token_data

        raise RuntimeError("Twitch OAuth stopped before completion.")

    async def _wait_for_callback_server(self, timeout: float = 10.0) -> None:
        parts = urlsplit(self.redirect_uri)
        health_url = urlunsplit((parts.scheme, parts.netloc, "/health", "", ""))
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline and not self._closing:
            try:
                client_timeout = aiohttp.ClientTimeout(total=1)
                async with aiohttp.ClientSession(timeout=client_timeout) as session:
                    async with session.get(health_url) as response:
                        if response.status < 500:
                            return
            except Exception:
                await asyncio.sleep(0.25)

        logger.warning(f"[TWITCH] OAuth callback server did not answer at {health_url}; continuing anyway.")

    def _has_required_scopes(self, scopes: list[str]) -> bool:
        return REQUIRED_SCOPE_NAMES.issubset(set(scopes))

    def _read_callback_file(self) -> dict[str, Any] | None:
        try:
            with CALLBACK_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return None
        except Exception as exc:
            logger.warning(f"[TWITCH] Unable to read OAuth callback file: {exc}")
            return None

        return data if isinstance(data, dict) else None

    def _delete_callback_file(self) -> None:
        try:
            CALLBACK_FILE.unlink()
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning(f"[TWITCH] Unable to delete OAuth callback file: {exc}")

    def _save_token_data(self, token_data: dict[str, Any]) -> None:
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = TOKEN_FILE.with_suffix(".tmp")

        with tmp_file.open("w", encoding="utf-8") as f:
            json.dump(token_data, f, indent=2)

        tmp_file.replace(TOKEN_FILE)
        self.token_data = dict(token_data)

    def _apply_token_identity(self, token_data: dict[str, Any]) -> None:
        self.user_id = str(token_data["user_id"])
        self.user_login = token_data.get("user_login")
        self.broadcaster_id = self.user_id
        self.moderator_id = self.user_id

    async def _start_twitch_api(self, token_data: dict[str, Any]) -> None:
        if self.twitch_api is not None:
            await self.twitch_api.close()

        twitch = await Twitch(self.client_id, self.client_secret)
        twitch.user_auth_refresh_callback = self._handle_twitchapi_token_refresh
        await twitch.set_user_authentication(
            token_data["access_token"],
            TWITCH_API_SCOPES,
            token_data["refresh_token"],
        )
        self.twitch_api = twitch
        logger.info("[TWITCH] TwitchAPI client ready.")

    async def _start_chat_client(self, token_data: dict[str, Any]) -> None:
        await self._stop_twitchio_client()

        if not self.user_id or not self.broadcaster_id:
            raise RuntimeError("Twitch user identity is unavailable.")

        client = AgathaTwitchIOClient(self, bot_id=self.user_id)
        await client.add_token(token_data["access_token"], token_data["refresh_token"])
        self.twitchio_client = client
        self._twitchio_task = asyncio.create_task(
            client.start(with_adapter=False, load_tokens=False, save_tokens=False)
        )

        ready_task = asyncio.create_task(client.wait_until_ready())
        done, pending = await asyncio.wait(
            {ready_task, self._twitchio_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if self._twitchio_task in done:
            error = self._twitchio_task.exception()
            ready_task.cancel()
            if error:
                raise error
            raise RuntimeError("TwitchIO client stopped before becoming ready.")

        for task in pending:
            if task is not self._twitchio_task:
                task.cancel()

        logger.info(f"[TWITCH] TwitchIO chat client ready for channel {self.user_login or self.user_id}.")

    async def handle_twitchio_token_refresh(self, payload: twitchio.TokenRefreshedPayload) -> None:
        token_data = dict(self.token_data or {})
        token_data.update(
            {
                "access_token": payload.token,
                "refresh_token": payload.refresh_token,
                "expires_at": time.time() + payload.expires_in,
                "scopes": list(payload.scopes),
                "user_id": payload.user_id,
                "user_login": self.user_login,
                "redirect_uri": self.redirect_uri,
            }
        )
        self._save_token_data(token_data)

        if self.twitch_api is not None:
            await self.twitch_api.set_user_authentication(
                payload.token,
                TWITCH_API_SCOPES,
                payload.refresh_token,
                validate=False,
            )

        logger.info("[TWITCH] TwitchIO token refreshed.")

    async def _handle_twitchapi_token_refresh(self, token: str, refresh_token: str) -> None:
        token_data = dict(self.token_data or {})
        token_data.update(
            {
                "access_token": token,
                "refresh_token": refresh_token,
                "scopes": list(REQUIRED_SCOPE_NAMES),
                "user_id": self.user_id,
                "user_login": self.user_login,
                "redirect_uri": self.redirect_uri,
            }
        )
        self._save_token_data(token_data)

        if self.twitchio_client is not None:
            await self.twitchio_client.add_token(token, refresh_token)

        logger.info("[TWITCH] TwitchAPI token refreshed.")

    def store_chat_message(self, message: twitchio.ChatMessage) -> None:
        chatter = message.chatter
        broadcaster = message.broadcaster
        
                                                                                                          
        if len(message.text) < 5 or message.text.startswith("!") or message.chatter.name.lower() in known_bots:
            logger.debug(f"[TWITCH] Ignoring chat message from {chatter.display_name} due to quality filters: {message.text}")
            return

        self._chat_messages.append(
            {
                "message_id": message.id,
                "content": message.text,
                "user_id": chatter.id,
                "user_login": chatter.name,
                "user_name": chatter.display_name,
                "broadcaster_id": broadcaster.id,
                "broadcaster_login": broadcaster.name,
                "broadcaster_name": broadcaster.display_name,
                "message_type": message.type,
                "received_at": time.time(),
                "raw": repr(message),
            }
        )
        logger.debug(f"[TWITCH] Chat message stored from {chatter.display_name}: {message.text}")

    def store_poll_result(self, poll: twitchio.ChannelPollEnd) -> None:
        choices = [
            {
                "id": choice.id,
                "title": choice.title,
                "votes": int(choice.votes or 0),
                "channel_points_votes": int(choice.channel_points_votes or 0),
            }
            for choice in poll.choices
        ]
        winner = max(choices, key=lambda choice: choice["votes"], default=None)
        result = {
            "poll_id": poll.id,
            "title": poll.title,
            "status": poll.status,
            "choices": choices,
            "winner": winner,
            "started_at": poll.started_at.isoformat() if poll.started_at else None,
            "ended_at": poll.ended_at.isoformat() if poll.ended_at else None,
            "received_at": time.time(),
            "raw": repr(poll),
        }
        self._poll_results.append(result)
        logger.info(
            "[TWITCH] Poll result stored: %s winner=%s",
            poll.title,
            winner["title"] if winner else "none",
        )

    def _take_chat_message(self, *, consume: bool = True) -> dict[str, Any] | None:
        if not self._chat_messages:
            return None

        index = random.randrange(len(self._chat_messages))
        message = self._chat_messages[index]

        if consume:
            del self._chat_messages[index]

        return dict(message)

    async def read_chat_message(self, *, consume: bool = True) -> dict[str, Any] | None:
        message = self._take_chat_message(consume=consume)
        if message is None:
            return None

        try:
            decision = await moderate_twitch_message(
                message.get("content"),
                viewer_id=message.get("user_id"),
            )
        except Exception as exc:
            logger.exception(
                "[MODERATION] OpenAI moderation failed for %s; message blocked without sanction: %s",
                message.get("user_name") or message.get("user_login") or "unknown viewer",
                exc,
            )
            return None

        message["moderation"] = decision
        if decision["result"] == "OK":
            return message

        if decision["ban"] == "yes":
            moderation_action = "BAN"
        elif isinstance(decision["timeout"], int):
            moderation_action = f"TIMEOUT duration={decision['timeout']}s"
        else:
            moderation_action = "NO SANCTION"

        logger.warning(
            "[MODERATION] Twitch message blocked from %s: action=%s reason=%s",
            message.get("user_name") or message.get("user_login") or "unknown viewer",
            moderation_action,
            decision["reason"],
        )
        if consume:
            await self._apply_moderation_sanction(message, decision)

        return None

    async def read_chat_msg(self, *, consume: bool = True) -> dict[str, Any] | None:
        return await self.read_chat_message(consume=consume)

    async def read_poll_result(self, *, consume: bool = True) -> dict[str, Any] | None:
        if not self._poll_results:
            return None

        result = self._poll_results[0]
        if consume:
            self._poll_results.popleft()

        return dict(result)

    async def _apply_moderation_sanction(
        self,
        message: dict[str, Any],
        decision: dict[str, str | int],
    ) -> None:
        user = str(message.get("user_id") or message.get("user_login") or "").strip()
        if not user:
            logger.warning("[MODERATION] Unable to sanction viewer without a Twitch user id/login.")
            return

        if self.broadcaster_id and user == self.broadcaster_id:
            logger.warning("[MODERATION] Refusing to sanction the broadcaster account.")
            return

        duration: int | None
        action: str
        if decision["ban"] == "yes":
            duration = None
            action = "permanent ban"
        elif isinstance(decision["timeout"], int):
            duration = decision["timeout"]
            action = f"{duration}s timeout"
        else:
            logger.warning("[MODERATION] NOT OK decision did not contain a sanction.")
            return

        try:
            await self.ban_user(
                user,
                reason=str(decision["reason"])[:500],
                duration=duration,
            )
        except Exception as exc:
            logger.exception(
                "[MODERATION] Failed to apply %s to %s; message remains blocked: %s",
                action,
                message.get("user_name") or message.get("user_login") or user,
                exc,
            )
            return

        logger.warning(
            "[MODERATION] Applied %s to %s: %s",
            action,
            message.get("user_name") or message.get("user_login") or user,
            decision["reason"],
        )

    async def create_poll(
        self,
        title: str,
        choices: list[str],
        *,
        duration: int = 60,
        channel_points_voting_enabled: bool = False,
        channel_points_per_vote: int | None = None,
        broadcaster_id: str | None = None,
    ) -> Any:
        self._require_api_ready()
        target_broadcaster_id = broadcaster_id or self.broadcaster_id

        if not target_broadcaster_id:
            raise RuntimeError("No Twitch broadcaster id available for poll creation.")

        if len(choices) < 2 or len(choices) > 5:
            raise ValueError("Twitch polls require between 2 and 5 choices.")

        poll = await self.twitch_api.create_poll(
            target_broadcaster_id,
            title,
            choices,
            duration,
            channel_points_voting_enabled,
            channel_points_per_vote,
        )
        return _object_to_dict(poll)

    async def ban_user(
        self,
        user: str,
        *,
        reason: str = "Banned by AgathaAI.",
        duration: int | None = None,
        broadcaster_id: str | None = None,
        moderator_id: str | None = None,
    ) -> Any:
        self._require_api_ready()
        target_broadcaster_id = broadcaster_id or self.broadcaster_id
        target_moderator_id = moderator_id or self.moderator_id

        if not target_broadcaster_id or not target_moderator_id:
            raise RuntimeError("No Twitch broadcaster/moderator id available for ban.")

        target_user_id = await self.resolve_user_id(user)
        response = await self.twitch_api.ban_user(
            target_broadcaster_id,
            target_moderator_id,
            target_user_id,
            reason,
            duration,
        )
        return _object_to_dict(response)

    async def resolve_user_id(self, user: str) -> str:
        self._require_api_ready()
        cleaned = user.strip().lstrip("@")

        if cleaned.isdigit():
            return cleaned

        async for twitch_user in self.twitch_api.get_users(logins=[cleaned.lower()]):
            return twitch_user.id

        raise ValueError(f"Twitch user not found: {user}")

    def _require_api_ready(self) -> None:
        if self.twitch_api is None:
            raise RuntimeError("Twitch plugin is not ready yet.")

    async def close(self) -> None:
        self._closing = True

        if self._runner_task:
            self._runner_task.cancel()

        await self._stop_clients()
        await self.oauth.close()

    async def _stop_clients(self) -> None:
        self.ready = False
        await self._stop_twitchio_client()

        if self.twitch_api is not None:
            try:
                await self.twitch_api.close()
            except Exception:
                pass

        self.twitch_api = None

    async def _stop_twitchio_client(self) -> None:
        if self.twitchio_client is not None:
            try:
                await self.twitchio_client.close(save_tokens=False)
            except Exception:
                pass

        if self._twitchio_task is not None:
            self._twitchio_task.cancel()
            try:
                await self._twitchio_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        self.twitchio_client = None
        self._twitchio_task = None


async def start_twitch_plugin() -> AgathaTwitchPlugin:
    global plugin
    plugin = AgathaTwitchPlugin()
    await plugin.start()
    return plugin


async def stop_twitch_plugin() -> None:
    global plugin

    if plugin is None:
        return

    await plugin.close()
    plugin = None


async def _read_chat_msg(*, consume: bool = True) -> dict[str, Any] | None:
    if plugin is None:
        return None

    return await plugin.read_chat_message(consume=consume)


async def _read_poll_result(*, consume: bool = True) -> dict[str, Any] | None:
    if plugin is None:
        return None

    return await plugin.read_poll_result(consume=consume)


async def _create_poll(
    title: str,
    choices: list[str],
    *,
    duration: int = 60,
    channel_points_voting_enabled: bool = False,
    channel_points_per_vote: int | None = None,
    broadcaster_id: str | None = None,
) -> Any:
    if plugin is None:
        return _twitch_action_error("create_poll", RuntimeError("Twitch plugin is not started."))

    try:
        return await plugin.create_poll(
            title,
            choices,
            duration=duration,
            channel_points_voting_enabled=channel_points_voting_enabled,
            channel_points_per_vote=channel_points_per_vote,
            broadcaster_id=broadcaster_id,
        )
    except (TwitchAPIException, ValueError) as exc:
        return _twitch_action_error(
            "create_poll",
            exc,
            title=title,
            choices=choices,
            duration=duration,
            broadcaster_id=broadcaster_id,
        )


async def _ban_user(
    user: str,
    *,
    reason: str = "Banned by AgathaAI.",
    duration: int | None = None,
    broadcaster_id: str | None = None,
    moderator_id: str | None = None,
) -> Any:
    if plugin is None:
        return _twitch_action_error("ban_user", RuntimeError("Twitch plugin is not started."), user=user)

    try:
        return await plugin.ban_user(
            user,
            reason=reason,
            duration=duration,
            broadcaster_id=broadcaster_id,
            moderator_id=moderator_id,
        )
    except (TwitchAPIException, ValueError) as exc:
        return _twitch_action_error(
            "ban_user",
            exc,
            user=user,
            reason=reason,
            duration=duration,
            broadcaster_id=broadcaster_id,
            moderator_id=moderator_id,
        )


async def _test_twitch() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s : %(message)s")

    cp_frontend = await _ensure_test_callback_server()
    twitch = await start_twitch_plugin()

    try:
        if not twitch.enabled:
            print("Twitch plugin disabled: TWITCH_ID or TWITCH_SECRET is missing in .env.")
            return

        print("Twitch test mode started.")
        print("If OAuth opens, authorize Twitch in the browser. Waiting for the plugin to be ready...")
        await _wait_until_test_ready(twitch)
        print(f"Twitch ready for channel: {twitch.user_login or twitch.user_id}")
        print("Commands: read, peek, poll, ban, timeout, help, quit")

        while True:
            command = (await _ainput("twitch> ")).strip().lower()

            if command in {"quit", "q", "exit"}:
                break

            if command in {"help", "h", "?"}:
                print("read    Read and consume one random buffered chat message.")
                print("peek    Read one random buffered chat message without consuming it.")
                print("poll    Create a Twitch poll.")
                print("ban     Permanently ban a user after confirmation.")
                print("timeout Timeout a user after confirmation.")
                print("quit    Stop the Twitch test mode.")
                continue

            if command == "read":
                _print_chat_message(await _read_chat_msg())
                continue

            if command == "peek":
                _print_chat_message(await _read_chat_msg(consume=False))
                continue

            if command == "poll":
                await _test_create_poll()
                continue

            if command == "ban":
                await _test_ban_user(duration=None)
                continue

            if command == "timeout":
                raw_duration = await _ainput("Duration seconds [600]: ")
                try:
                    duration = int(raw_duration.strip() or "600")
                except ValueError:
                    print("Invalid duration. Please enter a number of seconds.")
                    continue

                await _test_ban_user(duration=duration)
                continue

            if command:
                print("Unknown command. Type 'help' for available commands.")

    finally:
        await stop_twitch_plugin()

        if cp_frontend is not None:
            cp_frontend.terminate()
            try:
                await asyncio.wait_for(cp_frontend.wait(), timeout=5)
            except asyncio.TimeoutError:
                cp_frontend.kill()
                await cp_frontend.wait()


async def _ensure_test_callback_server() -> asyncio.subprocess.Process | None:
    if await _callback_server_healthy():
        return None

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "uvicorn",
        "web.routes:app",
        "--host",
        "local.example",
        "--port",
        "8080",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for _ in range(40):
        if await _callback_server_healthy():
            return proc

        if proc.returncode is not None:
            raise RuntimeError("Twitch OAuth callback server stopped during startup.")

        await asyncio.sleep(0.25)

    raise RuntimeError("Twitch OAuth callback server did not start on http://local.example:8080.")


async def _callback_server_healthy() -> bool:
    try:
        timeout = aiohttp.ClientTimeout(total=1)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get("http://local.example:8080/health") as response:
                return response.status < 500
    except Exception:
        return False

async def _wait_until_test_ready(twitch: AgathaTwitchPlugin) -> None:
    next_notice = time.monotonic()

    while not twitch._closing:
        if twitch.ready:
            return

        if twitch._runner_task is not None and twitch._runner_task.done():
            error = twitch._runner_task.exception()
            if error:
                raise error
            raise RuntimeError("Twitch plugin stopped before becoming ready.")

        now = time.monotonic()
        if now >= next_notice:
            print("Still waiting for Twitch OAuth/chat connection...")
            next_notice = now + 10

        await asyncio.sleep(0.5)

    raise RuntimeError("Twitch plugin was stopped before becoming ready.")


async def _test_create_poll() -> None:
    title = (await _ainput("Poll title: ")).strip()
    raw_choices = await _ainput("Choices, separated by comma or semicolon: ")

    try:
        duration = int((await _ainput("Duration seconds [60]: ")).strip() or "60")
    except ValueError:
        print("Invalid duration. Please enter a number of seconds.")
        return

    choices = [
        choice.strip()
        for separator_chunk in raw_choices.split(";")
        for choice in separator_chunk.split(",")
        if choice.strip()
    ]

    result = await _create_poll(title, choices, duration=duration)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


async def _test_ban_user(*, duration: int | None) -> None:
    user = (await _ainput("User login or id: ")).strip()
    reason = (await _ainput("Reason [Test AgathaAI]: ")).strip() or "Test AgathaAI"

    action = "timeout" if duration is not None else "permanent ban"
    confirmation = await _ainput(f"Type CONFIRM to apply {action} to {user}: ")

    if confirmation.strip() != "CONFIRM":
        print("Cancelled.")
        return

    result = await _ban_user(user, reason=reason, duration=duration)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


async def _ainput(prompt: str) -> str:
    return await asyncio.to_thread(input, prompt)


def _print_chat_message(message: dict[str, Any] | None) -> None:
    if message is None:
        print("No buffered chat message yet.")
        return

    print(json.dumps(message, indent=2, ensure_ascii=False, default=str))
