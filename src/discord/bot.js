import WebSocket from "ws";
import fs from "fs";
import { Client, Events, GatewayIntentBits, Partials } from "discord.js";
import {
  joinVoiceChannel,
  entersState,
  EndBehaviorType,
  VoiceConnectionStatus,
  StreamType,
  AudioPlayerStatus,
  createAudioPlayer,
  createAudioResource,
} from "@discordjs/voice";
import { PassThrough } from "node:stream";
import prism from "prism-media";
import { transcribe, warmup_stt } from "../AI/STT/stt.js";

let ws, connection, subscription, currentTTSstream;
let ws_loop = true;
let ttsCurrentlyPlaying = false;
let botDisabled = false;
let sttMuted = false;
let ttsDisabled = false;
const pendingPromptTimings = new Map();

const discord_token = process.env.DISCORD_KEY;
const audioPlayer = createAudioPlayer();

const VOICE_SAMPLE_RATE = 48000;
const DISCORD_OPUS_FRAME_SIZE = 960;
const LISTEN_END_SILENCE_MS = 200;
const GLOBAL_VOICE_SILENCE_MS = 200;
const DISCORD_SPEAKING_END_DELAY_MS = 100;
const GLOBAL_VOICE_FLUSH_DELAY_MS = Math.max(
  0,
  GLOBAL_VOICE_SILENCE_MS - DISCORD_SPEAKING_END_DELAY_MS,
);

const ttsPlaybackQueue = [];
let voiceConversationTurns = [];
let voiceConversationFlushTimer = null;
let voiceGlobalSilenceReady = false;

audioPlayer.on(AudioPlayerStatus.Playing, () => {
  console.log("[AUDIO] Player is playing");
});

audioPlayer.on(AudioPlayerStatus.Idle, () => {
  console.log("[AUDIO] Player is idle");
});

audioPlayer.on("error", (error) => {
  console.error("[AUDIO] Player error:", error);
});

function enqueueTTSResource(resource) {
  if (ttsDisabled || botDisabled) return;
  ttsPlaybackQueue.push(resource);
  playNextTTSResource();
}

function playNextTTSResource() {
  if (ttsCurrentlyPlaying) return;
  if (audioPlayer.state.status !== AudioPlayerStatus.Idle) return;

  const nextResource = ttsPlaybackQueue.shift();
  if (!nextResource) return;

  ttsCurrentlyPlaying = true;
  audioPlayer.play(nextResource);
}

audioPlayer.on(AudioPlayerStatus.Idle, () => {
  console.log("[AUDIO] Player is idle");
  ttsCurrentlyPlaying = false;
  playNextTTSResource();
});

function getFirstImageUrl(message) {
  const attachment = message.attachments.find((file) => {
    if (file.contentType?.startsWith("image/")) return true;
    return /\.(png|jpe?g|webp|gif)$/i.test(file.name ?? file.url ?? "");
  });

  return attachment?.url ?? null;
}

function sendControlSignal(name, value) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;

  ws.send(
    JSON.stringify({
      type: "control_panel.signal",
      payload: { name, value },
    }),
  );
}

function sendVoiceState(isInVc) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;

  ws.send(
    JSON.stringify({
      type: "discord.voice_state",
      payload: { is_in_vc: Boolean(isInVc) },
    }),
  );
}

function hasActiveVoiceConnection() {
  return connection?.state?.status === VoiceConnectionStatus.Ready;
}

function stopCurrentAudioPlayback() {
  ttsPlaybackQueue.length = 0;
  ttsCurrentlyPlaying = false;

  try {
    audioPlayer.stop(true);
  } catch {
  }

  if (currentTTSstream) {
    currentTTSstream.end();
    currentTTSstream = null;
  }
}

function applyDiscordControl(payload = {}) {
  botDisabled = Boolean(payload.botDisabled);
  sttMuted = Boolean(payload.muted);
  ttsDisabled = Boolean(payload.ttsDisabled);

  if (botDisabled || sttMuted) {
    resetVoiceConversationBuffer();
    sendControlSignal("humanSpeaking", false);
  }

  if (botDisabled || ttsDisabled) {
    stopCurrentAudioPlayback();
  }

  console.log(
    `[CONTROL] botDisabled=${botDisabled}, sttMuted=${sttMuted}, ttsDisabled=${ttsDisabled}`,
  );
}

function getAttachments(message) {
  const mediaExtensionPattern =
    /\.(3gp|aac|aiff|ape|avi|bmp|flac|gif|heic|ico|jpe?g|m4a|mkv|mov|mp3|mp4|mpe?g|ogg|opus|png|tiff?|wav|webm|webp|wmv)$/i;
  const executableExtensionPattern =
    /\.(apk|app|bin|class|com|dll|dmg|dylib|elf|exe|jar|msi|o|obj|scr|so)$/i;

  return message.attachments
    .filter((file) => {
      const contentType = file.contentType ?? "";
      const name = file.name ?? file.url ?? "";

      if (/^(audio|image|video)\//i.test(contentType)) return false;
      if (mediaExtensionPattern.test(name)) return false;
      if (executableExtensionPattern.test(name)) return false;

      return true;
    })
    .map((file) => ({
      id: file.id,
      name: file.name ?? `attachment-${file.id}`,
      url: file.url,
      contentType: file.contentType ?? null,
      size: file.size ?? null,
    }));
}

function clearVoiceConversationFlushTimer() {
  if (!voiceConversationFlushTimer) return;

  clearTimeout(voiceConversationFlushTimer);
  voiceConversationFlushTimer = null;
}

function markVoiceActivity() {
  clearVoiceConversationFlushTimer();
  voiceGlobalSilenceReady = false;
}

function resetVoiceConversationBuffer() {
  clearVoiceConversationFlushTimer();
  voiceConversationTurns = [];
  voiceGlobalSilenceReady = false;
}

function appendVoiceConversationTurn(member, transcript) {
  const content = transcript?.trim();
  if (!content) return;

  voiceConversationTurns.push({
    userId: member.user.id,
    userName: member.displayName || member.user.username,
    content,
  });
}

function flushVoiceConversationPrompt(context) {
  const turns = voiceConversationTurns.splice(0);
  voiceGlobalSilenceReady = false;

  if (!turns.length) return;

  const content = turns
    .map((turn) => `${turn.userName} : ${turn.content}`)
    .join("\n");

  console.log(`[VOICE PROMPT] Aggregated ${turns.length} turn(s):\n${content}`);

  if (!ws || ws.readyState !== WebSocket.OPEN) {
    console.warn("WS is not open. Dropping aggregated voice prompt.");
    return;
  }

  const firstTurn = turns[0];

  ws.send(
    JSON.stringify({
      type: "discord.llm_prompt_vc",
      request_id: crypto.randomUUID(),

      payload: {
        content,
        image_url: null,

        author_id: firstTurn.userId,
        author_name: "Conversation vocale",
        is_bot: false,

        channel_id: context.channelId,
        message_id: null,
        guild_id: context.guildId,
        is_dm: false,

        source: "voice_conversation",
        preformatted: true,
        speaker_turns: turns,
      },
    }),
  );
}

function flushVoiceConversationIfReady(context, listeningUsers, speakingUsers) {
  if (!voiceGlobalSilenceReady) return;
  if (listeningUsers.size > 0 || speakingUsers.size > 0) return;
  if (!voiceConversationTurns.length) return;

  flushVoiceConversationPrompt(context);
}

function scheduleVoiceConversationFlush(
  context,
  listeningUsers,
  speakingUsers,
) {
  if (speakingUsers.size > 0) return;

  if (voiceGlobalSilenceReady) {
    flushVoiceConversationIfReady(context, listeningUsers, speakingUsers);
    return;
  }

  if (voiceConversationFlushTimer) return;

  voiceConversationFlushTimer = setTimeout(() => {
    voiceConversationFlushTimer = null;
    voiceGlobalSilenceReady = true;
    flushVoiceConversationIfReady(context, listeningUsers, speakingUsers);
  }, GLOBAL_VOICE_FLUSH_DELAY_MS);
}

function mono48ToStereo48(buffer) {
  const mono = new Int16Array(
    buffer.buffer,
    buffer.byteOffset,
    buffer.length / Int16Array.BYTES_PER_ELEMENT,
  );

  const stereo = new Int16Array(mono.length * 2);

  for (let i = 0; i < mono.length; i++) {
    const sample = mono[i];
    const stereoIndex = i * 2;
    stereo[stereoIndex] = sample;
    stereo[stereoIndex + 1] = sample;
  }

  return Buffer.from(stereo.buffer, stereo.byteOffset, stereo.byteLength);
}

function connectToMainWS() {
  if (!ws_loop) return;

  console.log("Tring to connect to WebSocket server.");

  ws = new WebSocket("ws://local.example:8765");

  ws.on("open", () => {
    ws.send(
      JSON.stringify({
        type: "ws.identify",
        role: "discord",
      }),
    );
    sendVoiceState(hasActiveVoiceConnection());

    console.log("Connection established.");
  });

  ws.on("message", async (data) => {
    const msg = data.toString();
    let json_msg;

    try {
      json_msg = JSON.parse(msg);
    } catch (err) {
      console.log("[WS] Non-JSON message received:", msg);
    }

    if (msg === "shutdown") {
      console.log("Shutdown signal received.");
      console.log("Closing WS connection...");

      try {
        ws_loop = false;
        ws.close();
        console.log("WS connection successfully closed.");
      } catch (error) {
        console.error("Cannot close WS connection : ", error);
      }

      console.log("Disconnecting Discord bot...");
      try {
        if (subscription) {
          setTimeout(() => subscription.unsubscribe(), 5_000);
        }
        if (connection) {
          sendVoiceState(false);
          connection.destroy();
        }
        await client.destroy();
        console.log("Discord bot successfully disconnected.");
      } catch (error) {
        console.error("Unable to disconnet Discord bot : ", error);
      }
      return;
    }

    if (json_msg?.type === "discord.control") {
      applyDiscordControl(json_msg.payload);
      return;
    }

    if (json_msg?.type === "discord.llm_vocal_response") {
      if (ttsDisabled || botDisabled) {
        return;
      }


      if (!connection) {
        console.warn("No active voice connection. Cannot play TTS.");
        return;
      }

      if (!subscription) {
        subscription = connection.subscribe(audioPlayer);
      }

      const payload = json_msg.payload;

      if (payload.mode === "file") {
        const resource = createAudioResource(payload.path);
        audioPlayer.play(resource);
        return;
      }

      if (payload.mode === "stream_start") {
        currentTTSstream = new PassThrough();

        const resource = createAudioResource(currentTTSstream, {
          inputType: StreamType.Raw,
        });

        enqueueTTSResource(resource);
        return;
      }

      if (payload.mode === "stream_chunk") {
        if (!currentTTSstream) {
          console.warn("Received TTS chunk without stream_start.");
          return;
        }

        const monoChunk = Buffer.from(payload.audio, "base64");
        const stereoChunk = mono48ToStereo48(monoChunk);

        currentTTSstream.write(stereoChunk);
        return;
      }

      if (payload.mode === "stream_end") {
        if (currentTTSstream) {
          currentTTSstream.end();
          currentTTSstream = null;
        }
        return;
      }
    }

    if (json_msg?.type === "discord.llm_response") {
      const responseReceivedAtMs = Date.now();
      const requestId = json_msg.request_id;
      const pendingTiming = requestId ? pendingPromptTimings.get(requestId) : null;
      if (requestId) pendingPromptTimings.delete(requestId);

      const fetchStartedAtMs = Date.now();
      const channel = await client.channels.fetch(json_msg.channel_id);
      const fetchEndedAtMs = Date.now();
      const content = (json_msg.content ?? "").trim();

      if (!content) {
        console.warn("[LLM] Empty Discord response skipped.");
        return;
      }

      const sendStartedAtMs = Date.now();
      await channel.send({
        content,
        reply: json_msg.reply_to_message_id
          ? { messageReference: json_msg.reply_to_message_id }
          : undefined,
      });
      const sendEndedAtMs = Date.now();

      const timings = {
        ...(json_msg.timings ?? {}),
        discord_channel_fetch_s: (fetchEndedAtMs - fetchStartedAtMs) / 1000,
        discord_channel_send_s: (sendEndedAtMs - sendStartedAtMs) / 1000,
        discord_response_to_visible_s: (sendEndedAtMs - responseReceivedAtMs) / 1000,
      };
      if (pendingTiming) {
        timings.discord_ws_roundtrip_s = (responseReceivedAtMs - pendingTiming.sentAtMs) / 1000;
        timings.discord_total_until_visible_s = (sendEndedAtMs - pendingTiming.sentAtMs) / 1000;
        timings.discord_pre_ws_prepare_s = (pendingTiming.sentAtMs - pendingTiming.startedAtMs) / 1000;
      }
      console.log(
        `[PIPELINE:DISCORD_TEXT] request_id=${requestId ?? "unknown"} timings=${JSON.stringify(timings)} chars=${content.length}`,
      );
    }
  });

  ws.on("close", () => {
    console.log("Connection closed.");

    if (ws_loop) {
      setTimeout(connectToMainWS, 1000);
    }
  });

  ws.on("error", async (err) => {
    const ws_err = err.toString();
    console.error("WebSocket error : ", ws_err);
    console.log(
      "Due to this error, WS connection and Discord bot will be shut down...",
    );
    try {
      ws.close();
      await client.destroy();
      console.log(
        "WS connection and Discord bot disconnected due to an error event.",
      );
    } catch (error) {
      console.log(
        "Unable to shut down WS connection and / or Discord bot : ",
        error,
      );
    }
  });
}

async function connectToChannel(channel) {
  const connection = joinVoiceChannel({
    channelId: channel.id,
    guildId: channel.guild.id,
    adapterCreator: channel.guild.voiceAdapterCreator,
    selfDeaf: false,
  });

  try {
    await entersState(connection, VoiceConnectionStatus.Ready, 30_000);
    return connection;
  } catch (error) {
    connection.destroy();
    throw error;
  }
}

async function Listen(connection, userId) {
  return new Promise((resolve, reject) => {
    let settled = false;
    const opusStream = connection.receiver.subscribe(userId, {
      end: {
        behavior: EndBehaviorType.AfterSilence,
        duration: LISTEN_END_SILENCE_MS,
      },
    });

    const decoder = new prism.opus.Decoder({
      rate: VOICE_SAMPLE_RATE,
      channels: 1,
      frameSize: DISCORD_OPUS_FRAME_SIZE,
    });

    const { stream, result } = transcribe();

    const settle = (fn, value) => {
      if (settled) return;
      settled = true;
      fn(value);
    };

    const fail = (err) => {
      console.error("Discord audio error:", err);
      opusStream.destroy();
      decoder.destroy();
      stream.destroy(err);
      settle(reject, err);
    };

    opusStream.on("error", fail);
    decoder.on("error", fail);

    opusStream.pipe(decoder).pipe(stream);

    result
      .then((finalText) => settle(resolve, finalText))
      .catch((err) => {
        opusStream.destroy();
        decoder.destroy();
        settle(reject, err);
      });
  });
}

const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.GuildVoiceStates,
    GatewayIntentBits.MessageContent,
    GatewayIntentBits.DirectMessages,
  ],
  partials: [Partials.Channel, Partials.Message, Partials.User],
});

client.on(Events.ClientReady, (readyClient) => {
  console.log(`Logged in as ${readyClient.user.tag}!`);
});


client.on("messageCreate", async (message) => {
  if (message.author.id === client.user.id) return;

  const whitelist = JSON.parse(
    fs.readFileSync("./src/discord/whitelist.json", "utf-8"),
  );

  if (!Object.values(whitelist).includes(message.author.id)) return;

  const botMention = `<@${client.user.id}>`;
  const botMentionNick = `<@!${client.user.id}>`;

  if (message.content.includes("!join")) {
    if (!message.member?.voice.channel) {
      console.error("You must join a voice channel first");
      return;
    }

    try {
      if (subscription) {
        subscription.unsubscribe();
        subscription = null;
      }

      if (connection) {
        sendVoiceState(false);
        connection.destroy();
        connection = null;
      }

      resetVoiceConversationBuffer();

      connection = await connectToChannel(message.member.voice.channel);
      const voiceConnection = connection;
      subscription = voiceConnection.subscribe(audioPlayer);
      voiceConnection.once(VoiceConnectionStatus.Destroyed, () => {
        if (connection !== voiceConnection) return;
        connection = null;
        subscription = null;
        sendVoiceState(false);
      });
      sendVoiceState(true);

      const listeningUsers = new Set();
      const speakingUsers = new Set();
      const voiceContext = {
        channelId: message.channel.id,
        guildId: message.guild.id,
      };

      voiceConnection.receiver.speaking.on("end", (userId) => {
        speakingUsers.delete(userId);
        if (speakingUsers.size === 0 && listeningUsers.size === 0) {
          sendControlSignal("humanSpeaking", false);
        }
        scheduleVoiceConversationFlush(
          voiceContext,
          listeningUsers,
          speakingUsers,
        );
      });

      voiceConnection.receiver.speaking.on("start", async (userId) => {
        if (botDisabled || sttMuted) {
          sendControlSignal("humanSpeaking", false);
          return;
        }

        markVoiceActivity();
        speakingUsers.add(userId);
        sendControlSignal("humanSpeaking", true);

        if (listeningUsers.has(userId)) return;

        const member = await message.guild.members
          .fetch(userId)
          .catch(() => null);
        if (!member) {
          speakingUsers.delete(userId);
          scheduleVoiceConversationFlush(
            voiceContext,
            listeningUsers,
            speakingUsers,
          );
          return;
        }

        listeningUsers.add(userId);

        console.log(`Listening to ${member.user.username}`);

        try {
          const transcript = await Listen(voiceConnection, userId);

          if (!transcript?.trim()) return;

          console.log(
            `[VOICE TRANSCRIPT] ${member.user.username}:`,
            transcript,
          );

          appendVoiceConversationTurn(member, transcript);
        } catch (err) {
          console.error("Error while listening:", err);
        } finally {
          listeningUsers.delete(userId);
          if (!voiceConnection.receiver.speaking.users.has(userId)) {
            speakingUsers.delete(userId);
          }
          if (speakingUsers.size === 0 && listeningUsers.size === 0) {
            sendControlSignal("humanSpeaking", false);
          }
          scheduleVoiceConversationFlush(
            voiceContext,
            listeningUsers,
            speakingUsers,
          );
        }
      });

      console.log("Bot joined and is now listening");
    } catch (error) {
      sendVoiceState(false);
      console.error("An error occured in voice channel :\n", error);
    }
  }
  if (message.content.includes("!leave")) {
    try {
      if (connection) {
        sendVoiceState(false);
        connection.destroy();
        connection = null;
      }

      resetVoiceConversationBuffer();

      if (subscription) {
        subscription.unsubscribe();
        subscription = null;
      }
      console.log("Bot left the voice channel.");
    } catch (error) {
      console.error("!leave command failed : ", error);
    }
  }

  if (botDisabled) return;

  if (!message.guild) {
    console.log(
      `DM received from ${message.author.username}:`,
      message.content,
    );
    if (ws && ws.readyState === WebSocket.OPEN) {
      const promptStartedAtMs = Date.now();
      const cleanContent = message.content
        .replaceAll(botMention, "")
        .replaceAll(botMentionNick, "")
        .replace(/<@!?\d+>/g, "")
        .trim();
      const imageUrl = getFirstImageUrl(message);
      const attachments = getAttachments(message);
      const requestId = crypto.randomUUID();
      const sentAtMs = Date.now();
      pendingPromptTimings.set(requestId, {
        startedAtMs: promptStartedAtMs,
        sentAtMs,
      });

      ws.send(
        JSON.stringify({
          type: "discord.llm_prompt",
          request_id: requestId,

          payload: {
            content: cleanContent,
            image_url: imageUrl,
            author_id: message.author.id,
            author_name: message.author.username,
            is_bot: message.author.bot,
            channel_id: message.channel.id,
            message_id: message.id,
            guild_id: message.guild?.id ?? null,
            is_dm: !message.guild,
            attachments,
            client_sent_at_ms: sentAtMs,
          },
        }),
      );
    }
  }

  if (message.guild) {
    if (!message.mentions.has(client.user)) return;
    console.log(`Message received in ${message.guild.name} :`, message.content);
    if (ws && ws.readyState === WebSocket.OPEN) {
      const promptStartedAtMs = Date.now();
      const cleanContent = message.content
        .replaceAll(botMention, "")
        .replaceAll(botMentionNick, "")
        .replace(/<@!?\d+>/g, "")
        .trim();
      const imageUrl = getFirstImageUrl(message);
      const attachments = getAttachments(message);
      const requestId = crypto.randomUUID();
      const sentAtMs = Date.now();
      pendingPromptTimings.set(requestId, {
        startedAtMs: promptStartedAtMs,
        sentAtMs,
      });

      ws.send(
        JSON.stringify({
          type: "discord.llm_prompt",
          request_id: requestId,

          payload: {
            content: cleanContent,
            image_url: imageUrl,
            author_id: message.author.id,
            author_name: message.author.username,
            is_bot: message.author.bot,
            channel_id: message.channel.id,
            message_id: message.id,
            guild_id: message.guild?.id ?? null,
            is_dm: !message.guild,
            attachments,
            client_sent_at_ms: sentAtMs,
          },
        }),
      );
    }
  }
});

client.on("warn", console.warn);
client.on("error", console.error);

client.login(discord_token);
await warmup_stt();
connectToMainWS();
