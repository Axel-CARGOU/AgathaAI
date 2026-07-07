const clone = (value) => JSON.parse(JSON.stringify(value));

const defaultState = {
  updateScope: "full",
  runtime: {
    modulesActive: {
      bot: false,
      llm: false,
      tts: false,
      stt: false,
      vtuber: false,
      rag: false,
    },
    signals: {
      aiThinking: false,
      aiSpeaking: false,
      humanSpeaking: false,
    },
    controls: {
      botDisabled: false,
      llmDisabled: false,
      ttsDisabled: false,
      muted: false,
      vtuberDisabled: false,
    },
  },
  discord: {
    inVoice: false,
  },
  currentMessage: {
    source: "Idle",
    content: "",
    streamMode: "Waiting",
    streamPreview: "",
  },
  backendConnected: false,
  toggles: {
    twitchChat: false,
    twitchPoll: false,
    twitchBan: false,
    twitchTimeout: false,
    devMode: false,
    vision: false,
    webSearch: false,
    fileSearch: false,
    editFile: false,
  },
  twitch: {
    messages: [
      {
        user_name: "Viewer01",
        content: "Panel mock ready.",
      },
      {
        user_name: "Viewer02",
        content: "Twitch chat will be wired by the runtime websocket.",
      },
    ],
  },
  settings: {
    general: [
      {
        key: "LANGUAGE",
        label: "Language",
        value: "fr",
        type: "text",
        editable: true,
      },
      {
        key: "DEV_MODE",
        label: "Dev Mode",
        value: false,
        type: "boolean",
        editable: true,
      },
      {
        key: "VTUBING",
        label: "VTubing",
        value: true,
        type: "boolean",
        editable: true,
      },
      {
        key: "VTS_PORT",
        label: "VTS Port",
        value: 7801,
        type: "number",
        editable: true,
        lockedWhen: "vtuber",
      },
      {
        key: "RAG_ENABLED",
        label: "RAG Enabled",
        value: true,
        type: "boolean",
        editable: true,
      },
      {
        key: "RAG_QDRANT_PATH",
        label: "RAG Qdrant Path",
        value: "qdrant_data",
        type: "text",
        editable: true,
        lockedWhen: "rag",
        lockedReason: "Stop RAG first",
      },
      {
        key: "RAG_COLLECTION_NAME",
        label: "RAG Collection",
        value: "agathaai_long_term_memory",
        type: "text",
        editable: true,
        lockedWhen: "rag",
        lockedReason: "Stop RAG first",
      },
      {
        key: "RAG_EMBEDDING_MODEL",
        label: "RAG Embedding Model",
        value: "intfloat/multilingual-e5-small",
        type: "text",
        editable: true,
        lockedWhen: "rag",
        lockedReason: "Stop RAG first",
      },
      {
        key: "RAG_HASH_EMBEDDING_SIZE",
        label: "RAG Hash Embedding Size",
        value: 384,
        type: "number",
        editable: true,
        lockedWhen: "rag",
        lockedReason: "Stop RAG first",
      },
      {
        key: "RAG_SHORT_TERM_LIMIT",
        label: "RAG Short-Term Limit",
        value: 10,
        type: "number",
        editable: true,
      },
      {
        key: "RAG_LONG_TERM_CONTEXT_LIMIT",
        label: "RAG Long-Term Context Limit",
        value: 5,
        type: "number",
        editable: true,
      },
      {
        key: "RAG_MAX_CONTEXT_CHARS",
        label: "RAG Max Context Chars",
        value: 4000,
        type: "number",
        editable: true,
      },
      {
        key: "RAG_MAX_ENTRY_CHARS",
        label: "RAG Max Entry Chars",
        value: 650,
        type: "number",
        editable: true,
      },
      {
        key: "RAG_MAX_STORED_CHARS",
        label: "RAG Max Stored Chars",
        value: 3000,
        type: "number",
        editable: true,
      },
    ],
    llm: [
      {
        key: "LLM_MODEL_PATH",
        label: "Model Path",
        value: "agathaai_vision_v7_7b_qwen2.5-vl_gptq_int4",
        type: "text",
        editable: true,
        lockedWhen: "llm",
        lockedReason: "Unload LLM first",
      },
      {
        key: "LLM_MAX_NEW_TOKENS",
        label: "Max New Tokens",
        value: 512,
        type: "number",
        editable: true,
      },
      {
        key: "LLM_TEMPERATURE",
        label: "Temperature",
        value: 0.7,
        type: "number",
        editable: true,
        step: "0.05",
      },
      {
        key: "LLM_TOP_P",
        label: "Top P",
        value: 1.0,
        type: "number",
        editable: true,
        step: "0.05",
      },
      {
        key: "LLM_MIN_P",
        label: "Min P",
        value: 0.1,
        type: "number",
        editable: true,
        step: "0.05",
      },
      {
        key: "LLM_MAX_SEQ_LEN",
        label: "Max Seq Len",
        value: 10240,
        type: "number",
        editable: true,
        lockedWhen: "llm",
        lockedReason: "Unload LLM first",
      },
      {
        key: "LLM_REPETITION_PENALTY",
        label: "Repetition Penalty",
        value: 1.2,
        type: "number",
        editable: true,
        step: "0.05",
      },
      {
        key: "LLM_PRESENCE_PENALTY",
        label: "Presence Penalty",
        value: 0.3,
        type: "number",
        editable: true,
        step: "0.05",
      },
      {
        key: "KV_Q4",
        label: "KV Q4",
        value: true,
        type: "boolean",
        editable: true,
        lockedWhen: "llm",
        lockedReason: "Unload LLM first",
      },
      {
        key: "LLM_STARTUP_TIMEOUT",
        label: "Startup Timeout",
        value: 900,
        type: "number",
        editable: true,
      },
      {
        key: "LLM_REQUEST_TIMEOUT",
        label: "Request Timeout",
        value: 300,
        type: "number",
        editable: true,
      },
      {
        key: "LLM_GPU_MAX_USE",
        label: "GPU Max Use",
        value: 0.9,
        type: "number",
        editable: true,
        step: "0.05",
        lockedWhen: "llm",
        lockedReason: "Unload LLM first",
      },
      {
        key: "EDIT_FILE",
        label: "Edit File Tool",
        value: false,
        type: "boolean",
        editable: true,
      },
      {
        key: "FILES_SEARCH",
        label: "Files Search Tool",
        value: false,
        type: "boolean",
        editable: true,
      },
      {
        key: "WEB_SEARCH",
        label: "Web Search Tool",
        value: false,
        type: "boolean",
        editable: true,
      },
      {
        key: "TWITCH_CHAT",
        label: "Twitch Chat",
        value: false,
        type: "boolean",
        editable: true,
      },
      {
        key: "TWITCH_POLL",
        label: "Twitch Poll",
        value: false,
        type: "boolean",
        editable: true,
      },
      {
        key: "TWITCH_BAN",
        label: "Twitch Ban",
        value: false,
        type: "boolean",
        editable: true,
      },
      {
        key: "TWITCH_TIMEOUT",
        label: "Twitch Timeout",
        value: false,
        type: "boolean",
        editable: true,
      },
    ],
    stt: [
      {
        key: "SAMPLE_RATE",
        label: "Sample Rate",
        value: 48000,
        type: "number",
        editable: true,
        lockedWhen: "stt",
        lockedReason: "Mute or stop STT first",
      },
      {
        key: "STT_MODEL",
        label: "STT Model",
        value: "Google STT",
        type: "text",
        editable: false,
        lockedReason: "Runtime provider",
      },
      {
        key: "LANGUAGE_CODE",
        label: "Language Code",
        value: "fr-FR",
        type: "text",
        editable: false,
        lockedReason: "Node STT constant",
      },
      {
        key: "ALTERNATIVE_LANGUAGE_CODES",
        label: "Alternative Languages",
        value: "en-US",
        type: "text",
        editable: false,
        lockedReason: "Node STT constant",
      },
    ],
    tts: [
      {
        key: "VOICE_SAMPLE",
        label: "Voice Sample",
        value: "agatha.wav",
        type: "text",
        editable: true,
      },
      {
        key: "XTTS_STREAM_CHUNK_SIZE",
        label: "Stream Chunk Size",
        value: 20,
        type: "number",
        editable: false,
        lockedReason: "Code constant",
      },
      {
        key: "XTTS_STREAM_OVERLAP_WAV_LEN",
        label: "Stream Overlap",
        value: 1024,
        type: "number",
        editable: false,
        lockedReason: "Code constant",
      },
      {
        key: "SAMPLE_RATE_OUT",
        label: "Output Sample Rate",
        value: 48000,
        type: "number",
        editable: false,
        lockedReason: "Code constant",
      },
    ],
  },
  memories: [
    {
      id: "short-1",
      kind: "short",
      source: "discord",
      created_at: "2026-06-04 10:22:00 UTC",
      user_name: "Axel",
      user_prompt: "Explique rapidement ce que le control panel doit afficher.",
      llm_response:
        "Il doit montrer le message courant, les signaux runtime, les settings, la memoire et les modules.",
    },
    {
      id: "long-1",
      kind: "long",
      source: "rag",
      created_at: "2026-06-03 08:16:00 UTC",
      user_name: "Axel",
      user_prompt: "Quel style visuel pour AgathaAI ?",
      llm_response:
        "Un panel sombre, dense, avec accent jaune et actions dangereuses en rouge.",
    },
  ],
  moderation: {
    blacklist: ["pd", "négro"],
  },
  metrics: {
    frozenTimestamp: "",
    requestGraph: { exists: false, url: "", name: "", updatedAt: "" },
    engineGraph: { exists: false, url: "", name: "", updatedAt: "" },
    nvidiaGraph: { exists: false, url: "", name: "", updatedAt: "" },
    requestCsv: { exists: false, url: "", name: "", updatedAt: "" },
    requestSummaryCsv: { exists: false, url: "", name: "", updatedAt: "" },
    engineCsv: { exists: false, url: "", name: "", updatedAt: "" },
    nvidiaCsv: { exists: false, url: "", name: "", updatedAt: "" },
    autoRenderer: { status: "Stopped", pid: null, intervalSeconds: 5 },
    files: [],
  },
  contexts: {
    main: "main_context.json\n\nCurrent loaded persona prompt will appear here.",
    format: "format_context.json\n\nJSON output contract will appear here.",
    emotes: "emotes_context.json\n\nEmote rules will appear here.",
    tools: "tool_context.json\n\nTool instructions will appear here.",
    twitch:
      "twitch_context.json\n\nTwitch command instructions will appear here.",
  },
  seenByLlm: "Seen by LLM preview will be streamed by the runtime websocket.",
  temporaryContext: "",
  games: {
    lichess: {
      status: "Stopped",
    },
  },
};

let state = clone(defaultState);
let socket = null;
let socketPromise = null;
let requestCounter = 0;
const pendingRequests = new Map();
const subscribers = new Set();

function mockModeEnabled() {
  return new URLSearchParams(window.location.search).has("mock");
}

function panelWsUrl() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const hostname = window.location.hostname || "local.example";
  return `${protocol}//${hostname}:8765`;
}

function dispatchStateUpdate() {
  const snapshot = clone(state);
  subscribers.forEach((callback) => callback(snapshot));
  window.dispatchEvent(
    new CustomEvent("agatha-panel-state", { detail: snapshot }),
  );
}

function applyRemoteState(remoteState) {
  if (!remoteState || typeof remoteState !== "object") {
    return clone(state);
  }

  state.updateScope = remoteState.scope === "realtime" ? "realtime" : "full";

  if (remoteState.runtime) {
    state.runtime = {
      ...state.runtime,
      ...remoteState.runtime,
      modulesActive: {
        ...state.runtime.modulesActive,
        ...(remoteState.runtime.modulesActive ?? {}),
      },
      signals: {
        ...state.runtime.signals,
        ...(remoteState.runtime.signals ?? {}),
      },
      controls: {
        ...state.runtime.controls,
        ...(remoteState.runtime.controls ?? {}),
      },
    };
  }

  if (remoteState.currentMessage) {
    state.currentMessage = {
      ...state.currentMessage,
      ...remoteState.currentMessage,
    };
  }

  if (remoteState.toggles) {
    state.toggles = {
      ...state.toggles,
      ...remoteState.toggles,
    };
  }

  if (remoteState.twitch) {
    state.twitch = {
      ...state.twitch,
      ...remoteState.twitch,
    };
  }

  if (remoteState.discord) {
    state.discord = {
      ...state.discord,
      ...remoteState.discord,
    };
  }

  if (remoteState.moderation) {
    state.moderation = {
      ...state.moderation,
      ...remoteState.moderation,
    };
  }

  if (remoteState.games) {
    state.games = {
      ...state.games,
      ...remoteState.games,
    };
  }

  if (remoteState.metrics) {
    state.metrics = {
      ...state.metrics,
      ...remoteState.metrics,
    };
  }

  if (Array.isArray(remoteState.memories)) {
    state.memories = remoteState.memories;
  }

  if (remoteState.contexts) {
    state.contexts = {
      ...state.contexts,
      ...remoteState.contexts,
    };
  }

  if ("seenByLlm" in remoteState) {
    state.seenByLlm = String(remoteState.seenByLlm ?? "");
  }

  if ("temporaryContext" in remoteState) {
    state.temporaryContext = String(remoteState.temporaryContext ?? "");
  }

  if (remoteState.settingsValues) {
    applySettingValues(remoteState.settingsValues);
  }

  return clone(state);
}

function applySettingValues(values) {
  Object.values(state.settings).forEach((group) => {
    group.forEach((setting) => {
      if (Object.hasOwn(values, setting.key)) {
        setting.value = values[setting.key];
      }
    });
  });
}

function handleSocketMessage(event) {
  let message;
  try {
    message = JSON.parse(event.data);
  } catch {
    return;
  }

  if (message.type === "control_panel.state") {
    state.backendConnected = true;
    applyRemoteState(message.payload);
    dispatchStateUpdate();
    return;
  }

  if (message.type === "control_panel.result") {
    const pending = pendingRequests.get(message.request_id);
    if (!pending) return;

    pendingRequests.delete(message.request_id);

    if (message.payload?.state) {
      state.backendConnected = true;
      applyRemoteState(message.payload.state);
      dispatchStateUpdate();
    }

    if (message.ok === false) {
      pending.reject(
        new Error(message.payload?.error ?? "Control panel request failed"),
      );
      return;
    }

    pending.resolve(clone(state));
  }
}

async function connectSocket(timeoutMs = 450) {
  if (mockModeEnabled()) return null;
  if (socket?.readyState === WebSocket.OPEN) return socket;
  if (socketPromise) return socketPromise;

  socketPromise = new Promise((resolve) => {
    const ws = new WebSocket(panelWsUrl());
    let settled = false;

    const settle = (value) => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timer);
      socketPromise = null;
      resolve(value);
    };

    const timer = window.setTimeout(() => {
      try {
        ws.close();
      } catch {
      }
      settle(null);
    }, timeoutMs);

    ws.addEventListener("open", () => {
      socket = ws;
      ws.send(JSON.stringify({ type: "ws.identify", role: "control_panel" }));
      settle(ws);
    });

    ws.addEventListener("message", handleSocketMessage);

    ws.addEventListener("close", () => {
      if (socket === ws) socket = null;
      state.backendConnected = false;
      dispatchStateUpdate();
      pendingRequests.forEach((pending) =>
        pending.reject(new Error("Control panel websocket closed")),
      );
      pendingRequests.clear();
      settle(null);
    });

    ws.addEventListener("error", () => {
      settle(null);
    });
  });

  return socketPromise;
}

async function requestRuntime(type, payload = {}, timeoutMs = 2500) {
  const ws = await connectSocket();
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    return null;
  }

  const requestId = `cp-${Date.now()}-${++requestCounter}`;
  const message = {
    type,
    request_id: requestId,
    payload,
  };

  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(() => {
      pendingRequests.delete(requestId);
      reject(new Error(`${type} timed out`));
    }, timeoutMs);

    pendingRequests.set(requestId, {
      resolve: (value) => {
        window.clearTimeout(timer);
        resolve(value);
      },
      reject: (error) => {
        window.clearTimeout(timer);
        reject(error);
      },
    });

    ws.send(JSON.stringify(message));
  });
}

async function requestRuntimeOrFallback(type, payload = {}, timeoutMs = 2500) {
  try {
    return await requestRuntime(type, payload, timeoutMs);
  } catch (error) {
    if (socket?.readyState === WebSocket.OPEN) {
      throw error;
    }
    return null;
  }
}

function findSetting(group, key) {
  return state.settings[group]?.find((item) => item.key === key);
}

function coerceSettingValue(setting, value) {
  if (setting.type === "boolean") {
    return value === true || value === "true";
  }

  if (setting.type === "number") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : setting.value;
  }

  return String(value ?? "");
}

function formatInternalCommand(command) {
  return [
    "[INTERNAL_COMMAND]",
    `type: ${String(command?.type ?? "").trim()}`,
    `language: ${String(command?.language ?? "FR").trim().toUpperCase()}`,
    `context: ${String(command?.context ?? "").trim()}`,
    `instruction: ${String(command?.instruction ?? "").trim()}`,
  ].join("\n");
}

export function onPanelStateUpdate(callback) {
  subscribers.add(callback);
  return () => subscribers.delete(callback);
}

export async function getInitialPanelState() {
  await connectSocket();
  await requestRuntimeOrFallback("control_panel.get_state").catch(() => null);
  return clone(state);
}

export async function saveSetting(group, key, value) {
  const setting = findSetting(group, key);
  if (!setting) {
    throw new Error(`Unknown setting: ${group}.${key}`);
  }

  const nextValue = coerceSettingValue(setting, value);
  const remote = await requestRuntimeOrFallback("control_panel.set_setting", {
    key,
    value: nextValue,
  }, 1_200_000);
  if (remote) return remote;

  setting.value = nextValue;
  return clone(state);
}

export async function saveToggle(key, value) {
  const nextValue = Boolean(value);
  const remote = await requestRuntimeOrFallback("control_panel.set_toggle", {
    key,
    value: nextValue,
  });
  if (remote) return remote;

  state.toggles[key] = nextValue;
  return clone(state);
}

export async function runControlAction(action) {
  const remote = await requestRuntimeOrFallback(
    "control_panel.control",
    { action },
    1_200_000,
  );
  if (remote) return remote;

  const controls = state.runtime.controls;
  const active = state.runtime.modulesActive;

  if (action === "bot") controls.botDisabled = !controls.botDisabled;
  if (action === "llm") controls.llmDisabled = !controls.llmDisabled;
  if (action === "tts") controls.ttsDisabled = !controls.ttsDisabled;
  if (action === "mute") controls.muted = !controls.muted;
  if (action === "vtuber") controls.vtuberDisabled = !controls.vtuberDisabled;

  active.bot = !controls.botDisabled;
  active.llm = !controls.llmDisabled;
  active.tts = !controls.ttsDisabled;
  active.stt = !controls.muted;
  active.vtuber = !controls.vtuberDisabled && active.vtuber;

  return clone(state);
}

export async function saveContext(name, content) {
  const remote = await requestRuntimeOrFallback("control_panel.set_context", {
    name,
    content,
  });
  if (remote) return remote;

  state.contexts[name] = String(content ?? "");
  return clone(state);
}

export async function saveTemporaryContext(content) {
  const nextContent = String(content ?? "");
  const remote = await requestRuntimeOrFallback(
    "control_panel.set_temporary_context",
    { content: nextContent },
  );
  if (remote) return remote;

  state.temporaryContext = nextContent;
  return clone(state);
}

export async function sendInternalCommand(command) {
  const remote = await requestRuntimeOrFallback(
    "control_panel.internal_command",
    command,
    1_200_000,
  );
  if (remote) return remote;

  state.currentMessage = {
    ...state.currentMessage,
    source: "control_panel_command",
    content: formatInternalCommand(command),
    streamMode: "Queued",
  };
  return clone(state);
}

export async function addBlacklistWord(word) {
  const normalized = String(word ?? "")
    .trim()
    .toLowerCase();
  const remote = await requestRuntimeOrFallback(
    "control_panel.add_blacklist_word",
    { word: normalized },
  );
  if (remote) return remote.moderation.blacklist;

  if (normalized && !state.moderation.blacklist.includes(normalized)) {
    state.moderation.blacklist.push(normalized);
  }

  return clone(state.moderation.blacklist);
}

export async function deleteBlacklistWord(word) {
  const remote = await requestRuntimeOrFallback(
    "control_panel.delete_blacklist_word",
    { word },
  );
  if (remote) return remote.moderation.blacklist;

  state.moderation.blacklist = state.moderation.blacklist.filter(
    (item) => item !== word,
  );
  return clone(state.moderation.blacklist);
}

export async function updateGameStatus(game, status) {
  const action = `${game}-${String(status).toLowerCase() === "running" ? "start" : "stop"}`;
  const remote = await requestRuntimeOrFallback(
    "control_panel.game",
    { action },
    60_000,
  );
  if (remote) return remote.games[game];

  if (!state.games[game]) {
    state.games[game] = {};
  }

  state.games[game].status = status;
  return clone(state.games[game]);
}

export async function runPanelAction(action) {
  if (action === "abort-message") {
    const remote = await requestRuntimeOrFallback(
      "control_panel.abort",
      {},
      10_000,
    );
    if (remote) return remote;

    state.currentMessage.streamMode = "Aborted";
    state.runtime.signals.aiThinking = false;
    state.runtime.signals.aiSpeaking = false;
  }

  if (action === "clear-current") {
    const remote = await requestRuntimeOrFallback(
      "control_panel.clear_current",
      {},
      10_000,
    );
    if (remote) return remote;

    state.currentMessage = {
      source: "Idle",
      content: "",
      streamMode: "Waiting",
      streamPreview: "",
    };
    state.runtime.signals.aiThinking = false;
    state.runtime.signals.aiSpeaking = false;
  }

  if (action === "metrics-refresh") {
    const remote = await requestRuntimeOrFallback(
      "control_panel.refresh_metrics",
      {},
      60_000,
    );
    if (remote) return remote;
  }

  return clone(state);
}

export async function searchMemories(query = "") {
  const remote = await requestRuntimeOrFallback(
    "control_panel.search_memories",
    { query, limit: 100 },
    30_000,
  );
  if (remote) return remote;

  if (!query) return clone(state);

  const needle = String(query).toLowerCase();
  state.memories = state.memories.filter((memory) =>
    JSON.stringify(memory).toLowerCase().includes(needle),
  );
  return clone(state);
}

export async function saveMemoryEntry(entry) {
  const remote = await requestRuntimeOrFallback(
    "control_panel.save_memory",
    entry,
    30_000,
  );
  if (remote) return remote;

  if (entry.id) {
    const index = state.memories.findIndex((memory) => memory.id === entry.id);
    if (index >= 0) {
      state.memories[index] = { ...state.memories[index], ...entry };
    }
  } else {
    state.memories.unshift({
      id: `local-${Date.now()}`,
      kind: "short",
      source: "control_panel",
      created_at: new Date().toISOString(),
      user_name: entry.user_name || "Control Panel",
      user_prompt: entry.user_prompt || "",
      llm_response: entry.llm_response || "",
    });
  }

  return clone(state);
}

export async function deleteMemoryEntry(id) {
  const remote = await requestRuntimeOrFallback(
    "control_panel.delete_memory",
    { id },
    30_000,
  );
  if (remote) return remote;

  state.memories = state.memories.filter((memory) => memory.id !== id);
  return clone(state);
}

export async function clearShortTermMemories() {
  const remote = await requestRuntimeOrFallback(
    "control_panel.clear_short_memories",
    {},
    30_000,
  );
  if (remote) return remote;

  state.memories = [];
  return clone(state);
}

export async function importMemories(entries) {
  const normalizedEntries = entries.map((entry) => ({
      user_prompt: entry.user_prompt ?? entry.prompt ?? "",
      llm_response: entry.llm_response ?? entry.response ?? "",
      user_name: entry.user_name ?? "Imported",
  }));

  const remote = await requestRuntimeOrFallback(
    "control_panel.import_memories",
    { entries: normalizedEntries },
    30_000,
  );
  if (remote) return remote;

  for (const entry of normalizedEntries) {
    state.memories.unshift({
      id: `local-${Date.now()}-${Math.random().toString(16).slice(2)}`,
      kind: "short",
      source: "control_panel",
      created_at: new Date().toISOString(),
      ...entry,
    });
  }

  return clone(state);
}

export const panelBackend = {
  getInitialPanelState,
  saveSetting,
  saveToggle,
  runControlAction,
  saveContext,
  saveTemporaryContext,
  addBlacklistWord,
  deleteBlacklistWord,
  updateGameStatus,
  runPanelAction,
  onPanelStateUpdate,
  searchMemories,
  saveMemoryEntry,
  deleteMemoryEntry,
  clearShortTermMemories,
  importMemories,
};

window.agathaPanelBackend = panelBackend;
