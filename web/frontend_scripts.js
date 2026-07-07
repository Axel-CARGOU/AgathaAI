import {
  addBlacklistWord,
  clearShortTermMemories,
  deleteBlacklistWord,
  deleteMemoryEntry,
  getInitialPanelState,
  importMemories,
  onPanelStateUpdate,
  runControlAction,
  runPanelAction,
  saveContext,
  saveMemoryEntry,
  saveSetting,
  saveTemporaryContext,
  saveToggle,
  searchMemories,
  sendInternalCommand,
  updateGameStatus,
} from "./backend_scripts.js";

const ui = {
  state: null,
  view: "main",
  settingsTab: "general",
  promptTab: "main",
  selectedMemoryId: null,
  toastTimer: null,
  editingSettings: new Set(),
};

const selectors = {
  currentMessage: "#current-message",
  streamPreview: "#stream-preview",
  commandLanguage: "#command-language",
  commandType: "#command-type",
  commandContext: "#command-context",
  commandInstruction: "#command-instruction",
  commandSend: "#command-send",
  twitchFeed: "#twitch-feed",
  settingsList: "#settings-list",
  memoryList: "#memory-list",
  memoryDetail: "#memory-detail",
  memorySearch: "#memory-search",
  memoryUserPrompt: "#memory-user-prompt",
  memoryLlmResponse: "#memory-llm-response",
  memoryImportFile: "#memory-import-file",
  blacklistList: "#blacklist-list",
  blacklistWord: "#blacklist-word",
  moderationTest: "#moderation-test",
  matchPreview: "#match-preview",
  promptEditor: "#prompt-editor",
  seenByLlm: "#seen-by-llm",
  tempContext: "#temp-context",
  toast: "#toast",
  engineGraph: "#engine-graph",
  engineGraphEmpty: "#engine-graph-empty",
  engineGraphOpen: "#engine-graph-open",
  requestGraph: "#request-graph",
  requestGraphEmpty: "#request-graph-empty",
  requestGraphOpen: "#request-graph-open",
  nvidiaGraph: "#nvidia-graph",
  nvidiaGraphEmpty: "#nvidia-graph-empty",
  nvidiaGraphOpen: "#nvidia-graph-open",
  metricsFiles: "#metrics-files",
};

const controlLabels = {
  bot: ["Disable Bot", "Enable Bot", "botDisabled"],
  llm: ["Disable LLM", "Enable LLM", "llmDisabled"],
  tts: ["Disable TTS", "Enable TTS", "ttsDisabled"],
  mute: ["Mute", "Unmute", "muted"],
  vtuber: ["Disable VTuber", "Enable VTuber", "vtuberDisabled"],
};

const decorativeActions = new Set([
  "song-play",
  "song-stop",
  "song-pause",
  "song-resume",
  "vts-refresh",
  "vts-hotkey",
  "prop-spawn",
  "prop-despawn",
  "vts-chatting",
  "vts-fullscreen",
  "vts-reacting",
]);

const commandPrefills = {
  stream_intro: {
    type: "stream_intro",
    language: "FR",
    context: "Lancement du live Twitch, Axel est en vocal, jeu prévu: Total War: Warhammer III.",
    instruction: "Produis directement l'introduction à l'antenne. Ne réponds pas à la commande.",
  },
  stream_outro: {
    type: "stream_outro",
    language: "FR",
    context: "Fin du live Twitch, Axel est en vocal, remercie les viewers et annonce que le stream se termine.",
    instruction: "Produis directement l'outro du stream. Ne réponds pas à la commande.",
  },
  tweet_shitpost: {
    type: "tweet_shitpost",
    language: "FR",
    context: "Publication Twitter/X random et hors contexte, ton naturel d'Agatha, humour sec.",
    instruction: "Produis uniquement le texte du tweet. Ne réponds pas à la commande.",
  },
  tweet_stream_schedule: {
    type: "tweet_stream_schedule",
    language: "FR",
    context: "Annonce Twitter/X de l'emploi du temps de stream de Axel.",
    instruction: "Produis uniquement le texte du tweet de planning. Ne réponds pas à la commande.",
  },
};

const moderationCharacterEquivalents = Object.fromEntries([
  ["aàáâäãå", "aàáâäãå4@"],
  ["cç", "cç"],
  ["eéèêë", "eéèêë3"],
  ["iíìîï", "iíìîï1!|"],
  ["oóòôöõ", "oóòôöõ0"],
  ["s", "s5$"],
  ["uúùûü", "uúùûü"],
].flatMap(([characters, equivalents]) => (
  Array.from(characters, (character) => [character, equivalents])
)));

document.addEventListener("DOMContentLoaded", init);

async function init() {
  ui.state = await getInitialPanelState();
  ui.selectedMemoryId = ui.state.memories[0]?.id ?? null;
  onPanelStateUpdate((nextState) => {
    ui.state = nextState;
    if (ui.state.updateScope === "realtime") {
      renderRealtime();
      return;
    }
    render();
  });
  syncInitialRoute();
  bindEvents();
  render();
}

function bindEvents() {
  document.querySelectorAll("[data-view-link]").forEach((button) => {
    button.addEventListener("click", () => setView(button.dataset.viewLink));
  });

  window.addEventListener("hashchange", syncInitialRoute);

  document.querySelectorAll("[data-settings-tab]").forEach((button) => {
    button.addEventListener("click", () => {
      ui.settingsTab = button.dataset.settingsTab;
      renderSettings();
      updateActiveTabs("[data-settings-tab]", ui.settingsTab);
    });
  });

  document.querySelectorAll("[data-prompt-tab]").forEach((button) => {
    button.addEventListener("click", () => {
      ui.promptTab = button.dataset.promptTab;
      renderPromptEditor();
      updateActiveTabs("[data-prompt-tab]", ui.promptTab);
    });
  });

  document.querySelectorAll("[data-toggle]").forEach((input) => {
    input.addEventListener("change", async () => {
      try {
        ui.state = await saveToggle(input.dataset.toggle, input.checked);
        render();
        showToast(`${humanize(input.dataset.toggle)} saved`);
      } catch (error) {
        renderToggles();
        showToast(error.message);
      }
    });
  });

  document.querySelectorAll("[data-state-action]").forEach((button) => {
    button.addEventListener("click", async () => {
      const previousLabel = button.textContent;
      try {
        ui.state = await runControlAction(button.dataset.stateAction);
        render();
        showToast(`${previousLabel} requested`);
      } catch (error) {
        render();
        showToast(error.message);
      }
    });
  });

  document.querySelectorAll("[data-action]").forEach((button) => {
    button.addEventListener("click", () => handleAction(button.dataset.action));
  });

  document.querySelectorAll("[data-command-prefill]").forEach((button) => {
    button.addEventListener("click", () => applyCommandPrefill(button.dataset.commandPrefill));
  });

  document.querySelectorAll("[data-game-action]").forEach((button) => {
    button.addEventListener("click", () => handleGameAction(button.dataset.gameAction));
  });

  document.querySelector(selectors.moderationTest)?.addEventListener("input", renderMatchPreview);

  document.querySelector(selectors.memoryImportFile)?.addEventListener("change", handleMemoryImportFile);
}

function syncInitialRoute() {
  const route = window.location.hash.replace("#", "") || "main";
  const knownViews = ["main", "memory", "moderation", "vtube", "lobotomy", "games", "metrics"];
  setView(knownViews.includes(route) ? route : "main", { replaceHash: true });
}

function setView(view, options = {}) {
  ui.view = view;

  document.querySelectorAll("[data-view]").forEach((section) => {
    section.classList.toggle("is-active", section.dataset.view === view);
  });

  document.querySelectorAll("[data-view-link]").forEach((link) => {
    link.classList.toggle("is-active", link.dataset.viewLink === view);
  });

  if (!options.replaceHash) {
    window.location.hash = view;
  }
}

function render() {
  renderRealtime();
  renderToggles();
  renderSettings();
  renderMemories();
  renderBlacklist();
  renderPromptEditor();
  renderTemporaryContext();
  renderGameStatus();
  renderMetrics();
}

function renderRealtime() {
  renderBindings();
  renderSignals();
  renderControlButtons();
  renderTwitchChat();
  renderCommands();
}

function renderBindings() {
  setValue(selectors.currentMessage, ui.state.currentMessage.content);
  setValue(selectors.streamPreview, ui.state.currentMessage.streamPreview);
  bindText("currentSource", ui.state.currentMessage.source);
  bindText("streamMode", ui.state.currentMessage.streamMode);
  bindText("settingsStatus", ui.state.backendConnected ? "Runtime" : "Local");
  bindText("memoryStatus", ui.state.runtime.modulesActive.rag ? "RAG ready" : "RAG offline");
}

function renderSignals() {
  Object.entries(ui.state.runtime.signals).forEach(([key, value]) => {
    document.querySelector(`[data-signal="${key}"]`)?.classList.toggle("is-active", Boolean(value));
  });
}

function renderToggles() {
  Object.entries(ui.state.toggles).forEach(([key, value]) => {
    const input = document.querySelector(`[data-toggle="${key}"]`);
    if (input) {
      input.checked = Boolean(value);
    }
  });
}

function renderControlButtons() {
  Object.entries(controlLabels).forEach(([action, labels]) => {
    const [offLabel, onLabel, stateKey] = labels;
    const button = document.querySelector(`[data-state-action="${action}"]`);
    const active = Boolean(ui.state.runtime.controls[stateKey]);
    if (!button) return;

    button.textContent = active ? onLabel : offLabel;
    button.classList.toggle("danger", active);
    button.classList.toggle("secondary", !active);
  });
}

function renderTwitchChat() {
  const feed = document.querySelector(selectors.twitchFeed);
  const messages = ui.state.twitch.messages ?? [];
  bindText("twitchCount", `${messages.length} messages`);
  if (!feed) return;

  if (!messages.length) {
    feed.innerHTML = `<div class="chat-empty">Twitch messages</div>`;
    return;
  }

  feed.innerHTML = messages
    .map((message) => {
      return `
        <div class="chat-message">
          <strong>${escapeHtml(message.user_name ?? "viewer")}</strong>
          <span>${escapeHtml(message.content ?? "")}</span>
        </div>
      `;
    })
    .join("");
}

function renderCommands() {
  const inVoice = Boolean(ui.state.discord?.inVoice);
  const blocked = !inVoice
    || !Boolean(ui.state.runtime.modulesActive.bot)
    || !Boolean(ui.state.runtime.modulesActive.llm)
    || !Boolean(ui.state.runtime.modulesActive.tts)
    || Boolean(ui.state.runtime.controls.botDisabled)
    || Boolean(ui.state.runtime.controls.llmDisabled)
    || Boolean(ui.state.runtime.controls.ttsDisabled);
  const sendButton = document.querySelector(selectors.commandSend);

  if (sendButton) {
    sendButton.disabled = blocked;
  }

  let status = "Discord VC required";
  if (inVoice) status = "Voice ready";
  if (inVoice && blocked) status = "Module disabled";
  bindText("commandStatus", status);
}

function renderSettings() {
  const list = document.querySelector(selectors.settingsList);
  if (!list) return;

  const settings = ui.state.settings[ui.settingsTab] ?? [];
  list.innerHTML = settings.map(renderSettingRow).join("");

  list.querySelectorAll("[data-setting-save]").forEach((button) => {
    button.addEventListener("click", async () => {
      const { group, key } = button.dataset;
      const row = button.closest(".setting-row");
      const input = row?.querySelector("[data-setting-input]");
      if (!input) return;

      const editKey = settingEditKey(group, key);
      if (!ui.editingSettings.has(editKey)) {
        ui.editingSettings.add(editKey);
        renderSettings();
        focusSettingInput(editKey);
        return;
      }

      if (input.disabled) return;
      const nextValue = input.type === "checkbox" ? input.checked : input.value;
      try {
        ui.state = await saveSetting(group, key, nextValue);
        ui.editingSettings.delete(editKey);
        showToast(`${key} saved`);
        render();
      } catch (error) {
        render();
        showToast(error.message);
      }
    });
  });

  list.querySelectorAll("[data-setting-cancel]").forEach((button) => {
    button.addEventListener("click", () => {
      ui.editingSettings.delete(settingEditKey(button.dataset.group, button.dataset.key));
      renderSettings();
    });
  });

  list.querySelectorAll("[data-setting-input]").forEach((input) => {
    input.addEventListener("change", updateBooleanSettingPreview);
  });
}

function renderSettingRow(setting) {
  const locked = isSettingLocked(setting);
  const editKey = settingEditKey(ui.settingsTab, setting.key);
  const isEditing = ui.editingSettings.has(editKey);
  const disabled = locked || !setting.editable || !isEditing;
  const input = setting.type === "boolean"
    ? `
      <label class="toggle-line">
        <span class="toggle">
          <input data-setting-input type="checkbox" ${setting.value ? "checked" : ""} ${disabled ? "disabled" : ""} />
          <i></i>
        </span>
        <span data-setting-bool-value>${setting.value ? "true" : "false"}</span>
      </label>
    `
    : `
      <input
        class="input"
        data-setting-input
        type="${setting.type === "number" ? "number" : "text"}"
        value="${escapeAttribute(setting.value)}"
        ${setting.step ? `step="${escapeAttribute(setting.step)}"` : ""}
        ${disabled ? "disabled" : ""}
      />
    `;

  const meta = locked
    ? setting.lockedReason || "Module active"
    : setting.editable
      ? "Hot setting"
      : setting.lockedReason || "Read only";
  const rowClasses = ["setting-row"];
  if (locked) rowClasses.push("is-locked");
  if (!setting.editable) rowClasses.push("is-readonly");
  if (isEditing) rowClasses.push("is-editing");

  return `
    <div class="${rowClasses.join(" ")}">
      <label class="setting-label">
        ${escapeHtml(setting.label)}
        <span class="setting-meta">${escapeHtml(setting.key)} / ${escapeHtml(meta)}</span>
      </label>
      ${input}
      <div class="setting-actions">
        <button
          class="btn secondary"
          type="button"
          data-setting-save
          data-group="${escapeAttribute(ui.settingsTab)}"
          data-key="${escapeAttribute(setting.key)}"
          ${locked || !setting.editable ? "disabled" : ""}
        >${isEditing ? "Save" : "Edit"}</button>
        ${isEditing ? `
          <button
            class="btn secondary"
            type="button"
            data-setting-cancel
            data-group="${escapeAttribute(ui.settingsTab)}"
            data-key="${escapeAttribute(setting.key)}"
          >Cancel</button>
        ` : ""}
      </div>
    </div>
  `;
}

function isSettingLocked(setting) {
  if (!setting.lockedWhen) return false;
  return Boolean(ui.state.runtime.modulesActive[setting.lockedWhen]);
}

function settingEditKey(group, key) {
  return `${group}:${key}`;
}

function focusSettingInput(editKey) {
  window.requestAnimationFrame(() => {
    const [group, key] = editKey.split(":");
    const selector = `[data-setting-save][data-group="${escapeAttribute(group)}"][data-key="${escapeAttribute(key)}"]`;
    const row = document.querySelector(selector)?.closest(".setting-row");
    row?.querySelector("[data-setting-input]")?.focus();
  });
}

function updateBooleanSettingPreview(event) {
  if (event.target.type !== "checkbox") return;
  const label = event.target.closest(".toggle-line")?.querySelector("[data-setting-bool-value]");
  if (label) {
    label.textContent = event.target.checked ? "true" : "false";
  }
}

function renderMemories() {
  const list = document.querySelector(selectors.memoryList);
  if (!list) return;

  list.innerHTML = ui.state.memories.map((memory) => `
    <div class="memory-item">
      <span class="memory-kind ${memory.kind === "long" ? "long" : ""}"></span>
      <button class="memory-open" type="button" data-memory-open="${escapeAttribute(memory.id)}">
        <strong>${escapeHtml(memory.user_prompt)}</strong>
        <span>${escapeHtml(memory.created_at)} / ${escapeHtml(memory.source)}</span>
      </button>
      <button class="btn icon secondary" type="button" data-memory-edit="${escapeAttribute(memory.id)}" aria-label="Edit memory">Edit</button>
      <button class="btn icon danger" type="button" data-memory-delete="${escapeAttribute(memory.id)}" aria-label="Delete memory">Delete</button>
    </div>
  `).join("");

  list.querySelectorAll("[data-memory-open]").forEach((button) => {
    button.addEventListener("click", () => {
      ui.selectedMemoryId = button.dataset.memoryOpen;
      renderMemoryDetail();
    });
  });

  list.querySelectorAll("[data-memory-edit]").forEach((button) => {
    button.addEventListener("click", () => {
      ui.selectedMemoryId = button.dataset.memoryEdit;
      renderMemoryDetail();
      showToast("Memory loaded for editing");
    });
  });

  list.querySelectorAll("[data-memory-delete]").forEach((button) => {
    button.addEventListener("click", async () => {
      ui.selectedMemoryId = button.dataset.memoryDelete;
      ui.state = await deleteMemoryEntry(ui.selectedMemoryId);
      ui.selectedMemoryId = ui.state.memories[0]?.id ?? null;
      renderMemories();
      showToast("Memory deleted");
    });
  });

  renderMemoryDetail();
}

function renderMemoryDetail() {
  const detail = document.querySelector(selectors.memoryDetail);
  if (!detail) return;

  const memory = ui.state.memories.find((item) => item.id === ui.selectedMemoryId);
  if (!memory) {
    detail.innerHTML = `<div class="empty-detail">Select a memory entry</div>`;
    setValue(selectors.memoryUserPrompt, "", { preserveFocus: true });
    setValue(selectors.memoryLlmResponse, "", { preserveFocus: true });
    return;
  }

  detail.textContent = [
    `ID: ${memory.id}`,
    `Kind: ${memory.kind}`,
    `Source: ${memory.source}`,
    `Created: ${memory.created_at}`,
    `User: ${memory.user_name}`,
    "",
    "User prompt:",
    memory.user_prompt,
    "",
    "Agatha response:",
    memory.llm_response,
  ].join("\n");

  setValue(selectors.memoryUserPrompt, memory.user_prompt ?? "", { preserveFocus: true });
  setValue(selectors.memoryLlmResponse, memory.llm_response ?? "", { preserveFocus: true });
}

async function handleMemoryImportFile(event) {
  const file = event.target.files?.[0];
  event.target.value = "";
  if (!file) return;

  try {
    const parsed = JSON.parse(await file.text());
    const entries = Array.isArray(parsed) ? parsed : parsed.memories;
    if (!Array.isArray(entries)) {
      throw new Error("JSON must be an array or { memories: [...] }");
    }

    ui.state = await importMemories(entries);
    ui.selectedMemoryId = ui.state.memories[0]?.id ?? null;
    renderMemories();
    showToast("Memory import complete");
  } catch (error) {
    showToast(error.message);
  }
}

function renderBlacklist() {
  const list = document.querySelector(selectors.blacklistList);
  if (!list) return;

  list.innerHTML = ui.state.moderation.blacklist.map((word) => `
    <div class="blacklist-item">
      <span class="blacklist-word">${escapeHtml(word)}</span>
      <button class="btn icon danger" type="button" data-blacklist-delete="${escapeAttribute(word)}">Delete</button>
    </div>
  `).join("");

  list.querySelectorAll("[data-blacklist-delete]").forEach((button) => {
    button.addEventListener("click", async () => {
      ui.state.moderation.blacklist = await deleteBlacklistWord(button.dataset.blacklistDelete);
      renderBlacklist();
      renderMatchPreview();
      showToast("Blacklist word removed");
    });
  });

  renderMatchPreview();
}

function renderPromptEditor() {
  setValue(selectors.promptEditor, ui.state.contexts[ui.promptTab] ?? "", { preserveFocus: true });
  setValue(selectors.seenByLlm, ui.state.seenByLlm ?? "");
}

function renderTemporaryContext() {
  setValue(selectors.tempContext, ui.state.temporaryContext ?? "", { preserveFocus: true });
}

function renderGameStatus() {
  bindText("lichessStatus", ui.state.games.lichess.status);
}

function renderMetrics() {
  const metrics = ui.state.metrics ?? {};
  bindText("metricsRun", metrics.frozenTimestamp ? `Run ${metrics.frozenTimestamp}` : "No run");
  bindText("metricsDaemon", formatMetricsDaemon(metrics.autoRenderer));
  renderMetricImage("engine", metrics.engineGraph, "engineGraphUpdated");
  renderMetricImage("request", metrics.requestGraph, "requestGraphUpdated");
  renderMetricImage("nvidia", metrics.nvidiaGraph, "nvidiaGraphUpdated");
  renderMetricFiles(metrics.files ?? []);
}

function formatMetricsDaemon(autoRenderer) {
  const status = autoRenderer?.status ?? "Stopped";
  const interval = autoRenderer?.intervalSeconds ?? 5;
  const pid = autoRenderer?.pid ? ` / pid ${autoRenderer.pid}` : "";
  return `Renderer ${status} / ${interval}s${pid}`;
}

function renderMetricImage(kind, file, updatedBind) {
  const image = document.querySelector(selectors[`${kind}Graph`]);
  const open = document.querySelector(selectors[`${kind}GraphOpen`]);
  const frame = image?.closest(".metric-image-frame");
  const exists = Boolean(file?.exists && file?.url);

  if (image) {
    image.src = exists ? file.url : "";
  }

  frame?.classList.toggle("has-image", exists);

  if (open) {
    open.href = exists ? file.url : "#";
    open.setAttribute("aria-disabled", exists ? "false" : "true");
  }

  bindText(updatedBind, exists ? `${file.name} / ${file.updatedAt}` : "Waiting for graph");
}

function renderMetricFiles(files) {
  const list = document.querySelector(selectors.metricsFiles);
  if (!list) return;

  if (!files.length) {
    list.innerHTML = `<div class="chat-empty">No metrics files yet</div>`;
    return;
  }

  list.innerHTML = files.map((file) => `
    <div class="metric-file">
      <strong>${escapeHtml(file.name)}</strong>
      <span>${escapeHtml(file.updatedAt || "not generated")}</span>
      ${file.url ? `<a class="btn secondary" href="${escapeAttribute(file.url)}" target="_blank" rel="noreferrer">Open</a>` : `<span></span>`}
    </div>
  `).join("");
}

async function handleAction(action) {
  if (action === "blacklist-add") {
    const input = document.querySelector(selectors.blacklistWord);
    const word = input?.value.trim();
    if (!word) return;
    ui.state.moderation.blacklist = await addBlacklistWord(word);
    input.value = "";
    renderBlacklist();
    showToast("Blacklist word added");
    return;
  }

  if (action === "memory-search") {
    const input = document.querySelector(selectors.memorySearch);
    ui.state = await searchMemories(input?.value ?? "");
    if (!ui.state.memories.some((memory) => memory.id === ui.selectedMemoryId)) {
      ui.selectedMemoryId = ui.state.memories[0]?.id ?? null;
    }
    renderMemories();
    showToast("Memory search complete");
    return;
  }

  if (action === "memory-save") {
    const prompt = document.querySelector(selectors.memoryUserPrompt)?.value ?? "";
    const response = document.querySelector(selectors.memoryLlmResponse)?.value ?? "";
    const selected = ui.state.memories.find((memory) => memory.id === ui.selectedMemoryId);
    ui.state = await saveMemoryEntry({
      id: selected?.id,
      user_name: selected?.user_name ?? "Control Panel",
      user_prompt: prompt,
      llm_response: response,
    });
    ui.selectedMemoryId = selected?.id ?? ui.state.memories[0]?.id ?? null;
    renderMemories();
    showToast("Memory saved");
    return;
  }

  if (action === "memory-edit") {
    if (!ui.selectedMemoryId) return;
    renderMemoryDetail();
    document.querySelector(selectors.memoryUserPrompt)?.focus();
    showToast("Memory loaded for editing");
    return;
  }

  if (action === "memory-new") {
    ui.selectedMemoryId = null;
    setValue(selectors.memoryUserPrompt, "");
    setValue(selectors.memoryLlmResponse, "");
    renderMemoryDetail();
    showToast("New memory draft");
    return;
  }

  if (action === "memory-delete") {
    if (!ui.selectedMemoryId) return;
    ui.state = await deleteMemoryEntry(ui.selectedMemoryId);
    ui.selectedMemoryId = ui.state.memories[0]?.id ?? null;
    renderMemories();
    showToast("Memory deleted");
    return;
  }

  if (action === "memory-clear-short") {
    ui.state = await clearShortTermMemories();
    ui.selectedMemoryId = ui.state.memories[0]?.id ?? null;
    renderMemories();
    showToast("Short-term memory cleared");
    return;
  }

  if (action === "memory-export") {
    downloadJson("agathaai_memories.json", ui.state.memories);
    showToast("Memory export prepared");
    return;
  }

  if (action === "memory-import") {
    document.querySelector(selectors.memoryImportFile)?.click();
    return;
  }

  if (action === "command-send") {
    try {
      ui.state = await sendInternalCommand(readCommandForm());
      render();
      showToast("Internal command sent");
    } catch (error) {
      render();
      showToast(error.message);
    }
    return;
  }

  if (decorativeActions.has(action)) {
    showToast(`${humanize(action)} is decorative for now`);
    return;
  }

  if (action === "prompt-save") {
    const editor = document.querySelector(selectors.promptEditor);
    try {
      ui.state = await saveContext(ui.promptTab, editor?.value ?? "");
      renderPromptEditor();
      showToast(`${ui.promptTab} context saved`);
    } catch (error) {
      showToast(error.message);
    }
    return;
  }

  if (action === "temp-context-apply") {
    const textarea = document.querySelector(selectors.tempContext);
    ui.state = await saveTemporaryContext(textarea?.value ?? "");
    renderTemporaryContext();
    renderPromptEditor();
    showToast("Temporary context applied locally");
    return;
  }

  if (action === "temp-context-clear") {
    ui.state = await saveTemporaryContext("");
    render();
    showToast("Temporary context cleared");
    return;
  }

  if (action === "clear-current") {
    ui.state = await runPanelAction(action);
    render();
    showToast("Current message cleared");
    return;
  }

  try {
    ui.state = await runPanelAction(action);
    render();
    showToast(`${humanize(action)} queued`);
  } catch (error) {
    render();
    showToast(error.message);
  }
}

function applyCommandPrefill(key) {
  const template = commandPrefills[key];
  if (!template) return;

  setValue(selectors.commandLanguage, template.language);
  setValue(selectors.commandType, template.type);
  setValue(selectors.commandContext, template.context);
  setValue(selectors.commandInstruction, template.instruction);
}

function readCommandForm() {
  return {
    language: document.querySelector(selectors.commandLanguage)?.value ?? "FR",
    type: document.querySelector(selectors.commandType)?.value ?? "",
    context: document.querySelector(selectors.commandContext)?.value ?? "",
    instruction: document.querySelector(selectors.commandInstruction)?.value ?? "",
  };
}

async function handleGameAction(action) {
  if (action === "lichess-start") {
    ui.state.games.lichess = await updateGameStatus("lichess", "Running");
    showToast("Lichess launch requested");
  }

  if (action === "lichess-stop") {
    ui.state.games.lichess = await updateGameStatus("lichess", "Stopped");
    showToast("Lichess close requested");
  }

  renderGameStatus();
}

function renderMatchPreview() {
  const preview = document.querySelector(selectors.matchPreview);
  const textarea = document.querySelector(selectors.moderationTest);
  if (!preview || !textarea) return;

  const text = textarea.value;
  const matches = ui.state.moderation.blacklist.filter((word) => matchesWholeWord(text, word));

  preview.classList.toggle("has-match", matches.length > 0);
  preview.textContent = matches.length ? `Matched: ${matches.join(", ")}` : "No match";
}

function matchesWholeWord(text, word) {
  if (!text || !word) return false;
  const characters = Array.from(word);
  let body = characters.map(moderationCharacterPattern).join("");
  const lastCharacter = characters.at(-1);

  if (/\p{L}/u.test(lastCharacter) && lastCharacter.toLocaleLowerCase() !== "s") {
    body += `${moderationCharacterPattern("s")}?`;
  }

  const pattern = new RegExp(`(^|[^\\p{L}\\p{N}_])(${body})(?=$|[^\\p{L}\\p{N}_])`, "iu");
  return pattern.test(text);
}

function moderationCharacterPattern(character) {
  const equivalents = moderationCharacterEquivalents[character.toLocaleLowerCase()];
  return equivalents ? `[${escapeRegExp(equivalents)}]` : escapeRegExp(character);
}

function updateActiveTabs(selector, activeValue) {
  document.querySelectorAll(selector).forEach((button) => {
    const value = button.dataset.settingsTab || button.dataset.promptTab;
    button.classList.toggle("is-active", value === activeValue);
  });
}

function bindText(name, text) {
  document.querySelectorAll(`[data-bind="${name}"]`).forEach((element) => {
    element.textContent = text;
  });
}

function setValue(selector, value, options = {}) {
  const element = document.querySelector(selector);
  if (element) {
    if (options.preserveFocus && element === document.activeElement) return;
    element.value = value ?? "";
  }
}

function showToast(message) {
  const toast = document.querySelector(selectors.toast);
  if (!toast) return;

  window.clearTimeout(ui.toastTimer);
  toast.textContent = message;
  toast.classList.add("is-visible");
  ui.toastTimer = window.setTimeout(() => {
    toast.classList.remove("is-visible");
  }, 2400);
}

function downloadJson(filename, data) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function humanize(value) {
  return String(value ?? "")
    .replace(/[-_]/g, " ")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttribute(value) {
  return escapeHtml(value).replaceAll("\n", "&#10;");
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
