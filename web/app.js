"use strict";

const $ = (id) => document.getElementById(id);
const TARGET_RATE = 16000;

const ui = {
  dot: $("dot"), phase: $("phase"), timing: $("timing"), talk: $("talk"), stop: $("stop"),
  scope: $("scope"), log: $("log"), voice: $("voice"), model: $("model"), speed: $("speed"),
  minWords: $("minWords"), debounce: $("debounce"), gap: $("gap"), preview: $("preview"),
  reset: $("reset"), micLevel: $("micLevel"), outLevel: $("outLevel"), latency: $("latency"),
  info: $("sttInfo"), mic: $("mic"), alert: $("alert"), micMode: $("micMode"),
  deviceRow: $("deviceRow"),
};

let state = {};
let playCtx = null;

let recording = [];
let recordingChunks = 0;
let listening = false;      // actively capturing the microphone
let fromButton = false;     // this capture started on the big button
let busy = false;           // a turn is in flight
let speaking = false;
let micLevel = 0;
let outLevel = 0;

let turnId = 0;
let turnStartedAt = 0;
let firstAudioAt = 0;
let pending = [];
let history = [];

/* ---------------- scope ---------------- */

const scope = ui.scope.getContext("2d");
const bars = new Array(56).fill(0);

function draw() {
  const w = ui.scope.width, h = ui.scope.height, mid = h / 2;
  scope.clearRect(0, 0, w, h);

  const active = listening || speaking;
  const level = listening ? micLevel : outLevel;
  const hue = listening ? "251,113,133" : "94,234,212";

  scope.fillStyle = active ? `rgba(${hue},0.08)` : "rgba(139,149,165,0.04)";
  scope.beginPath();
  scope.roundRect(8, 8, w - 16, h - 16, 12);
  scope.fill();

  const step = (w - 52) / bars.length;
  for (let i = 0; i < bars.length; i++) {
    const wave = 0.35 + 0.65 * Math.abs(Math.sin(i * 0.62 + performance.now() / 190));
    const target = active ? level * wave : 0.015;
    bars[i] += (target - bars[i]) * (target > bars[i] ? 0.5 : 0.1);
    const bh = Math.max(2, bars[i] * (h - 70));
    scope.fillStyle = `rgba(${hue},${0.2 + Math.min(0.8, bars[i] * 1.4)})`;
    scope.beginPath();
    scope.roundRect(26 + i * step, mid - bh / 2, step - 5, bh, 3);
    scope.fill();
  }

  scope.fillStyle = "rgba(200,210,222,0.8)";
  scope.font = "12px ui-monospace, Consolas, monospace";
  const caption = listening ? "listening to your microphone"
    : speaking ? "the model is talking"
    : "ready - hold the button or the space bar";
  scope.fillText(caption, 26, 34);

  requestAnimationFrame(draw);
}
requestAnimationFrame(draw);

let lastPhase = "";

function setPhase(phase) {
  if (phase !== lastPhase) {
    lastPhase = phase;
    log("info", "phase: " + phase);
  }
  ui.phase.textContent = phase;
  ui.dot.className = "dot" + (phase === "idle" ? "" : " " + phase);
  if (phase !== "idle") ui.timing.textContent = phase;
}

/* ---------------- playback ---------------- */

async function audioRunning() {
  if (!audioCtx) return false;
  if (audioCtx.state !== "running") {
    try { await audioCtx.resume(); } catch (e) { log("warn", "resume: " + e); }
  }
  if (audioCtx.state !== "running") {
    alertBox("the browser blocked audio - click once anywhere on the page, then talk");
    log("warn", "audio context is " + audioCtx.state);
    return false;
  }
  return true;
}

async function ensurePlay() {
  if (!playCtx) playCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (playCtx.state !== "running") {
    try { await playCtx.resume(); } catch (e) { log("warn", "play resume: " + e); }
  }
  if (playCtx.state === "suspended") {
    alertBox("sound is blocked - click once anywhere on the page to allow playback");
  }
  return playCtx;
}

function toArrayBuffer(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes.buffer;
}

function outputMeter(analyser) {
  const data = new Uint8Array(analyser.fftSize);
  const tick = setInterval(() => {
    analyser.getByteTimeDomainData(data);
    let peak = 0;
    for (let i = 0; i < data.length; i++) {
      peak = Math.max(peak, Math.abs(data[i] - 128) / 128);
    }
    outLevel = Math.max(peak, outLevel * 0.8);
    ui.outLevel.textContent = outLevel.toFixed(2);
  }, 55);
  return () => clearInterval(tick);
}

async function playBuffer(buffer, onEnd) {
  const ctx = await ensurePlay();
  const epoch = playbackEpoch;
  if (epoch !== playbackEpoch) return;  // stopped while we were decoding
  if (ctx.state !== "running") {
    // A suspended context never advances its clock, so onended would never fire: the
    // output meter would stay lit and the server would wait forever for playback-done.
    log("warn", "skipping audio, playback is " + ctx.state);
    onEnd();
    return;
  }
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 1024;
  const node = ctx.createBufferSource();
  node.buffer = buffer;
  node.connect(analyser).connect(ctx.destination);

  speaking = true;
  setPhase("speaking");
  const stopMeter = outputMeter(analyser);
  pending.push(node);

  node.onended = () => {
    stopMeter();
    pending = pending.filter((n) => n !== node);
    if (!pending.length && epoch === playbackEpoch) onEnd();
  };
  node.start();
  if (!firstAudioAt) {
    firstAudioAt = performance.now();
    ui.latency.textContent = ((firstAudioAt - turnStartedAt) / 1000).toFixed(2) + "s";
  }
}

async function playChunk(b64) {
  const ctx = await ensurePlay();
  const buffer = await ctx.decodeAudioData(toArrayBuffer(b64));
  return playBuffer(buffer, playbackFinished);
}

function playbackFinished() {
  speaking = false;
  outLevel = 0;
  ui.outLevel.textContent = "0.00";
  // Only a real turn has a turn id to acknowledge; a preview must not clear the
  // server's playback flag for whatever reply happens to be running.
  if (turnId) post("/api/playback-done", { turn: turnId });
  if (!busy) setPhase("idle");
}

/* Playback epoch. Playback runs behind the network: Kokoro makes ~2s of audio per ~0.4s
   of synthesis, so 1-2 chunks are usually still queued when the user hits stop. Bumping the
   epoch on stop makes those queued steps no-ops, so the tail cannot restart playing (and
   re-declare "speaking") after the user asked it to stop. */
let playbackEpoch = 0;

function stopAudio() {
  playbackEpoch += 1;
  for (const node of pending) {
    try { node.stop(); } catch (e) { /* already finished */ }
  }
  pending = [];
  speaking = false;
  outLevel = 0;
  ui.outLevel.textContent = "0.00";
}

/* ---------------- reporting ---------------- */

function log(level, msg) {
  fetch("/api/client-log", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ level, msg: String(msg) }),
  }).catch(() => {});
}

function alertBox(text, ok) {
  ui.alert.textContent = text || "";
  ui.alert.className = "alert" + (ok ? " ok" : "");
}

window.addEventListener("error", (e) => log("error", "js: " + e.message + " @ " + e.filename + ":" + e.lineno));
window.addEventListener("unhandledrejection", (e) => log("error", "promise: " + (e.reason && e.reason.message ? e.reason.message : e.reason)));

/* ---------------- microphone ---------------- */

let micStream = null;
let micDevice = "";
let audioCtx = null;
let recorder = null;
let peakSeen = 0;
let floor = 0.004;

function setMeter(level) {
  micLevel = level;
  ui.micLevel.textContent = level.toFixed(2);
}

async function micOptions(deviceId) {
  return {
    audio: deviceId
      ? { deviceId: { exact: deviceId }, echoCancellation: true, noiseSuppression: true,
          autoGainControl: true }
      : { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    video: false,
  };
}

async function openMic(deviceId) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("getUserMedia unavailable (needs http://127.0.0.1 or localhost)");
  }
  if (micStream) micStream.getTracks().forEach((track) => track.stop());

  log("info", "requesting microphone" + (deviceId ? " " + deviceId : " (default)"));
  micStream = await navigator.mediaDevices.getUserMedia(await micOptions(deviceId));

  const track = micStream.getAudioTracks()[0];
  const label = track.label || "(unnamed input)";
  log("info", "microphone open: " + label + " | " +
    JSON.stringify(track.getSettings ? track.getSettings() : {}));

  if (!audioCtx || audioCtx.state === "closed") {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (audioCtx.state === "suspended") await audioCtx.resume();

  const source = audioCtx.createMediaStreamSource(micStream);
  const analyser = audioCtx.createAnalyser();
  analyser.fftSize = 1024;
  source.connect(analyser);
  const bins = new Uint8Array(analyser.fftSize);

  if (recorder) recorder.disconnect();
  recorder = audioCtx.createScriptProcessor(2048, 1, 1);
  let quiet = 0;

  recorder.onaudioprocess = (event) => {
    analyser.getByteTimeDomainData(bins);
    let peak = 0;
    for (let i = 0; i < bins.length; i++) peak = Math.max(peak, Math.abs(bins[i] - 128) / 128);
    setMeter(peak);
    if (listening) peakSeen = Math.max(peakSeen, peak);
    /* The floor has to fall when the room is quiet, otherwise it creeps up towards the
       voice (x1.02 per block) and, once it reaches 3x the speech level, treats a talking
       person as silence. Replayed against real speech that cut the user off at 6.3s. */
    floor = peak < floor ? peak : floor * 0.995 + 0.00005;

    if (!listening) return;
    recording.push(new Float32Array(event.inputBuffer.getChannelData(0)));
    recordingChunks++;

    quiet = peak < Math.max(0.012, floor * 3) ? quiet + 2048 / audioCtx.sampleRate : 0;
    if (quiet > 1.5 && secondsHeld() > 0.6) {
      log("info", "auto stop on silence");
      endTalk();
    }
  };

  const mute = audioCtx.createGain();
  mute.gain.value = 0;
  source.connect(recorder).connect(mute).connect(audioCtx.destination);

  alertBox("microphone ready: " + label, true);
  log("info", "graph at " + audioCtx.sampleRate + " Hz, state " + audioCtx.state +
    (audioCtx.state === "running" ? "" : " (click the page once to enable capture)"));
  return label;
}

/* The capture path records at the browser's rate (48k here) but is judged in seconds.
   Mixing the two up made a 0.4s minimum into 0.13s, so a cough or a door click was posted
   to whisper and came back as invented text. */
function secondsHeld() {
  const rate = audioCtx ? audioCtx.sampleRate : TARGET_RATE;
  return recordingChunks * 2048 / rate;
}

async function listMics() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const inputs = devices.filter((device) => device.kind === "audioinput");
    ui.mic.innerHTML = "";
    for (const device of inputs) {
      const option = document.createElement("option");
      option.value = device.deviceId;
      option.textContent = device.label || "microphone " + (ui.mic.length + 1);
      ui.mic.appendChild(option);
    }
    if (!inputs.length) {
      ui.mic.innerHTML = '<option value="">no inputs found</option>';
    }
    log("info", "inputs: " + inputs.length);
  } catch (err) {
    log("error", "enumerateDevices: " + err.message);
  }
}

async function initMic() {
  log("info", "page loaded, mic init starting");
  try {
    await openMic(micDevice);
    await listMics();
  } catch (err) {
    const name = err && err.name;
    const message = name === "NotAllowedError"
      ? "microphone permission was denied - allow it in the browser, then press 'enable microphone'"
      : name === "NotFoundError"
      ? "no microphone found on this machine"
      : name === "NotReadableError"
      ? "the microphone is busy in another app (close it and retry)"
      : "microphone failed: " + (err && err.message);
    alertBox(message);
    log("error", message);
  }
}

ui.mic.addEventListener("change", async () => {
  micDevice = ui.mic.value;
  await initMic();
});

function useServerMic() {
  return ui.micMode.value === "server";
}

ui.micMode.addEventListener("change", async () => {
  const server = useServerMic();
  ui.deviceRow.style.display = server ? "none" : "";
  if (server) {
    alertBox("using the app's microphone - click once and speak, it stops when you go quiet", true);
  } else {
    await initMic();
  }
  log("info", "microphone mode: " + ui.micMode.value);
});

function gainTrim(buffer) {
  let peak = 0;
  for (let i = 0; i < buffer.length; i++) {
    const value = Math.abs(buffer[i]);
    if (value > peak) peak = value;
  }
  if (peak < 0.02 || peak < 0.7) return buffer;
  const scale = 0.92 / peak;
  const out = new Float32Array(buffer.length);
  for (let i = 0; i < buffer.length; i++) out[i] = buffer[i] * scale;
  log("info", "gain trimmed by " + scale.toFixed(2) + " (was clipping at " + peak.toFixed(2) + ")");
  return out;
}

function downsample(buffer, from, to) {
  if (from === to) return buffer;
  const ratio = from / to;
  const length = Math.floor(buffer.length / ratio);
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const start = Math.floor(i * ratio);
    const end = Math.min(buffer.length, Math.floor((i + 1) * ratio));
    let sum = 0;
    for (let j = start; j < end; j++) sum += buffer[j];
    out[i] = sum / Math.max(1, end - start);
  }
  return out;
}

function toPcm16(float32) {
  const out = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

/* ---------------- conversation ---------------- */

function addTurn(who) {
  if (!history.length) ui.log.innerHTML = "";
  const wrap = document.createElement("div");
  wrap.className = "turn " + who;
  wrap.innerHTML = '<div class="who">' + who + '</div><p class="txt"></p>';
  ui.log.appendChild(wrap);
  const entry = { node: wrap.querySelector(".txt"), text: "" };
  history.push(entry);
  ui.log.scrollTop = ui.log.scrollHeight;
  return entry;
}

function appendText(entry, delta) {
  entry.text += (entry.text ? " " : "") + delta;
  entry.node.textContent = entry.text;
  ui.log.scrollTop = ui.log.scrollHeight;
}

/* ---------------- networking ---------------- */

async function getJSON(url) {
  return (await fetch(url)).json();
}

async function post(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  try {
    return await response.json();
  } catch (e) {
    return null;
  }
}

async function sendTurn(pcm) {
  busy = true;
  firstAudioAt = 0;
  turnStartedAt = performance.now();
  ui.latency.textContent = "...";
  ui.talk.disabled = true;
  ui.stop.disabled = false;
  setPhase(pcm ? "transcribing" : "listening");
  log("info", pcm
    ? "posting " + (pcm.length / 2 / TARGET_RATE).toFixed(2) + "s of recorded audio"
    : "asking the app to record");

  let bot = addTurn("bot");
  const caret = document.createElement("span");
  caret.className = "caret";
  caret.textContent = "▍";
  bot.node.appendChild(caret);

  /* try/finally is load bearing: without it any throw in here (a dropped connection, a
     half-written frame, a decode failure) left busy stuck true, which disabled the talk
     button permanently and only a page reload could clear. */
  try {
    const response = await fetch(pcm ? "/api/turn" : "/api/listen", {
      method: "POST",
      body: pcm ? pcm.buffer : "",
    });
    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw new Error("server said " + response.status + " " + detail.slice(0, 120));
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let chain = Promise.resolve();

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        const frame = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);
        const name = /^event: (.+)$/m.exec(frame);
        const data = /data: (.+)$/m.exec(frame);
        if (!name || !data) continue;
        const payload = JSON.parse(data[1]);
        if (payload.turn) turnId = payload.turn;

        if (name[1] === "transcript") {
          addTurn("you").node.textContent = payload.text;
        } else if (name[1] === "level") {
          setMeter(payload.level);
        } else if (name[1] === "status") {
          setPhase(payload.phase);
        } else if (name[1] === "text") {
          appendText(bot, payload.delta);
        } else if (name[1] === "audio") {
          chain = chain.then(() => playChunk(payload.wav));
        } else if (name[1] === "done") {
          bot.node.classList.add("spoken");
          if (payload.error) appendText(bot, payload.error);
          if (payload.reason === "no speech") {
            alertBox("the app heard no speech - check the microphone, then try again");
            log("warn", "server heard no speech");
          }
          if (payload.timings && payload.timings.total) {
            const llm = payload.timings.llm !== undefined ? payload.timings.llm : 0;
            ui.timing.textContent =
              "stt " + (payload.timings.stt || 0).toFixed(2) + "s · llm " +
              llm.toFixed(2) + "s · total " + payload.timings.total.toFixed(2) + "s";
          }
        }
      }
    }

    await chain;
  } catch (e) {
    log("error", "turn failed: " + (e && e.message ? e.message : e));
    appendText(bot, " (the turn failed: " + (e && e.message ? e.message : e) + ")");
    alertBox("something went wrong - the details are in the server log");
  } finally {
    if (caret.parentNode) caret.remove();
    busy = false;
    ui.talk.disabled = false;
    ui.stop.disabled = true;
    if (!pending.length) setPhase("idle");
  }
}

/* ---------------- interaction ---------------- */

async function startTalk() {
  if (busy || listening) return;
  /* Claim the button synchronously. startTalk awaits getUserMedia and resume(), and the
     matching pointerup/keyup lands in the middle of those awaits - endTalk then returned
     immediately because listening was still false, and we resumed into a recording nobody
     was holding. */
  listening = true;
  fromButton = true;
  ui.talk.classList.add("live");
  alertBox("");

  if (useServerMic()) {
    log("info", "server microphone capture");
    sendTurn(null);
    return;
  }
  if (!micStream) {
    alertBox("no microphone yet - retrying");
    await initMic();
    if (!micStream) {
      listening = false;
      ui.talk.classList.remove("live");
      return;
    }
  }
  if (!(await audioRunning())) {
    listening = false;
    ui.talk.classList.remove("live");
    return;
  }
  recording = [];
  recordingChunks = 0;
  peakSeen = 0;
  log("info", "capture started, audio " + audioCtx.state);
  setPhase("listening");
  setTimeout(() => {
    if (listening && recordingChunks === 0) {
      alertBox("this browser is not capturing - set 'microphone' to 'the app' below");
      log("warn", "no samples after 1.5s, audio ctx " + audioCtx.state);
    }
  }, 1500);
}

async function endTalk() {
  if (!listening) return;
  listening = false;
  fromButton = false;
  ui.talk.classList.remove("live");
  if (useServerMic()) {
    /* Release must end the *recording*, not the turn. Calling /api/interrupt here cancelled
       the whole reply, so pushing the button and letting go mid-sentence threw the answer
       away - and the panel still claimed to be talking. */
    if (busy) post("/api/stop-listening");
    return;
  }

  const merged = new Float32Array(recordingChunks * 2048);
  let offset = 0;
  for (const block of recording) {
    merged.set(block, offset);
    offset += block.length;
  }
  recording = [];
  const seconds = offset / (audioCtx ? audioCtx.sampleRate : TARGET_RATE);
  log("info", "captured " + seconds.toFixed(2) + "s, peak " + peakSeen.toFixed(3));

  if (seconds < 0.4) {
    setPhase("idle");
    alertBox("heard nothing - check the input meter and the microphone you picked");
    return;
  }
  if (peakSeen < 0.02) {
    alertBox("that was very quiet (" + peakSeen.toFixed(3) +
      ") - speaking further from the mic helps");
  }
  await sendTurn(toPcm16(downsample(gainTrim(merged), audioCtx.sampleRate, TARGET_RATE)));
}

ui.talk.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  startTalk();
});
window.addEventListener("pointerup", () => { if (fromButton) endTalk(); });
ui.talk.addEventListener("pointercancel", () => endTalk());

let space = false;
window.addEventListener("keydown", (e) => {
  if (e.code === "Space" && !space && !e.repeat) {
    e.preventDefault();
    space = true;
    startTalk();
  }
  if (e.code === "Escape" && busy) stopReply();
});
window.addEventListener("keyup", (e) => {
  if (e.code === "Space" && space) {
    e.preventDefault();
    space = false;
    endTalk();
  }
});

function stopReply() {
  stopAudio();
  post("/api/interrupt");
  ui.stop.disabled = true;
  setPhase("idle");
}
ui.stop.addEventListener("click", stopReply);
ui.reset.addEventListener("click", () => post("/api/reset", { model: state.model }));

/* ---------------- live controls ---------------- */

function flash(text) {
  ui.info.textContent = text;
}

function slider(id, key, format) {
  const input = $(id);
  const label = $(id + "Val");
  input.addEventListener("input", () => { label.textContent = format(input.value); });
  input.addEventListener("change", async () => {
    await post("/api/settings", { [key]: parseFloat(input.value) });
    flash(key + " = " + format(input.value));
  });
}

slider("speed", "speed", (v) => Number(v).toFixed(2));
slider("minWords", "min_words", (v) => v);
slider("debounce", "debounce", (v) => Number(v).toFixed(2));
slider("gap", "gap", (v) => Number(v).toFixed(2));

ui.voice.addEventListener("change", async () => {
  await post("/api/settings", { voice: ui.voice.value });
  flash("voice = " + ui.voice.value);
});

ui.model.addEventListener("change", async () => {
  await post("/api/settings", { model: ui.model.value });
  flash("model = " + ui.model.value);
});

ui.preview.addEventListener("click", async () => {
  ui.preview.disabled = true;
  try {
    const response = await fetch("/api/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: "Here is how I sound with the current settings." }),
    });
    const ctx = await ensurePlay();
    const buffer = await ctx.decodeAudioData(await response.arrayBuffer());
    // Clears the meter and the speaking caption like a real turn does; a bare
    // setPhase("idle") left the scope claiming the model was talking forever.
    playBuffer(buffer, playbackFinished);
  } catch (e) {
    log("error", "preview failed: " + (e && e.message ? e.message : e));
    alertBox("preview failed - see the server log");
  } finally {
    ui.preview.disabled = false;
  }
});

/* ---------------- boot ---------------- */

async function loadState() {
  state = await getJSON("/api/state");

  const voices = ui.voice;
  voices.innerHTML = "";
  for (const voice of state.voices || []) {
    const option = document.createElement("option");
    option.value = option.textContent = voice;
    voices.appendChild(option);
  }
  voices.value = state.voice;

  const models = ui.model;
  models.innerHTML = "";
  for (const model of state.models || []) {
    const option = document.createElement("option");
    option.value = option.textContent = model;
    models.appendChild(option);
  }
  models.value = state.model;

  ui.speed.value = state.speed;
  $("speedVal").textContent = Number(state.speed).toFixed(2);
  ui.minWords.value = state.min_words;
  $("minWordsVal").textContent = state.min_words;
  ui.debounce.value = state.debounce;
  $("debounceVal").textContent = Number(state.debounce).toFixed(2);
  ui.gap.value = state.gap;
  $("gapVal").textContent = Number(state.gap).toFixed(2);

  ui.info.textContent = "hearing: " + state.stt + " (" + state.whisper_model + ")" +
    (state.error ? " · tts problem: " + state.error : "");
}

setPhase("idle");
window.__uiAlive = true;
loadState();
initMic();
