"use strict";
/* audio2mp4 — record system audio to MP3 + MP4, open, trim, retag, save.
 *
 *   record : MediaRecorder (native, ~100x real time) -> decode once -> LAME once
 *            the same MP3 frames become the .mp3 file and the .mp4 audio track
 *   load   : WebAudio decodeAudioData for audio, <video> for video preview
 *   save   : MP4/M4A stream-copied with new tags; MP3 frames copied verbatim
 *   export : cut to Start..End and re-encode (H.264 video, MP3 audio)
 */
(() => {
  "use strict";
  const { Mp3Encoder } = lamejs;
  const MB = Mediabunny;
  const {
    Output, BufferTarget, Mp4OutputFormat, Mp3OutputFormat,
    EncodedAudioPacketSource, EncodedPacket, CanvasSource,
    BlobSource, ALL_FORMATS, Input,
  } = MB;

  // ------------------------------------------------------------------ dom
  const $ = (id) => document.getElementById(id);
  const ui = {
    status: $("status"), statusText: $("statusText"),
    source: $("source"), bitrate: $("bitrate"),
    btnRecord: $("btnRecord"), btnStop: $("btnStop"), recTime: $("recTime"),
    recHint: $("recHint"), recProgress: $("recProgress"),
    view: $("view"), preview: $("preview"),
    btnPlay: $("btnPlay"), tc: $("tc"), seek: $("seek"),
    btnFrameBack: $("btnFrameBack"), btnFrameFwd: $("btnFrameFwd"),
    fpsNum: $("fpsNum"), chkMute: $("chkMute"),
    wave: $("wave"), selInfo: $("selInfo"),
    btnSelStart: $("btnSelStart"), btnSelEnd: $("btnSelEnd"),
    btnSelClear: $("btnSelClear"), btnSelToClip: $("btnSelToClip"),
    trackVideo: $("trackVideo"), clipVideo: $("clipVideo"), phVideo: $("phVideo"),
    trackAudio: $("trackAudio"), clipAudio: $("clipAudio"), phAudio: $("phAudio"),
    inSec: $("inSec"), outSec: $("outSec"), saveFormat: $("saveFormat"),
    saveBitrate: $("saveBitrate"), btnSave: $("btnSave"), btnExport: $("btnExport"),
    btnReset: $("btnReset"), saveProgress: $("saveProgress"), editLog: $("editLog"),
    btnOpen: $("btnOpen"), btnSaveAs: $("btnSaveAs"), fileInput: $("fileInput"),
    fileList: $("fileList"), fileHint: $("fileHint"), loadLog: $("loadLog"),
    mTitle: $("mTitle"), mArtist: $("mArtist"), mAlbum: $("mAlbum"),
    mComment: $("mComment"), mYear: $("mYear"), coverInput: $("coverInput"),
    coverCv: $("coverCv"), coverInfo: $("coverInfo"),
    btnUseFrame: $("btnUseFrame"), btnClearCover: $("btnClearCover"),
    fileInfo: $("fileInfo"), saveLog: $("saveLog"),
  };
  const viewCtx = ui.view.getContext("2d", { alpha: false });
  const waveCtx = ui.wave.getContext("2d");
  const coverCtx = ui.coverCv.getContext("2d");

  // ------------------------------------------------------------------ state
  const state = {
    clips: [],
    active: null,
    file: null,                 // { blob, name, type, duration, audio, video, tags, cover, audioTrack, videoTrack }
    sel: { start: 0, end: 0 },
    pos: 0,
    playing: false,
    fps: 24,
    cover: null,
    busy: false,
  };
  let rec = null;
  let playUrl = null, playUrlFor = null;
  const audioEl = new Audio();
  audioEl.preload = "auto";
  audioEl.muted = false;

  const COLOR = { waveBg: "#181c24", base: "#7d8fab", sel: "#4c9aff", playhead: "#e2a93b" };

  // ------------------------------------------------------------------ helpers
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const fmtTime = (s) => {
    if (!Number.isFinite(s) || s < 0) s = 0;
    const m = Math.floor(s / 60);
    return `${m}:${(s - m * 60).toFixed(1).padStart(4, "0")}`;
  };
  const fmtBytes = (n) => n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(1)} KiB` : `${(n / 1048576).toFixed(1)} MiB`;
  const fmtBytesSI = (n) => n < 1e6 ? `${(n / 1e3).toFixed(0)} kB` : `${(n / 1e6).toFixed(1)} MB`;
  const baseName = (name) => (name || "audio").replace(/\.[^.]+$/, "").replace(/[^\w.\- ]+/g, "_").trim() || "audio";
  const setStatus = (text, kind) => { ui.statusText.textContent = text; ui.status.className = "status" + (kind ? " " + kind : ""); };
  const appendLog = (el, text) => { el.textContent += (el.textContent ? "\n" : "") + text; };
  const log = (el, text) => { el.textContent = text; };
  const showProgress = (el, show, frac, label) => {
    el.classList.toggle("show", !!show);
    if (!show) return;
    el.querySelector("i").style.width = `${clamp(frac, 0, 1) * 100}%`;
    if (label) el.querySelector(".lbl").textContent = label;
  };
  const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 60000);
  }

  function timestampName() {
    const d = new Date(), p = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
  }

  function metadataTags(fields) {
    const tags = {};
    if (fields.title) tags.title = fields.title;
    if (fields.artist) tags.artist = fields.artist;
    if (fields.album) tags.album = fields.album;
    if (fields.comment) tags.comment = fields.comment;
    if (fields.year) tags.date = String(fields.year);
    if (fields.cover) tags.images = [{ data: fields.cover.bytes, mimeType: fields.cover.mime, kind: "coverFront" }];
    return tags;
  }

  // ------------------------------------------------------------------ MP3 framing
  const MP3_BITRATES = {
    1: [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0],
    2: [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0],
  };
  const MP3_RATES = {
    1: [44100, 48000, 32000, 0],
    2: [22050, 24000, 16000, 0],
    0: [11025, 12000, 8000, 0],   // MPEG-2.5
  };

  /**
   * Parse a Layer III byte stream into exact frames.
   * Handles MPEG-1 (1152 samples/frame) and MPEG-2 / 2.5 (576 samples/frame),
   * because LAME downshifts to MPEG-2 when the requested bitrate is low.
   */
  function parseMp3Frames(bytes) {
    const frames = [];
    let at = 0, samplesPerFrame = 1152, rate = 0;
    while (at + 4 <= bytes.length) {
      if (bytes[at] !== 0xff || (bytes[at + 1] & 0xe0) !== 0xe0) { at++; continue; }
      const b1 = bytes[at + 1], b2 = bytes[at + 2];
      const versionBits = (b1 >> 3) & 3;          // 3 = MPEG-1, 2 = MPEG-2, 0 = MPEG-2.5
      const layerBits = (b1 >> 1) & 3;            // 1 = Layer III
      if (versionBits === 1 || layerBits !== 1) { at++; continue; }
      const ver = versionBits === 3 ? 1 : versionBits === 2 ? 2 : 0;
      const br = MP3_BITRATES[ver === 1 ? 1 : 2][(b2 >> 4) & 15] * 1000;
      const sr = MP3_RATES[ver][(b2 >> 2) & 3];
      if (!br || !sr) { at++; continue; }
      const len = Math.floor((ver === 1 ? 144 : 72) * br / sr) + ((b2 >> 1) & 1);
      if (at + len > bytes.length) break;
      frames.push({ offset: at, length: len });
      samplesPerFrame = ver === 1 ? 1152 : 576;
      rate = sr;
      at += len;
    }
    return { frames, spf: samplesPerFrame, sampleRate: rate };
  }

  function concatChunks(chunks) {
    const total = chunks.reduce((a, c) => a + c.length, 0);
    const out = new Uint8Array(total);
    let at = 0;
    for (const c of chunks) { out.set(c, at); at += c.length; }
    return out;
  }

  // ================================================================== recording
  const REC_MIME = (() => {
    for (const m of ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus"]) {
      if (window.MediaRecorder && MediaRecorder.isTypeSupported(m)) return m;
    }
    return "";
  })();

  function setRecUI(active) {
    ui.btnRecord.disabled = active;
    ui.btnStop.disabled = !active;
    ui.source.disabled = active;
    ui.bitrate.disabled = active;
  }

  async function startRecording() {
    const srcMode = ui.source.value;
    let audioCtx = null, stream = null, sourceNode = null, oscs = [];
    setStatus("requesting capture…", "busy");
    ui.btnRecord.disabled = true;
    try {
      if (srcMode === "tone") {
        audioCtx = new AudioContext({ sampleRate: 48000 });
        const gain = audioCtx.createGain();
        gain.gain.value = 0.20;
        const o1 = audioCtx.createOscillator(); o1.type = "sine"; o1.frequency.value = 440;
        const o2 = audioCtx.createOscillator(); o2.type = "triangle"; o2.frequency.value = 660;
        const g2 = audioCtx.createGain(); g2.gain.value = 0.45;
        o1.connect(gain); o2.connect(g2).connect(gain);
        o1.start(); o2.start();
        oscs = [o1, o2];
        sourceNode = gain;
      } else {
        if (srcMode === "system") {
          stream = await navigator.mediaDevices.getDisplayMedia({
            video: { frameRate: 1 },
            audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 2 },
            systemAudio: "include", selfBrowserSurface: "exclude", surfaceSwitching: "exclude",
          });
          if (stream.getAudioTracks().length === 0) {
            stream.getTracks().forEach((t) => t.stop());
            throw new Error("No audio in the capture. Tick “Share tab audio” (or “Share system audio”) in the picker.");
          }
          stream.getVideoTracks().forEach((t) => t.stop());
        } else {
          stream = await navigator.mediaDevices.getUserMedia({
            audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 2, sampleRate: 48000 },
          });
        }
        audioCtx = new AudioContext({ sampleRate: 48000 });
        sourceNode = audioCtx.createMediaStreamSource(stream);
      }
      if (audioCtx.state === "suspended") await audioCtx.resume();
      const sampleRate = audioCtx.sampleRate;

      // tap -> meter (visual) and tap -> MediaStreamDestination (recording)
      const tap = audioCtx.createGain();
      tap.gain.value = 1;
      sourceNode.connect(tap);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      tap.connect(analyser);
      const dest = audioCtx.createMediaStreamDestination();
      tap.connect(dest);

      const recorder = new MediaRecorder(dest.stream, REC_MIME ? { mimeType: REC_MIME, audioBitsPerSecond: 192000 } : undefined);
      const parts = [];
      recorder.ondataavailable = (e) => { if (e.data && e.data.size) parts.push(e.data); };
      const startedAt = performance.now();
      recorder.start(500);

      const meter = new Float32Array(analyser.fftSize);
      const timer = setInterval(() => {
        analyser.getFloatTimeDomainData(meter);
        let peak = 0;
        for (let i = 0; i < meter.length; i += 2) peak = Math.max(peak, Math.abs(meter[i]));
        drawRecordLevel(peak);
        ui.recTime.textContent = fmtTime((performance.now() - startedAt) / 1000);
      }, 100);

      rec = {
        startedAt, sampleRate,
        stop: () => new Promise((resolve) => {
          clearInterval(timer);
          const finish = () => {
            oscs.forEach((o) => { try { o.stop(); } catch (e) {} });
            if (stream) stream.getTracks().forEach((t) => t.stop());
            audioCtx.close().catch(() => {});
            resolve(new Blob(parts, { type: (REC_MIME || "audio/webm").split(";")[0] }));
          };
          recorder.onstop = finish;
          try { recorder.stop(); } catch (e) { finish(); }
        }),
      };
      setRecUI(true);
      setStatus(`● recording ${srcMode} · ${sampleRate} Hz`, "rec");
      ui.recHint.textContent = "Recording. MP3 + MP4 are written when you press Stop.";
    } catch (err) {
      setStatus("idle");
      ui.btnRecord.disabled = false;
      ui.recHint.innerHTML = `<span class="err">${escapeHtml(err && err.message ? err.message : String(err))}</span>`;
      if (stream) stream.getTracks().forEach((t) => t.stop());
      if (audioCtx) audioCtx.close().catch(() => {});
    }
  }

  async function stopRecording() {
    if (!rec) return;
    const session = rec;
    rec = null;
    const captured = (performance.now() - session.startedAt) / 1000;
    setStatus("writing files…", "busy");
    ui.btnStop.disabled = true;
    ui.btnRecord.disabled = true;
    showProgress(ui.recProgress, true, 0.05, "closing capture…");
    const convertStart = performance.now();
    try {
      const blob = await session.stop();
      showProgress(ui.recProgress, true, 0.15, "decoding capture…");
      const tags = currentTags();
      if (!tags.title) tags.title = timestampName();
      const result = await buildAudioOutputs(blob, {
        sampleRate: session.sampleRate,
        bitrateKbps: parseInt(ui.bitrate.value, 10) || 192,
        tags,
        cover: state.cover,
      });
      const convertSec = (performance.now() - convertStart) / 1000;
      showProgress(ui.recProgress, true, 1, "done");
      appendLog(ui.loadLog, `Recorded ${captured.toFixed(2)} s → ${fmtBytesSI(result.mp3.size)} MP3 + ${fmtBytesSI(result.mp4.size)} MP4 · converted in ${convertSec.toFixed(2)} s (${(captured / Math.max(convertSec, 0.001)).toFixed(1)}× real time)`);
      const mp3Clip = { name: `${baseName(tags.title)}.mp3`, type: "mp3", blob: result.mp3, size: result.mp3.size, edited: false };
      const mp4Clip = { name: `${baseName(tags.title)}.mp4`, type: "mp4", blob: result.mp4, size: result.mp4.size, edited: false };
      addClip(mp3Clip);
      addClip(mp4Clip);
      setStatus("recorded", "ok");
      await selectClip(mp4Clip);
    } catch (err) {
      appendLog(ui.loadLog, "Record failed: " + (err && err.message ? err.message : String(err)));
      setStatus("idle");
    } finally {
      setTimeout(() => showProgress(ui.recProgress, false), 600);
      setRecUI(false);
      ui.recHint.textContent = "System capture asks which tab/window to share — tick “Share tab audio” (or “Share system audio” on Windows) in the picker.";
    }
  }

  let recPeak = 0;
  function drawRecordLevel(peak) {
    const w = ui.wave.width, h = ui.wave.height;
    waveCtx.fillStyle = COLOR.waveBg;
    waveCtx.fillRect(0, 0, w, h);
    recPeak = Math.max(peak, recPeak * 0.92);
    const barH = clamp(recPeak, 0, 1) * (h - 10);
    waveCtx.fillStyle = recPeak > 0.98 ? "#ef5b5b" : "#35c46b";
    waveCtx.fillRect(0, h - 5 - barH, w, barH);
    waveCtx.fillStyle = "#c9d4e6";
    waveCtx.font = "12px ui-monospace, monospace";
    waveCtx.fillText(`recording · peak ${(20 * Math.log10(Math.max(recPeak, 1e-6))).toFixed(1)} dBFS`, 10, 20);
  }

  // ================================================================== audio encoding
  /** Decode a captured blob to a stereo AudioBuffer at `rate`. */
  async function decodeToAudioBuffer(blob, rate) {
    const bytes = await blob.arrayBuffer();
    const targetRate = rate || 48000;
    let decoded;
    try {
      decoded = await new OfflineAudioContext(2, 1, targetRate).decodeAudioData(bytes.slice(0));
    } catch (e) {
      const ctx = new AudioContext({ sampleRate: targetRate });
      try { decoded = await ctx.decodeAudioData(bytes); } finally { ctx.close(); }
    }
    return normalizeBuffer(decoded, targetRate);
  }

  function normalizeBuffer(decoded, rate) {
    if (decoded.numberOfChannels === 2 && Math.abs(decoded.sampleRate - rate) < 1) return decoded;
    const out = new AudioBuffer({ length: Math.ceil(decoded.duration * rate), numberOfChannels: 2, sampleRate: rate });
    for (let ch = 0; ch < 2; ch++) {
      const src = decoded.getChannelData(Math.min(ch, decoded.numberOfChannels - 1));
      out.copyToChannel(src.length > out.length ? src.subarray(0, out.length) : src, ch);
    }
    return out;
  }

  /** LAME-encode an AudioBuffer to raw MP3 bytes (one pass, UI-friendly). */
  const toKbps = (v) => clamp(Math.round((v > 1000 ? v / 1000 : v) || 192), 32, 320);

  async function encodeMp3Raw(ab, bitrateKbps, progressEl, from = 0, span = 1) {
    const enc = new Mp3Encoder(2, ab.sampleRate, toKbps(bitrateKbps));
    const l = ab.getChannelData(0), r = ab.getChannelData(1);
    const n = l.length, N = 1152;
    const l16 = new Int16Array(N), r16 = new Int16Array(N);
    const parts = [];
    const t0 = performance.now();
    for (let at = 0; at < n; at += N) {
      const cnt = Math.min(N, n - at);
      for (let i = 0; i < cnt; i++) {
        const a = l[at + i] * 32767, b = r[at + i] * 32767;
        l16[i] = a > 32767 ? 32767 : a < -32768 ? -32768 : Math.round(a);
        r16[i] = b > 32767 ? 32767 : b < -32768 ? -32768 : Math.round(b);
      }
      const out = enc.encodeBuffer(l16.subarray(0, cnt), r16.subarray(0, cnt));
      if (out.length) parts.push(new Uint8Array(out));
      if (at % (N * 1200) === 0) {
        if (progressEl) {
          const speed = (at / ab.sampleRate) / Math.max((performance.now() - t0) / 1000, 0.001);
          showProgress(progressEl, true, from + span * (at / n),
            `encoding MP3 ${Math.round((at / n) * 100)}% · ${speed > 999 ? ">999" : speed.toFixed(0)}× real time`);
        }
        await new Promise((res) => setTimeout(res, 0));
      }
    }
    const tail = enc.flush();
    if (tail.length) parts.push(new Uint8Array(tail));
    return concatChunks(parts);
  }

  /** Write exact MP3 frames as a tagged MP3 file. */
  async function writeMp3Frames(frameBytes, sampleRate, spf, tags) {
    const target = new BufferTarget();
    const output = new Output({ format: new Mp3OutputFormat(), target });
    const src = new EncodedAudioPacketSource("mp3");
    output.addAudioTrack(src, { numberOfChannels: 2, sampleRate });
    output.setMetadataTags(metadataTags(tags));
    await output.start();
    const dur = spf / sampleRate;
    for (let i = 0; i < frameBytes.length; i++) {
      await src.add(new EncodedPacket(frameBytes[i], "key", i * dur, dur, i),
        i === 0 ? { decoderConfig: { codec: "mp3", sampleRate, numberOfChannels: 2 } } : undefined);
    }
    await output.finalize();
    return new Blob([target.buffer], { type: "audio/mpeg" });
  }

  /** Generated cover art so a recording always has artwork. */
  function autoCover(bytes) {
    const size = 500;
    const c = document.createElement("canvas");
    c.width = size; c.height = size;
    const g = c.getContext("2d", { alpha: false });
    let h = 2166136261 >>> 0;
    for (let i = 0; i < bytes.length; i += 4096) { h ^= bytes[i]; h = Math.imul(h, 16777619) >>> 0; }
    const grad = g.createLinearGradient(0, 0, size, size);
    grad.addColorStop(0, "#101623"); grad.addColorStop(0.55, "#1c2740"); grad.addColorStop(1, "#0b0f18");
    g.fillStyle = grad; g.fillRect(0, 0, size, size);
    const bars = 42, bw = size / bars;
    for (let i = 0; i < bars; i++) {
      const t = ((h >>> (i % 24)) & 31) / 31;
      const bh = 40 + t * size * 0.55;
      g.fillStyle = `hsl(${200 + (i / bars) * 90}, 55%, ${28 + t * 28}%)`;
      g.fillRect(i * bw + 2, size - bh - 40, bw - 4, bh);
    }
    g.fillStyle = "#e8eefc";
    g.font = "600 34px -apple-system, system-ui, sans-serif";
    g.fillText((ui.mTitle.value || "Recording").slice(0, 22), 40, 70);
    g.fillStyle = "#8fa3c7";
    g.font = "20px -apple-system, system-ui, sans-serif";
    g.fillText(new Date().toISOString().slice(0, 10), 40, 100);
    const dataUrl = c.toDataURL("image/jpeg", 0.86);
    const bin = atob(dataUrl.split(",")[1]);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return { bytes: out, mime: "image/jpeg" };
  }

  /** Draw cover art (or a gradient) plus a title bar onto a 16:9 frame. */
  function drawCoverFrame(ctx, w, h, bitmap, title) {
    ctx.fillStyle = "#0b0f18";
    ctx.fillRect(0, 0, w, h);
    if (bitmap) {
      ctx.save();
      ctx.filter = "blur(42px)";
      const scale = Math.max(w / bitmap.width, h / bitmap.height) * 1.25;
      ctx.drawImage(bitmap, (w - bitmap.width * scale) / 2, (h - bitmap.height * scale) / 2, bitmap.width * scale, bitmap.height * scale);
      ctx.restore();
      ctx.fillStyle = "rgba(6,9,14,0.42)";
      ctx.fillRect(0, 0, w, h);
      const side = Math.round(h * 0.62);
      ctx.drawImage(bitmap, (w - side) / 2, (h - side) / 2, side, side);
    } else {
      const grad = ctx.createLinearGradient(0, 0, w, h);
      grad.addColorStop(0, "#101623"); grad.addColorStop(0.55, "#1c2740"); grad.addColorStop(1, "#0b0f18");
      ctx.fillStyle = grad; ctx.fillRect(0, 0, w, h);
    }
    if (title) {
      ctx.textAlign = "center";
      ctx.fillStyle = "rgba(0,0,0,0.35)";
      ctx.fillRect(0, h - 96, w, 96);
      ctx.fillStyle = "#e8eefc";
      ctx.font = "600 34px -apple-system, system-ui, sans-serif";
      ctx.fillText(title.slice(0, 44), w / 2, h - 44);
      ctx.textAlign = "start";
    }
  }

  /**
   * One decode + one LAME pass produces both outputs:
   *   mp3 = tagged MP3 file
   *   mp4 = MP4 carrying those exact MP3 frames + one still video frame
   */
  async function buildAudioOutputs(webmBlob, { sampleRate, bitrateKbps, tags, cover }) {
    const ab = await decodeToAudioBuffer(webmBlob, sampleRate);
    const art = cover || autoCover(new Uint8Array([Math.floor(ab.duration * 10)]));
    const outTags = { ...tags, cover: art };

    const raw = await encodeMp3Raw(ab, bitrateKbps, ui.recProgress, 0.15, 0.6);
    const parsed = parseMp3Frames(raw);
    if (!parsed.frames.length) throw new Error("MP3 encoder produced no frames.");
    const frames = parsed.frames.map((f) => raw.subarray(f.offset, f.offset + f.length));
    const rate = parsed.sampleRate || ab.sampleRate;
    showProgress(ui.recProgress, true, 0.8, "writing MP3 + MP4…");

    const [mp3, mp4] = await Promise.all([
      writeMp3Frames(frames, rate, parsed.spf, outTags),
      muxMp4FromFrames({ frames, sampleRate: rate, spf: parsed.spf, durationSec: ab.duration, cover: art, title: outTags.title, tags: outTags }),
    ]);
    return { mp3, mp4, tags: outTags };
  }

  /** MP4: MP3 audio track + a single still video frame (instant mux). */
  async function muxMp4FromFrames({ frames, sampleRate, spf, durationSec, cover, title, tags }) {
    const W = 1280, H = 720, fps = 1;
    const canvas = document.createElement("canvas");
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext("2d", { alpha: false });
    const bmp = cover ? await createImageBitmap(new Blob([cover.bytes], { type: cover.mime })).catch(() => null) : null;
    drawCoverFrame(ctx, W, H, bmp, title);

    const target = new BufferTarget();
    const output = new Output({ format: new Mp4OutputFormat({ fastStart: "in-memory" }), target });
    const vsrc = new CanvasSource(canvas, { codec: "avc", bitrate: 800000 });
    output.addVideoTrack(vsrc, { frameRate: fps });
    const asrc = new EncodedAudioPacketSource("mp3");
    output.addAudioTrack(asrc, { numberOfChannels: 2, sampleRate });
    output.setMetadataTags(metadataTags(tags));
    await output.start();
    await vsrc.add(0, Math.max(durationSec, 1 / fps));
    const dur = spf / sampleRate;
    for (let i = 0; i < frames.length; i++) {
      await asrc.add(new EncodedPacket(frames[i], "key", i * dur, dur, i),
        i === 0 ? { decoderConfig: { codec: "mp3", sampleRate, numberOfChannels: 2 } } : undefined);
    }
    await output.finalize();
    return new Blob([target.buffer], { type: "video/mp4" });
  }

  // ================================================================== loading
  async function loadFile(blob, name, { select = true } = {}) {
    const lower = name.toLowerCase();
    const type = lower.endsWith(".mp3") ? "mp3" : lower.endsWith(".m4a") ? "m4a" : "mp4";
    setStatus("loading…", "busy");
    appendLog(ui.loadLog, "");
    try {
      const input = new Input({ source: new BlobSource(blob), formats: ALL_FORMATS });
      const videoTrack = await input.getPrimaryVideoTrack();
      const audioTrack = await input.getPrimaryAudioTrack();
      if (!audioTrack && !videoTrack) throw new Error("No audio or video track found.");
      const duration = await input.computeDuration();
      showProgress(ui.saveProgress, true, 0.25, "decoding audio…");
      let audio = null;
      if (audioTrack) {
        const rate = audioTrack.sampleRate || 48000;
        let buffer = null;
        try {
          const decoded = await new OfflineAudioContext(2, 1, rate).decodeAudioData(await blob.arrayBuffer());
          buffer = normalizeBuffer(decoded, rate);
        } catch (e) {
          buffer = new AudioBuffer({ length: Math.max(1, Math.ceil(duration * rate)), numberOfChannels: 2, sampleRate: rate });
          const ch0 = buffer.getChannelData(0), ch1 = buffer.getChannelData(1);
          for await (const { buffer: b, timestamp } of new MB.AudioBufferSink(audioTrack).buffers()) {
            const at = Math.max(0, Math.round(timestamp * rate));
            for (let ch = 0; ch < Math.min(2, b.numberOfChannels); ch++) {
              const data = b.getChannelData(ch);
              (ch === 0 ? ch0 : ch1).set(data.subarray(0, Math.max(0, Math.min(data.length, ch0.length - at))), at);
            }
          }
        }
        audio = { buffer, sampleRate: buffer.sampleRate, duration: buffer.duration };
      }

      let video = null;
      if (videoTrack) {
        let fps = 0;
        try { fps = Math.round((await videoTrack.computePacketStats(60)).averagePacketRate) || 0; } catch (e) {}
        video = {
          track: videoTrack, width: videoTrack.displayWidth, height: videoTrack.displayHeight, duration,
          fps: fps >= 5 ? Math.min(fps, 60) : 1,
          still: fps < 5,
        };
      }

      const tags = {};
      try { Object.assign(tags, await input.getMetadataTags()); } catch (e) {}
      const cover = await readCover(tags);

      const model = { input, blob, name, type, duration, audio, video, tags, cover, audioTrack, videoTrack };
      state.file = model;
      state.pos = 0;
      fillTagFields(tags, cover);
      ui.fpsNum.value = String(video ? video.fps : 30);
      preparePreview();
      setRange(0, duration);
      drawWaveform();
      updateTimeline();
      ui.fileHint.textContent = "";
      ui.fileInfo.textContent = [
        name,
        `${duration.toFixed(2)} s · ${fmtBytes(blob.size)}`,
        audio ? `audio ${audio.sampleRate} Hz ${audio.buffer.numberOfChannels === 1 ? "mono" : "stereo"}` : "no audio",
        video ? `video ${video.width}×${video.height}${video.still ? " (still frame)" : ` @ ${video.fps} fps`}` : "no video",
      ].join("\n");
      if (select) renderFileList();
      showProgress(ui.saveProgress, false);
      setStatus("loaded", "ok");
      return model;
    } catch (err) {
      showProgress(ui.saveProgress, false);
      appendLog(ui.loadLog, `Could not load ${name}: ${err && err.message ? err.message : err}`);
      setStatus("idle");
      throw err;
    }
  }

  async function readCover(tags) {
    const img = tags && tags.images && tags.images.find((i) => i.kind === "coverFront" || i.kind === "unknown");
    if (!img) return null;
    const bytes = img.data instanceof Uint8Array ? img.data : new Uint8Array(img.data);
    return { bytes, mime: img.mimeType || "image/jpeg" };
  }

  function fillTagFields(tags, cover) {
    const first = (v) => (Array.isArray(v) ? v[0] : v) || "";
    ui.mTitle.value = first(tags && tags.title) || "";
    ui.mArtist.value = first(tags && tags.artist) || "";
    ui.mAlbum.value = first(tags && tags.album) || "";
    ui.mComment.value = first(tags && tags.comment) || first(tags && tags.lyrics) || "";
    ui.mYear.value = String(first(tags && tags.date) || "").slice(0, 4);
    state.cover = cover ? { bytes: cover.bytes, mime: cover.mime } : null;
    paintCover();
  }

  function paintCover() {
    coverCtx.fillStyle = "#1b2029";
    coverCtx.fillRect(0, 0, 84, 84);
    if (!state.cover) {
      coverCtx.fillStyle = "#5a6478";
      coverCtx.font = "11px ui-monospace, monospace";
      coverCtx.fillText("no cover", 16, 46);
      ui.coverInfo.textContent = "No cover art. “Use current frame” grabs the video frame at the playhead.";
      return;
    }
    const url = URL.createObjectURL(new Blob([state.cover.bytes], { type: state.cover.mime }));
    const img = new Image();
    img.onload = () => { coverCtx.drawImage(img, 0, 0, 84, 84); URL.revokeObjectURL(url); };
    img.onerror = () => { coverCtx.fillText("bad image", 16, 46); URL.revokeObjectURL(url); };
    img.src = url;
    ui.coverInfo.textContent = `${state.cover.mime} · ${fmtBytes(state.cover.bytes.length)}`;
  }

  function setCoverFromBytes(bytes, mime) {
    state.cover = { bytes, mime };
    paintCover();
  }

  // ================================================================== files list
  function addClip(clip) {
    clip.id = "c" + Math.random().toString(36).slice(2, 9);
    state.clips.push(clip);
    renderFileList();
  }

  function renderFileList() {
    ui.fileList.innerHTML = "";
    for (const clip of state.clips) {
      const row = document.createElement("div");
      row.className = "file-item" + (state.active === clip ? " active" : "");
      const meta = document.createElement("div");
      meta.className = "meta";
      const name = document.createElement("div");
      name.className = "name";
      name.textContent = clip.name;
      name.title = clip.name;
      const info = document.createElement("div");
      info.className = "info";
      info.textContent = `${clip.type.toUpperCase()} · ${fmtBytes(clip.size)}`;
      meta.append(name, info);
      const open = document.createElement("button");
      open.className = "small";
      open.textContent = state.active === clip ? "Loaded" : "Open";
      open.disabled = state.active === clip;
      open.onclick = () => selectClip(clip);
      const dl = document.createElement("button");
      dl.className = "small";
      dl.textContent = "Download";
      dl.onclick = () => downloadBlob(clip.blob, clip.name);
      const del = document.createElement("button");
      del.className = "small danger";
      del.textContent = "✕";
      del.title = "Remove from list";
      del.onclick = () => {
        state.clips = state.clips.filter((c) => c !== clip);
        if (state.active === clip) {
          state.active = null;
          state.file = null;
          state.cover = null;
          stopPlayback();
          ui.preview.removeAttribute("src");
          audioEl.removeAttribute("src");
          ui.fileInfo.textContent = "—";
          ui.selInfo.textContent = "No file loaded.";
          updateTimeline();
          drawWaveform();
          paintCover();
        }
        renderFileList();
      };
      row.append(meta, open, dl, del);
      ui.fileList.append(row);
    }
    if (!state.clips.length) ui.fileHint.textContent = "Nothing loaded. Record something, or open an MP3 / MP4 / M4A.";
    else ui.fileHint.textContent = "";
  }

  async function selectClip(clip) {
    if (state.active === clip && state.file) return;
    state.active = clip;
    renderFileList();
    try { await loadFile(clip.blob, clip.name, { select: false }); } catch (e) { /* logged */ }
    renderFileList();
  }

  // ================================================================== preview / playback
  function preparePreview() {
    stopPlayback();
    if (playUrl) { URL.revokeObjectURL(playUrl); playUrl = null; playUrlFor = null; }
    const file = state.file;
    const hasVideo = !!(file && file.video);
    ui.preview.style.display = hasVideo ? "block" : "none";
    ui.view.style.display = hasVideo ? "none" : "block";
    ui.preview.removeAttribute("src");
    audioEl.removeAttribute("src");
    if (hasVideo) {
      playUrl = URL.createObjectURL(file.blob);
      playUrlFor = file;
      ui.preview.src = playUrl;
      ui.preview.muted = ui.chkMute.checked;
    } else if (file && file.audio) {
      playUrl = URL.createObjectURL(new Blob([file.blob], { type: file.type === "mp3" ? "audio/mpeg" : "audio/mp4" }));
      playUrlFor = file;
      audioEl.src = playUrl;
      audioEl.muted = ui.chkMute.checked;
      drawIdleView();
    } else {
      drawIdleView();
    }
    ui.btnPlay.textContent = "▶ Play";
    updateTimecode(0);
    holdPlayhead(0);
  }

  function drawIdleView() {
    const w = ui.view.width, h = ui.view.height;
    viewCtx.fillStyle = "#0a0c10";
    viewCtx.fillRect(0, 0, w, h);
    viewCtx.fillStyle = "#39414f";
    viewCtx.font = "600 20px -apple-system, system-ui, sans-serif";
    viewCtx.textAlign = "center";
    viewCtx.fillText(state.file ? "audio only · no video" : "no file loaded", w / 2, h / 2);
    viewCtx.textAlign = "start";
  }

  function position() {
    if (state.playing) {
      const file = state.file;
      if (file && file.video) return ui.preview.currentTime || 0;
      if (file && file.audio) return audioEl.currentTime || 0;
    }
    return state.pos;
  }
  const totalDuration = () => (state.file ? state.file.duration : 0);

  function togglePlay() {
    if (!state.file) return;
    if (state.playing) pausePlayback();
    else startPlayback(state.pos >= totalDuration() - 0.05 ? 0 : state.pos);
  }

  function startPlayback(from) {
    const file = state.file;
    if (!file) return;
    const at = clamp(from, 0, Math.max(0, file.duration - 0.02));
    state.pos = at;
    if (file.video) {
      const v = ui.preview;
      v.currentTime = at;
      v.muted = ui.chkMute.checked;
      v.play().catch(() => {});
    } else if (file.audio) {
      audioEl.currentTime = at;
      audioEl.muted = ui.chkMute.checked;
      audioEl.play().catch(() => {});
    }
    state.playing = true;
    ui.btnPlay.textContent = "❚❚ Pause";
    ui.btnPlay.classList.add("primary");
    tick();
  }

  function pausePlayback() {
    state.pos = position();
    state.playing = false;
    ui.preview.pause();
    audioEl.pause();
    ui.btnPlay.textContent = "▶ Play";
    ui.btnPlay.classList.remove("primary");
    updateTimecode(state.pos);
  }

  function stopPlayback() {
    state.playing = false;
    try { ui.preview.pause(); } catch (e) {}
    try { audioEl.pause(); } catch (e) {}
    cancelAnimationFrame(state.rafId);
    ui.btnPlay.textContent = "▶ Play";
    ui.btnPlay.classList.remove("primary");
  }

  function tick() {
    cancelAnimationFrame(state.rafId);
    if (!state.playing) return;
    const at = clamp(position(), 0, totalDuration());
    state.pos = at;
    updateTimecode(at);
    holdPlayhead(at);
    if (at >= totalDuration() - 0.03) {
      state.pos = totalDuration();
      pausePlayback();
      return;
    }
    state.rafId = requestAnimationFrame(tick);
  }

  function updateTimecode(at) {
    ui.tc.textContent = `${fmtTime(at)} / ${fmtTime(totalDuration())}`;
    if (!ui.seek._dragging) ui.seek.value = String(totalDuration() ? Math.round((at / totalDuration()) * 1000) : 0);
  }

  function holdPlayhead(at) {
    const pct = clamp(at / Math.max(totalDuration(), 1e-6), 0, 1) * 100;
    ui.phVideo.style.left = pct + "%";
    ui.phAudio.style.left = pct + "%";
  }

  function seekTo(at) {
    at = clamp(at, 0, totalDuration());
    const file = state.file;
    state.pos = at;
    if (file) {
      if (file.video) { try { ui.preview.currentTime = at; } catch (e) {} }
      else if (file.audio) { try { audioEl.currentTime = at; } catch (e) {} }
    }
    updateTimecode(at);
    holdPlayhead(at);
  }

  // ================================================================== waveform
  function drawWaveform() {
    const dpr = window.devicePixelRatio || 1;
    const cssW = ui.wave.clientWidth || 700, cssH = 74;
    if (ui.wave.width !== Math.round(cssW * dpr) || ui.wave.height !== Math.round(cssH * dpr)) {
      ui.wave.width = Math.round(cssW * dpr);
      ui.wave.height = Math.round(cssH * dpr);
    }
    const w = ui.wave.width, h = ui.wave.height;
    waveCtx.fillStyle = COLOR.waveBg;
    waveCtx.fillRect(0, 0, w, h);
    const file = state.file;
    if (!file || !file.audio) {
      waveCtx.fillStyle = "#54607a";
      waveCtx.font = "12px ui-monospace, monospace";
      waveCtx.fillText(file ? "no audio track" : "waveform appears when a file is loaded", 10, h / 2 + 4);
      return;
    }
    // Peak envelope per pixel column. Sampling every Nth sample would alias a
    // periodic signal into a flat line, so each bucket takes its true min/max.
    const data = file.audio.buffer.getChannelData(0);
    const data1 = file.audio.buffer.numberOfChannels > 1 ? file.audio.buffer.getChannelData(1) : data;
    const n = data.length;
    const perCol = n / w;
    const mid = h / 2;
    const dur = Math.max(file.duration, 1e-6);
    const x0 = (state.sel.start / dur) * w, x1 = (state.sel.end / dur) * w;
    waveCtx.fillStyle = "rgba(76,154,255,0.13)";
    waveCtx.fillRect(x0, 0, Math.max(1, x1 - x0), h);
    waveCtx.strokeStyle = COLOR.base;
    waveCtx.lineWidth = 1;
    waveCtx.beginPath();
    for (let x = 0; x < w; x++) {
      const from = Math.floor(x * perCol);
      const to = Math.min(n, Math.max(from + 1, Math.floor((x + 1) * perCol)));
      const stride = Math.max(1, Math.floor((to - from) / 2048));   // bounded work per column
      let mn = 0, mx = 0;
      for (let i = from; i < to; i += stride) {
        const v = (data[i] + data1[i]) * 0.5;
        if (v < mn) mn = v; else if (v > mx) mx = v;
      }
      waveCtx.moveTo(x + 0.5, mid + mn * mid);
      waveCtx.lineTo(x + 0.5, Math.max(mid + mx * mid, mid + mn * mid + 0.5));
    }
    waveCtx.stroke();
    waveCtx.strokeStyle = COLOR.sel;
    waveCtx.lineWidth = 2 * dpr;
    waveCtx.beginPath();
    waveCtx.moveTo(x0, 0); waveCtx.lineTo(x0, h);
    waveCtx.moveTo(x1, 0); waveCtx.lineTo(x1, h);
    waveCtx.stroke();
  }

  function updateTimeline() {
    const file = state.file;
    const dur = Math.max(totalDuration(), 1e-6);
    ui.clipVideo.style.width = `${clamp((file && file.video ? file.duration : 0) / dur, 0, 1) * 100}%`;
    ui.clipAudio.style.width = `${clamp((file && file.audio ? file.audio.duration : 0) / dur, 0, 1) * 100}%`;
    ui.clipVideo.style.opacity = file && file.video ? "1" : "0.2";
    ui.clipAudio.style.opacity = file && file.audio ? "1" : "0.2";
  }

  // ================================================================== selection
  function setRange(start, end) {
    const dur = totalDuration();
    state.sel.start = clamp(start, 0, dur);
    state.sel.end = clamp(end, state.sel.start, dur);
    ui.inSec.value = state.sel.start.toFixed(2);
    ui.outSec.value = state.sel.end.toFixed(2);
    updateSelInfo();
    drawWaveform();
  }

  function updateSelInfo() {
    const { start, end } = state.sel;
    ui.selInfo.textContent = state.file
      ? `Selection ${fmtTime(start)} – ${fmtTime(end)} · ${(end - start).toFixed(2)} s of ${fmtTime(totalDuration())}`
      : "No file loaded.";
  }

  function readRangeInputs() {
    const dur = totalDuration();
    let a = clamp(parseFloat(ui.inSec.value) || 0, 0, dur);
    let b = clamp(parseFloat(ui.outSec.value) || dur, 0, dur);
    if (b <= a) b = Math.min(dur, a + 0.05);
    state.sel.start = a;
    state.sel.end = b;
    ui.inSec.value = a.toFixed(2);
    ui.outSec.value = b.toFixed(2);
    updateSelInfo();
    drawWaveform();
  }

  // ================================================================== tags / cover
  function currentTags() {
    return {
      title: ui.mTitle.value.trim(),
      artist: ui.mArtist.value.trim(),
      album: ui.mAlbum.value.trim(),
      comment: ui.mComment.value.trim(),
      year: ui.mYear.value.trim(),
      cover: state.cover,
    };
  }

  async function grabFrameCover() {
    const file = state.file;
    if (!file || !file.video) { ui.coverInfo.textContent = "No video in this file — choose an image instead."; return; }
    try {
      const c = document.createElement("canvas");
      c.width = 600; c.height = 600;
      const cx = c.getContext("2d", { alpha: false });
      cx.fillStyle = "#000"; cx.fillRect(0, 0, 600, 600);
      cx.drawImage(ui.preview, 0, 0, 600, 600);
      const blob = await new Promise((res) => c.toBlob(res, "image/jpeg", 0.9));
      setCoverFromBytes(new Uint8Array(await blob.arrayBuffer()), "image/jpeg");
    } catch (e) { ui.coverInfo.textContent = "Frame grab failed: " + e.message; }
  }

  // ================================================================== save (lossless)
  function selectionNeeded() {
    const { start, end } = state.sel;
    return start > 0.01 || (state.file && end < state.file.duration - 0.01);
  }

  function markEdited() {
    if (state.active) { state.active.edited = true; renderFileList(); }
  }

  async function saveCurrent() {
    const file = state.file;
    if (!file || state.busy) return;
    setBusy(true);
    setStatus("saving…", "busy");
    const t0 = performance.now();
    try {
      const format = ui.saveFormat.value;
      const tags = currentTags();
      const name = `${baseName(tags.title || file.name)}.${format}`;
      let blob;
      if (format === "mp3") {
        blob = file.type === "mp3" && !selectionNeeded()
          ? await retagMp3(file, tags)
          : await exportMp3(file, state.sel.start, state.sel.end, parseInt(ui.saveBitrate.value, 10));
      } else {
        blob = await remuxWithTags(file, tags, format);
      }
      addClip({ name, type: format, blob, size: blob.size, edited: false });
      markEdited();
      const elapsed = (performance.now() - t0) / 1000;
      const speed = `${(file.duration / Math.max(elapsed, 0.001)).toFixed(0)}×`;
      log(ui.saveLog, `Saved ${name} · ${fmtBytes(blob.size)} in ${elapsed.toFixed(2)} s (${speed} real time)`);
      setStatus("saved", "ok");
      downloadBlob(blob, name);
    } catch (err) {
      log(ui.saveLog, "Save failed: " + (err && err.message ? err.message : String(err)));
      setStatus("idle");
    } finally {
      setBusy(false);
    }
  }

  /** Copy the existing encoded tracks and replace the metadata. */
  async function remuxWithTags(file, tags, format) {
    showProgress(ui.saveProgress, true, 0, "muxing…");
    const target = new BufferTarget();
    const output = new Output({ format: new Mp4OutputFormat({ fastStart: "in-memory" }), target });
    let videoSrc = null, audioSrc = null;
    if (file.videoTrack) {
      videoSrc = new MB.EncodedVideoPacketSource("avc");
      output.addVideoTrack(videoSrc, { frameRate: (file.video && file.video.fps) || 1 });
    }
    if (file.audioTrack) {
      let codec = "mp3";
      try { codec = (await file.audioTrack.getCodec()) || "mp3"; } catch (e) {}
      if (codec !== "mp3" && codec !== "aac" && codec !== "opus") codec = "aac";
      audioSrc = new MB.EncodedAudioPacketSource(codec);
      output.addAudioTrack(audioSrc, { numberOfChannels: file.audioTrack.numberOfChannels, sampleRate: file.audioTrack.sampleRate });
    }
    if (format === "m4a" && videoSrc) throw new Error("M4A is audio-only — pick MP4 to keep the video.");
    output.setMetadataTags(metadataTags(tags));
    await output.start();
    const dur = Math.max(file.duration, 1e-6);

    if (videoSrc) {
      const sink = new MB.EncodedPacketSink(file.videoTrack);
      let first = true;
      for await (const packet of sink.packets()) {
        const meta = first ? { decoderConfig: await file.videoTrack.getDecoderConfig() } : undefined;
        await videoSrc.add(packet, meta);
        first = false;
        showProgress(ui.saveProgress, true, 0.5 * clamp(packet.timestamp / dur, 0, 1), "copying video…");
      }
    }
    if (audioSrc) {
      const sink = new MB.EncodedPacketSink(file.audioTrack);
      let first = true;
      for await (const packet of sink.packets()) {
        const meta = first ? { decoderConfig: await file.audioTrack.getDecoderConfig() } : undefined;
        await audioSrc.add(packet, meta);
        first = false;
        showProgress(ui.saveProgress, true, 0.5 + 0.5 * clamp(packet.timestamp / dur, 0, 1), "copying audio…");
      }
    }
    await output.finalize();
    showProgress(ui.saveProgress, false);
    return new Blob([target.buffer], { type: format === "m4a" ? "audio/mp4" : "video/mp4" });
  }

  /** MP3 retag: frames copied verbatim (no re-encode). */
  async function retagMp3(file, tags) {
    const bytes = new Uint8Array(await file.blob.arrayBuffer());
    const parsed = parseMp3Frames(bytes);
    if (!parsed.frames.length) throw new Error("No MP3 frames found.");
    return writeMp3Frames(parsed.frames.map((f) => bytes.subarray(f.offset, f.offset + f.length)), parsed.sampleRate, parsed.spf, tags);
  }

  // ================================================================== export (cut + encode)
  function sliceAudio(file, start, end) {
    const rate = file.audio.sampleRate;
    const from = Math.floor(start * rate);
    const to = Math.min(Math.floor(end * rate), file.audio.buffer.length);
    const len = Math.max(1, to - from);
    const ab = new AudioBuffer({ length: len, numberOfChannels: 2, sampleRate: rate });
    for (let ch = 0; ch < 2; ch++) {
      const src = file.audio.buffer.getChannelData(Math.min(ch, file.audio.buffer.numberOfChannels - 1));
      ab.copyToChannel(src.subarray(Math.min(from, src.length), Math.min(from + len, src.length)), ch);
    }
    return ab;
  }

  async function exportSelection() {
    const file = state.file;
    if (!file || state.busy) return;
    readRangeInputs();
    const { start, end } = state.sel;
    const format = ui.saveFormat.value;
    const bitrate = parseInt(ui.saveBitrate.value, 10) || 192000;
    if (end - start < 0.02) { log(ui.editLog, "The selection is empty — set Start and End first."); return; }
    setBusy(true);
    setStatus("exporting…", "busy");
    const t0 = performance.now();
    showProgress(ui.saveProgress, true, 0, "preparing…");
    try {
      const tags = currentTags();
      const name = `${baseName(tags.title || file.name)}.${format}`;
      let blob;
      if (format === "mp3") blob = await exportMp3(file, start, end, bitrate);
      else if (format === "m4a") blob = await exportToMp4(file, start, end, bitrate, tags, false);
      else blob = await exportToMp4(file, start, end, bitrate, tags, true);
      const elapsed = (performance.now() - t0) / 1000;
      appendLog(ui.editLog, `Exported ${name} · ${fmtBytes(blob.size)} · ${(end - start).toFixed(2)} s in ${elapsed.toFixed(2)} s (${((end - start) / Math.max(elapsed, 0.001)).toFixed(1)}× real time)`);
      addClip({ name, type: format, blob, size: blob.size, edited: false });
      markEdited();
      setStatus("exported", "ok");
      downloadBlob(blob, name);
    } catch (err) {
      appendLog(ui.editLog, "Export failed: " + (err && err.message ? err.message : String(err)));
      setStatus("idle");
    } finally {
      showProgress(ui.saveProgress, false);
      setBusy(false);
    }
  }

  async function exportMp3(file, start, end, bitrate) {
    if (!file.audio) throw new Error("No audio to export.");
    // Lossless path when the source is MP3: keep whole frames covering the range.
    if (file.type === "mp3") {
      const bytes = new Uint8Array(await file.blob.arrayBuffer());
      const parsed = parseMp3Frames(bytes);
      const frameDur = parsed.spf / parsed.sampleRate;
      const from = Math.max(0, Math.floor(start / frameDur));
      const to = Math.min(parsed.frames.length, Math.ceil(end / frameDur));
      if (to > from) {
        return writeMp3Frames(
          parsed.frames.slice(from, to).map((f) => bytes.subarray(f.offset, f.offset + f.length)),
          parsed.sampleRate, parsed.spf, currentTags());
      }
    }
    const ab = sliceAudio(file, start, end);
    const raw = await encodeMp3Raw(ab, bitrate, ui.saveProgress, 0.05, 0.9);
    const parsed = parseMp3Frames(raw);
    return writeMp3Frames(parsed.frames.map((f) => raw.subarray(f.offset, f.offset + f.length)), parsed.sampleRate, parsed.spf, currentTags());
  }

  async function exportToMp4(file, start, end, bitrate, tags, withVideo) {
    const fps = clamp(parseInt(ui.fpsNum.value, 10) || (file.video && file.video.fps) || 30, 5, 60);
    const outDur = end - start;
    const target = new BufferTarget();
    const output = new Output({ format: new Mp4OutputFormat({ fastStart: "in-memory" }), target });

    // ---- video ----
    let vsrc = null;
    if (withVideo) {
      const W = file.video ? file.video.width : 1280;
      const H = file.video ? file.video.height : 720;
      const canvas = document.createElement("canvas");
      canvas.width = W; canvas.height = H;
      const ctx = canvas.getContext("2d", { alpha: false });
      if (!file.video) {
        const bmp = state.cover ? await createImageBitmap(new Blob([state.cover.bytes], { type: state.cover.mime })).catch(() => null) : null;
        drawCoverFrame(ctx, W, H, bmp, tags.title);
      }
      const px = canvas.width * canvas.height;
      vsrc = new CanvasSource(canvas, { codec: "avc", bitrate: clamp(Math.round(px * fps * 0.12), 1200000, 12000000) });
      output.addVideoTrack(vsrc, { frameRate: fps });
      output._canvasCtx = ctx;
    }

    // ---- audio (MP3 frames, encoded once) ----
    let audioInfo = null;
    if (file.audio) {
      const ab = sliceAudio(file, start, end);
      const raw = await encodeMp3Raw(ab, bitrate, ui.saveProgress, 0.0, 0.45);
      const parsed = parseMp3Frames(raw);
      audioInfo = { frames: parsed.frames.map((f) => raw.subarray(f.offset, f.offset + f.length)), sampleRate: parsed.sampleRate, spf: parsed.spf };
    }

    const asrc = audioInfo ? new EncodedAudioPacketSource("mp3") : null;
    if (asrc) output.addAudioTrack(asrc, { numberOfChannels: 2, sampleRate: audioInfo.sampleRate });
    output.setMetadataTags(metadataTags(tags));
    await output.start();

    if (vsrc) {
      const ctx = output._canvasCtx;
      const t0 = performance.now();
      if (file.video) {
        let i = 0;
        const sink = new MB.VideoSampleSink(file.video.track);
        for await (const sample of sink.samples(start, end)) {
          ctx.drawImage(sample.toCanvasImageSource(), 0, 0, ctx.canvas.width, ctx.canvas.height);
          const ts = Math.max(0, sample.timestamp - start);
          sample.close();
          await vsrc.add(ts, 1 / fps);
          i++;
          if (i % 6 === 0) {
            showProgress(ui.saveProgress, true, 0.45 + 0.5 * clamp((ts / Math.max(outDur, 0.001)), 0, 1),
              `encoding video ${(ts / Math.max((performance.now() - t0) / 1000, 0.001)).toFixed(1)}× real time`);
            await new Promise((r) => setTimeout(r, 0));
          }
        }
      } else {
        await vsrc.add(0, Math.max(outDur, 1 / fps));
      }
    }

    if (asrc) {
      const dur = audioInfo.spf / audioInfo.sampleRate;
      for (let i = 0; i < audioInfo.frames.length; i++) {
        await asrc.add(new EncodedPacket(audioInfo.frames[i], "key", i * dur, dur, i),
          i === 0 ? { decoderConfig: { codec: "mp3", sampleRate: audioInfo.sampleRate, numberOfChannels: 2 } } : undefined);
      }
    }
    showProgress(ui.saveProgress, true, 0.99, "finalizing…");
    await output.finalize();
    return new Blob([target.buffer], { type: withVideo ? "video/mp4" : "audio/mp4" });
  }

  // ================================================================== misc ui
  function setBusy(busy) {
    state.busy = busy;
    for (const b of [ui.btnSave, ui.btnExport, ui.btnRecord, ui.btnOpen, ui.btnSaveAs]) b.disabled = busy;
    if (busy) ui.btnRecord.disabled = true;
  }

  // --- record
  ui.btnRecord.onclick = () => startRecording();
  ui.btnStop.onclick = () => stopRecording();

  // --- open / save
  ui.btnOpen.onclick = () => ui.fileInput.click();
  ui.fileInput.onchange = async () => {
    const files = [...(ui.fileInput.files || [])];
    ui.fileInput.value = "";
    let last = null;
    for (const f of files) {
      const type = f.name.toLowerCase().endsWith(".mp3") ? "mp3" : f.name.toLowerCase().endsWith(".m4a") ? "m4a" : "mp4";
      last = { name: f.name, type, blob: f, size: f.size, edited: false };
      addClip(last);
    }
    if (last) await selectClip(last);
  };
  ui.btnSaveAs.onclick = () => saveCurrent();

  // --- transport
  ui.btnPlay.onclick = () => togglePlay();
  ui.seek.addEventListener("pointerdown", () => { ui.seek._dragging = true; });
  ui.seek.addEventListener("pointerup", () => {
    ui.seek._dragging = false;
    seekTo((ui.seek.value / 1000) * totalDuration());
  });
  ui.seek.addEventListener("input", () => {
    if (!state.file) return;
    const at = (ui.seek.value / 1000) * totalDuration();
    updateTimecode(at);
    holdPlayhead(at);
  });
  const step = () => 1 / clamp(parseInt(ui.fpsNum.value, 10) || 30, 5, 60);
  ui.btnFrameBack.onclick = () => seekTo(position() - step());
  ui.btnFrameFwd.onclick = () => seekTo(position() + step());
  ui.chkMute.onchange = () => {
    audioEl.muted = ui.chkMute.checked;
    ui.preview.muted = ui.chkMute.checked;
  };
  ui.preview.addEventListener("timeupdate", () => { if (state.playing) { updateTimecode(ui.preview.currentTime); holdPlayhead(ui.preview.currentTime); } });
  ui.preview.addEventListener("ended", () => pausePlayback());
  audioEl.addEventListener("ended", () => pausePlayback());
  ui.preview.addEventListener("loadedmetadata", () => { if (state.file && !state.playing) seekTo(state.pos); });

  // --- trim
  ui.inSec.onchange = readRangeInputs;
  ui.outSec.onchange = readRangeInputs;
  ui.btnSelStart.onclick = () => setRange(position(), Math.max(state.sel.end, position()));
  ui.btnSelEnd.onclick = () => setRange(Math.min(state.sel.start, position()), position());
  ui.btnSelClear.onclick = () => setRange(0, totalDuration());
  ui.btnSelToClip.onclick = () => setRange(0, totalDuration());
  let waveDragging = false;
  ui.wave.addEventListener("pointerdown", (e) => {
    if (!state.file) return;
    waveDragging = true;
    ui.wave.setPointerCapture(e.pointerId);
    seekByWave(e);
  });
  ui.wave.addEventListener("pointermove", (e) => { if (waveDragging) seekByWave(e); });
  ui.wave.addEventListener("pointerup", () => { waveDragging = false; });
  function seekByWave(e) {
    const r = ui.wave.getBoundingClientRect();
    seekTo(clamp((e.clientX - r.left) / r.width, 0, 1) * totalDuration());
  }

  for (const t of [ui.trackVideo, ui.trackAudio]) {
    t.addEventListener("click", (e) => {
      if (!state.file) return;
      const r = t.getBoundingClientRect();
      seekTo(clamp((e.clientX - r.left) / r.width, 0, 1) * totalDuration());
    });
  }

  // --- save / export
  ui.btnSave.onclick = () => saveCurrent();
  ui.btnExport.onclick = () => exportSelection();
  ui.btnReset.onclick = () => {
    setRange(0, totalDuration());
    if (state.file) fillTagFields(state.file.tags, state.file.cover);
    log(ui.editLog, "Edits reset.");
  };
  ui.saveFormat.onchange = () => {
    if (state.file && !state.file.video && ui.saveFormat.value === "mp4") ui.saveFormat.value = "mp3";
  };

  // --- cover
  ui.coverInput.onchange = async () => {
    const f = (ui.coverInput.files || [])[0];
    if (!f) return;
    setCoverFromBytes(new Uint8Array(await f.arrayBuffer()), f.type || "image/jpeg");
  };
  ui.btnUseFrame.onclick = () => grabFrameCover();
  ui.btnClearCover.onclick = () => { state.cover = null; paintCover(); };

  // --- keyboard
  window.addEventListener("keydown", (e) => {
    const tag = (e.target.tagName || "").toLowerCase();
    if (tag === "input" || tag === "textarea" || tag === "select") return;
    if (e.code === "Space") { e.preventDefault(); togglePlay(); }
    else if (e.code === "ArrowLeft") { e.preventDefault(); seekTo(position() - step()); }
    else if (e.code === "ArrowRight") { e.preventDefault(); seekTo(position() + step()); }
    else if (e.key === "i" || e.key === "I") setRange(position(), Math.max(state.sel.end, position()));
    else if (e.key === "o" || e.key === "O") setRange(Math.min(state.sel.start, position()), position());
  });

  window.addEventListener("resize", () => drawWaveform());
  window.addEventListener("beforeunload", (e) => {
    if (rec) { e.preventDefault(); e.returnValue = ""; }
  });

  // --- init
  drawIdleView();
  drawWaveform();
  updateSelInfo();
  renderFileList();
  setStatus("idle");
})();
