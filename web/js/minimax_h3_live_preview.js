import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// Frontend half of minimax_h3/live_preview.py: every sampler step the backend sends one JPEG
// sprite sheet of the latent-to-RGB video frames; this widget plays it back on the node at the
// latent frame rate. Pointer and wheel events on the preview surface are re-dispatched to the
// graph canvas, so selecting the node, its context menu and wheel zoom work over the preview
// like over any other widget; only the small control bar handles its own clicks.

const NODE_CLASS = "MiniMaxH3LivePreview";
const EVENT = "minimax_h3_live_preview";
const STATE = Symbol("minimaxH3LivePreviewState");
const STYLE_ID = "minimax-h3-live-preview-style";
const SPEEDS = [0.5, 1, 2];

const CSS = `
.minimax-h3-live-preview {
    display: flex; flex-direction: column; gap: 4px; width: 100%; height: 100%;
    box-sizing: border-box; padding: 2px; font-family: sans-serif; user-select: none;
}
.minimax-h3-live-preview-stage {
    position: relative; flex: 1 1 auto; min-height: 0; border-radius: 6px; overflow: hidden;
    background: #101014;
}
.minimax-h3-live-preview-stage canvas { position: absolute; inset: 0; width: 100%; height: 100%; }
.minimax-h3-live-preview-idle {
    position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
    color: #8a8a94; font-size: 12px; text-align: center; padding: 8px;
}
.minimax-h3-live-preview-bar {
    display: flex; align-items: center; gap: 6px; flex: 0 0 auto; min-height: 20px;
    color: #cfcfd6; font-size: 11px; line-height: 1.2; white-space: nowrap; overflow: hidden;
}
.minimax-h3-live-preview-bar button {
    cursor: pointer; border: 1px solid #3a3a44; border-radius: 4px;
    background: #22222a; color: #e6e6ec; font-size: 11px; line-height: 1; padding: 3px 6px;
    min-width: 28px;
}
.minimax-h3-live-preview-bar button:hover { background: #2e2e38; }
.minimax-h3-live-preview-status { flex: 1 1 auto; overflow: hidden; text-overflow: ellipsis; }
`;

function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = CSS;
    document.head.appendChild(style);
}

// Prompt ids of nodes inside subgraphs are "parent:child[:grandchild]" paths.
function findNodeByExecutionId(executionId) {
    const parts = String(executionId).split(":");
    let graph = app.graph;
    for (let i = 0; i < parts.length - 1; i++) {
        const parent = graph?.getNodeById?.(Number(parts[i]));
        if (!parent?.subgraph) return null;
        graph = parent.subgraph;
    }
    return graph?.getNodeById?.(Number(parts[parts.length - 1])) ?? null;
}

function nodeLabel(executionId) {
    if (executionId == null) return "";
    const parts = String(executionId).split(":");
    // the top-level ancestor is what the user sees on the canvas (e.g. the preset subgraph)
    const node = findNodeByExecutionId(parts[0]);
    return node?.title || node?.type || "";
}

function formatSeconds(seconds) {
    if (!Number.isFinite(seconds)) return "—";
    if (seconds < 10) return `${seconds.toFixed(1)} s`;
    if (seconds < 90) return `${Math.round(seconds)} s`;
    const minutes = Math.floor(seconds / 60);
    return `${minutes} min ${Math.round(seconds - minutes * 60)} s`;
}

function base64ToBlob(base64, type) {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return new Blob([bytes], { type });
}

function installWidget(node) {
    if (node.comfyClass !== NODE_CLASS || node[STATE]) return;
    ensureStyle();

    const root = document.createElement("div");
    root.className = "minimax-h3-live-preview";
    const stage = document.createElement("div");
    stage.className = "minimax-h3-live-preview-stage";
    const canvas = document.createElement("canvas");
    const idle = document.createElement("div");
    idle.className = "minimax-h3-live-preview-idle";
    idle.textContent = "Live latent preview appears here while MiniMax H3 samples";
    stage.append(canvas, idle);
    const bar = document.createElement("div");
    bar.className = "minimax-h3-live-preview-bar";
    const playButton = document.createElement("button");
    playButton.title = "Pause / resume playback";
    playButton.textContent = "⏸";
    const speedButton = document.createElement("button");
    speedButton.title = "Playback speed (1× = the clip's real time)";
    speedButton.textContent = "1×";
    const status = document.createElement("div");
    status.className = "minimax-h3-live-preview-status";
    status.textContent = "Waiting for sampling…";
    bar.append(playButton, speedButton, status);
    root.append(stage, bar);

    // The DOM widget sits above the graph canvas. Hand everything that is not aimed at the
    // control bar back to the canvas as a cloned event, so click-to-select, the node context
    // menu and wheel zoom keep working over the preview (dragging still happens by the title).
    const forwardToCanvas = (event) => {
        if (bar.contains(event.target)) {
            event.stopPropagation();
            return;
        }
        const graphCanvas = app.canvas?.canvas;
        if (!graphCanvas) return;
        graphCanvas.dispatchEvent(new event.constructor(event.type, event));
        event.preventDefault();
        event.stopPropagation();
    };
    for (const eventName of ["pointerdown", "pointermove", "pointerup", "dblclick", "contextmenu"]) {
        root.addEventListener(eventName, forwardToCanvas);
    }
    root.addEventListener("wheel", forwardToCanvas, { passive: false });

    const context = canvas.getContext("2d");
    const state = {
        sheet: null, cols: 1, rows: 1, frameW: 1, frameH: 1, frames: 0, fps: 7,
        playing: true, speedIndex: 1, clock: 0, lastTick: null, raf: null,
        stepTimes: [], run: null, done: false, node,
    };
    node[STATE] = state;

    function draw() {
        const width = stage.clientWidth;
        const height = stage.clientHeight;
        if (!state.sheet || width < 2 || height < 2) return;
        const scale = Math.min(window.devicePixelRatio || 1, 2);
        const backingW = Math.round(width * scale);
        const backingH = Math.round(height * scale);
        if (canvas.width !== backingW || canvas.height !== backingH) {
            canvas.width = backingW;
            canvas.height = backingH;
        }
        const clipSeconds = state.frames / state.fps;
        const position = clipSeconds > 0 ? (state.clock % clipSeconds) / clipSeconds : 0;
        const index = Math.min(state.frames - 1, Math.floor(position * state.frames));
        const sx = (index % state.cols) * state.frameW;
        const sy = Math.floor(index / state.cols) * state.frameH;
        const fit = Math.min(backingW / state.frameW, backingH / state.frameH);
        const dw = Math.round(state.frameW * fit);
        const dh = Math.round(state.frameH * fit);
        const dx = Math.round((backingW - dw) / 2);
        const dy = Math.round((backingH - dh) / 2);
        context.imageSmoothingEnabled = true;
        context.imageSmoothingQuality = "high";
        context.clearRect(0, 0, backingW, backingH);
        context.drawImage(state.sheet, sx, sy, state.frameW, state.frameH, dx, dy, dw, dh);
    }

    function tick(now) {
        state.raf = requestAnimationFrame(tick);
        if (state.lastTick != null && state.playing) {
            state.clock += ((now - state.lastTick) / 1000) * SPEEDS[state.speedIndex];
        }
        state.lastTick = now;
        draw();
    }

    function startLoop() {
        if (state.raf == null) state.raf = requestAnimationFrame(tick);
    }

    function stopLoop() {
        if (state.raf != null) cancelAnimationFrame(state.raf);
        state.raf = null;
        state.lastTick = null;
    }

    playButton.addEventListener("click", (event) => {
        event.stopPropagation();
        state.playing = !state.playing;
        playButton.textContent = state.playing ? "⏸" : "▶";
    });
    speedButton.addEventListener("click", (event) => {
        event.stopPropagation();
        state.speedIndex = (state.speedIndex + 1) % SPEEDS.length;
        speedButton.textContent = `${SPEEDS[state.speedIndex]}×`;
    });

    function updateStatus(data) {
        const total = data.total || 0;
        const step = data.step || 0;
        const clipSeconds = data.video_frames / 24;
        let text = `step ${step}/${total} · ${data.latent_frames} latent frames → ${data.video_frames} frames (${clipSeconds.toFixed(1)} s)`;
        const times = state.stepTimes;
        if (times.length) {
            const average = times.reduce((sum, value) => sum + value, 0) / times.length / 1000;
            text += ` · ${average.toFixed(2)} s/step`;
            if (step < total) text += ` · ETA ${formatSeconds(average * (total - step))}`;
        }
        if (step >= total && total > 0) text += " · done";
        const label = nodeLabel(data.sampler);
        status.textContent = label ? `${label}: ${text}` : text;
        status.title = status.textContent;
    }

    state.handle = async (data) => {
        const runKey = `${data.sampler}`;
        if (data.step <= 1 || state.run !== runKey) {
            state.stepTimes = [];
            state.run = runKey;
        }
        if (Number.isFinite(data.step_ms)) state.stepTimes.push(data.step_ms);
        state.done = data.step >= data.total;
        updateStatus(data);
        let sheet;
        try {
            sheet = await createImageBitmap(base64ToBlob(data.sheet, "image/jpeg"));
        } catch (error) {
            console.warn("[MiniMaxH3LivePreview] could not decode preview sheet", error);
            return;
        }
        if (!node[STATE]) {
            sheet.close?.();
            return;
        }
        state.sheet?.close?.();
        state.sheet = sheet;
        state.cols = data.cols;
        state.rows = data.rows;
        state.frameW = data.w;
        state.frameH = data.h;
        state.frames = data.shown_frames;
        state.fps = data.fps || state.fps;
        if (idle.parentNode) idle.remove();
        startLoop();
    };

    state.interrupted = () => {
        if (state.sheet && !state.done) status.textContent = `${status.textContent} · interrupted`;
    };

    const widget = node.addDOMWidget("live_preview", "MINIMAX_H3_LIVE_PREVIEW", root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 140,
        getValue: () => undefined,
        setValue: () => {},
    });
    widget.serialize = false;

    const resizeObserver = new ResizeObserver(() => draw());
    resizeObserver.observe(stage);

    const size = node.size ?? [0, 0];
    node.setSize([Math.max(size[0], 420), Math.max(size[1], 360)]);

    const onRemoved = node.onRemoved;
    node.onRemoved = function () {
        stopLoop();
        resizeObserver.disconnect();
        state.sheet?.close?.();
        state.sheet = null;
        node[STATE] = null;
        return onRemoved?.apply(this, arguments);
    };
}

api.addEventListener(EVENT, ({ detail }) => {
    if (!detail || !Array.isArray(detail.nodes)) return;
    for (const executionId of detail.nodes) {
        const node = findNodeByExecutionId(executionId);
        node?.[STATE]?.handle?.(detail);
    }
});

api.addEventListener("execution_interrupted", () => {
    for (const graph of [app.graph, ...(app.graph?.subgraphs?.values?.() ?? [])]) {
        for (const node of graph?._nodes ?? []) node[STATE]?.interrupted?.();
    }
});

app.registerExtension({
    name: "TeaCache.MiniMaxH3LivePreview",
    nodeCreated(node) {
        installWidget(node);
    },
});
