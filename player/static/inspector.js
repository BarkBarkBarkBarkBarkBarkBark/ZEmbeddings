const state = {
    runs: [],
    selectedRunId: null,
    preview: null,
    socket: null,
    currentFrameIndex: -1,
    segments: [],
    lastBoundaryIndex: -1,
    playing: false,
    trajectory: {
        scene: null,
        camera: null,
        renderer: null,
        controls: null,
        line: null,
        boundaries: null,
        currentMarker: null,
    },
};

const els = {
    runList: document.getElementById('run-list'),
    schemaBrowser: document.getElementById('schema-browser'),
    dbStatus: document.getElementById('db-status'),
    serverStatus: document.getElementById('server-status'),
    runSource: document.getElementById('run-source'),
    stepReadout: document.getElementById('step-readout'),
    boundaryBadge: document.getElementById('boundary-badge'),
    attentionWindow: document.getElementById('attention-window'),
    streamText: document.getElementById('stream-text'),
    recomputeBtn: document.getElementById('recompute-btn'),
    playBtn: document.getElementById('play-btn'),
    stopBtn: document.getElementById('stop-btn'),
    resetViewBtn: document.getElementById('reset-view-btn'),
    windowSize: document.getElementById('window-size'),
    stride: document.getElementById('stride'),
    kSigma: document.getElementById('k-sigma'),
    emaAlpha: document.getElementById('ema-alpha'),
    kalmanMode: document.getElementById('kalman-mode'),
    paceMs: document.getElementById('pace-ms'),
    paceReadout: document.getElementById('pace-readout'),
    trajectoryCanvas: document.getElementById('trajectory-canvas'),
    transcriptTokenHint: document.getElementById('transcript-token-hint'),
    replayEstimate: document.getElementById('replay-estimate'),
    replayNote: document.getElementById('replay-note'),
    fullTranscript: document.getElementById('full-transcript'),
    metrics: {
        windows: document.getElementById('metric-windows'),
        boundaries: document.getElementById('metric-boundaries'),
        velocity: document.getElementById('metric-velocity'),
        path: document.getElementById('metric-path'),
        kalman: document.getElementById('metric-kalman'),
        kalmanAccel: document.getElementById('metric-kalman-accel'),
        ema: document.getElementById('metric-ema'),
        returnCluster: document.getElementById('metric-return'),
    },
};

function init3D() {
    const { trajectoryCanvas: canvas } = els;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0d1117);

    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    camera.position.set(2.5, 2.2, 3.4);

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    const controls = new THREE.OrbitControls(camera, canvas);
    controls.enableDamping = true;

    scene.add(new THREE.GridHelper(5, 24, 0x30363d, 0x1c2128));
    scene.add(new THREE.AxesHelper(1.5));

    const currentMarker = new THREE.Mesh(
        new THREE.SphereGeometry(0.07, 18, 18),
        new THREE.MeshBasicMaterial({ color: 0xffffff })
    );
    currentMarker.visible = false;
    scene.add(currentMarker);

    state.trajectory = { scene, camera, renderer, controls, line: null, boundaries: null, currentMarker };

    function resize() {
        const parent = canvas.parentElement;
        const w = Math.max(300, parent.clientWidth - 28);
        const h = Math.max(260, parent.clientHeight - 70);
        renderer.setSize(w, h, false);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
    }

    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }

    window.addEventListener('resize', resize);
    resize();
    animate();
}

function velocityColor(value) {
    const safe = value ?? 0;
    const t = Math.max(0, Math.min(safe / 0.18, 1));
    const hue = 220 - (220 * t);
    return `hsl(${hue}, 90%, 65%)`;
}

function renderRuns() {
    els.runList.innerHTML = '';
    if (!state.runs.length) {
        els.runList.innerHTML = '<div class="muted">No metrics YAML files found yet.</div>';
        return;
    }

    state.runs.forEach((run) => {
        const item = document.createElement('button');
        item.className = `run-item${run.id === state.selectedRunId ? ' active' : ''}`;
        item.type = 'button';
        item.innerHTML = `
            <div><strong>${escapeHtml(run.name)}</strong></div>
            <div class="meta">
                <span>${run.windows ?? '—'} windows</span>
                <span>${run.boundaries ?? '—'} boundaries</span>
            </div>
            <div class="meta">
                <span>${escapeHtml(run.timestamp || 'unknown')}</span>
                <span>${run.source_exists ? 'source ok' : 'source missing'}</span>
            </div>
        `;
        item.addEventListener('click', () => selectRun(run.id));
        els.runList.appendChild(item);
    });
}

function renderSchema(schema) {
    els.schemaBrowser.innerHTML = '';
    els.dbStatus.className = `pill ${schema.status === 'ok' ? 'ok' : 'warn'}`;
    els.dbStatus.textContent = schema.status === 'ok' ? 'db reachable' : 'db unavailable';

    if (schema.error) {
        const error = document.createElement('div');
        error.className = 'muted';
        error.textContent = schema.error.includes('connection to server at')
            ? 'PostgreSQL is not running locally. Schema browsing is optional for the inspector.'
            : schema.error;
        els.schemaBrowser.appendChild(error);
    }

    schema.tables.forEach((table) => {
        const details = document.createElement('details');
        details.className = 'schema-table';
        details.open = ['conversations', 'experiments', 'embeddings', 'metrics'].includes(table.name);
        const columns = table.columns.map((column) => `
            <div>
                <span>${escapeHtml(column.name)}</span>
                <span class="muted">${escapeHtml(column.udt_name || column.data_type)}${column.nullable ? '' : ' · not null'}</span>
            </div>
        `).join('');
        details.innerHTML = `
            <summary>${escapeHtml(table.name)} <span class="muted">(${table.columns.length})</span></summary>
            <div class="schema-columns">${columns}</div>
        `;
        els.schemaBrowser.appendChild(details);
    });
}

function selectRun(runId) {
    state.selectedRunId = runId;
    renderRuns();
    const run = state.runs.find((entry) => entry.id === runId);
    if (!run) return;
    els.windowSize.value = run.params.window.size;
    els.stride.value = run.params.window.stride;
    els.kSigma.value = run.params.boundary.k_sigma;
    els.emaAlpha.value = run.params.ema.alpha;
    els.kalmanMode.value = run.params.kalman.mode;
    els.runSource.textContent = run.source || 'Source transcript unavailable';
}

async function recomputePreview({ autoplay = false } = {}) {
    if (!state.selectedRunId) return;
    stopPlayback();
    els.serverStatus.textContent = 'computing';
    els.serverStatus.className = 'pill warn';
    els.recomputeBtn.disabled = true;

    try {
        const response = await fetch('/api/recompute', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                run_id: state.selectedRunId,
                window_size: Number(els.windowSize.value),
                stride: Number(els.stride.value),
                k_sigma: Number(els.kSigma.value),
                ema_alpha: Number(els.emaAlpha.value),
                kalman_mode: els.kalmanMode.value,
            }),
        });
        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(parseErrorDetail(error, 'Failed to recompute preview'));
        }

        state.preview = await response.json();
        state.currentFrameIndex = -1;
        state.lastBoundaryIndex = -1;
        state.segments = [];
        els.serverStatus.textContent = 'ready';
        els.serverStatus.className = 'pill ok';
        els.runSource.textContent = `${state.preview.run.source} · ${state.preview.summary.windows} windows`;
        drawTrajectory(state.preview.trajectory_3d, state.preview.timeseries.velocity, state.preview.timeseries.boundary_flags, state.preview.timeseries.kalman_accel_violations);
        renderSummary(state.preview.summary);
        seedUiAfterPreview();
        if (autoplay) startPlayback();
    } catch (error) {
        console.error(error);
        els.serverStatus.textContent = 'error';
        els.serverStatus.className = 'pill alert';
        els.runSource.textContent = error.message;
    } finally {
        els.recomputeBtn.disabled = false;
    }
}

function parseErrorDetail(error, fallback) {
    const d = error?.detail;
    if (typeof d === 'string') return d;
    if (Array.isArray(d)) return d.map((x) => x.msg || JSON.stringify(x)).join('; ') || fallback;
    return fallback;
}

function updateReplayEstimate() {
    if (!els.replayEstimate) return;
    if (!state.preview?.frames?.length) {
        els.replayEstimate.textContent = '';
        return;
    }
    const n = state.preview.frames.length;
    const ms = Math.round(Number(els.paceMs.value) || 160);
    const sec = (n * ms) / 1000;
    els.replayEstimate.textContent = `At ${ms} ms/step, full replay ≈ ${sec.toFixed(1)} s (${n} windows). Not wall-clock speech time.`;
}

function seedUiAfterPreview() {
    if (els.replayNote && state.preview?.replay?.note) {
        els.replayNote.textContent = state.preview.replay.note;
    }
    if (els.transcriptTokenHint && state.preview?.summary?.tokens != null) {
        els.transcriptTokenHint.textContent = `${state.preview.summary.tokens} tokens`;
    }
    if (els.fullTranscript) {
        els.fullTranscript.textContent = state.preview.transcript_text || '';
    }
    updateReplayEstimate();
    resetStreamArea();
    if (state.preview?.frames?.length) {
        const f0 = state.preview.frames[0];
        state.currentFrameIndex = 0;
        els.stepReadout.textContent = `window 1 / ${state.preview.frames.length} · press Play to replay`;
        els.attentionWindow.textContent = f0.window_text;
        updateCurrentMarker(0);
        drawAllSparklines(state.preview.timeseries, 0);
    } else {
        drawAllSparklines(state.preview.timeseries, -1);
    }
}

function streamWsHref(previewId, paceMs) {
    const u = new URL(`/ws/stream/${encodeURIComponent(previewId)}`, window.location.href);
    u.protocol = u.protocol === 'https:' ? 'wss:' : 'ws:';
    u.searchParams.set('pace_ms', String(Math.round(Number(paceMs) || 160)));
    return u.href;
}

function renderSummary(summary) {
    els.metrics.windows.textContent = summary.windows;
    els.metrics.boundaries.textContent = summary.boundaries;
    els.metrics.velocity.textContent = summary.mean_velocity.toFixed(4);
    els.metrics.path.textContent = summary.path_length.toFixed(4);
    els.metrics.kalman.textContent = summary.kalman_violations;
    els.metrics.kalmanAccel.textContent = summary.kalman_accel_violations;
    els.metrics.ema.textContent = '—';
    els.metrics.returnCluster.textContent = '—';
}

function drawTrajectory(coords3d, velocities, boundaries, accelViolations) {
    const { scene, currentMarker } = state.trajectory;
    removeTrajectoryObjects();
    currentMarker.visible = false;

    if (!coords3d || coords3d.length < 2) return;

    const positions = [];
    const colors = [];
    coords3d.forEach((coord, index) => {
        positions.push(coord[0], coord[1], coord[2]);
        const color = new THREE.Color(velocityColor(velocities[index]));
        colors.push(color.r, color.g, color.b);
    });

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    const line = new THREE.Line(
        geometry,
        new THREE.LineBasicMaterial({ vertexColors: true })
    );
    scene.add(line);
    state.trajectory.line = line;

    const boundaryGroup = new THREE.Group();
    const markerGeo = new THREE.SphereGeometry(0.04, 12, 12);
    coords3d.forEach((coord, index) => {
        if (!boundaries[index] && !accelViolations[index]) return;
        const mesh = new THREE.Mesh(
            markerGeo,
            new THREE.MeshBasicMaterial({ color: 0xf85149, transparent: true, opacity: 0.75 })
        );
        mesh.position.set(coord[0], coord[1], coord[2]);
        boundaryGroup.add(mesh);
    });
    scene.add(boundaryGroup);
    state.trajectory.boundaries = boundaryGroup;
}

function removeTrajectoryObjects() {
    const { scene, line, boundaries } = state.trajectory;
    if (line) {
        scene.remove(line);
        line.geometry.dispose();
        line.material.dispose();
        state.trajectory.line = null;
    }
    if (boundaries) {
        boundaries.children.forEach((child) => {
            child.geometry.dispose();
            child.material.dispose();
        });
        scene.remove(boundaries);
        state.trajectory.boundaries = null;
    }
}

function resetStreamArea() {
    els.stepReadout.textContent = 'step 0';
    els.boundaryBadge.textContent = 'stable';
    els.boundaryBadge.className = 'pill muted';
    els.attentionWindow.textContent = 'Attention window will appear here.';
    els.attentionWindow.classList.remove('flash');
    els.streamText.innerHTML = '';
}

function startPlayback() {
    if (!state.preview || state.playing) return;
    stopPlayback();
    state.playing = true;
    const paceMs = Number(els.paceMs.value);
    const ws = new WebSocket(streamWsHref(state.preview.preview_id, paceMs));
    state.socket = ws;

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'start') {
            els.streamText.innerHTML = '';
            state.segments = [];
            state.lastBoundaryIndex = -1;
        }
        if (data.type === 'frame') {
            applyFrame(data.frame);
        }
        if (data.type === 'done') {
            state.playing = false;
        }
    };
    ws.onclose = () => {
        state.playing = false;
        if (state.socket === ws) state.socket = null;
    };
}

function stopPlayback() {
    state.playing = false;
    if (state.socket) {
        state.socket.close();
        state.socket = null;
    }
}

function applyFrame(frame) {
    state.currentFrameIndex = frame.index;
    els.stepReadout.textContent = `step ${frame.index + 1}`;
    els.attentionWindow.textContent = frame.window_text;
    if (frame.boundary_flag) {
        state.lastBoundaryIndex = frame.index;
        els.boundaryBadge.textContent = `boundary @ ${frame.index}`;
        els.boundaryBadge.className = 'pill alert';
        els.attentionWindow.classList.add('flash');
        window.setTimeout(() => els.attentionWindow.classList.remove('flash'), 220);
        const separator = document.createElement('span');
        separator.className = 'boundary-separator';
        separator.textContent = '│';
        els.streamText.appendChild(separator);
    } else {
        els.boundaryBadge.textContent = 'stable';
        els.boundaryBadge.className = 'pill muted';
    }

    const segment = document.createElement('span');
    segment.className = 'segment';
    segment.textContent = frame.delta_text;
    segment.style.color = velocityColor(frame.velocity);
    els.streamText.appendChild(segment);
    state.segments.push({ node: segment, index: frame.index, velocity: frame.velocity });
    updateSegmentDecay();
    els.streamText.scrollTop = els.streamText.scrollHeight;

    updateLiveMetrics(frame);
    updateCurrentMarker(frame.index);
    drawAllSparklines(state.preview.timeseries, frame.index);
}

function updateSegmentDecay() {
    const alpha = Number(els.emaAlpha.value || 0.3);
    const decayRate = 0.08 + alpha * 0.24;
    state.segments.forEach((segment) => {
        const age = state.currentFrameIndex - segment.index;
        const boundaryDistance = segment.index > state.lastBoundaryIndex
            ? age
            : state.currentFrameIndex - state.lastBoundaryIndex + age;
        const opacity = Math.max(0.12, Math.exp(-boundaryDistance * decayRate));
        segment.node.style.opacity = opacity.toFixed(3);
    });
}

function updateLiveMetrics(frame) {
    els.metrics.windows.textContent = `${frame.index + 1}/${state.preview.summary.windows}`;
    els.metrics.boundaries.textContent = state.preview.timeseries.boundary_flags.slice(0, frame.index + 1).filter(Boolean).length;
    els.metrics.velocity.textContent = fmt(frame.velocity);
    els.metrics.path.textContent = fmt(frame.cumulative_path);
    els.metrics.kalman.textContent = fmt(frame.kalman_mahalanobis);
    els.metrics.kalmanAccel.textContent = fmt(frame.kalman_accel_mahalanobis);
    els.metrics.ema.textContent = fmt(frame.ema_drift);
    els.metrics.returnCluster.textContent = frame.return_flag ? String(frame.return_cluster_id) : '—';
}

function updateCurrentMarker(index) {
    const coords = state.preview.trajectory_3d[index];
    if (!coords) return;
    const marker = state.trajectory.currentMarker;
    marker.visible = true;
    marker.position.set(coords[0], coords[1], coords[2]);
}

function drawAllSparklines(timeseries, cursorIndex) {
    drawSparkline('spark-velocity', timeseries.velocity, '#58a6ff', cursorIndex);
    drawSparkline('spark-acceleration', timeseries.acceleration, '#bc8cff', cursorIndex);
    drawSparkline('spark-kalman', timeseries.kalman_mahalanobis, '#d29922', cursorIndex);
    drawSparkline('spark-kalman-accel', timeseries.kalman_accel_mahalanobis, '#f85149', cursorIndex);
    drawSparkline('spark-ema', timeseries.ema_drift, '#3fb950', cursorIndex);
}

function drawSparkline(canvasId, values, color, cursorIndex) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const clean = values.filter((value) => value !== null && !Number.isNaN(value));
    if (clean.length < 2) return;
    const min = Math.min(...clean);
    const max = Math.max(...clean);
    const range = max - min || 1;

    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    let first = true;
    values.forEach((value, index) => {
        if (value === null || Number.isNaN(value)) return;
        const x = (index / Math.max(values.length - 1, 1)) * w;
        const y = h - ((value - min) / range) * (h - 8) - 4;
        if (first) {
            ctx.moveTo(x, y);
            first = false;
        } else {
            ctx.lineTo(x, y);
        }
    });
    ctx.stroke();

    if (cursorIndex >= 0) {
        const x = (cursorIndex / Math.max(values.length - 1, 1)) * w;
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(255,255,255,0.45)';
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
    }
}

function fmt(value) {
    if (value === null || value === undefined || Number.isNaN(value)) return '—';
    return Number(value).toFixed(4);
}

function escapeHtml(value) {
    return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
}

async function bootstrap() {
    init3D();
    els.paceMs.addEventListener('input', () => {
        els.paceReadout.textContent = els.paceMs.value;
        updateReplayEstimate();
    });
    els.recomputeBtn.addEventListener('click', () => recomputePreview());
    els.playBtn.addEventListener('click', () => startPlayback());
    els.stopBtn.addEventListener('click', () => stopPlayback());
    els.resetViewBtn.addEventListener('click', () => {
        state.trajectory.camera.position.set(2.5, 2.2, 3.4);
        state.trajectory.controls.target.set(0, 0, 0);
        state.trajectory.controls.update();
    });

    const response = await fetch('/api/bootstrap');
    const payload = await response.json();
    state.runs = payload.runs;
    renderRuns();
    renderSchema(payload.schema);

    if (payload.default_run_id) {
        selectRun(payload.default_run_id);
        await recomputePreview();
    }
}

bootstrap().catch((error) => {
    console.error(error);
    els.serverStatus.textContent = 'error';
    els.serverStatus.className = 'pill alert';
});
