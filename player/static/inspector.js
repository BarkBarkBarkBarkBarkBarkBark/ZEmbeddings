const METRIC_LABEL_HELP = {
    windows: 'Progress: current window index vs total sliding windows in this run.',
    boundaries: 'Count of semantic boundary flags so far (velocity spike vs local statistics).',
    velocity: 'Semantic speed: how fast the embedding moves between consecutive windows (topic change rate).',
    path: 'Cumulative path length through embedding space along the trajectory so far.',
    kalman: 'Kalman Mahalanobis distance: how surprising this position is vs the filter’s prediction (innovation).',
    kalmanAccel: 'Mahalanobis distance on acceleration track — onset of change in the rate of topic motion.',
    ema: 'Drift from the exponential moving average of embeddings (working-memory deviation).',
    return: 'If the model flags a return to a prior semantic cluster, its cluster id; otherwise —.',
};

const METRIC_VALUE_DESC = {
    velocity: 'Rate of change between successive window embeddings. Higher often means a sharper topic or style shift.',
    path: 'Total distance traveled in embedding space. Larger runs cover more semantic ground.',
    kalman: 'Large values mean the new window is poorly predicted by the causal Kalman model — a “surprise” step.',
    kalmanAccel: 'Surprise in acceleration; spikes can align with boundaries even when velocity alone is smooth.',
    ema: 'Distance from the slow-moving EMA context; high values mean the current window diverges from recent gist.',
};

const state = {
    runs: [],
    selectedRunId: null,
    preview: null,
    socket: null,
    currentFrameIndex: -1,
    segments: [],
    lastBoundaryIndex: -1,
    playing: false,
    embeddingStripListeners: false,
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
    dialQ: document.getElementById('dial-q'),
    dialR: document.getElementById('dial-r'),
    dialK: document.getElementById('dial-k'),
    dialThreshold: document.getElementById('dial-threshold'),
    dialQReadout: document.getElementById('dial-q-readout'),
    dialRReadout: document.getElementById('dial-r-readout'),
    dialKReadout: document.getElementById('dial-k-readout'),
    dialThresholdReadout: document.getElementById('dial-threshold-readout'),
};

const KQ_LO = 1e-8;
const KQ_HI = 1e-1;
const KR_LO = 1e-5;
const KR_HI = 1.0;
const KK_LO = 0.05;
const KK_HI = 2.0;
const KTHR_LO = 0.5;
const KTHR_HI = 8.0;

function logSliderToValue(pos, lo, hi) {
    const t = Math.min(100, Math.max(0, pos)) / 100;
    return lo * (hi / lo) ** t;
}

function valueToLogSlider(v, lo, hi) {
    const t = Math.log(Math.max(v, lo) / lo) / Math.log(hi / lo);
    return Math.round(Math.min(100, Math.max(0, t * 100)));
}

function linSliderToValue(pos, lo, hi) {
    const t = Math.min(100, Math.max(0, pos)) / 100;
    return lo + (hi - lo) * t;
}

function valueToLinSlider(v, lo, hi) {
    return Math.round(Math.min(100, Math.max(0, ((v - lo) / (hi - lo)) * 100)));
}

function updateDialReadouts() {
    if (!els.dialQ) return;
    const q = logSliderToValue(Number(els.dialQ.value), KQ_LO, KQ_HI);
    const r = logSliderToValue(Number(els.dialR.value), KR_LO, KR_HI);
    const g = linSliderToValue(Number(els.dialK.value), KK_LO, KK_HI);
    const th = linSliderToValue(Number(els.dialThreshold.value), KTHR_LO, KTHR_HI);
    if (els.dialQReadout) els.dialQReadout.textContent = q.toExponential(2);
    if (els.dialRReadout) els.dialRReadout.textContent = r.toExponential(2);
    if (els.dialKReadout) els.dialKReadout.textContent = g.toFixed(2);
    if (els.dialThresholdReadout) els.dialThresholdReadout.textContent = th.toFixed(2);
}

function setKalmanDialsFromParams(k) {
    if (!k || !els.dialQ) return;
    const q = k.process_noise_scale ?? 1e-4;
    const r = k.measurement_noise_scale ?? 1e-2;
    const g = k.update_gain_scale ?? 1.0;
    const th = k.innovation_threshold ?? 3.0;
    els.dialQ.value = String(Math.max(0, Math.min(100, valueToLogSlider(q, KQ_LO, KQ_HI))));
    els.dialR.value = String(Math.max(0, Math.min(100, valueToLogSlider(r, KR_LO, KR_HI))));
    els.dialK.value = String(Math.max(0, Math.min(100, valueToLinSlider(g, KK_LO, KK_HI))));
    els.dialThreshold.value = String(Math.max(0, Math.min(100, valueToLinSlider(th, KTHR_LO, KTHR_HI))));
    updateDialReadouts();
}

function setKalmanDialsFromPreviewParams(prm) {
    if (!prm || !els.dialQ) return;
    setKalmanDialsFromParams({
        process_noise_scale: prm.kalman_process_noise_scale,
        measurement_noise_scale: prm.kalman_measurement_noise_scale,
        update_gain_scale: prm.kalman_update_gain_scale,
        innovation_threshold: prm.kalman_innovation_threshold,
    });
}

function getKalmanPayloadFromDials() {
    return {
        process_noise_scale: logSliderToValue(Number(els.dialQ.value), KQ_LO, KQ_HI),
        measurement_noise_scale: logSliderToValue(Number(els.dialR.value), KR_LO, KR_HI),
        update_gain_scale: linSliderToValue(Number(els.dialK.value), KK_LO, KK_HI),
        innovation_threshold: linSliderToValue(Number(els.dialThreshold.value), KTHR_LO, KTHR_HI),
    };
}

function seriesStats(arr) {
    const clean = (arr || []).filter((v) => v != null && !Number.isNaN(v)).map(Number).sort((a, b) => a - b);
    if (!clean.length) return null;
    const q = (p) => clean[Math.min(clean.length - 1, Math.floor(p * (clean.length - 1)))];
    return {
        min: clean[0],
        max: clean[clean.length - 1],
        p10: q(0.1),
        p50: q(0.5),
        p90: q(0.9),
    };
}

function percentileRank(arr, value) {
    if (value == null || Number.isNaN(value)) return null;
    const clean = (arr || []).filter((v) => v != null && !Number.isNaN(v)).map(Number);
    if (!clean.length) return null;
    const below = clean.filter((v) => v <= value).length;
    return Math.round((below / clean.length) * 100);
}

function buildValueTooltip(desc, tsKey, value) {
    if (!state.preview?.timeseries || desc == null) return '';
    let t = desc;
    const arr = tsKey ? state.preview.timeseries[tsKey] : null;
    if (arr && value != null && !Number.isNaN(value)) {
        const st = seriesStats(arr);
        if (st) {
            const f = (x) => (x == null || Number.isNaN(x) ? '—' : Number(x).toFixed(4));
            t += ` This run: min ${f(st.min)} · p50 ${f(st.p50)} · p90 ${f(st.p90)} · max ${f(st.max)}.`;
        }
        const pr = percentileRank(arr, value);
        if (pr != null) t += ` Current ≈ p${pr} of windows in this run.`;
    }
    return t;
}

function applyMetricLabelTitles() {
    const idMap = { kalmanAccel: 'metric-label-kalman-accel' };
    Object.entries(METRIC_LABEL_HELP).forEach(([key, text]) => {
        const id = idMap[key] || `metric-label-${key}`;
        const el = document.getElementById(id);
        if (el) el.title = text;
    });
    const cardMap = {
        windows: METRIC_LABEL_HELP.windows,
        boundaries: METRIC_LABEL_HELP.boundaries,
        velocity: METRIC_LABEL_HELP.velocity,
        path: METRIC_LABEL_HELP.path,
        kalman: METRIC_LABEL_HELP.kalman,
        kalmanAccel: METRIC_LABEL_HELP.kalmanAccel,
        ema: METRIC_LABEL_HELP.ema,
        return: METRIC_LABEL_HELP.return,
    };
    document.querySelectorAll('.metric-card').forEach((card) => {
        const h = cardMap[card.dataset.metric];
        if (h) card.setAttribute('aria-label', h);
    });
}

function syncConsoleDockReserve() {
    const dock = document.getElementById('floating-console');
    if (!dock) return;
    const h = dock.offsetHeight;
    document.documentElement.style.setProperty('--console-dock-reserve', `${h}px`);
}

function setMetricQualifier(elId, text) {
    const el = document.getElementById(elId);
    if (el) el.textContent = text || '';
}

function refreshMetricValueTooltip(metricElsKey, descKey, tsKey, rawValue) {
    const strong = els.metrics[metricElsKey];
    if (!strong) return;
    const desc = METRIC_VALUE_DESC[descKey];
    strong.title = buildValueTooltip(desc || '', tsKey, rawValue);
}

function drawEmbeddingStrip(chip, boundaryFlag) {
    state.lastEmbeddingChip = chip && chip.length ? chip : null;
    const canvas = document.getElementById('embedding-strip');
    if (!canvas) return;
    const wrap = document.querySelector('.embedding-strip-wrap');
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#161b22';
    ctx.fillRect(0, 0, w, h);
    if (!chip || !chip.length) {
        ctx.fillStyle = '#8b949e';
        ctx.font = '12px system-ui, sans-serif';
        ctx.fillText('No embedding signature for this step', 10, h / 2 + 4);
        if (wrap) wrap.classList.remove('boundary-flash');
        return;
    }
    const lo = Math.min(...chip);
    const hi = Math.max(...chip);
    const range = hi - lo || 1;
    const n = chip.length;
    const padBottom = 10;
    const drawH = h - padBottom;
    const gap = 0.5;
    const cellW = (w - (n - 1) * gap) / n;
    for (let i = 0; i < n; i += 1) {
        const t = (chip[i] - lo) / range;
        const hue = 220 * (1 - t);
        const light = 42 + t * 28;
        ctx.fillStyle = `hsl(${hue}, 72%, ${light}%)`;
        ctx.fillRect(i * (cellW + gap), 0, Math.max(0.5, cellW), drawH);
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.07)';
    ctx.lineWidth = 1;
    for (let g = 8; g < n; g += 8) {
        const x = g * (cellW + gap);
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, drawH);
        ctx.stroke();
    }
    if (boundaryFlag && wrap) {
        wrap.classList.remove('boundary-flash');
        void wrap.offsetWidth;
        wrap.classList.add('boundary-flash');
    }
}

function initEmbeddingStripInteractions() {
    if (state.embeddingStripListeners) return;
    const canvas = document.getElementById('embedding-strip');
    const tip = document.getElementById('embedding-strip-tooltip');
    if (!canvas || !tip) return;
    state.embeddingStripListeners = true;
    canvas.addEventListener('mousemove', (e) => {
        const chip = state.lastEmbeddingChip;
        if (!chip || !chip.length) {
            tip.hidden = true;
            return;
        }
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const n = chip.length;
        const i = Math.min(n - 1, Math.max(0, Math.floor((x / rect.width) * n)));
        const srcDim = state.preview?.embedding_source_dim;
        const chipLen = state.preview?.embedding_chip_len || n;
        const approxIdx = srcDim > chipLen && chipLen > 1
            ? Math.round((i / (chipLen - 1)) * (srcDim - 1))
            : i;
        tip.textContent = `dim≈${approxIdx} · sample ${i + 1}/${n} · value ${chip[i].toFixed(4)}`;
        tip.hidden = false;
        tip.style.left = `${Math.min(rect.width - 120, Math.max(0, x - 50))}px`;
    });
    canvas.addEventListener('mouseleave', () => {
        tip.hidden = true;
    });
}

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
    setKalmanDialsFromParams(run.params.kalman);
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
                ...getKalmanPayloadFromDials(),
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
        if (state.preview.params) setKalmanDialsFromPreviewParams(state.preview.params);
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
    const embNote = document.getElementById('embedding-strip-note');
    if (embNote) embNote.textContent = state.preview.embedding_note || '';
    initEmbeddingStripInteractions();
    applyMetricLabelTitles();
    updateReplayEstimate();
    resetStreamArea();
    if (state.preview?.frames?.length) {
        const f0 = state.preview.frames[0];
        state.currentFrameIndex = 0;
        els.stepReadout.textContent = `window 1 / ${state.preview.frames.length} · press Play to replay`;
        els.attentionWindow.textContent = f0.window_text;
        updateCurrentMarker(0);
        drawAllSparklines(state.preview.timeseries, 0);
        updateLiveMetrics(f0);
        drawEmbeddingStrip(f0.embedding_chip, f0.boundary_flag);
    } else {
        drawAllSparklines(state.preview.timeseries, -1);
        drawEmbeddingStrip(null, false);
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
    setMetricQualifier('metric-qual-windows', '');
    setMetricQualifier('metric-qual-boundaries', '');
    setMetricQualifier('metric-qual-return', '');
    const ts = state.preview?.timeseries;
    if (ts) {
        const mv = summary.mean_velocity;
        setMetricQualifier('metric-qual-velocity', percentileRank(ts.velocity, mv) != null
            ? `mean p${percentileRank(ts.velocity, mv)}`
            : '');
        refreshMetricValueTooltip('velocity', 'velocity', 'velocity', mv);
        const pl = summary.path_length;
        setMetricQualifier('metric-qual-path', percentileRank(ts.cumulative_path, pl) != null
            ? `end p${percentileRank(ts.cumulative_path, pl)}`
            : '');
        refreshMetricValueTooltip('path', 'path', 'cumulative_path', pl);
    }
    setMetricQualifier('metric-qual-kalman', '');
    setMetricQualifier('metric-qual-kalman-accel', '');
    setMetricQualifier('metric-qual-ema', '');
    els.metrics.kalman.title = 'Count of Kalman innovation violations in this run (Mahalanobis > threshold).';
    els.metrics.kalmanAccel.title = 'Count of acceleration-track Kalman violations.';
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
    if (!state.preview?.preview_id) {
        els.serverStatus.textContent = 'Recompute first';
        els.serverStatus.className = 'pill alert';
        els.playBtn.classList.add('play-btn-nudge');
        window.setTimeout(() => els.playBtn.classList.remove('play-btn-nudge'), 600);
        return;
    }
    if (state.playing) return;
    stopPlayback();
    state.playing = true;
    els.serverStatus.textContent = 'replaying';
    els.serverStatus.className = 'pill warn';
    els.playBtn.classList.add('play-btn-nudge');
    window.setTimeout(() => els.playBtn.classList.remove('play-btn-nudge'), 400);
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
            els.serverStatus.textContent = 'ready';
            els.serverStatus.className = 'pill ok';
        }
    };
    ws.onerror = () => {
        state.playing = false;
        els.serverStatus.textContent = 'replay error';
        els.serverStatus.className = 'pill alert';
    };
    ws.onclose = () => {
        state.playing = false;
        if (state.socket === ws) state.socket = null;
        if (els.serverStatus.textContent === 'replaying') {
            els.serverStatus.textContent = 'ready';
            els.serverStatus.className = 'pill ok';
        }
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
    drawEmbeddingStrip(frame.embedding_chip, frame.boundary_flag);
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
    const ts = state.preview.timeseries;
    els.metrics.windows.textContent = `${frame.index + 1}/${state.preview.summary.windows}`;
    els.metrics.boundaries.textContent = ts.boundary_flags.slice(0, frame.index + 1).filter(Boolean).length;
    els.metrics.velocity.textContent = fmt(frame.velocity);
    els.metrics.path.textContent = fmt(frame.cumulative_path);
    els.metrics.kalman.textContent = fmt(frame.kalman_mahalanobis);
    els.metrics.kalmanAccel.textContent = fmt(frame.kalman_accel_mahalanobis);
    els.metrics.ema.textContent = fmt(frame.ema_drift);
    els.metrics.returnCluster.textContent = frame.return_flag ? String(frame.return_cluster_id) : '—';
    setMetricQualifier('metric-qual-windows', '');
    setMetricQualifier('metric-qual-boundaries', '');
    setMetricQualifier('metric-qual-return', '');
    const pv = percentileRank(ts.velocity, frame.velocity);
    setMetricQualifier('metric-qual-velocity', pv != null ? `p${pv}` : '');
    refreshMetricValueTooltip('velocity', 'velocity', 'velocity', frame.velocity);
    const pp = percentileRank(ts.cumulative_path, frame.cumulative_path);
    setMetricQualifier('metric-qual-path', pp != null ? `p${pp}` : '');
    refreshMetricValueTooltip('path', 'path', 'cumulative_path', frame.cumulative_path);
    const pk = percentileRank(ts.kalman_mahalanobis, frame.kalman_mahalanobis);
    setMetricQualifier('metric-qual-kalman', pk != null ? `p${pk}` : '');
    refreshMetricValueTooltip('kalman', 'kalman', 'kalman_mahalanobis', frame.kalman_mahalanobis);
    const pa = percentileRank(ts.kalman_accel_mahalanobis, frame.kalman_accel_mahalanobis);
    setMetricQualifier('metric-qual-kalman-accel', pa != null ? `p${pa}` : '');
    refreshMetricValueTooltip('kalmanAccel', 'kalmanAccel', 'kalman_accel_mahalanobis', frame.kalman_accel_mahalanobis);
    const pe = percentileRank(ts.ema_drift, frame.ema_drift);
    setMetricQualifier('metric-qual-ema', pe != null ? `p${pe}` : '');
    refreshMetricValueTooltip('ema', 'ema', 'ema_drift', frame.ema_drift);
}

function updateCurrentMarker(index) {
    const coords = state.preview.trajectory_3d[index];
    if (!coords) return;
    const marker = state.trajectory.currentMarker;
    marker.visible = true;
    marker.position.set(coords[0], coords[1], coords[2]);
}

function drawAllSparklines(timeseries, cursorIndex) {
    const thr = state.preview?.params?.kalman_innovation_threshold;
    drawSparkline('spark-velocity', timeseries.velocity, '#58a6ff', cursorIndex);
    drawSparkline('spark-acceleration', timeseries.acceleration, '#bc8cff', cursorIndex);
    drawSparkline('spark-kalman', timeseries.kalman_mahalanobis, '#d29922', cursorIndex, {
        threshold: thr,
        thresholdLabel: 'innovation threshold',
    });
    drawSparkline('spark-kalman-accel', timeseries.kalman_accel_mahalanobis, '#f85149', cursorIndex);
    drawSparkline('spark-ema', timeseries.ema_drift, '#3fb950', cursorIndex);
}

function drawSparkline(canvasId, values, color, cursorIndex, opts = {}) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const clean = values.filter((value) => value !== null && !Number.isNaN(value));
    if (clean.length < 2) return;
    let min = Math.min(...clean);
    let max = Math.max(...clean);
    const { threshold } = opts;
    if (threshold != null && !Number.isNaN(threshold)) {
        min = Math.min(min, threshold);
        max = Math.max(max, threshold);
    }
    const range = max - min || 1;

    if (threshold != null && !Number.isNaN(threshold)) {
        const y = h - ((threshold - min) / range) * (h - 8) - 4;
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(255,255,255,0.22)';
        ctx.setLineDash([3, 4]);
        ctx.lineWidth = 1;
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
        ctx.setLineDash([]);
    }

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
    applyMetricLabelTitles();
    initEmbeddingStripInteractions();
    [els.dialQ, els.dialR, els.dialK, els.dialThreshold].forEach((el) => {
        el?.addEventListener('input', updateDialReadouts);
    });
    updateDialReadouts();
    const cbtn = document.getElementById('console-toggle');
    const cbody = document.getElementById('console-body');
    cbtn?.addEventListener('click', () => {
        const collapsed = cbody.classList.toggle('collapsed');
        cbtn.setAttribute('aria-expanded', String(!collapsed));
        cbtn.textContent = collapsed ? 'Show' : 'Hide';
        requestAnimationFrame(() => {
            requestAnimationFrame(syncConsoleDockReserve);
        });
    });
    const dockEl = document.getElementById('floating-console');
    if (dockEl && typeof ResizeObserver !== 'undefined') {
        new ResizeObserver(() => syncConsoleDockReserve()).observe(dockEl);
    }
    window.addEventListener('resize', syncConsoleDockReserve);
    syncConsoleDockReserve();
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
