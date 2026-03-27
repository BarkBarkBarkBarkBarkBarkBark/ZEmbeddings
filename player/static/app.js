/**
 * ZEmbeddings Live Player — Client-side app.
 *
 * Three.js 3D trajectory, WebSocket connection,
 * audio capture, sparkline rendering.
 */

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Three.js setup
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

const canvas = document.getElementById('trajectory-canvas');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0d1117);

const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
camera.position.set(2, 2, 3);

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
const controls = new THREE.OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.05;

// Grid
const grid = new THREE.GridHelper(4, 20, 0x30363d, 0x1c2128);
scene.add(grid);

// Axes
const axes = new THREE.AxesHelper(1.5);
scene.add(axes);

// Trajectory line + points
const trajectoryPoints = [];
const lineMaterial = new THREE.LineBasicMaterial({ color: 0x58a6ff, linewidth: 2 });
let trajectoryLine = null;
const pointGroup = new THREE.Group();
scene.add(pointGroup);

// Boundary markers
const boundaryGroup = new THREE.Group();
scene.add(boundaryGroup);

function resizeRenderer() {
    const panel = document.getElementById('panel-3d');
    const w = panel.clientWidth - 32;
    const h = panel.clientHeight - 60;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
}
window.addEventListener('resize', resizeRenderer);
resizeRenderer();

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}
animate();


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  WebSocket
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

let ws = null;

function connect() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}/ws`);

    ws.onopen = () => {
        setStatus('Connected', true);
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleMessage(data);
    };

    ws.onclose = () => {
        setStatus('Disconnected', false);
        setTimeout(connect, 2000);
    };

    ws.onerror = (err) => {
        console.error('WebSocket error', err);
        setStatus('Error', false);
    };
}

function setStatus(text, connected) {
    const el = document.getElementById('status');
    el.textContent = text;
    el.className = 'status' + (connected ? ' connected' : '');
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Message handler
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function handleMessage(data) {
    if (data.type === 'waiting') {
        document.getElementById('transcript').textContent = data.transcript;
        document.getElementById('m-windows').textContent = data.n_windows;
        return;
    }

    if (data.type === 'reset') {
        resetVisualization();
        return;
    }

    if (data.type === 'update') {
        // Update transcript
        document.getElementById('transcript').textContent = data.transcript;

        // Update metrics
        document.getElementById('m-windows').textContent = data.n_windows;
        document.getElementById('m-path').textContent = data.total_path_length.toFixed(4);
        document.getElementById('m-velocity').textContent = data.mean_velocity.toFixed(4);
        document.getElementById('m-boundaries').textContent = data.n_boundaries;
        document.getElementById('m-kalman').textContent = data.n_kalman_violations;
        document.getElementById('m-accel').textContent = data.n_accel_violations;
        document.getElementById('m-cloud-mean').textContent = data.cloud_mean_sim.toFixed(3);
        document.getElementById('m-cloud-std').textContent = data.cloud_std_sim.toFixed(3);

        // Update 3D trajectory
        updateTrajectory(data.trajectory_3d, data.velocity, data.boundary_flags,
                         data.kalman_accel_violations);

        // Update sparklines
        drawSparkline('spark-velocity', data.velocity, '#58a6ff');
        drawSparkline('spark-acceleration', data.acceleration, '#bc8cff');
        drawSparkline('spark-kalman', data.kalman_mahalanobis, '#d29922');
        drawSparkline('spark-kalman-accel', data.kalman_accel_mahalanobis, '#f85149');
        drawSparkline('spark-ema', data.ema_drift, '#3fb950');

        document.getElementById('point-count').textContent = `${data.n_windows} points`;
    }
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  3D trajectory
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function updateTrajectory(coords3d, velocities, boundaries, accelViolations) {
    // Clear existing
    if (trajectoryLine) scene.remove(trajectoryLine);
    while (pointGroup.children.length) pointGroup.remove(pointGroup.children[0]);
    while (boundaryGroup.children.length) boundaryGroup.remove(boundaryGroup.children[0]);

    if (!coords3d || coords3d.length < 2) return;

    // Build coloured line segments
    const positions = [];
    const colors = [];

    for (let i = 0; i < coords3d.length; i++) {
        const [x, y, z] = coords3d[i];
        positions.push(x, y, z);

        // Colour by velocity: blue (low) → red (high)
        const v = velocities[i] || 0;
        const t = Math.min(v / 0.3, 1.0); // normalise
        colors.push(
            0.34 + t * 0.63,   // R: blue→red
            0.65 - t * 0.45,   // G: bright→dim
            1.0 - t * 0.7      // B: blue→dark
        );
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    const material = new THREE.LineBasicMaterial({ vertexColors: true, linewidth: 2 });
    trajectoryLine = new THREE.Line(geometry, material);
    scene.add(trajectoryLine);

    // Point spheres (small)
    const sphereGeo = new THREE.SphereGeometry(0.02, 8, 8);
    for (let i = 0; i < coords3d.length; i++) {
        const [x, y, z] = coords3d[i];
        const v = velocities[i] || 0;
        const t = Math.min(v / 0.3, 1.0);
        const color = new THREE.Color(0.34 + t * 0.63, 0.65 - t * 0.45, 1.0 - t * 0.7);
        const mat = new THREE.MeshBasicMaterial({ color });
        const sphere = new THREE.Mesh(sphereGeo, mat);
        sphere.position.set(x, y, z);
        pointGroup.add(sphere);
    }

    // Boundary markers (larger red spheres)
    const boundaryGeo = new THREE.SphereGeometry(0.05, 12, 12);
    const boundaryMat = new THREE.MeshBasicMaterial({ color: 0xf85149, transparent: true, opacity: 0.7 });
    for (let i = 0; i < coords3d.length; i++) {
        if (boundaries[i] || accelViolations[i]) {
            const [x, y, z] = coords3d[i];
            const marker = new THREE.Mesh(boundaryGeo, boundaryMat);
            marker.position.set(x, y, z);
            boundaryGroup.add(marker);
        }
    }
}

function resetVisualization() {
    if (trajectoryLine) { scene.remove(trajectoryLine); trajectoryLine = null; }
    while (pointGroup.children.length) pointGroup.remove(pointGroup.children[0]);
    while (boundaryGroup.children.length) boundaryGroup.remove(boundaryGroup.children[0]);
    document.getElementById('transcript').innerHTML = '<span class="dim">Reset. Press "Start" to begin...</span>';
    document.getElementById('point-count').textContent = '0 points';
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Sparklines (canvas-based)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function drawSparkline(canvasId, values, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    // Filter nulls
    const clean = values.filter(v => v !== null && !isNaN(v));
    if (clean.length < 2) return;

    const min = Math.min(...clean);
    const max = Math.max(...clean);
    const range = max - min || 1;

    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;

    let first = true;
    for (let i = 0; i < values.length; i++) {
        if (values[i] === null) continue;
        const x = (i / (values.length - 1)) * w;
        const y = h - ((values[i] - min) / range) * (h - 4) - 2;
        if (first) { ctx.moveTo(x, y); first = false; }
        else ctx.lineTo(x, y);
    }
    ctx.stroke();
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Audio capture
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
const CHUNK_DURATION_MS = 5000; // 5 second chunks

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: { sampleRate: 16000, channelCount: 1 }
        });

        mediaRecorder = new MediaRecorder(stream, {
            mimeType: MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/ogg'
        });

        audioChunks = [];
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0 && ws && ws.readyState === WebSocket.OPEN) {
                // Send audio chunk as binary
                event.data.arrayBuffer().then(buffer => {
                    ws.send(buffer);
                });
            }
        };

        // Record in chunks
        mediaRecorder.start(CHUNK_DURATION_MS);
        isRecording = true;
        document.getElementById('btn-start').disabled = true;
        document.getElementById('btn-stop').disabled = false;
        document.getElementById('audio-status').textContent = '🔴 Recording...';

    } catch (err) {
        console.error('Mic access denied:', err);
        document.getElementById('audio-status').textContent = '⚠ Mic access denied';
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(t => t.stop());
        isRecording = false;
        document.getElementById('btn-start').disabled = false;
        document.getElementById('btn-stop').disabled = true;
        document.getElementById('audio-status').textContent = '';
    }
}


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Event listeners
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

document.getElementById('btn-start').addEventListener('click', startRecording);
document.getElementById('btn-stop').addEventListener('click', stopRecording);
document.getElementById('btn-reset').addEventListener('click', () => {
    stopRecording();
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'reset' }));
    }
    resetVisualization();
});

document.getElementById('btn-reset-view').addEventListener('click', () => {
    camera.position.set(2, 2, 3);
    controls.target.set(0, 0, 0);
    controls.update();
});

// Text input (for testing without microphone)
document.getElementById('btn-send').addEventListener('click', () => {
    const input = document.getElementById('text-input');
    const text = input.value.trim();
    if (text && ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'text', text }));
        input.value = '';
    }
});

document.getElementById('text-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        document.getElementById('btn-send').click();
    }
});

// Connect on load
connect();
