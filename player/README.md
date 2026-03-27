# 🎧 ZEmbeddings Live Player

Local-only real-time semantic trajectory visualisation.

Speak → Whisper transcribes → embeddings computed → 3D trajectory rendered live.

## Architecture

```
┌──────────────┐    WebSocket     ┌──────────────────┐
│  Browser UI  │ ◄──────────────► │  FastAPI server   │
│  Three.js 3D │                  │                   │
│  + Metrics   │                  │  Whisper (local)  │
│              │                  │  Embeddings (local)│
│  Mic capture │ ─── audio ───►  │  Kalman filters   │
└──────────────┘                  └──────────────────┘
```

## Quick Start

```bash
# Install player dependencies
pip install -e '.[player]'

# Start the server
cd player
python server.py

# Open http://localhost:8765 in your browser
```

## Components

| File | Purpose |
|------|---------|
| `server.py` | FastAPI + WebSocket server, Whisper + embedding pipeline |
| `static/index.html` | Single-page UI with Three.js 3D visualisation |
| `static/app.js` | Client-side audio capture, WebSocket, Three.js renderer |
| `static/style.css` | Dark-mode UI styling |

## Requirements

- Python 3.10+
- `openai-whisper` (local model, no API key needed)
- `sentence-transformers` (local embeddings)
- `fastapi` + `uvicorn` + `websockets`
- A modern browser with `getUserMedia` support (Chrome/Firefox/Safari)
- Microphone access

## How It Works

1. **Browser** captures microphone audio in 5-second chunks
2. **WebSocket** sends raw PCM audio to the server
3. **Whisper** (tiny/base model, local) transcribes the chunk
4. **Sliding window** tokeniser builds context windows
5. **Sentence-transformers** embeds each window locally
6. **Metrics** computed: velocity, acceleration, jerk, EMA drift
7. **Kalman filters** (scalar + acceleration) detect boundaries
8. **WebSocket** sends back: transcript, embeddings (PCA→3D), metrics, boundaries
9. **Three.js** renders the trajectory as a coloured 3D tube/ribbon
   - Colour = velocity (blue → calm, red → shifting)
   - Spheres = boundary events (Kalman violations)
   - Live sparklines for all metric channels
