# Research Diagram Studio

English | [简体中文](README.md)

A diagrams.net (draw.io) embedded web app that turns prompts / paper-figure screenshots into **editable, exportable** research diagrams you can continue editing in the browser.

## Key Features

- **Real-time diagrams.net canvas**: edit in the browser and export `mxGraph XML` (native diagrams.net format) or `JSON`
- **AI diagram generation (XML by default)**: generate and load directly into the canvas
- **Paper figure screenshot → editable diagram**: upload a reference image and go through a “Precision Calibrate” workflow before applying to canvas
- **Overlay extraction**: keep non-vector content (icons/3D/snapshots/real charts, etc.) as anchored image overlays (inside a node or globally)
- **Optional Image Model (text-to-image)**: generate or rewrite a reference image first, then extract structure and render into draw.io
- **Privacy-first**: provider API keys are not stored in the repo (see below)

## How It Works (Architecture & Data Flow)

- **Architecture**: `server/` (Node/Express) serves both the static frontend (`web/`) and the API. The diagrams.net canvas runs in the browser, and the app communicates with it via iframe messaging.
- **Provider requests**: all LLM / image-generation calls are proxied by the backend (`/api/flow/:format`, `/api/image/generate`) so API keys don’t need to be exposed to the frontend. `GET /api/providers` returns a sanitized provider list (no `apiKey`).
- **Three generation paths**:
  - **No reference image (direct)**: Prompt → `/api/flow/:format` → model returns XML/JSON → frontend loads it into the canvas.
  - **With reference image (Precision Calibrate)**: image → `/api/vision/structure` → structured JSON (nodes/text/edges/overlays) → user calibration → **deterministic** JSON→mxGraph XML → `Apply to Canvas`.
  - **Image model → then extract (Image Model assisted)**: when `Enable Image Model` is ON and your current input is judged to “need image generation / rewrite”:
    - **No uploaded image**: Prompt → `/api/image/generate` (Image Model, e.g. *nano banana pro*) → reference image (preview/download in UI) → `/api/vision/structure` → calibrate → apply to canvas.
    - **Uploaded image but you want redraw/edit/style-variant**: image + Prompt → `/api/image/generate` to create a new reference → `/api/vision/structure` → calibrate → apply to canvas.
- **Local vision service (Route-1)**: when `vision_service` is available, the pipeline prefers local CV+OCR+SAM2 for better bbox/text/overlay quality. If it’s not available, the API returns a clear error message instead of silently degrading output quality.
- **Task persistence**: calibration tasks are keyed by `imageHash + prompt` and stored in browser IndexedDB so you can reopen and continue editing. Segmented overlay PNGs are cached to avoid recomputation.
- **Leak prevention**: provider config is stored outside the repo by default (`~/.research-diagram-studio/providers.json`) + `.gitignore` + `npm run check:secrets` + GitHub Actions scanning.

## Requirements

- Node.js: **18+** (needs built-in `fetch`)
- Browser: Chrome / Edge
- (Optional) Local vision service: `vision_service/` (CV + OCR + SAM2, see `vision_service/README.md`)

## Quick Start

```bash
npm install
npm run dev
```

Open:

`http://localhost:3000`

Change port if needed:

```powershell
$env:PORT=3001; npm run dev
```

## Recommended Workflow

### 1) Configure a Provider (required)

Open `Settings` (top-right):

1. Choose language (中文 / English)
2. Choose a provider (default: `OpenAI-compatible`)
3. Fill required fields:
   - `API Key`
   - `Base URL` (OpenAI-compatible default: `https://0-0.pro/v1`)
   - `LLM Model` (structure/text extraction, planning, critic, etc.)
   - `Image Model` (optional, for image generation / rewrite)
4. Click `Save Provider`
5. (Optional) click `Get API` to open: https://0-0.pro/

### 2) No reference image: generate directly

1. Enter a `Prompt`
2. Choose output format (default: `XML`)
3. Click `Generate Diagram`

### 3) With a reference image: Precision Calibrate (recommended)

1. Upload a paper-figure screenshot (`Images`)
2. Click `Generate Diagram`
3. In `Precision Calibrate`:
   - Draw boxes to create overlays, pick FG/BG points for segmentation
   - Edit node bbox / shape / text assignment, etc.
4. Click `Apply to Canvas`

### 4) Enable Image Model (optional)

In `Settings`, enable `Enable Image Model` and fill `Image Model`:

- **Switch OFF**: the app will never call the image generation API
- **Switch ON**: the app decides whether to call the image model based on your intent (e.g., no uploaded image but you want to “generate a reference image”; or you uploaded an image but asked for “rewrite/redraw/style variant”)

After generation:

- You’ll see thumbnails in the UI; click to preview & download
- Click `×` (top-right) to remove a thumbnail

## Config & Data Storage (IMPORTANT: avoid leaks)

- Provider config file (contains `apiKey`) is stored **outside the repo**, by default at:
  - `~/.research-diagram-studio/providers.json`
  - You can override via:
    - `RDS_CONFIG_DIR`
    - `RDS_PROVIDERS_PATH`
- Template: `server/config/providers.sample.json`
- Precision calibration tasks are stored in browser IndexedDB (not pushed to GitHub)

Before pushing to GitHub, run:

```bash
npm run check:secrets
```

## Troubleshooting

- `fetch failed / network_error`: check `Base URL`, proxy, DNS, network connectivity
- `HTTP 401/403`: invalid API key / missing permissions (check Settings)
- `504 / timeout`: upstream gateway/model timeout; try a faster model, reduce complexity, or enable local `vision_service`

## API (for development)

- `GET /api/providers`: list providers (no `apiKey`)
- `POST /api/providers`: save provider (writes to user-home `providers.json`)
- `POST /api/flow/:format`: generate (`json` / `xml`)
- `POST /api/image/generate`: image generation (requires Image Model enabled)
- `POST /api/vision/structure`: structure extraction from reference image (can use local `vision_service`)

## Project Structure

- `web/`: frontend (`index.html` / `app.js` / `styles.css`)
- `server/`: backend (`server/index.js`)
- `vision_service/`: optional local vision service (FastAPI + CV + SAM2)

---

Contact (user group):
Telegram:@ryonliu
Affordable API gateway: https://0-0.pro/
