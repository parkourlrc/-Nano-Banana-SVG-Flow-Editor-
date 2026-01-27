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
