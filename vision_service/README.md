# Vision Service (Route-1)

This project can use a local CV/GPU vision service (Python) to extract:
- nodes (diagram modules)
- overlays (non-vector visuals to keep)
- edges (connectors; TODO in placeholder)

## Install (Windows)

```powershell
cd vision_service
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For Route-1 quality (OCR-enabled), install:

```powershell
pip install -r requirements.full.txt
```

## SAM2 (transparent overlay extraction)

To extract icon/illustration overlays with transparent background (`granularity="alphaMask"`), install SAM2 + Torch CUDA.

```powershell
cd vision_service
.\.venv\Scripts\Activate.ps1

# Torch CUDA (pick the CUDA index that matches your setup; cu121 is common on Windows)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# SAM2 + deps
pip install -r requirements.sam2.txt
```

On first use, the service auto-downloads the selected SAM2 checkpoint to `vision_service/.weights/`.
You can control it via env vars:
- `SAM2_MODEL` (default: `sam2_hiera_base_plus`)
- `SAM2_FORCE_CPU=1` (force CPU; slower)

## Run

```powershell
cd vision_service
.\.venv\Scripts\Activate.ps1
python app.py
```

Default URL: `http://127.0.0.1:7777`

## Enable in the Node server

If the service is running on the default URL (`http://127.0.0.1:7777`), `npm run dev` will auto-detect it.

To force a specific URL, set `VISION_SERVICE_URL` before running `npm run dev`:

```powershell
$env:VISION_SERVICE_URL="http://127.0.0.1:7777"
npm run dev
```

## Notes

- The pipeline uses OCR (if `requirements.full.txt` is installed) to keep labels editable, and extracts complex visuals as node-anchored overlay images.
- For best results on paper figure screenshots, install `requirements.full.txt` and use the Vision Debug modal in the UI to inspect detection output.
