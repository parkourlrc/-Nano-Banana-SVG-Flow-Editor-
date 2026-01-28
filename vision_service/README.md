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

## SAM3 (text-prompt grounding for cleaner proposals)

This repo vendors the `sam3/` Python package (borrowed from Edit-Banana) to produce cleaner node/overlay proposal boxes on paper-figure screenshots.
It is optional and only used when installed and enabled.

Install deps:

```powershell
cd vision_service
.\.venv\Scripts\Activate.ps1

# Torch CUDA (pick the CUDA index that matches your setup; cu121 is common on Windows)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.sam3.txt
```

### Download checkpoint

The official Hugging Face repo (`facebook/sam3`) may be gated (401/403) depending on your account.
Recommended: download from ModelScope (public) via Git LFS:

```powershell
cd vision_service
.\.venv\Scripts\Activate.ps1
python download_sam3.py
```

This downloads and caches the checkpoint into `vision_service/.weights/` (default: `sam3.pt`).
You can also use `model.safetensors`:

```powershell
python download_sam3.py model.safetensors
```

NOTE: If you prefer Hugging Face, request access and provide a token via `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`
(or login locally with `hf auth login`), or download the checkpoint manually and set `SAM3_CHECKPOINT_PATH`.

You can control it via env vars:
- `SAM3_ENABLED=0` (disable SAM3)
- `SAM3_FORCE_CPU=1` (force CPU)
- `SAM3_MODEL_ID` (default: `facebook/sam3`)
- `SAM3_CHECKPOINT_NAME` (default: `sam3.pt`)
- `SAM3_CHECKPOINT_PATH` (use a local checkpoint instead of auto-download)
- `SAM3_BPE_PATH` (override tokenizer vocab path; default points to `sam3/assets/bpe_simple_vocab_16e6.txt.gz`)

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
