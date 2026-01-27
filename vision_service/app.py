import base64
import hashlib
import importlib.util
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(title="Research Diagram Vision Service", version="1.0.0")

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), ".weights")
SAM2_MODEL_NAME = os.environ.get("SAM2_MODEL", "sam2_hiera_base_plus")
SAM2_FORCE_CPU = os.environ.get("SAM2_FORCE_CPU", "0") == "1"

SAM2_CONFIG_NAME_BY_MODEL = {
    "sam2_hiera_tiny": "sam2_hiera_t.yaml",
    "sam2_hiera_small": "sam2_hiera_s.yaml",
    "sam2_hiera_base_plus": "sam2_hiera_b+.yaml",
    "sam2_hiera_large": "sam2_hiera_l.yaml",
}

SAM2_CHECKPOINT_URL_BY_MODEL = {
    "sam2_hiera_tiny": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
    "sam2_hiera_small": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
    "sam2_hiera_base_plus": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
    "sam2_hiera_large": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
}

_sam2_state = {"model": None, "predictor": None, "amg": None, "amg_key": None, "device": None, "image_key": None}
_warn_counts: Dict[str, int] = {}


class ImageIn(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    dataUrl: str
    width: Optional[int] = None
    height: Optional[int] = None


class AnalyzeRequest(BaseModel):
    prompt: str = ""
    image: ImageIn
    imageWidth: int
    imageHeight: int
    textItems: Optional[List[Dict[str, Any]]] = None
    options: Optional[Dict[str, Any]] = None


class NodeAugmentIn(BaseModel):
    id: str
    bbox: Dict[str, Any]
    textBbox: Optional[Dict[str, Any]] = None


class AugmentRequest(BaseModel):
    image: ImageIn
    imageWidth: int
    imageHeight: int
    nodes: List[NodeAugmentIn]
    textItems: Optional[List[Dict[str, Any]]] = None
    options: Optional[Dict[str, Any]] = None


def _normalize_quality_mode(options: Optional[Dict[str, Any]]) -> str:
    try:
        mode = str((options or {}).get("qualityMode", "") or "").strip().lower()
    except Exception:
        mode = ""
    return "balanced" if mode == "balanced" else "max"


def _export_text_items_for_node_server(text_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for it in text_items or []:
        if not isinstance(it, dict):
            continue
        text = str(it.get("text", "") or "").strip()
        bb = it.get("bbox")
        if not text or not bb or not isinstance(bb, (tuple, list)) or len(bb) < 4:
            continue
        x, y, w, h = bb[0], bb[1], bb[2], bb[3]
        out.append({"text": text, "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}})
        if len(out) >= 420:
            break
    return out


def _rescale_text_items_inplace(raw_items: List[Dict[str, Any]], sx: float, sy: float) -> List[Dict[str, Any]]:
    if not raw_items:
        return raw_items
    if abs(float(sx) - 1.0) < 1e-6 and abs(float(sy) - 1.0) < 1e-6:
        return raw_items
    out: List[Dict[str, Any]] = []
    for it in raw_items:
        if not isinstance(it, dict):
            continue
        item = dict(it)
        bb = item.get("bbox")
        if isinstance(bb, dict):
            try:
                x = float(bb.get("x", 0.0)) * sx
                y = float(bb.get("y", 0.0)) * sy
                w = float(bb.get("w", bb.get("width", 0.0))) * sx
                h = float(bb.get("h", bb.get("height", 0.0))) * sy
                item["bbox"] = {"x": x, "y": y, "w": w, "h": h}
            except Exception:
                pass
        poly = item.get("poly")
        if isinstance(poly, list) and poly:
            try:
                poly2 = []
                for p in poly:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        poly2.append([float(p[0]) * sx, float(p[1]) * sy])
                if poly2:
                    item["poly"] = poly2
            except Exception:
                pass
        out.append(item)
    return out


class AnnotateBoxIn(BaseModel):
    id: str
    bbox: Dict[str, Any]
    label: Optional[str] = None
    color: Optional[str] = None  # hex string like "#ff0000"


class AnnotateBoxesRequest(BaseModel):
    image: ImageIn
    imageWidth: int
    imageHeight: int
    nodes: Optional[List[NodeAugmentIn]] = None
    overlays: Optional[List[AnnotateBoxIn]] = None


class ContactSheetItem(BaseModel):
    id: str
    label: Optional[str] = None
    bbox: Optional[Dict[str, Any]] = None
    previewDataUrl: Optional[str] = None
    nodeId: Optional[str] = None
    kind: Optional[str] = None


class ContactSheetOptions(BaseModel):
    tileSize: Optional[int] = 152
    cols: Optional[int] = 6
    maxItems: Optional[int] = 48
    padPx: Optional[int] = 3


class ContactSheetRequest(BaseModel):
    image: ImageIn
    imageWidth: int
    imageHeight: int
    items: List[ContactSheetItem]
    options: Optional[ContactSheetOptions] = None


class PointIn(BaseModel):
    x: float
    y: float


class OverlayResolveIn(BaseModel):
    id: str
    bbox: Dict[str, Any]
    granularity: str  # alphaMask|opaqueRect
    kind: Optional[str] = None
    fgPoints: Optional[List[PointIn]] = None
    bgPoints: Optional[List[PointIn]] = None


class OverlayResolveOptions(BaseModel):
    tightenBbox: Optional[bool] = True
    padPx: Optional[int] = None


class OverlayResolveRequest(BaseModel):
    image: ImageIn
    imageWidth: int
    imageHeight: int
    overlays: List[OverlayResolveIn]
    debug: Optional[bool] = False
    options: Optional[OverlayResolveOptions] = None
    textItems: Optional[List[Dict[str, Any]]] = None


def _download_file(url: str, dst_path: str) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(dst_path), suffix=".tmp") as tmp:
        tmp_path = tmp.name
    try:
        with urllib.request.urlopen(url) as resp, open(tmp_path, "wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(tmp_path, dst_path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _ensure_sam2_checkpoint(model_name: str) -> str:
    name = str(model_name or "").strip()
    if name not in SAM2_CHECKPOINT_URL_BY_MODEL:
        raise RuntimeError(f"Unknown SAM2 model '{name}'.")
    ckpt_name = f"{name}.pt" if not name.endswith(".pt") else name
    ckpt_path = os.path.join(WEIGHTS_DIR, ckpt_name)
    if os.path.exists(ckpt_path) and os.path.getsize(ckpt_path) > 50_000_000:
        return ckpt_path
    url = SAM2_CHECKPOINT_URL_BY_MODEL[name]
    _download_file(url, ckpt_path)
    return ckpt_path


def _resolve_sam2_config_path(model_name: str) -> str:
    name = str(model_name or "").strip()
    if name not in SAM2_CONFIG_NAME_BY_MODEL:
        raise RuntimeError(f"Unknown SAM2 model '{name}'.")

    explicit = str(os.environ.get("SAM2_CONFIG", "")).strip()
    if explicit and os.path.exists(explicit):
        return explicit

    cfg_name = SAM2_CONFIG_NAME_BY_MODEL[name]

    candidates: List[str] = []
    # Common local checkout layout: ./sam2/configs/sam2/<cfg>
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates.append(os.path.join(repo_root, "sam2", "configs", "sam2", cfg_name))
    candidates.append(os.path.join(repo_root, "sam2", "configs", cfg_name))

    # Installed package layout: <site-packages>/sam2/configs/sam2/<cfg>
    try:
        import sam2  # type: ignore

        sam2_dir = os.path.dirname(getattr(sam2, "__file__", "") or "")
        if sam2_dir:
            candidates.append(os.path.join(sam2_dir, "configs", "sam2", cfg_name))
            candidates.append(os.path.join(sam2_dir, "configs", cfg_name))
    except Exception:
        pass

    for c in candidates:
        if c and os.path.exists(c):
            return c

    preview = ", ".join(candidates[:6])
    raise RuntimeError(f"SAM2 config not found for model '{name}'. Tried: {preview}")


def _get_sam2_model() -> Tuple[Any, str]:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch is not installed. Install SAM2 + torch CUDA to enable alpha overlay extraction.") from exc

    try:
        from sam2.build_sam import build_sam2
    except Exception as exc:
        raise RuntimeError("SAM2 package is not installed. Install https://github.com/facebookresearch/sam2.") from exc

    device = "cuda" if (torch.cuda.is_available() and not SAM2_FORCE_CPU) else "cpu"

    model = _sam2_state.get("model")
    if model is None or _sam2_state.get("device") != device:
        cfg = _resolve_sam2_config_path(SAM2_MODEL_NAME)
        ckpt_path = _ensure_sam2_checkpoint(SAM2_MODEL_NAME)
        model = build_sam2(cfg, ckpt_path, device=device, apply_postprocessing=True)
        _sam2_state["model"] = model
        _sam2_state["device"] = device
        _sam2_state["predictor"] = None
        _sam2_state["image_key"] = None
        _sam2_state["amg"] = None
        _sam2_state["amg_key"] = None

    return model, device


def _get_sam2_predictor(img_bgr: Optional[np.ndarray] = None):
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except Exception as exc:
        raise RuntimeError("SAM2 package is not installed. Install https://github.com/facebookresearch/sam2.") from exc

    model, device = _get_sam2_model()

    predictor = _sam2_state.get("predictor")
    if predictor is None:
        predictor = SAM2ImagePredictor(model)
        _sam2_state["predictor"] = predictor
        _sam2_state["image_key"] = None

    if img_bgr is not None:
        # Cache image embeddings across calls for the same image.
        key = hashlib.sha256(img_bgr.tobytes()).hexdigest()
        if _sam2_state.get("image_key") != key:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            predictor.set_image(img_rgb)
            _sam2_state["image_key"] = key

    return predictor, device


def _get_sam2_amg(
    points_per_side: int,
    crop_n_layers: int,
    pred_iou_thresh: float = 0.62,
    stability_score_thresh: float = 0.62,
):
    try:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except Exception as exc:
        raise RuntimeError("SAM2 package is not installed. Install https://github.com/facebookresearch/sam2.") from exc

    model, device = _get_sam2_model()

    pps = int(max(4, min(96, int(points_per_side))))
    cnl = int(max(0, min(4, int(crop_n_layers))))
    piou = float(max(0.0, min(1.0, float(pred_iou_thresh))))
    stab = float(max(0.0, min(1.0, float(stability_score_thresh))))

    key = f"{device}|pps={pps}|cnl={cnl}|piou={piou:.3f}|stab={stab:.3f}"
    amg = _sam2_state.get("amg")
    if amg is None or _sam2_state.get("amg_key") != key:
        amg = SAM2AutomaticMaskGenerator(
            model,
            points_per_side=pps,
            points_per_batch=64,
            pred_iou_thresh=piou,
            stability_score_thresh=stab,
            box_nms_thresh=0.7,
            crop_n_layers=cnl,
            crop_nms_thresh=0.7,
            output_mode="binary_mask",
        )
        _sam2_state["amg"] = amg
        _sam2_state["amg_key"] = key

    return amg, device


def _encode_mask_png(mask01: np.ndarray) -> Optional[str]:
    m = (mask01.astype(np.uint8) * 255).astype(np.uint8)
    ok, png = cv2.imencode(".png", m)
    if not ok:
        return None
    return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode("ascii")


def _auto_points_for_bbox(x: int, y: int, w: int, h: int) -> Tuple[List[PointIn], List[PointIn]]:
    # Generate reasonable default prompt points in FULL-IMAGE coordinates.
    # These are used only when the LLM did not supply enough fg/bg points.
    cx = x + w / 2.0
    cy = y + h / 2.0
    inset = max(2.0, min(w, h) * 0.08)

    fg = [
        PointIn(x=cx, y=cy),
        PointIn(x=x + w * 0.25, y=y + h * 0.25),
        PointIn(x=x + w * 0.75, y=y + h * 0.25),
        PointIn(x=x + w * 0.25, y=y + h * 0.75),
        PointIn(x=x + w * 0.75, y=y + h * 0.75),
    ]

    bg = [
        PointIn(x=x + inset, y=y + inset),
        PointIn(x=x + w - inset, y=y + inset),
        PointIn(x=x + inset, y=y + h - inset),
        PointIn(x=x + w - inset, y=y + h - inset),
        PointIn(x=cx, y=y + inset),
        PointIn(x=cx, y=y + h - inset),
        PointIn(x=x + inset, y=cy),
        PointIn(x=x + w - inset, y=cy),
    ]

    return fg, bg


def _resolve_overlay_one(
    img_bgr: np.ndarray,
    ov: OverlayResolveIn,
    debug: bool = False,
    predictor=None,
    device: Optional[str] = None,
    sam2_available: bool = True,
    tighten_bbox: bool = True,
    pad_px: Optional[int] = None,
    text_ink_full: Optional[np.ndarray] = None,
    blue_full: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    h, w = img_bgr.shape[:2]
    bb = ov.bbox or {}
    x = int(bb.get("x", 0))
    y = int(bb.get("y", 0))
    ww = int(bb.get("w", bb.get("width", 1)))
    hh = int(bb.get("h", bb.get("height", 1)))
    x, y, ww, hh = clip_box(x, y, ww, hh, w, h)

    gran = str(ov.granularity or "").strip()
    if gran not in ("alphaMask", "opaqueRect"):
        return {"id": ov.id, "ok": False, "reason": "invalid_granularity"}

    roi = img_bgr[y : y + hh, x : x + ww]
    if roi.size == 0:
        return {"id": ov.id, "ok": False, "reason": "empty_roi"}

    roi_bbox = roi

    if gran == "opaqueRect":
        out_bbox = {"x": x, "y": y, "w": ww, "h": hh}
        if tighten_bbox:
            try:
                # Trim blank margins while keeping an opaque rectangular background.
                # Estimate background from border pixels and keep regions that differ.
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                hh0, ww0 = roi_rgb.shape[:2]
                if hh0 >= 4 and ww0 >= 4:
                    border = np.concatenate(
                        [
                            roi_rgb[0:1, :, :].reshape(-1, 3),
                            roi_rgb[hh0 - 1 : hh0, :, :].reshape(-1, 3),
                            roi_rgb[:, 0:1, :].reshape(-1, 3),
                            roi_rgb[:, ww0 - 1 : ww0, :].reshape(-1, 3),
                        ],
                        axis=0,
                    )
                    bg = np.median(border, axis=0).astype(np.float32)
                    diff = np.sqrt(np.sum((roi_rgb.astype(np.float32) - bg.reshape((1, 1, 3))) ** 2, axis=2))
                    # Adaptive threshold based on border variance.
                    border_std = float(np.mean(np.std(border.astype(np.float32), axis=0)))
                    thr = float(max(18.0, min(60.0, border_std * 2.6 + 18.0)))
                    mask = (diff > thr).astype(np.uint8) * 255

                    # Also include edges to keep thin strokes.
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    e = cv2.Canny(gray, 60, 160)
                    e = cv2.dilate(e, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
                    mask = cv2.bitwise_or(mask, e)

                    # Remove text/connector ink so external labels don't force the crop bbox to include them.
                    if text_ink_full is not None and text_ink_full.size > 0:
                        txt = text_ink_full[y : y + hh, x : x + ww]
                        if txt.shape[:2] == mask.shape[:2]:
                            mask[txt > 0] = 0
                    if blue_full is not None and blue_full.size > 0:
                        bl = blue_full[y : y + hh, x : x + ww]
                        if bl.shape[:2] == mask.shape[:2]:
                            m = int(max(2, min(20, round(min(ww0, hh0) * 0.06))))
                            bl2 = bl.copy()
                            if hh0 > 2 * m and ww0 > 2 * m:
                                bl2[m : hh0 - m, m : ww0 - m] = 0
                            mask[bl2 > 0] = 0

                    nz = cv2.findNonZero(mask)
                    if nz is not None:
                        rx, ry, rw, rh = cv2.boundingRect(nz)
                        rx, ry, rw, rh = clip_box(rx, ry, rw, rh, ww0, hh0)
                        area0 = float(ww0 * hh0)
                        area1 = float(rw * rh)
                        frac = area1 / max(1.0, area0)
                        # Only crop when it removes a meaningful margin and doesn't collapse to tiny content.
                        if 0.03 <= frac <= 0.97 and rw >= 6 and rh >= 6:
                            if pad_px is not None and isinstance(pad_px, int):
                                pad = int(max(0, min(96, pad_px)))
                            else:
                                pad = int(max(1, min(8, round(min(rw, rh) * 0.06))))
                            rx2 = max(0, rx - pad)
                            ry2 = max(0, ry - pad)
                            rw2 = min(ww0 - rx2, rw + 2 * pad)
                            rh2 = min(hh0 - ry2, rh + 2 * pad)
                            roi = roi[ry2 : ry2 + rh2, rx2 : rx2 + rw2]
                            out_bbox = {"x": x + int(rx2), "y": y + int(ry2), "w": int(rw2), "h": int(rh2)}
            except Exception:
                pass

        out = encode_png_rgba(roi, None)
        return {
            "id": ov.id,
            "ok": True,
            "granularity": gran,
            "bbox": out_bbox,
            "dataUrl": out,
            "stats": {"mode": "opaqueRect"},
        }

    # alphaMask via SAM2
    if not sam2_available:
        resp: Dict[str, Any] = {
            "id": ov.id,
            "ok": False,
            "granularity": gran,
            "bbox": {"x": x, "y": y, "w": ww, "h": hh},
            "dataUrl": None,
            "reason": "sam2_unavailable",
            "stats": {"mode": "alphaMask"},
        }
        if debug:
            resp["debug"] = {"crop": encode_png_rgba(roi, None), "mask": None}
        return resp

    fg = ov.fgPoints or []
    bg = ov.bgPoints or []
    if len(fg) < 2 or len(bg) < 2:
        auto_fg, auto_bg = _auto_points_for_bbox(x, y, ww, hh)
        # Keep any provided points and fill the rest with defaults.
        fg = (fg or []) + auto_fg
        bg = (bg or []) + auto_bg
        fg = fg[:10]
        bg = bg[:12]

    if predictor is None or not device:
        predictor, device = _get_sam2_predictor(img_bgr)

    pts = []
    labels = []
    for p in fg:
        pts.append([float(p.x), float(p.y)])
        labels.append(1)
    for p in bg:
        pts.append([float(p.x), float(p.y)])
        labels.append(0)

    pts_np = np.array(pts, dtype=np.float32)
    labels_np = np.array(labels, dtype=np.int32)
    box_xyxy = np.array([float(x), float(y), float(x + ww), float(y + hh)], dtype=np.float32)

    try:
        masks, ious, _ = predictor.predict(point_coords=pts_np, point_labels=labels_np, box=box_xyxy, multimask_output=True)
    except Exception as exc:
        return {"id": ov.id, "ok": False, "reason": "sam2_predict_failed", "detail": str(exc)[:200]}

    if masks is None or ious is None or len(masks) == 0:
        return {"id": ov.id, "ok": False, "reason": "sam2_no_mask"}

    best = int(np.argmax(ious))
    mask = masks[best]
    if mask.dtype != np.bool_:
        mask01 = mask > 0.0
    else:
        mask01 = mask

    # Crop to bbox and build RGBA.
    m_roi = mask01[y : y + hh, x : x + ww]
    m_roi_bbox = m_roi

    # Post-process: prefer the connected component that matches fgPoints.
    # This is crucial for small icons inside a larger bbox where SAM2 may return multiple components.
    chosen_mask = m_roi.astype(np.uint8)
    try:
        cc_in = (m_roi > 0).astype(np.uint8)
        num_cc, labels_cc, stats_cc, _ = cv2.connectedComponentsWithStats(cc_in, connectivity=8)
        if num_cc > 1:
            # Count which CC receives the most fg prompt points.
            counts: Dict[int, int] = {}
            for p in (ov.fgPoints or []):
                try:
                    px = int(round(float(p.x) - float(x)))
                    py = int(round(float(p.y) - float(y)))
                except Exception:
                    continue
                if px < 0 or py < 0 or px >= ww or py >= hh:
                    continue
                lbl = int(labels_cc[py, px])
                if lbl <= 0:
                    continue
                counts[lbl] = counts.get(lbl, 0) + 1

            chosen_lbl = 0
            if counts:
                chosen_lbl = max(counts.keys(), key=lambda k: (counts.get(k, 0), int(stats_cc[k, cv2.CC_STAT_AREA])))
            else:
                # Fall back to largest CC by area.
                areas = stats_cc[1:, cv2.CC_STAT_AREA]
                if areas.size:
                    chosen_lbl = int(1 + int(np.argmax(areas)))

            if chosen_lbl > 0:
                chosen_mask = (labels_cc == chosen_lbl).astype(np.uint8)
    except Exception:
        chosen_mask = (m_roi > 0).astype(np.uint8)

    # Remove text/connector ink from the mask (overlay content should not include editable labels/arrows).
    try:
        if text_ink_full is not None and text_ink_full.size > 0:
            txt = text_ink_full[y : y + hh, x : x + ww]
            if txt.shape[:2] == chosen_mask.shape[:2]:
                chosen_mask[txt > 0] = 0
        if blue_full is not None and blue_full.size > 0:
            bl = blue_full[y : y + hh, x : x + ww]
            if bl.shape[:2] == chosen_mask.shape[:2]:
                m = int(max(2, min(20, round(min(ww, hh) * 0.06))))
                bl2 = bl.copy()
                if hh > 2 * m and ww > 2 * m:
                    bl2[m : hh - m, m : ww - m] = 0
                chosen_mask[bl2 > 0] = 0
    except Exception:
        pass

    # Light cleanup.
    try:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        chosen_mask = cv2.morphologyEx(chosen_mask * 255, cv2.MORPH_OPEN, kernel, iterations=1)
        chosen_mask = cv2.morphologyEx(chosen_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        chosen_mask = (chosen_mask > 0).astype(np.uint8)
    except Exception:
        chosen_mask = (chosen_mask > 0).astype(np.uint8)

    # We always need some foreground; otherwise there is nothing to extract.
    nz = cv2.findNonZero((chosen_mask * 255).astype(np.uint8))
    if nz is None:
        return {"id": ov.id, "ok": False, "reason": "no_foreground"}

    rx2, ry2, rw2, rh2 = 0, 0, ww, hh
    m_tight = chosen_mask
    if tighten_bbox:
        rx, ry, rw, rh = cv2.boundingRect(nz)
        rx, ry, rw, rh = clip_box(rx, ry, rw, rh, ww, hh)

        # Small padding helps avoid cutting off strokes after SAM2 post-processing.
        if pad_px is not None and isinstance(pad_px, int):
            pad = int(max(0, min(48, pad_px)))
        else:
            pad = int(max(0, min(6, round(min(rw, rh) * 0.05))))

        rx2 = max(0, rx - pad)
        ry2 = max(0, ry - pad)
        rw2 = min(ww - rx2, rw + 2 * pad)
        rh2 = min(hh - ry2, rh + 2 * pad)

        m_tight = chosen_mask[ry2 : ry2 + rh2, rx2 : rx2 + rw2]
        roi = roi[ry2 : ry2 + rh2, rx2 : rx2 + rw2]

    alpha = (m_tight.astype(np.uint8) * 255).astype(np.uint8)
    area = int(np.count_nonzero(alpha))
    ratio = float(area) / float(max(1, int(rw2) * int(rh2)))
    iou = float(ious[best]) if np.ndim(ious) > 0 else float(ious)

    # Basic quality gates (do NOT rely on SAM2's IoU score; it can be very small even for good masks).
    if ratio < 0.001:
        reason = "mask_too_small"
        ok = False
    elif ratio > 0.995:
        reason = "mask_too_large"
        ok = False
    else:
        reason = ""
        ok = True

    out = encode_png_rgba(roi, alpha)
    out_bbox = {"x": x + int(rx2), "y": y + int(ry2), "w": int(rw2), "h": int(rh2)}
    resp: Dict[str, Any] = {
        "id": ov.id,
        "ok": ok,
        "granularity": gran,
        "bbox": out_bbox,
        "dataUrl": out if ok else None,
        "reason": reason if not ok else "",
        "stats": {"mode": "alphaMask", "maskArea": area, "maskRatio": ratio, "iou": iou, "device": device},
    }
    if debug:
        resp["debug"] = {
            "crop": encode_png_rgba(roi, None),
            "mask": _encode_mask_png(m_tight),
            "bboxCrop": encode_png_rgba(roi_bbox, None),
            "bboxMask": _encode_mask_png(m_roi_bbox),
        }
    return resp


def decode_data_url(data_url: str) -> np.ndarray:
    if not data_url.startswith("data:"):
        raise ValueError("Expected data URL")
    header, b64 = data_url.split(",", 1)
    raw = base64.b64decode(b64)
    buf = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    return img


def bgr_to_hex(bgr: np.ndarray) -> str:
    b = int(bgr[0])
    g = int(bgr[1])
    r = int(bgr[2])
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_bgr(hex_color: str) -> np.ndarray:
    s = str(hex_color or "").strip()
    if not s.startswith("#") or len(s) != 7:
        return np.array([255, 255, 255], dtype=np.uint8)
    try:
        r = int(s[1:3], 16)
        g = int(s[3:5], 16)
        b = int(s[5:7], 16)
        return np.array([b, g, r], dtype=np.uint8)
    except Exception:
        return np.array([255, 255, 255], dtype=np.uint8)


def clip_box(x: int, y: int, w: int, h: int, max_w: int, max_h: int) -> Tuple[int, int, int, int]:
    x = max(0, min(int(x), max_w - 1))
    y = max(0, min(int(y), max_h - 1))
    w = max(1, min(int(w), max_w - x))
    h = max(1, min(int(h), max_h - y))
    return x, y, w, h


def box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter) / float(max(1, union))


def nms_boxes(boxes: List[Tuple[int, int, int, int]], scores: List[float], iou_thr: float) -> List[int]:
    idxs = list(range(len(boxes)))
    idxs.sort(key=lambda i: scores[i], reverse=True)
    keep: List[int] = []
    while idxs:
        i = idxs.pop(0)
        keep.append(i)
        idxs = [j for j in idxs if box_iou(boxes[i], boxes[j]) < iou_thr]
    return keep


def find_blue_mask(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([90, 60, 50], dtype=np.uint8)
    upper = np.array([140, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def estimate_background_bgr(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if h < 2 or w < 2:
        return np.array([255, 255, 255], dtype=np.uint8)

    samples = []
    step = max(1, min(h, w) // 120)
    for x in range(0, w, step):
        samples.append(img_bgr[0, x])
        samples.append(img_bgr[h - 1, x])
    for y in range(0, h, step):
        samples.append(img_bgr[y, 0])
        samples.append(img_bgr[y, w - 1])

    arr = np.array(samples, dtype=np.uint8)
    if arr.size == 0:
        return np.array([255, 255, 255], dtype=np.uint8)
    med = np.median(arr.reshape(-1, 3), axis=0).astype(np.uint8)
    return med


def lab_distance_mask(img_bgr: np.ndarray, ref_bgr: np.ndarray, thr: float) -> np.ndarray:
    ref = ref_bgr.reshape((1, 1, 3)).astype(np.uint8)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    d = np.sqrt(np.sum((lab - ref_lab) ** 2, axis=2))
    return (d > float(thr)).astype(np.uint8) * 255


def box_bg_distance(img_bgr: np.ndarray, box: Tuple[int, int, int, int], bg_bgr: np.ndarray) -> float:
    h, w = img_bgr.shape[:2]
    x, y, ww, hh = box
    x, y, ww, hh = clip_box(x, y, ww, hh, w, h)
    if ww < 2 or hh < 2:
        return 0.0

    pad = 2
    x2 = max(0, min(w - 1, x + pad))
    y2 = max(0, min(h - 1, y + pad))
    ww2 = max(1, ww - 2 * pad)
    hh2 = max(1, hh - 2 * pad)
    roi = img_bgr[y2 : y2 + hh2, x2 : x2 + ww2]
    if roi.size == 0:
        return 0.0

    step = max(1, min(roi.shape[0], roi.shape[1]) // 28)
    samples = roi[::step, ::step].reshape(-1, 3)
    if samples.size == 0:
        return 0.0
    med = np.median(samples, axis=0).astype(np.uint8)

    a = cv2.cvtColor(med.reshape((1, 1, 3)), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    b = cv2.cvtColor(bg_bgr.reshape((1, 1, 3)), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    return float(np.linalg.norm(a - b))


def select_blue_connectors(blue_mask: np.ndarray) -> np.ndarray:
    """
    Keep only "thick" blue components (likely connectors/arrow strokes), and drop thin borders.
    This avoids wiping node borders when nodes use a blue stroke theme.
    """
    if blue_mask is None or blue_mask.size == 0:
        return np.zeros_like(blue_mask)

    m = (blue_mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)

    for lbl in range(1, num):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < 120:
            continue
        w = int(stats[lbl, cv2.CC_STAT_WIDTH])
        h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        if w <= 0 or h <= 0:
            continue
        fill_ratio = float(area) / float(max(1, w * h))
        ar = float(w) / float(max(1, h))

        # Thin borders tend to have very low fill ratio over a large bbox.
        # Connectors/arrow strokes have moderate fill ratio (thicker stroke relative to bbox).
        if fill_ratio >= 0.12 or (area >= 500 and (ar >= 7.0 or ar <= 1 / 7.0)):
            out[labels == lbl] = 1

    out = (out * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1)
    return out


@lru_cache(maxsize=1)
def get_ocr():
    try:
        from paddleocr import PaddleOCR  # type: ignore

        # Chinese + English mixed figures
        return PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
    except Exception:
        return None


def ocr_text(img_bgr: np.ndarray) -> List[Dict[str, Any]]:
    ocr = get_ocr()
    if ocr is None:
        return []
    # PaddleOCR expects RGB or BGR both ok; keep BGR.
    out = ocr.ocr(img_bgr, cls=True)
    items: List[Dict[str, Any]] = []
    if not out:
        return items
    for line in out[0]:
        poly = line[0]
        text, conf = line[1]
        if conf is None:
            conf = 0.0
        if conf < 0.5:
            continue
        pts = np.array(poly, dtype=np.float32)
        x = int(np.floor(np.min(pts[:, 0])))
        y = int(np.floor(np.min(pts[:, 1])))
        w = int(np.ceil(np.max(pts[:, 0])) - x)
        h = int(np.ceil(np.max(pts[:, 1])) - y)
        if w <= 2 or h <= 2:
            continue
        items.append(
            {
                "text": str(text),
                "confidence": float(conf),
                "poly": [[float(p[0]), float(p[1])] for p in poly],
                "bbox": (x, y, w, h),
            }
        )
    return items


def normalize_text_items_from_request(raw_items: Optional[List[Dict[str, Any]]], shape: Tuple[int, int]) -> List[Dict[str, Any]]:
    """
    Accept text items from the Node server (LLM OCR) and normalize into the same format as `ocr_text`:
    - text: str
    - confidence: float
    - bbox: (x,y,w,h)
    - poly: [[x,y], ...]  (rect poly when not provided)
    Also attaches a stable `_idx` for tracking/assignment.
    """
    h, w = shape
    if not raw_items:
        return []
    out: List[Dict[str, Any]] = []
    idx = 0
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "") or "").strip()
        bbox_raw = item.get("bbox") if isinstance(item.get("bbox"), (dict, list, tuple)) else None
        if bbox_raw is None:
            bbox_raw = item.get("box") if isinstance(item.get("box"), (dict, list, tuple)) else None
        if bbox_raw is None:
            bbox_raw = item.get("geometry") if isinstance(item.get("geometry"), (dict, list, tuple)) else None
        if bbox_raw is None:
            bbox_raw = item

        x = y = ww = hh = None
        if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) >= 4:
            try:
                x = float(bbox_raw[0])
                y = float(bbox_raw[1])
                ww = float(bbox_raw[2])
                hh = float(bbox_raw[3])
            except Exception:
                x = y = ww = hh = None
        elif isinstance(bbox_raw, dict):
            try:
                if "x" in bbox_raw and "y" in bbox_raw and ("w" in bbox_raw or "width" in bbox_raw):
                    x = float(bbox_raw.get("x", 0))
                    y = float(bbox_raw.get("y", 0))
                    ww = float(bbox_raw.get("w", bbox_raw.get("width", 0)))
                    hh = float(bbox_raw.get("h", bbox_raw.get("height", 0)))
                elif "x1" in bbox_raw and "y1" in bbox_raw and "x2" in bbox_raw and "y2" in bbox_raw:
                    x = float(bbox_raw.get("x1", 0))
                    y = float(bbox_raw.get("y1", 0))
                    ww = float(bbox_raw.get("x2", 0)) - x
                    hh = float(bbox_raw.get("y2", 0)) - y
            except Exception:
                x = y = ww = hh = None

        if x is None or y is None or ww is None or hh is None:
            continue
        xi = int(np.floor(x))
        yi = int(np.floor(y))
        wi = int(np.ceil(ww))
        hi = int(np.ceil(hh))
        xi, yi, wi, hi = clip_box(xi, yi, wi, hi, w, h)
        if wi <= 2 or hi <= 2:
            continue

        conf_raw = item.get("confidence", item.get("score", 0.75))
        try:
            conf = float(conf_raw)
        except Exception:
            conf = 0.75
        conf = float(max(0.0, min(1.0, conf)))

        poly = item.get("poly")
        if not poly:
            poly = [[float(xi), float(yi)], [float(xi + wi), float(yi)], [float(xi + wi), float(yi + hi)], [float(xi), float(yi + hi)]]
        else:
            try:
                pts = []
                for p in poly:
                    if not isinstance(p, (list, tuple)) or len(p) < 2:
                        continue
                    px = float(p[0])
                    py = float(p[1])
                    px = float(max(0.0, min(float(w), px)))
                    py = float(max(0.0, min(float(h), py)))
                    pts.append([px, py])
                if len(pts) >= 3:
                    poly = pts
                else:
                    poly = [[float(xi), float(yi)], [float(xi + wi), float(yi)], [float(xi + wi), float(yi + hi)], [float(xi), float(yi + hi)]]
            except Exception:
                poly = [[float(xi), float(yi)], [float(xi + wi), float(yi)], [float(xi + wi), float(yi + hi)], [float(xi), float(yi + hi)]]

        out.append({"text": text, "confidence": conf, "poly": poly, "bbox": (int(xi), int(yi), int(wi), int(hi)), "_idx": int(idx)})
        idx += 1

        if len(out) >= 320:
            break
    return out


def text_mask_from_polys(shape: Tuple[int, int], text_items: List[Dict[str, Any]]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for item in text_items:
        poly = item.get("poly")
        if not poly:
            continue
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def text_ink_mask(img_bgr: np.ndarray, text_items: List[Dict[str, Any]]) -> np.ndarray:
    """
    A conservative mask of *ink pixels* only (not the whole text bbox),
    to avoid wiping out overlays that sit behind/around labels.
    """
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if not text_items:
        return mask

    gray_all = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    for item in text_items:
        bb = item.get("bbox")
        if not bb:
            continue
        x, y, ww, hh = bb
        x, y, ww, hh = clip_box(x, y, ww, hh, w, h)
        if ww < 6 or hh < 6:
            continue

        crop = gray_all[y : y + hh, x : x + ww]
        if crop.size == 0:
            continue

        # Determine threshold direction by background brightness.
        med = float(np.median(crop))
        # Adaptive threshold block size must be odd and <= min(w,h)
        bs = int(max(9, min(31, min(ww, hh) | 1)))
        if bs % 2 == 0:
            bs += 1
        bs = max(3, min(bs, min(ww, hh) if min(ww, hh) % 2 == 1 else min(ww, hh) - 1))
        if bs < 3:
            continue

        # Bright bg -> dark ink; dark bg -> bright ink.
        thresh_type = cv2.THRESH_BINARY_INV if med > 150 else cv2.THRESH_BINARY
        try:
            ink = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresh_type, bs, 6)
        except Exception:
            _, ink = cv2.threshold(crop, 0, 255, thresh_type | cv2.THRESH_OTSU)

        poly = item.get("poly")
        if poly:
            local = np.zeros((hh, ww), dtype=np.uint8)
            pts = np.array(poly, dtype=np.int32)
            pts[:, 0] -= x
            pts[:, 1] -= y
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(local, [pts], 255)
            ink = cv2.bitwise_and(ink, local)

        # Keep mask thin.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel, iterations=1)
        ink = cv2.dilate(ink, kernel, iterations=1)
        mask[y : y + hh, x : x + ww] = cv2.bitwise_or(mask[y : y + hh, x : x + ww], ink)

    return mask


def canny_edges(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    e = cv2.Canny(g, 50, 140)
    return e


def detect_node_candidates(img_bgr: np.ndarray, blue_connectors: np.ndarray, text_ink: np.ndarray) -> List[Tuple[int, int, int, int]]:
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    edges = canny_edges(gray)
    edges[blue_connectors > 0] = 0
    edges[text_ink > 0] = 0

    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    k9 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

    # Boundary-first: emphasize closed-ish borders without merging whole backplates.
    edges2 = cv2.dilate(edges, k3, iterations=1)
    edges2 = cv2.morphologyEx(edges2, cv2.MORPH_CLOSE, k5, iterations=1)
    edges2 = cv2.morphologyEx(edges2, cv2.MORPH_CLOSE, k9, iterations=1)

    # Foreground (for light-filled shapes) by distance from estimated background.
    bg = estimate_background_bgr(img_bgr)
    fg = lab_distance_mask(img_bgr, bg, thr=10.0)
    fg[blue_connectors > 0] = 0
    fg[text_ink > 0] = 0
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k5, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k9, iterations=1)

    # IMPORTANT: keep internal contours (nodes inside a large light panel).
    contours_a, _ = cv2.findContours(edges2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_b, _ = cv2.findContours(fg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[Tuple[int, int, int, int]] = []
    scores: List[float] = []
    img_area = float(w * h)

    def consider_contours(contours: List[np.ndarray], base_score: float) -> None:
        for c in contours:
            x, y, ww, hh = cv2.boundingRect(c)
            area = ww * hh
            if area < 650:
                continue
            if area > 0.85 * img_area:
                continue

            ar = ww / max(1.0, hh)
            if ar > 28 or ar < (1 / 28):
                continue

            roi_edges = edges[y : y + hh, x : x + ww]
            edge_density = float(np.count_nonzero(roi_edges)) / float(max(1, area))

            if area > 0.10 * img_area and edge_density < 0.0018:
                roi = img_bgr[y : y + hh, x : x + ww]
                if roi.size > 0:
                    bg2 = estimate_background_bgr(img_bgr)
                    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
                    bg_lab = cv2.cvtColor(bg2.reshape((1, 1, 3)), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
                    d = float(np.mean(np.sqrt(np.sum((roi_lab - bg_lab) ** 2, axis=2))))
                    if d < 9.5:
                        continue

            size_score = min(1.0, area / (0.05 * img_area))
            border_score = min(1.0, edge_density * 220.0)
            score = base_score + 0.65 * border_score + 0.35 * size_score
            boxes.append((x, y, ww, hh))
            scores.append(float(score))

    consider_contours(contours_a, base_score=0.25)
    consider_contours(contours_b, base_score=0.15)

    if not boxes:
        return []

    keep = nms_boxes(boxes, scores, iou_thr=0.22)
    kept = [boxes[i] for i in keep]
    kept_scored = [(kept[i], scores[keep[i]]) for i in range(len(kept))]
    kept_scored.sort(key=lambda t: (t[1], t[0][2] * t[0][3]), reverse=True)
    return [b for (b, _s) in kept_scored[:80]]


def detect_node_candidates_sam2_amg(
    img_bgr: np.ndarray,
    blue_connectors: np.ndarray,
    text_ink: np.ndarray,
    text_items: Optional[List[Dict[str, Any]]] = None,
    quality_mode: str = "max",
) -> List[Tuple[int, int, int, int]]:
    """
    Best-effort node bbox detection using SAM2 AutomaticMaskGenerator (AMG).

    Design goals for paper-figure screenshots:
    - Prefer real diagram nodes (containers) over internal small fragments.
    - Keep small boxes/circles as nodes if (a) they contain text, or (b) a blue connector touches their border.
    - Drop huge background panels/backplates.
    """
    h, w = img_bgr.shape[:2]
    img_area = float(max(1, w * h))

    # Defensive: caller-provided masks can mismatch decoded image dims if client width/height differ.
    if text_ink is not None and getattr(text_ink, "size", 0) > 0 and text_ink.shape[:2] != (h, w):
        try:
            _warn_counts["node_text_ink_resize"] = int(_warn_counts.get("node_text_ink_resize", 0)) + 1
            if _warn_counts["node_text_ink_resize"] <= 5:
                print(f"[warn] node text_ink shape {text_ink.shape[:2]} != img {(h, w)}; resizing")
        except Exception:
            pass
        try:
            text_ink = cv2.resize(text_ink, (w, h), interpolation=cv2.INTER_NEAREST)
        except Exception:
            text_ink = None

    if blue_connectors is not None and getattr(blue_connectors, "size", 0) > 0 and blue_connectors.shape[:2] != (h, w):
        try:
            _warn_counts["node_blue_resize"] = int(_warn_counts.get("node_blue_resize", 0)) + 1
            if _warn_counts["node_blue_resize"] <= 5:
                print(f"[warn] node blue shape {blue_connectors.shape[:2]} != img {(h, w)}; resizing")
        except Exception:
            pass
        try:
            blue_connectors = cv2.resize(blue_connectors, (w, h), interpolation=cv2.INTER_NEAREST)
        except Exception:
            blue_connectors = None

    try:
        roi_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        max_dim = int(max(w, h))
        qm = str(quality_mode or "max").strip().lower()
        if qm == "balanced":
            points_per_side = int(max(32, min(80, round(max_dim / 12))))
            crop_layers = 1 if max_dim >= 900 else 0
            pred_iou = 0.58
            stab = 0.58
        else:
            points_per_side = int(max(64, min(128, round(max_dim / 8))))
            crop_layers = 3 if max_dim >= 1400 else (2 if max_dim >= 1100 else (1 if max_dim >= 650 else 0))
            pred_iou = 0.55
            stab = 0.55
        amg, _device = _get_sam2_amg(points_per_side=points_per_side, crop_n_layers=crop_layers, pred_iou_thresh=pred_iou, stability_score_thresh=stab)
        masks = amg.generate(roi_rgb)
    except Exception:
        return []

    # Precompute edges for border scoring.
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = canny_edges(gray)
    if blue_connectors is not None:
        edges[blue_connectors > 0] = 0
    if text_ink is not None:
        edges[text_ink > 0] = 0

    # Background estimate for panel filtering.
    bg = estimate_background_bgr(img_bgr)

    def count_text_in_box(box: Tuple[int, int, int, int]) -> int:
        if not text_items:
            return 0
        x, y, ww, hh = box
        x2 = x + ww
        y2 = y + hh
        cnt = 0
        for t in text_items:
            bb = t.get("bbox")
            if not bb:
                continue
            tx, ty, tw, th = bb
            # Use center point containment (robust to small OCR jitter).
            cx = tx + tw * 0.5
            cy = ty + th * 0.5
            if (x + 1) <= cx <= (x2 - 1) and (y + 1) <= cy <= (y2 - 1):
                cnt += 1
        return cnt

    candidates: List[Tuple[Tuple[int, int, int, int], float, int, float, float]] = []
    # (bbox, score, textCount, blueBorderRatio, borderEdgeDensity)

    for m in masks:
        seg = m.get("segmentation") if isinstance(m, dict) else None
        if seg is None:
            continue
        try:
            seg_u8 = (seg.astype(np.uint8) * 255) if hasattr(seg, "astype") else None
        except Exception:
            seg_u8 = None
        if seg_u8 is None or seg_u8.size == 0:
            continue

        # Align AMG mask to the decoded image shape (some builds/providers return transposed/resized masks).
        try:
            if seg_u8.ndim == 3 and seg_u8.shape[2] == 1:
                seg_u8 = seg_u8[:, :, 0]
            if seg_u8.shape[:2] != (h, w):
                if seg_u8.shape[:2] == (w, h):
                    try:
                        _warn_counts["node_amg_mask_transpose"] = int(_warn_counts.get("node_amg_mask_transpose", 0)) + 1
                        if _warn_counts["node_amg_mask_transpose"] <= 5:
                            print(f"[warn] node amg mask transposed: {seg_u8.shape[:2]} -> {(h, w)}")
                    except Exception:
                        pass
                    seg_u8 = seg_u8.T
                else:
                    try:
                        _warn_counts["node_amg_mask_resize"] = int(_warn_counts.get("node_amg_mask_resize", 0)) + 1
                        if _warn_counts["node_amg_mask_resize"] <= 5:
                            print(f"[warn] node amg mask shape {seg_u8.shape[:2]} != img {(h, w)}; resizing")
                    except Exception:
                        pass
                    seg_u8 = cv2.resize(seg_u8, (w, h), interpolation=cv2.INTER_NEAREST)
        except Exception:
            continue

        # Clean mask for bbox and solidity.
        seg_clean = seg_u8.copy()
        if text_ink is not None and text_ink.size > 0:
            try:
                if text_ink.shape[:2] == seg_clean.shape[:2]:
                    seg_clean[text_ink > 0] = 0
            except Exception:
                pass
        if blue_connectors is not None and blue_connectors.size > 0:
            try:
                if blue_connectors.shape[:2] == seg_clean.shape[:2]:
                    seg_clean[blue_connectors > 0] = 0
            except Exception:
                pass

        nz = cv2.findNonZero(seg_clean)
        if nz is None:
            continue
        x, y, ww, hh = cv2.boundingRect(nz)
        if ww < 18 or hh < 18:
            continue
        bbox_area = float(ww * hh)
        if bbox_area < 650:
            continue
        frac = bbox_area / img_area
        if frac > 0.75:
            continue
        ar = float(ww) / float(max(1, hh))
        if ar > 30.0 or ar < (1.0 / 30.0):
            continue

        area = float(np.count_nonzero(seg_clean))
        solidity = area / max(1.0, bbox_area)
        if solidity < 0.05:
            continue

        # Compute border ring stats.
        t = int(max(2, min(18, round(min(ww, hh) * 0.08))))
        ring_area = float(max(1, (2 * t * ww + 2 * t * hh - 4 * t * t)))
        edges_roi = edges[y : y + hh, x : x + ww]
        if edges_roi.size == 0:
            continue
        ring = np.zeros((hh, ww), dtype=np.uint8)
        ring[:t, :] = 1
        ring[-t:, :] = 1
        ring[:, :t] = 1
        ring[:, -t:] = 1
        border_edges = float(np.count_nonzero((edges_roi > 0) & (ring > 0)))
        border_edge_density = border_edges / ring_area

        blue_border_ratio = 0.0
        if blue_connectors is not None and blue_connectors.size > 0:
            blue_roi = blue_connectors[y : y + hh, x : x + ww]
            if blue_roi.size > 0:
                blue_border_ratio = float(np.count_nonzero((blue_roi > 0) & (ring > 0))) / ring_area

        # Drop likely huge, background-like panels.
        dist = box_bg_distance(img_bgr, (x, y, ww, hh), bg)
        if frac > 0.20 and dist < 12.0 and border_edge_density < 0.004:
            continue

        text_count = count_text_in_box((x, y, ww, hh))

        # Core node-ness rule:
        # - has text, or
        # - touched by connector at border, or
        # - large enough AND has strong border edges.
        is_node_like = bool(text_count >= 1 or blue_border_ratio >= 0.0015 or (frac >= 0.012 and border_edge_density >= 0.010))
        if not is_node_like:
            continue

        pred_iou = float(m.get("predicted_iou", 0.0)) if isinstance(m, dict) else 0.0
        stab = float(m.get("stability_score", 0.0)) if isinstance(m, dict) else 0.0
        score = float(0.60 * pred_iou + 0.40 * stab + (0.10 if text_count >= 1 else 0.0) + min(0.10, border_edge_density * 3.0))
        candidates.append(((int(x), int(y), int(ww), int(hh)), score, int(text_count), float(blue_border_ratio), float(border_edge_density)))

    if not candidates:
        return []

    # NMS to remove duplicates.
    boxes = [c[0] for c in candidates]
    scores = [float(c[1]) for c in candidates]
    keep = nms_boxes(boxes, scores, iou_thr=0.35)
    kept = [candidates[i] for i in keep]
    kept.sort(key=lambda c: (float(c[1]), float(c[0][2] * c[0][3])), reverse=True)

    # Drop backplates (contain many others + background-like).
    base_boxes = [k[0] for k in kept]
    contains_count: List[int] = []
    for i, a in enumerate(base_boxes):
        ax, ay, aw, ah = a
        ccount = 0
        for j, b in enumerate(base_boxes):
            if i == j:
                continue
            bx, by, bw, bh = b
            if bx >= ax + 3 and by >= ay + 3 and (bx + bw) <= (ax + aw - 3) and (by + bh) <= (ay + ah - 3):
                ccount += 1
        contains_count.append(ccount)

    pruned: List[Tuple[int, int, int, int]] = []
    for idx, (bb, score, text_count, blue_ratio, edge_den) in enumerate(kept):
        ax, ay, aw, ah = bb
        frac = float(aw * ah) / img_area
        dist = box_bg_distance(img_bgr, bb, bg)
        contains = int(contains_count[idx])
        is_panel = False
        if frac > 0.22 and contains >= 2 and dist < 11.0:
            is_panel = True
        if contains >= 6 and dist < 12.5:
            is_panel = True
        if frac > 0.30 and dist < 13.5:
            is_panel = True
        if is_panel:
            continue
        pruned.append(bb)

    # Remove internal fragments: boxes inside a larger node, with no text and no connector touch.
    pruned_sorted = sorted(pruned, key=lambda b: (b[2] * b[3]), reverse=True)
    final: List[Tuple[int, int, int, int]] = []
    for i, a in enumerate(pruned_sorted):
        ax, ay, aw, ah = a
        a_area = float(aw * ah)
        a_text = count_text_in_box(a)
        # If it has text, keep.
        if a_text >= 1:
            final.append(a)
            continue
        # If connector touches border, keep.
        if blue_connectors is not None and blue_connectors.size > 0:
            t = int(max(2, min(18, round(min(aw, ah) * 0.08))))
            ring = np.zeros((ah, aw), dtype=np.uint8)
            ring[:t, :] = 1
            ring[-t:, :] = 1
            ring[:, :t] = 1
            ring[:, -t:] = 1
            blue_roi = blue_connectors[ay : ay + ah, ax : ax + aw]
            ring_area = float(max(1, (2 * t * aw + 2 * t * ah - 4 * t * t)))
            blue_border = float(np.count_nonzero((blue_roi > 0) & (ring > 0))) / ring_area if blue_roi.size > 0 else 0.0
            if blue_border >= 0.0015:
                final.append(a)
                continue

        # Drop if almost fully contained by a larger kept box.
        drop = False
        for b in final:
            bx, by, bw, bh = b
            if ax >= bx + 2 and ay >= by + 2 and (ax + aw) <= (bx + bw - 2) and (ay + ah) <= (by + bh - 2):
                b_area = float(bw * bh)
                if a_area / max(1.0, b_area) <= 0.42:
                    drop = True
                    break
        if not drop:
            final.append(a)

    # Cap to a reasonable size.
    final = final[:60]
    return final


def classify_shape(contour: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[str, float]:
    x, y, w, h = bbox
    peri = cv2.arcLength(contour, True)
    if peri <= 0:
        # Default to editable container rather than downgrading to overlay.
        return ("roundRect", 0.72)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    area = cv2.contourArea(contour)
    if area <= 0:
        return ("roundRect", 0.72)
    circ = float(4 * np.pi * area / max(1.0, peri * peri))

    # ellipse-like
    if circ > 0.72 and 0.6 < (w / max(1.0, h)) < 1.6:
        return ("ellipse", 0.75)

    if len(approx) == 4:
        pts = approx.reshape((4, 2)).astype(np.float32)
        # sort by y then x
        pts = pts[np.argsort(pts[:, 1])]
        top = pts[:2]
        bot = pts[2:]
        top = top[np.argsort(top[:, 0])]
        bot = bot[np.argsort(bot[:, 0])]
        tl, tr = top
        bl, br = bot

        def _parallel(v1: np.ndarray, v2: np.ndarray, tol: float = 0.16) -> bool:
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 <= 1e-6 or n2 <= 1e-6:
                return False
            cross = float(abs(v1[0] * v2[1] - v1[1] * v2[0]))
            return (cross / (n1 * n2)) < float(tol)

        v_top = tr - tl
        v_bot = br - bl
        v_left = bl - tl
        v_right = br - tr
        top_len = float(np.linalg.norm(v_top))
        bot_len = float(np.linalg.norm(v_bot))

        # Trapezoid-like: top/bottom parallel (usually horizontal) with noticeably different widths,
        # and left/right sides NOT parallel. Many paper figures use this for "encoder/decoder".
        # For this product we prefer roundRect by default for editability; users can switch to trapezoid in calibration.
        try:
            is_tb_parallel = _parallel(v_top, v_bot, tol=0.18)
            is_lr_parallel = _parallel(v_left, v_right, tol=0.18)
            tb_ratio = (min(top_len, bot_len) / max(1e-6, max(top_len, bot_len))) if (top_len > 1e-6 and bot_len > 1e-6) else 1.0
            is_tb_horizontal = (abs(float(v_top[1])) / max(1e-6, top_len) < 0.28) and (abs(float(v_bot[1])) / max(1e-6, bot_len) < 0.28)
            if is_tb_parallel and (not is_lr_parallel) and is_tb_horizontal and tb_ratio < 0.93:
                return ("roundRect", 0.72)
        except Exception:
            pass

        # Measure angles via dot products
        def angle(a, b, c) -> float:
            ba = a - b
            bc = c - b
            denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
            cosv = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
            return float(np.degrees(np.arccos(cosv)))

        ang1 = angle(tr, tl, bl)
        ang2 = angle(tl, tr, br)
        ang3 = angle(tr, br, bl)
        ang4 = angle(br, bl, tl)
        angles = [ang1, ang2, ang3, ang4]
        rightish = sum(1 for a in angles if 70 <= a <= 110)
        if rightish >= 3:
            return ("roundRect", 0.7)

        # diamond / rhombus heuristic: corners near midpoints
        cx = x + w / 2.0
        cy = y + h / 2.0
        dists = [np.linalg.norm(p - np.array([cx, cy], dtype=np.float32)) for p in pts]
        if max(dists) - min(dists) < 0.18 * max(w, h):
            return ("rhombus", 0.6)

        # For this product we prefer editable containers by default.
        return ("roundRect", 0.68)

    # Unknown: keep an editable rounded rectangle; classify confidence remains high enough
    # to avoid server-side downgrade-to-overlay (overlay should be for true bitmaps only).
    return ("roundRect", 0.72)


def sample_colors(img_bgr: np.ndarray, contour: np.ndarray, text_mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, str]:
    x, y, w, h = bbox
    h_img, w_img = img_bgr.shape[:2]
    x, y, w, h = clip_box(x, y, w, h, w_img, h_img)

    mask = np.zeros((h, w), dtype=np.uint8)
    c2 = contour.copy()
    c2[:, 0, 0] -= x
    c2[:, 0, 1] -= y
    cv2.drawContours(mask, [c2], -1, 255, thickness=-1)

    # Remove *ink pixels* from color sampling (don't erase the whole label background).
    tm = text_mask[y : y + h, x : x + w]
    mask2 = cv2.bitwise_and(mask, cv2.bitwise_not(tm))

    # Fill color: sample eroded interior.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    inner = cv2.erode(mask2, kernel, iterations=1)
    ys, xs = np.where(inner > 0)
    if len(xs) < 40:
        ys, xs = np.where(mask2 > 0)
    if len(xs) < 20:
        fill = np.array([255, 255, 255], dtype=np.uint8)
    else:
        pixels = img_bgr[y : y + h, x : x + w][ys, xs]
        fill = np.median(pixels, axis=0).astype(np.uint8)

    # Stroke color: sample contour border ring.
    ring = cv2.dilate(mask2, kernel, iterations=1) - cv2.erode(mask2, kernel, iterations=1)
    ys2, xs2 = np.where(ring > 0)
    if len(xs2) < 30:
        stroke = np.array([26, 99, 235], dtype=np.uint8)
    else:
        pixels2 = img_bgr[y : y + h, x : x + w][ys2, xs2]
        stroke = np.median(pixels2, axis=0).astype(np.uint8)

    return {"fillColor": bgr_to_hex(fill), "strokeColor": bgr_to_hex(stroke)}


def assign_text_to_nodes(text_items: List[Dict[str, Any]], node_boxes: List[Tuple[int, int, int, int]]) -> Dict[int, List[Dict[str, Any]]]:
    # Assign each text bbox to node with max IoU/containment
    by_node: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(node_boxes))}
    for t in text_items:
        tb = t.get("bbox")
        if not tb:
            continue
        tx, ty, tw, th = tb
        best_i = -1
        best_score = 0.0
        for i, nb in enumerate(node_boxes):
            nx, ny, nw, nh = nb
            inter = box_iou((tx, ty, tw, th), nb)
            # containment boost
            contains = tx >= nx and ty >= ny and (tx + tw) <= (nx + nw) and (ty + th) <= (ny + nh)
            score = inter + (0.2 if contains else 0.0)
            if score > best_score:
                best_score = score
                best_i = i
        if best_i >= 0 and best_score > 0.02:
            by_node[best_i].append(t)
    return by_node


def combine_text(items: List[Dict[str, Any]]) -> Tuple[str, Optional[Tuple[int, int, int, int]]]:
    if not items:
        return ("", None)
    # Sort by y then x for reading order
    items_sorted = sorted(items, key=lambda it: (it["bbox"][1], it["bbox"][0]))
    text = "\n".join([it["text"] for it in items_sorted if it.get("text")])
    xs = [it["bbox"][0] for it in items_sorted]
    ys = [it["bbox"][1] for it in items_sorted]
    x2 = [it["bbox"][0] + it["bbox"][2] for it in items_sorted]
    y2 = [it["bbox"][1] + it["bbox"][3] for it in items_sorted]
    bb = (int(min(xs)), int(min(ys)), int(max(x2) - min(xs)), int(max(y2) - min(ys)))
    return (text, bb)


def encode_png_rgba(bgr: np.ndarray, alpha: Optional[np.ndarray] = None) -> Optional[str]:
    if bgr.size == 0:
        return None
    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    if alpha is None:
        rgba[:, :, 3] = 255
    else:
        if alpha.shape[:2] != rgba.shape[:2]:
            return None
        rgba[:, :, 3] = alpha
    ok, png = cv2.imencode(".png", rgba)
    if not ok:
        return None
    return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode("ascii")


def extract_overlay_png(
    img_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    init_mask: Optional[np.ndarray] = None,
    fill_bgr: Optional[np.ndarray] = None,
) -> Optional[str]:
    """
    Extract a PNG (optionally with alpha). Always keeps the *original bbox size*
    to preserve deterministic positioning when anchored in mxGraph.
    """
    x, y, w, h = bbox
    roi = img_bgr[y : y + h, x : x + w]
    if roi.size == 0:
        return None

    # Default: opaque crop (always valid; avoids "no_foreground" drops).
    fallback = encode_png_rgba(roi, None)
    if init_mask is None or fill_bgr is None:
        return fallback

    if init_mask.shape[:2] != roi.shape[:2]:
        return fallback

    try:
        gc_mask = np.full(roi.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
        gc_mask[init_mask > 0] = cv2.GC_PR_FGD

        # Strong background priors: pixels close to fill color.
        fill = fill_bgr.reshape((1, 1, 3)).astype(np.uint8)
        fill_lab = cv2.cvtColor(fill, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
        roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        d = np.sqrt(np.sum((roi_lab - fill_lab) ** 2, axis=2))
        gc_mask[d < 10.0] = cv2.GC_PR_BGD

        # Border is background.
        gc_mask[:1, :] = cv2.GC_BGD
        gc_mask[-1:, :] = cv2.GC_BGD
        gc_mask[:, :1] = cv2.GC_BGD
        gc_mask[:, -1:] = cv2.GC_BGD

        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(roi, gc_mask, None, bgd, fgd, 2, cv2.GC_INIT_WITH_MASK)

        alpha = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        # If grabcut failed, keep opaque crop.
        if int(np.count_nonzero(alpha)) < 25:
            return fallback

        # Light cleanup.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel, iterations=1)
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel, iterations=1)

        out = encode_png_rgba(roi, alpha)
        return out or fallback
    except Exception:
        return fallback


def detect_internal_elements(
    img_bgr: np.ndarray,
    node_bbox: Tuple[int, int, int, int],
    blue: np.ndarray,
    text_ink: np.ndarray,
    fill_bgr: np.ndarray,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns (innerShapes, nodeOverlays) inside a node.

    - innerShapes: simple bars/blocks we can represent as vector shapes.
    - nodeOverlays: complex icons/photos/charts to keep as bitmap overlays, anchored within the node group.
    """
    x, y, w, h = node_bbox
    roi = img_bgr[y : y + h, x : x + w]
    if roi.size == 0:
        return ([], [])

    roi_blue = blue[y : y + h, x : x + w]
    roi_text = text_ink[y : y + h, x : x + w]

    inner_mask = np.ones((h, w), dtype=np.uint8) * 255
    inner_mask[roi_blue > 0] = 0
    inner_mask[roi_text > 0] = 0
    # Avoid node borders (often stroke)
    border = 3
    inner_mask[:border, :] = 0
    inner_mask[-border:, :] = 0
    inner_mask[:, :border] = 0
    inner_mask[:, -border:] = 0

    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    fill_lab = cv2.cvtColor(fill_bgr.reshape((1, 1, 3)), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    d = np.sqrt(np.sum((roi_lab - fill_lab) ** 2, axis=2))
    valid = d[inner_mask > 0]
    if valid.size < 50:
        return ([], [])

    p85 = float(np.percentile(valid, 85))
    thr = max(12.0, min(28.0, p85 * 0.75))
    dist_mask = ((d > thr) & (inner_mask > 0)).astype(np.uint8) * 255
    try:
        inner_area = int(np.count_nonzero(inner_mask > 0))
        dist_area = int(np.count_nonzero(dist_mask > 0))
        frac = float(dist_area) / float(max(1, inner_area))
        # If the "non-fill" mask is too large, our fill estimate is likely off (e.g., strong gradients).
        # Tighten by increasing the threshold so we focus on high-contrast internal elements only.
        if frac > 0.45:
            p95 = float(np.percentile(valid, 95))
            thr2 = max(thr, min(45.0, p95 * 0.85))
            dist_mask = ((d > thr2) & (inner_mask > 0)).astype(np.uint8) * 255
    except Exception:
        pass

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 140)
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, k3, iterations=1)
    edges = cv2.bitwise_and(edges, inner_mask)

    m0 = cv2.bitwise_or(dist_mask, edges)
    # Expand to cover charts with white backgrounds: connect sparse edges into region blobs.
    # NOTE: we keep a copy of the pre-dilation mask (m0) so we can later "tighten" bbox and split merged blobs.
    k7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    m = cv2.dilate(m0, k7, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k7, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k3, iterations=1)

    def pick_spread_points_from_dist(
        dist: np.ndarray,
        allow_mask: np.ndarray,
        max_points: int,
        min_sep_px: int,
        x0: int,
        y0: int,
    ) -> List[Dict[str, float]]:
        if dist.size == 0:
            return []
        ys, xs = np.where(allow_mask > 0)
        if ys.size == 0:
            return []
        vals = dist[ys, xs].astype(np.float32, copy=False)
        order = np.argsort(vals)[::-1]
        chosen: List[Dict[str, float]] = []
        chosen_xy: List[Tuple[int, int]] = []
        sep2 = float(max(1, int(min_sep_px))) ** 2
        for idx in order:
            if len(chosen) >= int(max_points):
                break
            v = float(vals[idx])
            if not (v > 0.0):
                break
            px = int(xs[idx])
            py = int(ys[idx])
            ok = True
            for cx, cy in chosen_xy:
                dx = float(px - cx)
                dy = float(py - cy)
                if (dx * dx + dy * dy) < sep2:
                    ok = False
                    break
            if not ok:
                continue
            chosen_xy.append((px, py))
            chosen.append({"x": float(x0 + px), "y": float(y0 + py)})
        return chosen

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    inner_shapes: List[Dict[str, Any]] = []
    overlays: List[Dict[str, Any]] = []

    node_area = float(max(1, w * h))
    def handle_candidate_box(xx: int, yy: int, ww: int, hh: int) -> None:
        nonlocal inner_shapes, overlays

        if len(overlays) >= 24:
            return

        area = float(ww * hh)
        if area < 120:
            return
        if area > 0.80 * node_area:
            return
        ar = ww / max(1.0, hh)
        if ar > 20 or ar < (1 / 20):
            return

        crop = roi[yy : yy + hh, xx : xx + ww]
        if crop.size == 0:
            return

        # Compute simple stats for classification.
        std = float(np.mean(np.std(crop.reshape(-1, 3), axis=0)))
        left = np.mean(crop[:, : max(1, ww // 6)], axis=(0, 1))
        right = np.mean(crop[:, ww - max(1, ww // 6) :], axis=(0, 1))
        grad_dist = float(np.linalg.norm(left - right))

        is_bar = (ar > 2.2 and hh <= 0.45 * h) or (ar < 0.45 and ww <= 0.45 * w)
        # Treat thin gradient/stripe elements as vectorizable inner shapes (layout details), not bitmaps.
        bar_like = is_bar and area <= 0.25 * node_area and (std < 55 or grad_dist > 20)

        if bar_like:
            fill = bgr_to_hex(np.median(crop.reshape(-1, 3), axis=0))
            style: Dict[str, str] = {"fillColor": fill, "strokeColor": "none"}
            if grad_dist > 26 and ww >= 18:
                style["gradientColor"] = bgr_to_hex(right.astype(np.uint8))
                style["gradientDirection"] = "east"
            inner_shapes.append(
                {
                    "bbox": {"x": x + xx, "y": y + yy, "w": ww, "h": hh},
                    "shapeId": "rectangle",
                    "style": style,
                    "confidence": 0.65,
                }
            )
            return

        # Otherwise treat as an overlay *candidate* (actual extraction is done later by SAM2/crop).
        # We must aggressively avoid "text-only" detections, which commonly look like a light rectangle
        # containing only glyph strokes that OCR may miss.
        init = dist_mask[yy : yy + hh, xx : xx + ww].copy()
        mask01 = (init > 0).astype(np.uint8)
        mask_area = int(np.count_nonzero(mask01))
        mask_ratio = float(mask_area) / float(max(1.0, area))

        # Overlap with our text ink mask inside this node.
        txt = roi_text[yy : yy + hh, xx : xx + ww]
        txt_area = int(np.count_nonzero(txt > 0)) if txt is not None else 0
        txt_ratio = float(txt_area) / float(max(1.0, area))

        # Connected components on the non-fill mask: text tends to be many tiny components.
        try:
            num_cc, _, stats_cc, _ = cv2.connectedComponentsWithStats(mask01, connectivity=8)
            cc_areas = stats_cc[1:, cv2.CC_STAT_AREA] if num_cc > 1 else np.array([], dtype=np.int32)
            cc_count = int(num_cc - 1)
            largest_cc = int(cc_areas.max()) if cc_areas.size else 0
            largest_ratio = float(largest_cc) / float(max(1.0, area))
        except Exception:
            cc_count = 0
            largest_ratio = 0.0

        # Edge density inside candidate.
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop_edges = cv2.Canny(crop_gray, 60, 160)
        edge_ratio = float(np.count_nonzero(crop_edges)) / float(max(1.0, area))

        # Reject near-blank and text-only candidates.
        too_blank = mask_ratio < 0.008 and edge_ratio < 0.012
        too_texty = (
            area >= 900.0
            and mask_ratio < 0.20
            and largest_ratio < 0.10
            and cc_count >= 12
            and (txt_ratio >= 0.05 or edge_ratio >= 0.02)
        )
        if too_blank or too_texty:
            return

        # Classify overlay kind (controls later granularity choice).
        max_dim = max(ww, hh)
        if max_dim <= 92:
            kind = "icon"
        elif std >= 70 or mask_ratio >= 0.45:
            kind = "noise" if std >= 85 else "photo"
        else:
            kind = "icon"

        # Provide SAM2 prompt points (FULL image coordinates) to improve alphaMask extraction.
        # Foreground points are chosen from the pre-dilation blob mask (m0); background points are chosen
        # from within the bbox but outside the blob and away from edges/text.
        m_crop = m0[yy : yy + hh, xx : xx + ww]
        blob = (m_crop > 0).astype(np.uint8)
        allow_fg = blob.copy()
        allow_bg = (1 - blob).astype(np.uint8)

        # Further restrict background to non-text/non-connector interior.
        inner_crop = inner_mask[yy : yy + hh, xx : xx + ww]
        txt_crop = roi_text[yy : yy + hh, xx : xx + ww]
        if inner_crop is not None:
            allow_fg = cv2.bitwise_and(allow_fg, (inner_crop > 0).astype(np.uint8))
            allow_bg = cv2.bitwise_and(allow_bg, (inner_crop > 0).astype(np.uint8))
        if txt_crop is not None:
            allow_fg[txt_crop > 0] = 0
            allow_bg[txt_crop > 0] = 0

        try:
            dist_fg = cv2.distanceTransform((allow_fg > 0).astype(np.uint8), cv2.DIST_L2, 5)
        except Exception:
            dist_fg = allow_fg.astype(np.float32)
        try:
            dist_bg = cv2.distanceTransform((allow_bg > 0).astype(np.uint8), cv2.DIST_L2, 5)
        except Exception:
            dist_bg = allow_bg.astype(np.float32)

        min_sep = int(max(4, min(18, min(ww, hh) * 0.20)))
        fg_points = pick_spread_points_from_dist(dist_fg, allow_fg, max_points=6, min_sep_px=min_sep, x0=x + xx, y0=y + yy)
        bg_points = pick_spread_points_from_dist(dist_bg, allow_bg, max_points=10, min_sep_px=min_sep, x0=x + xx, y0=y + yy)

        # Ensure minimum points for SAM2.
        if len(fg_points) < 2 or len(bg_points) < 2:
            cx = float(x + xx + ww / 2.0)
            cy = float(y + yy + hh / 2.0)
            inset = float(max(2.0, min(ww, hh) * 0.12))
            fg_points = fg_points if len(fg_points) >= 2 else [{"x": cx, "y": cy}, {"x": cx - inset, "y": cy - inset}]
            bg_points = bg_points if len(bg_points) >= 2 else [
                {"x": float(x + xx + inset), "y": float(y + yy + inset)},
                {"x": float(x + xx + ww - inset), "y": float(y + yy + hh - inset)},
            ]

        overlays.append(
            {
                "kind": kind,
                "bbox": {"x": x + xx, "y": y + yy, "w": ww, "h": hh},
                "fgPoints": fg_points,
                "bgPoints": bg_points,
                "confidence": 0.72,
            }
        )

    for c in contours:
        xx, yy, ww, hh = cv2.boundingRect(c)
        # Tighten bbox back to pre-dilation mask to avoid oversized regions.
        sub0 = m0[yy : yy + hh, xx : xx + ww]
        if sub0.size == 0:
            continue
        nz0 = cv2.findNonZero(sub0)
        if nz0 is not None:
            rx, ry, rw, rh = cv2.boundingRect(nz0)
            xx, yy, ww, hh = int(xx + rx), int(yy + ry), int(rw), int(rh)
        else:
            continue

        area = float(ww * hh)

        # If a huge region is detected, split it by connected components on the base mask
        # (this often happens when dilation merges multiple internal elements).
        if area > 0.35 * node_area:
            sub = (m0[yy : yy + hh, xx : xx + ww] > 0).astype(np.uint8)
            try:
                num_cc, labels_cc, stats_cc, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
            except Exception:
                num_cc = 0
                labels_cc = None
                stats_cc = None
            parts: List[Tuple[int, int, int, int, int]] = []
            if num_cc and stats_cc is not None:
                for lbl in range(1, num_cc):
                    ax, ay, aw, ah, aarea = [int(v) for v in stats_cc[lbl, :5]]
                    if aarea < 90:
                        continue
                    parts.append((xx + ax, yy + ay, aw, ah, aarea))
            parts.sort(key=lambda t: t[4], reverse=True)
            # Process top components only.
            for (px, py, pw, ph, _aarea) in parts[:10]:
                handle_candidate_box(px, py, pw, ph)
                if len(overlays) >= 24:
                    break
            continue

        handle_candidate_box(xx, yy, ww, hh)
        if len(overlays) >= 24:
            break

    return (inner_shapes, overlays)


def detect_internal_overlays_sam2_amg(
    img_bgr: np.ndarray,
    node_bbox: Tuple[int, int, int, int],
    blue: np.ndarray,
    text_mask: np.ndarray,
    fill_bgr: np.ndarray,
    quality_mode: str = "max",
) -> List[Dict[str, Any]]:
    x, y, w, h = node_bbox
    if w < 24 or h < 24:
        return []

    roi = img_bgr[y : y + h, x : x + w]
    if roi.size == 0:
        return []

    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi_h, roi_w = roi.shape[:2]
    roi_blue = blue[y : y + h, x : x + w] if blue is not None else None
    roi_text = text_mask[y : y + h, x : x + w] if text_mask is not None else None

    # Defensive: upstream masks can be slightly different sizes (e.g., if caller-provided dims
    # mismatch decoded image dims). Always align ROI masks to the actual ROI shape.
    if roi_text is not None and roi_text.size > 0 and roi_text.shape[:2] != (roi_h, roi_w):
        try:
            _warn_counts["roi_text_resize"] = int(_warn_counts.get("roi_text_resize", 0)) + 1
            if _warn_counts["roi_text_resize"] <= 5:
                print(f"[warn] roi_text shape {roi_text.shape[:2]} != roi {(roi_h, roi_w)}; resizing")
        except Exception:
            pass
        try:
            roi_text = cv2.resize(roi_text, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
        except Exception:
            roi_text = None
    if roi_blue is not None and roi_blue.size > 0 and roi_blue.shape[:2] != (roi_h, roi_w):
        try:
            _warn_counts["roi_blue_resize"] = int(_warn_counts.get("roi_blue_resize", 0)) + 1
            if _warn_counts["roi_blue_resize"] <= 5:
                print(f"[warn] roi_blue shape {roi_blue.shape[:2]} != roi {(roi_h, roi_w)}; resizing")
        except Exception:
            pass
        try:
            roi_blue = cv2.resize(roi_blue, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
        except Exception:
            roi_blue = None

    max_dim = int(max(w, h))
    qm = str(quality_mode or "max").strip().lower()
    if qm == "balanced":
        points_per_side = int(max(20, min(44, round(max_dim / 10))))
        crop_layers = 1 if max_dim >= 240 else 0
        pred_iou = 0.62
        stab = 0.62
    else:
        points_per_side = int(max(32, min(64, round(max_dim / 6))))
        crop_layers = 2 if max_dim >= 260 else (1 if max_dim >= 180 else 0)
        pred_iou = 0.60
        stab = 0.60

    try:
        amg, device = _get_sam2_amg(points_per_side=points_per_side, crop_n_layers=crop_layers, pred_iou_thresh=pred_iou, stability_score_thresh=stab)
    except Exception:
        return []

    try:
        masks = amg.generate(roi_rgb)
    except Exception:
        return []

    roi_area = float(max(1, w * h))
    min_area = float(max(35, int(round(0.0009 * roi_area))))

    candidates: List[Dict[str, Any]] = []

    for m in masks:
        seg = m.get("segmentation") if isinstance(m, dict) else None
        if seg is None:
            continue
        try:
            seg_u8 = (seg.astype(np.uint8) * 255) if hasattr(seg, "astype") else None
        except Exception:
            seg_u8 = None
        if seg_u8 is None or seg_u8.size == 0:
            continue

        # SAM2 AMG outputs should match roi size, but some builds/providers may return transposed
        # or resized masks. Always align to the ROI shape to avoid crashes.
        try:
            if seg_u8.ndim == 3 and seg_u8.shape[2] == 1:
                seg_u8 = seg_u8[:, :, 0]
            if seg_u8.shape[:2] != (roi_h, roi_w):
                if seg_u8.shape[:2] == (roi_w, roi_h):
                    try:
                        _warn_counts["amg_mask_transpose"] = int(_warn_counts.get("amg_mask_transpose", 0)) + 1
                        if _warn_counts["amg_mask_transpose"] <= 5:
                            print(f"[warn] amg mask transposed: {seg_u8.shape[:2]} -> {(roi_h, roi_w)}")
                    except Exception:
                        pass
                    seg_u8 = seg_u8.T
                else:
                    try:
                        _warn_counts["amg_mask_resize"] = int(_warn_counts.get("amg_mask_resize", 0)) + 1
                        if _warn_counts["amg_mask_resize"] <= 5:
                            print(f"[warn] amg mask shape {seg_u8.shape[:2]} != roi {(roi_h, roi_w)}; resizing")
                    except Exception:
                        pass
                    seg_u8 = cv2.resize(seg_u8, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
        except Exception:
            continue

        if roi_text is not None and roi_text.size > 0 and roi_text.shape[:2] == seg_u8.shape[:2]:
            seg_u8[roi_text > 0] = 0

        nz = cv2.findNonZero(seg_u8)
        if nz is None:
            continue
        rx, ry, rw, rh = cv2.boundingRect(nz)
        if rw < 8 or rh < 8:
            continue

        bbox_area = float(rw * rh)
        area = float(np.count_nonzero(seg_u8))
        if area < min_area:
            continue
        area_frac = area / roi_area
        if area_frac > 0.92:
            continue
        if rw > 0.97 * w and rh > 0.97 * h:
            continue

        ar = float(rw) / float(max(1, rh))
        if ar > 18.0 or ar < (1.0 / 18.0):
            continue

        solidity = area / max(1.0, bbox_area)
        if solidity < 0.02:
            continue

        if roi_blue is not None:
            try:
                if roi_blue.shape[:2] != seg_u8.shape[:2]:
                    blue_overlap = 0.0
                else:
                    blue_overlap = float(np.count_nonzero((seg_u8 > 0) & (roi_blue > 0))) / float(max(1.0, area))
            except Exception:
                blue_overlap = 0.0
            # Likely a connector stroke: mostly blue and elongated/thin.
            if blue_overlap > 0.50 and (ar >= 4.0 or ar <= 1.0 / 4.0) and solidity < 0.25:
                continue

        touches_border = rx <= 1 or ry <= 1 or (rx + rw) >= (w - 1) or (ry + rh) >= (h - 1)
        if touches_border and (bbox_area / roi_area) > 0.65 and solidity < 0.35:
            continue

        try:
            crop = roi[ry : ry + rh, rx : rx + rw]
            mask_crop = seg_u8[ry : ry + rh, rx : rx + rw]
            if crop.size == 0:
                continue
            pixels = crop[mask_crop > 0]
            if pixels.size > 0:
                mean = np.mean(pixels.astype(np.float32), axis=0)
                dist = float(np.linalg.norm(mean - fill_bgr.astype(np.float32)))
                if dist < 7.5 and area_frac > 0.12:
                    continue
        except Exception:
            pass

        pred_iou = float(m.get("predicted_iou", 0.0)) if isinstance(m, dict) else 0.0
        stab = float(m.get("stability_score", 0.0)) if isinstance(m, dict) else 0.0
        score = float(0.65 * pred_iou + 0.35 * stab)

        candidates.append(
            {
                "bbox": (int(rx), int(ry), int(rw), int(rh)),
                "area": float(area),
                "bboxArea": float(bbox_area),
                "solidity": float(solidity),
                "score": float(score),
                "mask": seg_u8,
            }
        )

    if not candidates:
        return []

    boxes = [c["bbox"] for c in candidates]
    scores = [float(c["score"]) for c in candidates]
    keep = nms_boxes(boxes, scores, iou_thr=0.68)
    kept = [candidates[i] for i in keep]
    kept.sort(key=lambda c: (float(c["score"]), float(c["area"])), reverse=True)
    kept = kept[:60]

    def masks_touch(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        """
        Decide if two AMG mask fragments likely belong to the same overlay by checking
        connectivity after a small dilation within the overlap of their expanded bboxes.
        This is much less prone to over-merging than bbox-only proximity rules.
        """
        try:
            ax, ay, aw, ah = a["bbox"]
            bx, by, bw, bh = b["bbox"]
            ma = a.get("mask")
            mb = b.get("mask")
            if ma is None or mb is None:
                return False
            if ma.shape[:2] != (roi_h, roi_w) or mb.shape[:2] != (roi_h, roi_w):
                return False

            pad = int(max(1, min(10, round(min(aw, ah, bw, bh) * 0.10))))
            a1x, a1y, a2x, a2y = ax - pad, ay - pad, ax + aw + pad, ay + ah + pad
            b1x, b1y, b2x, b2y = bx - pad, by - pad, bx + bw + pad, by + bh + pad
            x1 = int(max(0, min(a1x, b1x)))
            y1 = int(max(0, min(a1y, b1y)))
            x2 = int(min(roi_w, max(a2x, b2x)))
            y2 = int(min(roi_h, max(a2y, b2y)))

            # Require expanded bboxes to overlap; otherwise they are too far apart.
            ox1 = max(a1x, b1x)
            oy1 = max(a1y, b1y)
            ox2 = min(a2x, b2x)
            oy2 = min(a2y, b2y)
            if ox2 <= ox1 or oy2 <= oy1:
                return False

            if x2 - x1 < 2 or y2 - y1 < 2:
                return False

            sa = (ma[y1:y2, x1:x2] > 0).astype(np.uint8) * 255
            sb = (mb[y1:y2, x1:x2] > 0).astype(np.uint8) * 255
            if int(np.count_nonzero(sa)) < 6 or int(np.count_nonzero(sb)) < 6:
                return False
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            da = cv2.dilate(sa, ker, iterations=1)
            # If a's dilated region touches b, consider same group.
            touch = int(np.count_nonzero((da > 0) & (sb > 0))) > 0
            if touch:
                return True
            db = cv2.dilate(sb, ker, iterations=1)
            return int(np.count_nonzero((db > 0) & (sa > 0))) > 0
        except Exception:
            return False

    # Semantic merge/denoise layer: group AMG fragments into a small number of top-level overlay candidates.
    # This reduces the "screen full of tiny boxes" issue for complex icons/illustrations without over-merging.
    parent = list(range(len(kept)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri = find(i)
        rj = find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(kept)):
        for j in range(i + 1, len(kept)):
            if masks_touch(kept[i], kept[j]):
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(len(kept)):
        r = find(i)
        groups.setdefault(r, []).append(i)

    def pick_points_from_mask(mask01: np.ndarray, want: int) -> List[Tuple[int, int]]:
        if mask01 is None or mask01.size == 0 or want <= 0:
            return []
        try:
            dist = cv2.distanceTransform(mask01.astype(np.uint8), cv2.DIST_L2, 5)
        except Exception:
            dist = None
        if dist is None or dist.size == 0:
            ys, xs = np.where(mask01 > 0)
            if xs.size == 0:
                return []
            step = int(max(1, round(xs.size / max(1, want))))
            pts = list(zip(xs[::step].tolist(), ys[::step].tolist()))
            return pts[:want]

        out_pts: List[Tuple[int, int]] = []
        d = dist.copy()
        r = int(max(3, min(18, round(min(mask01.shape[0], mask01.shape[1]) * 0.12))))
        for _ in range(want):
            _minv, maxv, _minloc, maxloc = cv2.minMaxLoc(d)
            if float(maxv) <= 0.0:
                break
            px, py = int(maxloc[0]), int(maxloc[1])
            out_pts.append((px, py))
            cv2.circle(d, (px, py), r, 0, -1)
        return out_pts

    # Build merged candidate list. Keep singletons as-is; merge multi-fragment groups.
    merged_cands: List[Dict[str, Any]] = []
    for gidx, idxs in enumerate(groups.values()):
        if not idxs:
            continue
        if len(idxs) == 1:
            merged_cands.append(kept[idxs[0]])
            continue

        gmask = np.zeros((h, w), dtype=np.uint8)
        gscore = 0.0
        for ii in idxs:
            c = kept[ii]
            m = c.get("mask")
            if m is not None and hasattr(m, "shape") and m.shape[:2] == gmask.shape[:2]:
                gmask = cv2.bitwise_or(gmask, m)
            gscore = max(gscore, float(c.get("score", 0.0)))

        nz = cv2.findNonZero(gmask)
        if nz is None:
            continue
        rx, ry, rw, rh = cv2.boundingRect(nz)
        if rw < 8 or rh < 8:
            continue

        merged_cands.append(
            {
                "bbox": (int(rx), int(ry), int(rw), int(rh)),
                "area": float(np.count_nonzero(gmask)),
                "bboxArea": float(rw * rh),
                "solidity": float(np.count_nonzero(gmask)) / float(max(1, rw * rh)),
                "score": float(gscore),
                "mask": gmask,
                "mergedCount": int(len(idxs)),
            }
        )

    # NMS again on merged candidates.
    boxes2 = [c["bbox"] for c in merged_cands]
    scores2 = [float(c.get("score", 0.0)) for c in merged_cands]
    keep2 = nms_boxes(boxes2, scores2, iou_thr=0.62)
    merged_kept = [merged_cands[i] for i in keep2]
    merged_kept.sort(key=lambda c: (float(c.get("score", 0.0)), float(c.get("area", 0.0))), reverse=True)
    merged_kept = merged_kept[:24]

    overlays: List[Dict[str, Any]] = []

    for idx, cand in enumerate(merged_kept):
        rx, ry, rw, rh = cand["bbox"]
        seg_u8 = cand["mask"]
        crop = roi[ry : ry + rh, rx : rx + rw]
        if crop.size == 0:
            continue
        mask_crop = seg_u8[ry : ry + rh, rx : rx + rw]
        if mask_crop.size == 0:
            continue

        area_in_bbox = float(np.count_nonzero(mask_crop))
        if area_in_bbox < 25:
            continue

        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            vals = gray[mask_crop > 0]
            std = float(np.std(vals)) if vals.size > 0 else float(np.std(gray))
            edges = cv2.Canny(gray, 60, 160)
            edge_count = float(np.count_nonzero(edges[mask_crop > 0])) if vals.size > 0 else float(np.count_nonzero(edges))
            edge_density = edge_count / float(max(1.0, area_in_bbox))

            pix = crop.reshape((-1, 3))
            mflat = (mask_crop.reshape((-1,)) > 0)
            pix_fg = pix[mflat]
            if pix_fg.shape[0] > 3200:
                step = int(max(1, round(pix_fg.shape[0] / 3200)))
                pix_fg = pix_fg[::step]
            if pix_fg.size > 0:
                bins = ((pix_fg[:, 2] >> 5).astype(np.int32) << 6) | ((pix_fg[:, 1] >> 5).astype(np.int32) << 3) | (
                    (pix_fg[:, 0] >> 5).astype(np.int32)
                )
                color_bins = int(np.unique(bins).size)
            else:
                color_bins = 0
        except Exception:
            std = 0.0
            edge_density = 0.0
            color_bins = 0.0

        bbox_area = float(max(1, rw * rh))
        solidity = float(cand.get("solidity", 0.0))
        bbox_frac = bbox_area / float(max(1.0, roi_area))

        kind = "icon"
        if bbox_frac >= 0.06 and solidity >= 0.90 and (std >= 95.0 or color_bins >= 160):
            kind = "noise"
        elif bbox_frac >= 0.06 and solidity >= 0.90 and (std >= 65.0 or color_bins >= 90 or edge_density >= 0.12):
            kind = "chart"
        elif bbox_frac >= 0.03 and solidity >= 0.88 and (std >= 75.0 or color_bins >= 120 or edge_density >= 0.16):
            kind = "chart"

        granularity = "opaqueRect" if kind in ("photo", "chart", "plot", "noise", "screenshot") else "alphaMask"

        fg_points: List[Dict[str, float]] = []
        bg_points: List[Dict[str, float]] = []
        alpha = None
        if granularity == "alphaMask":
            try:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                m2 = cv2.morphologyEx(mask_crop, cv2.MORPH_OPEN, kernel, iterations=1)
                m2 = cv2.morphologyEx(m2, cv2.MORPH_CLOSE, kernel, iterations=1)
                alpha = m2
            except Exception:
                alpha = mask_crop

            m01 = (alpha > 0).astype(np.uint8)
            inner = cv2.erode(m01, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
            if int(np.count_nonzero(inner)) < 16:
                inner = m01
            fg_local = pick_points_from_mask(inner, want=5)
            # Background points: pick from outside a dilated foreground to avoid holes.
            dil = cv2.dilate(m01, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
            bg_region = (dil == 0).astype(np.uint8)
            bg_local = pick_points_from_mask(bg_region, want=8)

            for (px, py) in fg_local:
                fg_points.append({"x": float(x + rx + px), "y": float(y + ry + py)})
            for (px, py) in bg_local:
                bg_points.append({"x": float(x + rx + px), "y": float(y + ry + py)})

        data_url = encode_png_rgba(crop, alpha)
        if not data_url:
            continue

        conf = float(max(0.0, min(1.0, cand.get("score", 0.7))))
        overlays.append(
            {
                "id": f"amg_{idx+1}",
                "kind": kind,
                "granularity": granularity,
                "bbox": {"x": int(x + rx), "y": int(y + ry), "w": int(rw), "h": int(rh)},
                "confidence": conf if conf > 0 else 0.72,
                "fgPoints": fg_points,
                "bgPoints": bg_points,
                "dataUrl": data_url,
                "stats": {
                    "source": "sam2_amg",
                    "device": device,
                    "pointsPerSide": points_per_side,
                    "cropLayers": crop_layers,
                    "mergedCount": int(cand.get("mergedCount", 1)),
                },
            }
        )

    return overlays


def build_edges_from_blue(
    img_bgr: np.ndarray,
    blue: np.ndarray,
    node_boxes: List[Tuple[int, int, int, int]],
    node_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    if not node_boxes:
        return []

    h, w = blue.shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    blue2 = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel, iterations=1)
    blue_bin = (blue2 > 0).astype(np.uint8) * 255

    def skeletonize(mask: np.ndarray) -> np.ndarray:
        img = (mask > 0).astype(np.uint8) * 255
        skel = np.zeros_like(img)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        it = 0
        while True:
            it += 1
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded
            if cv2.countNonZero(img) == 0 or it > 220:
                break
        return skel

    def endpoint_pixels(skel: np.ndarray) -> List[Tuple[int, int]]:
        s = (skel > 0).astype(np.uint8)
        nb = cv2.filter2D(s, -1, np.ones((3, 3), np.uint8)) - s
        ys, xs = np.where((s > 0) & (nb == 1))
        pts = list(zip(xs.tolist(), ys.tolist()))
        if len(pts) <= 2:
            return pts
        # Keep at most 6 farthest from centroid to avoid noisy endpoints.
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        pts.sort(key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2, reverse=True)
        return pts[:6]

    def node_distance2(px: int, py: int, box: Tuple[int, int, int, int]) -> float:
        nx, ny, nw, nh = box
        x1, y1, x2, y2 = nx, ny, nx + nw, ny + nh
        dx = 0.0 if x1 <= px <= x2 else (x1 - px if px < x1 else px - x2)
        dy = 0.0 if y1 <= py <= y2 else (y1 - py if py < y1 else py - y2)
        return dx * dx + dy * dy

    def closest_node(px: int, py: int, max_dist: float = 50.0) -> Optional[int]:
        best = None
        best_d = max_dist * max_dist
        for i, nb in enumerate(node_boxes):
            d2 = node_distance2(px, py, nb)
            if d2 < best_d:
                best_d = d2
                best = i
        return best

    def has_arrowhead(component_mask: np.ndarray, endpoint: Tuple[int, int]) -> bool:
        ex, ey = endpoint
        r = 22
        x1 = max(0, ex - r)
        y1 = max(0, ey - r)
        x2 = min(component_mask.shape[1] - 1, ex + r)
        y2 = min(component_mask.shape[0] - 1, ey + r)
        crop = component_mask[y1 : y2 + 1, x1 : x2 + 1]
        if crop.size == 0:
            return False
        contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < 35:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            if len(approx) == 3:
                return True
        return False

    num, labels, stats, _ = cv2.connectedComponentsWithStats((blue_bin > 0).astype(np.uint8), connectivity=8)
    edges: List[Dict[str, Any]] = []
    seen = set()

    for lbl in range(1, num):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < 250:
            continue
        x = int(stats[lbl, cv2.CC_STAT_LEFT])
        y = int(stats[lbl, cv2.CC_STAT_TOP])
        ww = int(stats[lbl, cv2.CC_STAT_WIDTH])
        hh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        if ww * hh < 350:
            continue

        comp = (labels == lbl).astype(np.uint8) * 255
        skel = skeletonize(comp)
        eps = endpoint_pixels(skel)
        if len(eps) < 2:
            continue

        # Map endpoints to nodes.
        mapped = []
        for (px, py) in eps:
            ni = closest_node(px, py)
            if ni is None:
                continue
            mapped.append((px, py, ni))
        # Deduplicate by node.
        uniq = {}
        for px, py, ni in mapped:
            if ni not in uniq:
                uniq[ni] = (px, py)
        if len(uniq) < 2:
            continue

        # Choose best two nodes by proximity of their endpoint points.
        items = list(uniq.items())  # (ni, (px,py))
        best_pair = None
        best_d = 1e18
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (ni, p1) = items[i]
                (nj, p2) = items[j]
                d2 = float((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                if d2 < best_d:
                    best_d = d2
                    best_pair = (ni, p1, nj, p2)
        if not best_pair:
            continue

        ni, p1, nj, p2 = best_pair
        # Determine direction by arrowhead if possible.
        arrow1 = has_arrowhead(comp, p1)
        arrow2 = has_arrowhead(comp, p2)
        if arrow1 and not arrow2:
            src_i, tgt_i = nj, ni
        elif arrow2 and not arrow1:
            src_i, tgt_i = ni, nj
        else:
            # Heuristic: left->right or top->bottom
            a = node_boxes[ni]
            b = node_boxes[nj]
            ax, ay, aw, ah = a
            bx, by, bw, bh = b
            dx = (bx + bw / 2.0) - (ax + aw / 2.0)
            dy = (by + bh / 2.0) - (ay + ah / 2.0)
            if abs(dx) >= abs(dy):
                src_i, tgt_i = (ni, nj) if dx >= 0 else (nj, ni)
            else:
                src_i, tgt_i = (ni, nj) if dy >= 0 else (nj, ni)

        key = (src_i, tgt_i)
        if key in seen:
            continue
        seen.add(key)

        a = node_boxes[src_i]
        b = node_boxes[tgt_i]
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        dx = (bx + bw / 2.0) - (ax + aw / 2.0)
        dy = (by + bh / 2.0) - (ay + ah / 2.0)
        if abs(dx) >= abs(dy):
            source_side = "right" if dx >= 0 else "left"
            target_side = "left" if dx >= 0 else "right"
        else:
            source_side = "bottom" if dy >= 0 else "top"
            target_side = "top" if dy >= 0 else "bottom"

        def nid(i: int) -> str:
            if node_ids and 0 <= i < len(node_ids):
                return str(node_ids[i])
            return f"n{i+1}"

        edges.append(
            {
                "id": f"e{len(edges)+1}",
                "source": nid(src_i),
                "target": nid(tgt_i),
                "sourceSide": source_side,
                "targetSide": target_side,
                "label": "",
                "confidence": 0.72 if (arrow1 or arrow2) else 0.62,
            }
        )
        if len(edges) >= 90:
            break

    return edges


def build_structure(img_bgr: np.ndarray, text_items_override: Optional[List[Dict[str, Any]]] = None, quality_mode: str = "max") -> Dict[str, Any]:
    h, w = img_bgr.shape[:2]
    blue_all = find_blue_mask(img_bgr)
    blue = select_blue_connectors(blue_all)
    text_items = text_items_override if (text_items_override and len(text_items_override) > 0) else ocr_text(img_bgr)
    text_poly = text_mask_from_polys((h, w), text_items)
    text_ink = text_ink_mask(img_bgr, text_items)

    # Prefer SAM2-AMG for node candidates (paper figures); fall back to CV edges if needed.
    node_boxes = detect_node_candidates_sam2_amg(img_bgr, blue, text_ink, text_items=text_items, quality_mode=quality_mode)
    if not node_boxes:
        node_boxes = detect_node_candidates(img_bgr, blue, text_ink)

    # Prune likely background panels/backplates (huge light boxes containing many other nodes).
    if node_boxes:
        bg = estimate_background_bgr(img_bgr)
        img_area = float(w * h)
        counts = []
        for i, a in enumerate(node_boxes):
            ax, ay, aw, ah = a
            contains = 0
            for j, b in enumerate(node_boxes):
                if i == j:
                    continue
                bx, by, bw, bh = b
                if bx >= ax + 3 and by >= ay + 3 and (bx + bw) <= (ax + aw - 3) and (by + bh) <= (ay + ah - 3):
                    contains += 1
            counts.append(contains)

        keep_idx = []
        for i, box in enumerate(node_boxes):
            area = float(box[2] * box[3])
            frac = area / max(1.0, img_area)
            contains = counts[i]
            dist = box_bg_distance(img_bgr, box, bg)

            is_panel = False
            if frac > 0.22 and contains >= 2 and dist < 11.0:
                is_panel = True
            if contains >= 6 and dist < 12.5:
                is_panel = True
            if frac > 0.30 and dist < 13.5:
                is_panel = True

            if not is_panel:
                keep_idx.append(i)

        node_boxes = [node_boxes[i] for i in keep_idx]
        # Keep a sane cap after pruning.
        node_boxes = node_boxes[:60]

    # refine nodes via contours within bbox for shape classification and colors
    nodes: List[Dict[str, Any]] = []
    node_contours: List[np.ndarray] = []

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    base_edges = canny_edges(gray)
    base_edges[blue > 0] = 0
    base_edges[text_poly > 0] = 0

    for idx, b in enumerate(node_boxes):
        x, y, ww, hh = b
        roi_edges = base_edges[y : y + hh, x : x + ww]
        # try to find best contour inside bbox
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        m = cv2.dilate(roi_edges, kernel, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cs:
            c = max(cs, key=cv2.contourArea)
            c[:, 0, 0] += x
            c[:, 0, 1] += y
            node_contours.append(c)
        else:
            # fallback to bbox rectangle contour
            c = np.array([[[x, y]], [[x + ww, y]], [[x + ww, y + hh]], [[x, y + hh]]], dtype=np.int32)
            node_contours.append(c)

    text_by_node = assign_text_to_nodes(text_items, node_boxes)

    img_area = float(max(1, w * h))

    for idx, bbox in enumerate(node_boxes):
        contour = node_contours[idx]
        shape_id, shape_conf = classify_shape(contour, bbox)
        colors = sample_colors(img_bgr, contour, text_ink, bbox)
        assigned = text_by_node.get(idx, [])
        label_text, label_bbox = combine_text(assigned)
        fill_bgr = hex_to_bgr(colors.get("fillColor", "#ffffff"))
        inner_shapes, node_overlays_cv = detect_internal_elements(img_bgr, bbox, blue, text_ink, fill_bgr)
        node_overlays = detect_internal_overlays_sam2_amg(img_bgr, bbox, blue, text_poly, fill_bgr, quality_mode=quality_mode) or node_overlays_cv

        # Heuristic: texture blocks/noise/photos should be rendered as overlay (opaqueRect),
        # not as editable shapes. This catches "data/noise" squares commonly seen in papers.
        x, y, ww, hh = bbox
        roi = img_bgr[y : y + hh, x : x + ww]
        roi_std = float(np.mean(np.std(roi.reshape(-1, 3), axis=0))) if roi.size else 0.0
        frac = float(ww * hh) / img_area
        is_texture = (label_text or "").strip() == "" and roi_std >= 70.0 and frac <= 0.20
        render_mode = "overlay" if is_texture else "shape"

        n: Dict[str, Any] = {
            "id": f"n{idx+1}",
            "bbox": {"x": int(bbox[0]), "y": int(bbox[1]), "w": int(bbox[2]), "h": int(bbox[3])},
            "text": label_text,
            "textBbox": (
                {"x": int(label_bbox[0]), "y": int(label_bbox[1]), "w": int(label_bbox[2]), "h": int(label_bbox[3])}
                if label_bbox
                else None
            ),
            "shapeId": shape_id,
            "render": render_mode,
            "confidence": {"bbox": 0.65, "text": 0.7 if label_text else 0.0, "shape": float(shape_conf)},
            "containerStyle": colors,
            "innerShapes": inner_shapes,
            "nodeOverlays": node_overlays,
        }
        if render_mode == "overlay":
            n["overlay"] = {
                "kind": "noise" if roi_std >= 90.0 else "photo",
                "granularity": "opaqueRect",
                "confidence": 0.85,
            }
        nodes.append(n)

    edges = build_edges_from_blue(img_bgr, blue, node_boxes)

    # Global overlays: keep none by default (we prefer anchoring overlays inside nodes).
    overlays: List[Dict[str, Any]] = []
    return {"nodes": nodes, "edges": edges, "overlays": overlays}


@app.get("/health")
def health() -> Dict[str, Any]:
    torch_ok = importlib.util.find_spec("torch") is not None
    sam2_ok = importlib.util.find_spec("sam2") is not None
    config_ok = False
    cfg_path = ""
    ckpt_ok = False
    try:
        if torch_ok and sam2_ok:
            cfg_path = _resolve_sam2_config_path(SAM2_MODEL_NAME)
            config_ok = os.path.exists(cfg_path)
            ckpt_name = f"{SAM2_MODEL_NAME}.pt" if not SAM2_MODEL_NAME.endswith(".pt") else SAM2_MODEL_NAME
            ckpt_path = os.path.join(WEIGHTS_DIR, ckpt_name)
            ckpt_ok = os.path.exists(ckpt_path) and os.path.getsize(ckpt_path) > 50_000_000
    except Exception:
        config_ok = False
        ckpt_ok = False
    return {
        "ok": "true",
        "sam2": "true" if (torch_ok and sam2_ok and config_ok and ckpt_ok) else "false",
        "sam2Model": SAM2_MODEL_NAME,
        "sam2Config": cfg_path,
        "sam2ConfigOk": config_ok,
        "sam2CheckpointOk": ckpt_ok,
        "capabilities": {"overlaysResolve": True, "augment": True, "analyze": True},
    }


@app.post("/overlays/resolve")
def overlays_resolve(req: OverlayResolveRequest) -> Dict[str, Any]:
    try:
        img = decode_data_url(req.image.dataUrl)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    h, w = img.shape[:2]
    overlays_in = req.overlays or []
    debug = bool(req.debug)
    opts = req.options or OverlayResolveOptions()
    tighten_bbox = bool(True if opts.tightenBbox is None else opts.tightenBbox)
    pad_px = None
    try:
        if opts.padPx is not None:
            pad_px = int(opts.padPx)
    except Exception:
        pad_px = None

    needs_alpha = any(str(o.granularity or "").strip() == "alphaMask" for o in overlays_in)

    # Build masks once per request to improve cropping/alpha extraction (avoid including external labels/blue arrows).
    blue_all = find_blue_mask(img)
    blue = select_blue_connectors(blue_all)
    provided = normalize_text_items_from_request(req.textItems, (h, w)) if req.textItems else []
    text_items = provided if provided else ocr_text(img)
    text_ink = text_ink_mask(img, text_items) if text_items else np.zeros((h, w), dtype=np.uint8)

    predictor = None
    device = None
    sam2_available = True
    sam2_error = ""
    if needs_alpha:
        try:
            predictor, device = _get_sam2_predictor(img)
        except Exception as exc:
            sam2_available = False
            sam2_error = str(exc)[:240]
            try:
                print(f"[sam2] init failed: {sam2_error}")
            except Exception:
                pass
            predictor = None
            device = None

    out = []
    for ov in overlays_in:
        try:
            item = _resolve_overlay_one(
                img,
                ov,
                debug=debug,
                predictor=predictor,
                device=device,
                sam2_available=sam2_available,
                tighten_bbox=tighten_bbox,
                pad_px=pad_px,
                text_ink_full=text_ink,
                blue_full=blue,
            )
            if not sam2_available and isinstance(item, dict) and item.get("reason") == "sam2_unavailable" and sam2_error:
                item["detail"] = sam2_error
            out.append(item)
        except Exception as exc:
            out.append({"id": ov.id, "ok": False, "reason": "exception", "detail": str(exc)[:300]})

    return {
        "overlays": out,
        "meta": {"backend": "sam2+crop", "imageWidth": w, "imageHeight": h, "ocr": ("provided" if provided else ("paddleocr" if get_ocr() is not None else "none"))},
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> Dict[str, Any]:
    try:
        img = decode_data_url(req.image.dataUrl)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    h, w = img.shape[:2]
    quality_mode = _normalize_quality_mode(req.options)

    # If the client-reported image size differs from decoded size, rescale provided OCR boxes.
    sx = float(w) / float(max(1, int(req.imageWidth or w)))
    sy = float(h) / float(max(1, int(req.imageHeight or h)))
    raw_items = _rescale_text_items_inplace(list(req.textItems or []), sx, sy) if req.textItems else []
    provided = normalize_text_items_from_request(raw_items, (h, w)) if raw_items else []
    text_items = provided if provided else ocr_text(img)
    structure = build_structure(img, text_items_override=text_items, quality_mode=quality_mode)
    ocr = "provided" if provided else ("paddleocr" if get_ocr() is not None else "none")
    return {
        "structure": structure,
        "meta": {
            "imageWidth": w,
            "imageHeight": h,
            "backend": "cv-route1",
            "ocr": ocr,
            "textItems": _export_text_items_for_node_server(text_items),
            "counts": {"nodes": len(structure.get("nodes", []) or []), "edges": len(structure.get("edges", []) or [])},
        },
    }


@app.post("/augment")
def augment(req: AugmentRequest) -> Dict[str, Any]:
    try:
        img = decode_data_url(req.image.dataUrl)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    h, w = img.shape[:2]
    quality_mode = _normalize_quality_mode(req.options)
    blue_all = find_blue_mask(img)
    blue = select_blue_connectors(blue_all)

    sx = float(w) / float(max(1, int(req.imageWidth or w)))
    sy = float(h) / float(max(1, int(req.imageHeight or h)))
    raw_items = _rescale_text_items_inplace(list(req.textItems or []), sx, sy) if req.textItems else []
    provided = normalize_text_items_from_request(raw_items, (h, w)) if raw_items else []
    text_items = provided if provided else ocr_text(img)
    text_ink = text_ink_mask(img, text_items) if text_items else np.zeros((h, w), dtype=np.uint8)
    text_poly = text_mask_from_polys((h, w), text_items) if text_items else np.zeros((h, w), dtype=np.uint8)

    out_nodes: List[Dict[str, Any]] = []
    node_boxes_for_edges: List[Tuple[int, int, int, int]] = []
    node_ids_for_edges: List[str] = []
    img_area = float(max(1, w * h))

    for n in req.nodes:
        bb = n.bbox or {}
        x = int(bb.get("x", 0))
        y = int(bb.get("y", 0))
        ww = int(bb.get("w", bb.get("width", 1)))
        hh = int(bb.get("h", bb.get("height", 1)))
        x, y, ww, hh = clip_box(x, y, ww, hh, w, h)

        contour = np.array([[[x, y]], [[x + ww, y]], [[x + ww, y + hh]], [[x, y + hh]]], dtype=np.int32)
        colors = sample_colors(img, contour, text_ink, (x, y, ww, hh))
        fill_bgr = hex_to_bgr(colors.get("fillColor", "#ffffff"))

        inner_shapes, node_overlays_cv = detect_internal_elements(img, (x, y, ww, hh), blue, text_ink, fill_bgr)
        node_overlays = detect_internal_overlays_sam2_amg(img, (x, y, ww, hh), blue, text_poly, fill_bgr, quality_mode=quality_mode) or node_overlays_cv

        # Keep a filtered set for connector detection (drop huge background panels).
        if float(ww * hh) / img_area <= 0.65:
            node_boxes_for_edges.append((x, y, ww, hh))
            node_ids_for_edges.append(str(n.id))

        out_nodes.append(
            {
                "id": str(n.id),
                "bbox": {"x": x, "y": y, "w": ww, "h": hh},
                "containerStyle": colors,
                "innerShapes": inner_shapes,
                "nodeOverlays": node_overlays,
            }
        )

    edges = build_edges_from_blue(img, blue, node_boxes_for_edges, node_ids=node_ids_for_edges)

    return {
        "nodes": out_nodes,
        "edges": edges,
        "meta": {
            "backend": "cv-augment",
            "imageWidth": w,
            "imageHeight": h,
            "ocr": ("provided" if provided else ("paddleocr" if get_ocr() is not None else "none")),
            "counts": {"edges": len(edges)},
        },
    }


@app.post("/debug/annotate")
def debug_annotate(req: AnalyzeRequest) -> Dict[str, Any]:
    try:
        img = decode_data_url(req.image.dataUrl)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    h, w = img.shape[:2]
    quality_mode = _normalize_quality_mode(req.options)
    sx = float(w) / float(max(1, int(req.imageWidth or w)))
    sy = float(h) / float(max(1, int(req.imageHeight or h)))
    raw_items = _rescale_text_items_inplace(list(req.textItems or []), sx, sy) if req.textItems else []
    provided = normalize_text_items_from_request(raw_items, (h, w)) if raw_items else []
    text_items = provided if provided else ocr_text(img)
    structure = build_structure(img, text_items_override=text_items, quality_mode=quality_mode)
    vis = img.copy()

    for n in structure.get("nodes", []):
        bb = n.get("bbox") or {}
        x, y, w, h = clip_box(bb.get("x", 0), bb.get("y", 0), bb.get("w", 1), bb.get("h", 1), vis.shape[1], vis.shape[0])
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(vis, str(n.get("id", "")), (x, max(0, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        tb = n.get("textBbox")
        if tb and isinstance(tb, dict):
            tx, ty, tw, th = clip_box(tb.get("x", 0), tb.get("y", 0), tb.get("w", 1), tb.get("h", 1), vis.shape[1], vis.shape[0])
            cv2.rectangle(vis, (tx, ty), (tx + tw, ty + th), (0, 200, 0), 2)

    for n in structure.get("nodes", []):
        for ov in n.get("nodeOverlays", []) or []:
            bb = ov.get("bbox") or {}
            x, y, w, h = clip_box(bb.get("x", 0), bb.get("y", 0), bb.get("w", 1), bb.get("h", 1), vis.shape[1], vis.shape[0])
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 255), 2)

    for e in structure.get("edges", []):
        # simple indicator: draw a line between node centers
        sid = str(e.get("source", ""))
        tid = str(e.get("target", ""))
        ns = next((n for n in structure["nodes"] if n["id"] == sid), None)
        nt = next((n for n in structure["nodes"] if n["id"] == tid), None)
        if not ns or not nt:
            continue
        b1 = ns["bbox"]
        b2 = nt["bbox"]
        p1 = (int(b1["x"] + b1["w"] / 2), int(b1["y"] + b1["h"] / 2))
        p2 = (int(b2["x"] + b2["w"] / 2), int(b2["y"] + b2["h"] / 2))
        cv2.line(vis, p1, p2, (255, 0, 0), 2)

    ok, png = cv2.imencode(".png", vis)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode PNG")
    data_url = "data:image/png;base64," + base64.b64encode(png.tobytes()).decode("ascii")
    ocr = "paddleocr" if get_ocr() is not None else "none"
    return {
        "annotated": data_url,
        "structure": structure,
        "meta": {
            "backend": "cv-route1",
            "ocr": ocr,
            "textItems": _export_text_items_for_node_server(text_items),
            "counts": {"nodes": len(structure.get("nodes", []) or []), "edges": len(structure.get("edges", []) or [])},
        },
    }


@app.post("/debug/annotate_boxes")
def debug_annotate_boxes(req: AnnotateBoxesRequest) -> Dict[str, Any]:
    try:
        img = decode_data_url(req.image.dataUrl)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    vis = img.copy()
    h, w = vis.shape[:2]

    def draw_label(text: str, x: int, y: int, color_bgr: np.ndarray) -> None:
        if not text:
            return
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55 if max(w, h) <= 1400 else 0.7
        thick = 1
        (tw, th), base = cv2.getTextSize(text, font, scale, thick)
        pad = 3
        x0 = max(0, min(int(x), w - 1))
        y0 = max(0, min(int(y), h - 1))
        y1 = max(0, y0 - th - base - pad * 2)
        x2 = min(w - 1, x0 + tw + pad * 2)
        y2 = min(h - 1, y0)
        cv2.rectangle(vis, (x0, y1), (x2, y2), tuple(int(v) for v in color_bgr.tolist()), thickness=-1)
        cv2.putText(
            vis,
            text,
            (x0 + pad, y2 - base - pad),
            font,
            scale,
            (255, 255, 255),
            thick,
            cv2.LINE_AA,
        )

    # Draw nodes (blue) and their text bboxes (green)
    for n in (req.nodes or []):
        bb = n.bbox or {}
        x = int(bb.get("x", 0))
        y = int(bb.get("y", 0))
        ww = int(bb.get("w", bb.get("width", 1)))
        hh = int(bb.get("h", bb.get("height", 1)))
        x, y, ww, hh = clip_box(x, y, ww, hh, w, h)
        node_color = hex_to_bgr("#2563eb")
        cv2.rectangle(vis, (x, y), (x + ww, y + hh), tuple(int(v) for v in node_color.tolist()), 2)
        draw_label(str(n.id), x, max(0, y - 2), node_color)

        tb = n.textBbox
        if tb and isinstance(tb, dict):
            tx = int(tb.get("x", 0))
            ty = int(tb.get("y", 0))
            tw = int(tb.get("w", tb.get("width", 1)))
            th = int(tb.get("h", tb.get("height", 1)))
            tx, ty, tw, th = clip_box(tx, ty, tw, th, w, h)
            txt_color = hex_to_bgr("#22c55e")
            cv2.rectangle(vis, (tx, ty), (tx + tw, ty + th), tuple(int(v) for v in txt_color.tolist()), 2)

    # Draw overlay candidate boxes (orange)
    for ov in (req.overlays or []):
        bb = ov.bbox or {}
        x = int(bb.get("x", 0))
        y = int(bb.get("y", 0))
        ww = int(bb.get("w", bb.get("width", 1)))
        hh = int(bb.get("h", bb.get("height", 1)))
        x, y, ww, hh = clip_box(x, y, ww, hh, w, h)
        color = hex_to_bgr(ov.color or "#f97316")
        cv2.rectangle(vis, (x, y), (x + ww, y + hh), tuple(int(v) for v in color.tolist()), 2)
        draw_label(str(ov.label or ov.id), x, max(0, y - 2), color)

    ok, png = cv2.imencode(".png", vis)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode PNG")
    data_url = "data:image/png;base64," + base64.b64encode(png.tobytes()).decode("ascii")

    return {
        "annotated": data_url,
        "meta": {"imageWidth": w, "imageHeight": h, "counts": {"nodes": len(req.nodes or []), "overlays": len(req.overlays or [])}},
    }


@app.post("/debug/contact_sheet")
def debug_contact_sheet(req: ContactSheetRequest) -> Dict[str, Any]:
    try:
        img = decode_data_url(req.image.dataUrl)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    h, w = img.shape[:2]
    opts = req.options or ContactSheetOptions()
    tile = int(max(96, min(256, int(opts.tileSize or 152))))
    cols = int(max(2, min(10, int(opts.cols or 6))))
    max_items = int(max(1, min(80, int(opts.maxItems or 48))))
    pad = int(max(0, min(12, int(opts.padPx or 3))))

    items = list(req.items or [])[:max_items]
    if not items:
        return {"image": "data:image/png;base64,", "meta": {"count": 0, "cols": cols, "rows": 0, "tileSize": tile}}

    def _clip(bb: Dict[str, Any]) -> Tuple[int, int, int, int]:
        x = int(bb.get("x", 0))
        y = int(bb.get("y", 0))
        ww = int(bb.get("w", bb.get("width", 1)))
        hh = int(bb.get("h", bb.get("height", 1)))
        return clip_box(x, y, ww, hh, w, h)

    def _make_tile(preview: Optional[str], bb: Optional[Dict[str, Any]], label: str) -> np.ndarray:
        thumb = None
        if preview and isinstance(preview, str) and preview.startswith("data:image/"):
            try:
                thumb = decode_data_url(preview)
            except Exception:
                thumb = None
        if thumb is None and bb and isinstance(bb, dict):
            x0, y0, ww, hh = _clip(bb)
            x1 = max(0, x0 - pad)
            y1 = max(0, y0 - pad)
            x2 = min(w, x0 + ww + pad)
            y2 = min(h, y0 + hh + pad)
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                thumb = crop

        if thumb is None or thumb.size == 0:
            thumb = np.full((tile, tile, 3), 255, dtype=np.uint8)

        # If RGBA, composite on white for readability.
        if thumb.ndim == 3 and thumb.shape[2] == 4:
            bgr = thumb[:, :, :3].astype(np.float32)
            a = (thumb[:, :, 3:4].astype(np.float32) / 255.0)
            bg = np.full_like(bgr, 255.0)
            thumb = (bgr * a + bg * (1.0 - a)).astype(np.uint8)

        th, tw = thumb.shape[:2]
        if th <= 0 or tw <= 0:
            thumb = np.full((tile, tile, 3), 255, dtype=np.uint8)
            th, tw = tile, tile
        scale = min(float(tile) / float(max(1, tw)), float(tile) / float(max(1, th)))
        nw = max(1, int(round(tw * scale)))
        nh = max(1, int(round(th * scale)))
        resized = cv2.resize(thumb, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
        out = np.full((tile, tile, 3), 255, dtype=np.uint8)
        ox = int((tile - nw) / 2)
        oy = int((tile - nh) / 2)
        out[oy : oy + nh, ox : ox + nw] = resized

        cv2.rectangle(out, (0, 0), (tile - 1, tile - 1), (249, 115, 22), 2)
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale2 = 0.48
            thick = 1
            (_tw2, th2), base = cv2.getTextSize(label, font, scale2, thick)
            y0 = int(min(tile - 1, th2 + base + 8))
            cv2.rectangle(out, (0, 0), (tile - 1, y0), (249, 115, 22), -1)
            cv2.putText(out, label[:18], (4, y0 - base - 3), font, scale2, (255, 255, 255), thick, cv2.LINE_AA)
        return out

    tiles: List[np.ndarray] = []
    for it in items:
        label = str(it.label or it.id or "").strip()
        tiles.append(_make_tile(it.previewDataUrl, it.bbox, label))

    rows = int(np.ceil(float(len(tiles)) / float(cols)))
    sheet_w = cols * tile
    sheet_h = rows * tile
    sheet = np.full((sheet_h, sheet_w, 3), 255, dtype=np.uint8)
    for idx, timg in enumerate(tiles):
        r = int(idx // cols)
        c = int(idx % cols)
        y0 = r * tile
        x0 = c * tile
        sheet[y0 : y0 + tile, x0 : x0 + tile] = timg

    ok, png = cv2.imencode(".png", sheet)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode PNG")
    data_url = "data:image/png;base64," + base64.b64encode(png.tobytes()).decode("ascii")
    return {"image": data_url, "meta": {"count": len(tiles), "cols": cols, "rows": rows, "tileSize": tile}}


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("VISION_HOST", "127.0.0.1")
    port = int(os.environ.get("VISION_PORT", "7777"))
    uvicorn.run(app, host=host, port=port)
