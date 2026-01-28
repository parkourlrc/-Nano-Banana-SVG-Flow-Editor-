const express = require('express');
const fs = require('fs');
const net = require('net');
const os = require('os');
const path = require('path');
const { spawn } = require('child_process');
const { ensureShapeCatalog, normalizeShapeId } = require('./shapes/catalog');

const app = express();
const PORT = process.env.PORT || 3000;
const PROVIDER_SAMPLE_PATH = path.join(__dirname, 'config', 'providers.sample.json');
const DEFAULT_USER_CONFIG_DIR = path.join(os.homedir(), '.research-diagram-studio');
const USER_CONFIG_DIR = process.env.RDS_CONFIG_DIR ? path.resolve(String(process.env.RDS_CONFIG_DIR)) : DEFAULT_USER_CONFIG_DIR;
const PROVIDER_PATH = process.env.RDS_PROVIDERS_PATH
  ? path.resolve(String(process.env.RDS_PROVIDERS_PATH))
  : path.join(USER_CONFIG_DIR, 'providers.json');

app.use(express.json({ limit: '15mb' }));
app.use(express.static(path.join(__dirname, '..', 'web')));

function nowIso() {
  return new Date().toISOString();
}

function safeJson(value, maxLen = 2000) {
  try {
    const text = JSON.stringify(value);
    if (!text) return '';
    return text.length > maxLen ? `${text.slice(0, maxLen)}...` : text;
  } catch (err) {
    return '';
  }
}

function logError(title, err, context = {}) {
  const message = err && err.message ? String(err.message) : String(err || '');
  const stack = err && err.stack ? String(err.stack) : '';
  console.error(`\n[${nowIso()}] ${title}`);
  if (message) console.error(`message: ${message}`);
  const ctxText = safeJson(context, 4000);
  if (ctxText) console.error(`context: ${ctxText}`);
  if (stack) console.error(stack);

  // Print nested causes (undici/network errors often stash useful info in err.cause).
  try {
    const seen = new Set();
    let cur = err && typeof err === 'object' ? err : null;
    for (let depth = 0; depth < 4; depth += 1) {
      const cause = cur && typeof cur === 'object' ? cur.cause : null;
      if (!cause || typeof cause !== 'object') break;
      if (seen.has(cause)) break;
      seen.add(cause);
      const cMsg = cause && cause.message ? String(cause.message) : String(cause || '');
      const cName = cause && cause.name ? String(cause.name) : 'Error';
      console.error(`cause[${depth + 1}]: ${cName}${cMsg ? `: ${cMsg}` : ''}`);
      if (cause && cause.stack) console.error(String(cause.stack));
      cur = cause;
    }
  } catch (causeErr) {
    // ignore
  }
}

function logApiResponseError(req, status, message, context = {}) {
  const safeMessage = String(message || '').slice(0, 1000);
  logError(`API ${req.method} ${req.originalUrl} -> ${status}`, new Error(safeMessage), {
    ip: req.ip,
    ...context
  });
}

process.on('unhandledRejection', (reason) => {
  logError('unhandledRejection', reason);
});

process.on('uncaughtException', (err) => {
  logError('uncaughtException', err);
});

const providerBaseUrls = {
  openai: 'https://api.openai.com/v1',
  'openai-compatible': 'https://0-0.pro/v1',
  openrouter: 'https://openrouter.ai/api/v1',
  groq: 'https://api.groq.com/openai/v1',
  deepseek: 'https://api.deepseek.com/v1',
  anthropic: 'https://api.anthropic.com/v1',
  gemini: 'https://generativelanguage.googleapis.com/v1beta',
  ollama: 'http://localhost:11434/api',
  custom: ''
};

const providerCatalog = [
  { type: 'openai-compatible', label: 'OpenAI-compatible', requiresBase: true, requiresApiKey: true, defaultModel: '', defaultPlannerModel: 'gemini-3-pro', defaultBase: providerBaseUrls['openai-compatible'] },
  { type: 'openai', label: 'OpenAI', requiresBase: false, requiresApiKey: true, defaultModel: 'gpt-4o-mini', defaultPlannerModel: 'gpt-4o-mini', defaultBase: providerBaseUrls.openai },
  { type: 'gemini', label: 'Gemini', requiresBase: false, requiresApiKey: true, defaultModel: 'gemini-2.5-flash', defaultPlannerModel: 'gemini-3-pro', defaultBase: '' },
  { type: 'anthropic', label: 'Anthropic', requiresBase: false, requiresApiKey: true, defaultModel: 'claude-3-5-sonnet-20241022', defaultPlannerModel: 'claude-3-5-sonnet-20241022', defaultBase: '' },
  { type: 'openrouter', label: 'OpenRouter', requiresBase: true, requiresApiKey: true, defaultModel: 'openai/gpt-4o-mini', defaultPlannerModel: 'openai/gpt-4o-mini', defaultBase: providerBaseUrls.openrouter },
  { type: 'groq', label: 'Groq', requiresBase: true, requiresApiKey: true, defaultModel: 'llama-3.1-70b-versatile', defaultPlannerModel: 'llama-3.1-70b-versatile', defaultBase: providerBaseUrls.groq },
  { type: 'deepseek', label: 'DeepSeek', requiresBase: true, requiresApiKey: true, defaultModel: 'deepseek-chat', defaultPlannerModel: 'deepseek-chat', defaultBase: providerBaseUrls.deepseek },
  { type: 'ollama', label: 'Ollama (Local)', requiresBase: true, requiresApiKey: false, defaultModel: 'llama3.1', defaultPlannerModel: 'llama3.1', defaultBase: 'http://localhost:11434/v1' },
  { type: 'custom', label: 'Custom', requiresBase: true, requiresApiKey: true, defaultModel: '', defaultPlannerModel: '', defaultBase: '' }
];

const providerCatalogByType = providerCatalog.reduce((acc, item) => {
  acc[item.type] = item;
  return acc;
}, {});

const SHAPE_CONFIDENCE_THRESHOLD = 0.6;
const STRUCTURE_MAX_ATTEMPTS = 3;
const VISION_HOST = '127.0.0.1';
const VISION_PORT_CANDIDATES = [7777, 7778, 7779];
const VISION_SERVICE_URL_ENV = process.env.VISION_SERVICE_URL ? String(process.env.VISION_SERVICE_URL).replace(/\/$/, '') : '';
const VISION_AUTOSTART = String(process.env.VISION_SERVICE_AUTOSTART || '1') !== '0';
let visionServiceUrl = VISION_SERVICE_URL_ENV || '';
let visionServiceProc = null;
let visionServiceProcPort = null;
let visionServiceReadyPromise = null;

const VISION_PROBE_PNG =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+X2uQAAAAASUVORK5CYII=';

function visionUrlForPort(port) {
  return `http://${VISION_HOST}:${port}`;
}

async function checkVisionService(url) {
  const status = await checkVisionServiceStatus(url);
  return Boolean(status && status.ok);
}

async function checkVisionServiceStatus(url) {
  const base = String(url || '').replace(/\/$/, '');
  if (!base) return { ok: false, incompatible: false, reason: 'missing_url' };
  try {
    const res = await fetchWithTimeout(`${base}/health`, { method: 'GET' }, 4000);
    if (!res.ok) return { ok: false, incompatible: false, reason: `health_http_${res.status}` };
    const json = await res.json().catch(() => null);
    const ok = Boolean(json && (json.ok === 'true' || json.ok === true));
    if (!ok) return { ok: false, incompatible: true, reason: 'health_not_ok', json };

    // Require SAM2 to be installed/configurable (checkpoint may be auto-downloaded later).
    const sam2Ok = Boolean(
      json &&
        (json.sam2ConfigOk === true ||
          json.sam2ConfigOk === 'true' ||
          json.sam2 === true ||
          json.sam2 === 'true')
    );
    if (!sam2Ok) return { ok: false, incompatible: true, reason: 'sam2_missing', json };

    // Require the v2 overlay endpoint; older vision services may return ok but lack overlay extraction.
    const probe = {
      image: { name: 'probe.png', type: 'image/png', dataUrl: VISION_PROBE_PNG },
      imageWidth: 1,
      imageHeight: 1,
      overlays: [],
      debug: false
    };
    const probeRes = await fetchWithTimeout(
      `${base}/overlays/resolve`,
      { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(probe) },
      8000
    );

    // We only need to confirm the endpoint exists (older services return 404).
    // Some environments may fail to decode the probe image and return 400/422, which is still acceptable here.
    if (Number(probeRes.status) === 404) return { ok: false, incompatible: true, reason: 'missing_overlays_resolve', json };
    return { ok: true, incompatible: false, reason: '', json };
  } catch (err) {
    return { ok: false, incompatible: false, reason: 'exception', error: err };
  }
}

function stopVisionServiceProc(logger, reason) {
  if (!visionServiceProc) return;
  try {
    if (!visionServiceProc.killed) visionServiceProc.kill();
  } catch (err) {
    // ignore
  }
  visionServiceProc = null;
  visionServiceProcPort = null;
  if (logger) logger(`Vision service stopped${reason ? ` (${reason})` : ''}.`);
}

async function isPortFree(port) {
  return await new Promise((resolve) => {
    const server = net.createServer();
    server.once('error', () => resolve(false));
    server.once('listening', () => {
      server.close(() => resolve(true));
    });
    server.listen(port, VISION_HOST);
  });
}

function pickPythonExecutable() {
  const candidates = [
    path.join(__dirname, '..', 'vision_service', '.venv', 'Scripts', 'python.exe'),
    path.join(__dirname, '..', 'vision_service', 'venv', 'Scripts', 'python.exe'),
    'python'
  ];
  for (const p of candidates) {
    if (p === 'python') return p;
    try {
      if (fs.existsSync(p)) return p;
    } catch (err) {
      // ignore
    }
  }
  return 'python';
}

function startVisionServiceIfPossible(logger) {
  if (!VISION_AUTOSTART) return false;
  if (visionServiceProc) return true;
  const appPath = path.join(__dirname, '..', 'vision_service', 'app.py');
  if (!fs.existsSync(appPath)) return false;

  const python = pickPythonExecutable();
  const chosenPort = Number.isFinite(Number(visionServiceProcPort)) ? Number(visionServiceProcPort) : VISION_PORT_CANDIDATES[0];
  const args = ['-m', 'uvicorn', 'vision_service.app:app', '--host', VISION_HOST, '--port', String(chosenPort)];
  if (logger) logger(`Starting local vision service: ${python} ${args.join(' ')}`);

  try {
    visionServiceProc = spawn(python, args, {
      cwd: path.join(__dirname, '..'),
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
      stdio: ['ignore', 'pipe', 'pipe'],
      windowsHide: true
    });
    visionServiceProcPort = chosenPort;

    const prefix = `[${nowIso()}] [vision_service]`;
    visionServiceProc.stdout.on('data', (chunk) => {
      try {
        const text = String(chunk || '').trimEnd();
        if (text) console.log(`${prefix} ${text}`);
      } catch (err) {
        // ignore
      }
    });
    visionServiceProc.stderr.on('data', (chunk) => {
      try {
        const text = String(chunk || '').trimEnd();
        if (text) console.error(`${prefix} ${text}`);
      } catch (err) {
        // ignore
      }
    });
    visionServiceProc.on('exit', (code) => {
      const msg = `vision_service exited (code=${code ?? 'null'})`;
      if (logger) logger(msg);
      else console.warn(`[${nowIso()}] ${msg}`);
      visionServiceProc = null;
      visionServiceProcPort = null;
      visionServiceUrl = VISION_SERVICE_URL_ENV || '';
    });

    process.on('exit', () => {
      try {
        if (visionServiceProc && !visionServiceProc.killed) visionServiceProc.kill();
      } catch (err) {
        // ignore
      }
    });

    return true;
  } catch (err) {
    visionServiceProc = null;
    visionServiceProcPort = null;
    if (logger) logger(`Failed to start vision service: ${err?.message || err}`);
    return false;
  }
}

async function ensureVisionServiceUrl(logger) {
  if (visionServiceUrl) {
    const status = await checkVisionServiceStatus(visionServiceUrl);
    if (status.ok) return visionServiceUrl;
    visionServiceUrl = '';
    if (status.incompatible) stopVisionServiceProc(logger, status.reason || 'incompatible');
  }
  if (visionServiceReadyPromise) return await visionServiceReadyPromise;

  visionServiceReadyPromise = (async () => {
    if (VISION_SERVICE_URL_ENV) {
      const ok = await checkVisionService(VISION_SERVICE_URL_ENV);
      if (ok) {
        visionServiceUrl = VISION_SERVICE_URL_ENV;
        return visionServiceUrl;
      }
      if (logger) logger(`VISION_SERVICE_URL set but health check failed: ${VISION_SERVICE_URL_ENV}`);
      return '';
    }

    for (const port of VISION_PORT_CANDIDATES) {
      // eslint-disable-next-line no-await-in-loop
      const ok = await checkVisionService(visionUrlForPort(port));
      if (ok) {
        visionServiceUrl = visionUrlForPort(port);
        return visionServiceUrl;
      }
    }

    if (!VISION_AUTOSTART) return '';

    for (const port of VISION_PORT_CANDIDATES) {
      // eslint-disable-next-line no-await-in-loop
      const free = await isPortFree(port);
      if (!free) continue;

      visionServiceProcPort = port;
      const started = startVisionServiceIfPossible(logger);
      if (!started) continue;

      const url = visionUrlForPort(port);
      const deadline = Date.now() + 30_000;
      while (Date.now() < deadline) {
        // eslint-disable-next-line no-await-in-loop
        const ok = await checkVisionService(url);
        if (ok) {
          visionServiceUrl = url;
          return visionServiceUrl;
        }
        // eslint-disable-next-line no-await-in-loop
        await new Promise((r) => setTimeout(r, 450));
      }

      try {
        if (visionServiceProc && !visionServiceProc.killed) visionServiceProc.kill();
      } catch (err) {
        // ignore
      }
      visionServiceProc = null;
      visionServiceProcPort = null;
    }

    if (logger) logger('Vision service not available (no compatible instance found).');
    return '';
  })().finally(() => {
    visionServiceReadyPromise = null;
  });

  return await visionServiceReadyPromise;
}

function cleanModelText(text) {
  if (!text) return '';
  const trimmed = String(text).trim();
  if (trimmed.startsWith('```')) {
    const lines = trimmed.split('\n');
    lines.shift();
    if (lines.length > 0 && lines[lines.length - 1].startsWith('```')) {
      lines.pop();
    }
    return lines.join('\n').trim();
  }
  return trimmed;
}

function parseJsonFromModel(text) {
  const cleaned = cleanModelText(text);
  if (!cleaned) return null;
  try {
    return JSON.parse(cleaned);
  } catch (err) {
    // fall through
  }

  const firstArray = cleaned.indexOf('[');
  const lastArray = cleaned.lastIndexOf(']');
  if (firstArray !== -1 && lastArray !== -1 && lastArray > firstArray) {
    try {
      return JSON.parse(cleaned.slice(firstArray, lastArray + 1));
    } catch (err) {
      // ignore
    }
  }

  const firstObj = cleaned.indexOf('{');
  const lastObj = cleaned.lastIndexOf('}');
  if (firstObj !== -1 && lastObj !== -1 && lastObj > firstObj) {
    try {
      return JSON.parse(cleaned.slice(firstObj, lastObj + 1));
    } catch (err) {
      // ignore
    }
  }

  return null;
}

function clampBox(box, maxW, maxH) {
  const x = Math.max(0, Math.min(Number(box.x) || 0, maxW));
  const y = Math.max(0, Math.min(Number(box.y) || 0, maxH));
  const w = Math.max(1, Math.min(Number(box.w ?? box.width ?? 1) || 1, maxW - x));
  const h = Math.max(1, Math.min(Number(box.h ?? box.height ?? 1) || 1, maxH - y));
  return { x, y, w, h };
}

function clampBoxWithin(box, container) {
  if (!box || typeof box !== 'object' || !container || typeof container !== 'object') return box;
  const cx = Number(container.x);
  const cy = Number(container.y);
  const cw = Number(container.w ?? container.width);
  const ch = Number(container.h ?? container.height);
  if (![cx, cy, cw, ch].every((v) => Number.isFinite(v))) return box;
  const x1 = cx;
  const y1 = cy;
  const x2 = cx + cw;
  const y2 = cy + ch;

  const bx = Number(box.x);
  const by = Number(box.y);
  const x = Math.max(x1, Math.min(Number.isFinite(bx) ? bx : x1, x2 - 1));
  const y = Math.max(y1, Math.min(Number.isFinite(by) ? by : y1, y2 - 1));
  const maxW = Math.max(1, x2 - x);
  const maxH = Math.max(1, y2 - y);
  const w = Math.max(1, Math.min(Number(box.w ?? box.width ?? 1) || 1, maxW));
  const h = Math.max(1, Math.min(Number(box.h ?? box.height ?? 1) || 1, maxH));
  return { x, y, w, h };
}

function boxIou(a, b) {
  if (!a || !b) return 0;
  const ax1 = Number(a.x) || 0;
  const ay1 = Number(a.y) || 0;
  const aw = Number(a.w ?? a.width) || 0;
  const ah = Number(a.h ?? a.height) || 0;
  const bx1 = Number(b.x) || 0;
  const by1 = Number(b.y) || 0;
  const bw = Number(b.w ?? b.width) || 0;
  const bh = Number(b.h ?? b.height) || 0;
  const ax2 = ax1 + aw;
  const ay2 = ay1 + ah;
  const bx2 = bx1 + bw;
  const by2 = by1 + bh;

  const x1 = Math.max(ax1, bx1);
  const y1 = Math.max(ay1, by1);
  const x2 = Math.min(ax2, bx2);
  const y2 = Math.min(ay2, by2);
  const iw = Math.max(0, x2 - x1);
  const ih = Math.max(0, y2 - y1);
  const inter = iw * ih;
  if (inter <= 0) return 0;
  const union = Math.max(1, aw * ah + bw * bh - inter);
  return inter / union;
}

function pointInBox(point, box) {
  if (!point || !box) return false;
  const px = Number(point.x);
  const py = Number(point.y);
  if (!Number.isFinite(px) || !Number.isFinite(py)) return false;
  const x = Number(box.x) || 0;
  const y = Number(box.y) || 0;
  const w = Number(box.w ?? box.width) || 0;
  const h = Number(box.h ?? box.height) || 0;
  return px >= x && px <= x + w && py >= y && py <= y + h;
}

function sortTextItemsReadingOrder(items) {
  return (items || []).slice().sort((a, b) => {
    const ab = a?.bbox;
    const bb = b?.bbox;
    const ay = Number(ab?.y ?? ab?.top ?? 0);
    const by = Number(bb?.y ?? bb?.top ?? 0);
    if (ay !== by) return ay - by;
    const ax = Number(ab?.x ?? ab?.left ?? 0);
    const bx = Number(bb?.x ?? bb?.left ?? 0);
    return ax - bx;
  });
}

function unionBoxes(boxes) {
  const list = (boxes || []).filter((b) => b && typeof b === 'object');
  if (!list.length) return null;
  let x1 = Infinity;
  let y1 = Infinity;
  let x2 = -Infinity;
  let y2 = -Infinity;
  list.forEach((b) => {
    const x = Number(b.x) || 0;
    const y = Number(b.y) || 0;
    const w = Number(b.w ?? b.width) || 0;
    const h = Number(b.h ?? b.height) || 0;
    x1 = Math.min(x1, x);
    y1 = Math.min(y1, y);
    x2 = Math.max(x2, x + w);
    y2 = Math.max(y2, y + h);
  });
  if (!Number.isFinite(x1) || !Number.isFinite(y1) || !Number.isFinite(x2) || !Number.isFinite(y2)) return null;
  return { x: x1, y: y1, w: Math.max(1, x2 - x1), h: Math.max(1, y2 - y1) };
}

function applyTextItemsToNodes(structure, textItems, imageWidth, imageHeight) {
  const nodes = Array.isArray(structure?.nodes) ? structure.nodes : [];
  const items = Array.isArray(textItems) ? textItems : [];
  if (!nodes.length || !items.length) return [];

  const byNodeId = new Map();
  const unassigned = [];

  items.forEach((it) => {
    if (!it || typeof it !== 'object') return;
    const text = String(it.text || '').trim();
    if (!text) return;
    if (!it.bbox || typeof it.bbox !== 'object') return;
    const bb = clampBox(it.bbox, imageWidth, imageHeight);
    const cx = bb.x + bb.w / 2;
    const cy = bb.y + bb.h / 2;

    let best = null;
    let bestArea = Infinity;
    for (const n of nodes) {
      if (!n || typeof n !== 'object' || !n.bbox) continue;
      if (String(n.render || '').toLowerCase() === 'text') continue;
      if (!pointInBox({ x: cx, y: cy }, n.bbox)) continue;
      const area = Number(n.bbox.w ?? n.bbox.width) * Number(n.bbox.h ?? n.bbox.height);
      if (!Number.isFinite(area)) continue;
      if (area < bestArea) {
        best = n;
        bestArea = area;
      }
    }

    if (!best) {
      unassigned.push({ ...it, bbox: bb });
      return;
    }

    const arr = byNodeId.get(String(best.id)) || [];
    arr.push({ ...it, bbox: bb });
    byNodeId.set(String(best.id), arr);
  });

  nodes.forEach((n) => {
    if (!n || typeof n !== 'object' || !n.bbox) return;
    if (String(n.render || '').toLowerCase() === 'text') return;
    const assigned = byNodeId.get(String(n.id)) || [];
    if (!assigned.length) return;
    const ordered = sortTextItemsReadingOrder(assigned);
    const joined = ordered.map((it) => String(it.text || '').trim()).filter(Boolean).join('\n');
    if (joined) n.text = joined;
    const union = unionBoxes(ordered.map((it) => it.bbox));
    if (union) {
      n.textBbox = clampBoxWithin(clampBox(union, imageWidth, imageHeight), n.bbox);
    }
  });

  return unassigned;
}

function appendTextNodes(structure, textItems, imageWidth, imageHeight) {
  const nodes = Array.isArray(structure?.nodes) ? structure.nodes : [];
  const items = Array.isArray(textItems) ? textItems : [];
  if (!nodes || !items.length) return;
  const idSet = new Set(nodes.map((n) => String(n?.id || '')));

  function uniqueId(prefix) {
    const base = String(prefix || 'txt');
    let i = 1;
    while (idSet.has(`${base}${i}`) || `${base}${i}` === '0' || `${base}${i}` === '1') i += 1;
    const id = `${base}${i}`;
    idSet.add(id);
    return id;
  }

  const existingTextNodes = nodes.filter(
    (n) => n && typeof n === 'object' && String(n.render || '').toLowerCase() === 'text' && n.bbox && typeof n.bbox === 'object'
  );

  items.forEach((it) => {
    if (!it || typeof it !== 'object') return;
    const text = String(it.text || '').trim();
    if (!text) return;
    const bbRaw = it.bbox && typeof it.bbox === 'object' ? it.bbox : null;
    if (!bbRaw) return;
    const bb = clampBox(bbRaw, imageWidth, imageHeight);
    if (!(bb.w >= 6 && bb.h >= 6)) return;

    const dup = existingTextNodes.some((n) => boxIou(n.bbox, bb) >= 0.75 && String(n.text || '').trim() === text);
    if (dup) return;

    const id = uniqueId('txt');
    const confRaw = Number(it.confidence);
    const conf = Number.isFinite(confRaw) ? Math.max(0, Math.min(1, confRaw)) : 0.9;
    const node = {
      id,
      bbox: bb,
      render: 'text',
      shapeId: 'label',
      text,
      textBbox: null,
      confidence: { bbox: 0.9, text: conf, shape: 1.0 },
      nodeOverlays: [],
      overlay: { kind: 'icon', granularity: 'ignore', fgPoints: [], bgPoints: [], confidence: 0.0 },
      containerStyle: null,
      innerShapes: []
    };
    nodes.push(node);
    existingTextNodes.push(node);
  });
}

function boxIntersectionArea(a, b) {
  if (!a || !b) return 0;
  const ax1 = Number(a.x) || 0;
  const ay1 = Number(a.y) || 0;
  const aw = Number(a.w ?? a.width) || 0;
  const ah = Number(a.h ?? a.height) || 0;
  const bx1 = Number(b.x) || 0;
  const by1 = Number(b.y) || 0;
  const bw = Number(b.w ?? b.width) || 0;
  const bh = Number(b.h ?? b.height) || 0;
  const ax2 = ax1 + aw;
  const ay2 = ay1 + ah;
  const bx2 = bx1 + bw;
  const by2 = by1 + bh;
  const x1 = Math.max(ax1, bx1);
  const y1 = Math.max(ay1, by1);
  const x2 = Math.min(ax2, bx2);
  const y2 = Math.min(ay2, by2);
  const iw = Math.max(0, x2 - x1);
  const ih = Math.max(0, y2 - y1);
  return iw * ih;
}

function clampPoint(point, maxW, maxH) {
  const x = Math.max(0, Math.min(Number(point?.x) || 0, maxW));
  const y = Math.max(0, Math.min(Number(point?.y) || 0, maxH));
  return { x, y };
}

function normalizePoints(points, maxW, maxH, maxCount) {
  const list = Array.isArray(points) ? points : [];
  const out = [];
  for (const p of list) {
    if (!p || typeof p !== 'object') continue;
    const x = Number(p.x);
    const y = Number(p.y);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    out.push(clampPoint({ x, y }, maxW, maxH));
    if (maxCount && out.length >= maxCount) break;
  }
  return out;
}

function autoPointsForBox(bbox) {
  const x = Number(bbox?.x) || 0;
  const y = Number(bbox?.y) || 0;
  const w = Number(bbox?.w ?? bbox?.width) || 0;
  const h = Number(bbox?.h ?? bbox?.height) || 0;
  if (!(w > 1 && h > 1)) return { fg: [], bg: [] };

  const cx = x + w / 2;
  const cy = y + h / 2;
  const inset = Math.max(2, Math.min(w, h) * 0.08);

  const fg = [
    { x: cx, y: cy },
    { x: x + w * 0.25, y: y + h * 0.25 },
    { x: x + w * 0.75, y: y + h * 0.25 },
    { x: x + w * 0.25, y: y + h * 0.75 },
    { x: x + w * 0.75, y: y + h * 0.75 }
  ];

  const bg = [
    { x: x + inset, y: y + inset },
    { x: x + w - inset, y: y + inset },
    { x: x + inset, y: y + h - inset },
    { x: x + w - inset, y: y + h - inset },
    { x: cx, y: y + inset },
    { x: cx, y: y + h - inset },
    { x: x + inset, y: cy },
    { x: x + w - inset, y: cy }
  ];

  return { fg, bg };
}

function ensureOverlaySeedPoints(fgPoints, bgPoints, bbox, imageWidth, imageHeight) {
  let fg = normalizePoints(fgPoints, imageWidth, imageHeight, 12);
  let bg = normalizePoints(bgPoints, imageWidth, imageHeight, 16);
  if (fg.length >= 2 && bg.length >= 2) {
    return { fg, bg };
  }
  const auto = autoPointsForBox(bbox);
  fg = [...fg, ...normalizePoints(auto.fg, imageWidth, imageHeight, 12)].slice(0, 10);
  bg = [...bg, ...normalizePoints(auto.bg, imageWidth, imageHeight, 16)].slice(0, 12);
  return { fg, bg };
}

function normalizeOverlayResolveOptions(value) {
  if (!value || typeof value !== 'object') return null;
  const out = {};
  if (typeof value.tightenBbox === 'boolean') out.tightenBbox = Boolean(value.tightenBbox);
  const pad = Number(value.padPx);
  if (Number.isFinite(pad)) out.padPx = Math.max(0, Math.min(96, Math.round(pad)));
  return Object.keys(out).length ? out : null;
}

function normalizeOverlayKind(value) {
  const kind = String(value || '').toLowerCase();
  const allowed = new Set(['icon', 'photo', 'chart', 'plot', '3d', 'noise', 'screenshot', 'node']);
  if (allowed.has(kind)) return kind;
  return kind ? 'icon' : 'icon';
}

function defaultGranularityForKind(kind) {
  const k = String(kind || '').toLowerCase();
  if (k === 'icon' || k === '3d') return 'alphaMask';
  if (k === 'photo' || k === 'chart' || k === 'plot' || k === 'noise' || k === 'screenshot') return 'opaqueRect';
  return 'opaqueRect';
}

function normalizeGranularity(value) {
  const g = String(value || '').trim();
  if (g === 'alphaMask' || g === 'opaqueRect' || g === 'ignore') return g;
  return 'opaqueRect';
}

function normalizeSide(value) {
  const v = String(value || '').toLowerCase();
  if (v === 'left' || v === 'right' || v === 'top' || v === 'bottom') return v;
  return 'right';
}

function buildStructureInstruction(imageWidth, imageHeight, shapeConfidenceThreshold) {
  const w = Number(imageWidth);
  const h = Number(imageHeight);
  const sizeHint = Number.isFinite(w) && Number.isFinite(h) && w > 0 && h > 0 ? `Reference image size: ${w}x${h} pixels.` : '';
  return [
    'You are extracting a diagram specification from a PAPER FIGURE SCREENSHOT for diagrams.net (draw.io).',
    'Return ONLY a single JSON object with keys: nodes, edges, overlays.',
    'All coordinates are in reference-image pixel space (origin top-left).',
    sizeHint,
    '',
    'nodes: array of {',
    '  id: string,',
    '  bbox:{x,y,w,h},',
    '  render:\"shape\"|\"overlay\",',
    '  shapeId?: string,',
    '  text: string,',
    '  textBbox?:{x,y,w,h},',
    '  confidence:{bbox:number,text:number,shape:number},',
    '  nodeOverlays?: array of {',
    '    id: string,',
    '    kind:\"icon\"|\"photo\"|\"chart\"|\"plot\"|\"3d\"|\"noise\"|\"screenshot\",',
    '    granularity:\"alphaMask\"|\"opaqueRect\"|\"ignore\",',
    '    bbox:{x,y,w,h},',
    '    fgPoints?: array of {x:number,y:number},',
    '    bgPoints?: array of {x:number,y:number},',
    '    confidence:number',
    '  },',
    '  overlay?: {',
    '    kind:\"icon\"|\"photo\"|\"chart\"|\"plot\"|\"3d\"|\"noise\"|\"screenshot\",',
    '    granularity:\"alphaMask\"|\"opaqueRect\"|\"ignore\",',
    '    fgPoints?: array of {x:number,y:number},',
    '    bgPoints?: array of {x:number,y:number},',
    '    confidence:number',
    '  }',
    '}.',
    'edges: array of { id, source, target, sourceSide:\"left\"|\"right\"|\"top\"|\"bottom\", targetSide:\"left\"|\"right\"|\"top\"|\"bottom\", label?:string, confidence:number }.',
    'overlays: array of { id, kind:\"icon\"|\"photo\"|\"chart\"|\"plot\"|\"3d\"|\"noise\"|\"screenshot\", granularity:\"alphaMask\"|\"opaqueRect\"|\"ignore\", bbox:{x,y,w,h}, fgPoints?:array, bgPoints?:array, confidence:number }.',
    '',
    'Rules:',
    '- ids must be unique strings (use n1,n2,... and e1,e2,...).',
    '- bbox MUST tightly cover the node/overlay. Do not include large blank margins or surrounding whitespace.',
    '- If a node has non-empty text, provide textBbox tightly around ONLY the rendered text (avoid including shape fill/lines).',
    '- textBbox must be fully inside bbox.',
    `- Only set render=\"shape\" when confidence.shape >= ${shapeConfidenceThreshold}. Otherwise set render=\"shape\" with shapeId=\"roundRect\" unless the node is an actual image/photo/plot (then render=\"overlay\").`,
    '- shapeId must be a draw.io shape style value: either a built-in shape (e.g. \"ellipse\",\"rhombus\",\"cloud\",\"cylinder\") or a stencil id like \"mxgraph.aws4.ec2\".',
    '- Overlays (nodeOverlays/overlays/overlay nodes) must NOT be text-only regions, arrows/connectors, blank areas, or large decorative background panels.',
    '- You MUST classify each overlay by granularity:',
    '  - alphaMask: icons/3D/robot/snowflake/illustrations that need transparent background (use SAM2). Provide 3-8 fgPoints and 3-8 bgPoints.',
    '  - opaqueRect: real photos, real charts, screenshots, noise blocks/texture blocks. Keep rectangular background. Do NOT provide points.',
    '  - ignore: text blocks, blank boxes, large background blocks, UI panels. Do not include unless requested.',
    '- For alphaMask points: fgPoints are inside the object; bgPoints are inside the bbox but on background pixels. Points are in FULL IMAGE pixel coordinates.',
    '- Do NOT output XML. Do NOT output markdown/code fences. Do NOT include explanations.'
  ]
    .filter(Boolean)
    .join('\\n');
}

function buildTextExtractionInstruction(imageWidth, imageHeight) {
  const w = Number(imageWidth);
  const h = Number(imageHeight);
  const sizeHint = Number.isFinite(w) && Number.isFinite(h) && w > 0 && h > 0 ? `Reference image size: ${w}x${h} pixels.` : '';
  return [
    'You are doing OCR for a paper-figure screenshot (diagram/flowchart).',
    'Return ONLY a JSON array of text items. No markdown.',
    sizeHint,
    '',
    'Schema:',
    '[',
    '  { "text": string, "bbox": { "x": number, "y": number, "w": number, "h": number } }',
    ']',
    '',
    'Rules:',
    '- Include ALL visible text in the diagram (Chinese/English).',
    '- bbox must tightly cover ONLY the glyphs (avoid large blank margins).',
    '- Use absolute pixel coordinates (origin top-left).',
    '- Do not include arrows/connectors as text.',
    '- Sort items by y then x (reading order).'
  ].join('\n');
}

function normalizeTextItemsFromModel(parsed, imageWidth, imageHeight) {
  const w = Number(imageWidth) || 0;
  const h = Number(imageHeight) || 0;
  const list = Array.isArray(parsed) ? parsed : [];
  const out = [];
  for (const item of list) {
    if (!item || typeof item !== 'object') continue;
    const text = String(item.text || '').trim();
    const bb = item.bbox && typeof item.bbox === 'object' ? item.bbox : null;
    if (!bb) continue;
    const x = Math.max(0, Math.min(Number(bb.x) || 0, w));
    const y = Math.max(0, Math.min(Number(bb.y) || 0, h));
    const ww = Math.max(1, Math.min(Number(bb.w ?? bb.width) || 0, w - x));
    const hh = Math.max(1, Math.min(Number(bb.h ?? bb.height) || 0, h - y));
    if (!(ww >= 2 && hh >= 2)) continue;
    if (!text) continue;
    out.push({ text, bbox: { x: Math.round(x), y: Math.round(y), w: Math.round(ww), h: Math.round(hh) } });
    if (out.length >= 260) break;
  }
  // Sort by y then x.
  out.sort((a, b) => (a.bbox.y - b.bbox.y) || (a.bbox.x - b.bbox.x));
  return out;
}

function unwrapTextItemsArray(parsed, depth = 0) {
  if (Array.isArray(parsed)) return parsed;
  if (!parsed || typeof parsed !== 'object') return null;

  const directKeys = ['items', 'textItems', 'text_items', 'texts', 'ocr', 'result', 'results', 'data', 'output'];
  for (const key of directKeys) {
    const value = parsed[key];
    if (Array.isArray(value)) return value;
  }

  // Some models wrap the array deeper, e.g. { textItems: { items: [...] } }.
  for (const key of directKeys) {
    const value = parsed[key];
    if (!value || typeof value !== 'object') continue;
    for (const nestedKey of ['items', 'textItems', 'text_items', 'texts', 'data', 'result', 'results']) {
      const nested = value[nestedKey];
      if (Array.isArray(nested)) return nested;
    }
  }

  // Sometimes the array is returned as a JSON string field.
  if (depth < 2) {
    const stringKeys = ['json', 'text', 'content'];
    for (const key of stringKeys) {
      const value = parsed[key];
      if (typeof value !== 'string') continue;
      const nestedParsed = parseJsonFromModel(value);
      const nestedArr = unwrapTextItemsArray(nestedParsed, depth + 1);
      if (nestedArr) return nestedArr;
    }
  }

  return null;
}

async function extractTextItems(provider, image, imageWidth, imageHeight) {
  const system = buildTextExtractionInstruction(imageWidth, imageHeight);
  let user = 'Return ONLY the JSON array.';
  let out = '';

  for (let attempt = 1; attempt <= 2; attempt += 1) {
    try {
      out = await callVisionJsonWithImages(provider, system, user, [image], { maxTokens: 4096, timeoutMs: 180000 });
    } catch (err) {
      if (err && typeof err === 'object' && !err.phase) err.phase = 'text_extract';
      throw err;
    }

    const parsed = parseJsonFromModel(out);
    const arr = unwrapTextItemsArray(parsed);
    if (arr) {
      return normalizeTextItemsFromModel(arr, imageWidth, imageHeight);
    }

    if (attempt === 1) {
      const snippet = cleanModelText(out).slice(0, 1200);
      user = [
        'Your previous response was invalid.',
        'Return ONLY a JSON array matching the schema exactly. No markdown, no code fences.',
        '',
        'Previous output:',
        snippet || '(empty)',
        '',
        'Return ONLY the corrected JSON array.'
      ].join('\n');
      continue;
    }
  }

  const err = new Error('Text extractor did not return a JSON array.');
  err.phase = 'text_extract';
  err.code = 'text_extract_invalid_json';
  err.modelTextSnippet = cleanModelText(out).slice(0, 1600);
  throw err;
}

function shouldContinueWithoutTextItems(err) {
  if (!err) return false;
  if (isProviderAuthHttpError(err)) return false;
  if (err.name === 'AbortError') return true;

  const status = Number(err?.http?.status);
  if (Number.isFinite(status) && (status === 408 || status === 429 || status === 500 || status === 502 || status === 503 || status === 504)) {
    return true;
  }

  const code = String(err?.code || '').toLowerCase();
  if (code === 'text_extract_invalid_json') return true;

  const msg = String(err?.message || '').toLowerCase();
  if (msg.includes('text extractor did not return')) return true;
  if (msg.includes('gateway timeout')) return true;
  if (msg.includes('timeout')) return true;
  if (msg.includes('temporarily unavailable')) return true;
  return false;
}

async function extractTextItemsBestEffort(provider, image, imageWidth, imageHeight, logContext) {
  try {
    const textItems = await extractTextItems(provider, image, imageWidth, imageHeight);
    return { textItems, textExtractError: null };
  } catch (err) {
    if (!shouldContinueWithoutTextItems(err)) throw err;
    const message = String(err?.message || '').trim() || 'Text extraction failed.';
    logError('Text extraction failed (continuing without text items)', err, {
      ...(logContext && typeof logContext === 'object' ? logContext : {}),
      modelTextSnippet: err?.modelTextSnippet || '',
      providerHttp: err?.http || null
    });
    return { textItems: [], textExtractError: message.slice(0, 900) };
  }
}

function normalizeProviders(raw) {
  const providers = {};
  const data = raw && typeof raw === 'object' ? raw : {};

  if (Array.isArray(data.providers)) {
    data.providers.forEach((item) => {
      const type = item?.type || item?.provider;
      if (type && providerCatalogByType[type]) {
        providers[type] = { ...item, type };
      }
    });
  } else if (data.providers && typeof data.providers === 'object') {
    Object.keys(data.providers).forEach((type) => {
      if (providerCatalogByType[type]) {
        providers[type] = { ...data.providers[type], type };
      }
    });
  }

  providerCatalog.forEach((entry) => {
    const existing = providers[entry.type] || {};
    const merged = { type: entry.type, ...existing };
    const model = String(merged.model || '').trim();
    const baseUrl = String(merged.baseUrl || '').trim();
    const apiKey = String(merged.apiKey || '');
    const imageModel = String(merged.imageModel || '').trim();
    const plannerModel = String(merged.plannerModel || merged.vlmModel || '').trim();

    providers[entry.type] = {
      ...merged,
      type: entry.type,
      model: model || entry.defaultModel || '',
      plannerModel: plannerModel || entry.defaultPlannerModel || '',
      imageModel,
      baseUrl: baseUrl || entry.defaultBase || '',
      apiKey
    };
  });

  let primary = typeof data.primary === 'string' ? data.primary : '';
  if (!providerCatalogByType[primary]) {
    primary = providerCatalog[0]?.type || '';
  }

  return { primary, providers };
}

function readProviderConfig() {
  let data = null;
  try {
    const raw = fs.readFileSync(PROVIDER_PATH, 'utf8');
    data = JSON.parse(raw);
  } catch (err) {
    // fall through
  }
  if (!data) {
    try {
      const raw = fs.readFileSync(PROVIDER_SAMPLE_PATH, 'utf8');
      data = JSON.parse(raw);
    } catch (err) {
      data = {};
    }
  }
  return normalizeProviders(data);
}

function writeProviderConfig(config) {
  const payload = { primary: config.primary, providers: config.providers };
  try {
    fs.mkdirSync(path.dirname(PROVIDER_PATH), { recursive: true });
  } catch (err) {
    // ignore
  }
  fs.writeFileSync(PROVIDER_PATH, JSON.stringify(payload, null, 2));
}

function sanitizeProvider(provider) {
  if (!provider) return null;
  const { apiKey, ...safe } = provider;
  return safe;
}

function resolveProvider(body, config) {
  const type = body?.providerType || body?.providerName || config.primary;
  if (!type) return null;
  return config.providers[type] || null;
}

function normalizeProvider(provider) {
  if (!provider) return null;
  const baseUrl = provider.baseUrl || providerBaseUrls[provider.type] || '';
  return { ...provider, baseUrl };
}

function normalizeProviderForVision(provider) {
  const normalized = normalizeProvider(provider);
  if (!normalized) return null;
  const plannerModel = String(provider?.plannerModel || '').trim();
  if (!plannerModel) return normalized;
  return { ...normalized, model: plannerModel };
}

function isProviderConfigured(provider) {
  if (!provider) return false;
  const meta = providerCatalogByType[provider.type];
  if (!meta) return false;
  if (!provider.model) return false;
  if (meta.requiresBase && !provider.baseUrl) return false;
  if (meta.requiresApiKey && !provider.apiKey) return false;
  return true;
}

function providerMissingFields(provider) {
  if (!provider) return ['provider'];
  const normalized = normalizeProvider(provider);
  const meta = normalized ? providerCatalogByType[normalized.type] : null;
  if (!meta) return ['type'];

  const missing = [];
  if (!normalized.model) missing.push('model');
  if (meta.requiresBase && !normalized.baseUrl) missing.push('baseUrl');
  if (meta.requiresApiKey && !normalized.apiKey) missing.push('apiKey');
  return missing;
}

function respondProviderNotConfigured(req, res, config, provider, providerTypeHint) {
  const providerType = String(providerTypeHint || config?.primary || '').trim();
  const missingFields = providerMissingFields(provider);
  const safeProvider = provider ? { type: provider.type, model: provider.model, baseUrl: provider.baseUrl } : null;
  const message =
    missingFields.length && missingFields[0] !== 'provider' && missingFields[0] !== 'type'
      ? `Provider not configured (missing: ${missingFields.join(', ')}).`
      : 'Provider not configured.';
  logApiResponseError(req, 400, message, { providerType, provider: safeProvider, missingFields });
  res.status(400).json({ error: message, code: 'provider_not_configured', providerType, missingFields });
}

function requireConfiguredProvider(req, res, config, provider) {
  const providerType = req.body?.providerType || req.body?.providerName || config.primary;
  if (!provider || !isProviderConfigured(provider)) {
    respondProviderNotConfigured(req, res, config, provider, providerType);
    return false;
  }
  return true;
}

function isProviderAuthHttpError(err) {
  const status = Number(err?.http?.status);
  if (!(status === 401 || status === 403)) return false;
  const url = String(err?.http?.url || '');
  if (!url) return true;
  return url.includes('/chat/completions') || url.includes('/models/') || url.includes('/messages');
}

function buildClientErrorPayload(err, fallbackMessage) {
  const upstreamStatus = Number(err?.http?.status);
  const phase = err && typeof err === 'object' && err.phase ? String(err.phase) : '';
  let code = err && typeof err === 'object' && err.code ? String(err.code) : '';

  let status = 500;
  if (Number.isFinite(upstreamStatus) && upstreamStatus >= 400 && upstreamStatus < 600) {
    status = upstreamStatus;
  } else if (err && err.name === 'AbortError') {
    status = 504;
    if (!code) code = 'timeout';
  }

  let message = err && err.message ? String(err.message).trim() : '';
  if (!message) message = String(fallbackMessage || 'Error').trim();

  const url = String(err?.http?.url || '');
  const lower = message.toLowerCase();
  let activationUrl = '';
  const m = message.match(/https?:\/\/console\.developers\.google\.com\/apis\/api\/generativelanguage\.googleapis\.com\/overview\?project=[0-9A-Za-z._-]+/);
  if (m) activationUrl = m[0];

  if (
    !code &&
    (lower.includes('service_disabled') ||
      lower.includes('service disabled') ||
      lower.includes('generative language api has not been used') ||
      lower.includes('generativelanguage.googleapis.com'))
  ) {
    code = 'provider_service_disabled';
  }

  if (isProviderAuthHttpError(err)) {
    if (!code) code = 'provider_auth_failed';
    if (code === 'provider_service_disabled') {
      message = `${message} (Enable Generative Language API in your Google project and retry).`;
    } else {
      message = `${message} (Check API key in Settings).`;
    }
  } else if (status === 404 && url.includes('/chat/completions')) {
    if (!code) code = 'provider_endpoint_not_found';
    message = `${message} (Check Base URL; it usually ends with /v1).`;
  } else if (!code && lower.includes('fetch failed')) {
    code = 'network_error';
    const cCode = err?.cause?.code || err?.code || '';
    if (status === 500) status = 502;
    message = `${message}${cCode ? ` (${String(cCode)})` : ''} (Network error reaching provider. Check Base URL / proxy / DNS and retry).`;
  }

  const body = { error: message };
  if (code) body.code = code;
  if (phase) body.phase = phase;
  if (Number.isFinite(upstreamStatus) && upstreamStatus) body.upstreamStatus = upstreamStatus;
  if (activationUrl) body.activationUrl = activationUrl;
  return { status, body };
}

function listProviders(config) {
  return providerCatalog.map((entry) => {
    const provider = config.providers[entry.type] || { type: entry.type };
    return {
      ...sanitizeProvider(provider),
      type: entry.type,
      configured: isProviderConfigured(provider)
    };
  });
}

function buildInstruction(format) {
  if (format === 'xml') {
    return [
      'Return ONLY draw.io mxGraph XML.',
      'XML root must be <mxfile><diagram>...<mxGraphModel><root>... structure.',
      'Include exactly one <mxCell id="0"/> and one <mxCell id="1" parent="0"/>.',
      'All other mxCell ids MUST be unique and MUST NOT be "0" or "1".',
      '<root> must contain ONLY <mxCell> elements as direct children (no <Array>, <Object>, or other tags under <root>).',
      'All vertices: <mxCell vertex="1" parent="1"> with <mxGeometry x y width height as="geometry"/>.',
      'All edges: <mxCell edge="1" parent="1" source="VERTEX_ID" target="VERTEX_ID"> with <mxGeometry relative="1" as="geometry"/>.',
      'Every edge source/target must reference an existing vertex id.',
      'Use simple, modern styling (consistent font sizes, subtle colors) but keep the structure minimal and valid.',
      'Do not wrap in Markdown or code fences.',
      'Keep it valid XML.'
    ].join(' ');
  }

  return [
    'Return ONLY valid React Flow JSON with keys: nodes, edges.',
    'nodes must be a non-empty array; edges must be an array.',
    'Node ids must be unique strings and MUST NOT be "0" or "1". Use ids like "n1","n2"...',
    'Edge ids must be unique strings. Use ids like "e1","e2"...',
    'Each node: { id, type:"flow", position:{x:number,y:number}, data:{label:string,shape?:string}, width:number, height:number }.',
    'Each edge: { id, source, target, label?:string, type:"smoothstep", sourceHandle?:string, targetHandle?:string }.',
    'Every edge source/target must reference an existing node id.',
    'Do not wrap in Markdown or code fences.'
  ].join(' ');
}

function buildOverlayInstruction(imageWidth, imageHeight) {
  const w = Number(imageWidth);
  const h = Number(imageHeight);
  const sizeHint = Number.isFinite(w) && Number.isFinite(h) && w > 0 && h > 0 ? `The image size is ${w}x${h} pixels.` : '';
  return [
    'Analyze the reference image and identify regions that are hard to represent with draw.io shapes, such as:',
    'photorealistic objects, complex icons, charts/screenshots, 3D diagrams, detailed plots, textures, or illustrations.',
    'Return ONLY a JSON array. Each item must be:',
    '{ "label": string, "kind": "icon"|"photo"|"chart"|"plot"|"3d", "x": number, "y": number, "width": number, "height": number, "confidence": number }.',
    sizeHint,
    'Coordinates are in pixels in the provided image coordinate system (origin top-left).',
    'Do NOT include regions that are primarily text (titles, labels, captions), blank/white space, simple rectangles, UI panels, or connector lines.',
    'Prefer boxes that contain a clearly recognizable non-text visual (icon/photo/chart/plot/3D) and would look wrong if redrawn as basic shapes.',
    'Return at most 6 items. Boxes should be tight, avoid background margins, and MUST NOT include the whole image.',
    'Do not include explanations, markdown, or code fences.'
  ].filter(Boolean).join(' ');
}

function buildNodeOverlayPlanInstruction(imageWidth, imageHeight, nodes) {
  const w = Number(imageWidth);
  const h = Number(imageHeight);
  const sizeHint = Number.isFinite(w) && Number.isFinite(h) && w > 0 && h > 0 ? `Reference image size: ${w}x${h} pixels.` : '';
  const list = Array.isArray(nodes) ? nodes : [];
  const compact = list
    .filter((n) => n && typeof n === 'object' && n.id && n.bbox)
    .slice(0, 60)
    .map((n) => ({
      id: String(n.id),
      bbox: n.bbox,
      text: typeof n.text === 'string' ? n.text : '',
      textBbox: n.textBbox && typeof n.textBbox === 'object' ? n.textBbox : null,
      candidates: Array.isArray(n.nodeOverlays)
        ? n.nodeOverlays
            .filter((ov) => ov && typeof ov === 'object' && ov.bbox)
            .slice(0, 10)
            .map((ov) => ({ kind: ov.kind || 'icon', bbox: ov.bbox }))
        : []
    }));

  return [
    'You are extracting bitmap overlay regions from a paper figure screenshot for diagrams.net (draw.io).',
    'Goal: identify ONLY the parts inside each node container that cannot be faithfully represented with draw.io vector shapes.',
    'Examples of overlays: small detailed icons, 3D illustrations, photorealistic objects, screenshots, charts/plots, textures/noise blocks inside nodes.',
    'Do NOT include: arrows/connectors, simple rectangles, simple gradient bars, rounded panels, plain fill regions, or text-only regions.',
    'IMPORTANT: text must remain editable; overlays must NOT include large text areas.',
    'You are given CV-generated candidate overlay boxes for each node (candidates). Use them as hints: refine by shrinking/splitting/dropping false positives.',
    'Do NOT invent huge overlay boxes. Prefer multiple small atomic overlays over one large merged box.',
    '',
    sizeHint,
    '',
    'INPUT NODES (id + bbox + candidates; bbox is authoritative and in reference-image pixel coordinates):',
    JSON.stringify(compact),
    '',
    'Return ONLY a single JSON object with key: nodeOverlaysByNodeId.',
    'nodeOverlaysByNodeId: { [nodeId: string]: Overlay[] }',
    'Overlay: {',
    '  id: string,',
    '  kind: "icon"|"3d"|"photo"|"chart"|"plot"|"noise"|"screenshot",',
    '  granularity: "alphaMask"|"opaqueRect",',
    '  bbox: {x:number,y:number,w:number,h:number},',
    '  fgPoints: [{x:number,y:number}, ...],',
    '  bgPoints: [{x:number,y:number}, ...],',
    '  confidence: number',
    '}',
    '',
    'Rules:',
    '- All coordinates are pixels in the reference image (origin top-left).',
    '- Each overlay bbox MUST be fully INSIDE its parent node bbox.',
    '- If textBbox is provided, overlays must NOT cover it (keep text editable).',
    '- bbox should be TIGHT around the non-text visual (avoid large blank margins).',
    '- fgPoints: at least 2 points INSIDE the overlay foreground (not background).',
    '- bgPoints: at least 2 points INSIDE the parent node bbox but OUTSIDE the overlay bbox, preferably on the node background.',
    '- Use granularity="alphaMask" for kind icon/3d (transparent background).',
    '- Use granularity="opaqueRect" for kind photo/chart/plot/noise/screenshot (keep rectangular background).',
    '- If a node has no overlays, return an empty array for that node id.',
    '- Do not output markdown, code fences, or explanations.'
  ]
    .filter(Boolean)
    .join('\n');
}

function boxArea(box) {
  if (!box || typeof box !== 'object') return 0;
  const w = Number(box.w ?? box.width) || 0;
  const h = Number(box.h ?? box.height) || 0;
  return Math.max(0, w) * Math.max(0, h);
}

function boxAspectRatio(box) {
  if (!box || typeof box !== 'object') return 0;
  const w = Number(box.w ?? box.width) || 0;
  const h = Number(box.h ?? box.height) || 0;
  if (!(w > 0 && h > 0)) return 0;
  return w / h;
}

function boxContains(outer, inner, pad = 2) {
  if (!outer || !inner) return false;
  const ox = Number(outer.x) || 0;
  const oy = Number(outer.y) || 0;
  const ow = Number(outer.w ?? outer.width) || 0;
  const oh = Number(outer.h ?? outer.height) || 0;
  const ix = Number(inner.x) || 0;
  const iy = Number(inner.y) || 0;
  const iw = Number(inner.w ?? inner.width) || 0;
  const ih = Number(inner.h ?? inner.height) || 0;
  if (![ox, oy, ow, oh, ix, iy, iw, ih].every((v) => Number.isFinite(v))) return false;
  return ix >= ox + pad && iy >= oy + pad && ix + iw <= ox + ow - pad && iy + ih <= oy + oh - pad;
}

function textCoverageFraction(textItems, nodes) {
  const items = Array.isArray(textItems) ? textItems : [];
  const list = Array.isArray(nodes) ? nodes : [];
  if (items.length === 0) return 1;
  if (list.length === 0) return 0;

  let covered = 0;
  for (const item of items) {
    const bb = item?.bbox;
    if (!bb || typeof bb !== 'object') continue;
    const w = Number(bb.w ?? bb.width) || 0;
    const h = Number(bb.h ?? bb.height) || 0;
    if (!(w >= 2 && h >= 2)) continue;

    const hit = list.some((n) => {
      const nb = n?.bbox;
      if (!nb || typeof nb !== 'object') return false;
      if (boxContains(nb, bb, 0)) return true;
      const inter = boxIntersectionArea(nb, bb);
      const area = Math.max(1, w * h);
      return inter / area >= 0.7;
    });
    if (hit) covered += 1;
  }
  return covered / Math.max(1, items.length);
}

function assessCvStructure(cvStruct, imageWidth, imageHeight, textItems) {
  const nodes = Array.isArray(cvStruct?.nodes) ? cvStruct.nodes : [];
  const edges = Array.isArray(cvStruct?.edges) ? cvStruct.edges : [];
  const imgArea = Math.max(1, Number(imageWidth || 0) * Number(imageHeight || 0));

  const areas = nodes.map((n) => boxArea(n?.bbox));
  const smallCount = nodes.reduce((acc, n) => {
    const bb = n?.bbox;
    const w = Number(bb?.w ?? bb?.width) || 0;
    const h = Number(bb?.h ?? bb?.height) || 0;
    const a = Math.max(0, w) * Math.max(0, h);
    const tiny = a < imgArea * 0.0009 || w < 14 || h < 14;
    return acc + (tiny ? 1 : 0);
  }, 0);
  const smallFrac = nodes.length ? smallCount / nodes.length : 1;

  // Count nodes that are almost fully contained in a larger node (common when CV picks arrowheads, icons, or text fragments as nodes).
  const sortedIdx = nodes
    .map((n, idx) => ({ idx, area: areas[idx] || 0 }))
    .sort((a, b) => b.area - a.area)
    .map((x) => x.idx);

  let containedCount = 0;
  for (let i = 0; i < sortedIdx.length; i += 1) {
    const ni = nodes[sortedIdx[i]];
    const bi = ni?.bbox;
    if (!bi) continue;
    for (let j = i + 1; j < sortedIdx.length; j += 1) {
      const nj = nodes[sortedIdx[j]];
      const bj = nj?.bbox;
      if (!bj) continue;
      if (boxContains(bi, bj, 3)) {
        containedCount += 1;
        break;
      }
    }
  }
  const containedFrac = nodes.length ? containedCount / nodes.length : 1;

  // Approximate overlap density on the top candidates only (O(n^2) bounded).
  const maxCheck = Math.min(40, nodes.length);
  let overlapPairs = 0;
  let pairTotal = 0;
  for (let i = 0; i < maxCheck; i += 1) {
    for (let j = i + 1; j < maxCheck; j += 1) {
      pairTotal += 1;
      const iou = boxIou(nodes[i]?.bbox, nodes[j]?.bbox);
      if (iou >= 0.45) overlapPairs += 1;
    }
  }
  const overlapRate = pairTotal ? overlapPairs / pairTotal : 0;

  const textFrac = textCoverageFraction(textItems, nodes);

  const ok =
    nodes.length >= 4 &&
    nodes.length <= 90 &&
    !(nodes.length > 18 && smallFrac > 0.55) &&
    !(nodes.length > 12 && containedFrac > 0.55) &&
    !(nodes.length > 12 && overlapRate > 0.22) &&
    !(Array.isArray(textItems) && textItems.length >= 8 && textFrac < 0.35) &&
    !(nodes.length < 12 && edges.length === 0);

  return {
    ok,
    metrics: {
      nodes: nodes.length,
      edges: edges.length,
      smallFrac: Number.isFinite(smallFrac) ? Number(smallFrac.toFixed(3)) : 1,
      containedFrac: Number.isFinite(containedFrac) ? Number(containedFrac.toFixed(3)) : 1,
      overlapRate: Number.isFinite(overlapRate) ? Number(overlapRate.toFixed(3)) : 0,
      textFrac: Number.isFinite(textFrac) ? Number(textFrac.toFixed(3)) : 0
    }
  };
}

function safeOverlayId(value, fallback) {
  const raw = String(value || '').trim() || String(fallback || '').trim() || 'ov';
  return raw.replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 80);
}

function pointInsideBox(p, bbox, pad = 0) {
  if (!p || !bbox) return false;
  const x = Number(p.x);
  const y = Number(p.y);
  const bx = Number(bbox.x) || 0;
  const by = Number(bbox.y) || 0;
  const bw = Number(bbox.w ?? bbox.width) || 0;
  const bh = Number(bbox.h ?? bbox.height) || 0;
  if (![x, y, bx, by, bw, bh].every((v) => Number.isFinite(v))) return false;
  return x >= bx + pad && y >= by + pad && x <= bx + bw - pad && y <= by + bh - pad;
}

function dedupePoints(points, maxCount) {
  const out = [];
  const seen = new Set();
  for (const p of Array.isArray(points) ? points : []) {
    if (!p || typeof p !== 'object') continue;
    const x = Math.round(Number(p.x));
    const y = Math.round(Number(p.y));
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    const k = `${x}:${y}`;
    if (seen.has(k)) continue;
    seen.add(k);
    out.push({ x: Number(p.x), y: Number(p.y) });
    if (maxCount && out.length >= maxCount) break;
  }
  return out;
}

function fillOverlayPlanPointsFromCandidates(finalOverlays, candidatesById, imageWidth, imageHeight) {
  const list = Array.isArray(finalOverlays) ? finalOverlays : [];
  return list.map((ov, idx) => {
    if (!ov || typeof ov !== 'object') return ov;
    const kind = normalizeOverlayKind(ov.kind || 'icon');
    const gran = defaultGranularityForKind(kind);

    const src = Array.isArray(ov.sourceCandidateIds) ? ov.sourceCandidateIds : [];

    let bboxFromCandidates = null;
    if (src.length > 0 && candidatesById instanceof Map) {
      let x1 = Infinity;
      let y1 = Infinity;
      let x2 = -Infinity;
      let y2 = -Infinity;
      src.forEach((id) => {
        const cand = candidatesById.get(String(id));
        if (!cand || !cand.bbox) return;
        const bb = clampBox(cand.bbox, imageWidth, imageHeight);
        const w = Number(bb.w ?? bb.width);
        const h = Number(bb.h ?? bb.height);
        if (!(w > 0 && h > 0)) return;
        x1 = Math.min(x1, Number(bb.x));
        y1 = Math.min(y1, Number(bb.y));
        x2 = Math.max(x2, Number(bb.x) + w);
        y2 = Math.max(y2, Number(bb.y) + h);
      });
      if ([x1, y1, x2, y2].every((v) => Number.isFinite(v)) && x2 > x1 && y2 > y1) {
        bboxFromCandidates = clampBox({ x: x1, y: y1, w: x2 - x1, h: y2 - y1 }, imageWidth, imageHeight);
      }
    }

    const bboxFromPlannerRaw = ov.bbox && typeof ov.bbox === 'object' ? ov.bbox : null;
    const bboxFromPlanner = bboxFromPlannerRaw ? clampBox(bboxFromPlannerRaw, imageWidth, imageHeight) : null;

    let bbox = bboxFromCandidates || bboxFromPlanner;
    if (bboxFromCandidates && bboxFromPlanner && src.length > 0 && candidatesById instanceof Map) {
      const coversAll = src.every((id) => {
        const cand = candidatesById.get(String(id));
        if (!cand || !cand.bbox) return true;
        const cb = clampBox(cand.bbox, imageWidth, imageHeight);
        const area = Math.max(1, boxArea(cb));
        const inter = boxIntersectionArea(cb, bboxFromPlanner);
        return inter / area >= 0.55;
      });
      const iou = boxIou(bboxFromCandidates, bboxFromPlanner);
      bbox = coversAll && iou >= 0.35 ? bboxFromPlanner : bboxFromCandidates;
    }

    // Normalize nodeId based on the source candidates (prevents "semantic" mis-anchoring).
    let nodeId = String(ov.nodeId || '');
    if (nodeId !== '__global__' && src.length > 0 && candidatesById instanceof Map) {
      const counts = new Map();
      src.forEach((id) => {
        const cand = candidatesById.get(String(id));
        const nid = String(cand?.nodeId || '');
        if (!nid) return;
        counts.set(nid, (counts.get(nid) || 0) + 1);
      });
      let best = '';
      let bestCount = 0;
      for (const [nid, count] of counts.entries()) {
        if (count > bestCount) {
          bestCount = count;
          best = nid;
        }
      }
      if (best) nodeId = best;
    }

    if (gran !== 'alphaMask') return { ...ov, nodeId, bbox, fgPoints: [], bgPoints: [] };

    let fg = normalizePoints(ov.fgPoints, imageWidth, imageHeight, 12);
    let bg = normalizePoints(ov.bgPoints, imageWidth, imageHeight, 16);
    if (bbox) {
      fg = fg.filter((p) => pointInsideBox(p, bbox, 0));
      bg = bg.filter((p) => pointInsideBox(p, bbox, 0));
    }

    if ((fg.length < 2 || bg.length < 2) && src.length > 0 && candidatesById instanceof Map) {
      const fgSeed = [];
      const bgSeed = [];
      src.forEach((id) => {
        const cand = candidatesById.get(String(id));
        if (!cand) return;
        if (Array.isArray(cand.fgPointsSeed)) fgSeed.push(...cand.fgPointsSeed);
        if (Array.isArray(cand.bgPointsSeed)) bgSeed.push(...cand.bgPointsSeed);
      });
      const fgMerged = dedupePoints([...fg, ...normalizePoints(fgSeed, imageWidth, imageHeight, 24)], 24);
      const bgMerged = dedupePoints([...bg, ...normalizePoints(bgSeed, imageWidth, imageHeight, 32)], 32);
      fg = bbox ? fgMerged.filter((p) => pointInsideBox(p, bbox, 0)) : fgMerged;
      bg = bbox ? bgMerged.filter((p) => pointInsideBox(p, bbox, 0)) : bgMerged;
    }

    if (bbox && (fg.length < 2 || bg.length < 2)) {
      const seeded = ensureOverlaySeedPoints(fg, bg, bbox, imageWidth, imageHeight);
      fg = seeded.fg;
      bg = seeded.bg;
    }

    return {
      ...ov,
      nodeId,
      bbox,
      fgPoints: fg,
      bgPoints: bg,
      granularity: ov.granularity || gran,
      id: ov.id || `p${idx + 1}`
    };
  });
}

function denoiseOverlayCandidatesForNode(node, candidates) {
  const nodeBox = node?.bbox;
  const nodeArea = Math.max(1, boxArea(nodeBox));
  const textBox = node?.textBbox && typeof node.textBbox === 'object' ? node.textBbox : null;

  const scored = (Array.isArray(candidates) ? candidates : [])
    .filter((c) => c && typeof c === 'object' && c.bbox && typeof c.bbox === 'object')
    .map((c) => {
      const area = Math.max(0, boxArea(c.bbox));
      const areaFrac = area / nodeArea;
      const ar = boxAspectRatio(c.bbox);
      const conf = Number(c.confidence);
      const confidence = Number.isFinite(conf) ? Math.max(0, Math.min(1, conf)) : 0.7;
      const score = confidence * Math.sqrt(Math.max(0, areaFrac));
      return { ...c, area, areaFrac, ar, confidence, score };
    })
    .filter((c) => {
      const w = Number(c.bbox.w ?? c.bbox.width) || 0;
      const h = Number(c.bbox.h ?? c.bbox.height) || 0;
      if (!(w >= 8 && h >= 8)) return false;
      if (c.areaFrac > 0.95) return false;
      if (c.ar > 18 || c.ar < 1 / 18) return false;

      const kind = normalizeOverlayKind(c.kindGuess || c.kind || 'icon');
      if ((kind === 'icon' || kind === '3d') && c.areaFrac > 0.65) return false;

      if (textBox && (kind === 'icon' || kind === '3d')) {
        const inter = boxIntersectionArea(c.bbox, textBox);
        if (inter / Math.max(1, c.area) > 0.25) return false;
      }

      return true;
    });

  if (scored.length === 0) return [];

  const maxScore = scored.reduce((m, c) => Math.max(m, Number(c.score) || 0), 0);
  const filtered = scored.filter((c) => {
    if (c.areaFrac < 0.002 && c.score < maxScore * 0.25) return false;
    if (c.areaFrac < 0.006 && c.score < maxScore * 0.15) return false;
    return true;
  });

  filtered.sort((a, b) => Number(b.score) - Number(a.score));

  const kept = [];
  for (const cand of filtered) {
    const dup = kept.some((k) => boxIou(k.bbox, cand.bbox) >= 0.78);
    if (dup) continue;
    const contained = kept.some((k) => boxContains(k.bbox, cand.bbox, 2) && cand.area / Math.max(1, k.area) <= 0.35);
    if (contained) continue;
    kept.push(cand);
  }

  // Keep a moderate top-level set; tiny fragments are better handled via manual add if needed.
  kept.sort((a, b) => Number(b.score) - Number(a.score));
  return kept.slice(0, 18);
}

function buildOverlaySemanticPlannerInstruction(imageWidth, imageHeight) {
  const w = Number(imageWidth);
  const h = Number(imageHeight);
  const sizeHint = Number.isFinite(w) && Number.isFinite(h) && w > 0 && h > 0 ? `Reference image size: ${w}x${h} pixels.` : '';

  return [
    'You are planning bitmap overlay extraction for diagrams.net (draw.io) from a paper-figure screenshot.',
    'Overlays are non-text visuals that are hard to recreate with vector shapes: detailed icons, 3D illustrations, photos, screenshots, charts/plots, textures/noise blocks.',
    'All TEXT must remain editable: overlays MUST NOT include text regions.',
    'Do NOT treat arrows/connectors/lines as overlays.',
    sizeHint,
    '',
    'You will receive images in this order:',
    '(1) reference screenshot',
    '(2) same screenshot annotated with boxes: nodes (blue) and overlay candidates (orange) labeled by candidate.tag',
    '(3) (optional) a thumbnail contact sheet of top candidates, each labeled by candidate.tag',
    '',
    'Return ONLY a JSON object with keys: finalOverlays, ignoreCandidateIds.',
    'finalOverlays: array of {',
    '  id: string,',
    '  nodeId: string | "__global__",',
    '  kind: "icon"|"3d"|"photo"|"chart"|"plot"|"noise"|"screenshot",',
    '  granularity: "alphaMask"|"opaqueRect"|"ignore",',
    '  bbox: {x:number,y:number,w:number,h:number},',
    '  sourceCandidateIds: string[],',
    '  fgPoints: [{x:number,y:number},...],',
    '  bgPoints: [{x:number,y:number},...],',
    '  confidence: number',
    '}',
    '',
	    'Rules:',
	    '- All coordinates are in full-image pixel coordinates (origin top-left).',
	    '- If nodeId is not "__global__", bbox MUST be fully inside that node bbox.',
	    '- Avoid covering node.textBbox when provided.',
	    '- IMPORTANT: Do NOT guess coordinates. If sourceCandidateIds is non-empty, bbox MUST be the tight union of the referenced candidate bbox(es).',
	    '- For kind icon/3d, use granularity="alphaMask".',
	    '- For kind photo/chart/plot/noise/screenshot, use granularity="opaqueRect".',
	    '- You may MERGE multiple candidate fragments into one overlay by listing multiple sourceCandidateIds (candidate.id or candidate.tag).',
	    '- Prefer a small number of top-level meaningful overlays; ignore tiny noisy fragments.',
	    '- Return ONLY JSON. No markdown/code fences. No explanations.'
	  ]
    .filter(Boolean)
    .join('\n');
}

async function callOverlaySemanticPlanner(provider, prompt, firstImage, imageWidth, imageHeight, nodes, candidates, visionUrl) {
  const nodeList = (Array.isArray(nodes) ? nodes : [])
    .filter((n) => n && typeof n === 'object' && n.id && n.bbox && String(n.render || '').toLowerCase() !== 'text')
    .slice(0, 80)
    .map((n) => ({
      id: String(n.id),
      bbox: n.bbox,
      text: typeof n.text === 'string' ? n.text : '',
      textBbox: n.textBbox && typeof n.textBbox === 'object' ? n.textBbox : null
    }));

  const candAll = Array.isArray(candidates) ? candidates : [];
  const candSorted = candAll
    .slice(0, 220)
    .filter((c) => c && typeof c === 'object' && c.id && c.nodeId && c.bbox)
    .slice()
    .sort((a, b) => {
      const na = String(a.nodeId || '');
      const nb = String(b.nodeId || '');
      if (na !== nb) return na.localeCompare(nb);
      const sa = Number(a.score ?? a.confidence ?? 0);
      const sb = Number(b.score ?? b.confidence ?? 0);
      return sb - sa;
    });

  const candList = candSorted.map((c, idx) => ({
    id: String(c.id),
    tag: `c${idx + 1}`,
    nodeId: String(c.nodeId),
    bbox: c.bbox,
    kindGuess: normalizeOverlayKind(c.kindGuess || c.kind || 'icon'),
    granularityGuess: normalizeGranularity(c.granularityGuess || c.granularity || defaultGranularityForKind(c.kindGuess || c.kind || 'icon')),
    confidence: typeof c.confidence === 'number' ? c.confidence : 0.7,
    previewDataUrl: typeof c.previewDataUrl === 'string' && c.previewDataUrl.startsWith('data:image/') ? c.previewDataUrl : null
  }));

  let annotatedImage = null;
  let contactSheetImage = null;
  try {
    if (visionUrl) {
      const overlaysForAnnotate = candList.slice(0, 220).map((c) => ({ id: c.id, bbox: c.bbox, label: c.tag, color: '#f97316' }));
      const nodesForAnnotate = nodeList.map((n) => ({ id: n.id, bbox: n.bbox, textBbox: n.textBbox }));
      const ann = await callVisionServiceAnnotateBoxes(visionUrl, firstImage, imageWidth, imageHeight, nodesForAnnotate, overlaysForAnnotate);
      if (ann && typeof ann.annotated === 'string' && ann.annotated.startsWith('data:image/')) {
        annotatedImage = { name: 'candidates.png', type: 'image/png', dataUrl: ann.annotated, width: imageWidth, height: imageHeight };
      }

      try {
        const items = candList
          .slice(0, 48)
          .map((c) => ({ id: c.id, label: c.tag, bbox: c.bbox, previewDataUrl: c.previewDataUrl, nodeId: c.nodeId }));
        const sheet = await callVisionServiceContactSheet(visionUrl, firstImage, imageWidth, imageHeight, items, {
          tileSize: 152,
          cols: 6,
          maxItems: 48
        });
        if (sheet && typeof sheet.image === 'string' && sheet.image.startsWith('data:image/')) {
          contactSheetImage = { name: 'thumbs.png', type: 'image/png', dataUrl: sheet.image, width: 0, height: 0 };
        }
      } catch (err) {
        logError('Vision contact_sheet failed (planner will proceed without thumbnails)', err, { visionUrl });
        contactSheetImage = null;
      }
    }
  } catch (err) {
    logError('Vision annotate_boxes failed (planner will use only the reference image)', err, { visionUrl });
    annotatedImage = null;
    contactSheetImage = null;
  }

  const system = buildOverlaySemanticPlannerInstruction(imageWidth, imageHeight);
  const user = [
    prompt ? `User request:\n${String(prompt)}` : 'User request: (none)',
    '',
    'INPUT (nodes + candidates):',
    JSON.stringify({ nodes: nodeList, candidates: candList })
  ]
    .filter(Boolean)
    .join('\n');

  const imagesForPlanner = [firstImage, annotatedImage, contactSheetImage].filter(Boolean);
  const raw = await callVisionJsonWithImages(provider, system, user, imagesForPlanner, { maxTokens: 6144, timeoutMs: 180000 });
  const parsed = parseJsonFromModel(raw);
  if (!parsed || typeof parsed !== 'object') return parsed;

  const tagToId = new Map(candList.map((c) => [String(c.tag), String(c.id)]));
  const resolveRef = (value) => {
    const s = String(value || '');
    return tagToId.get(s) || s;
  };

  if (Array.isArray(parsed.ignoreCandidateIds)) {
    parsed.ignoreCandidateIds = parsed.ignoreCandidateIds.map(resolveRef);
  }
  if (Array.isArray(parsed.finalOverlays)) {
    parsed.finalOverlays = parsed.finalOverlays.map((ov) => {
      if (!ov || typeof ov !== 'object') return ov;
      const src = Array.isArray(ov.sourceCandidateIds) ? ov.sourceCandidateIds.map(resolveRef) : [];
      return { ...ov, sourceCandidateIds: src };
    });
  }
  return parsed;
}

function applyOverlaySemanticPlanToStructure(structure, plan, imageWidth, imageHeight) {
  const out = { planned: 0, dropped: 0 };
  if (!structure || typeof structure !== 'object') return out;
  const nodes = Array.isArray(structure.nodes) ? structure.nodes : [];
  const byId = new Map(nodes.map((n) => [String(n?.id || ''), n]));

  nodes.forEach((n) => {
    if (!n || typeof n !== 'object') return;
    if (String(n.render || '').toLowerCase() === 'text') return;
    n.nodeOverlays = [];
  });
  structure.overlays = [];

  const items = Array.isArray(plan?.finalOverlays) ? plan.finalOverlays : [];
  const globalIdSet = new Set();
  const perNodeIdSet = new Map();

  items.forEach((ov, idx) => {
    if (!ov || typeof ov !== 'object') return;
    const nodeIdRaw = String(ov.nodeId || '');
    const isGlobal = nodeIdRaw === '__global__';
    const node = isGlobal ? null : byId.get(nodeIdRaw);
    if (!isGlobal && !node) return;

    const kind = normalizeOverlayKind(ov.kind || 'icon');
    let granularity = normalizeGranularity(ov.granularity || defaultGranularityForKind(kind));
    if (granularity !== 'ignore') {
      granularity = defaultGranularityForKind(kind);
    }
    if (granularity === 'ignore') {
      out.dropped += 1;
      return;
    }

    const bbRaw = ov.bbox && typeof ov.bbox === 'object' ? ov.bbox : null;
    if (!bbRaw) return;
    let bbox = clampBox(bbRaw, imageWidth, imageHeight);
    if (node?.bbox) bbox = clampBoxWithin(bbox, node.bbox);

    const w = Number(bbox.w ?? bbox.width);
    const h = Number(bbox.h ?? bbox.height);
    if (!(w >= 8 && h >= 8)) {
      out.dropped += 1;
      return;
    }

    const baseId = safeOverlayId(ov.id || `p${idx + 1}`, `p${idx + 1}`);
    const keyNode = isGlobal ? '__global__' : String(nodeIdRaw);
    if (!perNodeIdSet.has(keyNode)) perNodeIdSet.set(keyNode, new Set());
    const localSet = perNodeIdSet.get(keyNode);
    let id = baseId;
    let j = 2;
    while (localSet.has(id) || globalIdSet.has(`${keyNode}::${id}`)) {
      id = `${baseId}_${j}`;
      j += 1;
    }
    localSet.add(id);
    globalIdSet.add(`${keyNode}::${id}`);

    const fgPoints = normalizePoints(ov.fgPoints, imageWidth, imageHeight, 12);
    const bgPoints = normalizePoints(ov.bgPoints, imageWidth, imageHeight, 16);
    const conf = Number(ov.confidence);

    const overlayOut = {
      id,
      kind,
      granularity,
      bbox,
      fgPoints: granularity === 'alphaMask' ? fgPoints : [],
      bgPoints: granularity === 'alphaMask' ? bgPoints : [],
      confidence: Number.isFinite(conf) ? Math.max(0, Math.min(1, conf)) : 0.8
    };

    if (isGlobal) {
      structure.overlays.push(overlayOut);
    } else {
      if (!Array.isArray(node.nodeOverlays)) node.nodeOverlays = [];
      node.nodeOverlays.push(overlayOut);
    }
    out.planned += 1;
  });

  return out;
}

function buildCandidateLabelInstruction(imageWidth, imageHeight, candidates) {
  const w = Number(imageWidth);
  const h = Number(imageHeight);
  const sizeHint = Number.isFinite(w) && Number.isFinite(h) && w > 0 && h > 0 ? `Reference image size: ${w}x${h} pixels.` : '';
  const list = Array.isArray(candidates) ? candidates : [];
  const compact = list
    .filter((c) => c && typeof c === 'object' && c.id && c.bbox)
    .slice(0, 60)
    .map((c) => ({ id: String(c.id), bbox: c.bbox }));

  return [
    'You are extracting an accurate diagram spec from a reference image for diagrams.net (draw.io).',
    'You are given a list of CANDIDATE node containers (id + bbox) detected by CV.',
    'Your job is to: (1) choose which candidates are real nodes, (2) transcribe editable text, (3) infer edges between nodes.',
    sizeHint,
    '',
    'Return ONLY a single JSON object with keys: nodes, edges, overlays.',
    'All coordinates are in reference-image pixel space (origin top-left).',
    '',
    'CANDIDATES (do not change bbox values):',
    JSON.stringify(compact),
    '',
    'nodes: array of { id, bbox:{x,y,w,h}, text, textBbox?:{x,y,w,h}, shapeId, render:\"shape\", confidence:{bbox:number,text:number,shape:number} }.',
    'edges: array of { id, source, target, sourceSide:\"left\"|\"right\"|\"top\"|\"bottom\", targetSide:\"left\"|\"right\"|\"top\"|\"bottom\", label?:string, confidence:number }.',
    'overlays: ALWAYS return an empty array []. (Overlays are handled separately.)',
    '',
    'Rules:',
    '- nodes MUST be a subset of the provided candidates; every node.id MUST match a candidate id.',
    '- For every node you keep, COPY bbox EXACTLY from the candidate list; do not modify it.',
    '- text must be an exact transcription of what is inside the node bbox. If there is no text, use an empty string.',
    '- If text is non-empty, provide textBbox tightly around ONLY the rendered text and fully inside bbox.',
    '- shapeId should be a draw.io shape style value (built-in preferred): \"ellipse\",\"rhombus\",\"parallelogram\",\"cloud\",\"cylinder\",\"document\".',
    '- If unsure, use a rounded rectangle by setting shapeId=\"roundRect\".',
    '- render must be \"shape\" for all returned nodes.',
    '- Include ALL visible nodes from the candidate list; do NOT summarize or drop nodes.',
    '- Infer edges by following connector arrows in the image; every edge source/target must reference a returned node id.',
    '- Do NOT output XML. Do NOT output markdown/code fences. Do NOT include explanations.'
  ]
    .filter(Boolean)
    .join('\\n');
}

function buildUserPrompt(prompt, format, includeInstruction) {
  if (includeInstruction) {
    return `${prompt}\n\nOutput format requirements: ${buildInstruction(format)}`;
  }
  return prompt;
}

function trimBaseUrl(url) {
  if (!url) return '';
  return url.replace(/\/$/, '');
}

function redactUrlSecrets(url) {
  const raw = String(url || '');
  if (!raw) return '';
  try {
    const u = new URL(raw);
    const params = ['key', 'api_key', 'apikey', 'token', 'access_token'];
    params.forEach((p) => {
      if (u.searchParams.has(p)) u.searchParams.set(p, 'REDACTED');
    });
    return u.toString();
  } catch (err) {
    return raw.replace(/([?&](?:key|api_key|apikey|token|access_token)=)[^&]+/gi, '$1REDACTED');
  }
}

function buildHttpErrorMessage(response, text, data) {
  const status = response?.status;
  const statusText = response?.statusText;
  let message = '';

  if (data && typeof data === 'object') {
    const errVal = data.error ?? data.message ?? data.detail;
    if (errVal && typeof errVal === 'object') {
      message = errVal.message || safeJson(errVal, 1200);
    } else if (typeof errVal === 'string') {
      message = errVal;
    }
  } else if (typeof data === 'string') {
    message = data;
  }

  if (!message) {
    message = typeof text === 'string' ? text : '';
  }

  message = String(message || '').trim();
  if (!message) {
    const statusPart = status ? `HTTP ${status}` : 'HTTP error';
    message = statusText ? `${statusPart} ${statusText}` : statusPart;
  }
  return message;
}

function buildHttpError(response, text, data, url) {
  const err = new Error(buildHttpErrorMessage(response, text, data));
  err.http = {
    url: redactUrlSecrets(url || ''),
    status: response?.status,
    statusText: response?.statusText,
    bodySnippet: typeof text === 'string' ? text.slice(0, 2000) : '',
    error: data && typeof data === 'object' ? data.error : data
  };
  return err;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isTransientHttpStatus(status) {
  const s = Number(status);
  if (!Number.isFinite(s)) return false;
  if (s === 408 || s === 429) return true;
  // Some gateways temporarily return 403 during provider warmup / permissions propagation.
  if (s === 403) return true;
  if (s === 500 || s === 502 || s === 503 || s === 504) return true;
  return false;
}

function isTransientError(err) {
  if (!err) return false;
  if (err.name === 'AbortError') return true;

  const msg = String(err?.message || '').toLowerCase();
  if (msg.includes('fetch failed')) return true;

  const code = String(err?.cause?.code || err?.code || '').toUpperCase();
  if (
    code === 'ECONNRESET' ||
    code === 'ECONNREFUSED' ||
    code === 'EAI_AGAIN' ||
    code === 'ENOTFOUND' ||
    code === 'ETIMEDOUT' ||
    code === 'UND_ERR_CONNECT_TIMEOUT' ||
    code === 'UND_ERR_HEADERS_TIMEOUT' ||
    code === 'UND_ERR_SOCKET' ||
    code === 'UND_ERR_BODY_TIMEOUT'
  ) {
    return true;
  }

  const status = err?.http?.status;
  return isTransientHttpStatus(status);
}

function modelRequiresStreaming(model) {
  const m = String(model || '').trim().toLowerCase();
  if (!m) return false;
  if (m.startsWith('gpt')) return true;
  if (m.includes('/gpt')) return true;
  if (m.includes('gpt-')) return true;
  return false;
}

function shouldStreamChatRequest(model, baseUrl) {
  if (modelRequiresStreaming(model)) return true;
  const base = String(baseUrl || '').toLowerCase();
  if (base.includes('0-0.pro')) return true;
  return false;
}

async function fetchWithTimeout(url, init, timeoutMs) {
  const ms = Math.max(1000, Number(timeoutMs) || 60000);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } catch (err) {
    // Attach request info for better diagnostics (avoids "providerHttp: null" on network failures).
    const redacted = redactUrlSecrets(url || '');
    if (err && typeof err === 'object') {
      if (!err.http || typeof err.http !== 'object') {
        err.http = { url: redacted, status: null, statusText: '', bodySnippet: '', error: null };
      } else if (!err.http.url) {
        err.http.url = redacted;
      }
      if (err.http && err.http.error == null) {
        const code = err?.cause?.code || err?.code || '';
        const causeMsg = err?.cause?.message || '';
        err.http.error = { message: err.message || '', code: code ? String(code) : '', cause: causeMsg ? String(causeMsg).slice(0, 900) : '' };
      }
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

async function readChatCompletionsEventStream(response) {
  if (!response?.body) throw new Error('Missing streaming response body.');
  const decoder = new TextDecoder();
  const reader = response.body.getReader();
  let buf = '';
  let content = '';
  let finishReason = '';

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      let idx;
      while ((idx = buf.indexOf('\n')) !== -1) {
        const line = buf.slice(0, idx).trimEnd();
        buf = buf.slice(idx + 1);
        const trimmed = line.trim();
        if (!trimmed) continue;
        if (!trimmed.startsWith('data:')) continue;
        const payload = trimmed.slice(5).trim();
        if (!payload) continue;
        if (payload === '[DONE]') {
          try {
            await reader.cancel();
          } catch (err) {
            // ignore
          }
          return { content, finishReason };
        }

        let json = null;
        try {
          json = JSON.parse(payload);
        } catch (err) {
          continue;
        }

        const choice = json?.choices?.[0] || null;
        if (choice?.finish_reason) finishReason = String(choice.finish_reason);

        const delta = choice?.delta || null;
        if (delta && typeof delta === 'object') {
          if (typeof delta.content === 'string') content += delta.content;
          else if (typeof delta.text === 'string') content += delta.text;
        }

        const message = choice?.message || null;
        if (message && typeof message === 'object' && typeof message.content === 'string') {
          content += message.content;
        }
      }
    }
  } finally {
    try {
      reader.releaseLock();
    } catch (err) {
      // ignore
    }
  }

  return { content, finishReason };
}

async function fetchChatCompletionsJson(url, body, headers, options = {}) {
  const maxAttempts = Math.max(1, Math.min(5, Number(options.maxAttempts) || 3));
  const timeoutMs = Math.max(5000, Number(options.timeoutMs) || 60000);
  const wantsStream = Boolean(body && typeof body === 'object' && body.stream === true);
  const requestHeaders = wantsStream ? { ...headers, Accept: 'text/event-stream' } : headers;

  let lastErr = null;
  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      const response = await fetchWithTimeout(
        url,
        {
          method: 'POST',
          headers: requestHeaders,
          body: JSON.stringify(body)
        },
        timeoutMs
      );

      if (!response.ok) {
        const text = await response.text();
        let data = null;
        try {
          data = JSON.parse(text);
        } catch (err) {
          // ignore
        }

        const err = buildHttpError(response, text, data, url);
        lastErr = err;
        if (isTransientHttpStatus(response.status) && attempt < maxAttempts) {
          const backoff = 400 * attempt * attempt + Math.floor(Math.random() * 200);
          await sleep(backoff);
          continue;
        }
        throw err;
      }

      const contentType = String(response.headers.get('content-type') || '').toLowerCase();
      if (wantsStream && contentType.includes('text/event-stream')) {
        const streamOut = await readChatCompletionsEventStream(response);
        const content = String(streamOut?.content || '');
        const finishReason = String(streamOut?.finishReason || '');
        const data = { choices: [{ message: { content }, finish_reason: finishReason }] };
        return { response, text: content, data };
      }

      const text = await response.text();
      let data = null;
      try {
        data = JSON.parse(text);
      } catch (err) {
        // ignore
      }

      if (wantsStream && !data && text.includes('data:')) {
        let content = '';
        let finishReason = '';
        text
          .split('\n')
          .map((line) => line.trim())
          .forEach((line) => {
            if (!line.startsWith('data:')) return;
            const payload = line.slice(5).trim();
            if (!payload || payload === '[DONE]') return;
            try {
              const json = JSON.parse(payload);
              const choice = json?.choices?.[0] || null;
              if (choice?.finish_reason) finishReason = String(choice.finish_reason);
              const delta = choice?.delta || null;
              if (delta && typeof delta === 'object') {
                if (typeof delta.content === 'string') content += delta.content;
                else if (typeof delta.text === 'string') content += delta.text;
              }
              const message = choice?.message || null;
              if (message && typeof message === 'object' && typeof message.content === 'string') content += message.content;
            } catch (err) {
              // ignore
            }
          });
        if (content) {
          data = { choices: [{ message: { content }, finish_reason: finishReason }] };
          return { response, text: content, data };
        }
      }

      return { response, text, data };
    } catch (err) {
      lastErr = err;
      if (isTransientError(err) && attempt < maxAttempts) {
        const backoff = 400 * attempt * attempt + Math.floor(Math.random() * 200);
        await sleep(backoff);
        continue;
      }
      throw err;
    }
  }

  throw lastErr || new Error('Provider error');
}

function extractImageParts(images = []) {
  return images
    .filter((img) => img && img.dataUrl)
    .map((img) => ({ dataUrl: img.dataUrl, publicUrl: img.publicUrl, type: img.type || 'image/png' }));
}

async function callOpenAICompatible(provider, prompt, images, format) {
  const baseUrl = trimBaseUrl(provider.baseUrl);
  const url = `${baseUrl}/chat/completions`;
  const system = buildInstruction(format);
  const userPrompt = buildUserPrompt(prompt, format, false);
  const imgParts = extractImageParts(images);
  const wantsStream = shouldStreamChatRequest(provider.model, provider.baseUrl);

  async function requestCompletion(messages) {
    const headers = {
      Authorization: `Bearer ${provider.apiKey || ''}`,
      'Content-Type': 'application/json'
    };

    const body = {
      model: provider.model,
      messages,
      temperature: 0.2,
      max_tokens: 2048
    };
    if (wantsStream) body.stream = true;

    const { text, data } = await fetchChatCompletionsJson(
      url,
      body,
      headers,
      { maxAttempts: 3, timeoutMs: 65000 }
    );

    if (!data) return String(text || '').trim();
    return data.choices && data.choices[0] && data.choices[0].message ? data.choices[0].message.content : '';
  }

  if (imgParts.length === 0) {
    return requestCompletion([
      { role: 'system', content: system },
      { role: 'user', content: userPrompt }
    ]);
  }

  const visionMessages = [
    { role: 'system', content: system },
    {
      role: 'user',
      content: [
        { type: 'text', text: userPrompt },
        ...imgParts.map((img) => ({ type: 'image_url', image_url: { url: img.publicUrl || img.dataUrl } }))
      ]
    }
  ];

  try {
    return await requestCompletion(visionMessages);
  } catch (err) {
    const altMessages = [
      { role: 'system', content: system },
      {
        role: 'user',
        content: [
          { type: 'text', text: userPrompt },
          ...imgParts.map((img) => ({ type: 'image_url', image_url: img.publicUrl || img.dataUrl }))
        ]
      }
    ];

    try {
      return await requestCompletion(altMessages);
    } catch (altErr) {
      throw err;
    }
  }
}

async function callGemini(provider, prompt, images, format) {
  const baseUrl = trimBaseUrl(provider.baseUrl || providerBaseUrls.gemini);
  const url = `${baseUrl}/models/${provider.model}:generateContent?key=${provider.apiKey}`;
  const userPrompt = buildUserPrompt(prompt, format, true);
  const parts = [{ text: userPrompt }];
  const imgParts = extractImageParts(images);

  imgParts.forEach((img) => {
    const [header, data] = img.dataUrl.split(',');
    const mime = header.match(/data:(.*?);base64/)?.[1] || img.type || 'image/png';
    parts.push({ inline_data: { mime_type: mime, data } });
  });

  const response = await fetchWithTimeout(
    url,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ contents: [{ role: 'user', parts }] })
    },
    120000
  );

  const text = await response.text();
  let data = null;
  try {
    data = JSON.parse(text);
  } catch (err) {
    data = null;
  }
  if (!response.ok) throw buildHttpError(response, text, data, url);

  const partsOut = data?.candidates?.[0]?.content?.parts || [];
  return partsOut.map((part) => part.text || '').join('').trim();
}

async function callAnthropic(provider, prompt, images, format) {
  const baseUrl = trimBaseUrl(provider.baseUrl || providerBaseUrls.anthropic);
  const url = `${baseUrl}/messages`;
  const userPrompt = buildUserPrompt(prompt, format, true);
  const imgParts = extractImageParts(images);
  const content = [{ type: 'text', text: userPrompt }];

  imgParts.forEach((img) => {
    const [header, data] = img.dataUrl.split(',');
    const mime = header.match(/data:(.*?);base64/)?.[1] || img.type || 'image/png';
    content.push({ type: 'image', source: { type: 'base64', media_type: mime, data } });
  });

  const response = await fetchWithTimeout(
    url,
    {
      method: 'POST',
      headers: {
        'x-api-key': provider.apiKey || '',
        'anthropic-version': '2023-06-01',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: provider.model,
        max_tokens: 2048,
        messages: [{ role: 'user', content }]
      })
    },
    120000
  );

  const text = await response.text();
  let data = null;
  try {
    data = JSON.parse(text);
  } catch (err) {
    data = null;
  }
  if (!response.ok) throw buildHttpError(response, text, data, url);

  return data?.content && data.content[0] ? data.content[0].text : '';
}

async function callOllama(provider, prompt, images, format) {
  const baseUrl = trimBaseUrl(provider.baseUrl || providerBaseUrls.ollama);
  const url = `${baseUrl}/generate`;
  const combinedPrompt = `${buildInstruction(format)}\n\n${prompt}`;

  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: provider.model,
      prompt: combinedPrompt,
      stream: false
    })
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || 'Provider error');
  }

  return data.response || '';
}

function providerSupportsImageGeneration(provider) {
  const type = String(provider?.type || '');
  // This server only implements OpenAI-style image generation endpoints for OpenAI-compatible providers.
  if (type === 'gemini') return false;
  if (type === 'anthropic') return false;
  if (type === 'ollama') return false;
  return true;
}

function parseDataUrlToBuffer(dataUrl) {
  const text = String(dataUrl || '');
  const m = text.match(/^data:(.*?);base64,(.*)$/);
  if (!m) return null;
  const mime = String(m[1] || '').trim() || 'application/octet-stream';
  const b64 = m[2] || '';
  try {
    const buf = Buffer.from(b64, 'base64');
    if (!buf.length) return null;
    return { mime, buffer: buf };
  } catch (err) {
    return null;
  }
}

async function fetchImageUrlAsDataUrl(url) {
  const target = String(url || '').trim();
  if (!target) throw new Error('Image URL missing.');
  const response = await fetchWithTimeout(target, { method: 'GET' }, 120000);
  if (!response.ok) {
    const text = await response.text().catch(() => '');
    throw buildHttpError(response, text, null, target);
  }
  const mime = String(response.headers.get('content-type') || 'image/png').split(';')[0] || 'image/png';
  const buf = Buffer.from(await response.arrayBuffer());
  const b64 = buf.toString('base64');
  return `data:${mime};base64,${b64}`;
}

async function callOpenAICompatibleImageModel(provider, prompt, images) {
  const normalized = normalizeProvider(provider);
  if (!normalized) throw new Error('Provider not configured');

  const imageModel = String(normalized.imageModel || '').trim();
  if (!imageModel) {
    const err = new Error('Image model is not configured.');
    err.code = 'image_model_missing';
    throw err;
  }

  const baseUrl = trimBaseUrl(normalized.baseUrl);
  if (!baseUrl) {
    const err = new Error('Base URL is required for this provider.');
    err.code = 'provider_base_missing';
    throw err;
  }

  const headers = {
    Authorization: `Bearer ${normalized.apiKey || ''}`
  };

  const imgParts = extractImageParts(images);
  const hasEditImage = imgParts.length > 0 && imgParts[0] && typeof imgParts[0].dataUrl === 'string' && imgParts[0].dataUrl.startsWith('data:image/');

  // If a reference image is provided, try /images/edits (preferred for “modify/style” flows).
  if (hasEditImage) {
    const editUrl = `${baseUrl}/images/edits`;
    const parsed = parseDataUrlToBuffer(imgParts[0].dataUrl);
    if (!parsed) throw new Error('Invalid reference image.');

    const form = new FormData();
    form.append('model', imageModel);
    form.append('prompt', String(prompt || '').trim());
    form.append('size', '1024x1024');
    form.append('n', '1');
    form.append('response_format', 'b64_json');
    form.append('image', new Blob([parsed.buffer], { type: parsed.mime }), 'image.png');

    const response = await fetchWithTimeout(editUrl, { method: 'POST', headers, body: form }, 180000);
    const text = await response.text();
    let data = null;
    try {
      data = JSON.parse(text);
    } catch (err) {
      data = null;
    }
    if (!response.ok) throw buildHttpError(response, text, data, editUrl);
    const item = data?.data && Array.isArray(data.data) ? data.data[0] : null;
    const b64 = item?.b64_json || item?.b64 || item?.base64 || '';
    const outUrl = item?.url || '';
    if (b64) return `data:image/png;base64,${b64}`;
    if (outUrl) return await fetchImageUrlAsDataUrl(outUrl);
    throw new Error('Image model returned no image.');
  }

  const genUrl = `${baseUrl}/images/generations`;
  const response = await fetchWithTimeout(
    genUrl,
    {
      method: 'POST',
      headers: { ...headers, 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: imageModel,
        prompt: String(prompt || '').trim(),
        size: '1024x1024',
        n: 1,
        response_format: 'b64_json'
      })
    },
    180000
  );

  const text = await response.text();
  let data = null;
  try {
    data = JSON.parse(text);
  } catch (err) {
    data = null;
  }
  if (!response.ok) throw buildHttpError(response, text, data, genUrl);
  const item = data?.data && Array.isArray(data.data) ? data.data[0] : null;
  const b64 = item?.b64_json || item?.b64 || item?.base64 || '';
  const outUrl = item?.url || '';
  if (b64) return `data:image/png;base64,${b64}`;
  if (outUrl) return await fetchImageUrlAsDataUrl(outUrl);
  throw new Error('Image model returned no image.');
}

async function callProvider(provider, prompt, images, format) {
  const wantsVision = extractImageParts(images).length > 0;
  const normalized = wantsVision ? normalizeProviderForVision(provider) : normalizeProvider(provider);
  if (!normalized) {
    throw new Error('Provider not configured');
  }
  const type = normalized.type;

  if (type === 'gemini') {
    return callGemini(normalized, prompt, images, format);
  }

  if (type === 'anthropic') {
    return callAnthropic(normalized, prompt, images, format);
  }

  if (type === 'ollama') {
    return callOllama(normalized, prompt, images, format);
  }

  return callOpenAICompatible(normalized, prompt, images, format);
}

async function callOverlays(provider, images, imageWidth, imageHeight) {
  const normalized = normalizeProviderForVision(provider);
  if (!normalized) {
    throw new Error('Provider not configured');
  }

  const instruction = buildOverlayInstruction(imageWidth, imageHeight);
  const timeoutMs = 120000;
  const imgParts = extractImageParts(images);
  if (imgParts.length === 0) {
    return '[]';
  }

  if (normalized.type === 'gemini') {
    const baseUrl = trimBaseUrl(normalized.baseUrl || providerBaseUrls.gemini);
    const url = `${baseUrl}/models/${normalized.model}:generateContent?key=${normalized.apiKey}`;
    const parts = [{ text: instruction }];
    imgParts.forEach((img) => {
      const [header, data] = img.dataUrl.split(',');
      const mime = header.match(/data:(.*?);base64/)?.[1] || img.type || 'image/png';
      parts.push({ inline_data: { mime_type: mime, data } });
    });

    const response = await fetchWithTimeout(
      url,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ contents: [{ role: 'user', parts }] })
      },
      timeoutMs
    );
    const text = await response.text();
    let data = null;
    try {
      data = JSON.parse(text);
    } catch (err) {
      data = null;
    }
    if (!response.ok) throw buildHttpError(response, text, data, url);
    const partsOut = data?.candidates?.[0]?.content?.parts || [];
    return partsOut.map((part) => part.text || '').join('').trim();
  }

  if (normalized.type === 'anthropic') {
    const baseUrl = trimBaseUrl(normalized.baseUrl || providerBaseUrls.anthropic);
    const url = `${baseUrl}/messages`;
    const content = [{ type: 'text', text: instruction }];
    imgParts.forEach((img) => {
      const [header, data] = img.dataUrl.split(',');
      const mime = header.match(/data:(.*?);base64/)?.[1] || img.type || 'image/png';
      content.push({ type: 'image', source: { type: 'base64', media_type: mime, data } });
    });

    const response = await fetchWithTimeout(
      url,
      {
        method: 'POST',
        headers: {
          'x-api-key': normalized.apiKey || '',
          'anthropic-version': '2023-06-01',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model: normalized.model,
          max_tokens: 1024,
          messages: [{ role: 'user', content }]
        })
      },
      timeoutMs
    );

    const text = await response.text();
    let data = null;
    try {
      data = JSON.parse(text);
    } catch (err) {
      data = null;
    }
    if (!response.ok) throw buildHttpError(response, text, data, url);

    return data?.content && data.content[0] ? data.content[0].text : '';
  }

  if (normalized.type === 'ollama') {
    throw new Error('This provider does not support image overlays.');
  }

  // OpenAI-compatible vision JSON
  const baseUrl = trimBaseUrl(normalized.baseUrl);
  const url = `${baseUrl}/chat/completions`;
  const messages = [
    { role: 'system', content: instruction },
    {
      role: 'user',
      content: [
        { type: 'text', text: 'Return the overlays JSON array ONLY.' },
        ...imgParts.map((img) => ({ type: 'image_url', image_url: { url: img.publicUrl || img.dataUrl } }))
      ]
    }
  ];

  const headers = {
    Authorization: `Bearer ${normalized.apiKey || ''}`,
    'Content-Type': 'application/json'
  };

  const body = { model: normalized.model, messages, temperature: 0.1, max_tokens: 1024 };
  if (shouldStreamChatRequest(normalized.model, normalized.baseUrl)) body.stream = true;

  const { text, data } = await fetchChatCompletionsJson(
    url,
    body,
    headers,
    { maxAttempts: 3, timeoutMs: 90000 }
  );

  if (!data) return String(text || '').trim();
  return data.choices?.[0]?.message?.content || '';
}

function buildStructureRepairPrompt(originalInstruction, userPrompt, previousOutput, error) {
  const req = String(userPrompt || '').trim();
  return [
    originalInstruction,
    '',
    req ? `User request:\n${req}` : '',
    `Validation error:\n${String(error || '').slice(0, 1000)}`,
    `Previous JSON:\n${String(previousOutput || '').slice(0, 6000)}`,
    'Fix the JSON and return ONLY the corrected JSON object. Do not include explanations or markdown.'
  ]
    .filter(Boolean)
    .join('\n\n');
}

function buildOverlayRefinePrompt(params) {
  const imageWidth = Number(params?.imageWidth || 0);
  const imageHeight = Number(params?.imageHeight || 0);
  const overlayId = String(params?.overlayId || '');
  const kind = String(params?.kind || '');
  const granularity = String(params?.granularity || '');
  const bbox = params?.bbox && typeof params.bbox === 'object' ? params.bbox : null;
  const fgPoints = Array.isArray(params?.fgPoints) ? params.fgPoints : [];
  const bgPoints = Array.isArray(params?.bgPoints) ? params.bgPoints : [];
  const reason = String(params?.reason || '');
  const stats = params?.stats && typeof params.stats === 'object' ? params.stats : {};
  const parent = params?.parent && typeof params.parent === 'object' ? params.parent : {};
  const userPrompt = String(params?.prompt || '').trim();

  const sizeHint =
    imageWidth > 0 && imageHeight > 0 ? `Reference image size: ${imageWidth}x${imageHeight} pixels.` : 'Reference image size is unknown.';

  return [
    'You are fixing SAM2 prompt points/bbox for extracting an overlay bitmap from a paper figure screenshot.',
    'The goal is to extract ONLY the non-text visual content and avoid background/text.',
    sizeHint,
    '',
    `Overlay id: ${overlayId}`,
    kind ? `Kind: ${kind}` : '',
    granularity ? `Current granularity: ${granularity}` : '',
    bbox ? `Current bbox (FULL IMAGE coords): ${JSON.stringify(bbox)}` : '',
    fgPoints.length ? `Current fgPoints: ${JSON.stringify(fgPoints)}` : '',
    bgPoints.length ? `Current bgPoints: ${JSON.stringify(bgPoints)}` : '',
    reason ? `Failure reason: ${reason}` : '',
    Object.keys(stats).length ? `Segmentation stats: ${JSON.stringify(stats)}` : '',
    parent?.nodeText ? `Parent node text: ${String(parent.nodeText).slice(0, 120)}` : '',
    '',
    userPrompt ? `User request:\n${userPrompt}` : '',
    '',
    'You will receive TWO images: (1) the crop of the bbox, (2) the current binary mask.',
    'The crop image coordinate system is local: top-left is (0,0) and size is bbox.w x bbox.h.',
    'You MUST output points in FULL IMAGE coordinates. To convert from crop coords: add bbox.x to X and bbox.y to Y.',
    '',
    'Return ONLY a JSON object:',
    '{ "granularity":"alphaMask|opaqueRect|ignore", "bbox":{x,y,w,h}, "fgPoints":[{x,y}...], "bgPoints":[{x,y}...] }',
    '',
    'Rules:',
    '- If kind is "icon" or "3d", you MUST keep granularity="alphaMask". Do NOT switch to opaqueRect.',
    '- If the region is a real photo/screenshot/real chart/noise block, set granularity="opaqueRect" and omit points (or empty arrays). Do NOT try to alpha-mask it.',
    '- If the region is text-only/blank/background, set granularity="ignore".',
    '- If granularity="alphaMask": provide 4-10 fgPoints inside the visual foreground and 4-12 bgPoints on background inside bbox.',
    '- Bbox must still be tight and within image bounds; you may adjust bbox slightly if it is clearly too large/small.',
    '- Do NOT include any explanations or markdown. Return ONLY the JSON object.'
  ]
    .filter(Boolean)
    .join('\n');
}

async function callStructure(provider, prompt, images, imageWidth, imageHeight, attemptPromptOverride, options = {}) {
  const normalized = normalizeProviderForVision(provider);
  if (!normalized) throw new Error('Provider not configured');

  const instruction = buildStructureInstruction(imageWidth, imageHeight, SHAPE_CONFIDENCE_THRESHOLD);
  const userPrompt = String(prompt || '').trim();
  const combined = attemptPromptOverride || (userPrompt ? `${instruction}\n\nUser request:\n${userPrompt}` : instruction);

  const imgParts = extractImageParts(images);
  if (imgParts.length === 0) {
    throw new Error('Reference image is required.');
  }

  if (normalized.type === 'ollama') {
    throw new Error('This provider does not support vision/image input for structure extraction.');
  }

  const timeoutMs = Math.max(15000, Number(options.timeoutMs) || 120000);
  const maxTokens = Math.max(256, Math.min(8192, Number(options.maxTokens) || 3072));

  if (normalized.type === 'gemini') {
    const baseUrl = trimBaseUrl(normalized.baseUrl || providerBaseUrls.gemini);
    const url = `${baseUrl}/models/${normalized.model}:generateContent?key=${normalized.apiKey}`;
    const parts = [{ text: combined }];
    imgParts.forEach((img) => {
      const [header, data] = img.dataUrl.split(',');
      const mime = header.match(/data:(.*?);base64/)?.[1] || img.type || 'image/png';
      parts.push({ inline_data: { mime_type: mime, data } });
    });

    const response = await fetchWithTimeout(
      url,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ contents: [{ role: 'user', parts }], generationConfig: { temperature: 0.1, maxOutputTokens: maxTokens } })
      },
      timeoutMs
    );
    const text = await response.text();
    let data = null;
    try {
      data = JSON.parse(text);
    } catch (err) {
      data = null;
    }
    if (!response.ok) throw buildHttpError(response, text, data, url);
    const partsOut = data?.candidates?.[0]?.content?.parts || [];
    return partsOut.map((part) => part.text || '').join('').trim();
  }

  if (normalized.type === 'anthropic') {
    const baseUrl = trimBaseUrl(normalized.baseUrl || providerBaseUrls.anthropic);
    const url = `${baseUrl}/messages`;
    const content = [{ type: 'text', text: combined }];
    imgParts.forEach((img) => {
      const [header, data] = img.dataUrl.split(',');
      const mime = header.match(/data:(.*?);base64/)?.[1] || img.type || 'image/png';
      content.push({ type: 'image', source: { type: 'base64', media_type: mime, data } });
    });

    const response = await fetchWithTimeout(
      url,
      {
        method: 'POST',
        headers: {
          'x-api-key': normalized.apiKey || '',
          'anthropic-version': '2023-06-01',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ model: normalized.model, max_tokens: maxTokens, messages: [{ role: 'user', content }] })
      },
      timeoutMs
    );
    const text = await response.text();
    let data = null;
    try {
      data = JSON.parse(text);
    } catch (err) {
      data = null;
    }
    if (!response.ok) throw buildHttpError(response, text, data, url);
    return data?.content && data.content[0] ? data.content[0].text : '';
  }

  const baseUrl = trimBaseUrl(normalized.baseUrl);
  const url = `${baseUrl}/chat/completions`;
  const systemContent = attemptPromptOverride || instruction;
  const userText = attemptPromptOverride
    ? 'Fix the JSON and return ONLY the corrected JSON object.'
    : userPrompt
      ? `User request:\n${userPrompt}`
      : 'Extract the diagram structure from the image and return ONLY the JSON object.';

  const messages = [
    { role: 'system', content: systemContent },
    {
      role: 'user',
      content: [{ type: 'text', text: userText }, ...imgParts.map((img) => ({ type: 'image_url', image_url: { url: img.publicUrl || img.dataUrl } }))]
    }
  ];

  const headers = {
    Authorization: `Bearer ${normalized.apiKey || ''}`,
    'Content-Type': 'application/json'
  };

  const body = { model: normalized.model, messages, temperature: 0.1, max_tokens: maxTokens };
  if (shouldStreamChatRequest(normalized.model, normalized.baseUrl)) body.stream = true;

  const { text, data } = await fetchChatCompletionsJson(
    url,
    body,
    headers,
    { maxAttempts: 3, timeoutMs }
  );

  if (!data) return String(text || '').trim();
  const finishReason = data.choices?.[0]?.finish_reason || '';
  const content = data.choices?.[0]?.message?.content || '';
  if (String(finishReason).toLowerCase() === 'length') {
    throw new Error('Model output truncated (finish_reason=length).');
  }
  // Best-effort truncation detection for providers that omit finish_reason.
  const cleaned = cleanModelText(content);
  if (cleaned && (cleaned.startsWith('{') || cleaned.startsWith('['))) {
    const lastBrace = Math.max(cleaned.lastIndexOf('}'), cleaned.lastIndexOf(']'));
    if (lastBrace < cleaned.length - 1) {
      // still fine
    } else {
      const open = cleaned.startsWith('{') ? '{' : '[';
      const close = cleaned.startsWith('{') ? '}' : ']';
      if (!cleaned.trimEnd().endsWith(close)) {
        throw new Error('Model output appears truncated (missing closing bracket).');
      }
      if (!cleaned.includes(close)) {
        throw new Error('Model output appears truncated.');
      }
    }
  }
  return content;
}

async function callVisionJsonWithImages(provider, systemContent, userText, images, options = {}) {
  const normalized = normalizeProviderForVision(provider);
  if (!normalized) throw new Error('Provider not configured');

  const imgParts = extractImageParts(images);
  if (imgParts.length === 0) {
    throw new Error('Reference image is required.');
  }

  if (normalized.type === 'ollama') {
    throw new Error('This provider does not support vision/image input.');
  }

  const timeoutMs = Math.max(15000, Number(options.timeoutMs) || 120000);
  const maxTokens = Math.max(256, Math.min(8192, Number(options.maxTokens) || 2048));

  if (normalized.type === 'gemini') {
    const baseUrl = trimBaseUrl(normalized.baseUrl || providerBaseUrls.gemini);
    const url = `${baseUrl}/models/${normalized.model}:generateContent?key=${normalized.apiKey}`;
    const parts = [{ text: String(systemContent || '') }, { text: String(userText || '') }].filter((p) => p.text);
    imgParts.forEach((img) => {
      const [header, data] = img.dataUrl.split(',');
      const mime = header.match(/data:(.*?);base64/)?.[1] || img.type || 'image/png';
      parts.push({ inline_data: { mime_type: mime, data } });
    });

    const response = await fetchWithTimeout(
      url,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ contents: [{ role: 'user', parts }], generationConfig: { temperature: 0.1, maxOutputTokens: maxTokens } })
      },
      timeoutMs
    );
    const text = await response.text();
    let data = null;
    try {
      data = JSON.parse(text);
    } catch (err) {
      data = null;
    }
    if (!response.ok) throw buildHttpError(response, text, data, url);
    const partsOut = data?.candidates?.[0]?.content?.parts || [];
    return partsOut.map((part) => part.text || '').join('').trim();
  }

  if (normalized.type === 'anthropic') {
    const baseUrl = trimBaseUrl(normalized.baseUrl || providerBaseUrls.anthropic);
    const url = `${baseUrl}/messages`;
    const content = [{ type: 'text', text: `${String(systemContent || '')}\n\n${String(userText || '')}`.trim() }];
    imgParts.forEach((img) => {
      const [header, data] = img.dataUrl.split(',');
      const mime = header.match(/data:(.*?);base64/)?.[1] || img.type || 'image/png';
      content.push({ type: 'image', source: { type: 'base64', media_type: mime, data } });
    });

    const response = await fetchWithTimeout(
      url,
      {
        method: 'POST',
        headers: {
          'x-api-key': normalized.apiKey || '',
          'anthropic-version': '2023-06-01',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ model: normalized.model, max_tokens: maxTokens, messages: [{ role: 'user', content }] })
      },
      timeoutMs
    );
    const text = await response.text();
    let data = null;
    try {
      data = JSON.parse(text);
    } catch (err) {
      data = null;
    }
    if (!response.ok) throw buildHttpError(response, text, data, url);
    return data?.content && data.content[0] ? data.content[0].text : '';
  }

  // OpenAI-compatible vision
  const baseUrl = trimBaseUrl(normalized.baseUrl);
  const url = `${baseUrl}/chat/completions`;
  const messages = [
    { role: 'system', content: String(systemContent || '') },
    {
      role: 'user',
      content: [{ type: 'text', text: String(userText || '') }, ...imgParts.map((img) => ({ type: 'image_url', image_url: { url: img.publicUrl || img.dataUrl } }))]
    }
  ];

  const headers = {
    Authorization: `Bearer ${normalized.apiKey || ''}`,
    'Content-Type': 'application/json'
  };

  const body = { model: normalized.model, messages, temperature: 0.1, max_tokens: maxTokens };
  if (shouldStreamChatRequest(normalized.model, normalized.baseUrl)) body.stream = true;

  const { text, data } = await fetchChatCompletionsJson(
    url,
    body,
    headers,
    { maxAttempts: 3, timeoutMs }
  );

  if (!data) return String(text || '').trim();
  const finishReason = data.choices?.[0]?.finish_reason || '';
  const contentOut = data.choices?.[0]?.message?.content || '';
  if (String(finishReason).toLowerCase() === 'length') {
    throw new Error('Model output truncated (finish_reason=length).');
  }
  const cleaned = cleanModelText(contentOut);
  if (cleaned && (cleaned.startsWith('{') || cleaned.startsWith('['))) {
    const close = cleaned.startsWith('{') ? '}' : ']';
    if (!cleaned.trimEnd().endsWith(close)) {
      throw new Error('Model output appears truncated.');
    }
  }
  return contentOut;
}

async function extractStructureV2(provider, prompt, firstImage, imageWidth, imageHeight) {
  let lastErr = null;
  let lastOut = '';
  let parsed = null;
  let attemptPromptOverride = null;
  let maxTokens = 4096;

  const instruction = buildStructureInstruction(imageWidth, imageHeight, SHAPE_CONFIDENCE_THRESHOLD);

  for (let attempt = 1; attempt <= Math.max(1, STRUCTURE_MAX_ATTEMPTS); attempt += 1) {
    try {
      lastOut = await callStructure(provider, prompt, [firstImage], imageWidth, imageHeight, attemptPromptOverride, { maxTokens });
      parsed = parseJsonFromModel(lastOut);
      if (!parsed || typeof parsed !== 'object') {
        throw new Error('Model did not return a JSON object.');
      }
      break;
    } catch (err) {
      if (err && typeof err === 'object' && !err.phase) err.phase = 'structure_extract';
      lastErr = err;
      const msg = String(err?.message || '').toLowerCase();
      if (msg.includes('truncated')) {
        maxTokens = 6144;
      }
      attemptPromptOverride = buildStructureRepairPrompt(instruction, prompt, lastOut, err?.message || String(err || ''));
    }
  }

  if (!parsed) {
    throw lastErr || new Error('Failed to extract structure.');
  }
  return parsed;
}

function pickArray(obj, keys) {
  for (const k of keys) {
    if (Array.isArray(obj?.[k])) return obj[k];
  }
  return [];
}

function pickObject(obj, keys) {
  for (const k of keys) {
    if (obj?.[k] && typeof obj[k] === 'object') return obj[k];
  }
  return null;
}

async function normalizeStructureFromModel(parsed, imageWidth, imageHeight, catalog) {
  const nodesIn = Array.isArray(parsed.nodes) ? parsed.nodes : [];
  const edgesIn = Array.isArray(parsed.edges) ? parsed.edges : [];
  const overlaysIn = Array.isArray(parsed.overlays) ? parsed.overlays : [];

  const nodeIds = new Set();
  const nodes = [];

  function uniqueNodeId(base) {
    const raw = String(base || '').trim() || `n${nodes.length + 1}`;
    let id = raw;
    let i = 2;
    while (nodeIds.has(id) || id === '0' || id === '1') {
      id = `${raw}_${i}`;
      i += 1;
    }
    nodeIds.add(id);
    return id;
  }

  for (const n of nodesIn) {
    if (!n || typeof n !== 'object') continue;

    const id = uniqueNodeId(n.id);
    const bbox = clampBox(n.bbox || n, imageWidth, imageHeight);
    const text = typeof n.text === 'string' ? n.text : '';

    let textBbox = null;
    if (n.textBbox && typeof n.textBbox === 'object') {
      textBbox = clampBox(n.textBbox, imageWidth, imageHeight);
      if (
        textBbox.x < bbox.x ||
        textBbox.y < bbox.y ||
        textBbox.x + textBbox.w > bbox.x + bbox.w ||
        textBbox.y + textBbox.h > bbox.y + bbox.h
      ) {
        textBbox = null;
      }
    }

    const conf = n.confidence && typeof n.confidence === 'object' ? n.confidence : {};
    const bboxC = Number(conf.bbox);
    const textC = Number(conf.text);
    const shapeC = Number(conf.shape);
    const confidence = {
      bbox: Number.isFinite(bboxC) ? Math.max(0, Math.min(1, bboxC)) : 0.75,
      text: Number.isFinite(textC) ? Math.max(0, Math.min(1, textC)) : text ? 0.8 : 0.0,
      shape: Number.isFinite(shapeC) ? Math.max(0, Math.min(1, shapeC)) : 0.7
    };

    let render = String(n.render || 'shape').toLowerCase();
    if (render !== 'shape' && render !== 'overlay' && render !== 'text') render = 'shape';

    let rawShapeId = normalizeShapeId(String(n.shapeId || n.shape || '').trim());
    if (!rawShapeId) rawShapeId = render === 'text' ? 'label' : 'roundRect';

    let shapeId = 'roundRect';
    if (render === 'shape') {
      try {
        shapeId = (await catalog.isSupported(rawShapeId)) ? rawShapeId : 'roundRect';
      } catch (err) {
        shapeId = 'roundRect';
      }
    }

    const nodeOverlaysIn = pickArray(n, ['nodeOverlays', 'overlays']);
    const nodeOverlays = nodeOverlaysIn
      .filter((o) => o && typeof o === 'object' && (o.bbox || o.geometry || (o.x !== undefined && o.y !== undefined)))
      .slice(0, 80)
      .map((o, idx) => {
        const oid = String(o.id || `nov_${id}_${idx + 1}`);
        const bb = clampBox(o.bbox || o.geometry || o, imageWidth, imageHeight);
        const bbIn = clampBoxWithin(bb, bbox);
        const ow = Number(bbIn.w ?? bbIn.width);
        const oh = Number(bbIn.h ?? bbIn.height);
        const oArea = Math.max(0, ow) * Math.max(0, oh);
        if (!(ow >= 8 && oh >= 8)) return null;
        const ar = ow / Math.max(1, oh);
        if (ar > 12 || ar < 1 / 12) return null;
        const bbW0 = Number(bb.w ?? bb.width);
        const bbH0 = Number(bb.h ?? bb.height);
        const area0 = Math.max(0, bbW0) * Math.max(0, bbH0);
        if (area0 > 0 && oArea / area0 < 0.35) return null;
        const nodeW = Number(bbox.w ?? bbox.width);
        const nodeH = Number(bbox.h ?? bbox.height);
        const nodeArea = Math.max(1, nodeW * nodeH);
        const kind = normalizeOverlayKind(o.kind || o.type || o.label || '');
        const granIn = String(o.granularity || o.mode || '').trim();
        const granularity = granIn ? normalizeGranularity(granIn) : defaultGranularityForKind(kind);
        const frac = oArea / nodeArea;
        if (frac > 0.80) return null;
        if ((kind === 'icon' || kind === '3d') && frac > 0.55) return null;
        if (textBbox) {
          const inter = boxIntersectionArea(bbIn, textBbox);
          if (inter / Math.max(1, oArea) > 0.35) return null;
        }
        const fg = normalizePoints(o.fgPoints || o.fg || o.foregroundPoints, imageWidth, imageHeight, 12);
        const bg = normalizePoints(o.bgPoints || o.bg || o.backgroundPoints, imageWidth, imageHeight, 16);
        const confO = Number(o.confidence);
        return {
          id: oid,
          kind,
          granularity,
          bbox: bbIn,
          fgPoints: fg,
          bgPoints: bg,
          confidence: Number.isFinite(confO) ? Math.max(0, Math.min(1, confO)) : 0.7
        };
      })
      .filter(Boolean);

    let overlayCfg = pickObject(n, ['overlay']);
    if (!overlayCfg || typeof overlayCfg !== 'object') {
      overlayCfg = { kind: 'photo', granularity: 'opaqueRect', confidence: 0.8 };
    }
    const overlayKind = normalizeOverlayKind(
      overlayCfg.kind || overlayCfg.type || overlayCfg.label || (render === 'overlay' ? 'photo' : '')
    );
    const overlayGranIn = String(overlayCfg.granularity || overlayCfg.mode || '').trim();
    const overlay = {
      kind: overlayKind,
      granularity: overlayGranIn ? normalizeGranularity(overlayGranIn) : render === 'overlay' ? defaultGranularityForKind(overlayKind) : 'ignore',
      fgPoints: normalizePoints(overlayCfg.fgPoints || overlayCfg.fg || overlayCfg.foregroundPoints, imageWidth, imageHeight, 12),
      bgPoints: normalizePoints(overlayCfg.bgPoints || overlayCfg.bg || overlayCfg.backgroundPoints, imageWidth, imageHeight, 16),
      confidence: (() => {
        const c = Number(overlayCfg.confidence);
        return Number.isFinite(c) ? Math.max(0, Math.min(1, c)) : 0.75;
      })()
    };

    nodes.push({
      id,
      bbox,
      render,
      shapeId: render === 'shape' ? shapeId : rawShapeId || '',
      text,
      textBbox,
      confidence,
      nodeOverlays,
      overlay,
      containerStyle: n.containerStyle && typeof n.containerStyle === 'object' ? n.containerStyle : null,
      innerShapes: Array.isArray(n.innerShapes) ? n.innerShapes : []
    });
  }

  const edges = edgesIn
    .filter((e) => e && typeof e === 'object')
    .map((e, idx) => {
      const id = String(e.id || `e${idx + 1}`);
      const source = String(e.source || '');
      const target = String(e.target || '');
      const conf = Number(e.confidence);
      return {
        id,
        source,
        target,
        sourceSide: normalizeSide(e.sourceSide),
        targetSide: normalizeSide(e.targetSide),
        label: typeof e.label === 'string' ? e.label : '',
        confidence: Number.isFinite(conf) ? Math.max(0, Math.min(1, conf)) : 0.7
      };
    })
    .filter((e) => nodeIds.has(e.source) && nodeIds.has(e.target));

  const overlays = overlaysIn
    .filter((o) => o && typeof o === 'object' && (o.bbox || o.geometry || (o.x !== undefined && o.y !== undefined)))
    .slice(0, 60)
    .map((o, idx) => {
      const id = String(o.id || `ov${idx + 1}`);
      const bbox = clampBox(o.bbox || o.geometry || o, imageWidth, imageHeight);
      const ow = Number(bbox.w ?? bbox.width);
      const oh = Number(bbox.h ?? bbox.height);
      if (!(ow >= 8 && oh >= 8)) return null;
      const ar = ow / Math.max(1, oh);
      if (ar > 18 || ar < 1 / 18) return null;
      const kind = normalizeOverlayKind(o.kind || o.type || o.label || '');
      const granIn = String(o.granularity || o.mode || '').trim();
      const granularity = granIn ? normalizeGranularity(granIn) : defaultGranularityForKind(kind);
      const fg = normalizePoints(o.fgPoints || o.fg || o.foregroundPoints, imageWidth, imageHeight, 12);
      const bg = normalizePoints(o.bgPoints || o.bg || o.backgroundPoints, imageWidth, imageHeight, 16);
      const conf = Number(o.confidence);
      return {
        id,
        kind,
        granularity,
        bbox,
        fgPoints: fg,
        bgPoints: bg,
        confidence: Number.isFinite(conf) ? Math.max(0, Math.min(1, conf)) : 0.7
      };
    })
    .filter(Boolean);

  return { nodes, edges, overlays };
}

function collectOverlaySlots(structure, options = {}) {
  const onlyMissing = Boolean(options.onlyMissing);
  const slots = [];
  const nodes = Array.isArray(structure.nodes) ? structure.nodes : [];
  const overlays = Array.isArray(structure.overlays) ? structure.overlays : [];

  nodes.forEach((n) => {
    if (!n || typeof n !== 'object' || !n.bbox) return;
    const nodeId = String(n.id || '');

    if (String(n.render || '') === 'overlay') {
      const cfg = n.overlay && typeof n.overlay === 'object' ? n.overlay : { kind: 'photo', granularity: 'opaqueRect' };
      const slotId = `node_${nodeId}`;
      if (!(onlyMissing && typeof n.dataUrl === 'string' && n.dataUrl.startsWith('data:image/'))) {
        const ww = Number(n.bbox?.w ?? n.bbox?.width);
        const hh = Number(n.bbox?.h ?? n.bbox?.height);
        if (!(ww >= 8 && hh >= 8)) return;
        slots.push({
          slotId,
          kind: normalizeOverlayKind(cfg.kind || 'photo'),
          granularity: normalizeGranularity(cfg.granularity || 'opaqueRect'),
          bbox: n.bbox,
          fgPoints: Array.isArray(cfg.fgPoints) ? cfg.fgPoints : [],
          bgPoints: Array.isArray(cfg.bgPoints) ? cfg.bgPoints : [],
          parentNode: n,
          target: { type: 'node', node: n },
          attempts: 0
        });
      }
    }

    if (Array.isArray(n.nodeOverlays)) {
      n.nodeOverlays.forEach((ov) => {
        if (!ov || typeof ov !== 'object' || !ov.bbox) return;
        if (onlyMissing && typeof ov.dataUrl === 'string' && ov.dataUrl.startsWith('data:image/')) return;
        const ww = Number(ov.bbox?.w ?? ov.bbox?.width);
        const hh = Number(ov.bbox?.h ?? ov.bbox?.height);
        if (!(ww >= 8 && hh >= 8)) return;
        const slotId = `nov_${nodeId}_${String(ov.id || '')}`;
        slots.push({
          slotId,
          kind: normalizeOverlayKind(ov.kind || 'icon'),
          granularity: normalizeGranularity(ov.granularity || 'opaqueRect'),
          bbox: ov.bbox,
          fgPoints: Array.isArray(ov.fgPoints) ? ov.fgPoints : [],
          bgPoints: Array.isArray(ov.bgPoints) ? ov.bgPoints : [],
          parentNode: n,
          target: { type: 'nodeOverlay', node: n, overlay: ov },
          attempts: 0
        });
      });
    }
  });

  overlays.forEach((ov) => {
    if (!ov || typeof ov !== 'object' || !ov.bbox) return;
    if (onlyMissing && typeof ov.dataUrl === 'string' && ov.dataUrl.startsWith('data:image/')) return;
    const ww = Number(ov.bbox?.w ?? ov.bbox?.width);
    const hh = Number(ov.bbox?.h ?? ov.bbox?.height);
    if (!(ww >= 8 && hh >= 8)) return;
    const slotId = `ov_${String(ov.id || '')}`;
    slots.push({
      slotId,
      kind: normalizeOverlayKind(ov.kind || 'icon'),
      granularity: normalizeGranularity(ov.granularity || 'opaqueRect'),
      bbox: ov.bbox,
      fgPoints: Array.isArray(ov.fgPoints) ? ov.fgPoints : [],
      bgPoints: Array.isArray(ov.bgPoints) ? ov.bgPoints : [],
      parentNode: null,
      target: { type: 'globalOverlay', overlay: ov },
      attempts: 0
    });
  });

  return slots;
}

async function resolveOverlaysWithRetries(provider, visionUrl, firstImage, imageWidth, imageHeight, structure, prompt, options = {}) {
  const slots = collectOverlaySlots(structure, options);
  const failures = [];
  const overlayOptions = options && typeof options === 'object' ? options.overlayOptions : null;
  const textItems = options && typeof options === 'object' ? options.textItems : null;

  async function runResolveBatch(batch) {
    const payload = batch
      .filter((s) => s && s.bbox && s.granularity !== 'ignore')
      .map((s) => ({
        id: s.slotId,
        bbox: s.bbox,
        kind: s.kind,
        granularity: s.granularity,
        fgPoints: s.granularity === 'alphaMask' ? s.fgPoints : [],
        bgPoints: s.granularity === 'alphaMask' ? s.bgPoints : []
      }));
    if (payload.length === 0) return [];
    const out = await callVisionServiceResolveOverlays(visionUrl, firstImage, imageWidth, imageHeight, payload, true, overlayOptions, textItems);
    return Array.isArray(out.overlays) ? out.overlays : [];
  }

  function applyResolveResult(slot, result) {
    const ok = Boolean(result?.ok);
    const dataUrl = typeof result?.dataUrl === 'string' ? result.dataUrl : null;
    const reason = typeof result?.reason === 'string' ? result.reason : '';
    const detail = typeof result?.detail === 'string' ? result.detail : '';
    const stats = result?.stats && typeof result.stats === 'object' ? result.stats : {};
	    const debug = result?.debug && typeof result.debug === 'object' ? result.debug : null;
	    const resultBoxRaw = result?.bbox && typeof result.bbox === 'object' ? result.bbox : null;
	    let resultBox = resultBoxRaw ? clampBox(resultBoxRaw, imageWidth, imageHeight) : null;

	    slot.attempts += 1;
	    slot.last = { ok, reason, detail, stats, debug };

    const usableDataUrl = ok && dataUrl && dataUrl.startsWith('data:image/') ? dataUrl : null;

    if (slot.target.type === 'node') {
      if (ok && resultBox) {
        slot.bbox = resultBox;
        slot.target.node.bbox = resultBox;
      }
      slot.target.node.dataUrl = usableDataUrl;
      slot.target.node.overlayResult = { ok, reason, stats };
    } else if (slot.target.type === 'nodeOverlay') {
      if (ok && resultBox) {
        slot.bbox = resultBox;
        slot.target.overlay.bbox = resultBox;
      }
      slot.target.overlay.dataUrl = usableDataUrl;
      slot.target.overlay.result = { ok, reason, stats };
    } else if (slot.target.type === 'globalOverlay') {
      if (ok && resultBox) {
        slot.bbox = resultBox;
        slot.target.overlay.bbox = resultBox;
      }
      slot.target.overlay.dataUrl = usableDataUrl;
      slot.target.overlay.result = { ok, reason, stats };
    }
    return ok;
  }

  // Initial resolve for all overlays.
  const results0 = await runResolveBatch(slots);
  const byId0 = new Map(results0.map((r) => [String(r.id || ''), r]));
  slots.forEach((s) => {
    const r = byId0.get(String(s.slotId));
    if (r) applyResolveResult(s, r);
  });

  // Retry alphaMask failures with provider-assisted point refinement (max 3 attempts total).
  for (const slot of slots) {
    if (slot.granularity !== 'alphaMask') continue;
    if (slot.last?.ok) continue;

      for (let attempt = 2; attempt <= 3; attempt += 1) {
        const debug = slot.last?.debug;
        const crop = debug?.bboxCrop || debug?.crop;
        const mask = debug?.bboxMask || debug?.mask;
        if (!crop || !mask) break;

        const refinePrompt = buildOverlayRefinePrompt({
          imageWidth,
          imageHeight,
        overlayId: slot.slotId,
        kind: slot.kind,
        granularity: slot.granularity,
        bbox: slot.bbox,
        fgPoints: slot.fgPoints,
        bgPoints: slot.bgPoints,
        reason: slot.last?.reason || '',
        stats: slot.last?.stats || {},
        parent: { nodeText: slot.parentNode?.text || '' },
        prompt
      });

      let refined = null;
      try {
        const raw = await callVisionJsonWithImages(
          provider,
          refinePrompt,
          'Return ONLY the JSON object.',
          [
            { dataUrl: crop, type: 'image/png' },
            { dataUrl: mask, type: 'image/png' }
          ],
          { maxTokens: 2048, timeoutMs: 120000 }
        );
        refined = parseJsonFromModel(raw);
      } catch (err) {
        logError('Overlay refine provider call failed', err, { overlayId: slot.slotId });
        refined = null;
      }
      if (!refined || typeof refined !== 'object') break;

      let newGran = normalizeGranularity(refined.granularity || slot.granularity);
      let newBox = refined.bbox && typeof refined.bbox === 'object' ? clampBox(refined.bbox, imageWidth, imageHeight) : slot.bbox;
      if (slot.target.type === 'nodeOverlay') {
        newBox = clampBoxWithin(newBox, slot.parentNode?.bbox || newBox);
      }
      const newFg = normalizePoints(refined.fgPoints, imageWidth, imageHeight, 12);
      const newBg = normalizePoints(refined.bgPoints, imageWidth, imageHeight, 16);

      // Do not degrade icon/3d overlays to opaqueRect/ignore during refinement.
      if (slot.kind === 'icon' || slot.kind === '3d') {
        newGran = 'alphaMask';
      } else {
        newGran = defaultGranularityForKind(slot.kind || 'photo');
      }

      slot.granularity = newGran;
      slot.bbox = newBox;
      slot.fgPoints = newGran === 'alphaMask' ? newFg : [];
      slot.bgPoints = newGran === 'alphaMask' ? newBg : [];

      // Apply changes back to structure objects.
      if (slot.target.type === 'node') {
        slot.target.node.bbox = newBox;
        slot.target.node.overlay = { ...(slot.target.node.overlay || {}), granularity: newGran, fgPoints: slot.fgPoints, bgPoints: slot.bgPoints };
      } else if (slot.target.type === 'nodeOverlay') {
        slot.target.overlay.bbox = newBox;
        slot.target.overlay.granularity = newGran;
        slot.target.overlay.fgPoints = slot.fgPoints;
        slot.target.overlay.bgPoints = slot.bgPoints;
      } else if (slot.target.type === 'globalOverlay') {
        slot.target.overlay.bbox = newBox;
        slot.target.overlay.granularity = newGran;
        slot.target.overlay.fgPoints = slot.fgPoints;
        slot.target.overlay.bgPoints = slot.bgPoints;
      }

      if (newGran === 'ignore') {
        slot.last = { ok: false, reason: 'ignored', stats: {}, debug: null };
        break;
      }

      // Re-run resolve for this slot only.
      const resultsN = await runResolveBatch([slot]);
      const r = resultsN[0];
      if (r) {
        const ok = applyResolveResult(slot, r);
        if (ok) break;
      }
    }

  }

  slots.forEach((slot) => {
    if (!slot || slot.granularity === 'ignore') return;
    if (slot.last?.ok) return;
    failures.push({
      id: slot.slotId,
      kind: slot.kind,
      granularity: slot.granularity,
      reason: slot.last?.reason || 'failed',
      detail: slot.last?.detail || '',
      stats: slot.last?.stats || {},
      bbox: slot.bbox
    });
  });

  return { failures };
}

async function callCandidateLabel(provider, prompt, images, imageWidth, imageHeight, candidates) {
  const normalized = normalizeProviderForVision(provider);
  if (!normalized) throw new Error('Provider not configured');

  const instruction = buildCandidateLabelInstruction(imageWidth, imageHeight, candidates);
  const userPrompt = String(prompt || '').trim();
  const combined = userPrompt ? `${instruction}\n\nUser request:\n${userPrompt}` : instruction;

  const imgParts = extractImageParts(images);
  if (imgParts.length === 0) {
    throw new Error('Reference image is required.');
  }

  if (normalized.type === 'ollama') {
    throw new Error('This provider does not support vision/image input for candidate labeling.');
  }

  if (normalized.type === 'gemini') {
    const baseUrl = trimBaseUrl(normalized.baseUrl || providerBaseUrls.gemini);
    const url = `${baseUrl}/models/${normalized.model}:generateContent?key=${normalized.apiKey}`;
    const parts = [{ text: combined }];
    imgParts.forEach((img) => {
      const [header, data] = img.dataUrl.split(',');
      const mime = header.match(/data:(.*?);base64/)?.[1] || img.type || 'image/png';
      parts.push({ inline_data: { mime_type: mime, data } });
    });

    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ contents: [{ role: 'user', parts }], generationConfig: { temperature: 0.1, maxOutputTokens: 4096 } })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error?.message || 'Provider error');
    const partsOut = data.candidates?.[0]?.content?.parts || [];
    return partsOut.map((part) => part.text || '').join('').trim();
  }

  if (normalized.type === 'anthropic') {
    const baseUrl = trimBaseUrl(normalized.baseUrl || providerBaseUrls.anthropic);
    const url = `${baseUrl}/messages`;
    const content = [{ type: 'text', text: combined }];
    imgParts.forEach((img) => {
      const [header, data] = img.dataUrl.split(',');
      const mime = header.match(/data:(.*?);base64/)?.[1] || img.type || 'image/png';
      content.push({ type: 'image', source: { type: 'base64', media_type: mime, data } });
    });

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'x-api-key': normalized.apiKey || '',
        'anthropic-version': '2023-06-01',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ model: normalized.model, max_tokens: 4096, messages: [{ role: 'user', content }] })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error?.message || 'Provider error');
    return data.content && data.content[0] ? data.content[0].text : '';
  }

  const baseUrl = trimBaseUrl(normalized.baseUrl);
  const url = `${baseUrl}/chat/completions`;
  const messages = [
    { role: 'system', content: combined },
    {
      role: 'user',
      content: [
        { type: 'text', text: 'Return only the JSON object.' },
        ...imgParts.map((img) => ({ type: 'image_url', image_url: { url: img.publicUrl || img.dataUrl } }))
      ]
    }
  ];

  const headers = {
    Authorization: `Bearer ${normalized.apiKey || ''}`,
    'Content-Type': 'application/json'
  };

  const body = { model: normalized.model, messages, temperature: 0.1, max_tokens: 4096 };
  if (shouldStreamChatRequest(normalized.model, normalized.baseUrl)) body.stream = true;

  const { text, data } = await fetchChatCompletionsJson(
    url,
    body,
    headers,
    { maxAttempts: 3, timeoutMs: 120000 }
  );

  if (!data) return String(text || '').trim();
  const contentOut = data.choices?.[0]?.message?.content || '';
  return contentOut;
}

async function callVisionServiceStructure(serviceUrl, prompt, image, imageWidth, imageHeight, textItems, options) {
  const url = `${String(serviceUrl || '').replace(/\/$/, '')}/analyze`;
  const payload = {
    prompt: String(prompt || ''),
    image,
    imageWidth: Number(imageWidth || 0),
    imageHeight: Number(imageHeight || 0),
    ...(Array.isArray(textItems) && textItems.length ? { textItems } : {}),
    ...(options && typeof options === 'object' ? { options } : {})
  };

  const response = await fetchWithTimeout(
    url,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    },
    120000
  );

  const text = await response.text();
  let data = null;
  try {
    data = JSON.parse(text);
  } catch (err) {
    data = null;
  }
  if (!response.ok) {
    throw buildHttpError(response, text, data, url);
  }
  if (!data || typeof data !== 'object' || typeof data.structure !== 'object') {
    const err = new Error('Vision service returned invalid response.');
    err.http = { url, status: response.status, statusText: response.statusText, bodySnippet: text.slice(0, 2000), error: data };
    throw err;
  }
  return data;
}

async function callVisionServiceAugment(serviceUrl, image, imageWidth, imageHeight, nodes, textItems, options) {
  const url = `${String(serviceUrl || '').replace(/\/$/, '')}/augment`;
  const payload = {
    image,
    imageWidth: Number(imageWidth || 0),
    imageHeight: Number(imageHeight || 0),
    nodes: Array.isArray(nodes) ? nodes : [],
    ...(Array.isArray(textItems) && textItems.length ? { textItems } : {}),
    ...(options && typeof options === 'object' ? { options } : {})
  };

  const response = await fetchWithTimeout(
    url,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    },
    120000
  );

  const text = await response.text();
  let data = null;
  try {
    data = JSON.parse(text);
  } catch (err) {
    data = null;
  }
  if (!response.ok) {
    throw buildHttpError(response, text, data, url);
  }
  if (!data || typeof data !== 'object' || !Array.isArray(data.nodes)) {
    const err = new Error('Vision service /augment returned invalid response.');
    err.http = { url, status: response.status, statusText: response.statusText, bodySnippet: text.slice(0, 2000), error: data };
    throw err;
  }
  return data;
}

async function callVisionServiceAnnotateBoxes(serviceUrl, image, imageWidth, imageHeight, nodes, overlays) {
  const url = `${String(serviceUrl || '').replace(/\/$/, '')}/debug/annotate_boxes`;
  const payload = {
    image,
    imageWidth: Number(imageWidth || 0),
    imageHeight: Number(imageHeight || 0),
    nodes: Array.isArray(nodes) ? nodes : [],
    overlays: Array.isArray(overlays) ? overlays : []
  };

  const response = await fetchWithTimeout(
    url,
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) },
    120000
  );
  const text = await response.text();
  let data = null;
  try {
    data = JSON.parse(text);
  } catch (err) {
    data = null;
  }
  if (!response.ok) {
    throw buildHttpError(response, text, data, url);
  }
  if (!data || typeof data !== 'object' || typeof data.annotated !== 'string') {
    const err = new Error('Vision service /debug/annotate_boxes returned invalid response.');
    err.http = { url, status: response.status, statusText: response.statusText, bodySnippet: text.slice(0, 2000), error: data };
    throw err;
  }
  return data;
}

async function callVisionServiceContactSheet(serviceUrl, image, imageWidth, imageHeight, items, options) {
  const url = `${String(serviceUrl || '').replace(/\/$/, '')}/debug/contact_sheet`;
  const payload = {
    image,
    imageWidth: Number(imageWidth || 0),
    imageHeight: Number(imageHeight || 0),
    items: Array.isArray(items) ? items : [],
    ...(options && typeof options === 'object' ? { options } : {})
  };

  const response = await fetchWithTimeout(
    url,
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) },
    180000
  );
  const text = await response.text();
  let data = null;
  try {
    data = JSON.parse(text);
  } catch (err) {
    data = null;
  }
  if (!response.ok) {
    throw buildHttpError(response, text, data, url);
  }
  if (!data || typeof data !== 'object' || typeof data.image !== 'string') {
    const err = new Error('Vision service /debug/contact_sheet returned invalid response.');
    err.http = { url, status: response.status, statusText: response.statusText, bodySnippet: text.slice(0, 2000), error: data };
    throw err;
  }
  return data;
}

async function callVisionServiceResolveOverlays(serviceUrl, image, imageWidth, imageHeight, overlays, debug, options, textItems) {
  const url = `${String(serviceUrl || '').replace(/\/$/, '')}/overlays/resolve`;
  const payload = {
    image,
    imageWidth: Number(imageWidth || 0),
    imageHeight: Number(imageHeight || 0),
    overlays: Array.isArray(overlays) ? overlays : [],
    debug: Boolean(debug),
    ...(options && typeof options === 'object' ? { options } : {}),
    ...(Array.isArray(textItems) && textItems.length ? { textItems } : {})
  };

  const response = await fetchWithTimeout(
    url,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    },
    180000
  );

  const text = await response.text();
  let data = null;
  try {
    data = JSON.parse(text);
  } catch (err) {
    data = null;
  }
  if (!response.ok) {
    throw buildHttpError(response, text, data, url);
  }
  if (!data || typeof data !== 'object' || !Array.isArray(data.overlays)) {
    const err = new Error('Vision service /overlays/resolve returned invalid response.');
    err.http = { url, status: response.status, statusText: response.statusText, bodySnippet: text.slice(0, 2000), error: data };
    throw err;
  }
  return data;
}

async function resolveImageDimsViaVisionService(serviceUrl, image) {
  if (!serviceUrl || !image || typeof image !== 'object' || !image.dataUrl) return null;
  try {
    const out = await callVisionServiceResolveOverlays(serviceUrl, image, 0, 0, [], false);
    const w = Number(out?.meta?.imageWidth);
    const h = Number(out?.meta?.imageHeight);
    if (w > 0 && h > 0) return { imageWidth: w, imageHeight: h };
  } catch (err) {
    // ignore
  }
  return null;
}

function providerCanAttemptImages(provider) {
  const type = String(provider?.type || '');
  // Ollama route in this server doesn't send images.
  if (type === 'ollama') return false;
  // For everything else, try to send images and fall back if the provider rejects.
  return true;
}

function shouldFallbackWithoutImages(errorMessage) {
  const msg = String(errorMessage || '').toLowerCase();
  if (!msg) return false;

  // Likely "vision not supported" / schema mismatch cases.
  if (msg.includes('input should be a valid string')) return true;
  if (msg.includes('image_url') && msg.includes('input should be a valid')) return true;
  if (msg.includes('does not support') && msg.includes('image')) return true;
  if (msg.includes('unsupported') && msg.includes('image')) return true;
  if (msg.includes('invalid') && msg.includes('image_url')) return true;

  // If the provider can parse the request but fails to decode/process the image,
  // do NOT silently fall back (users expect the image to be used).
  if (msg.includes('unable to process input image')) return false;

  // Default: don't hide failures.
  return false;
}

app.get('/api/providers', (req, res) => {
  const config = readProviderConfig();
  res.json({ primary: config.primary, providers: listProviders(config) });
});

app.post('/api/providers', (req, res) => {
  const payload = req.body || {};
  const config = readProviderConfig();

  if (payload.primary && providerCatalogByType[payload.primary]) {
    config.primary = payload.primary;
  }

  if (payload.type) {
    const type = payload.type;
    const meta = providerCatalogByType[type];
    if (!meta) {
      logApiResponseError(req, 400, 'Unknown provider type.', { type });
      res.status(400).json({ error: 'Unknown provider type.' });
      return;
    }

    const existing = config.providers[type] || { type };
    const next = {
      ...existing,
      type,
      model: String(payload.model || '').trim() || existing.model || meta.defaultModel || '',
      plannerModel: String(payload.plannerModel || '').trim() || existing.plannerModel || meta.defaultPlannerModel || '',
      imageModel: String(payload.imageModel || '').trim() || existing.imageModel || '',
      baseUrl: String(payload.baseUrl || '').trim() || existing.baseUrl || meta.defaultBase || '',
      apiKey: String(payload.apiKey || '').trim() ? String(payload.apiKey).trim() : existing.apiKey || ''
    };

    if (!next.model) {
      logApiResponseError(req, 400, 'Model is required.', { type });
      res.status(400).json({ error: 'Model is required.' });
      return;
    }
    if (meta.requiresBase && !next.baseUrl) {
      logApiResponseError(req, 400, 'Base URL is required for this provider type.', { type });
      res.status(400).json({ error: 'Base URL is required for this provider type.' });
      return;
    }

    config.providers[type] = next;
  }

  try {
    writeProviderConfig(config);
    res.json({ primary: config.primary, providers: listProviders(config) });
  } catch (err) {
    logError('Failed to save provider config', err);
    res.status(500).json({ error: 'Failed to save provider.' });
  }
});

app.post('/api/image/generate', async (req, res) => {
  const config = readProviderConfig();
  const provider = resolveProvider(req.body, config);
  if (!requireConfiguredProvider(req, res, config, provider)) return;

  const allow = req.body?.allowImageModel === true || req.body?.allowImageModel === 1 || req.body?.allowImageModel === '1';
  if (!allow) {
    logApiResponseError(req, 400, 'Image generation disabled by client setting.', { providerType: provider?.type || '' });
    res.status(400).json({ error: 'Image generation is disabled by client setting.', code: 'image_model_disabled' });
    return;
  }

  const prompt = String(req.body?.prompt || '').trim();
  if (!prompt) {
    logApiResponseError(req, 400, 'Prompt is required.', {});
    res.status(400).json({ error: 'Prompt is required.' });
    return;
  }

  if (!providerSupportsImageGeneration(provider)) {
    logApiResponseError(req, 400, 'Image generation is not supported for this provider type.', { providerType: provider?.type || '' });
    res.status(400).json({ error: 'Image generation is not supported for this provider type.' });
    return;
  }

  if (!String(provider?.imageModel || '').trim()) {
    logApiResponseError(req, 400, 'Image model is not configured.', { providerType: provider?.type || '' });
    res.status(400).json({ error: 'Image model is not configured. Set it in Settings.', code: 'image_model_missing' });
    return;
  }

  try {
    const images = Array.isArray(req.body?.images) ? req.body.images : [];
    const dataUrl = await callOpenAICompatibleImageModel(provider, prompt, images);
    res.json({
      image: {
        name: 'generated.png',
        type: 'image/png',
        dataUrl
      }
    });
  } catch (err) {
    if (err && typeof err === 'object' && !err.phase) err.phase = 'image_generate';
    const out = buildClientErrorPayload(err, 'Provider error');
    logError('API /api/image/generate failed', err, {
      providerType: req.body?.providerType || req.body?.providerName || config.primary,
      provider: provider ? { type: provider.type, model: provider.model, imageModel: provider.imageModel, baseUrl: provider.baseUrl } : null,
      promptLen: typeof req.body?.prompt === 'string' ? req.body.prompt.length : 0,
      imagesCount: Array.isArray(req.body?.images) ? req.body.images.length : 0,
      providerHttp: err?.http || null,
      phase: out.body?.phase || null,
      code: out.body?.code || err?.code || null
    });
    res.status(out.status).json(out.body);
  }
});

app.post('/api/flow/:format', async (req, res) => {
  const format = req.params.format;
  if (format !== 'json' && format !== 'xml') {
    logApiResponseError(req, 400, 'Format must be json or xml.', { format });
    res.status(400).json({ error: 'Format must be json or xml.' });
    return;
  }

  const config = readProviderConfig();
  const provider = resolveProvider(req.body, config);
  if (!requireConfiguredProvider(req, res, config, provider)) return;

  try {
    const images = Array.isArray(req.body.images) ? req.body.images : [];
    let warning = null;
    let warningCode = null;
    let imagesToUse = images;
    const rawPrompt = String(req.body.prompt || '');
    let promptToUse = rawPrompt;

    if (images.length > 0 && !providerCanAttemptImages(provider)) {
      imagesToUse = [];
      warningCode = 'images_not_supported';
      warning = 'This provider/model does not support image input; generated without images.';
    }

    if (imagesToUse.length > 0 && format === 'xml') {
      const first = imagesToUse[0] || {};
      const w = Number(first.width || 0);
      const h = Number(first.height || 0);
      const sizeHint = Number.isFinite(w) && Number.isFinite(h) && w > 0 && h > 0 ? `Reference image size: ${w}x${h} pixels.` : '';
      promptToUse = [
        rawPrompt,
        '',
        'When generating the mxGraph XML, replicate the reference image layout as closely as possible.',
        sizeHint,
        'Use the reference image pixel coordinate system for mxGeometry x/y/width/height (origin at top-left).',
        'Keep relative positions and arrow directions consistent with the reference image; do NOT reorganize the structure.',
        'Do not add extra decorative elements that are not present in the reference.'
      ]
        .filter(Boolean)
        .join('\n');
    }

    let output;
    try {
      output = await callProvider(provider, promptToUse, imagesToUse, format);
    } catch (err) {
      if (imagesToUse.length > 0) {
        const detail = String(err?.message || '').slice(0, 500);
        if (shouldFallbackWithoutImages(detail)) {
          warningCode = 'images_rejected';
          warning = 'Provider rejected image input; generated without images.';
          if (detail) {
            warning = `${warning} (${detail})`;
          }
          output = await callProvider(provider, promptToUse, [], format);
        } else {
          throw err;
        }
      } else {
        throw err;
      }
    }

    res.json({ output, warning, warningCode });
  } catch (err) {
    const config = readProviderConfig();
    const provider = resolveProvider(req.body, config);
    if (err && typeof err === 'object' && !err.phase) err.phase = 'flow_generate';
    const out = buildClientErrorPayload(err, 'Provider error');
    logError('API /api/flow failed', err, {
      format: req.params.format,
      providerType: req.body?.providerType || req.body?.providerName || config.primary,
      provider: provider ? { type: provider.type, model: provider.model, baseUrl: provider.baseUrl } : null,
      promptLen: typeof req.body?.prompt === 'string' ? req.body.prompt.length : 0,
      imagesCount: Array.isArray(req.body?.images) ? req.body.images.length : 0,
      providerHttp: err?.http || null,
      phase: out.body?.phase || null,
      code: out.body?.code || null
    });
    res.status(out.status).json(out.body);
  }
});

app.post('/api/vision/structure', async (req, res) => {
  const config = readProviderConfig();
  const provider = resolveProvider(req.body, config);
  if (!requireConfiguredProvider(req, res, config, provider)) return;

  const visionUrl = await ensureVisionServiceUrl((msg) => console.log(`[${nowIso()}] ${msg}`));
  if (!visionUrl) {
    logApiResponseError(req, 400, 'Vision service not available (SAM2 required).', {});
    res.status(400).json({
      error:
        'Vision service not available. SAM2 is required for high-quality overlay extraction. Install vision_service deps and start it (or set VISION_SERVICE_URL).'
    });
    return;
  }

  const images = Array.isArray(req.body.images) ? req.body.images : [];
  const first = images[0] || null;
  if (!first || !first.dataUrl) {
    logApiResponseError(req, 400, 'Reference image is required.', { imagesCount: images.length });
    res.status(400).json({ error: 'Reference image is required.' });
    return;
  }

  let imageWidth = Number(req.body.imageWidth || first.width || 0);
  let imageHeight = Number(req.body.imageHeight || first.height || 0);
  if (!(imageWidth > 0 && imageHeight > 0)) {
    const dims = await resolveImageDimsViaVisionService(visionUrl, first);
    if (dims) {
      imageWidth = dims.imageWidth;
      imageHeight = dims.imageHeight;
    }
  }
  if (!(imageWidth > 0 && imageHeight > 0)) {
    logApiResponseError(req, 400, 'Image width/height required.', { imageWidth, imageHeight });
    res.status(400).json({ error: 'Image width/height required.' });
    return;
  }

  let catalog;
  try {
    catalog = await ensureShapeCatalog((msg) => console.log(`[${nowIso()}] ${msg}`));
  } catch (err) {
    logError('Failed to build/load shape catalog', err);
    res.status(500).json({ error: 'Failed to build shape catalog.' });
    return;
  }

  const prompt = String(req.body.prompt || '').trim();
  const instruction = buildStructureInstruction(imageWidth, imageHeight, SHAPE_CONFIDENCE_THRESHOLD);
  const qualityMode = String(req.body?.qualityMode || '').toLowerCase() === 'balanced' ? 'balanced' : 'max';
  const visionOptions = { qualityMode };

  try {
    let textItems = [];
    let textExtractError = null;

    // In max-quality mode, prefer local OCR (PaddleOCR) from the vision service so the pipeline can
    // still work even if the configured provider OCR is unavailable or misconfigured.
    let cvOutForText = null;
    if (qualityMode === 'max') {
      try {
        const cvTry = await callVisionServiceStructure(visionUrl, prompt, first, imageWidth, imageHeight, null, visionOptions);
        const ti = Array.isArray(cvTry?.meta?.textItems) ? cvTry.meta.textItems : [];
        if (ti.length) {
          textItems = ti;
          // Only reuse this CV pass when it was computed with usable OCR (node detection benefits from text masks).
          cvOutForText = cvTry;
        }
      } catch (err) {
        logError('Vision service /analyze failed (max mode, local OCR unavailable)', err, { visionUrl });
      }
    }

    if (!textItems.length) {
      const out = await extractTextItemsBestEffort(provider, first, imageWidth, imageHeight, {
        providerType: req.body?.providerType || req.body?.providerName || config.primary,
        provider: provider ? { type: provider.type, model: provider.model, baseUrl: provider.baseUrl } : null,
        imageWidth,
        imageHeight
      });
      textItems = Array.isArray(out.textItems) ? out.textItems : [];
      textExtractError = out.textExtractError || null;
    }

    // Prefer CV (route-1) for paper-figure screenshots; fall back to provider-only extraction if CV is weak.
    let parsedBase = null;
    let baseBackend = 'provider+sam2';

	    try {
	      const cvOut =
	        cvOutForText ||
	        (await callVisionServiceStructure(
	          visionUrl,
	          prompt,
	          first,
	          imageWidth,
	          imageHeight,
	          textItems && textItems.length ? textItems : null,
	          visionOptions
	        ));
	      const cvStruct = cvOut && typeof cvOut === 'object' ? cvOut.structure : null;
	      const cvNodes = Array.isArray(cvStruct?.nodes) ? cvStruct.nodes : [];
	      const cvEdges = Array.isArray(cvStruct?.edges) ? cvStruct.edges : [];
	      if (!textItems.length) {
	        const cvText = Array.isArray(cvOut?.meta?.textItems) ? cvOut.meta.textItems : [];
	        if (cvText.length) textItems = cvText;
	      }
	      const assessment = assessCvStructure(cvStruct, imageWidth, imageHeight, textItems);
	      if (assessment.ok) {
	        parsedBase = cvStruct;
	        baseBackend = String(cvOut?.meta?.backend || 'cv-route1');
	        try {
	          console.log(`[${nowIso()}] CV route accepted (${baseBackend}): ${JSON.stringify(assessment.metrics)}`);
	        } catch (err) {
	          // ignore
	        }
	      } else {
	        try {
	          console.log(`[${nowIso()}] CV route rejected: ${JSON.stringify(assessment.metrics)}`);
	        } catch (err) {
	          // ignore
	        }
	      }
	    } catch (err) {
	      logError('Vision service /analyze failed (falling back to provider structure)', err, { visionUrl });
	      parsedBase = null;
	    }

    if (!parsedBase) {
      const parsedV2 = await extractStructureV2(provider, prompt, first, imageWidth, imageHeight);
      parsedBase = parsedV2;
      baseBackend = 'provider-structure';
    }

    const normalizedV2 = await normalizeStructureFromModel(parsedBase, imageWidth, imageHeight, catalog);
    const unassignedTextItems = applyTextItemsToNodes(normalizedV2, textItems, imageWidth, imageHeight);

    // Optional: CV augment for container colors and simple inner bars (vectorizable).
    let cvEdges = null;
    try {
      const aug = await callVisionServiceAugment(
        visionUrl,
        first,
        imageWidth,
        imageHeight,
        normalizedV2.nodes.map((n) => ({ id: n.id, bbox: n.bbox, textBbox: n.textBbox })),
        textItems,
        visionOptions
      );
      cvEdges = Array.isArray(aug.edges) ? aug.edges : null;
      const byId = new Map((Array.isArray(aug.nodes) ? aug.nodes : []).map((n) => [String(n.id), n]));
      normalizedV2.nodes.forEach((n) => {
        const a = byId.get(String(n.id));
        if (!a || typeof a !== 'object') return;
        if (a.containerStyle && typeof a.containerStyle === 'object') n.containerStyle = a.containerStyle;
        if (Array.isArray(a.innerShapes)) {
          n.innerShapes = a.innerShapes
            .filter((s) => s && typeof s === 'object' && (s.bbox || s.geometry))
            .slice(0, 30)
            .map((s) => {
              const bb = clampBox(s.bbox || s.geometry, imageWidth, imageHeight);
              const style = s.style && typeof s.style === 'object' ? s.style : {};
              return {
                bbox: bb,
                shapeId: normalizeShapeId(String(s.shapeId || s.shape || '').trim()),
                style,
                confidence: typeof s.confidence === 'number' ? Math.max(0, Math.min(1, s.confidence)) : 0.65
              };
            });
        }

        if (Array.isArray(a.nodeOverlays)) {
          const existing = Array.isArray(n.nodeOverlays) ? n.nodeOverlays : [];
          const merged = [...existing];
          const idSet = new Set(merged.map((ov) => String(ov?.id || '')));

          function uniqueOverlayId(base) {
            const raw = String(base || '').trim() || `cv_${idSet.size + 1}`;
            let next = raw;
            let i = 2;
            while (idSet.has(next)) {
              next = `${raw}_${i}`;
              i += 1;
            }
            idSet.add(next);
            return next;
          }

           const candidates = a.nodeOverlays
             .filter((ov) => ov && typeof ov === 'object' && (ov.bbox || ov.geometry || (ov.x !== undefined && ov.y !== undefined)))
             .slice(0, 24)
             .map((ov, idx) => {
               const bb = clampBox(ov.bbox || ov.geometry || ov, imageWidth, imageHeight);
               const bbIn = clampBoxWithin(bb, n.bbox);
               const kind = normalizeOverlayKind(ov.kind || ov.type || 'icon');
               const granularity =
                 kind === 'photo' || kind === 'chart' || kind === 'plot' || kind === 'noise' || kind === 'screenshot'
                   ? 'opaqueRect'
                   : 'alphaMask';
               const dataUrl = typeof ov.dataUrl === 'string' ? ov.dataUrl : null;
               const fg = normalizePoints(ov.fgPoints || ov.fg || ov.foregroundPoints, imageWidth, imageHeight, 12);
               const bg = normalizePoints(ov.bgPoints || ov.bg || ov.backgroundPoints, imageWidth, imageHeight, 16);
               const conf = Number(ov.confidence);
               return {
                 id: uniqueOverlayId(String(ov.id || `cv_${idx + 1}`)),
                 kind,
                 granularity,
                 bbox: bbIn,
                 fgPoints: granularity === 'alphaMask' ? fg : [],
                 bgPoints: granularity === 'alphaMask' ? bg : [],
                 confidence: Number.isFinite(conf) ? Math.max(0, Math.min(1, conf)) : 0.7,
                 dataUrl: dataUrl && dataUrl.startsWith('data:image/') ? dataUrl : null
               };
             });

          candidates.forEach((cand) => {
            const bb = cand?.bbox;
            const ww = Number(bb?.w ?? bb?.width);
            const hh = Number(bb?.h ?? bb?.height);
            if (!(ww >= 10 && hh >= 10)) return;
            const dupIdx = merged.findIndex((m) => boxIou(m?.bbox, cand.bbox) >= 0.72);
            if (dupIdx >= 0) {
              const existingOv = merged[dupIdx];
              if (existingOv && typeof existingOv === 'object') {
                // Prefer resolved overlays (dataUrl) and tighter bboxes from vision service.
                if (!existingOv.dataUrl && cand.dataUrl) existingOv.dataUrl = cand.dataUrl;
                existingOv.kind = cand.kind || existingOv.kind;
                existingOv.granularity = cand.granularity || existingOv.granularity;
                existingOv.bbox = cand.bbox || existingOv.bbox;
                if (Array.isArray(cand.fgPoints) && cand.fgPoints.length) existingOv.fgPoints = cand.fgPoints;
                if (Array.isArray(cand.bgPoints) && cand.bgPoints.length) existingOv.bgPoints = cand.bgPoints;
                if (typeof cand.confidence === 'number') existingOv.confidence = cand.confidence;
              }
              return;
            }
            merged.push(cand);
          });

          n.nodeOverlays = merged.slice(0, 80);
        }
      });
    } catch (err) {
      logError('Vision augment failed (continuing without CV colors/innerShapes)', err, { visionUrl });
    }

    // Prefer CV-extracted blue connectors when available; it is usually more accurate than LLM-inferred edges.
    if (Array.isArray(cvEdges) && cvEdges.length >= 2) {
      const nodeIdSet = new Set(normalizedV2.nodes.map((n) => String(n.id)));
      const seenPairs = new Set();
      const out = [];
      cvEdges.forEach((e, idx) => {
        if (!e || typeof e !== 'object') return;
        const source = String(e.source || '');
        const target = String(e.target || '');
        if (!nodeIdSet.has(source) || !nodeIdSet.has(target)) return;
        const key = `${source}=>${target}`;
        if (seenPairs.has(key)) return;
        seenPairs.add(key);
        const conf = Number(e.confidence);
        out.push({
          id: `cv_e${idx + 1}`,
          source,
          target,
          sourceSide: normalizeSide(e.sourceSide),
          targetSide: normalizeSide(e.targetSide),
          label: typeof e.label === 'string' ? e.label : '',
          confidence: Number.isFinite(conf) ? Math.max(0, Math.min(1, conf)) : 0.65
        });
      });
      if (out.length >= 2) normalizedV2.edges = out;
    }

    // Overlay pipeline: AMG candidates -> rule denoise -> Gemini semantic plan -> SAM2 predictor resolve.
    try {
      const plannerCandidates = [];
      const fallback = { finalOverlays: [] };

      normalizedV2.nodes
        .filter(
          (n) =>
            n &&
            typeof n === 'object' &&
            n.bbox &&
            String(n.render || '').toLowerCase() !== 'text' &&
            String(n.render || '').toLowerCase() !== 'overlay'
        )
        .forEach((n) => {
          const nodeId = String(n.id || '');
          const raw = Array.isArray(n.nodeOverlays) ? n.nodeOverlays : [];
          const candRaw = raw
            .filter((ov) => ov && typeof ov === 'object' && ov.bbox)
            .map((ov, idx) => {
              const kindGuess = normalizeOverlayKind(ov.kind || 'icon');
              const granularityGuess = defaultGranularityForKind(kindGuess);
              const bb = clampBoxWithin(clampBox(ov.bbox, imageWidth, imageHeight), n.bbox);
              const conf = Number(ov.confidence);
              const seeds =
                granularityGuess === 'alphaMask'
                  ? ensureOverlaySeedPoints(ov.fgPoints, ov.bgPoints, bb, imageWidth, imageHeight)
                  : { fg: [], bg: [] };
              return {
                id: `cand_${nodeId}_${safeOverlayId(ov.id || `o${idx + 1}`, `o${idx + 1}`)}`,
                nodeId,
                bbox: bb,
                kindGuess,
                granularityGuess,
                confidence: Number.isFinite(conf) ? Math.max(0, Math.min(1, conf)) : 0.7,
                fgPointsSeed: seeds.fg,
                bgPointsSeed: seeds.bg,
                previewDataUrl: typeof ov.dataUrl === 'string' && ov.dataUrl.startsWith('data:image/') ? ov.dataUrl : null
              };
            });

          const denoised = denoiseOverlayCandidatesForNode(n, candRaw);
          denoised.forEach((c) => {
            plannerCandidates.push(c);
            const seedFg = c.granularityGuess === 'alphaMask' ? (Array.isArray(c.fgPointsSeed) ? c.fgPointsSeed : []) : [];
            const seedBg = c.granularityGuess === 'alphaMask' ? (Array.isArray(c.bgPointsSeed) ? c.bgPointsSeed : []) : [];
            fallback.finalOverlays.push({
              id: c.id,
              nodeId: c.nodeId,
              kind: c.kindGuess,
              granularity: c.granularityGuess,
              bbox: c.bbox,
              sourceCandidateIds: [c.id],
              fgPoints: seedFg,
              bgPoints: seedBg,
              confidence: c.confidence
            });
          });
        });

      // Apply denoised fallback first (reduces clutter even if the semantic planner fails).
      if (fallback.finalOverlays.length > 0) {
        applyOverlaySemanticPlanToStructure(normalizedV2, fallback, imageWidth, imageHeight);
        baseBackend = `${baseBackend}+overlay-denoise`;
      }

      if (plannerCandidates.length > 0) {
        const planned = await callOverlaySemanticPlanner(provider, prompt, first, imageWidth, imageHeight, normalizedV2.nodes, plannerCandidates, visionUrl);
        if (planned && Array.isArray(planned.finalOverlays) && planned.finalOverlays.length > 0) {
          const candidatesById = new Map(plannerCandidates.map((c) => [String(c?.id || ''), c]));
          planned.finalOverlays = fillOverlayPlanPointsFromCandidates(planned.finalOverlays, candidatesById, imageWidth, imageHeight);
          const applied = applyOverlaySemanticPlanToStructure(normalizedV2, planned, imageWidth, imageHeight);
          if (applied.planned > 0) {
            baseBackend = `${baseBackend}+overlay-semantic`;
          }
        }
      }
    } catch (err) {
      logError('Overlay semantic planner failed (continuing with denoised candidates)', err, {
        providerType: provider?.type || req.body?.providerType || config.primary,
        imageWidth,
        imageHeight,
        providerHttp: err?.http || null
      });
    }

    const overlayOptions = normalizeOverlayResolveOptions(req.body?.overlayOptions);
    const overlayOut = await resolveOverlaysWithRetries(provider, visionUrl, first, imageWidth, imageHeight, normalizedV2, prompt, {
      onlyMissing: true,
      overlayOptions,
      textItems
    });

    appendTextNodes(normalizedV2, unassignedTextItems, imageWidth, imageHeight);

    res.json({
      structure: normalizedV2,
      meta: {
        imageWidth,
        imageHeight,
        backend: baseBackend,
        overlayFailures: overlayOut.failures || [],
        textItems,
        ...(textExtractError ? { textExtractError } : {}),
        shapeCatalog: {
          count: catalog.set?.size || catalog.meta?.count || 0,
          stencilsBase: catalog.stencilsBase || catalog.meta?.source?.stencilsBase || ''
        }
      }
    });
    return;

    // Route-1: Use local vision service when configured.
    if (visionUrl) {
      const out = await callVisionServiceStructure(visionUrl, prompt, first, imageWidth, imageHeight, null, { qualityMode: 'balanced' });
      const parsed = out.structure;
      // Continue through the same validation/shape-catalog gating below by assigning `parsed`.

      const nodesIn = Array.isArray(parsed.nodes) ? parsed.nodes : [];
      const edgesIn = Array.isArray(parsed.edges) ? parsed.edges : [];
      const overlaysIn = Array.isArray(parsed.overlays) ? parsed.overlays : [];

      const nodeIds = new Set();
      const nodes = [];

      for (let idx = 0; idx < nodesIn.length; idx += 1) {
        const n = nodesIn[idx];
        if (!n || typeof n !== 'object') continue;

        const id = String(n.id || `n${idx + 1}`);
        nodeIds.add(id);
        const bbox = clampBox(n.bbox || n, imageWidth, imageHeight);
        const text = typeof n.text === 'string' ? n.text : '';
        const rawShapeId = normalizeShapeId(String(n.shapeId || n.shape || '').trim());

        const conf = n.confidence && typeof n.confidence === 'object' ? n.confidence : {};
        const bboxC = Number(conf.bbox);
        const textC = Number(conf.text);
        const shapeC = Number(conf.shape);
        const confidence = {
          bbox: Number.isFinite(bboxC) ? Math.max(0, Math.min(1, bboxC)) : 0.7,
          text: Number.isFinite(textC) ? Math.max(0, Math.min(1, textC)) : 0.7,
          shape: Number.isFinite(shapeC) ? Math.max(0, Math.min(1, shapeC)) : 0.5
        };
        const shapeLow = confidence.shape < SHAPE_CONFIDENCE_THRESHOLD;

        let render = String(n.render || '').toLowerCase();
        if (render !== 'shape' && render !== 'overlay') render = 'shape';

        let supported = false;
        let shapeId = rawShapeId;

        if (render === 'shape' && !shapeLow && shapeId) {
          try {
            supported = await catalog.isSupported(shapeId);
          } catch (err) {
            supported = false;
          }
        }

        if (!shapeId) supported = false;

        if (shapeLow) {
          render = 'overlay';
          shapeId = '';
        } else if (!supported) {
          render = 'overlay';
          shapeId = '';
        }

        let textBbox = null;
        if (n.textBbox && typeof n.textBbox === 'object') {
          textBbox = clampBox(n.textBbox, imageWidth, imageHeight);
        }

        const reason =
          render === 'overlay' && shapeLow
            ? 'low_shape_confidence'
            : render === 'overlay' && !supported
              ? 'unsupported_shape'
              : '';

        nodes.push({
          id,
          bbox,
          text,
          textBbox,
          shapeId,
          render,
          reason,
          confidence,
          // Route-1 extra structure for deterministic rendering
          innerShapes: Array.isArray(n.innerShapes)
            ? n.innerShapes
                .filter((s) => s && typeof s === 'object' && (s.bbox || s.geometry))
                .slice(0, 20)
                .map((s) => {
                  const bb = clampBox(s.bbox || s.geometry, imageWidth, imageHeight);
                  const style = s.style && typeof s.style === 'object' ? s.style : {};
                  return {
                    bbox: bb,
                    shapeId: normalizeShapeId(String(s.shapeId || s.shape || '').trim()),
                    style,
                    confidence: typeof s.confidence === 'number' ? Math.max(0, Math.min(1, s.confidence)) : 0.5
                  };
                })
            : [],
          nodeOverlays: Array.isArray(n.nodeOverlays)
            ? n.nodeOverlays
                .filter((o) => o && typeof o === 'object' && (o.bbox || o.geometry))
                .slice(0, 10)
                .map((o, oidx) => {
                  const bb = clampBox(o.bbox || o.geometry, imageWidth, imageHeight);
                  const kind = String(o.kind || '').toLowerCase();
                  const allowed = new Set(['icon', 'photo', 'chart', 'plot', '3d', 'node']);
                  return {
                    id: String(o.id || `nov${oidx + 1}`),
                    kind: allowed.has(kind) ? kind : 'icon',
                    bbox: bb,
                    confidence: typeof o.confidence === 'number' ? Math.max(0, Math.min(1, o.confidence)) : 0.6,
                    dataUrl: typeof o.dataUrl === 'string' ? o.dataUrl : null
                  };
                })
            : [],
          containerStyle: n.containerStyle && typeof n.containerStyle === 'object' ? n.containerStyle : null
        });
      }

      const edges = edgesIn
        .filter((e) => e && typeof e === 'object')
        .map((e, idx) => {
          const id = String(e.id || `e${idx + 1}`);
          const source = String(e.source || '');
          const target = String(e.target || '');
          const conf = Number(e.confidence);
          return {
            id,
            source,
            target,
            sourceSide: normalizeSide(e.sourceSide),
            targetSide: normalizeSide(e.targetSide),
            label: typeof e.label === 'string' ? e.label : '',
            confidence: Number.isFinite(conf) ? Math.max(0, Math.min(1, conf)) : 0.6
          };
        })
        .filter((e) => nodeIds.has(e.source) && nodeIds.has(e.target));

      const overlays = overlaysIn
        .filter((o) => o && typeof o === 'object')
        .map((o, idx) => {
          const id = String(o.id || `ov${idx + 1}`);
          const bbox = clampBox(o.bbox || o, imageWidth, imageHeight);
          const kind = String(o.kind || '').toLowerCase();
          const allowed = new Set(['icon', 'photo', 'chart', 'plot', '3d']);
          const conf = Number(o.confidence);
          return {
            id,
            kind: allowed.has(kind) ? kind : 'icon',
            bbox,
            confidence: Number.isFinite(conf) ? Math.max(0, Math.min(1, conf)) : 0.6,
            dataUrl: typeof o.dataUrl === 'string' ? o.dataUrl : null
          };
        });

      // If local CV fails to detect any nodes, fall back to provider vision extraction (route-1 hybrid).
      const imgArea = Math.max(1, imageWidth * imageHeight);
      const nodeArea = nodes.reduce((sum, n) => sum + Math.max(0, Number(n?.bbox?.w || 0)) * Math.max(0, Number(n?.bbox?.h || 0)), 0);
      const areaFrac = nodeArea / imgArea;
      const hasAnyText = nodes.some((n) => typeof n?.text === 'string' && n.text.trim().length > 0);
      const ocrMode = String(out?.meta?.ocr || '');
      const shouldFallbackToProvider =
        nodes.length === 0 ||
        (nodes.length < 8 && areaFrac < 0.16) ||
        (nodes.length < 12 && edges.length === 0) ||
        !hasAnyText;

      if (!shouldFallbackToProvider) {
        res.json({
          structure: { nodes, edges, overlays },
          meta: {
            imageWidth,
            imageHeight,
            backend: out.meta?.backend || 'vision-service',
            ocr: out.meta?.ocr || null,
            counts: out.meta?.counts || { nodes: nodes.length, edges: edges.length, overlays: overlays.length },
            shapeCatalog: {
              count: catalog.set?.size || catalog.meta?.count || 0,
              stencilsBase: catalog.stencilsBase || catalog.meta?.source?.stencilsBase || ''
            }
          }
        });
        return;
      }

      // If CV has geometry but low-quality semantics (missing text/edges), try a provider "labeling" pass
      // using the CV node bboxes as fixed candidates, so we keep tight geometry but get accurate editable text.
      if (provider && nodes.length >= 6) {
        try {
          const candidates = nodes.slice(0, 60).map((n) => ({ id: n.id, bbox: n.bbox }));
          const labeledRaw = await callCandidateLabel(provider, prompt, [first], imageWidth, imageHeight, candidates);
          const labeledParsed = parseJsonFromModel(labeledRaw);
          if (!labeledParsed || typeof labeledParsed !== 'object') {
            throw new Error('Candidate labeling did not return a JSON object.');
          }

          const labeledNodesIn = Array.isArray(labeledParsed.nodes) ? labeledParsed.nodes : [];
          const labeledEdgesIn = Array.isArray(labeledParsed.edges) ? labeledParsed.edges : [];
          const candidateIds = new Set(candidates.map((c) => String(c.id)));
          const cvById = new Map(nodes.map((n) => [String(n.id), n]));

          const labeledNodes = [];
          for (const n of labeledNodesIn) {
            if (!n || typeof n !== 'object') continue;
            const id = String(n.id || '');
            if (!candidateIds.has(id)) continue;
            const base = cvById.get(id);
            if (!base) continue;

            const text = typeof n.text === 'string' ? n.text : '';
            const rawShapeId = normalizeShapeId(String(n.shapeId || n.shape || base.shapeId || '').trim());
            let textBbox = null;
            if (n.textBbox && typeof n.textBbox === 'object') {
              textBbox = clampBox(n.textBbox, imageWidth, imageHeight);
              if (
                textBbox.x < base.bbox.x ||
                textBbox.y < base.bbox.y ||
                textBbox.x + textBbox.w > base.bbox.x + base.bbox.w ||
                textBbox.y + textBbox.h > base.bbox.y + base.bbox.h
              ) {
                textBbox = null;
              }
            }

            const conf = n.confidence && typeof n.confidence === 'object' ? n.confidence : {};
            const shapeC = Number(conf.shape);
            const textC = Number(conf.text);
            const confidence = {
              bbox: base.confidence?.bbox ?? 0.7,
              text: Number.isFinite(textC) ? Math.max(0, Math.min(1, textC)) : text ? 0.8 : 0.0,
              shape: Number.isFinite(shapeC) ? Math.max(0, Math.min(1, shapeC)) : 0.75
            };

            let shapeId = rawShapeId;
            let supported = false;
            const shapeLow = confidence.shape < SHAPE_CONFIDENCE_THRESHOLD;
            if (shapeId && !shapeLow) {
              try {
                supported = await catalog.isSupported(shapeId);
              } catch (err) {
                supported = false;
              }
            }
            if (!shapeId || shapeLow || !supported) {
              shapeId = base.shapeId || '';
            }

            labeledNodes.push({
              ...base,
              text,
              textBbox,
              shapeId,
              render: 'shape',
              reason: '',
              confidence
            });
          }

          if (labeledNodes.length < Math.ceil(nodes.length * 0.6)) {
            throw new Error(`Candidate labeling dropped too many nodes (${labeledNodes.length}/${nodes.length}).`);
          }

          const labeledIds = new Set(labeledNodes.map((n) => String(n.id)));
          const labeledEdges = labeledEdgesIn
            .filter((e) => e && typeof e === 'object')
            .map((e, idx) => {
              const id = String(e.id || `e${idx + 1}`);
              const source = String(e.source || '');
              const target = String(e.target || '');
              const conf = Number(e.confidence);
              return {
                id,
                source,
                target,
                sourceSide: normalizeSide(e.sourceSide),
                targetSide: normalizeSide(e.targetSide),
                label: typeof e.label === 'string' ? e.label : '',
                confidence: Number.isFinite(conf) ? Math.max(0, Math.min(1, conf)) : 0.7
              };
            })
            .filter((e) => labeledIds.has(e.source) && labeledIds.has(e.target));

          res.json({
            structure: { nodes: labeledNodes, edges: labeledEdges, overlays },
            meta: {
              imageWidth,
              imageHeight,
              backend: `${out.meta?.backend || 'vision-service'}+provider-label`,
              ocr: out.meta?.ocr || null,
              counts: { nodes: labeledNodes.length, edges: labeledEdges.length, overlays: overlays.length },
              shapeCatalog: {
                count: catalog.set?.size || catalog.meta?.count || 0,
                stencilsBase: catalog.stencilsBase || catalog.meta?.source?.stencilsBase || ''
              }
            }
          });
          return;
        } catch (err) {
          logError('Candidate labeling failed (falling back to provider full extraction)', err, { visionUrl });
        }
      }

      if (!provider) {
        throw new Error('Vision service returned 0 nodes and no provider is configured for fallback.');
      }
      console.log(
        `[${nowIso()}] Vision service result deemed low-quality; falling back to provider (nodes=${nodes.length}, edges=${edges.length}, areaFrac=${areaFrac.toFixed(3)}, hasText=${hasAnyText}, ocr=${ocrMode || 'unknown'})`
      );
    }

    let lastErr = null;
    let lastOut = '';
    let parsed = null;
    let attemptPromptOverride = null;
    let maxTokens = 2048;

    for (let attempt = 1; attempt <= STRUCTURE_MAX_ATTEMPTS; attempt += 1) {
      try {
        lastOut = await callStructure(provider, prompt, [first], imageWidth, imageHeight, attemptPromptOverride, { maxTokens });
        parsed = parseJsonFromModel(lastOut);
        if (!parsed || typeof parsed !== 'object') {
          throw new Error('Model did not return a JSON object.');
        }
        break;
      } catch (err) {
        lastErr = err;
        const msg = String(err?.message || '').toLowerCase();
        if (msg.includes('truncated')) {
          maxTokens = 4096;
        }
        attemptPromptOverride = buildStructureRepairPrompt(instruction, prompt, lastOut, err?.message || String(err || ''));
      }
    }

    if (!parsed) {
      throw lastErr || new Error('Failed to extract structure.');
    }

    const nodesIn = Array.isArray(parsed.nodes) ? parsed.nodes : [];
    const edgesIn = Array.isArray(parsed.edges) ? parsed.edges : [];
    const overlaysIn = Array.isArray(parsed.overlays) ? parsed.overlays : [];

    const nodeIds = new Set();
    const nodes = [];

    for (let idx = 0; idx < nodesIn.length; idx += 1) {
      const n = nodesIn[idx];
      if (!n || typeof n !== 'object') continue;

      const id = String(n.id || `n${idx + 1}`);
      nodeIds.add(id);
      const bbox = clampBox(n.bbox || n, imageWidth, imageHeight);
      const text = typeof n.text === 'string' ? n.text : '';
      const rawShapeId = normalizeShapeId(String(n.shapeId || n.shape || '').trim());

      const conf = n.confidence && typeof n.confidence === 'object' ? n.confidence : {};
      const bboxC = Number(conf.bbox);
      const textC = Number(conf.text);
      const shapeC = Number(conf.shape);
      const confidence = {
        bbox: Number.isFinite(bboxC) ? Math.max(0, Math.min(1, bboxC)) : 0.7,
        text: Number.isFinite(textC) ? Math.max(0, Math.min(1, textC)) : 0.7,
        shape: Number.isFinite(shapeC) ? Math.max(0, Math.min(1, shapeC)) : 0.5
      };
      const shapeLow = confidence.shape < SHAPE_CONFIDENCE_THRESHOLD;

      let render = String(n.render || '').toLowerCase();
      if (render !== 'shape' && render !== 'overlay') render = 'shape';

      let supported = false;
      let shapeId = rawShapeId;

      if (render === 'shape' && !shapeLow && shapeId) {
        try {
          supported = await catalog.isSupported(shapeId);
        } catch (err) {
          supported = false;
        }
      }

      if (!shapeId) supported = false;

      if (shapeLow) {
        render = 'overlay';
        shapeId = '';
      } else if (!supported) {
        render = 'overlay';
        shapeId = '';
      }

      let textBbox = null;
      if (n.textBbox && typeof n.textBbox === 'object') {
        textBbox = clampBox(n.textBbox, imageWidth, imageHeight);
      }

      const reason =
        render === 'overlay' && shapeLow
          ? 'low_shape_confidence'
          : render === 'overlay' && !supported
            ? 'unsupported_shape'
            : '';

      nodes.push({
        id,
        bbox,
        text,
        textBbox,
        shapeId,
        render,
        reason,
        confidence
      });
    }

    const edges = edgesIn
      .filter((e) => e && typeof e === 'object')
      .map((e, idx) => {
        const id = String(e.id || `e${idx + 1}`);
        const source = String(e.source || '');
        const target = String(e.target || '');
        const conf = Number(e.confidence);
        return {
          id,
          source,
          target,
          sourceSide: normalizeSide(e.sourceSide),
          targetSide: normalizeSide(e.targetSide),
          label: typeof e.label === 'string' ? e.label : '',
          confidence: Number.isFinite(conf) ? Math.max(0, Math.min(1, conf)) : 0.6
        };
      })
      .filter((e) => nodeIds.has(e.source) && nodeIds.has(e.target));

    const overlays = overlaysIn
      .filter((o) => o && typeof o === 'object')
      .map((o, idx) => {
        const id = String(o.id || `ov${idx + 1}`);
        const bbox = clampBox(o.bbox || o, imageWidth, imageHeight);
        const kind = String(o.kind || '').toLowerCase();
        const allowed = new Set(['icon', 'photo', 'chart', 'plot', '3d']);
        const conf = Number(o.confidence);
        return {
          id,
          kind: allowed.has(kind) ? kind : 'icon',
          bbox,
          confidence: Number.isFinite(conf) ? Math.max(0, Math.min(1, conf)) : 0.6
        };
      });

    // If the local vision service is available, augment provider-extracted nodes with:
    // - containerStyle (fill/stroke colors)
    // - innerShapes (bars/icons as vector)
    // - nodeOverlays (bitmaps anchored to node)
    if (visionUrl && nodes.length > 0) {
      try {
        const aug = await callVisionServiceAugment(
          visionUrl,
          first,
          imageWidth,
          imageHeight,
          nodes.map((n) => ({ id: n.id, bbox: n.bbox, textBbox: n.textBbox })),
          null,
          { qualityMode: String(req.body?.qualityMode || '').toLowerCase() === 'balanced' ? 'balanced' : 'max' }
        );
        const byId = new Map((Array.isArray(aug.nodes) ? aug.nodes : []).map((n) => [String(n.id), n]));
        nodes.forEach((n) => {
          const a = byId.get(String(n.id));
          if (!a || typeof a !== 'object') return;
          if (a.containerStyle && typeof a.containerStyle === 'object') n.containerStyle = a.containerStyle;
          if (Array.isArray(a.innerShapes)) {
            n.innerShapes = a.innerShapes
              .filter((s) => s && typeof s === 'object' && (s.bbox || s.geometry))
              .slice(0, 24)
              .map((s) => {
                const bb = clampBox(s.bbox || s.geometry, imageWidth, imageHeight);
                const style = s.style && typeof s.style === 'object' ? s.style : {};
                return {
                  bbox: bb,
                  shapeId: normalizeShapeId(String(s.shapeId || s.shape || '').trim()),
                  style,
                  confidence: typeof s.confidence === 'number' ? Math.max(0, Math.min(1, s.confidence)) : 0.6
                };
              });
          }
          if (Array.isArray(a.nodeOverlays)) {
            n.nodeOverlays = a.nodeOverlays
              .filter((o) => o && typeof o === 'object' && (o.bbox || o.geometry))
              .slice(0, 16)
              .map((o, oidx) => {
                const bb = clampBox(o.bbox || o.geometry, imageWidth, imageHeight);
                const kind = String(o.kind || '').toLowerCase();
                const allowed = new Set(['icon', 'photo', 'chart', 'plot', '3d', 'node']);
                return {
                  id: String(o.id || `nov${oidx + 1}`),
                  kind: allowed.has(kind) ? kind : 'icon',
                  bbox: bb,
                  confidence: typeof o.confidence === 'number' ? Math.max(0, Math.min(1, o.confidence)) : 0.75,
                  dataUrl: typeof o.dataUrl === 'string' ? o.dataUrl : null
                };
              })
              .filter((o) => Boolean(o.dataUrl));
          }
        });
      } catch (err) {
        logError('Vision augment failed (continuing without augmentation)', err, { visionUrl });
      }
    }

    res.json({
      structure: { nodes, edges, overlays },
      meta: {
        imageWidth,
        imageHeight,
        backend: visionUrl ? 'provider+cv-augment' : 'provider',
        shapeCatalog: {
          count: catalog.set?.size || catalog.meta?.count || 0,
          stencilsBase: catalog.stencilsBase || catalog.meta?.source?.stencilsBase || ''
        }
      }
    });
  } catch (err) {
    if (err && typeof err === 'object' && !err.phase) err.phase = 'structure_pipeline';
    const out = buildClientErrorPayload(err, 'Structure extraction failed.');
    logError('API /api/vision/structure failed', err, {
      providerType: req.body?.providerType || req.body?.providerName || config.primary,
      provider: provider ? { type: provider.type, model: provider.model, baseUrl: provider.baseUrl } : null,
      visionUrl: visionUrl || null,
      imageWidth,
      imageHeight,
      promptLen: typeof req.body?.prompt === 'string' ? req.body.prompt.length : 0,
      providerHttp: err?.http || null,
      phase: out.body?.phase || null,
      code: out.body?.code || null
    });
    res.status(out.status).json(out.body);
  }
});

app.get('/api/shapes/catalog', async (req, res) => {
  try {
    const catalog = await ensureShapeCatalog((msg) => console.log(`[${nowIso()}] ${msg}`));
    res.json({
      shapes: Array.from(catalog.set || []).sort(),
      meta: {
        count: catalog.set?.size || 0,
        stencilsBase: catalog.meta?.source?.stencilsBase || ''
      }
    });
  } catch (err) {
    logError('API /api/shapes/catalog failed', err);
    res.status(500).json({ error: 'Failed to load shape catalog.' });
  }
});

app.post('/api/vision/overlays/resolve-one', async (req, res) => {
  const visionUrl = await ensureVisionServiceUrl((msg) => console.log(`[${nowIso()}] ${msg}`));
  if (!visionUrl) {
    logApiResponseError(req, 400, 'Vision service not available (SAM2 required).', {});
    res.status(400).json({
      error:
        'Vision service not available. SAM2 is required for high-quality overlay extraction. Install vision_service deps and start it (or set VISION_SERVICE_URL).'
    });
    return;
  }

  const image = req.body?.image && typeof req.body.image === 'object' ? req.body.image : null;
  if (!image || !image.dataUrl) {
    logApiResponseError(req, 400, 'Reference image is required.', {});
    res.status(400).json({ error: 'Reference image is required.' });
    return;
  }

  let imageWidth = Number(req.body?.imageWidth || image.width || 0);
  let imageHeight = Number(req.body?.imageHeight || image.height || 0);
  if (!(imageWidth > 0 && imageHeight > 0)) {
    const dims = await resolveImageDimsViaVisionService(visionUrl, image);
    if (dims) {
      imageWidth = dims.imageWidth;
      imageHeight = dims.imageHeight;
    }
  }
  if (!(imageWidth > 0 && imageHeight > 0)) {
    logApiResponseError(req, 400, 'Image width/height required.', { imageWidth, imageHeight });
    res.status(400).json({ error: 'Image width/height required.' });
    return;
  }

  const overlay = req.body?.overlay && typeof req.body.overlay === 'object' ? req.body.overlay : null;
  if (!overlay) {
    logApiResponseError(req, 400, 'Overlay required.', {});
    res.status(400).json({ error: 'Overlay required.' });
    return;
  }

  const payload = [
    {
      id: String(overlay.id || 'ov1'),
      bbox: clampBox(overlay.bbox || overlay, imageWidth, imageHeight),
      kind: normalizeOverlayKind(overlay.kind || 'icon'),
      granularity: normalizeGranularity(overlay.granularity || 'alphaMask'),
      fgPoints: Array.isArray(overlay.fgPoints) ? overlay.fgPoints : [],
      bgPoints: Array.isArray(overlay.bgPoints) ? overlay.bgPoints : []
    }
  ];

  try {
    const overlayOptions = normalizeOverlayResolveOptions(req.body?.overlayOptions);
    const textItems = Array.isArray(req.body?.textItems) ? req.body.textItems : [];
    const out = await callVisionServiceResolveOverlays(
      visionUrl,
      image,
      imageWidth,
      imageHeight,
      payload,
      false,
      overlayOptions,
      textItems && textItems.length ? textItems : undefined
    );
    res.json({
      overlay: Array.isArray(out.overlays) ? out.overlays[0] || null : null,
      meta: out.meta || { backend: 'sam2+crop', imageWidth, imageHeight }
    });
  } catch (err) {
    logError('API /api/vision/overlays/resolve-one failed', err, { visionUrl, imageWidth, imageHeight, providerHttp: err?.http || null });
    res.status(500).json({ error: err.message || 'Overlay resolve failed.' });
  }
});

app.post('/api/vision/overlays/retry', async (req, res) => {
  const config = readProviderConfig();
  const provider = resolveProvider(req.body, config);
  if (!requireConfiguredProvider(req, res, config, provider)) return;

  const visionUrl = await ensureVisionServiceUrl((msg) => console.log(`[${nowIso()}] ${msg}`));
  if (!visionUrl) {
    logApiResponseError(req, 400, 'Vision service not available (SAM2 required).', {});
    res.status(400).json({
      error:
        'Vision service not available. SAM2 is required for high-quality overlay extraction. Install vision_service deps and start it (or set VISION_SERVICE_URL).'
    });
    return;
  }

  const images = Array.isArray(req.body.images) ? req.body.images : [];
  const first = images[0] || null;
  if (!first || !first.dataUrl) {
    logApiResponseError(req, 400, 'Reference image is required.', { imagesCount: images.length });
    res.status(400).json({ error: 'Reference image is required.' });
    return;
  }

  let imageWidth = Number(req.body.imageWidth || first.width || 0);
  let imageHeight = Number(req.body.imageHeight || first.height || 0);
  if (!(imageWidth > 0 && imageHeight > 0)) {
    const dims = await resolveImageDimsViaVisionService(visionUrl, first);
    if (dims) {
      imageWidth = dims.imageWidth;
      imageHeight = dims.imageHeight;
    }
  }
  if (!(imageWidth > 0 && imageHeight > 0)) {
    logApiResponseError(req, 400, 'Image width/height required.', { imageWidth, imageHeight });
    res.status(400).json({ error: 'Image width/height required.' });
    return;
  }

  const structure = req.body.structure;
  if (!structure || typeof structure !== 'object') {
    logApiResponseError(req, 400, 'Structure JSON required.', {});
    res.status(400).json({ error: 'Structure JSON required.' });
    return;
  }

  const prompt = String(req.body.prompt || '').trim();

  try {
    let textItems = Array.isArray(req.body?.textItems) ? req.body.textItems : [];
    let textExtractError = null;
    if (!textItems.length) {
      const out = await extractTextItemsBestEffort(provider, first, imageWidth, imageHeight, {
        providerType: req.body?.providerType || req.body?.providerName || config.primary,
        provider: provider ? { type: provider.type, model: provider.model, baseUrl: provider.baseUrl } : null,
        imageWidth,
        imageHeight
      });
      textItems = out.textItems;
      textExtractError = out.textExtractError;
    }
    const overlayOptions = normalizeOverlayResolveOptions(req.body?.overlayOptions);
    const overlayOut = await resolveOverlaysWithRetries(provider, visionUrl, first, imageWidth, imageHeight, structure, prompt, {
      onlyMissing: true,
      overlayOptions,
      textItems
    });
    res.json({
      structure,
      meta: {
        imageWidth,
        imageHeight,
        backend: 'overlays-retry+sam2',
        overlayFailures: overlayOut.failures || [],
        textItems,
        ...(textExtractError ? { textExtractError } : {})
      }
    });
  } catch (err) {
    if (err && typeof err === 'object' && !err.phase) err.phase = 'overlays_retry';
    const out = buildClientErrorPayload(err, 'Retry failed overlays failed.');
    logError('API /api/vision/overlays/retry failed', err, {
      providerType: req.body?.providerType || req.body?.providerName || config.primary,
      provider: provider ? { type: provider.type, model: provider.model, baseUrl: provider.baseUrl } : null,
      visionUrl: visionUrl || null,
      imageWidth,
      imageHeight,
      providerHttp: err?.http || null,
      phase: out.body?.phase || null,
      code: out.body?.code || null
    });
    res.status(out.status).json(out.body);
  }
});

app.post('/api/vision/critic', async (req, res) => {
  const config = readProviderConfig();
  const provider = resolveProvider(req.body, config);
  if (!requireConfiguredProvider(req, res, config, provider)) return;

  const visionUrl = await ensureVisionServiceUrl((msg) => console.log(`[${nowIso()}] ${msg}`));
  if (!visionUrl) {
    logApiResponseError(req, 400, 'Vision service not available (SAM2 required).', {});
    res.status(400).json({
      error:
        'Vision service not available. SAM2 is required for high-quality overlay extraction. Install vision_service deps and start it (or set VISION_SERVICE_URL).'
    });
    return;
  }

  const images = Array.isArray(req.body.images) ? req.body.images : [];
  const first = images[0] || null;
  const rendered = String(req.body.rendered || '').trim();
  if (!first || !first.dataUrl) {
    logApiResponseError(req, 400, 'Reference image is required.', { imagesCount: images.length });
    res.status(400).json({ error: 'Reference image is required.' });
    return;
  }
  if (!rendered.startsWith('data:image/')) {
    logApiResponseError(req, 400, 'Rendered PNG dataUrl required.', { renderedLen: rendered.length });
    res.status(400).json({ error: 'Rendered PNG dataUrl required.' });
    return;
  }

  let imageWidth = Number(req.body.imageWidth || first.width || 0);
  let imageHeight = Number(req.body.imageHeight || first.height || 0);
  if (!(imageWidth > 0 && imageHeight > 0)) {
    const dims = await resolveImageDimsViaVisionService(visionUrl, first);
    if (dims) {
      imageWidth = dims.imageWidth;
      imageHeight = dims.imageHeight;
    }
  }
  if (!(imageWidth > 0 && imageHeight > 0)) {
    logApiResponseError(req, 400, 'Image width/height required.', { imageWidth, imageHeight });
    res.status(400).json({ error: 'Image width/height required.' });
    return;
  }

  let catalog;
  try {
    catalog = await ensureShapeCatalog((msg) => console.log(`[${nowIso()}] ${msg}`));
  } catch (err) {
    logError('Failed to build/load shape catalog', err);
    res.status(500).json({ error: 'Failed to build shape catalog.' });
    return;
  }

  const prompt = String(req.body.prompt || '').trim();
  const baseInstruction = buildStructureInstruction(imageWidth, imageHeight, SHAPE_CONFIDENCE_THRESHOLD);
  const criticSystem = [
    'You are a strict critic improving a diagram extraction pipeline.',
    'You will receive TWO images:',
    '(1) the reference paper figure screenshot (authoritative for coordinates),',
    '(2) the current rendered draw.io preview (to spot mistakes/missing elements).',
    'Your job: produce a corrected structure JSON that matches image (1) as closely as possible.',
    'Use image (2) only to identify what is missing/wrong; NEVER copy its geometry blindly.',
    'Pay special attention to: missing nodes, incorrect text transcription, wrong edge directions/connections, overlay granularity misclassification.',
    '',
    baseInstruction
  ].join('\n');

  const renderedImg = { name: 'rendered.png', type: 'image/png', dataUrl: rendered, width: 0, height: 0 };

  let parsed = null;
  let lastOut = '';
  let lastErr = null;
  for (let attempt = 1; attempt <= 2; attempt += 1) {
    try {
      lastOut = await callVisionJsonWithImages(
        provider,
        criticSystem,
        prompt ? `User request:\n${prompt}` : 'Return ONLY the corrected JSON object.',
        [first, renderedImg],
        { maxTokens: 6144, timeoutMs: 180000 }
      );
      parsed = parseJsonFromModel(lastOut);
      if (!parsed || typeof parsed !== 'object') {
        throw new Error('Critic did not return a JSON object.');
      }
      break;
    } catch (err) {
      lastErr = err;
      const repairSystem = buildStructureRepairPrompt(baseInstruction, prompt, lastOut, err?.message || String(err || ''));
      try {
        lastOut = await callVisionJsonWithImages(provider, repairSystem, 'Fix and return ONLY the corrected JSON object.', [first, renderedImg], {
          maxTokens: 6144,
          timeoutMs: 180000
        });
        parsed = parseJsonFromModel(lastOut);
        if (parsed && typeof parsed === 'object') break;
      } catch (err2) {
        lastErr = err2;
      }
    }
  }
  if (!parsed) {
    const err = lastErr || new Error('Critic failed');
    if (err && typeof err === 'object' && !err.phase) err.phase = 'critic';
    const out = buildClientErrorPayload(err, 'Critic failed.');
    logError('API /api/vision/critic failed', err, {
      providerType: req.body?.providerType || req.body?.providerName || config.primary,
      provider: provider ? { type: provider.type, model: provider.model, baseUrl: provider.baseUrl } : null,
      providerHttp: err?.http || null,
      phase: out.body?.phase || null,
      code: out.body?.code || null
    });
    res.status(out.status).json(out.body);
    return;
  }

  try {
    const normalized = await normalizeStructureFromModel(parsed, imageWidth, imageHeight, catalog);

    const qualityMode = String(req.body?.qualityMode || '').toLowerCase() === 'balanced' ? 'balanced' : 'max';
    const visionOptions = { qualityMode };

    let textItems = [];
    let textExtractError = null;
    if (qualityMode === 'max') {
      try {
        const cv = await callVisionServiceStructure(visionUrl, prompt, first, imageWidth, imageHeight, null, visionOptions);
        const ti = Array.isArray(cv?.meta?.textItems) ? cv.meta.textItems : [];
        if (ti.length) textItems = ti;
      } catch (err) {
        logError('Vision service /analyze failed (critic pass, local OCR unavailable)', err, { visionUrl });
      }
    }
    if (!textItems.length) {
      const out = await extractTextItemsBestEffort(provider, first, imageWidth, imageHeight, {
        providerType: req.body?.providerType || req.body?.providerName || config.primary,
        provider: provider ? { type: provider.type, model: provider.model, baseUrl: provider.baseUrl } : null,
        imageWidth,
        imageHeight
      });
      textItems = Array.isArray(out.textItems) ? out.textItems : [];
      textExtractError = out.textExtractError || null;
    }
    const unassignedTextItems = applyTextItemsToNodes(normalized, textItems, imageWidth, imageHeight);

    // Optional: CV augment for container colors and simple inner bars.
    try {
       const aug = await callVisionServiceAugment(
         visionUrl,
         first,
         imageWidth,
         imageHeight,
         normalized.nodes.map((n) => ({ id: n.id, bbox: n.bbox, textBbox: n.textBbox })),
         textItems,
         { qualityMode: String(req.body?.qualityMode || '').toLowerCase() === 'balanced' ? 'balanced' : 'max' }
       );
      const byId = new Map((Array.isArray(aug.nodes) ? aug.nodes : []).map((n) => [String(n.id), n]));
      normalized.nodes.forEach((n) => {
        const a = byId.get(String(n.id));
        if (!a || typeof a !== 'object') return;
        if (a.containerStyle && typeof a.containerStyle === 'object') n.containerStyle = a.containerStyle;
        if (Array.isArray(a.innerShapes)) {
          n.innerShapes = a.innerShapes
            .filter((s) => s && typeof s === 'object' && (s.bbox || s.geometry))
            .slice(0, 30)
            .map((s) => {
              const bb = clampBox(s.bbox || s.geometry, imageWidth, imageHeight);
              const style = s.style && typeof s.style === 'object' ? s.style : {};
              return {
                bbox: bb,
                shapeId: normalizeShapeId(String(s.shapeId || s.shape || '').trim()),
                style,
                confidence: typeof s.confidence === 'number' ? Math.max(0, Math.min(1, s.confidence)) : 0.65
              };
            });
        }
      });
    } catch (err) {
      logError('Vision augment failed (critic pass, continuing)', err, { visionUrl });
    }

    const overlayOptions = normalizeOverlayResolveOptions(req.body?.overlayOptions);
    const overlayOut = await resolveOverlaysWithRetries(provider, visionUrl, first, imageWidth, imageHeight, normalized, prompt, {
      onlyMissing: true,
      overlayOptions,
      textItems
    });

    appendTextNodes(normalized, unassignedTextItems, imageWidth, imageHeight);

    res.json({
      structure: normalized,
      meta: {
        imageWidth,
        imageHeight,
        backend: 'critic+provider+sam2',
        overlayFailures: overlayOut.failures || [],
        textItems,
        ...(textExtractError ? { textExtractError } : {}),
        shapeCatalog: {
          count: catalog.set?.size || catalog.meta?.count || 0,
          stencilsBase: catalog.stencilsBase || catalog.meta?.source?.stencilsBase || ''
        }
      }
    });
  } catch (err) {
    if (err && typeof err === 'object' && !err.phase) err.phase = 'critic_postprocess';
    const out = buildClientErrorPayload(err, 'Critic postprocess failed.');
    logError('API /api/vision/critic postprocess failed', err, {
      providerType: req.body?.providerType || req.body?.providerName || config.primary,
      provider: provider ? { type: provider.type, model: provider.model, baseUrl: provider.baseUrl } : null,
      providerHttp: err?.http || null,
      phase: out.body?.phase || null,
      code: out.body?.code || null
    });
    res.status(out.status).json(out.body);
  }
});

app.post('/api/vision/debug/annotate', async (req, res) => {
  const visionUrl = await ensureVisionServiceUrl((msg) => console.log(`[${nowIso()}] ${msg}`));
  if (!visionUrl) {
    logApiResponseError(req, 400, 'Vision service not available.', {});
    res
      .status(400)
      .json({ error: 'Vision service not available. Install vision_service deps and/or set VISION_SERVICE_URL.' });
    return;
  }
  const images = Array.isArray(req.body.images) ? req.body.images : [];
  const first = images[0] || null;
  if (!first || !first.dataUrl) {
    logApiResponseError(req, 400, 'Reference image is required.', { imagesCount: images.length });
    res.status(400).json({ error: 'Reference image is required.' });
    return;
  }

  const imageWidth = Number(req.body.imageWidth || first.width || 0);
  const imageHeight = Number(req.body.imageHeight || first.height || 0);
  const prompt = String(req.body.prompt || '').trim();

  try {
    const url = `${visionUrl.replace(/\/$/, '')}/debug/annotate`;
    const response = await fetchWithTimeout(
      url,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, image: first, imageWidth, imageHeight })
      },
      120000
    );

    const text = await response.text();
    let data = null;
    try {
      data = JSON.parse(text);
    } catch (err) {
      data = null;
    }
    if (!response.ok) {
      throw buildHttpError(response, text, data, url);
    }
    res.json(data || {});
  } catch (err) {
    logError('API /api/vision/debug/annotate failed', err, { providerHttp: err?.http || null });
    res.status(500).json({ error: err.message || 'Vision debug failed.' });
  }
});

app.post('/api/overlays/detect', async (req, res) => {
  const config = readProviderConfig();
  const provider = resolveProvider(req.body, config);
  if (!requireConfiguredProvider(req, res, config, provider)) return;

  try {
    const images = Array.isArray(req.body.images) ? req.body.images : [];
    if (images.length === 0) {
      res.json({ overlays: [] });
      return;
    }

    const imageWidth = Number(req.body.imageWidth || images[0]?.width || 0);
    const imageHeight = Number(req.body.imageHeight || images[0]?.height || 0);

    const raw = await callOverlays(provider, images, imageWidth, imageHeight);
    const parsed = parseJsonFromModel(raw);
    const list = Array.isArray(parsed) ? parsed : Array.isArray(parsed?.overlays) ? parsed.overlays : [];

    const overlays = list
      .filter((item) => item && typeof item === 'object')
      .slice(0, 6)
      .map((item, idx) => {
        const x = Number(item.x);
        const y = Number(item.y);
        const width = Number(item.width);
        const height = Number(item.height);
        const confidence = Number(item.confidence);
        const label = String(item.label || `overlay-${idx + 1}`);
        const kind = String(item.kind || '');
        return { label, kind, x, y, width, height, confidence };
      })
      .filter((item) => Number.isFinite(item.x) && Number.isFinite(item.y) && Number.isFinite(item.width) && Number.isFinite(item.height))
      .map((item) => {
        const maxW = imageWidth > 0 ? imageWidth : 1e9;
        const maxH = imageHeight > 0 ? imageHeight : 1e9;
        const x = Math.max(0, Math.min(item.x, maxW));
        const y = Math.max(0, Math.min(item.y, maxH));
        const w = Math.max(1, Math.min(item.width, maxW - x));
        const h = Math.max(1, Math.min(item.height, maxH - y));
        return { ...item, x, y, width: w, height: h };
      });

    res.json({ overlays });
  } catch (err) {
    if (err && typeof err === 'object' && !err.phase) err.phase = 'overlay_detect';
    const out = buildClientErrorPayload(err, 'Overlay detection failed.');
    logError('API /api/overlays/detect failed', err, {
      providerType: req.body?.providerType || req.body?.providerName,
      provider: provider ? { type: provider.type, model: provider.model, baseUrl: provider.baseUrl } : null,
      imageWidth: req.body?.imageWidth,
      imageHeight: req.body?.imageHeight,
      providerHttp: err?.http || null,
      imagesCount: Array.isArray(req.body?.images) ? req.body.images.length : 0,
      phase: out.body?.phase || null,
      code: out.body?.code || null
    });
    res.status(out.status).json(out.body);
  }
});

app.post('/api/client-error', (req, res) => {
  try {
    const body = req.body || {};
    logError('ClientError', new Error(String(body.message || 'Client error')), {
      name: body.name,
      stack: body.stack ? String(body.stack).slice(0, 4000) : '',
      context: body.context || {},
      userAgent: body.userAgent ? String(body.userAgent).slice(0, 300) : ''
    });
  } catch (err) {
    logError('ClientError handler failed', err);
  }
  res.json({ ok: true });
});

app.use((err, req, res, next) => {
  logError('Express error middleware', err, { method: req.method, url: req.originalUrl });
  res.status(500).json({ error: 'Internal server error.' });
});

app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '..', 'web', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
  ensureVisionServiceUrl((msg) => console.log(`[${nowIso()}] ${msg}`))
    .then((url) => {
      if (url) console.log(`[${nowIso()}] Vision service ready: ${url}`);
      else console.log(`[${nowIso()}] Vision service not available (LLM structure extraction fallback).`);
    })
    .catch(() => {
      console.log(`[${nowIso()}] Vision service not available (LLM structure extraction fallback).`);
    });
});
