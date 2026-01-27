const fs = require('fs');
const path = require('path');

const CATALOG_PATH = path.join(__dirname, 'catalog.json');
const DEFAULT_SHAPES_PATH = path.join(__dirname, 'default-shapes.json');
const DEFAULT_STENCILS_BASE = 'https://embed.diagrams.net/stencils';
const DEFAULT_APP_JS = 'https://embed.diagrams.net/js/app.min.js';

function builtInShapes() {
  const base = new Set([
    'rectangle',
    'rect',
    'roundRect',
    'ellipse',
    'rhombus',
    'diamond',
    'triangle',
    'hexagon',
    'cloud',
    'cylinder',
    'parallelogram',
    'trapezoid',
    'actor',
    'process',
    'document',
    'note',
    'label',
    'image',
    'line',
    'arrow',
    'curlyBracket',
    'callout'
  ]);

  // Add extracted default shapes from diagrams.net runtime if available.
  try {
    if (fs.existsSync(DEFAULT_SHAPES_PATH)) {
      const raw = fs.readFileSync(DEFAULT_SHAPES_PATH, 'utf8');
      const parsed = JSON.parse(raw);
      const list = Array.isArray(parsed.shapes) ? parsed.shapes : [];
      list.forEach((s) => {
        const name = String(s || '').trim();
        if (name) base.add(name);
      });
    }
  } catch (err) {
    // ignore
  }

  return base;
}

function normalizeShapeId(shapeId) {
  const raw = String(shapeId || '').trim();
  if (!raw) return '';
  const lower = raw.toLowerCase();
  if (lower === 'diamond') return 'rhombus';
  return raw;
}

function normalizeStencilShapeName(name) {
  return String(name || '')
    .trim()
    .replace(/ /g, '_')
    .toLowerCase();
}

function fileForMxgraphShape(shapeId) {
  const s = String(shapeId || '');
  if (!s.startsWith('mxgraph.')) return null;
  const parts = s.split('.');
  if (parts.length < 3) return null;
  // mxgraph.<lib...>.<shape>
  const libParts = parts.slice(1, -1);
  const libPath = libParts.join('/');
  return `${libPath}.xml`;
}

async function fetchText(url) {
  const res = await fetch(url, { method: 'GET' });
  if (!res.ok) {
    const err = new Error(`HTTP ${res.status} for ${url}`);
    err.status = res.status;
    throw err;
  }
  return await res.text();
}

async function ensureDefaultShapes(logger) {
  if (fs.existsSync(DEFAULT_SHAPES_PATH)) return;
  const url = String(process.env.DRAWIO_APP_MIN_JS || DEFAULT_APP_JS);
  if (logger) logger(`Downloading draw.io runtime JS to extract default shapes: ${url}`);
  const text = await fetchText(url);

  const re = /mxCellRenderer\.defaultShapes\.([A-Za-z0-9_]+)\b/g;
  const set = new Set();
  let m;
  while ((m = re.exec(text))) {
    const name = String(m[1] || '').trim();
    if (name) set.add(name);
  }

  const payload = { source: url, extractedAt: new Date().toISOString(), count: set.size, shapes: Array.from(set).sort() };
  try {
    fs.writeFileSync(DEFAULT_SHAPES_PATH, JSON.stringify(payload, null, 2), 'utf8');
  } catch (err) {
    // ignore
  }
}

function parseStencilXmlNames(xmlText) {
  // Gets stencilset name and raw <shape name=""> names.
  const rootMatch = xmlText.match(/<(?:shapes|stencils)\b[^>]*\bname\s*=\s*"(.*?)"[^>]*>/i);
  const rootName = rootMatch ? String(rootMatch[1] || '').trim() : '';

  const names = [];
  const re = /<shape\b[^>]*\bname\s*=\s*"(.*?)"[^>]*>/gi;
  let match;
  while ((match = re.exec(xmlText))) {
    const name = String(match[1] || '').trim();
    if (name) names.push(name);
  }

  return { rootName, names };
}

function loadCatalogFromDisk() {
  if (!fs.existsSync(CATALOG_PATH)) return null;
  try {
    const raw = fs.readFileSync(CATALOG_PATH, 'utf8');
    const data = JSON.parse(raw);
    const shapes = Array.isArray(data.shapes) ? data.shapes : [];
    return { meta: data, set: new Set(shapes) };
  } catch (err) {
    return null;
  }
}

function persistCatalog(meta, set) {
  try {
    const out = {
      ...meta,
      updatedAt: new Date().toISOString(),
      count: set.size,
      shapes: Array.from(set).sort()
    };
    fs.writeFileSync(CATALOG_PATH, JSON.stringify(out, null, 2), 'utf8');
  } catch (err) {
    // ignore
  }
}

let cached = null;
let stencilCache = new Map(); // file -> Set(shapeIds)

async function ensureShapeCatalog(logger) {
  if (cached) return cached;

  await ensureDefaultShapes(logger);

  const existing = loadCatalogFromDisk();
  if (existing) {
    cached = existing;
  } else {
    const base = builtInShapes();
    const meta = {
      source: { stencilsBase: process.env.DRAWIO_STENCILS_BASE || DEFAULT_STENCILS_BASE },
      generatedAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      count: base.size,
      shapes: Array.from(base).sort()
    };
    cached = { meta, set: base };
    persistCatalog(meta, base);
  }

  const stencilsBase = String(process.env.DRAWIO_STENCILS_BASE || cached.meta?.source?.stencilsBase || DEFAULT_STENCILS_BASE).replace(/\/$/, '');

  async function ensureMxgraphShape(shapeId) {
    const normalized = normalizeShapeId(shapeId);
    if (cached.set.has(normalized)) return true;
    const file = fileForMxgraphShape(normalized);
    if (!file) return false;

    if (stencilCache.has(file)) {
      const set = stencilCache.get(file);
      const ok = set.has(normalized.toLowerCase());
      if (ok) {
        cached.set.add(normalized);
        persistCatalog(cached.meta, cached.set);
      }
      return ok;
    }

    const url = `${stencilsBase}/${file}`;
    if (logger) logger(`Fetching stencil set ${url}`);

    let text;
    try {
      text = await fetchText(url);
    } catch (err) {
      // Try a fallback for common irregularities: some libraries are stored in a single xml at the first segment.
      // Example: mxgraph.aws4.* -> aws4.xml already, so no change. For others, fallback may help.
      if (err && err.status === 404) {
        stencilCache.set(file, new Set());
        return false;
      }
      throw err;
    }

    const parsed = parseStencilXmlNames(text);
    const rootName = String(parsed.rootName || '').trim().toLowerCase();
    const set = new Set();
    if (rootName) {
      parsed.names.forEach((n) => {
        const norm = normalizeStencilShapeName(n);
        set.add(`${rootName}.${norm}`);
      });
    }
    stencilCache.set(file, set);

    const ok = set.has(normalized.toLowerCase());
    if (ok) {
      cached.set.add(normalized);
      persistCatalog(cached.meta, cached.set);
    }
    return ok;
  }

  async function isSupported(shapeId) {
    const normalized = normalizeShapeId(shapeId);
    if (!normalized) return false;
    if (cached.set.has(normalized)) return true;
    if (!normalized.startsWith('mxgraph.')) {
      // Built-in shapes: allow if in built-in list (already in cached set).
      return false;
    }
    return ensureMxgraphShape(normalized);
  }

  cached.isSupported = isSupported;
  cached.stencilsBase = stencilsBase;
  return cached;
}

module.exports = { ensureShapeCatalog, normalizeShapeId };
