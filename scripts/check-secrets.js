/* eslint-disable no-console */
const fs = require('fs');
const path = require('path');

const ROOT = process.cwd();

const IGNORE_DIR_NAMES = new Set([
  'node_modules',
  '.git',
  '.idea',
  '.vscode',
  '.venv',
  'venv',
  '__pycache__',
  '.pytest_cache',
  '.mypy_cache',
  '.ruff_cache',
  '.weights',
  '.cache'
]);

const IGNORE_EXT = new Set([
  '.png',
  '.jpg',
  '.jpeg',
  '.webp',
  '.gif',
  '.ico',
  '.pdf',
  '.zip',
  '.gz',
  '.tar',
  '.7z',
  '.bin',
  '.pt',
  '.pth',
  '.onnx',
  '.mp4',
  '.mov'
]);

const PATTERNS = [
  { name: 'OpenAI API key', re: /sk-[A-Za-z0-9]{16,}/g },
  { name: 'Google API key', re: /AIza[0-9A-Za-z\-_]{20,}/g },
  { name: 'GitHub token', re: /ghp_[A-Za-z0-9]{20,}/g },
  { name: 'Slack token', re: /xox[bap]-[A-Za-z0-9-]{10,}/g },
  { name: 'Private key', re: /-----BEGIN (?:RSA|OPENSSH|EC|DSA) PRIVATE KEY-----/g },
  {
    name: 'Non-empty apiKey field',
    re: /"apiKey"\s*:\s*"(?!\s*$)(?!YOUR_)(?!<YOUR_)(?!REPLACE)(?!CHANGEME)(?!TODO)(?!EXAMPLE)(?!placeholder)([^"\r\n]{8,})"/gi
  }
];

function isIgnoredDir(fullPath) {
  const parts = fullPath.split(path.sep);
  return parts.some((p) => IGNORE_DIR_NAMES.has(p));
}

function redact(text) {
  let out = String(text || '');
  PATTERNS.forEach(({ re }) => {
    out = out.replace(re, '<REDACTED>');
  });
  out = out.replace(/Bearer\s+([A-Za-z0-9\-._~+/]+=*)/g, 'Bearer <REDACTED>');
  return out;
}

function findMatchesInText(text) {
  const matches = [];
  const content = String(text || '');
  PATTERNS.forEach(({ name, re }) => {
    re.lastIndex = 0;
    let m;
    // eslint-disable-next-line no-cond-assign
    while ((m = re.exec(content))) {
      matches.push({ name, index: m.index, length: m[0].length });
      if (matches.length >= 50) break;
    }
  });
  return matches;
}

function indexToLineCol(text, index) {
  const slice = text.slice(0, Math.max(0, index));
  const lines = slice.split('\n');
  const line = lines.length;
  const col = lines[lines.length - 1].length + 1;
  return { line, col };
}

function walk(dir) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (IGNORE_DIR_NAMES.has(entry.name)) continue;
      if (isIgnoredDir(full)) continue;
      files.push(...walk(full));
      continue;
    }
    files.push(full);
  }
  return files;
}

function main() {
  const allFiles = walk(ROOT);
  const findings = [];

  for (const file of allFiles) {
    const ext = path.extname(file).toLowerCase();
    if (IGNORE_EXT.has(ext)) continue;
    if (isIgnoredDir(file)) continue;

    let text = '';
    try {
      text = fs.readFileSync(file, 'utf8');
    } catch (err) {
      continue;
    }

    const matches = findMatchesInText(text);
    if (!matches.length) continue;

    matches.forEach((match) => {
      const pos = indexToLineCol(text, match.index);
      findings.push({ file: path.relative(ROOT, file), ...pos, name: match.name });
    });
  }

  if (!findings.length) {
    console.log('OK: no obvious secrets detected.');
    process.exit(0);
  }

  console.error('ERROR: potential secrets detected. Please remove them before pushing:');
  findings.slice(0, 50).forEach((f) => {
    console.error(`- ${f.file}:${f.line}:${f.col} (${f.name})`);
  });
  if (findings.length > 50) console.error(`...and ${findings.length - 50} more`);
  console.error('\nTip: keep API keys in local config / env vars and never commit them.');
  process.exit(1);
}

main();
