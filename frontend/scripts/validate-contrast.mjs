#!/usr/bin/env node
// validate-contrast.mjs — WCAG contrast gate for the Sepia Etude palette.
//
// The token PAIRS below are hardcoded from VISUAL_DESIGN.md ("Color contrast"):
// the seven documented combinations plus the full pie matrix — every pie
// family's text companion on ground, panel, and muted in both modes. The hex
// VALUES are parsed live from src/app.css so this check cannot drift from the
// palette actually shipped. Exits nonzero if any pair falls below WCAG AA for
// normal text (4.5:1).

import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const cssPath = join(dirname(fileURLToPath(import.meta.url)), '..', 'src', 'app.css');
const css = readFileSync(cssPath, 'utf8');

// --- Parse custom properties from app.css -------------------------------

function parseVars(block) {
  const vars = {};
  for (const match of block.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)) {
    vars[match[1]] = match[2].trim();
  }
  return vars;
}

function extractBlock(source, opener) {
  const start = source.indexOf(opener);
  if (start === -1) throw new Error(`Could not find "${opener}" in ${cssPath}`);
  let depth = 0;
  for (let i = source.indexOf('{', start); i < source.length; i++) {
    if (source[i] === '{') depth++;
    if (source[i] === '}' && --depth === 0) return source.slice(start, i + 1);
  }
  throw new Error(`Unbalanced braces after "${opener}"`);
}

const lightVars = parseVars(extractBlock(css, ':root'));
const darkBlock = extractBlock(css, '@media (prefers-color-scheme: dark)');
const darkVars = { ...lightVars, ...parseVars(extractBlock(darkBlock, ':root')) };

function resolve(vars, name, seen = new Set()) {
  if (seen.has(name)) throw new Error(`Circular var reference at ${name}`);
  seen.add(name);
  const value = vars[name];
  if (value === undefined) throw new Error(`Token ${name} not found in app.css`);
  const ref = value.match(/^var\((--[\w-]+)\)$/);
  if (ref) return resolve(vars, ref[1], seen);
  if (!/^#[0-9a-f]{6}$/i.test(value)) {
    throw new Error(`Token ${name} resolves to non-hex value "${value}"`);
  }
  return value.toLowerCase();
}

// --- WCAG 2.x relative luminance and contrast ratio ----------------------

function luminance(hex) {
  const channel = (i) => {
    const c = parseInt(hex.slice(1 + 2 * i, 3 + 2 * i), 16) / 255;
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * channel(0) + 0.7152 * channel(1) + 0.0722 * channel(2);
}

function contrast(fgHex, bgHex) {
  const [a, b] = [luminance(fgHex), luminance(bgHex)];
  return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
}

// --- The contract: VISUAL_DESIGN.md "Color Contrast" table ---------------

const AA_NORMAL_TEXT = 4.5;

const pairs = [
  // Documented combinations (VISUAL_DESIGN.md contrast table).
  { scheme: 'light', fg: '--text', bg: '--bg', label: 'body text on ground' },
  { scheme: 'light', fg: '--text-secondary', bg: '--bg-muted', label: 'secondary text on muted' },
  { scheme: 'light', fg: '--accent-text', bg: '--bg', label: 'bronze display on ground' },
  { scheme: 'light', fg: '--ivory', bg: '--accent', label: 'ivory on accent' },
  { scheme: 'dark', fg: '--text', bg: '--bg', label: 'body text on ground' },
  { scheme: 'dark', fg: '--ivory', bg: '--accent', label: 'ivory on accent' },
  { scheme: 'dark', fg: '--accent-text', bg: '--bg', label: 'bronze display on ground' },
];

// Full pie matrix: every family's text companion on all three ground levels,
// both modes.
const pieInks = {
  '--success-text': 'forest ink',
  '--warning-text': 'plains ink',
  '--info-text': 'island ink',
  '--error-text': 'mountain ink',
  '--neutral-text': 'swamp ink',
};
const grounds = { '--bg': 'ground', '--bg-surface': 'panel', '--bg-muted': 'muted' };
for (const scheme of ['light', 'dark']) {
  for (const [fg, fgLabel] of Object.entries(pieInks)) {
    for (const [bg, bgLabel] of Object.entries(grounds)) {
      pairs.push({ scheme, fg, bg, label: `${fgLabel} on ${bgLabel}` });
    }
  }
}

const rows = pairs.map(({ scheme, fg, bg, label }) => {
  const vars = scheme === 'dark' ? darkVars : lightVars;
  const fgHex = resolve(vars, fg);
  const bgHex = resolve(vars, bg);
  const ratio = contrast(fgHex, bgHex);
  const grade = ratio >= 7 ? 'AAA' : ratio >= AA_NORMAL_TEXT ? 'AA' : 'FAIL';
  return {
    scheme,
    pair: `${fg} on ${bg}`,
    label,
    colors: `${fgHex} / ${bgHex}`,
    ratio: `${ratio.toFixed(2)}:1`,
    grade,
    pass: ratio >= AA_NORMAL_TEXT,
  };
});

const headers = ['scheme', 'pair', 'label', 'colors', 'ratio', 'grade'];
const widths = headers.map((h) => Math.max(h.length, ...rows.map((r) => String(r[h]).length)));
const line = (cells) => cells.map((c, i) => String(c).padEnd(widths[i])).join('  ');

console.log(line(headers));
console.log(line(widths.map((w) => '-'.repeat(w))));
for (const row of rows) console.log(line(headers.map((h) => row[h])));

const failures = rows.filter((r) => !r.pass);
if (failures.length > 0) {
  console.error(`\n${failures.length} pair(s) below WCAG AA (${AA_NORMAL_TEXT}:1).`);
  process.exit(1);
}
console.log(`\nAll ${rows.length} pairs meet WCAG AA (>= ${AA_NORMAL_TEXT}:1 for normal text).`);
