// The design lint: `DESIGN.md`'s permitted list against the literals in
// `app.css` and the board's Svelte styles, and the contrast of the text tokens
// on the grounds they sit on, in both themes.
import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const ui = join(dirname(fileURLToPath(import.meta.url)), '..', '..');

type Permitted = {
  families: Set<string>;
  sizes: Map<number, number>;
  weights: Set<string>;
  radii: Set<string>;
  shadows: Set<string>;
};

function readPermitted(design: string): Permitted {
  const block = /```permitted\n([\s\S]*?)```/.exec(design);
  if (!block) throw new Error('DESIGN.md has no ```permitted block');
  const lists = new Map<string, string[]>();
  for (const line of block[1].trim().split('\n')) {
    const [key, values] = line.split(':');
    lists.set(key.trim(), values.split(',').map(value => value.trim()));
  }
  const list = (key: string): string[] => lists.get(key) ?? [];
  return {
    families: new Set([...list('families'), ...list('fallbacks')]),
    sizes: new Map(list('sizes').map(pair => pair.split('/').map(Number) as [number, number])),
    weights: new Set(list('weights')),
    radii: new Set(list('radii')),
    shadows: new Set(list('shadows')),
  };
}

type Finding = { file: string; line: number; declaration: string; why: string };

/** Every `property: value` in a stylesheet, with its line, comments removed. */
function declarations(css: string): { property: string; value: string; line: number; atRule: string }[] {
  const source = css.replace(/\/\*[\s\S]*?\*\//g, comment => comment.replace(/[^\n]/g, ' '));
  const found: { property: string; value: string; line: number; atRule: string }[] = [];
  const contexts: string[] = [];
  let start = 0;
  for (let i = 0; i < source.length; i += 1) {
    const char = source[i];
    if (char === '{') {
      contexts.push(source.slice(start, i).trim());
      start = i + 1;
    } else if (char === ';' || char === '}') {
      const text = source.slice(start, i);
      const colon = text.indexOf(':');
      if (colon > 0 && !text.trim().startsWith('@')) {
        const property = text.slice(0, colon).trim();
        if (/^-?-?[a-z-]+$/i.test(property)) {
          const line = source.slice(0, start + text.search(/\S/)).split('\n').length;
          found.push({ property, value: text.slice(colon + 1).trim(), line, atRule: contexts.find(c => c.startsWith('@')) ?? '' });
        }
      }
      if (char === '}') contexts.pop();
      start = i + 1;
    }
  }
  return found;
}

const COLOUR_LITERAL = /#[0-9a-f]{3,8}\b|\b(?:rgba?|hsla?|oklch|lab|lch)\(|\b(?:white|black|red|green|blue|gray|grey|ivory|gold)\b/i;
const VAR = /^var\((--[\w-]+)\)$/;

/** Split at a separator outside parentheses: `0 0 0 1px color-mix(a, b), var(--x)`. */
function splitTop(value: string, separator: RegExp): string[] {
  const parts: string[] = [];
  let depth = 0;
  let current = '';
  for (const char of value) {
    if (char === '(') depth += 1;
    if (char === ')') depth -= 1;
    if (depth === 0 && separator.test(char)) {
      if (current.trim()) parts.push(current.trim());
      current = '';
    } else current += char;
  }
  if (current.trim()) parts.push(current.trim());
  return parts;
}

function familyNames(value: string): string[] {
  return value.split(',').map(name => name.trim().replace(/^['"]|['"]$/g, ''));
}

function checkSize(value: string, permitted: Permitted): string | null {
  if (value === 'inherit') return null;
  if (/^calc\(var\(--pip\) \* [\d.]+\)$/.test(value)) return null;
  const px = /^(\d+(?:\.\d+)?)px$/.exec(value);
  if (!px || !permitted.sizes.has(Number(px[1]))) return `size ${value} is not on the scale`;
  return null;
}

function checkLineHeight(value: string, permitted: Permitted): string | null {
  if (value === '1' || value === 'inherit' || value === 'normal') return null;
  const px = /^(\d+)px$/.exec(value);
  if (!px || ![...permitted.sizes.values()].includes(Number(px[1]))) return `line height ${value} is not on the scale`;
  return null;
}

/** The game's and the interface's faces; monospace only in app.css, for code. */
function checkFamily(value: string, file: string): string | null {
  if (value === 'inherit') return null;
  const faces = file.endsWith('app.css') ? /^var\(--font-(sans|serif|mono)\)$/ : /^var\(--font-(sans|serif)\)$/;
  return faces.test(value) ? null : `family ${value} is not the game's or the interface's face`;
}

type Shorthand = { style: string | null; weight: string | null; size: string; lineHeight: string | null; family: string };

function parseFont(value: string): Shorthand | null {
  const words = splitTop(value, /\s/);
  const style = words[0] === 'italic' || words[0] === 'normal' ? words.shift()! : null;
  const weight = words[0] && /^\d{3}$/.test(words[0]) ? words.shift()! : null;
  const sizeWord = words.shift();
  if (!sizeWord || words.length === 0) return null;
  const slash = sizeWord.lastIndexOf('/');
  const [size, lineHeight] = slash > 0 && !sizeWord.slice(slash).includes(')') ? [sizeWord.slice(0, slash), sizeWord.slice(slash + 1)] : [sizeWord, null];
  return { style, weight, size, lineHeight, family: words.join(' ') };
}

function checkShadow(value: string, permitted: Permitted): string | null {
  if (value === 'none') return null;
  for (const part of splitTop(value, /,/)) {
    const token = VAR.exec(part);
    if (token) {
      if (!permitted.shadows.has(token[1])) return `shadow ${token[1]} is not a permitted shadow`;
      continue;
    }
    // An edge or ring: offsets and no blur, in a token colour.
    const lengths = splitTop(part.replace(/^inset\s+/, ''), /\s/).filter(word => /^-?[\d.]+(px)?$/.test(word));
    if (lengths.length < 3 || Number.parseFloat(lengths[2]) !== 0) return `shadow ${part} blurs; use a shadow token`;
  }
  return null;
}

function checkRadius(value: string, permitted: Permitted): string | null {
  for (const part of value.split(/\s+/)) {
    if (part === '0' || part === '50%' || part === 'inherit') continue;
    const token = VAR.exec(part);
    if (!token || !permitted.radii.has(token[1])) return `radius ${part} is not a permitted radius`;
  }
  return null;
}

/** The literals in one stylesheet that DESIGN.md does not permit. */
function lint(css: string, file: string, permitted: Permitted): Finding[] {
  const findings: Finding[] = [];
  for (const { property, value, line, atRule } of declarations(css)) {
    const why: (string | null)[] = [];
    if (atRule.startsWith('@font-face')) {
      if (property === 'font-family' && !familyNames(value).every(name => permitted.families.has(name))) why.push(`family ${value} is not permitted`);
    } else if (property.startsWith('--')) {
      if (property.startsWith('--font-') && !familyNames(value).every(name => permitted.families.has(name))) why.push(`family ${value} is not permitted`);
    } else {
      if (COLOUR_LITERAL.test(value) && property !== 'mask-image') why.push('colour literal; use a token');
      if (property === 'font-size') why.push(checkSize(value, permitted));
      if (property === 'line-height') why.push(checkLineHeight(value, permitted));
      if (property === 'font-family') why.push(checkFamily(value, file));
      if (property === 'font-weight' && !permitted.weights.has(value) && value !== 'inherit') why.push(`weight ${value} is not permitted`);
      if (property === 'font' && value !== 'inherit') {
        const font = parseFont(value);
        if (!font) why.push(`font ${value} is not weight size/line family`);
        else {
          if (font.style === 'italic') why.push('italic is not in the system');
          if (font.weight && !permitted.weights.has(font.weight)) why.push(`weight ${font.weight} is not permitted`);
          why.push(checkSize(font.size, permitted));
          if (font.lineHeight) why.push(checkLineHeight(font.lineHeight, permitted));
          why.push(checkFamily(font.family, file));
        }
      }
      if (property === 'font-style' && value === 'italic') why.push('italic is not in the system');
      if (property === 'letter-spacing' && value !== '0' && value !== 'normal') why.push('tracking is not in the system');
      if (property === 'text-transform' && value !== 'none') why.push('labels are sentence case, not transformed');
      if (property === 'border-radius') why.push(checkRadius(value, permitted));
      if (property === 'box-shadow') why.push(checkShadow(value, permitted));
      if (property === 'text-shadow' || (property === 'filter' && value.includes('drop-shadow'))) why.push('shadow outside box-shadow tokens');
    }
    for (const reason of why) if (reason) findings.push({ file, line, declaration: `${property}: ${value}`, why: reason });
  }
  return findings;
}

/** The `<style>` blocks of a Svelte file, padded so lines keep their numbers. */
function svelteStyles(source: string): string {
  const lines = source.split('\n').map(() => '');
  for (const match of source.matchAll(/<style[^>]*>([\s\S]*?)<\/style>/g)) {
    const first = source.slice(0, match.index! + match[0].indexOf('>') + 1).split('\n').length - 1;
    match[1].split('\n').forEach((text, offset) => { lines[first + offset] = text; });
  }
  return lines.join('\n');
}

/** Tailwind's arbitrary values (`text-[13px]`, `rounded-[5px]`) in markup bypass the voices. */
function arbitraryClasses(source: string, file: string): Finding[] {
  const markup = source.replace(/<style[\s\S]*?<\/style>/g, '').replace(/<script[\s\S]*?<\/script>/g, block => block.replace(/[^\n]/g, ' '));
  const findings: Finding[] = [];
  markup.split('\n').forEach((text, index) => {
    for (const match of text.matchAll(/\b(?:text|font|rounded|shadow|tracking|leading)-\[[^\]]+\]/g)) {
      findings.push({ file, line: index + 1, declaration: match[0], why: 'arbitrary Tailwind value' });
    }
  });
  return findings;
}

const permitted = readPermitted(readFileSync(join(ui, 'DESIGN.md'), 'utf8'));

function lintFile(path: string): Finding[] {
  const source = readFileSync(path, 'utf8');
  const file = relative(ui, path);
  if (path.endsWith('.css')) return lint(source, file, permitted);
  return [...lint(svelteStyles(source), file, permitted), ...arbitraryClasses(source, file)];
}

describe('the design lint', () => {
  it('finds no literal outside DESIGN.md in app.css or the board', () => {
    const board = join(ui, 'src', 'lib', 'board');
    const files = [join(ui, 'src', 'app.css'), ...readdirSync(board).filter(name => name.endsWith('.svelte')).map(name => join(board, name))];
    const findings = files.flatMap(lintFile).map(f => `${f.file}:${f.line} ${f.declaration} (${f.why})`);
    expect(findings).toEqual([]);
  });

  it('refuses what the system leaves out', () => {
    const css = `.a { font: 700 13px/1.3 var(--font-mono); color: #fff; border-radius: 5px; box-shadow: 0 2px 6px rgb(0 0 0 / 0.4); text-transform: uppercase; letter-spacing: 0.1em; }
@font-face { font-family: 'Lato'; }`;
    const why = lint(css, 'sample.css', permitted).map(f => f.why);
    expect(why).toEqual(expect.arrayContaining([
      'weight 700 is not permitted', 'size 13px is not on the scale', 'line height 1.3 is not on the scale',
      "family var(--font-mono) is not the game's or the interface's face", 'colour literal; use a token',
      'radius 5px is not a permitted radius', 'labels are sentence case, not transformed', 'tracking is not in the system',
      "family 'Lato' is not permitted",
    ]));
    expect(why.some(reason => reason.startsWith('shadow 0 2px 6px'))).toBe(true);
    expect(lint('.b { font: 600 14px/20px var(--font-sans); box-shadow: 0 0 0 1px var(--border), var(--lift); border-radius: var(--radius-sm); }', 'ok.css', permitted)).toEqual([]);
  });
});

// Contrast: the text tokens on their grounds, from app.css's own values.

function tokens(css: string, opener: string): Map<string, string> {
  const start = css.indexOf(opener);
  let depth = 0;
  let end = start;
  for (let i = css.indexOf('{', start); i < css.length; i += 1) {
    if (css[i] === '{') depth += 1;
    if (css[i] === '}' && --depth === 0) { end = i; break; }
  }
  return new Map([...css.slice(start, end).matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)].map(m => [m[1], m[2].trim()]));
}

function hex(theme: Map<string, string>, light: Map<string, string>, name: string): string {
  const value = theme.get(name) ?? light.get(name);
  const ref = value && VAR.exec(value);
  if (ref) return hex(theme, light, ref[1]);
  if (!value || !/^#[0-9a-f]{6}$/i.test(value)) throw new Error(`${name} is not a six-digit colour: ${value}`);
  return value;
}

function luminance(colour: string): number {
  const channel = (i: number): number => {
    const c = Number.parseInt(colour.slice(1 + 2 * i, 3 + 2 * i), 16) / 255;
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * channel(0) + 0.7152 * channel(1) + 0.0722 * channel(2);
}

function contrast(a: string, b: string): number {
  const [x, y] = [luminance(a), luminance(b)].sort((p, q) => q - p);
  return (x + 0.05) / (y + 0.05);
}

describe('the text tokens', () => {
  const css = readFileSync(join(ui, 'src', 'app.css'), 'utf8');
  const light = tokens(css, ':root');
  const themes = { light, dark: tokens(css, '@media (prefers-color-scheme: dark)') };
  const pairs: [string, string][] = [['--on-accent', '--accent']];
  for (const ink of ['--text', '--text-secondary', '--text-tertiary', '--success-text', '--warning-text', '--error-text', '--info-text', '--neutral-text', '--planned-text']) {
    for (const ground of ['--bg', '--bg-surface', '--bg-muted']) pairs.push([ink, ground]);
  }
  for (const colour of ['w', 'u', 'b', 'r', 'g', 'c']) pairs.push([`--card-${colour}-ink`, `--card-${colour}`]);

  for (const [name, theme] of Object.entries(themes)) {
    it(`read at 4.5:1 or better in the ${name} theme`, () => {
      const low = pairs
        .map(([ink, ground]) => ({ pair: `${ink} on ${ground}`, ratio: contrast(hex(theme, light, ink), hex(theme, light, ground)) }))
        .filter(({ ratio }) => ratio < 4.5)
        .map(({ pair, ratio }) => `${pair} ${ratio.toFixed(2)}:1`);
      expect(low).toEqual([]);
    });
  }
});
