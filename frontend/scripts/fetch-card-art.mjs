#!/usr/bin/env node
// Fetch card art for every curated-pack identity from Scryfall, once, into
// src/lib/card-art/ (bundler-known, so missing files are never requested). The frontend layers these over the procedural treatments
// (which remain the fallback), so play never depends on Scryfall at runtime
// and Scryfall is never queried per-session — versioned content, not
// opportunistic runtime fetches.
//
// Etiquette per https://scryfall.com/docs/api: identified User-Agent,
// >=100ms between requests, cached results are never re-fetched.
import { mkdir, readFile, writeFile, access } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const manifestPath = path.join(
  here,
  '../src/lib/packs/tla-ur-lessons-vs-gw-allies/v1/manifest.json',
);
const outDir = path.join(here, '../src/lib/card-art');

const USER_AGENT = 'EtudeFantasia/0.1 (research fan project; github.com/loopflowstudio/etude)';
const DELAY_MS = 150;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export function artSlug(name) {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '');
}

async function exists(file) {
  try {
    await access(file);
    return true;
  } catch {
    return false;
  }
}

async function scryfall(url) {
  const response = await fetch(url, {
    headers: { 'User-Agent': USER_AGENT, Accept: 'application/json' },
  });
  if (!response.ok) {
    return null;
  }
  return response.json();
}

async function findCard(name, kind) {
  if (kind === 'token') {
    // Tokens hide behind layout:token; exact-name lookup misses them.
    const result = await scryfall(
      `https://api.scryfall.com/cards/search?q=${encodeURIComponent(`!"${name}" include:extras`)}`,
    );
    return result?.data?.find((card) => card.layout === 'token') ?? result?.data?.[0] ?? null;
  }
  return scryfall(
    `https://api.scryfall.com/cards/named?exact=${encodeURIComponent(name)}`,
  );
}

function artCropUri(card) {
  return card?.image_uris?.art_crop ?? card?.card_faces?.[0]?.image_uris?.art_crop ?? null;
}

const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
const identities = Object.entries(manifest.identities).map(([name, identity]) => ({
  name,
  kind: identity.kind,
}));

await mkdir(outDir, { recursive: true });

let fetched = 0;
let cached = 0;
let missing = 0;

// The classic Deckmaster card back, from Scryfall's canonical back image.
const BACK_URI = 'https://backs.scryfall.io/normal/0/a/0aeebaf5-8c7d-4636-9e82-8c27447861f7.jpg';
const backFile = path.join(outDir, 'card-back.jpg');
if (await exists(backFile)) {
  cached += 1;
} else {
  await sleep(DELAY_MS);
  const back = await fetch(BACK_URI, { headers: { 'User-Agent': USER_AGENT } });
  if (back.ok) {
    await writeFile(backFile, Buffer.from(await back.arrayBuffer()));
    console.log('  ✓ card back (Deckmaster)');
    fetched += 1;
  } else {
    console.log(`  – card back: fetch failed (${back.status}); violet material stays`);
    missing += 1;
  }
}

for (const { name, kind } of identities) {
  const file = path.join(outDir, `${artSlug(name)}.jpg`);
  if (await exists(file)) {
    cached += 1;
    continue;
  }

  await sleep(DELAY_MS);
  const card = await findCard(name, kind);
  const uri = artCropUri(card);
  if (!uri) {
    console.log(`  – ${name}: not found on Scryfall (procedural treatment stays)`);
    missing += 1;
    continue;
  }

  await sleep(DELAY_MS);
  const image = await fetch(uri, { headers: { 'User-Agent': USER_AGENT } });
  if (!image.ok) {
    console.log(`  – ${name}: art fetch failed (${image.status})`);
    missing += 1;
    continue;
  }
  await writeFile(file, Buffer.from(await image.arrayBuffer()));
  console.log(`  ✓ ${name}`);
  fetched += 1;
}

console.log(
  `\ncard art: ${fetched} fetched, ${cached} already cached, ${missing} missing (fallback treatments cover them)`,
);
