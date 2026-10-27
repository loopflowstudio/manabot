#!/usr/bin/env node
// Fetch card art for every curated-pack identity from Scryfall, once, into
// src/lib/card-art/ (bundler-known, so missing files are never requested). The frontend layers these over the procedural treatments
// (which remain the fallback), so play never depends on Scryfall at runtime
// and Scryfall is never queried per-session — versioned content, not
// opportunistic runtime fetches.
//
// Etiquette per https://scryfall.com/docs/api: identified User-Agent,
// >=100ms between requests, cached results are never re-fetched.
import { mkdir, readFile, writeFile, access, readdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const manifestPath = path.join(
  here,
  '../src/lib/packs/tla-ur-lessons-vs-gw-allies/v2/manifest.json',
);
const outDir = path.join(here, '../src/lib/card-art');
const fullDir = path.join(outDir, 'full');
const creditsFile = path.join(outDir, 'credits.json');

const USER_AGENT = 'Academy/0.1 (research fan project; github.com/loopflowstudio/manabot)';
const DELAY_MS = 1000;

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

async function request(url, accept) {
  for (let attempt = 0; attempt < 3; attempt += 1) {
    const response = await fetch(url, {
      headers: { 'User-Agent': USER_AGENT, ...(accept ? { Accept: accept } : {}) },
      signal: AbortSignal.timeout(20_000),
    });
    if (response.status !== 429) return response;
    const retryAfter = response.headers.get('Retry-After');
    const seconds = /^\d+$/.test(retryAfter ?? '') ? Number(retryAfter) :
      retryAfter ? Math.ceil((Date.parse(retryAfter) - Date.now()) / 1000) : 60;
    const delay = Number.isFinite(seconds) ? Math.max(60, seconds) : 60;
    console.log(`  Scryfall rate limit: waiting ${delay} seconds before retrying.`);
    await response.body?.cancel();
    await sleep(delay * 1000);
  }
  throw new Error('Scryfall remains rate limited. Cached artwork is retained; retry later.');
}

async function scryfall(url) {
  const response = await request(url, 'application/json');
  if (!response.ok) {
    if (response.status === 404) return null;
    throw new Error(`Scryfall card lookup failed (${response.status}): ${url}`);
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

const manifests = await Promise.all([manifestPath, path.join(here, '../src/lib/packs/elemental-world/v2/manifest.json')]
  .map(async filename => JSON.parse(await readFile(filename, 'utf8'))));
const identities = [...new Map(manifests.flatMap(manifest => Object.entries(manifest.identities)) )]
  .map(([name, identity]) => ({ name, kind: identity.kind }));
// Directed demos use ordinary cards outside the curated two-deck pack.
const fixtureDir = path.join(here, '../../fixtures');
const known = new Set(identities.map(identity => identity.name));
for (const filename of (await readdir(fixtureDir)).filter(name => name.includes('demo') && name.endsWith('.json'))) {
  const fixture = JSON.parse(await readFile(path.join(fixtureDir, filename), 'utf8'));
  for (const deck of [fixture.config?.hero_deck, fixture.config?.villain_deck]) {
    if (!deck || typeof deck !== 'object' || Array.isArray(deck)) continue;
    for (const name of Object.keys(deck)) {
      if (!known.has(name)) {
        identities.push({ name, kind: 'card' });
        known.add(name);
      }
    }
  }
}
if (!known.has('Treasure')) identities.push({ name: 'Treasure', kind: 'token' });

await mkdir(outDir, { recursive: true });
await mkdir(fullDir, { recursive: true });
const credits = await exists(creditsFile)
  ? JSON.parse(await readFile(creditsFile, 'utf8')) : {};

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
  const back = await request(BACK_URI);
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
  const fullFile = path.join(fullDir, `${artSlug(name)}.jpg`);
  if (await exists(file) && await exists(fullFile) && credits[name]) {
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
  const image = await request(uri);
  if (!image.ok) {
    console.log(`  – ${name}: art fetch failed (${image.status})`);
    missing += 1;
    continue;
  }
  await writeFile(file, Buffer.from(await image.arrayBuffer()));
  const fullUri = card?.image_uris?.normal ?? card?.card_faces?.[0]?.image_uris?.normal;
  if (fullUri) {
    await sleep(DELAY_MS);
    const fullImage = await request(fullUri);
    if (!fullImage.ok) throw new Error(`Card preview fetch failed (${fullImage.status}): ${name}`);
    await writeFile(fullFile, Buffer.from(await fullImage.arrayBuffer()));
  }
  credits[name] = {
    artist: card.artist ?? card.card_faces?.[0]?.artist ?? 'Artist not listed',
    source: card.scryfall_uri,
    copyright: '© Wizards of the Coast LLC',
  };
  await writeFile(creditsFile, `${JSON.stringify(credits, null, 2)}\n`);
  console.log(`  ✓ ${name}`);
  fetched += 1;
}

console.log(
  `\ncard art: ${fetched} fetched, ${cached} already cached, ${missing} missing (fallback treatments cover them)`,
);
