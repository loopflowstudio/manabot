import manifestData from './packs/tla-ur-lessons-vs-gw-allies/v1/manifest.json';

export interface PackDeck {
  deck_id: string;
  display_name: string;
  card_count: number;
  cards: Record<string, number>;
}

export interface PackTreatment {
  palette: [string, string, string];
  motif: string;
  seed: number;
  rights_ref: string;
}

export interface PackIdentity {
  kind: 'card' | 'token';
  provenance: Record<string, unknown>;
  treatment: PackTreatment;
}

export interface CuratedPackManifest {
  schema_version: 1;
  pack: { id: string; version: string; title: string };
  matchup: {
    hero: PackDeck;
    villain: PackDeck;
    reachable_tokens: string[];
  };
  rights: Record<string, Record<string, unknown>>;
  fallback: {
    version: 'fallback-v1';
    algorithm: 'fnv1a-32-utf8';
    rights_ref: string;
    palettes: [string, string, string][];
    motifs: string[];
  };
  identities: Record<string, PackIdentity>;
}

export interface ResolvedTreatment extends PackTreatment {
  source: 'pack' | 'fallback';
  identityKind: 'card' | 'token' | 'unknown';
}

const HEX_COLOR = /^#[0-9a-f]{6}$/i;
const REMOTE_URL = /^(?:https?:)?\/\//i;
const RIGHTS_FIELDS = [
  'asset_kind',
  'contains_third_party_art',
  'creator',
  'copyright_notice',
  'license',
] as const;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function requireRecord(value: unknown, path: string): Record<string, unknown> {
  if (!isRecord(value)) {
    throw new Error(`${path} must be an object`);
  }
  return value;
}

function requireString(value: unknown, path: string): string {
  if (typeof value !== 'string' || value.length === 0) {
    throw new Error(`${path} must be a non-empty string`);
  }
  return value;
}

function requirePalette(value: unknown, path: string): [string, string, string] {
  if (
    !Array.isArray(value) ||
    value.length !== 3 ||
    !value.every((color) => typeof color === 'string' && HEX_COLOR.test(color))
  ) {
    throw new Error(`${path} must contain three #RRGGBB colors`);
  }
  return value as [string, string, string];
}

function containsRemoteValue(value: unknown): boolean {
  if (typeof value === 'string') {
    return REMOTE_URL.test(value);
  }
  if (Array.isArray(value)) {
    return value.some(containsRemoteValue);
  }
  if (isRecord(value)) {
    return Object.values(value).some(containsRemoteValue);
  }
  return false;
}

function validateDeck(value: unknown, path: string): PackDeck {
  const deck = requireRecord(value, path);
  const cards = requireRecord(deck.cards, `${path}.cards`);
  let total = 0;
  for (const [name, count] of Object.entries(cards)) {
    requireString(name, `${path}.cards key`);
    if (!Number.isInteger(count) || (count as number) <= 0) {
      throw new Error(`${path}.cards.${name} must be a positive integer`);
    }
    total += count as number;
  }
  if (!Number.isInteger(deck.card_count) || (deck.card_count as number) <= 0) {
    throw new Error(`${path}.card_count must be a positive integer`);
  }
  if (total !== deck.card_count) {
    throw new Error(`${path} declares ${deck.card_count} cards but contains ${total}`);
  }
  return {
    deck_id: requireString(deck.deck_id, `${path}.deck_id`),
    display_name: requireString(deck.display_name, `${path}.display_name`),
    card_count: deck.card_count as number,
    cards: cards as Record<string, number>,
  };
}

export function validateCuratedPack(value: unknown): CuratedPackManifest {
  const root = requireRecord(value, 'manifest');
  if (root.schema_version !== 1) {
    throw new Error('manifest.schema_version must be 1');
  }

  const pack = requireRecord(root.pack, 'manifest.pack');
  const rights = requireRecord(root.rights, 'manifest.rights');
  for (const [rightsId, rawRights] of Object.entries(rights)) {
    const record = requireRecord(rawRights, `manifest.rights.${rightsId}`);
    for (const field of RIGHTS_FIELDS) {
      if (!(field in record)) {
        throw new Error(`manifest.rights.${rightsId}.${field} is required`);
      }
    }
    requireString(record.asset_kind, `manifest.rights.${rightsId}.asset_kind`);
    requireString(record.creator, `manifest.rights.${rightsId}.creator`);
    requireString(record.copyright_notice, `manifest.rights.${rightsId}.copyright_notice`);
    requireString(record.license, `manifest.rights.${rightsId}.license`);
    if (typeof record.contains_third_party_art !== 'boolean') {
      throw new Error(`manifest.rights.${rightsId}.contains_third_party_art must be boolean`);
    }
  }

  const matchup = requireRecord(root.matchup, 'manifest.matchup');
  const hero = validateDeck(matchup.hero, 'manifest.matchup.hero');
  const villain = validateDeck(matchup.villain, 'manifest.matchup.villain');
  if (!Array.isArray(matchup.reachable_tokens) || matchup.reachable_tokens.length === 0) {
    throw new Error('manifest.matchup.reachable_tokens must be a non-empty list');
  }
  const reachableTokens = matchup.reachable_tokens.map((name, index) =>
    requireString(name, `manifest.matchup.reachable_tokens[${index}]`),
  );

  const fallback = requireRecord(root.fallback, 'manifest.fallback');
  if (fallback.version !== 'fallback-v1' || fallback.algorithm !== 'fnv1a-32-utf8') {
    throw new Error('manifest fallback version or algorithm is unsupported');
  }
  const fallbackRights = requireString(fallback.rights_ref, 'manifest.fallback.rights_ref');
  if (!(fallbackRights in rights)) {
    throw new Error(`manifest.fallback.rights_ref is unknown: ${fallbackRights}`);
  }
  if (!Array.isArray(fallback.palettes) || fallback.palettes.length === 0) {
    throw new Error('manifest.fallback.palettes must be a non-empty list');
  }
  const fallbackPalettes = fallback.palettes.map((palette, index) =>
    requirePalette(palette, `manifest.fallback.palettes[${index}]`),
  );
  if (!Array.isArray(fallback.motifs) || fallback.motifs.length === 0) {
    throw new Error('manifest.fallback.motifs must be a non-empty list');
  }
  const fallbackMotifs = fallback.motifs.map((motif, index) =>
    requireString(motif, `manifest.fallback.motifs[${index}]`),
  );

  const rawIdentities = requireRecord(root.identities, 'manifest.identities');
  const expectedNames = new Set([
    ...Object.keys(hero.cards),
    ...Object.keys(villain.cards),
    ...reachableTokens,
  ]);
  const actualNames = Object.keys(rawIdentities);
  if (
    actualNames.length !== expectedNames.size ||
    actualNames.some((name) => !expectedNames.has(name))
  ) {
    throw new Error('manifest identity inventory does not match the matchup');
  }

  const tokenNames = new Set(reachableTokens);
  const identities: Record<string, PackIdentity> = {};
  for (const [name, rawIdentity] of Object.entries(rawIdentities)) {
    const identity = requireRecord(rawIdentity, `manifest.identities.${name}`);
    const kind = tokenNames.has(name) ? 'token' : 'card';
    if (identity.kind !== kind) {
      throw new Error(`manifest.identities.${name}.kind must be ${kind}`);
    }
    const provenance = requireRecord(
      identity.provenance,
      `manifest.identities.${name}.provenance`,
    );
    requireString(provenance.provider, `manifest.identities.${name}.provenance.provider`);
    requireString(provenance.source_uri, `manifest.identities.${name}.provenance.source_uri`);
    requireString(provenance.retrieved_at, `manifest.identities.${name}.provenance.retrieved_at`);
    const provenanceRights = requireString(
      provenance.rights_ref,
      `manifest.identities.${name}.provenance.rights_ref`,
    );
    if (!(provenanceRights in rights)) {
      throw new Error(`manifest.identities.${name}.provenance.rights_ref is unknown`);
    }
    requireString(
      provenance[kind === 'token' ? 'local_identity' : 'oracle_id'],
      `manifest.identities.${name}.provenance identity`,
    );

    const rawTreatment = requireRecord(
      identity.treatment,
      `manifest.identities.${name}.treatment`,
    );
    if (containsRemoteValue(rawTreatment)) {
      throw new Error(`manifest.identities.${name}.treatment must be local`);
    }
    const treatmentRights = requireString(
      rawTreatment.rights_ref,
      `manifest.identities.${name}.treatment.rights_ref`,
    );
    if (!(treatmentRights in rights)) {
      throw new Error(`manifest.identities.${name}.treatment.rights_ref is unknown`);
    }
    if (!Number.isInteger(rawTreatment.seed)) {
      throw new Error(`manifest.identities.${name}.treatment.seed must be an integer`);
    }
    identities[name] = {
      kind,
      provenance,
      treatment: {
        palette: requirePalette(
          rawTreatment.palette,
          `manifest.identities.${name}.treatment.palette`,
        ),
        motif: requireString(
          rawTreatment.motif,
          `manifest.identities.${name}.treatment.motif`,
        ),
        seed: rawTreatment.seed as number,
        rights_ref: treatmentRights,
      },
    };
  }

  return {
    schema_version: 1,
    pack: {
      id: requireString(pack.id, 'manifest.pack.id'),
      version: requireString(pack.version, 'manifest.pack.version'),
      title: requireString(pack.title, 'manifest.pack.title'),
    },
    matchup: { hero, villain, reachable_tokens: reachableTokens },
    rights: rights as Record<string, Record<string, unknown>>,
    fallback: {
      version: 'fallback-v1',
      algorithm: 'fnv1a-32-utf8',
      rights_ref: fallbackRights,
      palettes: fallbackPalettes,
      motifs: fallbackMotifs,
    },
    identities,
  };
}

export const CURATED_PACK = validateCuratedPack(manifestData);

export function fnv1a32(value: string): number {
  let hash = 0x811c9dc5;
  for (const byte of new TextEncoder().encode(value)) {
    hash ^= byte;
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash;
}

export function resolveTreatment(name: string): ResolvedTreatment {
  const identity = CURATED_PACK.identities[name];
  if (identity) {
    return {
      ...identity.treatment,
      source: 'pack',
      identityKind: identity.kind,
    };
  }

  const hash = fnv1a32(name);
  const palettes = CURATED_PACK.fallback.palettes;
  const motifs = CURATED_PACK.fallback.motifs;
  return {
    palette: palettes[hash % palettes.length],
    motif: motifs[Math.floor(hash / palettes.length) % motifs.length],
    seed: hash,
    rights_ref: CURATED_PACK.fallback.rights_ref,
    source: 'fallback',
    identityKind: 'unknown',
  };
}

export function treatmentBackground(treatment: ResolvedTreatment): string {
  const [shadow, midtone, highlight] = treatment.palette;
  const x = 18 + (treatment.seed % 65);
  const y = 14 + (Math.floor(treatment.seed / 7) % 68);
  const angle = 105 + (treatment.seed % 80);
  return [
    `radial-gradient(circle at ${x}% ${y}%, ${highlight} 0, transparent 34%)`,
    `linear-gradient(${angle}deg, ${shadow} 0%, ${midtone} 58%, ${shadow} 100%)`,
  ].join(', ');
}
