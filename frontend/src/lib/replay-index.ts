import type {
  Command,
  ExperienceFrame,
  InteractionOffer,
  PresentationEvent,
} from './types';
import { validatePresentationTail } from './presentation';

export const CANONICAL_REPLAY_VERSION = 1 as const;

export type ReplayDecisionSource = 'client' | 'policy';

export interface ReplayDecision {
  ordinal: number;
  viewer: number;
  source: ReplayDecisionSource;
  revision: number;
  prompt_id: number;
  offer_id: number;
  command_id: string;
  presentation_cursor: number;
  frame: ExperienceFrame;
  offer: InteractionOffer;
  command: Command;
}

/** The only canonical replay shape allowed across a client boundary. */
export interface CanonicalReplayProjectionV1 {
  version: typeof CANONICAL_REPLAY_VERSION;
  replay_id: string;
  match_id: string;
  content_hash: string;
  asset_manifest_hash: string;
  viewer: number;
  decisions: ReplayDecision[];
  presentation_head: number;
  presentation: PresentationEvent[];
}

export interface AddressedReplayDecision extends ReplayDecision {
  address: string;
}

export interface CanonicalReplayProjectionResponseV1
  extends Omit<CanonicalReplayProjectionV1, 'decisions'> {
  decisions: AddressedReplayDecision[];
}

export interface RestoredReplayDecision {
  address: string;
  ordinal: number;
  viewer: number;
  revision: number;
  presentation_cursor: number;
  frame: ExperienceFrame;
  offer: InteractionOffer;
  command: Command;
  continuation: PresentationEvent[];
}

/** Decimal strings retain every u64 without passing through JS Number. */
export interface ReplayDecisionAddress {
  version: typeof CANONICAL_REPLAY_VERSION;
  replay_id: string;
  match_id: string;
  ordinal: string;
  viewer: string;
  revision: string;
  prompt_id: string;
  offer_id: string;
  command_id: string;
  presentation_cursor: string;
  decision_sha256: string;
}

const DECIMAL = /^(0|[1-9][0-9]*)$/;
const SHA256 = /^[0-9a-f]{64}$/;

function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (value !== null && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, child]) => [key, canonical(child)]),
    );
  }
  return value;
}

function sameJson(left: unknown, right: unknown): boolean {
  return JSON.stringify(canonical(left)) === JSON.stringify(canonical(right));
}

function safeInteger(value: number, field: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new Error(`invalid canonical replay: ${field} is not a safe non-negative integer`);
  }
}

function encodeBase64Url(bytes: Uint8Array): string {
  let binary = '';
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replaceAll('+', '-').replaceAll('/', '_').replace(/=+$/, '');
}

function decodeBase64Url(value: string): Uint8Array {
  if (value.length === 0 || value.includes('=') || !/^[A-Za-z0-9_-]+$/.test(value)) {
    throw new Error('invalid replay decision address');
  }
  const padded = value.replaceAll('-', '+').replaceAll('_', '/')
    + '='.repeat((4 - (value.length % 4)) % 4);
  const binary = atob(padded);
  return Uint8Array.from(binary, (character) => character.charCodeAt(0));
}

export function serializeReplayDecisionAddress(address: ReplayDecisionAddress): string {
  const numeric = [
    address.ordinal,
    address.viewer,
    address.revision,
    address.prompt_id,
    address.offer_id,
    address.presentation_cursor,
  ];
  if (
    numeric.some((value) => !DECIMAL.test(value))
    || !address.replay_id
    || !address.match_id
    || !address.command_id
    || !SHA256.test(address.decision_sha256)
  ) {
    throw new Error('invalid replay decision address');
  }
  const payload = [
    1,
    address.replay_id,
    address.match_id,
    address.ordinal,
    address.viewer,
    address.revision,
    address.prompt_id,
    address.offer_id,
    address.command_id,
    address.presentation_cursor,
    address.decision_sha256,
  ];
  return `erd1.${encodeBase64Url(new TextEncoder().encode(JSON.stringify(payload)))}`;
}

export function parseReplayDecisionAddress(value: string): ReplayDecisionAddress {
  try {
    if (!value.startsWith('erd1.')) throw new Error('prefix');
    const payload = JSON.parse(
      new TextDecoder('utf-8', { fatal: true }).decode(decodeBase64Url(value.slice(5))),
    ) as unknown;
    if (!Array.isArray(payload) || payload.length !== 11 || payload[0] !== 1) {
      throw new Error('shape');
    }
    if (payload.slice(1).some((item) => typeof item !== 'string')) {
      throw new Error('type');
    }
    const address: ReplayDecisionAddress = {
      version: 1,
      replay_id: payload[1] as string,
      match_id: payload[2] as string,
      ordinal: payload[3] as string,
      viewer: payload[4] as string,
      revision: payload[5] as string,
      prompt_id: payload[6] as string,
      offer_id: payload[7] as string,
      command_id: payload[8] as string,
      presentation_cursor: payload[9] as string,
      decision_sha256: payload[10] as string,
    };
    if (serializeReplayDecisionAddress(address) !== value) throw new Error('canonical');
    return address;
  } catch (error) {
    throw new Error('invalid replay decision address', { cause: error });
  }
}

export function assertAddressBindsReplayDecision(
  address: ReplayDecisionAddress,
  projection: CanonicalReplayProjectionV1,
  row: ReplayDecision,
): void {
  if (
    address.replay_id !== projection.replay_id
    || address.match_id !== projection.match_id
    || address.ordinal !== String(row.ordinal)
    || address.viewer !== String(row.viewer)
    || address.revision !== String(row.revision)
    || address.prompt_id !== String(row.prompt_id)
    || address.offer_id !== String(row.offer_id)
    || address.command_id !== row.command_id
    || address.presentation_cursor !== String(row.presentation_cursor)
  ) {
    throw new Error('replay decision address identity drifted');
  }
}

export function assertViewerSafeReplayProjection(
  projection: CanonicalReplayProjectionV1,
): void {
  if (projection.version !== 1) throw new Error('invalid canonical replay version');
  if (!projection.replay_id) throw new Error('canonical replay projection requires replay_id');
  safeInteger(projection.viewer, 'viewer');
  safeInteger(projection.presentation_head, 'presentation_head');
  let previousOrdinal = -1;
  let previousCursor = 0;
  const commandIds = new Set<string>();
  const promptIds = new Set<string>();
  for (const row of projection.decisions) {
    for (const [field, value] of Object.entries({
      ordinal: row.ordinal,
      viewer: row.viewer,
      revision: row.revision,
      prompt_id: row.prompt_id,
      offer_id: row.offer_id,
      presentation_cursor: row.presentation_cursor,
    })) safeInteger(value, field);
    if (row.viewer !== projection.viewer) {
      throw new Error('canonical replay projection mixes viewer rows');
    }
    if (row.ordinal <= previousOrdinal) {
      throw new Error('canonical replay projection ordinals are not increasing');
    }
    if (
      row.presentation_cursor < previousCursor
      || row.presentation_cursor > projection.presentation_head
    ) {
      throw new Error('canonical replay projection cursor drifted');
    }
    const prompt = row.frame.prompt;
    const selected = row.frame.offers.find(({ id }) => id === row.offer_id);
    if (
      row.frame.match_id !== projection.match_id
      || row.frame.content_hash !== projection.content_hash
      || row.frame.asset_manifest_hash !== projection.asset_manifest_hash
      || prompt === null
      || prompt.id !== row.prompt_id
      || prompt.actor !== row.viewer
      || row.frame.revision !== row.revision
      || row.frame.projection.agent.player_index !== row.viewer
      || row.frame.projection.opponent.hand.length !== 0
      || row.offer.id !== row.offer_id
      || row.offer.actor !== row.viewer
      || selected === undefined
      || !sameJson(selected, row.offer)
      || row.command.command_id !== row.command_id
      || row.command.match_id !== row.frame.match_id
      || row.command.expected_revision !== row.revision
      || row.command.prompt_id !== row.prompt_id
      || row.command.offer_id !== row.offer_id
    ) {
      throw new Error('canonical replay decision binding drifted');
    }
    const promptKey = `${row.revision}:${row.prompt_id}`;
    if (commandIds.has(row.command_id) || promptIds.has(promptKey)) {
      throw new Error('canonical replay projection has duplicate identity');
    }
    commandIds.add(row.command_id);
    promptIds.add(promptKey);
    previousOrdinal = row.ordinal;
    previousCursor = row.presentation_cursor;
  }
  if (validatePresentationTail(projection.presentation, 0) !== projection.presentation_head) {
    throw new Error('canonical replay projection presentation head drifted');
  }
}
