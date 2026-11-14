/** Display-only player identity. Recorded names, versions and filter values stay intact. */
import type { RecordedPlayer } from './types';

export interface PlayerIdentity {
  compact: string;
  full: string;
  /** Checkpoint SHA-256 only; control bots and humans have none. */
  digest: string | null;
  shortDigest: string | null;
  text: string;
  description: string;
  producer?: string;
  provenance?: Record<string, unknown>;
}

/** The ten-hex prefix every page shows for a recorded hash. */
export function shortHash(value: string): string {
  return value.slice(0, 10);
}

export function playerIdentity(player: {
  name: string;
  kind?: string;
  digest?: string | null;
  producer?: string;
  provenance?: Record<string, unknown>;
}): PlayerIdentity {
  const digest = player.digest ?? null;
  const parts = player.name.split('/');
  const compact = player.kind === 'bot' && parts.length >= 3
    ? `${parts[0]}/…/${parts.at(-1)}` : player.name;
  const shortDigest = digest ? shortHash(digest) : null;
  return { compact, full: player.name, digest, shortDigest,
    text: compact + (shortDigest ? ` · ${shortDigest}` : ''),
    description: player.name + (digest ? ` · ${digest}` : ''),
    producer: player.producer, provenance: player.provenance };
}

/**
 * A recorded player's `version` is the checkpoint SHA-256 only for a checkpoint;
 * for control bots it hashes inference settings, so it is never shown as a digest.
 */
export function recordedIdentity(
  player: Pick<RecordedPlayer, 'name' | 'kind' | 'version' | 'player_id'> & { inference?: Record<string, unknown> | null },
  extra: { producer?: string; provenance?: Record<string, unknown> } = {},
): PlayerIdentity {
  const checkpoint = player.inference?.policy === 'checkpoint' || player.player_id.startsWith('checkpoint:');
  return playerIdentity({ name: player.name, kind: player.kind, digest: checkpoint ? player.version : null, ...extra });
}
