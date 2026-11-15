/** Display-only player identity. Recorded names, versions and filter values stay intact. */
import type { CheckpointProvenance, PolicyRuntimeIdentity, RecordedPlayer } from './types';

export interface PlayerIdentity {
  compact: string;
  full: string;
  /** Policy ID for registered runtimes, file SHA-256 for historical checkpoints. */
  digest: string | null;
  /** Compact name with the short digest: the one-line form for prose and labels. */
  text: string;
  /** Full name with the full digest. */
  description: string;
  producer?: string;
  provenance?: CheckpointProvenance;
  runtime?: PolicyRuntimeIdentity;
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
  provenance?: CheckpointProvenance;
  runtime?: PolicyRuntimeIdentity;
}): PlayerIdentity {
  const digest = player.digest ?? null;
  const parts = player.name.split('/');
  const compact = player.kind === 'bot' && parts.length >= 3
    ? `${parts[0]}/…/${parts.at(-1)}` : player.name;
  return { compact, full: player.name, digest,
    text: compact + (digest ? ` · ${shortHash(digest)}` : ''),
    description: player.name + (digest ? ` · ${digest}` : ''),
    producer: player.producer, provenance: player.provenance, runtime: player.runtime };
}

/** A checkpoint as the catalog, a table snapshot or a trace's `opponent` names it. */
export function checkpointIdentity(checkpoint: { name: string; sha256: string; producer?: string }): PlayerIdentity {
  return playerIdentity({ name: checkpoint.name, kind: 'bot', digest: checkpoint.sha256, producer: checkpoint.producer });
}

/**
 * A recorded player's `version` is the runtime Policy ID or historical checkpoint SHA-256;
 * for control bots it hashes inference settings, so it is never shown as a digest.
 */
export function recordedIdentity(
  player: Pick<RecordedPlayer, 'name' | 'kind' | 'version' | 'player_id'> & { inference?: (Record<string, unknown> & { runtime?: PolicyRuntimeIdentity }) | null },
): PlayerIdentity {
  const checkpoint = ['checkpoint', 'policy'].includes(String(player.inference?.policy)) || player.player_id.startsWith('checkpoint:') || player.player_id.startsWith('policy:');
  return playerIdentity({ name: player.name, kind: player.kind, digest: checkpoint ? player.version : null, runtime: player.inference?.runtime });
}
