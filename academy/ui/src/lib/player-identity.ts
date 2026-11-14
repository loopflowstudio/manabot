/** Display-only player identity. Recorded names, versions and filter values stay intact. */
export interface PlayerIdentity {
  compact: string;
  full: string;
  digest: string | null;
  shortDigest: string | null;
  text: string;
  description: string;
  producer?: string;
  provenance?: Record<string, unknown>;
}

export function playerIdentity(player: {
  name: string;
  kind?: string;
  digest?: string | null;
  version?: string | null;
  producer?: string;
  provenance?: Record<string, unknown>;
}): PlayerIdentity {
  const digest = player.digest ?? player.version ?? null;
  const parts = player.name.split('/');
  const compact = player.kind === 'bot' && parts.length >= 3
    ? `${parts[0]}/…/${parts.at(-1)}` : player.name;
  const shortDigest = digest?.slice(0, 10) ?? null;
  return { compact, full: player.name, digest, shortDigest,
    text: compact + (shortDigest ? ` · ${shortDigest}` : ''),
    description: player.name + (digest ? ` · ${digest}` : ''),
    producer: player.producer, provenance: player.provenance };
}
