/** Retain selected execution identity so runner changes can be explained on restart. */
import type { PolicyRuntimeIdentity, TrainedOpponent } from './types';

const key = 'etude.policy.selections';

function selections(): PolicyRuntimeIdentity[] {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(key) ?? '[]');
    return Array.isArray(value) ? value.filter(p => p && typeof p.id === 'string'
      && typeof p.provider === 'string' && typeof p.runner === 'string'
      && p.artifacts && p.config) : [];
  } catch { return []; }
}

export function rememberedPolicy(id: string): PolicyRuntimeIdentity | null {
  return selections().find(p => p.id === id) ?? null;
}

export function rememberPolicy(policy: PolicyRuntimeIdentity): void {
  try {
    localStorage.setItem(key, JSON.stringify([
      ...selections().filter(p => p.id !== policy.id).slice(-63), policy,
    ]));
  } catch { /* Storage may be unavailable; selection itself remains usable. */ }
}

function equalRecord(a: Record<string, unknown>, b: Record<string, unknown>): boolean {
  // Config objects may contain nested records whose insertion order differs.
  function canonical(value: unknown): unknown {
    if (Array.isArray(value)) return value.map(canonical);
    if (value && typeof value === 'object') return Object.fromEntries(
      Object.entries(value).sort(([a], [b]) => a.localeCompare(b))
        .map(([key, value]) => [key, canonical(value)]));
    return value;
  }
  return JSON.stringify(canonical(a)) === JSON.stringify(canonical(b));
}

export function runnerReplacement(
  prior: PolicyRuntimeIdentity, entries: TrainedOpponent[],
): TrainedOpponent | null {
  const matches = entries.filter(({ policy: p }) => p && p.id !== prior.id
    && p.provider === prior.provider && p.runner !== prior.runner
    && p.contract_version === prior.contract_version
    && equalRecord(p.artifacts, prior.artifacts) && equalRecord(p.config, prior.config));
  return matches.length === 1 ? matches[0] : null;
}
