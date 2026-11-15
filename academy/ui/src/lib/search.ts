export interface SearchConfig {
  algorithm: 'local_update';
  belief: 'physical_deal';
  worlds: number;
  depth: number;
  decision_seconds: number;
  alpha: number;
  beta: number;
}

export interface PendingSearch {
  code: string;
  message: string;
  match_id: string;
  expected_revision: number;
}

export function parseSearchConfigs(value: unknown): [SearchConfig | null, SearchConfig | null] {
  function valid(spec: unknown): spec is SearchConfig | null {
    if (spec === null) return true;
    if (typeof spec !== 'object' || spec === null) return false;
    const fields = spec as Record<string, unknown>;
    return fields.algorithm === 'local_update' && fields.belief === 'physical_deal'
      && typeof fields.worlds === 'number' && Number.isInteger(fields.worlds) && fields.worlds >= 1 && fields.worlds <= 64
      && typeof fields.depth === 'number' && Number.isInteger(fields.depth) && fields.depth >= 1 && fields.depth <= 40
      && typeof fields.decision_seconds === 'number' && fields.decision_seconds > 0 && fields.decision_seconds <= 30
      && typeof fields.alpha === 'number' && Number.isFinite(fields.alpha) && fields.alpha >= 0
      && typeof fields.beta === 'number' && Number.isFinite(fields.beta) && fields.beta > 0;
  }
  if (!Array.isArray(value) || value.length !== 2 || !valid(value[0]) || !valid(value[1])) {
    throw new Error('Invalid two-seat Search settings.');
  }
  return [value[0], value[1]];
}
