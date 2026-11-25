/**
 * An analysis: what a policy would have done at each of your decisions in a
 * played game (`managym/academy/analysis.py`), read beside what you did.
 *
 * Nothing here grades the person. A decision is "same" when you played the
 * policy's first choice and "differs" otherwise; the two estimates after your
 * choice and after its own sit on one 0–100 scale from your seat.
 */
import type { PolicyRuntimeIdentity } from './types';

export interface AnalysedDecision {
  /** The trace event the decision is (`Position.event`). */
  event: number;
  /** The offer index you played. */
  played: number;
  /** Probabilities in the decision's offer order; the value at the decision from your seat. */
  annotation: { probabilities: number[] | null; value: number | null };
  semantic: string | null;
  /** The 0–1 score from your seat after each offer, in offer order; null where unread. */
  after: (number | null)[];
}

export interface Analysis {
  id: string;
  seat: number;
  policy: PolicyRuntimeIdentity & { name: string };
  status: 'running' | 'complete' | 'stopped';
  reason: string | null;
  /** Decisions the analysis reads. */
  total: number;
  created_at: string;
  decisions: AnalysedDecision[];
}

/** Who may analyse a game (`/analyses/policies`), read once when Review opens. */
export interface AnalysisChoices {
  /** Policies that may analyse this game, by registered id. */
  policies: { sha256: string; name: string }[];
  /** The opponent's own policy, when it may analyse the game. */
  default: string | null;
  /** Why no policy may analyse this game. */
  unavailable: string | null;
  /** The events an analysis of your seat reads. */
  events: number[];
}

/** What the 0–100 line measures, by the policy's value semantic; null without a scale. */
export function scoreName(semantic: string | null): string | null {
  return semantic === 'signed_outcome' ? 'Expected score' : semantic === 'win_logit' ? 'Win chance' : null;
}

/** A policy value as a 0–1 score from the analysed seat, as the server maps it. */
export function score(value: number | null, semantic: string | null): number | null {
  if (value === null) return null;
  if (semantic === 'signed_outcome') return Math.min(1, Math.max(0, (value + 1) / 2));
  if (semantic === 'win_logit') return 1 / (1 + Math.exp(-Math.max(-60, Math.min(60, value))));
  return null;
}

/** "54": a score on the 0–100 scale. */
export function formatScore(s: number): string { return String(Math.round(s * 100)); }

/** An analysis's decisions by trace event. */
export function byEvent(analysis: Analysis | null): Map<number, AnalysedDecision> {
  return new Map((analysis?.decisions ?? []).map(decision => [decision.event, decision]));
}

/** The policy's first choice at a decision: the offer index of its largest probability. */
export function firstChoice(decision: AnalysedDecision): number | null {
  const p = decision.annotation.probabilities;
  if (!p?.length) return null;
  return p.reduce((best, value, i) => (value > p[best] ? i : best), 0);
}

export interface DecisionMark {
  same: boolean;
  /** Scores after your choice and after the policy's first choice, 0–1. */
  yours: number | null;
  its: number | null;
}

/** How your choice sits beside the policy's. */
export function decisionMark(decision: AnalysedDecision): DecisionMark | null {
  const first = firstChoice(decision);
  if (first === null) return null;
  return { same: first === decision.played, yours: decision.after[decision.played] ?? null, its: decision.after[first] ?? null };
}

/** "Analysing 34 of 81", "Analysis stopped at decision 12: …", or null once complete. */
export function analysisStatus(analysis: Analysis): string | null {
  if (analysis.status === 'running') return `Analysing ${analysis.decisions.length} of ${analysis.total}`;
  if (analysis.status === 'stopped') return `Analysis stopped after ${analysis.decisions.length} of ${analysis.total} decisions${analysis.reason ? `: ${analysis.reason}` : ''}`;
  return null;
}
