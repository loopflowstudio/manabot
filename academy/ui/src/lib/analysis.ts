/**
 * An analysis: what a policy would have done at each of your decisions in a
 * played game (`managym/academy/analysis.py`), read beside what you did.
 *
 * Nothing here grades the person. A decision is "same" when you played the
 * policy's first choice and "differs" otherwise; the two estimates after your
 * choice and after its own sit on one 0–100 scale from your seat.
 */
import type { ExperienceFrame, PolicyRuntimeIdentity } from './types';

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

/** A policy that may read a game, by registered id. */
export interface AnalysisPolicy { sha256: string; name: string }

/** Who may analyse a game (`/analyses/policies`), read once when Review opens. */
export interface AnalysisChoices {
  policies: AnalysisPolicy[];
  /** The opponent's own policy, when it may analyse the game. */
  default: string | null;
  /** Why no policy may analyse this game. */
  unavailable: string | null;
  /** The events an analysis of your seat reads. */
  events: number[];
}

// ———————————————————————————————————————— Analysis while you play

/**
 * The server's answer to Show analysis (`analysis_state`): whether it is
 * shown, by which policy, which policies may read this game, and why none
 * may.
 */
export interface LiveAnalysisState {
  type: 'analysis_state';
  shown: boolean;
  policy: AnalysisPolicy | null;
  policies: AnalysisPolicy[];
  unavailable: string | null;
}

/**
 * One live reading (`analysis`), sent after the decision's frame and never
 * inside it: the policy's probabilities and estimates at the decision
 * published at `revision`, or why the analysis stopped.
 */
export type LiveAnalysisMessage = { type: 'analysis'; policy: AnalysisPolicy } & (
  | { event: number; revision: number; annotation: AnalysedDecision['annotation']; semantic: string | null; after: (number | null)[] }
  | { stopped: string }
  /** The line so far, sent once when analysis turns on: the policy's score at each decision you already made. */
  | { history: { event: number; turn: number; label: string; score: number | null; semantic: string | null }[] }
);

/** What Play holds of a live analysis. Off at the start of every game. */
export interface LiveAnalysis {
  shown: boolean;
  /** A Show analysis request is on its way; the switch waits for its answer. */
  pending: boolean;
  policy: AnalysisPolicy | null;
  /** Known once the server has answered a request. */
  policies: AnalysisPolicy[] | null;
  unavailable: string | null;
  stopped: string | null;
  /** The latest reading, for the decision published at its revision. */
  reading: { revision: number; decision: AnalysedDecision } | null;
  /**
   * The game's line so far: the policy's score at each of your decisions read
   * this game, in order. It survives turning analysis off and on, and starts
   * again with another policy, whose scores are another line.
   */
  points: LivePoint[];
}

/**
 * One point of the live line: your decision at trace `event`. A decision read
 * as you faced it carries `at`, the action log's length then, which places it
 * after the last move logged. One read after the fact (`history`) carries its
 * `turn` and the move you chose there instead.
 */
export interface LivePoint { event: number; at: number | null; score: number | null; semantic: string | null; turn?: number; label?: string }

export const LIVE_ANALYSIS_OFF: LiveAnalysis = { shown: false, pending: false, policy: null, policies: null, unavailable: null, stopped: null, reading: null, points: [] };

/**
 * The next live analysis after a server message. A reading arriving after the
 * person turned it off is dropped. `logLength` is the action log's length when
 * the message arrives, which places a reading's point on the game's line.
 */
export function nextLiveAnalysis(current: LiveAnalysis, message: LiveAnalysisState | LiveAnalysisMessage, logLength = 0): LiveAnalysis {
  if (message.type === 'analysis_state') {
    const samePolicy = message.policy === null || message.policy.sha256 === current.policy?.sha256;
    return { ...LIVE_ANALYSIS_OFF, shown: message.shown, policy: message.policy ?? current.policy, policies: message.policies, unavailable: message.unavailable, points: samePolicy ? current.points : [] };
  }
  if ('history' in message) {
    // The line so far can outrun the answer to the request that asked for it;
    // holding it under its policy lets that answer keep it.
    if (!current.shown && !current.pending) return current;
    const same = current.policy?.sha256 === message.policy.sha256;
    const read = new Set(message.history.map(point => point.event));
    const earlier = message.history.map(point => ({ ...point, at: null }));
    return { ...current, policy: message.policy, points: [...earlier, ...(same ? current.points.filter(point => !read.has(point.event)) : [])] };
  }
  if (!current.shown || current.pending) return current;
  if ('stopped' in message) return { ...current, stopped: message.stopped, reading: null };
  const point = { event: message.event, at: logLength, score: score(message.annotation.value, message.semantic), semantic: message.semantic };
  return {
    ...current,
    stopped: null,
    reading: { revision: message.revision, decision: { event: message.event, played: -1, annotation: message.annotation, semantic: message.semantic, after: message.after } },
    points: [...current.points.filter(earlier => earlier.event !== message.event), point],
  };
}

/**
 * Follow the server's frames: each carries the live analysis's policy while it
 * is shown, so a reloaded page reads it as on, and a new game reads it as off.
 */
export function followShown(current: LiveAnalysis, policy: AnalysisPolicy | null): LiveAnalysis {
  if (current.pending) return current;
  if (policy === null) return current.shown ? { ...LIVE_ANALYSIS_OFF, policy: current.policy, policies: current.policies, points: current.points } : current;
  return current.shown && current.policy?.sha256 === policy.sha256 ? current : { ...current, shown: true, policy };
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

/**
 * How your choices in one move sit beside the policy's. A move can hold
 * several decisions (a cast, its target, its payment): `of` counts the
 * analysed ones and `agreed` those where you chose its first choice, so the
 * marks add up to the summary's count. The estimates are the move's first
 * decision's.
 */
export interface DecisionMark {
  agreed: number;
  of: number;
  /** Scores after your choice and after the policy's first choice, 0–1. */
  yours: number | null;
  its: number | null;
}

/** How your choice sits beside the policy's. */
export function decisionMark(decision: AnalysedDecision): DecisionMark | null {
  const first = firstChoice(decision);
  if (first === null) return null;
  return { agreed: first === decision.played ? 1 : 0, of: 1, yours: decision.after[decision.played] ?? null, its: decision.after[first] ?? null };
}

/** The mark's words: "same", "differs", or "1 of 3 same" for a move whose decisions split. */
export function markText(mark: DecisionMark): string {
  if (mark.agreed === mark.of) return 'same';
  if (mark.agreed === 0) return 'differs';
  return `${mark.agreed} of ${mark.of} same`;
}

/** "Analysing 34 of 81", "Analysis stopped at decision 12: …", or null once complete. */
export function analysisStatus(analysis: Analysis): string | null {
  if (analysis.status === 'running') return `Analysing ${analysis.decisions.length} of ${analysis.total}`;
  if (analysis.status === 'stopped') return `Analysis stopped after ${analysis.decisions.length} of ${analysis.total} decisions${analysis.reason ? `: ${analysis.reason}` : ''}`;
  return null;
}

// ———————————————— Key moments

export interface KeyMoment {
  /** The trace event of the decision. */
  event: number;
  kind: 'differs' | 'moved';
  /** What makes it a key moment, in words: a difference, never a grade. */
  reason: string;
}

/** Up to this many of each kind of key moment. */
const KEY_MOMENTS = { differs: 5, moved: 3 };
/** Smaller differences and movements are no key moment (an untrained policy's flat 50/50). */
const LEAST = 0.05;

/**
 * Where reading the game beside the policy is most worth a look, in game
 * order: the decisions where its first choice had the most probability over
 * yours, and where its estimate from your seat moved most between one of your
 * decisions and the next. A decision that is both is listed once, as a difference.
 */
export function keyMoments(analysis: Analysis, seatSwaps: readonly { event: number }[] = []): KeyMoment[] {
  const differs: (KeyMoment & { size: number })[] = [];
  const moved: (KeyMoment & { size: number })[] = [];
  analysis.decisions.forEach((decision, i) => {
    const p = decision.annotation.probabilities;
    const first = firstChoice(decision);
    if (p && first !== null && first !== decision.played) {
      const size = p[first] - (p[decision.played] ?? 0);
      if (size >= LEAST) differs.push({ event: decision.event, kind: 'differs', size, reason: `Differs · ${percent(p[first])} on its first choice, ${percent(p[decision.played] ?? 0)} on yours` });
    }
    const next = analysis.decisions[i + 1];
    const crossedSwap = next !== undefined && seatSwaps.some(swap => decision.event < swap.event && swap.event <= next.event);
    const here = score(decision.annotation.value, decision.semantic);
    const then = next && !crossedSwap ? score(next.annotation.value, next.semantic) : null;
    if (here !== null && then !== null && Math.abs(then - here) >= LEAST) {
      const name = scoreName(decision.semantic)?.toLowerCase() ?? 'estimate';
      moved.push({ event: decision.event, kind: 'moved', size: Math.abs(then - here), reason: `Its ${name} moved ${formatScore(here)} → ${formatScore(then)} by your next decision` });
    }
  });
  const largest = <T extends { size: number }>(items: T[], count: number): T[] => [...items].sort((a, b) => b.size - a.size).slice(0, count);
  const chosen = largest(differs, KEY_MOMENTS.differs);
  const taken = new Set(chosen.map(moment => moment.event));
  chosen.push(...largest(moved.filter(moment => !taken.has(moment.event)), KEY_MOMENTS.moved));
  return chosen.sort((a, b) => a.event - b.event).map(({ event, kind, reason }) => ({ event, kind, reason }));
}

function percent(p: number): string { return `${Math.round(p * 100)}%`; }

// ———————————————— Its line

/** A move of a line (`managym/academy/analysis.py` `LineStep`); `projection` is your own view after it. */
export interface LineStep {
  actor: 'agent' | 'opponent';
  description: string;
  projection: ExperienceFrame['projection'];
}

/** The policy's first choice at a decision, played on (`AgentLine`): one play-out, nothing saved. */
export interface AgentLine {
  event: number;
  /** The offer index the policy chose. */
  offer: number;
  steps: LineStep[];
  end: 'game_over' | 'turn' | 'limit';
  winner: number | null;
}
