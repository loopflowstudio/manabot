/**
 * A whole combat declaration, drafted on the board before any of it is sent.
 *
 * The engine asks about one creature at a time. The board lets a person draft
 * the whole attack (which creatures attack) or block (which blocker blocks
 * which attacker), asks the native preview (`/combat-preview`, read-only) for
 * the exact per-creature offers that declare it, and on confirm sends those
 * offers one by one. `stepOffer` checks each new frame against the step the
 * preview predicted; any other frame stops the sequence. Nothing here is a
 * rules solver: which creatures may attack or block, and whether the plan is
 * legal, come from the preview alone.
 *
 * A draft names creatures by their display ids (`PermanentState.id`, the
 * preview's `ObjectRenderId.entity`).
 */
import type { ExperienceFrame, InteractionOffer } from '$lib/types';
import { PRIMARY_CHARS } from './prompt';

// ———————————————————————————————————————— The preview's records

export interface RenderId { entity: number; incarnation: number }
export interface PlannedBlock { blocker: RenderId; attacker: RenderId }
export interface CombatPlan { attackers: RenderId[]; blocks: PlannedBlock[] }
export interface DeclarableCreature { creature: RenderId; label: string; can_block: RenderId[]; must_block: boolean; must_attack: boolean }
export interface DeclarationStep {
  revision: number; fingerprint: string; creature: RenderId; action_index: number; attack: boolean; blocks: RenderId | null;
}
export interface CombatPreview {
  kind: 'attackers' | 'blockers';
  declarable: DeclarableCreature[];
  steps: DeclarationStep[];
  refused: { creature: RenderId; label: string; reason: string } | null;
  declared_attackers: RenderId[];
  declared_blocks: PlannedBlock[];
}
export interface CombatAddress { match_id: string; revision: number; prompt_id: number; frame_hash: string }
export interface CombatResponse extends CombatAddress { viewer: number; plan: CombatPlan; preview: CombatPreview }
export type CombatFetcher = (request: CombatAddress & { plan: CombatPlan }, signal: AbortSignal) => Promise<CombatResponse>;

// ———————————————————————————————————————— Drafts

/** Attackers in the attack, or blocker → attacker; display ids. */
export interface Draft { attackers: number[]; blocks: [number, number][] }
export const EMPTY_DRAFT: Draft = { attackers: [], blocks: [] };

export function isDeclaration(frame: ExperienceFrame): boolean {
  return frame.action_space === 'DECLARE_ATTACKER' || frame.action_space === 'DECLARE_BLOCKER';
}

export function addressOf(frame: ExperienceFrame): CombatAddress | null {
  return frame.prompt ? { match_id: frame.match_id, revision: frame.revision, prompt_id: frame.prompt.id, frame_hash: frame.frame_hash } : null;
}

/** Add the creatures caught by a sweep, keeping earlier selections and their order. */
export function addAttackers(draft: Draft, hits: ReadonlySet<number>): Draft {
  const attackers = new Set([...draft.attackers, ...hits]);
  return attackers.size === draft.attackers.length ? draft : { ...draft, attackers: [...attackers] };
}

export function toggleAttacker(draft: Draft, id: number): Draft {
  const attackers = draft.attackers.includes(id) ? draft.attackers.filter(other => other !== id) : [...draft.attackers, id];
  return { ...draft, attackers };
}

/** Place or remove one blocker, leaving every other assignment intact. */
export function setBlock(draft: Draft, blocker: number, attacker: number | null): Draft {
  if (draft.blocks.find(([id]) => id === blocker)?.[1] === attacker) return draft;
  const blocks = draft.blocks.filter(([id]) => id !== blocker);
  if (attacker !== null) blocks.push([blocker, attacker]);
  return { ...draft, blocks };
}

export function toggleBlock(draft: Draft, blocker: number, attacker: number): Draft {
  const assigned = draft.blocks.some(([b, a]) => b === blocker && a === attacker);
  return setBlock(draft, blocker, assigned ? null : attacker);
}

/** Draft the first `count` of a pile's identical creatures into the attack. */
export function setPileCount(draft: Draft, pile: number[], count: number): Draft {
  const rest = draft.attackers.filter(id => !pile.includes(id));
  return { ...draft, attackers: [...rest, ...pile.slice(0, Math.max(0, Math.min(pile.length, count)))] };
}

/** The plan to send: the draft named by the preview's own exact identities. */
export function planOf(draft: Draft, preview: CombatPreview): CombatPlan {
  const ids = new Map<number, RenderId>();
  for (const creature of preview.declarable) {
    ids.set(creature.creature.entity, creature.creature);
    for (const attacker of creature.can_block) ids.set(attacker.entity, attacker);
  }
  const id = (entity: number): RenderId => ids.get(entity) ?? { entity, incarnation: 0 };
  return {
    attackers: draft.attackers.map(id),
    blocks: draft.blocks.map(([blocker, attacker]) => ({ blocker: id(blocker), attacker: id(attacker) })),
  };
}

export function planKey(plan: CombatPlan): string {
  const key = (id: RenderId): string => `${id.entity}@${id.incarnation}`;
  return JSON.stringify([plan.attackers.map(key).sort(), plan.blocks.map(block => `${key(block.blocker)}>${key(block.attacker)}`).sort()]);
}

/** The board as drafted: drafted attackers lean, drafted blockers draw their line. Presentation only. */
export function drawDraft(frame: ExperienceFrame, draft: Draft): ExperienceFrame {
  if (!draft.attackers.length && !draft.blocks.length) return frame;
  const attacking = new Set(draft.attackers);
  const blocking = new Map(draft.blocks);
  const draw = (permanent: ExperienceFrame['projection']['agent']['battlefield'][number]) =>
    attacking.has(permanent.id) ? { ...permanent, attacking: true }
      : blocking.has(permanent.id) ? { ...permanent, blocking: blocking.get(permanent.id) } : permanent;
  const { agent, opponent } = frame.projection;
  return {
    ...frame,
    projection: {
      ...frame.projection,
      agent: { ...agent, battlefield: agent.battlefield.map(draw) },
      opponent: { ...opponent, battlefield: opponent.battlefield.map(draw) },
    },
  };
}

/**
 * What the engine already declared in this frame without asking: a creature
 * that must attack, or a blocker with one legal block (`skip_trivial`).
 */
export function committedOf(frame: ExperienceFrame): Draft {
  const permanents = [...frame.projection.agent.battlefield, ...frame.projection.opponent.battlefield];
  return {
    attackers: permanents.filter(permanent => permanent.attacking).map(permanent => permanent.id),
    blocks: permanents.flatMap(permanent => (permanent.blocking === undefined ? [] : [[permanent.id, permanent.blocking] as [number, number]])),
  };
}

// ———————————————————————————————————————— Words

/** "Tiger-Seal", "Ally ×2": names in first-seen order, repeats counted. */
function counted(names: string[]): string[] {
  const counts = new Map<string, number>();
  for (const name of names) counts.set(name, (counts.get(name) ?? 0) + 1);
  return [...counts].map(([name, count]) => (count > 1 ? `${name} ×${count}` : name));
}

function list(parts: string[]): string {
  return parts.length <= 1 ? parts.join('') : `${parts.slice(0, -1).join(', ')} and ${parts.at(-1)}`;
}

export const BLOCK_INSTRUCTION = 'Drag a blocker onto an attacker, or click each';

export interface DraftWords {
  /** The primary button: "Attack with 3", "No attack". */
  text: string;
  /** The whole plan: "Attack with 3 · not with Tiger-Seal". */
  plan: string;
}

/** Words for the declaration as it will stand: `committed` (see `committedOf`) plus the draft. */
export function draftWords(
  kind: CombatPreview['kind'], drafted: Draft, declarable: DeclarableCreature[], names: Map<number, string>, committed: Draft,
): DraftWords {
  const name = (id: number): string => names.get(id) ?? 'a creature';
  const draft: Draft = {
    attackers: [...committed.attackers.filter(id => !drafted.attackers.includes(id)), ...drafted.attackers],
    blocks: [...committed.blocks.filter(([blocker]) => !drafted.blocks.some(([b]) => b === blocker)), ...drafted.blocks],
  };
  if (kind === 'attackers') {
    const count = draft.attackers.length;
    if (count === 0) return { text: 'No attack', plan: 'No attack · click creatures to add them' };
    const out = declarable.filter(creature => !draft.attackers.includes(creature.creature.entity)).map(creature => creature.label);
    const one = `Attack with ${name(draft.attackers[0])}`;
    const text = count === 1 && one.length <= PRIMARY_CHARS ? one : `Attack with ${count}`;
    if (!out.length) return { text, plan: count > 1 ? `Attack with all ${count}` : `Attack with ${name(draft.attackers[0])}` };
    const left = counted(out);
    return { text, plan: `${count === 1 ? `Attack with ${name(draft.attackers[0])}` : text} · not with ${left.length > 2 ? `${out.length} others` : list(left)}` };
  }
  const count = draft.blocks.length;
  if (count === 0) return { text: 'No blocks', plan: `No blocks · ${BLOCK_INSTRUCTION}` };
  const byAttacker = new Map<number, string[]>();
  for (const [blocker, attacker] of draft.blocks) byAttacker.set(attacker, [...(byAttacker.get(attacker) ?? []), name(blocker)]);
  const lines = [...byAttacker].map(([attacker, blockers]) => `${list(counted(blockers))} on ${name(attacker)}`);
  return { text: `Block with ${count}`, plan: lines.join(' · ') };
}

/** Identical declarable creatures drawn as one pile: "4 of 9 Allies". */
export interface PileCount { name: string; ids: number[]; drafted: number }

export function pileCounts(declarable: DeclarableCreature[], draft: Draft, pileOf: (id: number) => string | null): PileCount[] {
  const groups = new Map<string, PileCount>();
  for (const creature of declarable) {
    const key = pileOf(creature.creature.entity);
    if (key === null) continue;
    const group = groups.get(key) ?? { name: creature.label, ids: [], drafted: 0 };
    group.ids.push(creature.creature.entity);
    if (draft.attackers.includes(creature.creature.entity)) group.drafted += 1;
    groups.set(key, group);
  }
  return [...groups.values()].filter(group => group.ids.length > 1);
}

// ———————————————————————————————————————— Sending

/**
 * The offer that answers `step` in `frame`, or null when the frame is not the
 * decision the preview predicted: another kind, another creature, or an offer
 * that does not say the planned answer. The offer is the frame's own; the
 * caller binds it to this frame's revision.
 */
export function stepOffer(frame: ExperienceFrame, kind: CombatPreview['kind'], step: DeclarationStep): InteractionOffer | null {
  if (frame.action_space !== (kind === 'attackers' ? 'DECLARE_ATTACKER' : 'DECLARE_BLOCKER')) return null;
  const offer = frame.offers.find(candidate => candidate.id === step.action_index);
  if (!offer?.subjects) return null;
  const about = (role: string): number[] => offer.subjects!.filter(subject => subject.role === role).map(subject => subject.object.display_id);
  const named = offer.subjects.map(subject => subject.object.display_id);
  if (!named.includes(step.creature.entity)) return null;
  if (kind === 'attackers') {
    const attacks = about('source').includes(step.creature.entity);
    return attacks === step.attack ? offer : null;
  }
  const targets = about('target');
  if (step.blocks === null) return targets.length === 0 ? offer : null;
  return targets.length === 1 && targets[0] === step.blocks.entity ? offer : null;
}

// ———————————————————————————————————————— Reload

const DRAFT_KEY = 'academy.table.combat-draft';

/** A draft survives a reload of the same decision in this tab, and no other. */
export function saveDraft(frame: ExperienceFrame, draft: Draft): void {
  try {
    sessionStorage.setItem(DRAFT_KEY, JSON.stringify({ match_id: frame.match_id, frame_hash: frame.frame_hash, draft }));
  } catch { /* A draft is presentation; losing it loses nothing committed. */ }
}

export function loadDraft(frame: ExperienceFrame): Draft | null {
  try {
    const saved = JSON.parse(sessionStorage.getItem(DRAFT_KEY) ?? 'null');
    return saved && saved.match_id === frame.match_id && saved.frame_hash === frame.frame_hash ? saved.draft as Draft : null;
  } catch {
    return null;
  }
}
