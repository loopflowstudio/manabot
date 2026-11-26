/**
 * Typed grouped-input descriptors, plans, previews and unified draft storage.
 *
 * Mirrors `managym.decision_group` (`managym/decision_group.py`) and the live `decision_group` payload
 * attached by `managym/academy/server.py`. The browser never infers rules or
 * compiles steps: it reads `LiveDecisionGroup` from the server, previews drafted plans
 * via `/decision-group-preview`, and submits one atomic `decision_group_command` on Confirm.
 */
import type { ObjectRenderId } from './types';

export type LibraryEnd = 'top' | 'bottom';

export interface PlannedAttack extends ObjectRenderId {
  defender?: ObjectRenderId | null;
}

export interface PlannedBlock {
  blocker: ObjectRenderId;
  attacker: ObjectRenderId;
}

export interface DecisionCard {
  item: ObjectRenderId;
  label: string;
}

export interface AttackerItem {
  item: ObjectRenderId;
  label: string;
  can_attack?: ObjectRenderId[];
  must_attack: boolean;
}

export interface BlockerItem {
  item: ObjectRenderId;
  label: string;
  can_block: ObjectRenderId[];
  must_block: boolean;
}

export interface DecisionGroupAddress {
  match_id: string;
  revision: number;
  prompt_id: number;
  frame_hash: string;
}

export interface DeclareAttackersDecisionGroup extends DecisionGroupAddress {
  kind: 'declare_attackers';
  declarable: AttackerItem[];
  committed: PlannedAttack[];
}

export interface DeclareBlockersDecisionGroup extends DecisionGroupAddress {
  kind: 'declare_blockers';
  declarable: BlockerItem[];
  committed: PlannedBlock[];
}

export interface ScryDecisionGroup extends DecisionGroupAddress {
  kind: 'scry';
  items: DecisionCard[];
  committed_top: DecisionCard[];
  committed_bottom: DecisionCard[];
}

export interface SurveilDecisionGroup extends DecisionGroupAddress {
  kind: 'surveil';
  items: DecisionCard[];
  committed_graveyard: DecisionCard[];
}

export interface OrderLibraryDecisionGroup extends DecisionGroupAddress {
  kind: 'order_library';
  end: LibraryEnd;
  items: DecisionCard[];
  committed: DecisionCard[];
}

export interface PartitionPilesDecisionGroup extends DecisionGroupAddress {
  kind: 'partition_piles';
  items: DecisionCard[];
  committed_face_up: DecisionCard[];
  committed_face_down: DecisionCard[];
}

export type CombatDecisionGroup = DeclareAttackersDecisionGroup | DeclareBlockersDecisionGroup;
export type ArrangeDecisionGroup = ScryDecisionGroup | SurveilDecisionGroup | OrderLibraryDecisionGroup;
export type LiveDecisionGroup = CombatDecisionGroup | ArrangeDecisionGroup | PartitionPilesDecisionGroup;
export type DecisionGroupKind = LiveDecisionGroup['kind'];

export interface DeclareAttackersPlan {
  kind: 'declare_attackers';
  attackers: PlannedAttack[];
}

export interface DeclareBlockersPlan {
  kind: 'declare_blockers';
  blocks: PlannedBlock[];
}

export interface ScryPlan {
  kind: 'scry';
  top: ObjectRenderId[];
  bottom: ObjectRenderId[];
}

export interface SurveilPlan {
  kind: 'surveil';
  graveyard: ObjectRenderId[];
}

export interface OrderLibraryPlan {
  kind: 'order_library';
  order: ObjectRenderId[];
}

export interface PartitionPilesPlan {
  kind: 'partition_piles';
  face_up: ObjectRenderId[];
}

export type DecisionGroupPlan =
  | DeclareAttackersPlan
  | DeclareBlockersPlan
  | ScryPlan
  | SurveilPlan
  | OrderLibraryPlan
  | PartitionPilesPlan;

export interface DecisionGroupRefusal {
  item: ObjectRenderId | null;
  label: string;
  reason: string;
}

export type DecisionGroupOutcome =
  | { kind: 'declare_attackers'; attackers: PlannedAttack[] }
  | { kind: 'declare_blockers'; blocks: PlannedBlock[] }
  | { kind: 'scry'; top: ObjectRenderId[]; bottom: ObjectRenderId[] }
  | { kind: 'surveil'; graveyard: ObjectRenderId[]; kept: ObjectRenderId[] }
  | { kind: 'order_library'; end: LibraryEnd; order: ObjectRenderId[] }
  | { kind: 'partition_piles'; face_up: ObjectRenderId[]; face_down: ObjectRenderId[] };

export interface DecisionGroupPreviewRequest extends DecisionGroupAddress {
  plan: DecisionGroupPlan;
}

export interface DecisionGroupPreviewResponse extends DecisionGroupAddress {
  viewer: number;
  complete: boolean;
  refused: DecisionGroupRefusal | null;
  outcome: DecisionGroupOutcome | null;
}

export type DecisionGroupPreviewFetcher = (
  request: DecisionGroupPreviewRequest,
  signal: AbortSignal,
) => Promise<DecisionGroupPreviewResponse>;

export interface DecisionGroupCommand {
  command_id: string;
  match_id: string;
  expected_revision: number;
  prompt_id: number;
  frame_hash: string;
  plan: DecisionGroupPlan;
}

export function isCombatDecisionGroup(group: LiveDecisionGroup | null): group is CombatDecisionGroup {
  return group !== null && (group.kind === 'declare_attackers' || group.kind === 'declare_blockers');
}

export function isArrangeDecisionGroup(group: LiveDecisionGroup | null): group is ArrangeDecisionGroup {
  return group !== null && (group.kind === 'scry' || group.kind === 'surveil' || group.kind === 'order_library');
}

export function isPartitionDecisionGroup(group: LiveDecisionGroup | null): group is PartitionPilesDecisionGroup {
  return group !== null && group.kind === 'partition_piles';
}

function renderIdKey(id: ObjectRenderId): string {
  return `${id.entity}@${id.incarnation}`;
}

/** Canonical comparison key for preview deduplication. Preserves order where order matters. */
export function decisionGroupPlanKey(plan: DecisionGroupPlan): string {
  switch (plan.kind) {
    case 'declare_attackers':
      return JSON.stringify([
        plan.kind,
        plan.attackers
          .map(attack => (attack.defender ? `${renderIdKey(attack)}>${renderIdKey(attack.defender)}` : renderIdKey(attack)))
          .sort(),
      ]);
    case 'declare_blockers':
      return JSON.stringify([
        plan.kind,
        plan.blocks.map(block => `${renderIdKey(block.blocker)}>${renderIdKey(block.attacker)}`).sort(),
      ]);
    case 'scry':
      return JSON.stringify([plan.kind, plan.top.map(renderIdKey), plan.bottom.map(renderIdKey)]);
    case 'surveil':
      return JSON.stringify([plan.kind, plan.graveyard.map(renderIdKey).sort()]);
    case 'order_library':
      return JSON.stringify([plan.kind, plan.order.map(renderIdKey)]);
    case 'partition_piles':
      return JSON.stringify([plan.kind, plan.face_up.map(renderIdKey).sort()]);
  }
}

function validRenderId(value: unknown): value is ObjectRenderId {
  if (!value || typeof value !== 'object') return false;
  const id = value as Partial<ObjectRenderId>;
  return Number.isSafeInteger(id.entity) && Number.isSafeInteger(id.incarnation);
}

function validRenderIdList(value: unknown): value is ObjectRenderId[] {
  return Array.isArray(value) && value.every(validRenderId);
}

export function validDecisionGroupPlan(value: unknown): value is DecisionGroupPlan {
  if (!value || typeof value !== 'object') return false;
  const plan = value as Record<string, unknown>;
  switch (plan.kind) {
    case 'declare_attackers':
      return Array.isArray(plan.attackers) && plan.attackers.every(
        entry => validRenderId(entry)
          && ((entry as Partial<PlannedAttack>).defender === undefined
            || (entry as Partial<PlannedAttack>).defender === null
            || validRenderId((entry as Partial<PlannedAttack>).defender)),
      );
    case 'declare_blockers':
      return Array.isArray(plan.blocks) && plan.blocks.every(
        entry => !!entry && typeof entry === 'object'
          && validRenderId((entry as Partial<PlannedBlock>).blocker)
          && validRenderId((entry as Partial<PlannedBlock>).attacker),
      );
    case 'scry':
      return validRenderIdList(plan.top) && validRenderIdList(plan.bottom);
    case 'surveil':
      return validRenderIdList(plan.graveyard);
    case 'order_library':
      return validRenderIdList(plan.order);
    case 'partition_piles':
      return validRenderIdList(plan.face_up);
    default:
      return false;
  }
}

export interface DecisionGroupDisclosureCopy {
  summary: string;
  caption: string;
}

/** Rail disclosure wording for the collapsed raw single-step offers of a live group. */
export function decisionGroupDisclosureCopy(kind: DecisionGroupKind): DecisionGroupDisclosureCopy {
  if (kind === 'declare_attackers' || kind === 'declare_blockers') {
    return {
      summary: "Each creature's choice",
      caption: 'Confirming the draft commits all declarations together, or choose one creature below.',
    };
  }
  if (kind === 'partition_piles') {
    return {
      summary: "Each card's choice",
      caption: 'Confirming the split commits all cards together, or choose one card below.',
    };
  }
  return {
    summary: "Each card's choice",
    caption: 'Confirming the arrangement commits all cards together, or choose one step below.',
  };
}

// ———————————————————————————————————————— Unified draft persistence

const DECISION_GROUP_DRAFT_KEY = 'academy.table.decision-group-draft';

export interface DecisionGroupDraftScope {
  match_id: string;
  frame_hash: string;
  kind: DecisionGroupKind;
}

/** Keep only distinct numeric entity ids present in `allowed`, preserving order. */
export function sanitizeEntityIds(
  raw: unknown,
  allowed: ReadonlySet<number>,
  seen: Set<number> = new Set(),
): number[] {
  if (!Array.isArray(raw)) return [];
  const clean: number[] = [];
  for (const id of raw) {
    if (typeof id === 'number' && allowed.has(id) && !seen.has(id)) {
      seen.add(id);
      clean.push(id);
    }
  }
  return clean;
}

/** Persist a live grouped draft scoped to `(match_id, frame_hash, kind)`. */
export function saveDecisionGroupDraft<T>(scope: DecisionGroupDraftScope, draft: T): void {
  try {
    sessionStorage.setItem(
      DECISION_GROUP_DRAFT_KEY,
      JSON.stringify({ match_id: scope.match_id, frame_hash: scope.frame_hash, kind: scope.kind, draft }),
    );
  } catch { /* A draft is presentation; losing it loses nothing committed. */ }
}

/** Restore a saved draft only when it matches the exact decision scope. */
export function loadDecisionGroupDraft<T>(scope: DecisionGroupDraftScope): T | null {
  try {
    const saved = JSON.parse(sessionStorage.getItem(DECISION_GROUP_DRAFT_KEY) ?? 'null') as
      | { match_id?: unknown; frame_hash?: unknown; kind?: unknown; draft?: unknown }
      | null;
    if (
      saved
      && saved.match_id === scope.match_id
      && saved.frame_hash === scope.frame_hash
      && saved.kind === scope.kind
    ) {
      return saved.draft as T;
    }
    return null;
  } catch {
    return null;
  }
}
