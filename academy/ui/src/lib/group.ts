/**
 * Typed grouped-input descriptors, plans, previews and unified draft storage.
 *
 * Mirrors `managym.group` (`managym/group.py`) and the live `group` payload
 * attached by `managym/academy/server.py`. The browser never infers rules or
 * compiles steps: it reads `LiveGroup` from the server, previews drafted plans
 * via `/group-preview`, and submits one atomic `group_command` on Confirm.
 */
import type { ObjectRenderId } from './types';

export type LibraryEnd = 'top' | 'bottom';

export interface PlannedBlock {
  blocker: ObjectRenderId;
  attacker: ObjectRenderId;
}

export interface GroupCard {
  item: ObjectRenderId;
  label: string;
}

export interface AttackerItem {
  item: ObjectRenderId;
  label: string;
  must_attack: boolean;
}

export interface BlockerItem {
  item: ObjectRenderId;
  label: string;
  can_block: ObjectRenderId[];
  must_block: boolean;
}

export interface GroupAddress {
  match_id: string;
  revision: number;
  prompt_id: number;
  frame_hash: string;
}

export interface DeclareAttackersGroup extends GroupAddress {
  kind: 'declare_attackers';
  declarable: AttackerItem[];
  committed: ObjectRenderId[];
}

export interface DeclareBlockersGroup extends GroupAddress {
  kind: 'declare_blockers';
  declarable: BlockerItem[];
  committed: PlannedBlock[];
}

export interface ScryGroup extends GroupAddress {
  kind: 'scry';
  items: GroupCard[];
  committed_top: ObjectRenderId[];
  committed_bottom: ObjectRenderId[];
}

export interface SurveilGroup extends GroupAddress {
  kind: 'surveil';
  items: GroupCard[];
  committed_graveyard: ObjectRenderId[];
}

export interface OrderLibraryGroup extends GroupAddress {
  kind: 'order_library';
  end: LibraryEnd;
  items: GroupCard[];
  committed: ObjectRenderId[];
}

export type CombatGroup = DeclareAttackersGroup | DeclareBlockersGroup;
export type ArrangeGroup = ScryGroup | SurveilGroup | OrderLibraryGroup;
export type LiveGroup = CombatGroup | ArrangeGroup;
export type GroupKind = LiveGroup['kind'];

export interface DeclareAttackersPlan {
  kind: 'declare_attackers';
  attackers: ObjectRenderId[];
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

export type GroupPlan =
  | DeclareAttackersPlan
  | DeclareBlockersPlan
  | ScryPlan
  | SurveilPlan
  | OrderLibraryPlan;

export interface GroupRefusal {
  item: ObjectRenderId | null;
  label: string;
  reason: string;
}

export type GroupOutcome =
  | { kind: 'declare_attackers'; attackers: ObjectRenderId[] }
  | { kind: 'declare_blockers'; blocks: PlannedBlock[] }
  | { kind: 'scry'; top: ObjectRenderId[]; bottom: ObjectRenderId[] }
  | { kind: 'surveil'; graveyard: ObjectRenderId[]; kept: ObjectRenderId[] }
  | { kind: 'order_library'; end: LibraryEnd; order: ObjectRenderId[] };

export interface GroupPreviewRequest extends GroupAddress {
  plan: GroupPlan;
}

export interface GroupPreviewResponse extends GroupAddress {
  viewer: number;
  complete: boolean;
  refused: GroupRefusal | null;
  outcome: GroupOutcome | null;
}

export type GroupPreviewFetcher = (
  request: GroupPreviewRequest,
  signal: AbortSignal,
) => Promise<GroupPreviewResponse>;

export interface GroupCommand {
  command_id: string;
  match_id: string;
  expected_revision: number;
  prompt_id: number;
  frame_hash: string;
  plan: GroupPlan;
}

export function isCombatGroup(group: LiveGroup | null): group is CombatGroup {
  return group !== null && (group.kind === 'declare_attackers' || group.kind === 'declare_blockers');
}

export function isArrangeGroup(group: LiveGroup | null): group is ArrangeGroup {
  return group !== null && (group.kind === 'scry' || group.kind === 'surveil' || group.kind === 'order_library');
}

function renderIdKey(id: ObjectRenderId): string {
  return `${id.entity}@${id.incarnation}`;
}

/** Canonical comparison key for preview deduplication. Preserves order where order matters. */
export function groupPlanKey(plan: GroupPlan): string {
  switch (plan.kind) {
    case 'declare_attackers':
      return JSON.stringify([plan.kind, plan.attackers.map(renderIdKey).sort()]);
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

export function validGroupPlan(value: unknown): value is GroupPlan {
  if (!value || typeof value !== 'object') return false;
  const plan = value as Record<string, unknown>;
  switch (plan.kind) {
    case 'declare_attackers':
      return validRenderIdList(plan.attackers);
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
    default:
      return false;
  }
}

export interface GroupDisclosureCopy {
  summary: string;
  caption: string;
}

/** Rail disclosure wording for the collapsed raw single-step offers of a live group. */
export function groupDisclosureCopy(kind: GroupKind): GroupDisclosureCopy {
  if (kind === 'declare_attackers' || kind === 'declare_blockers') {
    return {
      summary: "Each creature's choice",
      caption: 'Confirming the draft commits all declarations together, or choose one creature below.',
    };
  }
  return {
    summary: "Each card's choice",
    caption: 'Confirming the arrangement commits all cards together, or choose one step below.',
  };
}

// ———————————————————————————————————————— Unified draft persistence

const GROUP_DRAFT_KEY = 'academy.table.group-draft';

export interface GroupDraftScope {
  match_id: string;
  frame_hash: string;
  kind: GroupKind;
}

/** Persist a live grouped draft scoped to `(match_id, frame_hash, kind)`. */
export function saveGroupDraft<T>(scope: GroupDraftScope, draft: T): void {
  try {
    sessionStorage.setItem(
      GROUP_DRAFT_KEY,
      JSON.stringify({ match_id: scope.match_id, frame_hash: scope.frame_hash, kind: scope.kind, draft }),
    );
  } catch { /* A draft is presentation; losing it loses nothing committed. */ }
}

/** Restore a saved draft only when it matches the exact decision scope. */
export function loadGroupDraft<T>(scope: GroupDraftScope): T | null {
  try {
    const saved = JSON.parse(sessionStorage.getItem(GROUP_DRAFT_KEY) ?? 'null') as
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
