export interface CardTypes {
  is_creature: boolean;
  is_land: boolean;
  is_spell: boolean;
  is_artifact: boolean;
  is_enchantment: boolean;
  is_planeswalker: boolean;
  is_battle: boolean;
}

export interface CardState {
  id: number;
  registry_key: number;
  name: string;
  zone: string;
  owner_id: number;
  power: number;
  toughness: number;
  mana_value: number;
  types: CardTypes;
}

export interface PermanentState {
  id: number;
  name: string | null;
  controller_id: number;
  tapped: boolean;
  damage: number;
  summoning_sick: boolean;
  power: number | null;
  toughness: number | null;
  base_power: number | null;
  base_toughness: number | null;
  plus1_counters: number;
}

export interface PlayerState {
  player_index: number;
  id: number;
  is_active: boolean;
  is_agent: boolean;
  life: number;
  zone_counts: Record<string, number>;
  library_count: number;
  hand_hidden_count?: number;
  hand: CardState[];
  graveyard: CardState[];
  exile: CardState[];
  stack: CardState[];
  battlefield: PermanentState[];
}

export interface Observation {
  game_over: boolean;
  won: boolean;
  turn: {
    turn_number: number;
    phase: string;
    step: string;
    active_player_id: number;
    agent_player_id: number;
  };
  agent: PlayerState;
  opponent: PlayerState;
}

export interface ActionOption {
  index: number;
  type: string;
  focus: number[];
  description: string;
}

export interface PromptView {
  id: number;
  actor: number;
  kind: string;
  title: string;
  instruction: string;
}

export interface ObjectRenderId {
  entity: number;
  incarnation: number;
}

export type SubjectRef =
  | { kind: 'object'; id: ObjectRenderId }
  | { kind: 'stack'; id: number }
  | { kind: 'player'; id: number };

export const OFFER_VERBS = [
  'cast',
  'play_land',
  'activate',
  'pass_priority',
  'declare_attackers',
  'declare_blockers',
  'choose',
  'pay',
  'special',
] as const;

export type OfferVerb = (typeof OFFER_VERBS)[number];

export type CandidateValue =
  | { kind: 'subject'; subject: SubjectRef }
  | { kind: 'mode'; key: string }
  | { kind: 'payment_plan'; id: string }
  | { kind: 'boolean'; value: boolean };

export interface Candidate {
  id: number;
  value: CandidateValue;
  label: string;
  help: string | null;
  preview: string | null;
}

export interface CandidateSource {
  id: number;
  depends_on: number[];
  initial: Candidate[] | null;
}

export type ChoiceStep =
  | {
      kind: 'select';
      role: number;
      label: string;
      candidates: CandidateSource;
      min: number;
      max: number;
      ordered: boolean;
      distinct: boolean;
    }
  | { kind: 'number'; role: number; label: string; min: number; max: number }
  | {
      kind: 'assign';
      role: number;
      label: string;
      sources: CandidateSource;
      destinations: CandidateSource;
      min_per_source: number;
      max_per_source: number;
    }
  | { kind: 'order'; role: number; label: string; candidates: CandidateSource }
  | {
      kind: 'payment';
      role: number;
      label: string;
      plans: CandidateSource;
      allow_auto: boolean;
    };

export type ChoiceAnswer =
  | { kind: 'candidates'; role: number; candidates: number[] }
  | { kind: 'number'; role: number; value: number }
  | { kind: 'assignments'; role: number; pairs: [number, number][] }
  | { kind: 'order'; role: number; candidates: number[] }
  | { kind: 'payment'; role: number; plan: string };

// Protocol-v1's first adapter slice wraps one current engine action in each
// offer. action_type/focus are temporary bridges for the established table.
export interface InteractionOffer {
  id: number;
  actor: number;
  verb: OfferVerb;
  source: SubjectRef | null;
  label: string;
  help: string | null;
  choices: ChoiceStep[];
  confirm_label: string;
  action_type: string;
  focus: number[];
}

export interface ExperienceFrame {
  protocol: ProtocolVersion;
  match_id: string;
  revision: number;
  frame_hash: string;
  content_hash: string;
  asset_manifest_hash: string;
  status: AuthorityStatus;
  prompt: PromptView | null;
  projection: Observation;
  offers: InteractionOffer[];
  winner: number | null;
  action_space: string;
  stops: StopsConfig;
  deck_names?: DeckNames;
  asset_pack?: AssetPackReference | null;
  log?: string[];
  auto_passed?: number;
}

export interface Command {
  command_id: string;
  match_id: string;
  expected_revision: number;
  prompt_id: number;
  offer_id: number;
  answers: ChoiceAnswer[];
}

export interface CommandReceipt {
  command_id: string;
  actor: number;
  accepted_at: number;
  resulting_revision: number;
  resulting_frame_hash: string;
}

export const PROTOCOL_VERSION = 1 as const;
export type ProtocolVersion = typeof PROTOCOL_VERSION;

export const AUTHORITY_STATUSES = [
  'ready',
  'thinking',
  'resolving',
  'reconnecting',
  'game_over',
] as const;
export type AuthorityStatus = (typeof AUTHORITY_STATUSES)[number];

export const PRESENTATION_IMPORTANCES = [
  'ambient',
  'normal',
  'emphasized',
  'critical',
] as const;
export type PresentationImportance = (typeof PRESENTATION_IMPORTANCES)[number];

export const PRESENTATION_KIND_NAMES = [
  'cast',
  'targeted',
  'resolved',
  'damage',
  'destroyed',
  'died',
] as const;

/**
 * Protocol-v1 semantic theater. Targeted/resolved are the first explicit
 * additions beyond the original design sketch, because a target choice and a
 * spell leaving the stack must not be reconstructed from snapshot diffs.
 */
export type PresentationKind =
  | {
      kind: 'cast';
      object: ObjectRenderId;
      controller: number;
      stack: number;
    }
  | {
      kind: 'targeted';
      source: SubjectRef;
      target: SubjectRef;
    }
  | {
      kind: 'resolved';
      stack: number;
    }
  | {
      kind: 'damage';
      source: SubjectRef | null;
      target: SubjectRef;
      amount: number;
    }
  | {
      kind: 'destroyed';
      objects: ObjectRenderId[];
    }
  | {
      /** Actual creature death, including lethal damage and zero toughness. */
      kind: 'died';
      objects: ObjectRenderId[];
    };

export interface PresentationEvent {
  seq: number;
  from_revision: number;
  to_revision: number;
  caused_by: string | null;
  group: number;
  importance: PresentationImportance;
  suggested_ms: number;
  sound: string | null;
  kind: PresentationKind;
}

export interface FrameUpdate {
  base_revision: number;
  frame: ExperienceFrame;
  presentation: PresentationEvent[];
  receipt: CommandReceipt | null;
}

export interface RecoveryEnvelope {
  protocol: ProtocolVersion;
  engine_version: string;
  content_hash: string;
  asset_manifest_hash: string;
  reason: RecoveryReason;
  frame: ExperienceFrame;
  presentation_tail: PresentationEvent[];
  accepted_commands: CommandReceipt[];
  replay_cursor: number;
  checkpoint: string | null;
}

/** Root of the checked-in Rust-generated cross-language fixture. */
export interface ProtocolV1ConformanceBundle {
  recovery: RecoveryEnvelope;
  command: Command;
}

export const RECOVERY_REASONS = [
  'initial_connect',
  'explicit_resync',
  'revision_gap',
  'reconnect',
  'duplicate_command',
  'stale_command',
  'authority_restart',
] as const;
export type RecoveryReason = (typeof RECOVERY_REASONS)[number];

export interface CommandRejection {
  command_id: string;
  code: string;
  message: string;
  current_revision: number;
  current_prompt: number | null;
}

export interface GameLogEntry {
  id: string;
  actor: 'hero' | 'villain' | 'system';
  text: string;
}

export interface TraceSummary {
  id: string;
  timestamp: string | null;
  winner: number | null;
  end_reason: string | null;
  num_events: number;
}

export type VillainType = 'passive' | 'random' | 'search' | 'checkpoint';

export interface OpponentConfig {
  villain_type: VillainType;
  villain_sims?: number;
  villain_checkpoint?: string;
  villain_deterministic?: boolean;
}

export interface TraceConfig {
  hero_deck: Record<string, number>;
  villain_deck: Record<string, number>;
  villain_type: string;
  seed?: number | null;
  hero_deck_name?: string;
  villain_deck_name?: string;
  villain_sims?: number | null;
  villain_checkpoint?: string | null;
  villain_deterministic?: boolean;
  asset_pack?: AssetPackReference | null;
}

export interface AssetPackReference {
  id: string;
  version: string;
  manifest_sha256: string;
}

// Display names for the current matchup, echoed by the server on every
// observation/game_over payload.
export interface DeckNames {
  hero: string;
  villain: string;
}

export interface TraceEvent {
  actor: 'hero' | 'villain';
  observation: Observation;
  actions: ActionOption[];
  action: number;
  action_description: string;
  reward: number;
  /** Optional for backwards compatibility with traces written before v1. */
  presentation?: PresentationEvent[];
}

export interface Trace {
  id?: string;
  config: TraceConfig;
  events: TraceEvent[];
  final_observation: Observation;
  winner: number | null;
  end_reason: string;
  timestamp: string;
}

export interface ReplayFrame {
  observation: Observation;
  actionDescription: string | null;
  actor: 'hero' | 'villain' | null;
  presentation: PresentationEvent[];
}

export type StopSide = 'my' | 'opponent';

// Effective priority-stop configuration; the server echoes this on every
// observation/game_over payload (see gui/server.py: _stops_payload).
export interface StopsConfig {
  my: string[];
  opponent: string[];
  stop_on_stack: boolean;
  auto_pass: boolean;
}

export type ServerMessage =
  | {
      type: 'observation';
      data: Observation;
      actions: ActionOption[];
      // ActionSpaceEnum name (PRIORITY, SCRY, PAY_OR_NOT, MODAL, ...) — what
      // kind of decision the hero is being asked to make.
      action_space?: string;
      log?: string[];
      stops?: StopsConfig;
      deck_names?: DeckNames;
      asset_pack?: AssetPackReference | null;
      auto_passed?: number;
      session_id?: string;
      resume_token?: string;
      frame?: ExperienceFrame;
      recovery?: RecoveryEnvelope;
    }
  | {
      type: 'game_over';
      data: Observation;
      winner: number | null;
      log?: string[];
      stops?: StopsConfig;
      deck_names?: DeckNames;
      asset_pack?: AssetPackReference | null;
      auto_passed?: number;
      frame?: ExperienceFrame;
      recovery?: RecoveryEnvelope;
    }
  | ({ type: 'command_outcome' } & (
      | { status: 'accepted'; update: FrameUpdate }
      | {
          status: 'duplicate';
          receipt: CommandReceipt;
          recovery: RecoveryEnvelope;
        }
      | {
          status: 'rejected';
          rejection: CommandRejection;
          recovery?: RecoveryEnvelope;
        }
    ))
  | { type: 'error'; message: string };

export type ClientMessage =
  | { type: 'new_game'; config?: Record<string, unknown> }
  | { type: 'command'; command: Command }
  | {
      type: 'set_stops';
      stops: { my: string[]; opponent: string[] };
      stop_on_stack: boolean;
      auto_pass: boolean;
    }
  | { type: 'pass_turn' }
  | { type: 'resume'; session_id: string; resume_token: string };

export type ConnectionState =
  | 'disconnected'
  | 'connecting'
  | 'connected'
  | 'reconnecting';
