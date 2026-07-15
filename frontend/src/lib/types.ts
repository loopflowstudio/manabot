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

// Protocol-v1's first adapter slice wraps one current engine action in each
// offer. action_type/focus are temporary bridges for the established table.
export interface InteractionOffer {
  id: number;
  actor: number;
  verb: string;
  source: null;
  label: string;
  help: string | null;
  choices: [];
  confirm_label: string;
  action_type: string;
  focus: number[];
}

export interface ExperienceFrame {
  protocol: 1;
  match_id: string;
  revision: number;
  frame_hash: string;
  content_hash: string;
  asset_manifest_hash: string;
  status: 'ready' | 'game_over';
  prompt: PromptView | null;
  projection: Observation;
  offers: InteractionOffer[];
  winner: number | null;
  action_space: string;
  stops: StopsConfig;
  deck_names?: DeckNames;
  log?: string[];
  auto_passed?: number;
}

export interface Command {
  command_id: string;
  match_id: string;
  expected_revision: number;
  prompt_id: number;
  offer_id: number;
  answers: [];
}

export interface CommandReceipt {
  command_id: string;
  actor: number;
  accepted_at: number;
  resulting_revision: number;
  resulting_frame_hash: string;
}

export interface ObjectRenderId {
  entity: number;
  incarnation: number;
}

export type SubjectRef =
  | { kind: 'object'; id: ObjectRenderId }
  | { kind: 'stack'; id: number }
  | { kind: 'player'; id: number };

export type PresentationImportance = 'ambient' | 'normal' | 'emphasized' | 'critical';

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
  protocol: 1;
  engine_version: string;
  content_hash: string;
  asset_manifest_hash: string;
  reason: string;
  frame: ExperienceFrame;
  presentation_tail: PresentationEvent[];
  accepted_commands: CommandReceipt[];
  replay_cursor: number;
  checkpoint: string | null;
}

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
  /** Optional until the trace writer persists protocol presentation events. */
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
