export interface CardTypes {
  is_creature: boolean;
  is_land: boolean;
  is_spell: boolean;
  is_artifact: boolean;
  is_enchantment: boolean;
  is_planeswalker: boolean;
  is_battle: boolean;
}

export interface CardPreview {
  name: string | null;
  text_box?: string;
  power: number | null;
  toughness: number | null;
}

export interface CardState {
  printed_loyalty?: number | null;
  text_box?: string;
  id: number;
  registry_key: number;
  name: string;
  zone: string;
  owner_id: number;
  power: number;
  toughness: number;
  mana_value: number;
  types: CardTypes;
  /** Face down in exile; only views of players who may look at it list it. */
  face_down?: boolean;
  /** The permission to play this card from its current zone, when it has one. */
  cast_permission?: CastPermission;
  cast_permissions?: CastPermission[];
  /** The value of X announced for this spell on the stack. */
  x?: number;
  /** Counters on the card off the battlefield ("with a stash counter on it"), by kind. */
  counters?: Record<string, number>;
}

/** A public permission to play a card in its current zone; only set terms are present. */
export interface CastPermission {
  player_id: number;
  /** Mana symbols paid to cast it; absent for a land play. */
  cost?: string;
  this_turn?: boolean;
  /** Ends with the permitted player's next turn. */
  through_next_turn?: boolean;
  own_turn_only?: boolean;
  /** Usable only on a turn numbered after this one. */
  after_turn?: number;
  /** The public permanent sustaining this grant. */
  source_permanent_id?: number;
  flash?: boolean;
  any_mana?: boolean;
  waterbend?: boolean;
  foretold?: boolean;
  flashback?: boolean;
  follows_source_controller?: boolean;
  on_adventure?: boolean;
  /** This exact exiled copy stays prepared while linked to its source. */
  prepared?: boolean;
  /** Face down in exile without Foretell. */
  face_down?: boolean;
  /** A land face may be played with it; otherwise it casts spells only. */
  play_lands?: boolean;
  /** What must hold to use it, such as "you control a Wizard". */
  condition?: string;
}

/** Mana spendable only on the named spell or ability-source subtypes. */
export interface RestrictedMana {
  spells?: string[];
  abilities?: string[];
  /** Spendable on anything but casting nonartifact spells (Powerstone). */
  except_nonartifact_spells?: boolean;
  mana: number[];
}

/** One continuous-effect record on a permanent or player (CR 611, 613). */
export interface ContinuousEffect {
  /** The engine's `Modification` variant (`ModifyPt`, `AddKeywords`, ...). */
  modification: string;
  power?: number;
  toughness?: number;
  amount?: number;
  keywords?: string[];
  subtypes?: string[];
  color?: string;
  /** `EndOfTurn`, `Indefinite`, `WhileYouControlSource` or `UntilYourNextTurn`. */
  duration: string;
  /** The source's printed name. */
  source?: string;
  /** Timestamp order (CR 613.7); larger is later. */
  timestamp: number;
}

export interface PermanentState {
  printed_loyalty?: number | null;
  loyalty_activated_this_turn?: boolean;
  attacked_permanent?: number;
  types?: CardTypes;
  is_token?: boolean;
  temp_subtypes?: string[];
  exile_if_dies_this_turn?: boolean;
  /** Exhaust abilities this object has used. */
  transforms?: number;
  exhausted_abilities?: number;
  /** "Once each turn" abilities this object has used this turn. */
  once_each_turn_used?: number;
  /** Firebending granted until end of turn. */
  granted_firebending?: number;
  text_box?: string;
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
  /** Counters other than +1/+1, by kind (stun, fire, double_strike). */
  counters?: Record<string, number>;
  /** The permanent this Aura or Equipment is attached to. */
  attached_to?: number;
  /** Creature types set by an effect; `name` is then the set name. */
  creature_types?: string[];
  /** A land only through an effect. */
  is_granted_land?: boolean;
  /** Attacking in the current combat. */
  attacking?: boolean;
  first_strike_participant?: boolean;
  attacked_this_turn?: boolean;
  /** The attacking permanent this one blocks. */
  blocking?: number;
  /** It has lost all abilities. */
  abilities_lost?: boolean;
  /** Face down (CR 708.2). */
  face_down?: boolean;
  /** Keywords it has after continuous effects; absent when it has none. */
  keywords?: string[];
  /** Activated abilities (indices in its ability list) whose limit is spent. */
  used_abilities?: number[];
  combat_damage_to_players?: number[];
  damaged_permanents?: number[];
  ability_resolutions?: [number, number][];
  /** It was dealt damage this turn. */
  dealt_damage_this_turn?: boolean;
  entered_this_turn?: boolean;
  /** Continuous effects on it, in timestamp order. */
  effects?: ContinuousEffect[];
  /** The copy effect that made it a copy (CR 707.2). */
  copy_of?: CopyEffect;
}

/** A copy effect: the printed card it copies and the exceptions in its copiable values (CR 707.9b). */
export interface CopyEffect {
  name: string;
  exceptions?: ContinuousEffect[];
}

export interface SideboardCard {
  text_box?: string;
  candidate_id: number;
  owner_id: number;
  registry_key: number;
  name: string;
}

export interface PlayerTurnHistory {
  attacked?: boolean;
  permanents_sacrificed?: number;
  land_entered?: boolean;
  cards_drawn?: number;
  /** Bending kinds done this turn: bits water 1, earth 2, fire 4, air 8. */
  bent?: number;
  /** Life lost this turn (CR 119.3). */
  life_lost?: number;
  life_gains?: number;
  life_gained?: number;
  creatures_died?: number;
  permanents_left?: number;
  nonland_permanents_left?: number;
  warped_this_turn?: boolean;
  mana_spent_casting?: number;
}

export interface PlayerState {
  pooled_mana_classes?: { mana: number[]; effect?: string | null; restriction?: string | null; combat: boolean }[];
  mana?: number[];
  lesson_mana?: number[];
  /** `lesson_mana` split by restriction class. */
  restricted_mana?: RestrictedMana[];
  /** Player id whose next turn ends this player's hand-only casting. */
  spent_graveyard_types?: [number, string][];
  hand_only_until?: number;
  no_maximum_hand_size?: boolean;
  counters?: Record<string, number>;
  combat_mana?: number;
  inspected?: CardState[];
  sideboard?: SideboardCard[];
  known_hand?: Record<string, number>;
  /** Zero-based position from the top to definition, as this viewer knows it. */
  known_library?: Record<string, number>;
  /** Decklist counts, present only when the viewer knows this player's
   *  decklist (its own, or an open opponent's). Absent means closed. */
  main_counts?: Record<string, number>;
  unseen_counts?: Record<string, number>;
  sideboard_counts?: Record<string, number>;
  remaining_sideboard_counts?: Record<string, number>;
  player_index: number;
  id: number;
  is_active: boolean;
  is_agent: boolean;
  life: number;
  zone_counts: Record<string, number>;
  library_count: number;
  hand_hidden_count?: number;
  /** Unspent red mana stays until end of turn. */
  keeps_red_mana?: boolean;
  /** What this player has done this turn. */
  turn_history?: PlayerTurnHistory;
  /** Spells this player cast this turn. */
  spells_cast?: number;
  /** Extra turns this player will take after this one. */
  extra_turns?: number;
  /** Additional combat phases still to come this turn. */
  additional_combats?: number;
  /** Continuous effects on this player, in timestamp order. */
  effects?: ContinuousEffect[];
  hand: CardState[];
  graveyard: CardState[];
  exile: CardState[];
  command?: CardState[];
  stack: CardState[];
  battlefield: PermanentState[];
}

export interface DecisionObjectView {
  entity_id: number;
  subject: SubjectRef;
  display_id: number;
  label: string;
}

export interface OfferSubjectView {
  role: string;
  object: DecisionObjectView;
}

export interface DecisionTargetView {
  clause: number;
  object: DecisionObjectView;
}

/** One object's payment toward the pending cost: "sacrificed Gray Ogre". */
export interface PaidView {
  phrase: string;
  /** That object as it is now; absent when nothing visible remains of it. */
  object?: DecisionObjectView;
}

export interface DecisionContextView {
  pending?: string[];
  recent_facts: string[];
  actor: number;
  stage: string;
  instruction: string;
  source: DecisionObjectView | null;
  targets: DecisionTargetView[];
  reserved_sacrifice: DecisionObjectView | null;
  mana_cost: string | null;
  chosen_mode: number | null;
  /** `source` is being announced, so it is on the stack by the rules (CR 601.2a, 602.2a). */
  announcing?: 'spell' | 'activated_ability';
  /** What objects have already paid toward the cost being paid, in paid order. */
  paid?: PaidView[];
}

export interface StackEntryView {
  mana_spent?: number;
  combat_damage_to_players?: number[];
  ability_resolutions?: number;
  id: number;
  kind: string;
  controller: number;
  label: string;
  source_permanent?: number;
  ability_index?: number;
  chosen_mode?: number;
  /** The announced value of X, when above zero. */
  x?: number;
  /** A spell cast from a prepared copy; its stack copies are not. */
  from_prepared?: boolean;
  /** A spell cast with flashback, exiled as it leaves the stack. */
  flashback?: boolean;
}

export interface PlayerControlView {
  player: number;
  controller: number;
  whole_turn: boolean;
  active: boolean;
  effective: boolean;
}

export interface CombatView {
  player_index: number;
  /** Absent means no additional attacker restriction; legal offers still govern. */
  attackers?: string;
}

export interface ScheduleView {
  extra_turns: number[];
  resume_with?: number;
  current_combat?: CombatView;
  additional_combats: CombatView[];
}

export interface Observation {
  /** Missing on historical captures means unknown, not an empty schedule. */
  schedule?: ScheduleView;
  player_controls?: PlayerControlView[];
  stack_objects?: StackEntryView[];
  decision_context?: DecisionContextView;
  definition_names?: Record<string, string>;
  game_over: boolean;
  won: boolean;
  turn: {
    turn_number: number;
    creature_died_this_turn?: boolean;
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
  /** Native yes/no declaration intent for combat prompts. */
  declared?: boolean | null;
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

// Each offer binds one current native decision. Raw action_type/focus remain
// for retained readers and outside-game Learn candidate navigation.
export interface InteractionOffer {
  subjects?: OfferSubjectView[];
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
  log?: LogLine[];
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
  'attack_group',
  'blocked',
  'turn_started',
  'land_played',
  'discarded',
  'revealed',
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
    }
  | {
      kind: 'attack_group';
      attackers: ObjectRenderId[];
      defender: SubjectRef;
    }
  | {
      kind: 'blocked';
      attacker: ObjectRenderId;
      blockers: ObjectRenderId[];
    }
  | {
      kind: 'turn_started';
      player: number;
      turn_number: number;
    }
  | {
      kind: 'land_played';
      object: ObjectRenderId;
      player: number;
    }
  | {
      kind: 'discarded';
      objects: ObjectRenderId[];
      player: number;
    }
  | {
      /** Cards revealed to every player, by name: a reveal has no public object. */
      kind: 'revealed';
      cards: string[];
      player: number;
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
  /** Compatibility narration stays outside the immutable canonical frame. */
  log?: LogLine[];
  auto_passed?: number;
  /** The person's board after each step of this batch, as patches back from the frame (`board/boards.ts`). */
  boards?: StepBoardPatch[];
}

/** One step board on the wire: the board after presentation events `first`..`last`. */
export interface StepBoardPatch {
  first: number;
  last: number;
  /** A `PatchNode` (`board/boards.ts`) onto the board after this one, or the frame's. */
  patch: unknown;
}

export interface RecoveryEnvelope {
  protocol: ProtocolVersion;
  engine_version: string;
  content_hash: string;
  asset_manifest_hash: string;
  reason: RecoveryReason;
  frame: ExperienceFrame;
  /** Sequence address of the first tail event, or the next sequence when empty. */
  presentation_cursor: number;
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

/** One line of the server's narration: either player's move or an automated action, in the turn it was taken. */
export interface LogLine {
  actor: 'hero' | 'villain' | 'auto';
  turn: number;
  text: string;
}

/** One move of the board's log; automation is its own actor, never the person's. */
export interface GameLogEntry {
  id: string;
  actor: 'hero' | 'villain' | 'auto' | 'system';
  /** The turn the move was made in; null for notes outside any turn. */
  turn: number | null;
  text: string;
}

export interface PolicyRuntimeIdentity {
  id: string;
  provider: string;
  runner: string;
  artifacts: Record<string, string>;
  config: Record<string, unknown>;
  supports_search: boolean;
  contract_version?: number;
}

export interface RecordedPlayer {
  seat: number;
  player_id: string;
  kind: 'human' | 'bot';
  name: string;
  deck: string;
  /** The deck's display name ("GW Allies"); the recording index supplies it. */
  deck_name?: string;
  version: string | null;
  inference: (Record<string, unknown> & { runtime?: PolicyRuntimeIdentity }) | null;
}

export interface GameSummary extends TraceSummary {
  status: 'active' | 'completed' | 'stopped' | 'interrupted';
  ended_at: string | null;
  players: RecordedPlayer[];
  origin: string;
  record_version: number;
  revision: number | null;
  mine: boolean;
  replay_available: boolean;
  play_session_id?: string | null;
  game_number?: number | null;
  feedback_truncated?: boolean;
  feedback: { note: string; created_at: string; author_id: string | null; decision_address: string | null }[];
}

export interface TraceSummary {
  id: string;
  timestamp: string | null;
  winner: number | null;
  end_reason: string | null;
  num_events: number;
}

export type VillainType = 'passive' | 'random' | 'search' | 'policy';

// Provider-authored presentation from GET /api/opponents.
export interface TrainedOpponent {
  supports_search?: boolean;
  search_reason?: string | null;
  policy?: PolicyRuntimeIdentity;
  sha256: string;
  name: string;
  /** How it decides: "scripted", "checkpoint w17", "search". */
  kind?: string;
  producer: string;
  group: string;
  details: { label: string; value: string }[];
  available: boolean;
  reason: string | null;
}

export interface OpponentCatalog {
  advice_available?: boolean;
  extension?: string | null;
  watch_players?: string[];
  opponents: TrainedOpponent[];
  catalog_error: string | null;
}

export interface OpponentConfig {
  villain_type: VillainType;
  villain_sims?: number;
  opponent_sha256?: string;
}

/** What a game recorded about its policy; an unrecorded value is absent. */
export interface CheckpointProvenance {
  policy?: PolicyRuntimeIdentity;
  source?: string | null;
  world?: string | null;
  model?: string | null;
  step?: number | null;
}

export interface TraceConfig {
  opponent?: { name: string; sha256: string; producer?: string } | null;
  opponent_checkpoint?: CheckpointProvenance | null;
  watch_players?: string[] | null;
  mirror?: import('./search').SearchConfig | null;
  hero_deck: Record<string, number>;
  villain_deck: Record<string, number>;
  villain_type: string;
  seed?: number | null;
  hero_deck_name?: string;
  villain_deck_name?: string;
  hero_sideboard?: Record<string, number>;
  villain_sideboard?: Record<string, number>;
  villain_sims?: number | null;
  villain_checkpoint?: string | null;
  villain_deterministic?: boolean;
  asset_pack?: AssetPackReference | null;
  decklists?: Decklists;
}

/** Open: both players know each other's main deck and sideboard. Closed: neither. */
export type Decklists = 'open' | 'closed';

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
  auto?: boolean;
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
  can_feedback?: boolean;
  read_only?: boolean;
  players?: RecordedPlayer[];
  record_version?: number;
  origin?: string;
  id?: string;
  config: TraceConfig;
  events: TraceEvent[];
  final_observation: Observation;
  winner: number | null;
  end_reason: string;
  timestamp: string;
}

/** Viewer-safe names captured from authoritative projections around a beat. */
export interface PresentationLabels {
  objects: Record<string, string>;
  players: Record<string, string>;
  stacks: Record<string, string>;
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

export type WorkStatus = {
  operation_id: string;
  sequence: number;
  kind: 'load' | 'choose' | 'search' | 'advance';
  started_at: number;
  state: 'running' | 'complete' | 'failed';
} & ({
  scope: 'setup';
  setup_id: string;
  loading_seat: number | null;
  match_id: null;
  revision: null;
  chooser: null;
} | {
  scope: 'match';
  setup_id: null;
  loading_seat: null;
  match_id: string;
  revision: number;
  chooser: number | null;
});

export type ServerMessage = (
  | { type: 'work_status'; session_id: string; work: WorkStatus }

  | {
      type: 'observation';
      data: Observation;
      actions: ActionOption[];
      // ActionSpaceEnum name (PRIORITY, SCRY, PAY_OR_NOT, MODAL, ...) — what
      // kind of decision the hero is being asked to make.
      action_space?: string;
      log?: LogLine[];
      stops?: StopsConfig;
      deck_names?: DeckNames;
      asset_pack?: AssetPackReference | null;
      auto_passed?: number;
      boards?: StepBoardPatch[];
      session_id?: string;
      resume_token?: string;
      frame?: ExperienceFrame;
      recovery?: RecoveryEnvelope;
    }
  | {
      type: 'game_over';
      data: Observation;
      winner: number | null;
      log?: LogLine[];
      stops?: StopsConfig;
      deck_names?: DeckNames;
      asset_pack?: AssetPackReference | null;
      auto_passed?: number;
      boards?: StepBoardPatch[];
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
  | { type: 'error'; message: string; session_ended?: boolean }
) & {
  work?: WorkStatus | null;
  mirror_retry?: import('./search').PendingSearch | null;
  mirror?: import('./search').SearchConfig | null;
  table?: import('./shared-table-protocol').TableSnapshot;
  session_id?: string;
  resume_token?: string;
  participant_id?: string;
  launch?: string;
  launch_setup?: LaunchSetup;
};

export interface LaunchSetup {
  mirror?: import('./search').SearchConfig | null;
  hero_deck?: string;
  villain_deck?: string;
  villain_type: string;
  villain_sims?: number;
  opponent_sha256?: string;
  decklists: Decklists;
  deck_import?: { hero: { source: string; filename: string }; villain: { source: string; filename: string } };
}

export type ClientMessage = import('./shared-table-protocol').SharedTableRequest;

export type ConnectionState =
  | 'disconnected'
  | 'connecting'
  | 'connected'
  | 'reconnecting';
