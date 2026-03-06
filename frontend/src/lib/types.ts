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
  card: string | null;
  focus: number[];
  description: string;
}

export interface BoardActionTarget {
  objectId: number;
  actionIndexes: number[];
}

export interface GameLogEntry {
  id: string;
  actor: 'hero' | 'villain' | 'system';
  text: string;
  details: string[];
}

export interface TraceSummary {
  id: string;
  timestamp: string | null;
  winner: number | null;
  end_reason: string | null;
  num_events: number;
}

export interface TraceConfig {
  hero_deck: Record<string, number>;
  villain_deck: Record<string, number>;
  villain_type: string;
  seed?: number | null;
}

export interface TraceEvent {
  actor: 'hero' | 'villain';
  observation: Observation;
  actions: ActionOption[];
  action: number;
  action_description: string;
  reward: number;
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
}

export type ServerMessage =
  | {
      type: 'observation';
      data: Observation;
      actions: ActionOption[];
      log?: string[];
      session_id?: string;
      resume_token?: string;
    }
  | { type: 'game_over'; data: Observation; winner: number | null; log?: string[] }
  | { type: 'error'; message: string };

export type ClientMessage =
  | { type: 'new_game'; config?: Record<string, unknown> }
  | { type: 'action'; index: number }
  | { type: 'resume'; session_id: string; resume_token: string };

export type ConnectionState =
  | 'disconnected'
  | 'connecting'
  | 'connected'
  | 'reconnecting';
