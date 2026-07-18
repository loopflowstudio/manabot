import type { RestoredReplayDecision } from './replay-index';
import type { AdviceRequestIdentity } from './advice';
import type { Command, Observation, PresentationEvent } from './types';

export const TESTING_HOUSE_VERSION = 'testing-house-v1' as const;

export const TESTING_HOUSE_REQUEST_TYPES = [
  'join_table',
  'resume',
  'new_game',
  'rematch',
  'command',
  'action',
  'pass_turn',
  'set_stops',
  'transfer_pilot',
  'author_belief',
  'share_belief',
  'restore_decision',
  'retry_decision',
  'branch_reveal',
  'branch_preview',
  'return_from_branch',
  'return_to_live',
] as const;

export type ViewerRole = 'pilot' | 'watcher';

export type TableCapability =
  | 'view_table'
  | 'author_belief'
  | 'share_belief'
  | 'compare_advice'
  | 'explore_study'
  | 'submit_live_command'
  | 'configure_match'
  | 'transfer_pilot';

export interface ViewerIdentity {
  viewer_id: string;
  table_id: string;
  rules_viewer: 0;
}

export interface ViewerAccess {
  identity: ViewerIdentity;
  role: ViewerRole;
  capabilities: TableCapability[];
  grant_revision: number;
}

export interface ParticipantPresence {
  viewer_id: string;
  role: ViewerRole;
  connected: boolean;
}

export interface BeliefScenario {
  id: string;
  author_viewer_id: string;
  source: {
    decision_address: string;
    gam6_scenario_id: string;
    advice_identity: AdviceRequestIdentity;
  };
  audience: { kind: 'personal' } | { kind: 'table'; table_id: string };
  provenance: {
    kind: 'player_authored';
    created_at_table_revision: number;
    shared_at_table_revision: number | null;
  };
}

export interface TableDecisionSummary {
  address: string;
  ordinal: number;
  revision: number;
  prompt_id: number;
  offer_id: number;
}

export interface TableSnapshot {
  contract: typeof TESTING_HOUSE_VERSION;
  table_id: string;
  table_revision: number;
  mode: 'live' | 'study';
  access: ViewerAccess;
  participants: ParticipantPresence[];
  beliefs: BeliefScenario[];
  decisions: TableDecisionSummary[];
  opponent_label: string | null;
  watcher_invite: string | null;
}

interface GrantedRequest {
  grant_revision: number;
}

export type TestingHouseRequest =
  | {
      type: 'join_table';
      table_id: string;
      invite_token: string;
      presentation_cursor?: number | null;
    }
  | {
      type: 'resume';
      session_id: string;
      resume_token: string;
      presentation_cursor?: number | null;
    }
  | ({ type: 'new_game'; grant_revision?: number | null; config?: Record<string, unknown> })
  | (GrantedRequest & { type: 'rematch'; config?: Record<string, unknown> })
  | ({ type: 'command'; grant_revision?: number | null; command: Command })
  | ({ type: 'action'; grant_revision?: number | null; index: number })
  | ({ type: 'pass_turn'; grant_revision?: number | null })
  | ({
      type: 'set_stops';
      grant_revision?: number | null;
      stops?: Record<string, string[]> | null;
      stop_on_stack?: boolean | null;
      auto_pass?: boolean | null;
    })
  | (GrantedRequest & { type: 'transfer_pilot'; target_viewer_id: string })
  | (GrantedRequest & { type: 'author_belief'; scenario_id: string })
  | (GrantedRequest & { type: 'share_belief'; belief_id: string })
  | (GrantedRequest & { type: 'restore_decision'; address: string })
  | (GrantedRequest & { type: 'retry_decision'; address: string; command: Command })
  | (GrantedRequest & { type: 'branch_reveal'; attempt_id: string })
  | (GrantedRequest & {
      type: 'branch_preview';
      attempt_id: string;
      plan: 'played' | 'policy' | 'search';
    })
  | (GrantedRequest & { type: 'return_from_branch'; attempt_id: string })
  | (GrantedRequest & { type: 'return_to_live' });

export interface BranchRetryPayload {
  attempt_id: string;
  address: string;
  retry: {
    command: Command;
    projection: Observation;
    presentation: PresentationEvent[];
  };
}

export type TestingHouseControlEvent =
  | { type: 'table_snapshot'; table: TableSnapshot }
  | { type: 'belief_changed'; table: TableSnapshot; belief: BeliefScenario }
  | { type: 'role_changed'; table: TableSnapshot }
  | {
      type: 'decision_restored';
      address: string;
      restored: RestoredReplayDecision;
      table?: TableSnapshot | null;
    }
  | {
      type: 'branch_updated';
      attempt_id: string;
      phase: 'retry' | 'revealed' | 'preview';
      payload: BranchRetryPayload | Record<string, unknown>;
      table?: TableSnapshot | null;
    }
  | {
      type: 'branch_returned';
      restored: RestoredReplayDecision;
      table?: TableSnapshot | null;
    }
  | {
      type: 'control_error';
      code:
        | 'invalid_message'
        | 'unsupported_message'
        | 'forbidden'
        | 'stale_grant'
        | 'not_found'
        | 'conflict';
      message: string;
      table?: TableSnapshot | null;
    };

export interface TestingHouseV1ConformanceBundle {
  contract: typeof TESTING_HOUSE_VERSION;
  requests: TestingHouseRequest[];
  events: TestingHouseControlEvent[];
}
