import type { TableSnapshot } from '../../src/lib/shared-table-protocol';

let attempt = 0;

/** A fresh authoritative attempt for controlled WebSocket game fixtures. */
export function tableSnapshot(): TableSnapshot {
  return {
    contract: 'testing-house-v1', table_id: 'fixture-table', table_revision: 0,
    attempt_id: `fixture-attempt-${++attempt}`, mode: 'live',
    access: { identity: { viewer_id: 'fixture-viewer', table_id: 'fixture-table', rules_viewer: 0 },
      role: 'pilot', capabilities: ['view_table', 'submit_live_command', 'configure_match', 'explore_study'], grant_revision: 0 },
    participants: [], beliefs: [], decisions: [], opponent_label: null, watcher_invite: null,
  };
}
