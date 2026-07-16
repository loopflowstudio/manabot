//! Deterministic post-commit stabilization before priority.
//!
//! CR 117.5 and 704.3 require the engine to finish applicable state-based
//! actions, collect and stack waiting triggers, and then check state-based
//! actions again before a player receives priority. This module is the one
//! authority for that fixed point.

use crate::{agent::action::ActionSpace, flow::game::Game};

impl Game {
    /// Reach the CR 117.5 / 704.3 fixed point before publishing priority.
    ///
    /// A trigger target choice is itself a required player decision, so it may
    /// suspend stabilization. Completing that choice re-enters this routine
    /// through the ordinary action boundary before priority can be granted.
    pub(crate) fn stabilize_before_priority(&mut self) -> Option<ActionSpace> {
        self.journal_priority();
        self.state.priority.sba_done = false;

        loop {
            // Only committed facts enter trigger matching. Processing is
            // deterministic because events, trigger sources, and enqueue
            // order all have canonical sequence order.
            self.process_game_events();

            // CR 704.3 — after any SBA is performed, check the complete set
            // again before putting waiting triggers on the stack.
            if self.perform_state_based_actions() {
                if self.is_game_over() {
                    return None;
                }
                continue;
            }

            if self.is_game_over() {
                return None;
            }

            // CR 603.3, 603.3b — put waiting triggers on the stack in the
            // engine's deterministic APNAP/enqueue order. Stacking triggers
            // can emit committed facts, so always re-enter the fixed point.
            let had_waiting_triggers = self.state.pending_trigger_choice.is_some()
                || !self.state.pending_triggers.is_empty();
            if had_waiting_triggers {
                if let Some(choice) = self.flush_triggers() {
                    return Some(choice);
                }
                continue;
            }

            self.journal_priority();
            self.state.priority.sba_done = true;
            return None;
        }
    }
}
