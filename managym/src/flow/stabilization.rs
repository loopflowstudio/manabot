//! Deterministic post-commit stabilization before priority.
//!
//! CR 117.5 and 704.3 require the engine to finish applicable state-based
//! actions, collect and stack waiting triggers, and then check state-based
//! actions again before a player receives priority. This module is the one
//! authority for that fixed point.

use crate::{
    agent::action::ActionSpace,
    flow::{game::Game, sba::SbaProgress, turn::StepKind},
};

impl Game {
    /// Reach the CR 117.5 / 704.3 fixed point before publishing priority.
    ///
    /// A trigger target choice is itself a required player decision, so it may
    /// suspend stabilization. Completing that choice re-enters this routine
    /// through the ordinary action boundary before priority can be granted.
    pub(crate) fn stabilize_before_priority(&mut self) -> Option<ActionSpace> {
        self.state.priority_mut().sba_done = false;
        let cleanup = self.state.turn.current_step_kind() == StepKind::Cleanup;

        loop {
            self.refresh_control();
            // CR 603.3b: finish this placement round before another SBA or
            // collection pass, including after an ordering/target suspension.
            if self.state.trigger_placement.is_some() {
                if let Some(choice) = self.flush_triggers() {
                    return Some(choice);
                }
            }
            self.collect_waiting_triggers();

            // CR 704.3 — after any SBA is performed, check the complete set
            // again before putting waiting triggers on the stack.
            let progress = self.perform_state_based_actions();
            if cleanup && progress != SbaProgress::Stable {
                self.state.set_cleanup_priority(true);
            }
            if self.is_game_over() {
                return None;
            }
            match progress {
                SbaProgress::Pending => return self.suspended_decision_action_space(),
                SbaProgress::Changed => continue,
                SbaProgress::Stable => {}
            }

            // CR 603.3b (2026-09-25): controller choices within each of two
            // APNAP passes. Placement arrivals wait for the next round.
            if !self.state.pending_triggers.is_empty() {
                if cleanup {
                    self.state.set_cleanup_priority(true);
                }
                if let Some(choice) = self.flush_triggers() {
                    return Some(choice);
                }
                continue;
            }
            self.state.priority_mut().sba_done = true;
            return None;
        }
    }
}
