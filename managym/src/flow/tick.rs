// tick.rs
// Game loop: step, tick, turn_tick, tick_priority, and turn-based actions.

use crate::{
    agent::action::{Action, ActionSpace, ActionSpaceKind, AgentError},
    flow::{
        combat::CombatState,
        event::GameEvent,
        game::Game,
        turn::{StepKind, TurnState},
    },
    state::game_object::PlayerId,
};

impl Game {
    pub fn step(&mut self, action: usize) -> Result<bool, AgentError> {
        if self.is_game_over() {
            return Err(AgentError("game is over".to_string()));
        }

        let action_space = self
            .current_action_space
            .take()
            .ok_or_else(|| AgentError("no active action space".to_string()))?;

        if action >= action_space.actions.len() {
            self.current_action_space = Some(action_space.clone());
            return Err(AgentError(format!(
                "Action index {action} out of bounds: {}",
                action_space.actions.len()
            )));
        }

        let selected_action = action_space.actions[action].clone();
        if let Err(error) = self.execute_action(&selected_action) {
            self.current_action_space = Some(action_space);
            return Err(error);
        }
        self.state.priority.sba_done = false;

        let game_over = self.tick();
        if game_over {
            if let Some(winner) = self.winner_index() {
                self.trackers[winner].on_game_won();
            }
        }
        self.assert_stack_consistent();

        Ok(game_over)
    }

    pub fn play(&mut self) {
        while !self.is_game_over() {
            let _ = self.step(0);
        }
    }

    pub fn drain_events(&mut self) -> Vec<GameEvent> {
        std::mem::take(&mut self.state.events)
    }

    pub(crate) fn tick(&mut self) -> bool {
        loop {
            let action_space = self.turn_tick();
            if self.is_game_over() {
                self.current_action_space = Some(ActionSpace::game_over());
                return true;
            }

            if let Some(space) = action_space {
                if !self.skip_trivial || space.actions.len() > 1 {
                    self.current_action_space = Some(space);
                    return false;
                }

                self.skip_trivial_count += 1;
                if let Some(action) = space.actions.first() {
                    if self.execute_action(action).is_err() {
                        return true;
                    }
                }
                continue;
            }
        }
    }

    fn turn_tick(&mut self) -> Option<ActionSpace> {
        let step = self.state.turn.current_step_kind();

        if !self.state.turn.step_initialized {
            self.on_step_start(step);
            self.state.turn.step_initialized = true;
            self.state.turn.turn_based_actions_complete = false;
        }

        if !self.state.turn.turn_based_actions_complete {
            if let Some(space) = self.perform_turn_based_actions(step) {
                return Some(space);
            }
            self.state.turn.turn_based_actions_complete = true;
        }

        if TurnState::step_has_priority(step) {
            if let Some(space) = self.tick_priority() {
                return Some(space);
            }
        }

        // CR 106.4 — Unspent mana empties at the end of each step and phase.
        self.clear_mana_pools();
        self.on_step_end(step);
        self.state.turn.advance_step();

        None
    }

    fn on_step_start(&mut self, step: StepKind) {
        self.state.priority.start_round(self.active_player());
        if step == StepKind::Untap {
            self.emit(GameEvent::TurnStarted {
                player: self.active_player(),
            });
        }
        self.emit(GameEvent::StepStarted { step });
        match step {
            StepKind::BeginningOfCombat => {
                // CR 507.1 — Beginning of combat creates/refreshes combat state.
                self.state.combat = Some(CombatState::default());
            }
            StepKind::DeclareAttackers => {
                // CR 508.1 — Active player declares attackers.
                let active = self.active_player();
                let eligible = self.eligible_attackers(active);
                let combat = self.state.combat.get_or_insert_with(CombatState::default);
                combat.attackers_to_declare = eligible;
            }
            StepKind::DeclareBlockers => {
                // CR 509.1 — Defending player declares blockers.
                let defender = self.non_active_player();
                let eligible = self.eligible_blockers(defender);
                let combat = self.state.combat.get_or_insert_with(CombatState::default);
                combat.blockers_to_declare = eligible;
            }
            _ => {}
        }
    }

    fn on_step_end(&mut self, step: StepKind) {
        if matches!(step, StepKind::EndOfCombat) {
            // CR 511.3 — Creatures stop being attacking as combat ends.
            for permanent in self.state.permanents.iter_mut().flatten() {
                permanent.attacking = false;
            }
            self.state.combat = None;
        }
    }

    fn perform_turn_based_actions(&mut self, step: StepKind) -> Option<ActionSpace> {
        match step {
            StepKind::Untap => {
                // CR 502.2 — Active player untaps permanents they control.
                let active = self.active_player();
                self.mark_permanents_not_summoning_sick(active);
                self.untap_all_permanents(active);
                None
            }
            StepKind::Draw => {
                // CR 504.1 — Active player draws one card in the draw step.
                let active = self.active_player();
                // The player who goes first skips their draw on turn 1.
                let is_first_player_first_turn =
                    self.state.turn.turn_number == 1 && active == PlayerId(0);
                if !is_first_player_first_turn {
                    self.draw_cards(active, 1);
                }
                None
            }
            StepKind::DeclareAttackers => {
                let active = self.active_player();
                let combat = self.state.combat.get_or_insert_with(CombatState::default);
                let attacker = combat.attackers_to_declare.pop()?;
                Some(ActionSpace {
                    player: Some(active),
                    kind: ActionSpaceKind::DeclareAttacker,
                    actions: vec![
                        Action::DeclareAttacker {
                            player: active,
                            permanent: attacker,
                            attack: true,
                        },
                        Action::DeclareAttacker {
                            player: active,
                            permanent: attacker,
                            attack: false,
                        },
                    ],
                    focus: Vec::new(),
                })
            }
            StepKind::DeclareBlockers => {
                let defending = self.non_active_player();
                let combat = self.state.combat.get_or_insert_with(CombatState::default);
                let Some(blocker) = combat.blockers_to_declare.pop() else {
                    self.cleanup_illegal_menace_blocks();
                    return None;
                };
                let attackers = combat.attackers.clone();

                let mut actions = Vec::with_capacity(attackers.len() + 1);
                for attacker in &attackers {
                    if self.blocker_can_block_attacker(blocker, *attacker) {
                        actions.push(Action::DeclareBlocker {
                            player: defending,
                            blocker,
                            attacker: Some(*attacker),
                        });
                    }
                }
                actions.push(Action::DeclareBlocker {
                    player: defending,
                    blocker,
                    attacker: None,
                });

                Some(ActionSpace {
                    player: Some(defending),
                    kind: ActionSpaceKind::DeclareBlocker,
                    actions,
                    focus: Vec::new(),
                })
            }
            StepKind::CombatDamage => {
                // CR 510.1 — Assign and deal combat damage.
                self.resolve_combat_damage();
                None
            }
            StepKind::Cleanup => {
                // CR 514.2 — Damage marked on permanents is removed during cleanup.
                self.clear_damage();
                self.clear_temporary_modifiers();
                None
            }
            _ => None,
        }
    }

    fn tick_priority(&mut self) -> Option<ActionSpace> {
        loop {
            if let Some(choice_space) = self.pending_choice_action_space() {
                return Some(choice_space);
            }

            if !self.state.priority.sba_done {
                // CR 117.5, 704.3 — Check state-based actions before granting priority.
                self.perform_state_based_actions();
                self.state.priority.sba_done = true;
                if self.is_game_over() {
                    return None;
                }
            }

            // CR 603.3 — Flush pending triggers before granting priority.
            if self.state.pending_trigger_choice.is_some()
                || !self.state.pending_triggers.is_empty()
            {
                if let Some(space) = self.flush_triggers() {
                    return Some(space);
                }
            }

            if self.state.priority.consecutive_passes >= self.state.players.len() {
                self.state.priority.start_round(self.active_player());
                if !self.state.stack_objects.is_empty() {
                    // CR 117.4, 405.2 — If all players pass with a nonempty stack, resolve top object.
                    self.resolve_top_of_stack();
                    self.state.priority.on_non_pass_action(self.active_player());
                    continue;
                }
                return None;
            }

            let player = self.state.priority.holder;

            if self.skip_trivial && !self.can_player_act(player) {
                let next = self.next_player(player);
                self.state.priority.on_pass(next);
                continue;
            }

            let actions = self.compute_player_actions(player);
            return Some(ActionSpace {
                player: Some(player),
                kind: ActionSpaceKind::Priority,
                actions,
                focus: Vec::new(),
            });
        }
    }
}
