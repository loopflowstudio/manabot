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

        Ok(self.finish_action_step())
    }

    /// Finish one externally committed action after its complete declaration
    /// has been applied. Structured commands reuse this boundary so an atomic
    /// cast-plus-target has the same SBA, priority, tracking, and consistency
    /// behavior as the legacy positional path.
    pub(crate) fn finish_action_step(&mut self) -> bool {
        self.state.priority.sba_done = false;

        let game_over = self.tick();
        if game_over {
            if let Some(winner) = self.winner_index() {
                self.trackers[winner].on_game_won();
            }
        }
        self.assert_stack_consistent();

        game_over
    }

    /// Publish a newly computed external decision and invalidate commands
    /// decoded from every previously published structured offer set.
    pub(crate) fn publish_action_space(&mut self, action_space: ActionSpace) {
        self.decision_epoch = self
            .decision_epoch
            .checked_add(1)
            .expect("decision epoch exhausted");
        self.current_action_space = Some(action_space);
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
                self.publish_action_space(ActionSpace::game_over());
                return true;
            }

            if let Some(space) = action_space {
                if !self.skip_trivial || space.actions.len() > 1 {
                    self.publish_action_space(space);
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
            let player = self.active_player();
            let event = GameEvent::TurnStarted {
                player,
                turn_number: self.state.turn.turn_number,
            };
            self.state.events.push(event.clone());
            self.emit_observation_event(event);
        }
        // Emitted (not just logged) so step-based triggers such as
        // "at the beginning of your upkeep" can see it.
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
            // Until-end-of-combat mana (firebending) empties now.
            self.clear_combat_mana_pools();
        }
        if matches!(step, StepKind::Cleanup) {
            // Safety net: no combat mana survives the turn.
            self.clear_combat_mana_pools();
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
                let Some(attacker) = combat.attackers_to_declare.pop() else {
                    // CR 508.1 — Attack triggers fire once the whole batch of
                    // attackers has been declared.
                    let attackers = combat.attackers.clone();
                    if !attackers.is_empty() {
                        self.emit(GameEvent::AttackersDeclared {
                            player: active,
                            attackers: attackers.clone(),
                        });
                        let attackers = attackers
                            .into_iter()
                            .filter_map(|permanent| self.permanent_event_ref(permanent))
                            .collect::<Vec<_>>();
                        if !attackers.is_empty() {
                            self.emit_observation_event(GameEvent::CombatAttackersDeclared {
                                player: active,
                                defender: self.non_active_player(),
                                attackers,
                            });
                        }
                    }
                    return None;
                };
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
                    let assignments: Vec<_> = self
                        .state
                        .combat
                        .as_ref()
                        .map(|combat| {
                            combat
                                .attacker_to_blockers
                                .iter()
                                .map(|(attacker, blockers)| (*attacker, blockers.clone()))
                                .collect()
                        })
                        .unwrap_or_default();
                    let assignments = assignments
                        .into_iter()
                        .filter_map(|(attacker, blockers)| {
                            let attacker = self.permanent_event_ref(attacker)?;
                            let blockers = blockers
                                .into_iter()
                                .filter_map(|blocker| self.permanent_event_ref(blocker))
                                .collect::<Vec<_>>();
                            (!blockers.is_empty()).then_some((attacker, blockers))
                        })
                        .collect::<Vec<_>>();
                    if !assignments.is_empty() {
                        self.emit_observation_event(GameEvent::BlockersDeclared { assignments });
                    }
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
            // A suspended resolution owns the game until its decision is
            // made — no SBAs, triggers, or priority in between (CR 608.2).
            if let Some(decision_space) = self.suspended_decision_action_space() {
                return Some(decision_space);
            }

            if let Some(choice_space) = self.pending_choice_action_space() {
                return Some(choice_space);
            }

            if let Some(space) = self.stabilize_before_priority() {
                return Some(space);
            }
            if self.is_game_over() {
                return None;
            }

            if self.state.priority.consecutive_passes >= self.state.players.len() {
                self.state.priority.start_round(self.active_player());
                if !self.stack_is_empty() {
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
