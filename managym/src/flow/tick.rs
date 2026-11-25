// tick.rs
// Game loop: step, tick, turn_tick, tick_priority, and turn-based actions.

use crate::{
    agent::action::{Action, ActionSpace, ActionSpaceKind, AgentError},
    flow::{
        event::GameEvent,
        game::{Game, GameOutcome, GameProcedure},
        priority::LoopSample,
        proposed_event::EventBinding,
        turn::{PhaseKind, StepKind, TurnState, PHASE_STEPS},
    },
    state::{card::StaticAbility, game_object::PlayerId, permanent::CounterKind, zone::ZoneType},
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

        let pending_memory = crate::agent::decision_memory::capture(
            self,
            &action_space,
            &action_space.actions[action],
        );
        let exposure = self.exposure_choice(&action_space, action);
        let selected_action = action_space.actions[action].clone();
        if let Err(error) = self.execute_action(&selected_action) {
            self.current_action_space = Some(action_space);
            return Err(error);
        }

        crate::agent::decision_memory::commit(self, pending_memory);
        if let Some(choice) = exposure {
            self.commit_capture_choice(choice);
        }
        Ok(self.finish_action_step())
    }

    /// Finish one externally committed action after its complete declaration
    /// has been applied. Structured commands reuse this boundary so an atomic
    /// cast-plus-target has the same SBA, priority, tracking, and consistency
    /// behavior as the legacy positional path.
    pub(crate) fn finish_action_step(&mut self) -> bool {
        self.state.priority_mut().sba_done = false;

        let game_over = self.tick();
        self.assert_stack_consistent();

        game_over
    }

    /// Publish a newly computed external decision and invalidate commands
    /// decoded from every previously published structured offer set.
    pub(crate) fn publish_action_space(&mut self, mut action_space: ActionSpace) {
        self.refresh_visible_library_tops();
        self.assert_library_knowledge();
        self.decision_epoch = self
            .decision_epoch
            .checked_add(1)
            .expect("decision epoch exhausted");
        action_space.player = action_space
            .actions
            .first()
            .map(|action| action.player())
            .map(|player| self.decision_controller(player));
        self.remember_controlled_hands();
        self.current_action_space = Some(action_space);
    }

    pub fn play(&mut self) {
        while !self.is_game_over() {
            let _ = self.step(0);
        }
    }

    pub fn drain_events(&mut self) -> Vec<GameEvent> {
        self.state.events_mut().take()
    }

    pub(crate) fn tick(&mut self) -> bool {
        loop {
            let action_space = self.opening_actions().or_else(|| self.turn_tick());
            if self.is_game_over() {
                self.finish_combat_capture();
                self.publish_action_space(ActionSpace::game_over());
                return true;
            }

            if let Some(space) = action_space {
                if let Some(action) = self.automatic_payment_action(&space) {
                    self.execute_action(&action)
                        .expect("automatic payment witness executes");
                    self.skip_trivial_count += 1;
                    continue;
                }
                if !self.skip_trivial || space.actions.len() > 1 {
                    if space.kind != ActionSpaceKind::Priority && space.actions.len() > 1 {
                        // A player chooses: the chain is no longer mandatory.
                        self.state.priority_mut().clear_loop_watch();
                    }
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
            self.state.set_step_initialized(true);
            self.state.set_turn_based_actions_complete(false);
        }

        if let Some(space) = self.suspended_decision_action_space() {
            return Some(space);
        }

        if !self.state.turn.turn_based_actions_complete {
            if let Some(space) = self.perform_turn_based_actions(step) {
                return Some(space);
            }
            self.state.set_turn_based_actions_complete(true);
            if let Some(space) = self.suspended_decision_action_space() {
                return Some(space);
            }
        }

        if step != StepKind::Untap {
            if let Some(space) = self.tick_priority(step) {
                return Some(space);
            }
            if !self.state.turn.step_initialized || self.is_game_over() {
                return None;
            }
        }

        self.on_step_end(step);
        // Read before writing: a mutable borrow of `state` advances every
        // derived-cache revision, and most damage steps have no second pass.
        if step == StepKind::CombatDamage
            && self
                .state
                .combat
                .as_ref()
                .is_some_and(|combat| combat.in_first_strike_damage_step())
        {
            self.state.combat.as_mut().unwrap().regular_damage_step = true;
            self.state.turn.step_initialized = false;
            self.state.turn.turn_based_actions_complete = false;
            return None;
        }
        if self.state.turn.cleanup_priority {
            // CR 514.3a: repeat cleanup after its priority round finishes.
            self.state.turn.start_cleanup();
        } else {
            self.state.advance_step();
            self.skip_combat_phases_if_required();
        }

        None
    }

    /// CR 725.1d: leaving a combat step early also ends its combat phase.
    pub(crate) fn end_step_for_cleanup(&mut self) {
        let step = if self.state.turn.current_phase_kind() == PhaseKind::Combat {
            StepKind::EndOfCombat
        } else {
            self.state.turn.current_step_kind()
        };
        self.on_step_end(step);
        self.state.turn.start_cleanup();
    }

    /// A player skipping their combat phases passes from the precombat main
    /// phase to the postcombat one, past any additional combat (CR 500.11).
    fn skip_combat_phases_if_required(&mut self) {
        while self.state.turn.current_phase_kind() == PhaseKind::Combat
            && self.state.turn.current_step == 0
            && self.skips_combat_phases(self.active_player())
        {
            self.state.turn.current_step = PHASE_STEPS[PhaseKind::Combat as usize].len() - 1;
            self.state.advance_step();
        }
    }

    /// Give the active player priority with no passes yet in the round, as
    /// at a step's start and after any non-pass action (CR 117.3).
    pub(crate) fn restart_priority_round(&mut self) {
        let active = self.active_player();
        self.state.priority_mut().start_round(active);
    }

    fn on_step_start(&mut self, step: StepKind) {
        self.restart_priority_round();
        self.state.priority_mut().clear_loop_watch();
        if step == StepKind::Untap {
            let player = self.active_player();
            if let Some(exposure) = self.exposure_mut() {
                exposure.mana_turns[player.0].push(Default::default());
            }
            let event = GameEvent::TurnStarted {
                player,
                turn_number: self.state.turn.turn_number,
            };
            self.state.events_mut().push(event.clone());
            self.emit_observation_event(event);
        }
        // Emitted (not just logged) so step-based triggers such as
        // "at the beginning of your upkeep" can see it.
        self.emit(GameEvent::StepStarted { step });
        match step {
            StepKind::BeginningOfCombat => {
                // CR 507.1 — Beginning of combat creates/refreshes combat state.
                self.state.combat = Some(Box::default());
            }
            StepKind::DeclareAttackers => {
                // CR 508.1 — Active player declares attackers.
                let active = self.active_player();
                let eligible = self.eligible_attackers(active);
                let combat = self.state.combat.get_or_insert_default();
                combat.already_attacking = combat.attack_targets.len();
                combat.attackers_to_declare = eligible;
            }
            StepKind::DeclareBlockers => {
                // CR 509.1 — Defending player declares blockers.
                let defender = self.non_active_player();
                let eligible = self.eligible_blockers(defender);
                let combat = self.state.combat.get_or_insert_default();
                combat.blockers_to_declare = eligible;
            }
            _ => {}
        }
    }

    fn on_step_end(&mut self, step: StepKind) {
        // CR 106.4 — Unspent mana empties at the end of each step and phase.
        self.clear_mana_pools();
        if step == StepKind::EndOfCombat {
            // CR 511.3 — Creatures stop being attacking as combat ends.
            for permanent in self.state.permanents.iter_mut().flatten() {
                permanent.attacking = false;
            }
            self.finish_combat_capture();
            self.state.combat = None;
        }
        if matches!(step, StepKind::EndOfCombat | StepKind::Cleanup) {
            self.clear_combat_mana_pools();
        }
        self.remember_controlled_hands();
    }

    fn perform_turn_based_actions(&mut self, step: StepKind) -> Option<ActionSpace> {
        match step {
            StepKind::Untap => {
                // CR 502.2 — Active player untaps permanents they control.
                // "Until your next turn" ends as the turn begins, and their
                // phased-out permanents phase in first (CR 502.1).
                let active = self.active_player();
                self.end_until_your_next_turn_effects(active);
                self.phase_in(active);
                self.expire_cast_restrictions(active);
                self.mark_permanents_not_summoning_sick(active);
                self.untap_all_permanents(active);
                self.untap_during_other_untap_steps(active);
                None
            }
            StepKind::Main => {
                // CR 714.3c — As the precombat main phase begins, the active
                // player puts a lore counter on each Saga they control.
                let active = self.active_player();
                let events = self
                    .battlefield_permanents(active)
                    .into_iter()
                    .filter(|&id| self.characteristics(id).final_chapter().is_some())
                    .map(
                        |id| crate::flow::proposed_event::ProposedEvent::CounterChange {
                            provenance: crate::flow::proposed_event::CounterProvenance::new(
                                crate::flow::proposed_event::EventCause::TurnAction,
                            ),
                            target: self.permanent_object(id).expect("battlefield"),
                            kind: CounterKind::Lore,
                            delta: 1,
                            by: Some(active),
                        },
                    );
                let mut frame = crate::flow::decision::EffectFrame::new(active, Vec::new());
                frame.batch = Some(self.event_batch(EventBinding::None, events));
                frame.completion = crate::flow::decision::FrameCompletion::Continue;
                self.run_frame(frame);
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
                let combat = self.state.combat.get_or_insert_default();
                let Some(attacker) = combat.attackers_to_declare.pop() else {
                    // CR 508.1 — Attack triggers fire once the whole batch of
                    // attackers has been declared.
                    let first = std::mem::take(&mut combat.already_attacking);
                    if first == combat.attack_targets.len() {
                        return None;
                    }
                    let declared = combat.attack_targets[first..].to_vec();
                    let mut attackers = Vec::with_capacity(declared.len());
                    let mut permanent_defenders = Vec::new();
                    let mut visible_defenders = Vec::new();
                    self.state.turn.history[active.0].attacked = true;
                    for (attacker, target) in declared {
                        attackers.push(attacker);
                        let object = self.permanent_object(attacker).expect("declared attacker");
                        self.state.turn.record_attack(object);
                        if let Some(target @ super::combat::AttackTarget::Permanent(defender)) =
                            target
                        {
                            permanent_defenders.push((attacker, target));
                            if let (Some(a), Some(d)) = (
                                self.permanent_event_ref(attacker),
                                self.object_event_ref(defender),
                            ) {
                                visible_defenders.push((a, d));
                            }
                        }
                    }
                    let visible_attackers: Vec<_> = attackers
                        .iter()
                        .filter_map(|&permanent| self.permanent_event_ref(permanent))
                        .collect();
                    self.emit(GameEvent::AttackersDeclared {
                        permanent_defenders: (!permanent_defenders.is_empty())
                            .then(|| Box::new(permanent_defenders)),
                        player: active,
                        attackers,
                    });
                    if !visible_attackers.is_empty() {
                        self.emit_observation_event(GameEvent::CombatAttackersDeclared {
                            player: active,
                            defender: self.non_active_player(),
                            permanent_defenders: (!visible_defenders.is_empty())
                                .then(|| Box::new(visible_defenders)),
                            attackers: visible_attackers,
                        });
                    }
                    return None;
                };
                let must_attack = self.must_attack(attacker);
                let mut targets = self.declaration_targets(active);
                let actions = targets
                    .next()
                    .map(Some)
                    .into_iter()
                    .chain((!must_attack).then_some(None))
                    .chain(targets.map(Some))
                    .map(|attack| Action::DeclareAttacker {
                        player: active,
                        permanent: attacker,
                        attack,
                    })
                    .collect();
                Some(ActionSpace {
                    player: Some(active),
                    kind: ActionSpaceKind::DeclareAttacker,
                    actions,
                    focus: Vec::new(),
                })
            }
            StepKind::DeclareBlockers => {
                let defending = self.non_active_player();
                let combat = self.state.combat.get_or_insert_default();
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
                let attackers = combat.attackers().collect::<Vec<_>>();

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
                // CR 509.1c: a blocker able to block an attacker every able
                // creature must block declares one of those blocks.
                let required = |action: &Action| {
                    matches!(action, Action::DeclareBlocker { attacker: Some(attacker), .. }
                        if self.must_be_blocked_by_all_able(*attacker))
                };
                if actions.iter().any(required) {
                    actions.retain(required);
                } else {
                    actions.push(Action::DeclareBlocker {
                        player: defending,
                        blocker,
                        attacker: None,
                    });
                }

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
                let player = self.active_player();
                let unlimited = self.has_no_maximum_hand_size_effect(player)
                    || self.has_functioning_static(Some(player), |ability| {
                        matches!(ability, StaticAbility::NoMaximumHandSize)
                    });
                if !unlimited && self.state.zones.size(ZoneType::Hand, player) > 7 {
                    self.suspend_rule_decision(crate::flow::decision::Decision::Discard {
                        player,
                        remaining: 1,
                    });
                    return self.suspended_decision_action_space();
                }
                // CR 514.2 — Damage marked on permanents is removed during cleanup.
                self.clear_damage();
                self.end_turn_modifiers();
                None
            }
            _ => None,
        }
    }

    fn tick_priority(&mut self, step: StepKind) -> Option<ActionSpace> {
        // An automatic pass changes only the priority holder. Reuse this
        // invocation's fixed-point proof until a stack object resolves.
        let mut stabilized = false;
        loop {
            // A suspended resolution owns the game until its decision is
            // made — no SBAs, triggers, or priority in between (CR 608.2).
            if let Some(decision_space) = self.suspended_decision_action_space() {
                return Some(decision_space);
            }

            if let Some(choice_space) = self.pending_choice_action_space() {
                return Some(choice_space);
            }

            if !stabilized {
                if let Some(space) = self.stabilize_before_priority() {
                    return Some(space);
                }
                // CR 514.3a: only an SBA or a waiting trigger opens cleanup priority.
                if !TurnState::step_has_priority(step) && !self.state.turn.cleanup_priority {
                    return None;
                }
                stabilized = true;
            }
            if self.is_game_over() {
                return None;
            }
            if self.state.priority.resolution_owed {
                // CR 117.5: state-based losses and newly stacked triggers
                // determine whether the resolution ended the game or the chain.
                self.state.priority_mut().resolution_owed = false;
                if self.end_mandatory_loop() {
                    return None;
                }
            }

            if self.state.priority.consecutive_passes >= self.state.players.len() {
                self.restart_priority_round();
                if !self.stack_is_empty() {
                    // CR 117.4, 405.2 — If all players pass with a nonempty stack, resolve top object.
                    self.resolve_top_of_stack();
                    if !self.state.turn.step_initialized || self.is_game_over() {
                        // A turn-ending resolution leaves this priority round.
                        return None;
                    }
                    self.state.priority_mut().resolution_owed = true;
                    stabilized = false;
                    continue;
                }
                return None;
            }

            let player = self.state.priority.holder;
            self.capture_priority_mana(player);
            let space = self.priority_action_space(player);
            if self.skip_trivial && space.actions.len() == 1 {
                self.skipped_priority_passes += 1;
                let next = self.next_player(player);
                self.state.priority_mut().on_pass(next);
                continue;
            }

            return Some(space);
        }
    }

    /// Count one finished resolution toward the CR 104.4b watch, ending the
    /// game in a draw once it shows a mandatory loop; true if it did. An
    /// emptied stabilized stack ends any chain, so it ends the watch instead.
    fn end_mandatory_loop(&mut self) -> bool {
        if self.stack_is_empty() {
            self.state.priority_mut().clear_loop_watch();
            return false;
        }
        // Taken out of the state so the sample can read the whole state.
        let mut watch = self
            .state
            .priority_mut()
            .loop_watch
            .take()
            .unwrap_or_default();
        let looped = watch.repeats(|| {
            let seats = [PlayerId(0), PlayerId(1)];
            LoopSample {
                fingerprint: self.state.loop_fingerprint(),
                life: seats.map(|seat| self.state.players[seat.0].life),
                poison: seats
                    .map(|seat| self.state.players[seat.0].counters.get(CounterKind::Poison)),
                can_lose: seats.map(|seat| !self.protected_from_loss(seat)),
            }
        });
        self.state.priority_mut().loop_watch = Some(watch);
        if looped {
            self.state.procedure = GameProcedure::Finished(GameOutcome::Draw {
                mandatory_loop: true,
            });
        }
        looped
    }
}
