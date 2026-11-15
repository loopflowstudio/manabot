// tick.rs
// Game loop: step, tick, turn_tick, tick_priority, and turn-based actions.

use crate::{
    agent::action::{Action, ActionSpace, ActionSpaceKind, AgentError},
    flow::{
        combat::CombatState,
        declaration::{DeclarationCollector, DeclarationCompletion, DeclarationStage},
        event::GameEvent,
        game::Game,
        turn::{PhaseKind, StepKind, TurnState, PHASE_STEPS},
    },
    state::{
        card::StaticAbility, game_object::PlayerId, permanent::CounterKind,
        stack_object::StackObject, zone::ZoneType,
    },
};

impl Game {
    pub fn step(&mut self, action: usize) -> Result<bool, AgentError> {
        self.step_with_completion(action, None)
    }

    pub(crate) fn step_with_completion(
        &mut self,
        action: usize,
        completion: Option<&mut DeclarationCollector>,
    ) -> Result<bool, AgentError> {
        self.ensure_playable()?;
        if self.is_game_over() {
            return Err(AgentError::Rules("game is over".to_string()));
        }
        let action_space = self
            .current_action_space
            .take()
            .ok_or_else(|| AgentError::Rules("no active action space".to_string()))?;

        if action >= action_space.actions.len() {
            self.current_action_space = Some(action_space.clone());
            return Err(AgentError::Rules(format!(
                "Action index {action} out of bounds: {}",
                action_space.actions.len()
            )));
        }

        let exposure = self.exposure_choice(&action_space, action);
        let selected_action = action_space.actions[action].clone();
        if let Err(error) = self.execute_action(&selected_action) {
            self.current_action_space = Some(action_space);
            return Err(error);
        }

        if let Some(choice) = exposure {
            self.card_capture
                .as_mut()
                .unwrap()
                .exposure
                .as_mut()
                .unwrap()
                .choices
                .push(choice);
        }
        self.finish_action_step_with_completion(completion)
    }

    /// Finish one externally committed action after its complete declaration
    /// has been applied. Structured commands reuse this boundary so an atomic
    /// cast-plus-target has the same SBA, priority, tracking, and consistency
    /// behavior as the legacy positional path.
    pub(crate) fn finish_action_step(&mut self) -> Result<bool, AgentError> {
        self.finish_action_step_with_completion(None)
    }

    fn finish_action_step_with_completion(
        &mut self,
        completion: Option<&mut DeclarationCollector>,
    ) -> Result<bool, AgentError> {
        self.state.priority_mut().sba_done = false;

        let game_over = self.tick_with_completion(completion)?;
        if game_over {
            if let Some(winner) = self.winner_index() {
                self.trackers[winner].on_game_won();
            }
        }
        self.assert_stack_consistent();

        Ok(game_over)
    }

    /// Reject a partially advanced failed stream until its owner resets it.
    pub fn ensure_playable(&self) -> Result<(), AgentError> {
        match &self.progression_error {
            Some(error) => Err(error.as_ref().clone()),
            None => Ok(()),
        }
    }

    /// Publish a newly computed external decision and invalidate commands
    /// decoded from every previously published structured offer set.
    pub(crate) fn publish_action_space(&mut self, mut action_space: ActionSpace) {
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

    pub fn play(&mut self) -> Result<(), AgentError> {
        self.ensure_playable()?;
        while !self.is_game_over() {
            self.step(0)?;
        }
        Ok(())
    }

    pub fn drain_events(&mut self) -> Vec<GameEvent> {
        self.state.events_mut().take()
    }

    pub(crate) fn tick(&mut self) -> Result<bool, AgentError> {
        self.tick_with_completion(None)
    }

    fn tick_with_completion(
        &mut self,
        completion: Option<&mut DeclarationCollector>,
    ) -> Result<bool, AgentError> {
        self.ensure_playable()?;
        let result = self.progress_until_decision(completion);
        if let Err(error) = &result {
            self.current_action_space = None;
            self.progression_error = Some(Box::new(error.clone()));
        }
        result
    }

    fn progress_until_decision(
        &mut self,
        mut completion: Option<&mut DeclarationCollector>,
    ) -> Result<bool, AgentError> {
        loop {
            let action_space = self.turn_tick(completion.as_deref_mut())?;
            if self.is_game_over() {
                self.publish_action_space(ActionSpace::game_over());
                return Ok(true);
            }

            if let Some(space) = action_space {
                if !self.skip_trivial || space.actions.len() > 1 {
                    self.publish_action_space(space);
                    return Ok(false);
                }
                self.skip_trivial_count += 1;
                if let Some(action) = space.actions.first() {
                    self.execute_action(action)?;
                }
                continue;
            }
        }
    }

    fn turn_tick(
        &mut self,
        mut completion: Option<&mut DeclarationCollector>,
    ) -> Result<Option<ActionSpace>, AgentError> {
        let step = self.state.turn.current_step_kind();

        if !self.state.turn.step_initialized {
            self.on_step_start(step);
            self.state.set_step_initialized(true);
            self.state.set_turn_based_actions_complete(false);
        }

        if let Some(space) = self.suspended_decision_action_space() {
            return Ok(Some(space));
        }

        if !self.state.turn.turn_based_actions_complete {
            if let Some(space) = self.perform_turn_based_actions(step, completion.as_deref_mut())? {
                return Ok(Some(space));
            }
            self.state.set_turn_based_actions_complete(true);
            if let Some(space) = self.suspended_decision_action_space() {
                return Ok(Some(space));
            }
        }

        if step == StepKind::Cleanup {
            if !self.state.turn.cleanup_priority {
                let choice = self.stabilize_before_priority();
                if choice.is_some() {
                    self.state.set_cleanup_priority(true);
                }
                if let Some(space) = choice {
                    return Ok(Some(space));
                }
            }
            if self.state.turn.cleanup_priority {
                if let Some(space) = self.tick_priority() {
                    return Ok(Some(space));
                }
                // CR 514.3a: repeat cleanup after this priority round finishes.
                self.state.set_cleanup_priority(false);
                self.state.set_turn_based_actions_complete(false);
                return Ok(None);
            }
        }

        if TurnState::step_has_priority(step) {
            if let Some(space) = self.tick_priority() {
                return Ok(Some(space));
            }
            if !self.state.turn.step_initialized {
                if let Some(collector) = completion {
                    collector.close(step);
                }
                return Ok(None);
            }
        }

        // CR 106.4 — Unspent mana empties at the end of each step and phase.
        self.clear_mana_pools();
        self.on_step_end(step);
        if let Some(collector) = completion {
            collector.close(step);
        }
        self.remember_controlled_hands();
        self.state.advance_step();
        self.skip_combat_phases_if_required();

        Ok(None)
    }

    /// End the turn (CR 723.1): exile every spell on the stack and remove
    /// every ability from it (723.1a), drop triggered abilities waiting to be
    /// put on it (723.1b), end combat, and skip every remaining phase and
    /// step but cleanup (723.1d). The cleanup step then runs normally, and
    /// may repeat if anything triggers during it (723.1e).
    pub(crate) fn end_the_turn(&mut self) {
        while let Some(object) = self.state.stack_objects.last() {
            match object {
                StackObject::Spell(spell) => {
                    let object_id = spell.object;
                    self.move_object(object_id, ZoneType::Exile);
                }
                _ => {
                    self.state.stack_objects_mut().pop();
                }
            }
        }
        self.state.pending_triggers_mut().clear();
        if self.state.combat.is_some() {
            self.on_step_end(StepKind::EndOfCombat);
        }
        self.clear_mana_pools();
        let turn = &mut self.state.turn;
        turn.additional_combats.clear();
        turn.combat_attackers = None;
        let ending = PhaseKind::Ending as usize;
        turn.current_phase = ending;
        turn.current_step = PHASE_STEPS[ending].len() - 1;
        turn.step_initialized = false;
        turn.turn_based_actions_complete = false;
        turn.cleanup_priority = false;
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
        if step == StepKind::Untap {
            let player = self.active_player();
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
            for index in 0..self.state.permanents.len() {
                let permanent_id = crate::state::game_object::PermanentId(index);
                if self.state.permanents[permanent_id].is_some() {
                    self.state.permanents[permanent_id]
                        .as_mut()
                        .expect("checked permanent")
                        .attacking = false;
                }
            }
            self.state.combat = None;
            // Until-end-of-combat mana (firebending) empties now.
            self.clear_combat_mana_pools();
        }
        if matches!(step, StepKind::Cleanup) {
            // Safety net: no combat mana survives the turn.
            self.clear_combat_mana_pools();
            self.expire_turn_cast_grants();
        }
    }

    fn perform_turn_based_actions(
        &mut self,
        step: StepKind,
        completion: Option<&mut DeclarationCollector>,
    ) -> Result<Option<ActionSpace>, AgentError> {
        Ok(match step {
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
                    )
                    .collect();
                let mut frame = crate::flow::decision::EffectFrame::new(active, Vec::new());
                frame.batch = Some(self.event_batch(events));
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
                let combat = self.state.combat.get_or_insert_with(CombatState::default);
                let Some(attacker) = combat.attackers_to_declare.pop() else {
                    // CR 508.1 — Attack triggers fire once the whole batch of
                    // attackers has been declared.
                    let attackers = combat.attackers.clone();
                    let fact = DeclarationCompletion::from_game(self, DeclarationStage::Attackers)?;
                    if let Some(collector) = completion {
                        collector.observe(fact)?;
                    }
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
                    return Ok(None);
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
                    crate::flow::block_declaration::BlockProblem::from_game(self)?.validate(
                        self,
                        None,
                        &[],
                    )?;
                    let fact = DeclarationCompletion::from_game(self, DeclarationStage::Blockers)?;
                    if let Some(collector) = completion {
                        collector.observe(fact)?;
                    }
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
                    return Ok(None);
                };
                let actions = self.block_declaration_actions(blocker)?;

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
                    || self.battlefield_permanents(player).iter().any(|id| {
                        self.characteristics(*id)
                            .static_abilities
                            .contains(&StaticAbility::NoMaximumHandSize)
                    });
                if !unlimited && self.state.zones.size(ZoneType::Hand, player) > 7 {
                    self.suspend_rule_decision(crate::flow::decision::Decision::Discard {
                        player,
                        remaining: 1,
                    });
                    return Ok(self.suspended_decision_action_space());
                }
                // CR 514.2 — Damage marked on permanents is removed during cleanup.
                self.clear_damage();
                self.end_turn_modifiers();
                None
            }
            _ => None,
        })
    }

    fn tick_priority(&mut self) -> Option<ActionSpace> {
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
                stabilized = true;
            }
            if self.is_game_over() {
                return None;
            }

            if self.state.priority.consecutive_passes >= self.state.players.len() {
                self.restart_priority_round();
                if !self.stack_is_empty() {
                    // CR 117.4, 405.2 — If all players pass with a nonempty stack, resolve top object.
                    self.resolve_top_of_stack();
                    if !self.state.turn.step_initialized {
                        // The resolution ended the turn (CR 723.1d): the
                        // cleanup step begins without priority.
                        return None;
                    }
                    self.restart_priority_round();
                    stabilized = false;
                    continue;
                }
                return None;
            }

            let player = self.state.priority.holder;
            let actions = self.compute_player_actions(player);
            if self.skip_trivial && actions.len() == 1 {
                let next = self.next_player(player);
                self.state.priority_mut().on_pass(next);
                continue;
            }

            return Some(ActionSpace {
                player: Some(player),
                kind: ActionSpaceKind::Priority,
                actions,
                focus: Vec::new(),
            });
        }
    }
}
