// triggers.rs
// Triggered ability processing: event handling, trigger matching, and target selection.

use crate::{
    agent::action::{Action, ActionSpace, ActionSpaceKind, AgentError},
    flow::{event::GameEvent, game::Game, trigger::PendingTrigger, turn::StepKind},
    state::{
        ability::{Ability, TargetSpec, TriggerCondition, TriggerSubject},
        game_object::{CardId, PermanentId, PlayerId, Target},
        predicate::CardPredicate,
        stack_object::{StackObject, TriggeredAbilityOnStack},
        target::Target as ActionTarget,
        zone::ZoneType,
    },
};

impl Game {
    pub(crate) fn choose_trigger_target(
        &mut self,
        player: PlayerId,
        target: Target,
    ) -> Result<(), AgentError> {
        let pending_trigger = self
            .state
            .pending_trigger_choice
            .take()
            .ok_or_else(|| AgentError("no pending target choice".to_string()))?;

        if pending_trigger.controller != player {
            return Err(AgentError("wrong player for target selection".to_string()));
        }

        let action_target = match target {
            Target::Player(p) => ActionTarget::Player(p),
            Target::Permanent(p) => ActionTarget::Permanent(p),
            Target::StackSpell(_) => {
                return Err(AgentError(
                    "invalid target for triggered ability".to_string(),
                ));
            }
        };

        let Some(Some(target_spec)) = self.trigger_target_spec(&pending_trigger) else {
            return Err(AgentError("triggered ability no longer exists".to_string()));
        };
        if !self.is_valid_target_for_spec(action_target, target_spec) {
            return Err(AgentError("selected target is not legal".to_string()));
        }

        self.place_triggered_ability_on_stack(pending_trigger, Some(target));
        Ok(())
    }

    pub(crate) fn flush_triggers(&mut self) -> Option<ActionSpace> {
        while let Some(trigger) = self
            .state
            .pending_trigger_choice
            .take()
            .or_else(|| self.pop_next_pending_trigger())
        {
            match self.trigger_target_spec(&trigger) {
                // The ability no longer exists.
                None => continue,
                // No target required — the trigger goes straight on the stack.
                Some(None) => {
                    self.place_triggered_ability_on_stack(trigger, None);
                    continue;
                }
                Some(Some(target_spec)) => {
                    let legal_targets = self.legal_targets_for_spec(target_spec);
                    if legal_targets.is_empty() {
                        // CR 603.3d — Triggered abilities with no legal required targets are removed.
                        continue;
                    }

                    let controller = trigger.controller;
                    self.state.pending_trigger_choice = Some(trigger);
                    return Some(ActionSpace {
                        player: Some(controller),
                        kind: ActionSpaceKind::ChooseTarget,
                        actions: legal_targets
                            .into_iter()
                            .map(|legal_target| Action::ChooseTarget {
                                player: controller,
                                target: legal_target,
                            })
                            .collect(),
                        focus: Vec::new(),
                    });
                }
            }
        }

        None
    }

    /// Outer `None`: the ability doesn't exist. Inner option: its target spec.
    fn trigger_target_spec<'a>(
        &'a self,
        trigger: &PendingTrigger,
    ) -> Option<Option<&'a TargetSpec>> {
        let ability = self
            .state
            .cards
            .get(trigger.source_card.0)
            .and_then(|card| card.abilities.get(trigger.ability_index))?;
        Some(ability.target_spec())
    }

    fn pop_next_pending_trigger(&mut self) -> Option<PendingTrigger> {
        let active = self.active_player();
        let next_index = self
            .state
            .pending_triggers
            .iter()
            .enumerate()
            .min_by_key(|(_, trigger)| {
                let apnap_rank = if trigger.controller == active {
                    0_u8
                } else {
                    1_u8
                };
                (apnap_rank, trigger.enqueue_order)
            })
            .map(|(index, _)| index)?;
        Some(self.state.pending_triggers.remove(next_index))
    }

    fn place_triggered_ability_on_stack(
        &mut self,
        trigger: PendingTrigger,
        target: Option<Target>,
    ) {
        let source_card_registry_key = self.state.cards[trigger.source_card].registry_key;
        let id = self.state.id_gen.next_id();
        let targets = target.into_iter().collect();
        self.push_to_stack(StackObject::TriggeredAbility(TriggeredAbilityOnStack {
            id,
            controller: trigger.controller,
            source_card: trigger.source_card,
            source_card_registry_key,
            ability_index: trigger.ability_index,
            targets,
        }));
        self.emit(GameEvent::AbilityTriggered {
            source_card: trigger.source_card,
            controller: trigger.controller,
        });
        self.state.priority.start_round(self.active_player());
    }

    pub(crate) fn legal_targets_for_spec(&self, target_spec: &TargetSpec) -> Vec<ActionTarget> {
        match target_spec {
            TargetSpec::Creature | TargetSpec::CreatureOrPlayer => {
                let mut out = Vec::new();
                if matches!(target_spec, TargetSpec::CreatureOrPlayer) {
                    out.push(ActionTarget::Player(PlayerId(0)));
                    out.push(ActionTarget::Player(PlayerId(1)));
                }
                for player in [PlayerId(0), PlayerId(1)] {
                    for card_id in self.state.zones.zone_cards(ZoneType::Battlefield, player) {
                        let Some(permanent_id) = self.state.card_to_permanent[card_id] else {
                            continue;
                        };
                        if self.state.permanents[permanent_id].is_none() {
                            continue;
                        }
                        if self.state.cards[card_id].types.is_creature() {
                            out.push(ActionTarget::Permanent(permanent_id));
                        }
                    }
                }
                out
            }
            TargetSpec::Spell => Vec::new(),
        }
    }

    pub(crate) fn is_valid_target_for_spec(
        &self,
        target: ActionTarget,
        target_spec: &TargetSpec,
    ) -> bool {
        match (target, target_spec) {
            (ActionTarget::Player(_), TargetSpec::CreatureOrPlayer) => true,
            (
                ActionTarget::Permanent(permanent_id),
                TargetSpec::Creature | TargetSpec::CreatureOrPlayer,
            ) => {
                let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
                    return false;
                };
                let card = &self.state.cards[permanent.card];
                card.types.is_creature()
                    && self.state.zones.zone_of(permanent.card) == Some(ZoneType::Battlefield)
            }
            _ => false,
        }
    }

    /// Match pending game events against every triggered ability in play and
    /// enqueue the ones that fire. Matching only enqueues triggers (it never
    /// emits events), but effects that run between calls may, so drain in a
    /// loop.
    pub(crate) fn process_game_events(&mut self) {
        while !self.state.pending_events.is_empty() {
            let events = std::mem::take(&mut self.state.pending_events);
            for event in events {
                self.check_triggers_for_event(&event);
            }
        }
    }

    fn check_triggers_for_event(&mut self, event: &GameEvent) {
        let mut fired: Vec<(CardId, usize, PlayerId)> = Vec::new();
        for (source_card, source_controller) in self.trigger_ability_sources(event) {
            for (ability_index, ability) in
                self.state.cards[source_card].abilities.iter().enumerate()
            {
                let Ability::Triggered { condition, .. } = ability;
                if self.trigger_condition_fires(condition, source_card, source_controller, event) {
                    fired.push((source_card, ability_index, source_controller));
                }
            }
        }

        for (source_card, ability_index, controller) in fired {
            self.state.pending_triggers.push(PendingTrigger {
                source_card,
                ability_index,
                controller,
                enqueue_order: self.state.trigger_enqueue_counter,
            });
            self.state.trigger_enqueue_counter =
                self.state.trigger_enqueue_counter.saturating_add(1);
        }
    }

    /// Cards whose triggered abilities can see this event: every permanent on
    /// the battlefield, plus — for zone changes — the moved card itself, so
    /// that leave-the-battlefield abilities look back in time (CR 603.6c).
    fn trigger_ability_sources(&self, event: &GameEvent) -> Vec<(CardId, PlayerId)> {
        let mut sources = Vec::new();
        for player in [PlayerId(0), PlayerId(1)] {
            for card_id in self.state.zones.zone_cards(ZoneType::Battlefield, player) {
                let Some(permanent_id) = self.state.card_to_permanent[card_id] else {
                    continue;
                };
                let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
                    continue;
                };
                if !self.state.cards[card_id].abilities.is_empty() {
                    sources.push((*card_id, permanent.controller));
                }
            }
        }

        if let GameEvent::CardMoved {
            card, controller, ..
        } = event
        {
            let on_battlefield = self.state.zones.zone_of(*card) == Some(ZoneType::Battlefield);
            if !on_battlefield && !self.state.cards[*card].abilities.is_empty() {
                sources.push((*card, *controller));
            }
        }

        sources
    }

    fn trigger_condition_fires(
        &self,
        condition: &TriggerCondition,
        source_card: CardId,
        source_controller: PlayerId,
        event: &GameEvent,
    ) -> bool {
        match (condition, event) {
            (
                TriggerCondition::EntersTheBattlefield { subject },
                GameEvent::CardMoved {
                    card,
                    from,
                    to,
                    controller,
                },
            ) => {
                let entered = *to == ZoneType::Battlefield && *from != Some(ZoneType::Battlefield);
                entered
                    && self.subject_matches_moved_card(
                        subject,
                        source_card,
                        source_controller,
                        *card,
                        *controller,
                    )
            }
            (
                TriggerCondition::Dies { subject },
                GameEvent::CardMoved {
                    card,
                    from,
                    to,
                    controller,
                },
            ) => {
                let died = *from == Some(ZoneType::Battlefield) && *to == ZoneType::Graveyard;
                died && self.subject_matches_moved_card(
                    subject,
                    source_card,
                    source_controller,
                    *card,
                    *controller,
                )
            }
            (
                TriggerCondition::Attacks { subject },
                GameEvent::AttackersDeclared { player, attackers },
            ) => match subject {
                TriggerSubject::This => self
                    .state
                    .card_to_permanent
                    .get(source_card.0)
                    .copied()
                    .flatten()
                    .is_some_and(|permanent_id| attackers.contains(&permanent_id)),
                TriggerSubject::AnotherYouControl(predicate) => {
                    *player == source_controller
                        && attackers.iter().any(|attacker| {
                            self.permanent_is_other(*attacker, source_card)
                                && self.permanent_matches_predicate(*attacker, predicate)
                        })
                }
                TriggerSubject::AnyYouControl(predicate) => {
                    *player == source_controller
                        && attackers
                            .iter()
                            .any(|attacker| self.permanent_matches_predicate(*attacker, predicate))
                }
            },
            (
                TriggerCondition::BecomesTapped { subject },
                GameEvent::PermanentTapped { permanent, .. },
            ) => {
                self.subject_matches_permanent(subject, source_card, source_controller, *permanent)
            }
            (
                TriggerCondition::TappedForMana { subject },
                GameEvent::PermanentTapped {
                    permanent,
                    for_mana: true,
                },
            ) => {
                self.subject_matches_permanent(subject, source_card, source_controller, *permanent)
            }
            (
                TriggerCondition::BeginningOfYourUpkeep,
                GameEvent::StepStarted {
                    step: StepKind::Upkeep,
                },
            ) => self.active_player() == source_controller,
            (
                TriggerCondition::YouDrawNthCardThisTurn { n },
                GameEvent::CardDrawn {
                    player,
                    nth_this_turn,
                },
            ) => *player == source_controller && nth_this_turn == n,
            _ => false,
        }
    }

    fn subject_matches_moved_card(
        &self,
        subject: &TriggerSubject,
        source_card: CardId,
        source_controller: PlayerId,
        moved_card: CardId,
        moved_controller: PlayerId,
    ) -> bool {
        match subject {
            TriggerSubject::This => moved_card == source_card,
            TriggerSubject::AnotherYouControl(predicate) => {
                moved_card != source_card
                    && moved_controller == source_controller
                    && self.source_is_on_battlefield(source_card)
                    && predicate.matches_card(&self.state.cards[moved_card])
            }
            TriggerSubject::AnyYouControl(predicate) => {
                moved_controller == source_controller
                    && self.source_is_on_battlefield(source_card)
                    && predicate.matches_card(&self.state.cards[moved_card])
            }
        }
    }

    fn subject_matches_permanent(
        &self,
        subject: &TriggerSubject,
        source_card: CardId,
        source_controller: PlayerId,
        permanent_id: PermanentId,
    ) -> bool {
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return false;
        };
        match subject {
            TriggerSubject::This => permanent.card == source_card,
            TriggerSubject::AnotherYouControl(predicate) => {
                permanent.card != source_card
                    && permanent.controller == source_controller
                    && self.permanent_matches_predicate(permanent_id, predicate)
            }
            TriggerSubject::AnyYouControl(predicate) => {
                permanent.controller == source_controller
                    && self.permanent_matches_predicate(permanent_id, predicate)
            }
        }
    }

    fn source_is_on_battlefield(&self, source_card: CardId) -> bool {
        self.state.zones.zone_of(source_card) == Some(ZoneType::Battlefield)
    }

    fn permanent_is_other(&self, permanent_id: PermanentId, source_card: CardId) -> bool {
        self.state.permanents[permanent_id]
            .as_ref()
            .is_some_and(|permanent| permanent.card != source_card)
    }

    /// Match a battlefield permanent against a predicate using its effective
    /// power (counters and until-EOT modifiers included).
    pub(crate) fn permanent_matches_predicate(
        &self,
        permanent_id: PermanentId,
        predicate: &CardPredicate,
    ) -> bool {
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return false;
        };
        let card = &self.state.cards[permanent.card];
        predicate.matches_card_with_power(card, permanent.effective_power(card))
    }
}
