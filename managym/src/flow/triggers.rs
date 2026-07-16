// triggers.rs
// Triggered ability processing: event handling, trigger matching, and target selection.

use crate::{
    agent::action::{Action, ActionSpace, ActionSpaceKind, AgentError},
    flow::{event::GameEvent, game::Game, trigger::PendingTrigger, turn::StepKind},
    state::{
        ability::{Ability, TargetSpec, TriggerCondition, TriggerSubject},
        game_object::{CardId, ObjectLki, ObjectRef, PermanentId, PlayerId, Target},
        stack_object::{StackObject, TriggeredAbilityOnStack},
        target::Target as ActionTarget,
        zone::ZoneType,
    },
};

/// Exact battlefield source facts used while matching one or more committed
/// events. SBA batches reuse one sorted snapshot so simultaneous departures
/// see the same pre-batch abilities (CR 603.6c, 704.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TriggerSourceSnapshot {
    card: CardId,
    controller: PlayerId,
    lki: ObjectLki,
}

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

        let Some(Some(target_spec)) = self.trigger_target_spec(&pending_trigger) else {
            return Err(AgentError("triggered ability no longer exists".to_string()));
        };
        let target_spec = target_spec.clone();
        if !self.target_is_legal(target, &target_spec, pending_trigger.controller) {
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
                    let target_spec = target_spec.clone();
                    let optional = self.trigger_target_optional(&trigger);
                    let legal_targets =
                        self.legal_targets_for_spec(&target_spec, trigger.controller);
                    if legal_targets.is_empty() {
                        if optional {
                            // "Up to one target" with none available: the
                            // trigger still goes on the stack (and does
                            // nothing on resolution).
                            self.place_triggered_ability_on_stack(trigger, None);
                        }
                        // CR 603.3d — Triggered abilities with no legal required targets are removed.
                        continue;
                    }

                    let controller = trigger.controller;
                    self.state.pending_trigger_choice = Some(trigger);
                    let mut actions: Vec<Action> = legal_targets
                        .into_iter()
                        .map(|legal_target| Action::ChooseTarget {
                            player: controller,
                            target: match legal_target {
                                Target::Player(p) => ActionTarget::Player(p),
                                Target::Permanent(p) => ActionTarget::Permanent(p),
                                Target::StackSpell(c) => ActionTarget::StackSpell(c),
                            },
                        })
                        .collect();
                    if optional {
                        actions.push(Action::Decline { player: controller });
                    }
                    return Some(ActionSpace {
                        player: Some(controller),
                        kind: ActionSpaceKind::ChooseTarget,
                        actions,
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
        trigger: &'a PendingTrigger,
    ) -> Option<Option<&'a TargetSpec>> {
        if let Some(effects) = &trigger.inline_effects {
            // Delayed triggers carry their (targetless) effects inline.
            return Some(effects.iter().find_map(|effect| effect.target_spec()));
        }
        let ability = self
            .state
            .cards
            .get(trigger.source_card.0)
            .and_then(|card| card.abilities.get(trigger.ability_index))?;
        Some(ability.target_spec())
    }

    /// Whether the pending trigger's target is "up to one" (a Decline
    /// action is offered alongside the legal targets).
    fn trigger_target_optional(&self, trigger: &PendingTrigger) -> bool {
        if trigger.inline_effects.is_some() {
            return false;
        }
        self.state
            .cards
            .get(trigger.source_card.0)
            .and_then(|card| card.abilities.get(trigger.ability_index))
            .is_some_and(|ability| ability.target_optional())
    }

    /// Decline choosing a target for the pending "up to one target"
    /// trigger — it goes on the stack with no targets.
    pub(crate) fn decline_trigger_target(&mut self, player: PlayerId) -> Result<(), AgentError> {
        let Some(pending_trigger) = self.state.pending_trigger_choice.take() else {
            return Err(AgentError("no pending target choice".to_string()));
        };
        if pending_trigger.controller != player {
            self.state.pending_trigger_choice = Some(pending_trigger);
            return Err(AgentError("wrong player for target selection".to_string()));
        }
        if !self.trigger_target_optional(&pending_trigger) {
            self.state.pending_trigger_choice = Some(pending_trigger);
            return Err(AgentError("trigger target is not optional".to_string()));
        }
        self.place_triggered_ability_on_stack(pending_trigger, None);
        Ok(())
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
        let source_definition_id = trigger
            .source_lki
            .map(|lki| lki.definition_id)
            .unwrap_or(self.state.cards[trigger.source_card].definition_id);
        let id = self.state.id_gen.next_id();
        let targets = target.into_iter().collect();
        self.push_to_stack(StackObject::TriggeredAbility(TriggeredAbilityOnStack {
            id,
            controller: trigger.controller,
            source_card: trigger.source_card,
            source_ref: trigger.source_ref,
            source_lki: trigger.source_lki,
            source_definition_id,
            ability_index: trigger.ability_index,
            targets,
            context: trigger.context,
            inline_effects: trigger.inline_effects,
        }));
        self.emit(GameEvent::AbilityTriggered {
            source_card: trigger.source_card,
            controller: trigger.controller,
        });
        self.state.priority.start_round(self.active_player());
    }

    /// All currently legal targets for a spec, from `controller`'s
    /// perspective ("you").
    pub(crate) fn legal_targets_for_spec(
        &self,
        target_spec: &TargetSpec,
        controller: PlayerId,
    ) -> Vec<Target> {
        match target_spec {
            TargetSpec::Creature
            | TargetSpec::CreatureOrPlayer
            | TargetSpec::CreatureYouControl
            | TargetSpec::CreatureOpponentControls
            | TargetSpec::LandYouControl
            | TargetSpec::PermanentOpponentControls { .. } => {
                let mut out = Vec::new();
                if matches!(target_spec, TargetSpec::CreatureOrPlayer) {
                    out.push(Target::Player(PlayerId(0)));
                    out.push(Target::Player(PlayerId(1)));
                }
                for player in [PlayerId(0), PlayerId(1)] {
                    for card_id in self.state.zones.zone_cards(ZoneType::Battlefield, player) {
                        let Some(permanent_id) = self.state.card_to_permanent[card_id] else {
                            continue;
                        };
                        let target = Target::Permanent(permanent_id);
                        if self.target_is_legal(target, target_spec, controller) {
                            out.push(target);
                        }
                    }
                }
                out
            }
            TargetSpec::Spell => self.stack_spell_targets(0),
            TargetSpec::SpellOrPermanent { min_mana_value } => {
                let mut out = self.stack_spell_targets(*min_mana_value);
                for player in [PlayerId(0), PlayerId(1)] {
                    for card_id in self.state.zones.zone_cards(ZoneType::Battlefield, player) {
                        let Some(permanent_id) = self.state.card_to_permanent[card_id] else {
                            continue;
                        };
                        let target = Target::Permanent(permanent_id);
                        if self.target_is_legal(target, target_spec, controller) {
                            out.push(target);
                        }
                    }
                }
                out
            }
        }
    }

    fn stack_spell_targets(&self, min_mana_value: u8) -> Vec<Target> {
        self.state
            .stack_objects
            .iter()
            .rev()
            .filter_map(|object| match object {
                StackObject::Spell(spell) if self.card_mana_value(spell.card) >= min_mana_value => {
                    Some(Target::StackSpell(spell.card))
                }
                _ => None,
            })
            .collect()
    }

    /// Match pending committed events against the applicable exact sources.
    pub(crate) fn process_game_events(&mut self) {
        while !self.state.pending_events.is_empty() {
            let events = std::mem::take(&mut self.state.pending_events);
            for event in events {
                let sources = self.trigger_ability_sources(&event);
                self.check_triggers_for_event(&event, &sources);
            }
        }
    }

    /// Process a simultaneous committed batch using the same pre-batch source
    /// facts for every event. Matching only enqueues triggers, so the snapshot
    /// remains immutable throughout collection.
    pub(crate) fn process_game_events_with_sources(&mut self, sources: &[TriggerSourceSnapshot]) {
        while !self.state.pending_events.is_empty() {
            let events = std::mem::take(&mut self.state.pending_events);
            for event in events {
                self.check_triggers_for_event(&event, sources);
            }
        }
    }

    /// The object from an event that a fired ability's effects may
    /// reference (e.g. ward counters the spell that targeted its permanent).
    fn trigger_context_for_event(event: &GameEvent) -> Option<Target> {
        match event {
            GameEvent::PermanentTargeted { spell, .. } => Some(Target::StackSpell(*spell)),
            _ => None,
        }
    }

    fn check_triggers_for_event(&mut self, event: &GameEvent, sources: &[TriggerSourceSnapshot]) {
        let mut fired = Vec::new();
        for source in sources {
            for (ability_index, ability) in
                self.state.cards[source.card].abilities.iter().enumerate()
            {
                let Ability::Triggered { condition, .. } = ability;
                if self.trigger_condition_fires(
                    condition,
                    source.card,
                    source.controller,
                    true,
                    event,
                ) {
                    fired.push((source.card, ability_index, source.controller, source.lki));
                }
            }
        }

        let context = Self::trigger_context_for_event(event);
        for (source_card, ability_index, controller, source_lki) in fired {
            self.state.pending_triggers.push(PendingTrigger {
                source_card,
                source_ref: Some(source_lki.object_ref),
                source_lki: Some(source_lki),
                ability_index,
                controller,
                enqueue_order: self.state.trigger_enqueue_counter,
                context,
                inline_effects: None,
            });
            self.state.trigger_enqueue_counter =
                self.state.trigger_enqueue_counter.saturating_add(1);
        }
    }

    /// Enqueue a fired delayed trigger with inline effects (no
    /// `Card::abilities` entry to reference).
    pub(crate) fn enqueue_delayed_trigger(
        &mut self,
        source: ObjectRef,
        source_lki: ObjectLki,
        controller: PlayerId,
        effects: Vec<crate::state::ability::Effect>,
    ) {
        self.state.pending_triggers.push(PendingTrigger {
            source_card: source_lki.card,
            source_ref: Some(source),
            source_lki: Some(source_lki),
            ability_index: 0,
            controller,
            enqueue_order: self.state.trigger_enqueue_counter,
            context: None,
            inline_effects: Some(effects),
        });
        self.state.trigger_enqueue_counter = self.state.trigger_enqueue_counter.saturating_add(1);
    }

    /// Cards whose triggered abilities can see this event: every permanent on
    /// the battlefield, plus — for zone changes — the moved card itself, so
    /// that leave-the-battlefield abilities look back in time (CR 603.6c).
    pub(crate) fn snapshot_trigger_sources(&self) -> Vec<TriggerSourceSnapshot> {
        let mut sources: Vec<_> = self
            .state
            .permanents
            .iter()
            .flatten()
            .filter(|permanent| !self.state.cards[permanent.card].abilities.is_empty())
            .filter_map(|permanent| {
                self.snapshot_current_permanent(permanent.card)
                    .map(|lki| TriggerSourceSnapshot {
                        card: permanent.card,
                        controller: permanent.controller,
                        lki,
                    })
            })
            .collect();
        sources.sort_by_key(|source| source.lki.object_ref);
        sources
    }

    fn trigger_ability_sources(&self, event: &GameEvent) -> Vec<TriggerSourceSnapshot> {
        let mut sources = self.snapshot_trigger_sources();

        if let GameEvent::CardMoved {
            card, controller, ..
        } = event
        {
            let on_battlefield = self.state.zones.zone_of(*card) == Some(ZoneType::Battlefield);
            if !on_battlefield && !self.state.cards[*card].abilities.is_empty() {
                let source_lki = self
                    .state
                    .object_lki
                    .iter()
                    .rev()
                    .find_map(|(object_ref, lki)| (object_ref.entity.0 == card.0).then_some(*lki));
                if let Some(lki) = source_lki {
                    sources.push(TriggerSourceSnapshot {
                        card: *card,
                        controller: *controller,
                        lki,
                    });
                }
            }
        }

        sources.sort_by_key(|source| source.lki.object_ref);
        sources.dedup_by_key(|source| source.lki.object_ref);
        sources
    }

    fn trigger_condition_fires(
        &self,
        condition: &TriggerCondition,
        source_card: CardId,
        source_controller: PlayerId,
        source_was_on_battlefield: bool,
        event: &GameEvent,
    ) -> bool {
        match (condition, event) {
            // Conditionally-granted abilities: the ability only exists (and
            // so only triggers) while the condition holds for its
            // controller (CR 603.4 fire-time check).
            (
                TriggerCondition::ActiveIf {
                    active_if,
                    condition,
                },
                _,
            ) => {
                self.check_static_condition(active_if, source_controller)
                    && self.trigger_condition_fires(
                        condition,
                        source_card,
                        source_controller,
                        source_was_on_battlefield,
                        event,
                    )
            }
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
                        source_was_on_battlefield,
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
                    source_was_on_battlefield,
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
                TriggerCondition::BecomesTargeted { subject },
                GameEvent::PermanentTargeted {
                    permanent,
                    spell_controller,
                    ..
                },
            ) => {
                // Ward triggers only for spells opponents control (CR 702.21a).
                *spell_controller != source_controller
                    && self.subject_matches_permanent(
                        subject,
                        source_card,
                        source_controller,
                        *permanent,
                    )
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
        source_was_on_battlefield: bool,
        moved_card: CardId,
        moved_controller: PlayerId,
    ) -> bool {
        match subject {
            TriggerSubject::This => moved_card == source_card,
            TriggerSubject::AnotherYouControl(predicate) => {
                moved_card != source_card
                    && moved_controller == source_controller
                    && source_was_on_battlefield
                    && predicate.matches_card(&self.state.cards[moved_card])
            }
            TriggerSubject::AnyYouControl(predicate) => {
                moved_controller == source_controller
                    && source_was_on_battlefield
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

    fn permanent_is_other(&self, permanent_id: PermanentId, source_card: CardId) -> bool {
        self.state.permanents[permanent_id]
            .as_ref()
            .is_some_and(|permanent| permanent.card != source_card)
    }
}
