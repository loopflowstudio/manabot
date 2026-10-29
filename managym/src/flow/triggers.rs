// triggers.rs
// Triggered ability processing: event handling, trigger matching, and target selection.

use crate::{
    agent::action::{Action, ActionSpace, ActionSpaceKind, AgentError},
    flow::{
        event::GameEvent,
        game::Game,
        trigger::{PendingTrigger, TriggerId, TriggerOrderingClass, TriggerPlacement},
        turn::StepKind,
    },
    state::{
        ability::{Ability, TargetSpec, TriggerCondition, TriggerSubject},
        card::CardType,
        game_object::{CardId, ObjectLki, ObjectRef, PermanentId, PlayerId, Target},
        stack_object::{StackObject, TriggeredAbilityOnStack},
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
    fn trigger_requirements(
        &self,
        trigger: &PendingTrigger,
    ) -> Option<Vec<crate::state::ability::TargetRequirement>> {
        let effects = match &trigger.inline_effects {
            Some(effects) => effects.as_slice(),
            None => self
                .state
                .cards
                .get(trigger.source_card.0)?
                .abilities
                .get(trigger.ability_index)?
                .effects(),
        };
        Some(crate::state::ability::effect_target_requirements(effects))
    }

    /// "Another target" excludes the exact source object.
    fn excluded_source(
        &self,
        trigger: &PendingTrigger,
        requirement: &crate::state::ability::TargetRequirement,
    ) -> Option<PermanentId> {
        if !requirement.another {
            return None;
        }
        let source = trigger.source_ref?;
        self.lookup_current_permanent(source).ok()
    }

    pub(crate) fn choose_trigger_target(
        &mut self,
        player: PlayerId,
        target: Target,
    ) -> Result<(), AgentError> {
        let trigger = self
            .state
            .trigger_placement
            .as_ref()
            .and_then(|p| p.placing.as_ref())
            .ok_or_else(|| AgentError("no pending target choice".into()))?;
        let requirements = self
            .trigger_requirements(trigger)
            .ok_or_else(|| AgentError("missing triggered ability".into()))?;
        let requirement = requirements
            .get(trigger.requirement_index)
            .ok_or_else(|| AgentError("no pending target requirement".into()))?;
        let duplicate = trigger
            .targets
            .iter()
            .zip(&trigger.target_req_indices)
            .any(|(chosen, index)| *index == trigger.requirement_index && *chosen == target);
        if trigger.controller != player
            || duplicate
            || !self.target_is_legal(target, &requirement.spec, player)
            || self
                .excluded_source(trigger, requirement)
                .is_some_and(|id| target == Target::Permanent(id))
        {
            return Err(AgentError("illegal trigger target".into()));
        }
        let trigger = self
            .state
            .trigger_placement
            .as_mut()
            .and_then(|p| p.placing.as_mut())
            .expect("validated trigger");
        trigger.targets.push(target);
        trigger.target_req_indices.push(trigger.requirement_index);
        if trigger
            .target_req_indices
            .iter()
            .filter(|index| **index == trigger.requirement_index)
            .count()
            >= requirement.max
        {
            trigger.requirement_index += 1;
        }
        Ok(())
    }

    pub(crate) fn flush_triggers(&mut self) -> Option<ActionSpace> {
        if self.state.trigger_placement.is_none() {
            if self.state.pending_triggers.is_empty() {
                return None;
            }
            self.state.trigger_placement = Some(TriggerPlacement {
                remaining: std::mem::take(&mut self.state.pending_triggers),
                active: self.active_player(),
                group: 0,
                placing: None,
            });
        }
        loop {
            let placement = self.state.trigger_placement.as_mut().unwrap();
            if placement.placing.is_none() {
                while placement.group < 4 {
                    let candidates: Vec<_> = placement
                        .remaining
                        .iter()
                        .filter(|t| placement.in_current_group(t))
                        .map(|t| t.id)
                        .collect();
                    if candidates.len() > 1 {
                        let player = placement.controller();
                        return Some(ActionSpace {
                            player: Some(player),
                            kind: ActionSpaceKind::OrderTriggers,
                            actions: candidates
                                .into_iter()
                                .map(|trigger_id| Action::ChooseTrigger { player, trigger_id })
                                .collect(),
                            focus: Vec::new(),
                        });
                    }
                    if let Some(id) = candidates.first() {
                        let index = placement
                            .remaining
                            .iter()
                            .position(|t| t.id == *id)
                            .unwrap();
                        placement.placing = Some(placement.remaining.remove(index));
                        break;
                    }
                    placement.group += 1;
                }
            }
            let Some(mut trigger) = placement.placing.take() else {
                self.state.trigger_placement = None;
                return None;
            };
            let Some(requirements) = self.trigger_requirements(&trigger) else {
                continue;
            };
            let mut impossible = false;
            while let Some(requirement) = requirements.get(trigger.requirement_index) {
                let count = trigger
                    .target_req_indices
                    .iter()
                    .filter(|index| **index == trigger.requirement_index)
                    .count();
                let source = self.excluded_source(&trigger, requirement);
                let legal: Vec<_> = self
                    .legal_targets_for_spec(&requirement.spec, trigger.controller)
                    .into_iter()
                    .filter(|target| source.is_none_or(|id| *target != Target::Permanent(id)))
                    .filter(|target| {
                        !trigger.targets.iter().zip(&trigger.target_req_indices).any(
                            |(chosen, index)| {
                                *index == trigger.requirement_index && chosen == target
                            },
                        )
                    })
                    .collect();
                if legal.is_empty() {
                    if count < requirement.min {
                        impossible = true;
                        break;
                    }
                    trigger.requirement_index += 1;
                    continue;
                }
                let controller = trigger.controller;
                let mut actions: Vec<_> = legal
                    .into_iter()
                    .map(|target| Action::ChooseTarget {
                        player: controller,
                        target,
                    })
                    .collect();
                if count >= requirement.min && requirement.max > requirement.min {
                    actions.push(Action::Decline { player: controller });
                }
                self.state.trigger_placement.as_mut().unwrap().placing = Some(trigger);
                return Some(ActionSpace {
                    player: Some(controller),
                    kind: ActionSpaceKind::ChooseTarget,
                    actions,
                    focus: Vec::new(),
                });
            }
            if !impossible {
                self.place_triggered_ability_on_stack(trigger);
            }
        }
    }

    pub(crate) fn decline_trigger_target(&mut self, player: PlayerId) -> Result<(), AgentError> {
        let trigger = self
            .state
            .trigger_placement
            .as_ref()
            .and_then(|p| p.placing.as_ref())
            .ok_or_else(|| AgentError("no pending trigger".into()))?;
        let requirements = self
            .trigger_requirements(trigger)
            .ok_or_else(|| AgentError("missing triggered ability".into()))?;
        let requirement = requirements
            .get(trigger.requirement_index)
            .ok_or_else(|| AgentError("no pending target requirement".into()))?;
        let count = trigger
            .target_req_indices
            .iter()
            .filter(|index| **index == trigger.requirement_index)
            .count();
        if trigger.controller != player
            || count < requirement.min
            || requirement.max == requirement.min
        {
            return Err(AgentError("trigger cannot decline target".into()));
        }
        self.state
            .trigger_placement
            .as_mut()
            .and_then(|p| p.placing.as_mut())
            .expect("validated trigger")
            .requirement_index += 1;
        Ok(())
    }

    pub(crate) fn choose_trigger(
        &mut self,
        player: PlayerId,
        id: TriggerId,
    ) -> Result<(), AgentError> {
        let placement = self
            .state
            .trigger_placement
            .as_mut()
            .ok_or_else(|| AgentError("no trigger order pending".into()))?;
        if placement.placing.is_some() || placement.controller() != player {
            return Err(AgentError("not the pending trigger controller".into()));
        }
        let index = placement
            .remaining
            .iter()
            .position(|t| t.id == id && placement.in_current_group(t))
            .ok_or_else(|| AgentError("trigger is not in the current group".into()))?;
        placement.placing = Some(placement.remaining.remove(index));
        Ok(())
    }

    fn place_triggered_ability_on_stack(&mut self, trigger: PendingTrigger) {
        let source_definition_id = trigger
            .source_lki
            .map(|lki| lki.definition_id)
            .unwrap_or(self.state.cards[trigger.source_card].definition_id);
        let id = self.state.id_gen.next_id();
        let targets = trigger.targets;
        self.push_to_stack(StackObject::TriggeredAbility(TriggeredAbilityOnStack {
            id,
            controller: trigger.controller,
            source_card: trigger.source_card,
            source_ref: trigger.source_ref,
            source_lki: trigger.source_lki,
            source_definition_id,
            ability_index: trigger.ability_index,
            targets,
            target_req_indices: trigger.target_req_indices,
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
        let mut candidates: Vec<Target> = match target_spec {
            TargetSpec::GraveyardCardYouOwn { .. } => self
                .state
                .zones
                .zone_cards(ZoneType::Graveyard, controller)
                .iter()
                .filter_map(|card| self.current_object_ref(*card))
                .map(Target::GraveyardCard)
                .collect(),
            TargetSpec::Player | TargetSpec::Opponent | TargetSpec::CreatureOrPlayer => {
                vec![Target::Player(PlayerId(0)), Target::Player(PlayerId(1))]
            }
            TargetSpec::Spell | TargetSpec::SpellOrPermanent { .. } => self
                .state
                .stack_objects
                .iter()
                .rev()
                .filter_map(|object| match object {
                    StackObject::Spell(spell) => Some(Target::StackSpell(spell.card)),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        };
        // Preserve offer order: players or newest spells first, then permanents
        // in player/zone order. Enumeration supplies candidates; legality below
        // owns restrictions at selection (CR 601.2c) and resolution (608.2b).
        if matches!(
            target_spec,
            TargetSpec::PermanentYouControl { .. }
                | TargetSpec::Permanent { .. }
                | TargetSpec::Creature
                | TargetSpec::CreatureOrPlayer
                | TargetSpec::CreatureYouControl
                | TargetSpec::CreatureOpponentControls
                | TargetSpec::LandYouControl
                | TargetSpec::PermanentOpponentControls { .. }
                | TargetSpec::SpellOrPermanent { .. }
        ) {
            for player in [PlayerId(0), PlayerId(1)] {
                for card_id in self.state.zones.zone_cards(ZoneType::Battlefield, player) {
                    if let Some(permanent_id) = self.state.card_to_permanent[card_id] {
                        candidates.push(Target::Permanent(permanent_id));
                    }
                }
            }
        }
        candidates.retain(|target| self.target_is_legal(*target, target_spec, controller));
        candidates
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
    fn trigger_context_for_event(&self, event: &GameEvent) -> Option<Target> {
        match event {
            GameEvent::PermanentTargeted { spell, .. } => Some(Target::StackSpell(*spell)),
            _ => None,
        }
    }

    fn check_triggers_for_event(&mut self, event: &GameEvent, sources: &[TriggerSourceSnapshot]) {
        let mut fired = Vec::new();
        for source in sources.iter().filter(|source| !source.lki.abilities_lost) {
            for (ability_index, ability) in
                self.state.cards[source.card].abilities.iter().enumerate()
            {
                let Ability::Triggered { condition, .. } = ability;
                if (!matches!(condition, TriggerCondition::OnceEachTurn { .. })
                    || !self
                        .state
                        .turn
                        .ability_triggers_this_turn
                        .contains_key(&(source.lki.object_ref, ability_index)))
                    && self.trigger_condition_fires(
                        condition,
                        source.card,
                        source.controller,
                        true,
                        event,
                    )
                {
                    fired.push((source.card, ability_index, source.controller, source.lki));
                }
            }
        }

        for (source_card, ability_index, controller, source_lki) in fired {
            if matches!(
                &self.state.cards[source_card].abilities[ability_index],
                Ability::Triggered {
                    condition: TriggerCondition::OnceEachTurn { .. },
                    ..
                }
            ) {
                self.journal_turn();
                *self
                    .state
                    .turn
                    .ability_triggers_this_turn
                    .entry((source_lki.object_ref, ability_index))
                    .or_default() += 1;
            }
            let context = if self.state.cards[source_card].abilities[ability_index]
                .effects()
                .iter()
                .any(|effect| {
                    matches!(
                        effect,
                        crate::state::ability::Effect::ModifyTriggeringPermanent { .. }
                    )
                }) {
                match event {
                    GameEvent::CardMoved {
                        card,
                        to: ZoneType::Battlefield,
                        ..
                    } => self.state.card_to_permanent[*card].map(Target::Permanent),
                    _ => None,
                }
            } else {
                self.trigger_context_for_event(event)
            };
            self.state.pending_triggers.push(PendingTrigger {
                targets: Vec::new(),
                target_req_indices: Vec::new(),
                requirement_index: 0,
                source_card,
                source_ref: Some(source_lki.object_ref),
                source_lki: Some(source_lki),
                ability_index,
                controller,
                id: TriggerId(self.state.trigger_enqueue_counter),
                ordering_class: TriggerOrderingClass::Ordinary,
                context,
                inline_effects: None,
            });
            self.state.trigger_enqueue_counter = self
                .state
                .trigger_enqueue_counter
                .checked_add(1)
                .expect("trigger occurrence identity exhausted");
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
            targets: Vec::new(),
            target_req_indices: Vec::new(),
            requirement_index: 0,
            source_card: source_lki.card,
            source_ref: Some(source),
            source_lki: Some(source_lki),
            ability_index: 0,
            controller,
            id: TriggerId(self.state.trigger_enqueue_counter),
            ordering_class: TriggerOrderingClass::Ordinary,
            context: None,
            inline_effects: Some(effects),
        });
        self.state.trigger_enqueue_counter = self
            .state
            .trigger_enqueue_counter
            .checked_add(1)
            .expect("trigger occurrence identity exhausted");
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
            (TriggerCondition::WasCast { condition }, _) => {
                self.state.card_to_permanent[source_card]
                    .and_then(|id| self.state.permanents[id].as_ref())
                    .is_some_and(|source| source.was_cast)
                    && self.trigger_condition_fires(
                        condition,
                        source_card,
                        source_controller,
                        source_was_on_battlefield,
                        event,
                    )
            }
            (TriggerCondition::OnceEachTurn { condition }, _) => self.trigger_condition_fires(
                condition,
                source_card,
                source_controller,
                source_was_on_battlefield,
                event,
            ),
            (
                TriggerCondition::Sacrificed {
                    predicate,
                    controller_only,
                },
                GameEvent::PermanentSacrificed { card, controller },
            ) => {
                (!controller_only || *controller == source_controller)
                    && self.state.card_to_permanent[*card]
                        .is_some_and(|id| self.permanent_matches_predicate(id, predicate))
            }
            (TriggerCondition::YouCastSpell { predicate }, GameEvent::SpellCast { card, .. }) => {
                self.state.cards[*card].owner == source_controller
                    && predicate.matches_card(&self.state.cards[*card])
            }
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
                TriggerCondition::BeginningOfYourCombat,
                GameEvent::StepStarted {
                    step: StepKind::BeginningOfCombat,
                },
            ) => self.active_player() == source_controller,
            (
                TriggerCondition::BeginningOfYourUpkeep,
                GameEvent::StepStarted {
                    step: StepKind::Upkeep,
                },
            )
            | (
                TriggerCondition::BeginningOfYourEndStep,
                GameEvent::StepStarted {
                    step: StepKind::End,
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
        let animated = self.state.card_to_permanent[moved_card]
            .and_then(|id| self.state.permanents[id].as_ref())
            .map(|p| p.animated_creature(&self.state.cards[moved_card]))
            .unwrap_or_else(|| {
                self.state
                    .object_lki
                    .iter()
                    .rev()
                    .find_map(|(_, lki)| (lki.card == moved_card).then_some(lki.animated_creature))
                    .unwrap_or(false)
            });
        let matches = |predicate: &crate::state::predicate::CardPredicate| {
            predicate.matches(
                &self.state.cards[moved_card],
                self.state.cards[moved_card].power.unwrap_or(0),
                if animated { &[CardType::Creature] } else { &[] },
            )
        };
        match subject {
            TriggerSubject::This => moved_card == source_card,
            TriggerSubject::AnotherYouControl(predicate) => {
                moved_card != source_card
                    && moved_controller == source_controller
                    && source_was_on_battlefield
                    && matches(predicate)
            }
            TriggerSubject::AnyYouControl(predicate) => {
                moved_controller == source_controller
                    && source_was_on_battlefield
                    && matches(predicate)
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

#[cfg(test)]
#[path = "target_authority_tests.rs"]
mod target_authority_tests;
