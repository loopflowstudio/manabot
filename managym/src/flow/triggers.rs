// triggers.rs
// Triggered ability processing: event handling, trigger matching, and target selection.

use crate::{
    agent::action::{Action, ActionSpace, ActionSpaceKind, AgentError},
    flow::{
        event::GameEvent,
        game::Game,
        statics::ScopeView,
        trigger::{
            DelayedTrigger, DelayedTriggerEvent, PendingTrigger, TriggerId, TriggerOrderingClass,
            TriggerPlacement,
        },
        turn::StepKind,
    },
    state::{
        ability::{Ability, Effect, TargetSpec, TriggerCondition, TriggerSubject},
        card::CardType,
        effect::{EffectDuration, Modification},
        game_object::{CardId, ObjectLki, ObjectRef, PermanentId, PlayerId, Target},
        mana::{Color, Mana},
        stack_object::{StackObject, TriggeredAbilityOnStack},
        zone::ZoneType,
    },
};

/// One occurrence of a matched triggered ability, before it waits to be
/// placed: a printed ability by index, or a granted one's effects.
struct FiredTrigger {
    source: TriggerSourceSnapshot,
    ability_index: usize,
    granted: Option<Vec<Effect>>,
}

/// Exact battlefield source facts used while matching one or more committed
/// events. SBA batches reuse one sorted snapshot so simultaneous departures
/// see the same pre-batch abilities (CR 603.6c, 704.3).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TriggerSourceSnapshot {
    card: CardId,
    controller: PlayerId,
    lki: ObjectLki,
}

impl Game {
    /// Whether `target` is the exact permanent whose ability triggered.
    fn is_trigger_source(&self, trigger: &PendingTrigger, target: Target) -> bool {
        matches!(target, Target::Permanent(id)
            if trigger.source_ref.is_some() && self.permanent_object_ref(id) == trigger.source_ref)
    }

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
            || (requirement.another && self.is_trigger_source(trigger, target))
            || !self.target_is_legal(target, &requirement.spec, player)
            || !self.within_source_bound(
                target,
                &requirement.spec,
                trigger.source_card,
                trigger.source_ref,
            )
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
            // CR 700.2b: a modal trigger's mode is chosen first; with no
            // mode that has legal targets it is removed (CR 700.2a).
            if let Some(modes) = self.unchosen_trigger_modes(&trigger) {
                let legal = self.legal_trigger_modes(&trigger, &modes);
                let up_to = matches!(
                    self.trigger_mode_rule(&trigger),
                    Some(Effect::TriggerModes { up_to: Some(_), .. })
                );
                if up_to && (legal.is_empty() || self.trigger_modes_left(&trigger) == 0) {
                    // "Choose up to": the chosen modes, in printed order.
                    trigger.inline_effects = Some(self.chosen_mode_effects(&trigger, &modes));
                } else if legal.is_empty() {
                    continue;
                } else {
                    let controller = trigger.controller;
                    self.state.trigger_placement.as_mut().unwrap().placing = Some(trigger);
                    return Some(ActionSpace {
                        player: Some(controller),
                        kind: ActionSpaceKind::Modal,
                        actions: legal
                            .into_iter()
                            .map(|mode| Action::ChooseMode {
                                player: controller,
                                mode,
                            })
                            .chain(up_to.then_some(Action::Decline { player: controller }))
                            .collect(),
                        focus: Vec::new(),
                    });
                }
            }
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
                        self.within_source_bound(
                            *target,
                            &requirement.spec,
                            trigger.source_card,
                            trigger.source_ref,
                        )
                    })
                    .filter(|target| {
                        !(requirement.another && self.is_trigger_source(&trigger, *target))
                    })
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

    /// The modes of a placing trigger whose modes are not yet chosen.
    pub(crate) fn unchosen_trigger_modes(
        &self,
        trigger: &PendingTrigger,
    ) -> Option<Vec<Vec<Effect>>> {
        if trigger.inline_effects.is_some() {
            return None;
        }
        match self.trigger_mode_rule(trigger)? {
            Effect::TriggerModes { modes, .. } => Some(modes.clone()),
            _ => None,
        }
    }

    /// The placing trigger's `Effect::TriggerModes`, its only effect.
    fn trigger_mode_rule(&self, trigger: &PendingTrigger) -> Option<&Effect> {
        match self
            .state
            .cards
            .get(trigger.source_card.0)?
            .abilities
            .get(trigger.ability_index)?
            .effects()
        {
            [rule @ Effect::TriggerModes { .. }] => Some(rule),
            _ => None,
        }
    }

    /// How many modes the placing trigger may still choose: one, or "up to"
    /// its value less those chosen.
    fn trigger_modes_left(&self, trigger: &PendingTrigger) -> usize {
        match self.trigger_mode_rule(trigger) {
            Some(Effect::TriggerModes {
                up_to: Some(value), ..
            }) => usize::try_from(self.controller_value(value, trigger.controller))
                .unwrap_or(0)
                .saturating_sub(trigger.chosen_modes.len()),
            _ => 1,
        }
    }

    /// Modes the placing trigger may choose next: every required target has
    /// a legal candidate, it is not chosen already, and an "unchosen" rule's
    /// source has not chosen it before.
    fn legal_trigger_modes(&self, trigger: &PendingTrigger, modes: &[Vec<Effect>]) -> Vec<usize> {
        let unchosen = matches!(
            self.trigger_mode_rule(trigger),
            Some(Effect::TriggerModes { unchosen: true, .. })
        );
        let chosen_before = match trigger
            .source_ref
            .map(|source| self.lookup_current_permanent(source))
        {
            Some(Ok(id)) if unchosen => self.chosen_modes(id),
            _ => Vec::new(),
        };
        self.targetable_trigger_modes(trigger, modes)
            .into_iter()
            .filter(|mode| !trigger.chosen_modes.contains(mode) && !chosen_before.contains(mode))
            .collect()
    }

    /// Modes whose every required target has a legal candidate.
    fn targetable_trigger_modes(
        &self,
        trigger: &PendingTrigger,
        modes: &[Vec<Effect>],
    ) -> Vec<usize> {
        let source = trigger
            .source_ref
            .and_then(|source| self.lookup_current_permanent(source).ok());
        (0..modes.len())
            .filter(|mode| {
                crate::state::ability::effect_target_requirements(&modes[*mode])
                    .iter()
                    .filter(|requirement| requirement.min > 0)
                    .all(|requirement| {
                        self.legal_targets_for_spec(&requirement.spec, trigger.controller)
                            .into_iter()
                            .any(|target| {
                                !(requirement.another
                                    && source.is_some_and(|id| target == Target::Permanent(id)))
                            })
                    })
            })
            .collect()
    }

    /// Choose the placing modal trigger's mode; its targets follow.
    pub(crate) fn choose_trigger_mode(
        &mut self,
        player: PlayerId,
        mode: usize,
    ) -> Result<(), AgentError> {
        let trigger = self
            .state
            .trigger_placement
            .as_ref()
            .and_then(|p| p.placing.as_ref())
            .ok_or_else(|| AgentError("no pending trigger mode".into()))?;
        let modes = self
            .unchosen_trigger_modes(trigger)
            .ok_or_else(|| AgentError("no pending trigger mode".into()))?;
        if trigger.controller != player
            || !self.legal_trigger_modes(trigger, &modes).contains(&mode)
        {
            return Err(AgentError("illegal trigger mode".into()));
        }
        let rule = self.trigger_mode_rule(trigger).cloned();
        let source = trigger
            .source_ref
            .and_then(|source| self.lookup_current_permanent(source).ok());
        let trigger = self
            .state
            .trigger_placement
            .as_mut()
            .and_then(|p| p.placing.as_mut())
            .expect("validated trigger");
        match rule {
            Some(Effect::TriggerModes { up_to: Some(_), .. }) => trigger.chosen_modes.push(mode),
            Some(Effect::TriggerModes { unchosen, .. }) => {
                trigger.inline_effects = Some(modes[mode].clone());
                if let (true, Some(source)) = (unchosen, source) {
                    let affected = self.affected_permanent(source);
                    let controller = player;
                    let card = self.state.permanents[source].as_ref().map(|p| p.card);
                    self.apply_continuous(
                        card,
                        controller,
                        affected,
                        Modification::ModeChosen {
                            mode: u8::try_from(mode).unwrap_or(u8::MAX),
                        },
                        EffectDuration::Indefinite,
                    );
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// The effects of the placing trigger's chosen modes, in printed order.
    fn chosen_mode_effects(&self, trigger: &PendingTrigger, modes: &[Vec<Effect>]) -> Vec<Effect> {
        let mut chosen = trigger.chosen_modes.clone();
        chosen.sort_unstable();
        chosen
            .into_iter()
            .flat_map(|mode| modes[mode].iter().cloned())
            .collect()
    }

    /// Stop choosing an "up to" trigger's modes.
    fn stop_choosing_trigger_modes(&mut self, player: PlayerId) -> Result<(), AgentError> {
        let trigger = self
            .state
            .trigger_placement
            .as_ref()
            .and_then(|p| p.placing.as_ref())
            .ok_or_else(|| AgentError("no pending trigger".into()))?;
        let modes = self
            .unchosen_trigger_modes(trigger)
            .ok_or_else(|| AgentError("no pending trigger mode".into()))?;
        if trigger.controller != player {
            return Err(AgentError("not the trigger's controller".into()));
        }
        let effects = self.chosen_mode_effects(trigger, &modes);
        self.state
            .trigger_placement
            .as_mut()
            .and_then(|p| p.placing.as_mut())
            .expect("validated trigger")
            .inline_effects = Some(effects);
        Ok(())
    }

    pub(crate) fn decline_trigger_target(&mut self, player: PlayerId) -> Result<(), AgentError> {
        let trigger = self
            .state
            .trigger_placement
            .as_ref()
            .and_then(|p| p.placing.as_ref())
            .ok_or_else(|| AgentError("no pending trigger".into()))?;
        if matches!(
            self.trigger_mode_rule(trigger),
            Some(Effect::TriggerModes { up_to: Some(_), .. })
        ) && trigger.inline_effects.is_none()
        {
            return self.stop_choosing_trigger_modes(player);
        }
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
            .as_ref()
            .map(|lki| lki.definition_id)
            .unwrap_or(self.state.cards[trigger.source_card].definition_id);
        let id = self.state.id_gen.next_id();
        let targets = trigger.targets;
        self.emit_permanents_targeted(&targets, Target::StackAbility(id), trigger.controller);
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
            player: trigger.player,
            x: trigger.x,
            inline_effects: trigger.inline_effects,
        }));
        self.emit(GameEvent::AbilityTriggered {
            source_card: trigger.source_card,
            controller: trigger.controller,
            ability: id,
            attacker: trigger.attacker,
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
            TargetSpec::GraveyardCardYouOwn { .. }
            | TargetSpec::GraveyardCardYouOwnWithinSourcePower { .. } => self
                .state
                .zones
                .zone_cards(ZoneType::Graveyard, controller)
                .iter()
                .filter_map(|card| self.current_object_ref(*card))
                .map(Target::GraveyardCard)
                .collect(),
            TargetSpec::GraveyardCard { .. } => [PlayerId(0), PlayerId(1)]
                .into_iter()
                .flat_map(|player| self.state.zones.zone_cards(ZoneType::Graveyard, player))
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
            TargetSpec::SpellOrAbilityWithSingleTarget => self
                .state
                .stack_objects
                .iter()
                .rev()
                .map(|object| match object {
                    StackObject::Spell(spell) => Target::StackSpell(spell.card),
                    object => Target::StackAbility(object.id()),
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
            GameEvent::PermanentTargeted { by, .. } => Some(*by),
            GameEvent::AbilityTriggered {
                ability,
                attacker: Some(_),
                ..
            } => Some(Target::StackAbility(*ability)),
            _ => None,
        }
    }

    fn check_triggers_for_event(&mut self, event: &GameEvent, sources: &[TriggerSourceSnapshot]) {
        self.granted_firebending_triggers(event);
        self.graveyard_triggers(event);
        self.delayed_triggers_for_event(event);
        let mut fired: Vec<FiredTrigger> = Vec::new();
        for source in sources.iter().filter(|source| !source.lki.abilities_lost) {
            for (ability_index, ability) in
                self.state.cards[source.card].abilities.iter().enumerate()
            {
                let Ability::Triggered {
                    condition,
                    intervening_if,
                    ..
                } = ability;
                // CR 603.4: an intervening "if" is checked as the trigger
                // event occurs, and again on resolution.
                if intervening_if.as_ref().is_none_or(|condition| {
                    self.check_intervening_if(
                        condition,
                        source.controller,
                        Some(source.lki.object_ref),
                        self.triggering_permanent(event),
                    )
                }) && (!matches!(condition, TriggerCondition::OnceEachTurn { .. })
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
                    let times =
                        self.trigger_multiplicity(condition, source.card, source.controller, event)
                            + self.additional_trigger_times(source);
                    for _ in 0..times {
                        fired.push(FiredTrigger {
                            source: source.clone(),
                            ability_index,
                            granted: None,
                        });
                    }
                }
            }
            for ability in self.granted_triggered_abilities(source) {
                let Ability::Triggered {
                    condition, effects, ..
                } = &ability;
                if self.trigger_condition_fires(
                    condition,
                    source.card,
                    source.controller,
                    true,
                    event,
                ) {
                    let times =
                        self.trigger_multiplicity(condition, source.card, source.controller, event)
                            + self.additional_trigger_times(source);
                    for _ in 0..times {
                        fired.push(FiredTrigger {
                            source: source.clone(),
                            ability_index: 0,
                            granted: Some(effects.clone()),
                        });
                    }
                }
            }
        }

        for FiredTrigger {
            source,
            ability_index,
            granted,
        } in fired
        {
            let TriggerSourceSnapshot {
                card: source_card,
                controller,
                lki: source_lki,
            } = source;
            if granted.is_none()
                && matches!(
                    &self.state.cards[source_card].abilities[ability_index],
                    Ability::Triggered {
                        condition: TriggerCondition::OnceEachTurn { .. },
                        ..
                    }
                )
            {
                self.spend_once_each_turn((source_lki.object_ref, ability_index));
            }
            let effects = match &granted {
                Some(effects) => effects.as_slice(),
                None => self.state.cards[source_card].abilities[ability_index].effects(),
            };
            let reads_triggering_permanent = effects.iter().any(Effect::reads_triggering_permanent);
            let asks_triggering_spell = effects.iter().any(Effect::asks_triggering_spell);
            let context = if reads_triggering_permanent {
                self.triggering_permanent(event)
            } else if let (true, GameEvent::SpellCast { card, .. }) = (asks_triggering_spell, event)
            {
                Some(Target::StackSpell(*card))
            } else {
                self.trigger_context_for_event(event)
            };
            let player = match event {
                GameEvent::SpellCast { card, .. } => Some(self.spell_caster(*card)),
                GameEvent::LibrarySearched { player } => Some(*player),
                _ => None,
            };
            // CR 107.3m: an entering permanent's own trigger sees its spell's
            // X; a counters trigger's X is how many were put ("that much").
            let x = match event {
                GameEvent::CountersPut { count, .. } => u8::try_from(*count).unwrap_or(u8::MAX),
                GameEvent::CardMoved {
                    card,
                    to: ZoneType::Battlefield,
                    ..
                } if *card == source_card => self.state.card_to_permanent[*card]
                    .and_then(|id| self.state.permanents[id].as_ref())
                    .map_or(0, |permanent| permanent.cast_x),
                _ => 0,
            };
            let attacker = match event {
                GameEvent::AttackersDeclared { attackers, .. } => attackers
                    .iter()
                    .copied()
                    .find(|id| self.permanent_object_ref(*id) == Some(source_lki.object_ref)),
                _ => None,
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
                player,
                x,
                chosen_modes: Vec::new(),
                attacker,
                inline_effects: granted,
            });
            self.state.trigger_enqueue_counter = self
                .state
                .trigger_enqueue_counter
                .checked_add(1)
                .expect("trigger occurrence identity exhausted");
        }
    }

    /// The permanent whose event fired a trigger: an entering one, a
    /// departed one (by the slot it occupied, read from its last known
    /// information) or a lone attacker.
    fn triggering_permanent(&self, event: &GameEvent) -> Option<Target> {
        match event {
            GameEvent::CardMoved {
                card,
                to: ZoneType::Battlefield,
                ..
            } => self.state.card_to_permanent[*card].map(Target::Permanent),
            GameEvent::CardMoved {
                card,
                from: Some(ZoneType::Battlefield),
                ..
            } => self
                .state
                .object_lki
                .values()
                .filter(|lki| lki.card == *card)
                .max_by_key(|lki| lki.object_ref)
                .map(|lki| Target::Permanent(lki.permanent)),
            GameEvent::AttackersDeclared { attackers, .. } if attackers.len() == 1 => {
                Some(Target::Permanent(attackers[0]))
            }
            _ => None,
        }
    }

    /// Triggered abilities static grants give the battlefield source
    /// (CR 613.1f), in battlefield order of the granting permanents.
    fn granted_triggered_abilities(&self, source: &TriggerSourceSnapshot) -> Vec<Ability> {
        let Ok(target) = self.lookup_current_permanent(source.lki.object_ref) else {
            return Vec::new();
        };
        self.static_sources()
            .flat_map(|(grantor, controller, card)| {
                card.triggered_grants()
                    .filter(move |grant| {
                        self.static_scope_applies(
                            &grant.scope,
                            grantor,
                            controller,
                            target,
                            ScopeView::Rules,
                        )
                    })
                    .map(|grant| (*grant.ability).clone())
            })
            .collect()
    }

    /// Extra occurrences of one of `source`'s triggered abilities: one for
    /// each permanent its controller controls whose "triggers an additional
    /// time" covers the source (CR 603.2d), as the source is or last was.
    fn additional_trigger_times(&self, source: &TriggerSourceSnapshot) -> usize {
        let current = self.lookup_current_permanent(source.lki.object_ref).ok();
        let card = &self.state.cards[source.card];
        self.static_sources()
            .filter(|(_, controller, _)| *controller == source.controller)
            .map(|(_, _, grantor)| {
                grantor
                    .additional_trigger_predicates()
                    .filter(|predicate| match current {
                        Some(id) => self.permanent_matches_predicate(id, predicate),
                        None => predicate.matches_card_with_power(card, source.lki.power),
                    })
                    .count()
            })
            .sum()
    }

    /// Granted firebending (CR 702.187): each attacking creature with
    /// firebending until end of turn triggers "add that much {R} until end
    /// of combat".
    fn granted_firebending_triggers(&mut self, event: &GameEvent) {
        let GameEvent::AttackersDeclared { attackers, .. } = event else {
            return;
        };
        for attacker in attackers.clone() {
            let amount = self.granted_firebending(attacker);
            let Some(permanent) = self.state.permanents[attacker].as_ref() else {
                continue;
            };
            let (controller, card) = (permanent.controller, permanent.card);
            if amount == 0 {
                continue;
            }
            let Some(source) = self.permanent_object_ref(attacker) else {
                continue;
            };
            let mut mana = Mana::default();
            mana.mana[Color::Red as usize] = amount;
            let effect = Effect::AddMana {
                mana,
                until_end_of_combat: true,
            };
            self.enqueue_inline_trigger(card, Some(source), controller, vec![effect], None);
            if let Some(trigger) = self.state.pending_triggers.last_mut() {
                trigger.attacker = Some(attacker);
            }
        }
    }

    /// Wait for a delayed triggered ability's event (CR 603.7).
    pub(crate) fn add_delayed_trigger(&mut self, trigger: DelayedTrigger) {
        self.journal_delayed_triggers();
        self.state.delayed_triggers.push(trigger);
    }

    /// The delayed triggers `event` fires: the dying-creature, end-step and
    /// cast events. (`Leaves` fires as its object leaves the battlefield.)
    /// One-shot triggers are removed as they fire (CR 603.7c).
    fn delayed_triggers_for_event(&mut self, event: &GameEvent) {
        let creature_died = match event {
            GameEvent::CardMoved {
                card,
                from: Some(ZoneType::Battlefield),
                to: ZoneType::Graveyard,
                ..
            } => {
                self.state.cards[*card].types.is_creature()
                    || self
                        .state
                        .object_lki
                        .values()
                        .filter(|lki| lki.card == *card)
                        .max_by_key(|lki| lki.object_ref)
                        .is_some_and(|lki| lki.animated_creature)
            }
            _ => false,
        };
        let mut fired = Vec::new();
        let mut spent = Vec::new();
        for (index, trigger) in self.state.delayed_triggers.iter().enumerate() {
            let context = match (&trigger.event, event) {
                (DelayedTriggerEvent::CreatureDies, _) if creature_died => None,
                (
                    DelayedTriggerEvent::NextEndStep { context },
                    GameEvent::StepStarted {
                        step: StepKind::End,
                    },
                ) => {
                    spent.push(index);
                    *context
                }
                (
                    DelayedTriggerEvent::YouCastSpell(predicate),
                    GameEvent::SpellCast { card, .. },
                ) if self.spell_caster(*card) == trigger.controller
                    && predicate.matches_card(&self.state.cards[*card]) =>
                {
                    spent.push(index);
                    Some(Target::StackSpell(*card))
                }
                _ => continue,
            };
            fired.push((trigger.clone(), context));
        }
        if !spent.is_empty() {
            self.journal_delayed_triggers();
            for index in spent.into_iter().rev() {
                self.state.delayed_triggers.remove(index);
            }
        }
        for (trigger, context) in fired {
            self.fire_delayed_trigger(&trigger, context);
        }
    }

    /// Put a delayed trigger's occurrence in the waiting list.
    pub(crate) fn fire_delayed_trigger(
        &mut self,
        trigger: &DelayedTrigger,
        context: Option<Target>,
    ) {
        self.enqueue_inline_trigger(
            trigger.source_card,
            trigger.source_ref,
            trigger.controller,
            trigger.effects.clone(),
            context,
        );
    }

    /// Put an occurrence with inline effects (no `Card::abilities` entry to
    /// reference: a delayed, reflexive or granted-firebending ability) in the
    /// waiting list, with its source's last known information when that was
    /// a battlefield object.
    pub(crate) fn enqueue_inline_trigger(
        &mut self,
        source_card: CardId,
        source_ref: Option<ObjectRef>,
        controller: PlayerId,
        effects: Vec<Effect>,
        context: Option<Target>,
    ) {
        let source_lki = source_ref.and_then(|source| {
            self.object_lki(source).cloned().or_else(|| {
                (self.current_object_ref(source_card) == Some(source))
                    .then(|| self.snapshot_current_permanent(source_card))
                    .flatten()
            })
        });
        self.state.pending_triggers.push(PendingTrigger {
            targets: Vec::new(),
            target_req_indices: Vec::new(),
            requirement_index: 0,
            source_card,
            source_ref,
            source_lki,
            ability_index: 0,
            controller,
            id: TriggerId(self.state.trigger_enqueue_counter),
            ordering_class: TriggerOrderingClass::Ordinary,
            context,
            player: None,
            x: 0,
            chosen_modes: Vec::new(),
            attacker: None,
            inline_effects: Some(effects),
        });
        self.state.trigger_enqueue_counter = self
            .state
            .trigger_enqueue_counter
            .checked_add(1)
            .expect("trigger occurrence identity exhausted");
    }

    /// Abilities that function from their owner's graveyard (CR 113.6k):
    /// the exact graveyard object is the source, and its owner controls it.
    fn graveyard_triggers(&mut self, event: &GameEvent) {
        let mut fired = Vec::new();
        for owner in [PlayerId(0), PlayerId(1)] {
            for &card in self.state.zones.zone_cards(ZoneType::Graveyard, owner) {
                for (ability_index, ability) in self.state.cards[card].abilities.iter().enumerate()
                {
                    let Ability::Triggered {
                        condition: TriggerCondition::InGraveyard { condition },
                        ..
                    } = ability
                    else {
                        continue;
                    };
                    if self.trigger_condition_fires(condition, card, owner, false, event) {
                        fired.push((card, ability_index, owner));
                    }
                }
            }
        }
        for (card, ability_index, owner) in fired {
            self.state.pending_triggers.push(PendingTrigger {
                targets: Vec::new(),
                target_req_indices: Vec::new(),
                requirement_index: 0,
                source_card: card,
                source_ref: self.current_object_ref(card),
                source_lki: None,
                ability_index,
                controller: owner,
                id: TriggerId(self.state.trigger_enqueue_counter),
                ordering_class: TriggerOrderingClass::Ordinary,
                context: None,
                player: None,
                x: 0,
                chosen_modes: Vec::new(),
                attacker: None,
                inline_effects: None,
            });
            self.state.trigger_enqueue_counter = self
                .state
                .trigger_enqueue_counter
                .checked_add(1)
                .expect("trigger occurrence identity exhausted");
        }
    }

    /// Cards whose triggered abilities can see this event: every permanent on
    /// the battlefield, plus — for zone changes — the moved card itself, so
    /// that leave-the-battlefield abilities look back in time (CR 603.6c).
    pub(crate) fn snapshot_trigger_sources(&self) -> Vec<TriggerSourceSnapshot> {
        let grants = self
            .static_sources()
            .any(|(_, _, card)| card.triggered_grants().next().is_some());
        let mut sources: Vec<_> = self
            .state
            .permanents
            .iter()
            .flatten()
            .filter(|permanent| grants || !self.state.cards[permanent.card].abilities.is_empty())
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
                    .find(|(object_ref, _)| object_ref.entity.0 == card.0)
                    .map(|(_, lki)| lki.clone());
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
            (TriggerCondition::OpponentSearchesLibrary, GameEvent::LibrarySearched { player }) => {
                *player != source_controller
            }
            (TriggerCondition::FromZone { zone, condition }, _) => {
                self.event_origin(event) == Some(*zone)
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
                    another,
                },
                GameEvent::PermanentSacrificed { card, controller },
            ) => {
                (!controller_only || *controller == source_controller)
                    && !(*another && *card == source_card)
                    && self.state.card_to_permanent[*card]
                        .is_some_and(|id| self.permanent_matches_predicate(id, predicate))
            }
            (TriggerCondition::YouCastSpell { predicate }, GameEvent::SpellCast { card, .. }) => {
                self.spell_caster(*card) == source_controller
                    && predicate.matches_card(&self.state.cards[*card])
            }
            (
                TriggerCondition::PlayerCastsSpell { predicate },
                GameEvent::SpellCast { card, .. },
            ) => predicate.matches_card(&self.state.cards[*card]),
            (
                TriggerCondition::AttacksWithAnother,
                GameEvent::AttackersDeclared { attackers, .. },
            ) => {
                attackers.len() >= 2
                    && self
                        .state
                        .card_to_permanent
                        .get(source_card.0)
                        .copied()
                        .flatten()
                        .is_some_and(|permanent_id| attackers.contains(&permanent_id))
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
                    ..
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
                    ..
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
                TriggerCondition::LeavesTheBattlefield { subject },
                GameEvent::CardMoved {
                    card,
                    from: Some(ZoneType::Battlefield),
                    to,
                    controller,
                    ..
                },
            ) if *to != ZoneType::Battlefield => self.subject_matches_moved_card(
                subject,
                source_card,
                source_controller,
                source_was_on_battlefield,
                *card,
                *controller,
            ),
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
                TriggerSubject::EnchantedCreature => attackers
                    .iter()
                    .any(|attacker| self.is_enchanted_by(*attacker, source_card)),
            },
            (
                TriggerCondition::AttacksAlone { subject },
                GameEvent::AttackersDeclared { player, attackers },
            ) => {
                let [attacker] = attackers.as_slice() else {
                    return false;
                };
                match subject {
                    TriggerSubject::This => self
                        .state
                        .card_to_permanent
                        .get(source_card.0)
                        .copied()
                        .flatten()
                        .is_some_and(|permanent_id| permanent_id == *attacker),
                    TriggerSubject::AnotherYouControl(predicate) => {
                        *player == source_controller
                            && self.permanent_is_other(*attacker, source_card)
                            && self.permanent_matches_predicate(*attacker, predicate)
                    }
                    TriggerSubject::AnyYouControl(predicate) => {
                        *player == source_controller
                            && self.permanent_matches_predicate(*attacker, predicate)
                    }
                    TriggerSubject::EnchantedCreature => {
                        self.is_enchanted_by(*attacker, source_card)
                    }
                }
            }
            (
                TriggerCondition::CreatureDealsCombatDamageToPlayer { .. },
                GameEvent::CombatDamageToPlayers { .. },
            ) => self.trigger_multiplicity(condition, source_card, source_controller, event) > 0,
            (
                TriggerCondition::OneOrMoreDealCombatDamageToPlayer { predicate },
                GameEvent::CombatDamageToPlayers { creatures },
            ) => creatures.iter().any(|creature| {
                self.state.permanents[*creature]
                    .as_ref()
                    .is_some_and(|permanent| permanent.controller == source_controller)
                    && self.permanent_matches_predicate(*creature, predicate)
            }),
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
                    by_controller,
                    ..
                },
            ) => {
                // Ward triggers only for spells and abilities opponents
                // control (CR 702.21a).
                *by_controller != source_controller
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
            )
            | (
                TriggerCondition::BeginningOfYourFirstMainPhase,
                GameEvent::StepStarted {
                    step: StepKind::Main,
                },
            ) => self.active_player() == source_controller,
            (
                TriggerCondition::YouDrawNthCardThisTurn { n },
                GameEvent::CardDrawn {
                    player,
                    nth_this_turn,
                },
            ) => *player == source_controller && nth_this_turn == n,
            (
                TriggerCondition::YourAttackingCreatureTriggers,
                GameEvent::AbilityTriggered {
                    controller,
                    attacker: Some(attacker),
                    ..
                },
            ) => {
                *controller == source_controller
                    && self.state.permanents[*attacker]
                        .as_ref()
                        .is_some_and(|permanent| permanent.controller == source_controller)
            }
            (TriggerCondition::YouScry, GameEvent::Scried { player }) => {
                *player == source_controller
            }
            (
                TriggerCondition::YouPutCountersOnCreature { kind },
                GameEvent::CountersPut {
                    permanent,
                    kind: put,
                    by,
                    ..
                },
            ) => {
                put == kind
                    && *by == source_controller
                    && self.permanent_is_battlefield_creature(*permanent)
            }
            (
                TriggerCondition::OpponentDrawsNthCardThisTurn { n },
                GameEvent::CardDrawn {
                    player,
                    nth_this_turn,
                },
            ) => *player != source_controller && nth_this_turn == n,
            _ => false,
        }
    }

    /// The zone a cast spell was cast from, or a moved card came from.
    fn event_origin(&self, event: &GameEvent) -> Option<ZoneType> {
        match event {
            GameEvent::SpellCast { card, .. } => {
                self.state
                    .stack_objects
                    .iter()
                    .find_map(|object| match object {
                        StackObject::Spell(spell) if spell.card == *card => Some(spell.cast_from),
                        _ => None,
                    })
            }
            GameEvent::CardMoved { from, .. } => *from,
            _ => None,
        }
    }

    /// How many times one event triggers a matching ability: once, except
    /// "whenever a creature ... deals combat damage to a player", which
    /// triggers for each creature that did (CR 603.2c).
    fn trigger_multiplicity(
        &self,
        condition: &TriggerCondition,
        source_card: CardId,
        source_controller: PlayerId,
        event: &GameEvent,
    ) -> usize {
        match (condition, event) {
            (
                TriggerCondition::CreatureDealsCombatDamageToPlayer { subject },
                GameEvent::CombatDamageToPlayers { creatures },
            ) => creatures
                .iter()
                .filter(|creature| {
                    self.subject_matches_permanent(
                        subject,
                        source_card,
                        source_controller,
                        **creature,
                    )
                })
                .count(),
            _ => 1,
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
            .filter(|id| self.state.permanents[*id].is_some())
            .map(|id| self.permanent_is_animated(id))
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
            // The Aura is still attached to the departed object as the
            // event is matched (CR 603.10a).
            TriggerSubject::EnchantedCreature => self.state.card_to_permanent[source_card]
                .and_then(|aura| self.state.permanents[aura].as_ref())
                .and_then(|aura| aura.attached_to)
                .is_some_and(|attached| {
                    attached.entity.0 == moved_card.0
                        && self.state.object_lki.contains_key(&attached)
                }),
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
            TriggerSubject::EnchantedCreature => self.is_enchanted_by(permanent_id, source_card),
        }
    }

    /// Whether the Aura card `aura` is on the battlefield attached to the
    /// permanent.
    fn is_enchanted_by(&self, permanent_id: PermanentId, aura: CardId) -> bool {
        self.state.card_to_permanent[aura].and_then(|aura| self.attached_permanent(aura))
            == Some(permanent_id)
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
