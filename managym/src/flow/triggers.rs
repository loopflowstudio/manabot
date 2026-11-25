// triggers.rs
// Triggered ability processing: event handling, trigger matching, and target selection.

use std::sync::Arc;

use crate::{
    agent::action::{Action, ActionSpace, ActionSpaceKind, AgentError},
    flow::{
        event::{CombatPlayerHit, DamageTarget, EventKind, GameEvent},
        game::Game,
        trigger::{
            DelayedTrigger, DelayedTriggerEvent, PendingTrigger, TriggerId, TriggerPlacement,
        },
        turn::{Bending, StepKind},
    },
    state::{
        ability::{
            Ability, Actor, DamageRecipientFilter, DelayedFrequency, Effect, EffectValue,
            EffectiveTrigger, KeywordAction, Per, SpellTargetFilter, TargetRequirement, TargetSpec,
            TriggerCondition, TriggerSubject, TurnStep,
        },
        card::CardDefId,
        effect::{EffectDuration, Modification},
        game_object::{AbilitySource, ObjectId, ObjectLki, PermanentId, PlayerId, Target},
        permanent::CounterKind,
        stack_object::{StackObject, TriggeredAbilityOnStack},
        zone::ZoneType,
    },
};

/// One occurrence of a matched triggered ability, before it waits to be
/// placed: a printed ability by index, or a granted one's effects.
#[derive(Clone, Copy)]
struct FiredTrigger<'a> {
    context: Option<Target>,
    player: Option<PlayerId>,
    amount: i32,
    once: bool,
    source: &'a TriggerSourceSnapshot,
    ability: TriggerAbility<'a>,
}

#[derive(Clone, Copy)]
enum TriggerAbility<'a> {
    Printed(usize),
    Granted(&'a EffectiveTrigger),
}

/// The event member bound by a per-object ability. Batch abilities observe the whole event.
#[derive(Clone, Copy)]
enum TriggerMember {
    Event,
    Attacker(PermanentId),
    CombatHit(CombatPlayerHit),
    CombatTotal(ObjectId, u32),
}

/// Exact battlefield source facts used while matching one or more committed
/// events. SBA batches reuse one sorted snapshot so simultaneous departures
/// see the same pre-batch abilities (CR 603.6c, 704.3).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TriggerSourceSnapshot {
    pub(crate) lki: ObjectLki,
}

impl serde::Serialize for TriggerSourceSnapshot {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        // V16 records the source slot under `card`; LKI owns the exact id.
        let mut snapshot = serializer.serialize_struct("TriggerSourceSnapshot", 3)?;
        snapshot.serialize_field("card", &self.lki.object)?;
        snapshot.serialize_field("controller", &self.lki.controller)?;
        snapshot.serialize_field("lki", &self.lki)?;
        snapshot.end()
    }
}

/// A set of `EventKind`s, one bit per variant, and a bit of its own for
/// state triggers (CR 603.8), which are asked of their source, never an
/// event.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct EventKinds(u32);

impl EventKinds {
    const NONE: Self = Self(0);
    const STATE_TRIGGER: Self = Self(1 << 31);

    fn of(kind: EventKind) -> Self {
        Self(1 << kind as u32)
    }

    fn contains(self, kind: EventKind) -> bool {
        self.intersects(Self::of(kind))
    }

    fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }
}

impl std::ops::BitOr for EventKinds {
    type Output = Self;

    fn bitor(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// The kinds of event a trigger condition can fire on. Wrappers defer to
/// the condition they wrap. Debug builds assert, each time a condition
/// fires, that its event's kind is listed here, so the index below never
/// leaves out a source that would have fired.
fn condition_event_kinds(condition: &TriggerCondition) -> EventKinds {
    match condition.leaf() {
        TriggerCondition::Sacrificed { .. } => EventKinds::of(EventKind::PermanentSacrificed),
        // `check_state_triggers` asks these, never an event.
        TriggerCondition::NoCounters { .. } | TriggerCondition::State(_) => {
            EventKinds::STATE_TRIGGER
        }
        TriggerCondition::Casts { .. }
        | TriggerCondition::Expend(_)
        | TriggerCondition::YouCastThis
        | TriggerCondition::SpellTargetsOnlyThis { .. } => EventKinds::of(EventKind::SpellCast),
        TriggerCondition::EntersTheBattlefield { .. }
        | TriggerCondition::Dies { .. }
        | TriggerCondition::LeavesTheBattlefield { .. } => EventKinds::of(EventKind::CardMoved),
        TriggerCondition::Attacks { .. } => EventKinds::of(EventKind::AttackersDeclared),
        TriggerCondition::Performs { any_of, .. } => {
            any_of.iter().fold(EventKinds::NONE, |kinds, action| {
                kinds | EventKinds::of(keyword_action_event(*action))
            })
        }
        TriggerCondition::PutIntoGraveyard { .. } => EventKinds::of(EventKind::CardMoved),
        TriggerCondition::DealsCombatDamage { to: Some(_), .. } => {
            EventKinds::of(EventKind::CombatDamageToPlayers)
        }
        TriggerCondition::DealsDamage { .. } => EventKinds::of(EventKind::DamageDealt),
        TriggerCondition::DealsCombatDamage { to: None, .. } => {
            EventKinds::of(EventKind::CombatDamageTotals)
        }
        TriggerCondition::BecomesTapped { .. } | TriggerCondition::TappedForMana { .. } => {
            EventKinds::of(EventKind::PermanentTapped)
        }
        TriggerCondition::BecomesUntapped { .. } => EventKinds::of(EventKind::PermanentUntapped),
        TriggerCondition::BecomesTargeted { .. } => EventKinds::of(EventKind::PermanentTargeted),
        TriggerCondition::BeginningOf { .. } => EventKinds::of(EventKind::StepStarted),
        TriggerCondition::YourAttackingCreatureTriggers => {
            EventKinds::of(EventKind::AbilityTriggered)
        }
        TriggerCondition::GainsLife { .. } | TriggerCondition::LosesLife { .. } => {
            EventKinds::of(EventKind::LifeChanged)
        }
        TriggerCondition::Draws { .. } => EventKinds::of(EventKind::CardDrawn),
        TriggerCondition::NthCounter { .. }
        | TriggerCondition::PutsCounters { .. }
        | TriggerCondition::PutsCountersOn { .. }
        | TriggerCondition::Chapter { .. } => EventKinds::of(EventKind::CountersPut),
        wrapper => unreachable!("TriggerCondition::leaf left wrapper {wrapper:?}"),
    }
}

/// The amount a fired trigger binds from its event (`EffectValue::TriggeringAmount`):
/// the damage of its own combat hit, combat total, or damage event, or the life
/// a player actually lost.
fn event_amount(event: &GameEvent, member: TriggerMember) -> i32 {
    match (event, member) {
        (_, TriggerMember::CombatHit(hit)) => hit.amount,
        (_, TriggerMember::CombatTotal(_, total)) => total as i32,
        (GameEvent::LifeChanged { old, new, .. }, _) if new < old => old - new,
        (GameEvent::DamageDealt { amount, .. }, _) => *amount as i32,
        _ => 0,
    }
}

/// The event that reports a keyword action.
fn keyword_action_event(action: KeywordAction) -> EventKind {
    match action {
        KeywordAction::Search => EventKind::LibrarySearched,
        KeywordAction::Scry => EventKind::Scried,
        KeywordAction::Surveil => EventKind::Surveilled,
        KeywordAction::Waterbend
        | KeywordAction::Earthbend
        | KeywordAction::Firebend
        | KeywordAction::Airbend => EventKind::Bent,
        KeywordAction::Discard => EventKind::CardMoved,
    }
}

/// The player and keyword action an event reports, if it reports one.
fn performed_action(event: &GameEvent) -> Option<(PlayerId, KeywordAction)> {
    match event {
        GameEvent::LibrarySearched { player } => Some((*player, KeywordAction::Search)),
        GameEvent::Scried { player } => Some((*player, KeywordAction::Scry)),
        GameEvent::Surveilled { player } => Some((*player, KeywordAction::Surveil)),
        GameEvent::Bent { player, kind } => Some((
            *player,
            match kind {
                Bending::Water => KeywordAction::Waterbend,
                Bending::Earth => KeywordAction::Earthbend,
                Bending::Fire => KeywordAction::Firebend,
                Bending::Air => KeywordAction::Airbend,
            },
        )),
        // The engine discards by moving a card from hand to its owner's
        // graveyard, the event's controller.
        GameEvent::CardMoved {
            from: Some(ZoneType::Hand),
            to: ZoneType::Graveyard,
            controller,
            ..
        } => Some((*controller, KeywordAction::Discard)),
        _ => None,
    }
}

/// The trigger step a started step is, if abilities can watch its beginning.
fn turn_step(step: StepKind) -> Option<TurnStep> {
    match step {
        StepKind::Upkeep => Some(TurnStep::Upkeep),
        StepKind::Draw => Some(TurnStep::Draw),
        StepKind::Main => Some(TurnStep::FirstMain),
        StepKind::BeginningOfCombat => Some(TurnStep::Combat),
        StepKind::End => Some(TurnStep::End),
        _ => None,
    }
}

/// A battlefield permanent and the event kinds its printed triggered
/// abilities can fire on.
#[derive(Debug, PartialEq, Eq)]
struct TriggerSourceEntry {
    object: ObjectId,
    kinds: EventKinds,
}

/// Which battlefield permanents can have a triggered ability that fires on
/// each kind of event, at one `triggers` revision (`Revisions`, cached in
/// `DerivedCache`): every permanent with the kinds of its printed abilities
/// and those it inherits from a `HasAbilitiesOfSource` record, in object
/// order like `snapshot_trigger_sources`, plus the kinds of every
/// static triggered-ability grant, which may reach any permanent its scope
/// covers.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct TriggerIndex {
    sources: Vec<TriggerSourceEntry>,
}

impl Game {
    pub(crate) fn pending_trigger_definition(&self, trigger: &PendingTrigger) -> CardDefId {
        trigger.source_lki.as_ref().map_or_else(
            || self.state.source_base(trigger.source_object()).0,
            |lki| lki.definition_id,
        )
    }

    fn placing_trigger(&self) -> Option<&PendingTrigger> {
        self.state.trigger_placement.as_ref()?.placing.as_ref()
    }

    fn placing_trigger_mut(&mut self) -> Option<&mut PendingTrigger> {
        self.state
            .trigger_placement_mut()
            .as_mut()?
            .placing
            .as_mut()
    }

    /// The target requirements for this occurrence's chosen effects.
    fn trigger_requirements(&self, trigger: &PendingTrigger) -> Option<Vec<TargetRequirement>> {
        let effects = match &trigger.inline_effects {
            Some(effects) => std::borrow::Cow::Borrowed(effects.as_slice()),
            None => crate::state::ability::with_chosen_modes(
                self.source_face(
                    trigger.source_object(),
                    self.pending_trigger_definition(trigger),
                )?
                .abilities
                .get(trigger.ability_index)?
                .effects(),
                &trigger.chosen_modes,
            ),
        };
        Some(crate::state::ability::effect_target_requirements(&effects))
    }

    /// Check against earlier targets of this clause. Paid X belongs to the trigger.
    fn fits_trigger_target_clause(
        &self,
        trigger: &PendingTrigger,
        requirement: &TargetRequirement,
        target: Target,
    ) -> bool {
        self.fits_target_clause(
            requirement,
            trigger.x,
            trigger.current_clause_targets(),
            target,
        )
    }

    pub(crate) fn fits_target_clause(
        &self,
        requirement: &TargetRequirement,
        x: u8,
        chosen: impl Iterator<Item = Target> + Clone,
        target: Target,
    ) -> bool {
        // A graveyard card's owner fixes which graveyard it was targeted in,
        // even if that appearance has left before a copied spell is retargeted.
        let graveyard = |target| match target {
            Target::GraveyardCard(id) => {
                self.state.objects.get(id.slot()).map(|object| object.owner)
            }
            _ => None,
        };
        let same_graveyard = !requirement.same_graveyard
            || graveyard(target)
                .is_some_and(|owner| chosen.clone().all(|other| graveyard(other) == Some(owner)));
        same_graveyard
            && (!requirement.total_mana_value_x
                || self.target_mana_value(target)
                    + chosen
                        .map(|other| self.target_mana_value(other))
                        .sum::<u32>()
                    <= u32::from(x))
    }

    pub(crate) fn target_mana_value(&self, target: Target) -> u32 {
        match target {
            Target::GraveyardCard(object) => u32::from(self.object_mana_value(object)),
            Target::Permanent(id) => self.state.permanents[id]
                .as_ref()
                .map_or(0, |p| u32::from(self.object_mana_value(p.object))),
            Target::StackSpell(object_id) => u32::from(self.object_mana_value(object_id)),
            _ => 0,
        }
    }

    pub(crate) fn choose_trigger_target(
        &mut self,
        player: PlayerId,
        target: Target,
    ) -> Result<(), AgentError> {
        let trigger = self
            .placing_trigger()
            .ok_or_else(|| AgentError("no pending target choice".into()))?;
        let requirements = self
            .trigger_requirements(trigger)
            .ok_or_else(|| AgentError("missing triggered ability".into()))?;
        let requirement = requirements
            .get(trigger.requirement_index)
            .ok_or_else(|| AgentError("no pending target requirement".into()))?;
        if trigger.controller != player
            || trigger
                .current_clause_targets()
                .any(|chosen| chosen == target)
            || !self.fits_trigger_target_clause(trigger, requirement, target)
            || !self.target_meets_requirement(target, requirement, player, trigger.source_identity)
        {
            return Err(AgentError("illegal trigger target".into()));
        }
        let trigger = self.placing_trigger_mut().expect("validated trigger");
        trigger.targets.push(target);
        trigger.target_req_indices.push(trigger.requirement_index);
        if trigger.current_clause_targets().count() >= requirement.max {
            trigger.requirement_index += 1;
        }
        Ok(())
    }

    pub(crate) fn flush_triggers(&mut self) -> Option<ActionSpace> {
        if self.state.trigger_placement.is_none() {
            if self.state.pending_triggers.is_empty() {
                return None;
            }
            *self.state.trigger_placement_mut() = Some(Box::new(TriggerPlacement {
                remaining: std::mem::take(self.state.pending_triggers_mut()),
                active: self.active_player(),
                group: 0,
                placing: None,
            }));
        }
        loop {
            let placement = self.state.trigger_placement_mut().as_mut().unwrap();
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
                *self.state.trigger_placement_mut() = None;
                return None;
            };
            // CR 700.2b: a modal trigger's mode is chosen first; with no
            // mode that has legal targets it is removed (CR 700.2a).
            if let Some((modes, up_to, unchosen)) = self.unchosen_trigger_mode_rule(&trigger) {
                let legal = self.legal_trigger_modes(&trigger, modes, unchosen);
                let modes_left = up_to.map(|value| {
                    usize::try_from(self.controller_value(value, trigger.controller))
                        .unwrap_or(0)
                        .saturating_sub(trigger.chosen_modes.len())
                });
                if modes_left.is_some_and(|left| legal.is_empty() || left == 0) {
                    // "Choose up to": the chosen modes, in printed order.
                    trigger.modes_chosen = true;
                } else if legal.is_empty() {
                    continue;
                } else {
                    let controller = trigger.controller;
                    self.state.trigger_placement_mut().as_mut().unwrap().placing = Some(trigger);
                    return Some(ActionSpace {
                        player: Some(controller),
                        kind: ActionSpaceKind::Modal,
                        actions: legal
                            .into_iter()
                            .map(|mode| Action::ChooseMode {
                                player: controller,
                                mode,
                            })
                            .chain(
                                modes_left
                                    .is_some()
                                    .then_some(Action::Decline { player: controller }),
                            )
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
                let count = trigger.current_clause_targets().count();
                let legal: Vec<_> = self
                    .targets_for_source(requirement, trigger.controller, trigger.source_identity)
                    .filter(|target| {
                        self.fits_trigger_target_clause(&trigger, requirement, *target)
                            && !trigger
                                .current_clause_targets()
                                .any(|chosen| chosen == *target)
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
                self.state.trigger_placement_mut().as_mut().unwrap().placing = Some(trigger);
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
    ) -> Option<&[Vec<Effect>]> {
        self.unchosen_trigger_mode_rule(trigger)
            .map(|(modes, _, _)| modes)
    }

    /// The placing trigger's `Effect::TriggerModes` parameters when its modes
    /// are not yet chosen.
    fn unchosen_trigger_mode_rule(
        &self,
        trigger: &PendingTrigger,
    ) -> Option<(&[Vec<Effect>], Option<&EffectValue>, bool)> {
        if trigger.inline_effects.is_some() || trigger.modes_chosen {
            return None;
        }
        match self
            .source_face(
                trigger.source_object(),
                self.pending_trigger_definition(trigger),
            )?
            .abilities
            .get(trigger.ability_index)?
            .effects()
        {
            [Effect::TriggerModes {
                modes,
                up_to,
                unchosen,
            }] => Some((modes, up_to.as_ref(), *unchosen)),
            _ => None,
        }
    }

    /// Modes the placing trigger may choose next: every required target has
    /// a legal candidate, it is not chosen already, and an "unchosen" rule's
    /// source has not chosen it before.
    fn legal_trigger_modes(
        &self,
        trigger: &PendingTrigger,
        modes: &[Vec<Effect>],
        unchosen: bool,
    ) -> Vec<usize> {
        let chosen_before = match trigger
            .source_ref()
            .map(|source| self.lookup_current_permanent(source))
        {
            Some(Ok(id)) if unchosen => self.chosen_modes(id),
            _ => Vec::new(),
        };
        (0..modes.len())
            .filter(|mode| !trigger.chosen_modes.contains(mode) && !chosen_before.contains(mode))
            .filter(|mode| {
                crate::state::ability::effect_target_requirements(&modes[*mode])
                    .iter()
                    .filter(|requirement| requirement.min > 0)
                    .all(|requirement| {
                        self.targets_for_source(
                            requirement,
                            trigger.controller,
                            trigger.source_identity,
                        )
                        .next()
                        .is_some()
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
            .placing_trigger()
            .ok_or_else(|| AgentError("no pending trigger mode".into()))?;
        let (modes, up_to, unchosen) = self
            .unchosen_trigger_mode_rule(trigger)
            .ok_or_else(|| AgentError("no pending trigger mode".into()))?;
        let is_up_to = up_to.is_some();
        if trigger.controller != player
            || !self
                .legal_trigger_modes(trigger, modes, unchosen)
                .contains(&mode)
        {
            return Err(AgentError("illegal trigger mode".into()));
        }
        let source = trigger
            .source_ref()
            .and_then(|source| self.lookup_current_permanent(source).ok());
        let trigger = self.placing_trigger_mut().expect("validated trigger");
        trigger.chosen_modes.push(mode);
        if !is_up_to {
            trigger.modes_chosen = true;
            if let (true, Some(source)) = (unchosen, source) {
                let affected = self.affected_permanent(source);
                let object = self.state.permanents[source].as_ref().map(|p| p.object);
                self.apply_continuous(
                    object,
                    player,
                    affected,
                    Modification::ModeChosen {
                        mode: u8::try_from(mode).unwrap_or(u8::MAX),
                    },
                    EffectDuration::Indefinite,
                );
            }
        }
        Ok(())
    }

    pub(crate) fn decline_trigger_target(&mut self, player: PlayerId) -> Result<(), AgentError> {
        let trigger = self
            .placing_trigger()
            .ok_or_else(|| AgentError("no pending trigger".into()))?;
        if matches!(
            self.unchosen_trigger_mode_rule(trigger),
            Some((_, Some(_), _))
        ) {
            if trigger.controller != player {
                return Err(AgentError("not the trigger's controller".into()));
            }
            self.placing_trigger_mut()
                .expect("validated trigger")
                .modes_chosen = true;
            return Ok(());
        }
        let requirements = self
            .trigger_requirements(trigger)
            .ok_or_else(|| AgentError("missing triggered ability".into()))?;
        let requirement = requirements
            .get(trigger.requirement_index)
            .ok_or_else(|| AgentError("no pending target requirement".into()))?;
        let count = trigger.current_clause_targets().count();
        if trigger.controller != player
            || count < requirement.min
            || requirement.max == requirement.min
        {
            return Err(AgentError("trigger cannot decline target".into()));
        }
        self.placing_trigger_mut()
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
        let source_definition_id = self.pending_trigger_definition(&trigger);
        let id = self.state.id_gen_mut().next_id();
        let targets = trigger.targets;
        self.emit_permanents_targeted(&targets, Target::StackAbility(id), trigger.controller);
        self.push_to_stack(StackObject::TriggeredAbility(TriggeredAbilityOnStack {
            triggering_amount: trigger.triggering_amount,
            program: trigger.program,
            remembered: trigger.remembered,
            source_transforms: trigger.source_lki.as_ref().map_or(0, |lki| lki.transforms),
            id,
            controller: trigger.controller,
            source_identity: trigger.source_identity,
            source_lki: trigger.source_lki,
            source_definition_id,
            ability_index: trigger.ability_index,
            targets,
            target_req_indices: trigger.target_req_indices,
            context: trigger.context,
            player: trigger.player,
            x: trigger.x,
            spent: trigger.spent.clone(),
            chosen_modes: trigger.chosen_modes,
            inline_effects: trigger.inline_effects,
        }));
        self.emit(GameEvent::AbilityTriggered {
            source_object: trigger.source_identity.object(),
            controller: trigger.controller,
            ability: id,
            attacker: trigger.attacker,
        });
        self.restart_priority_round();
    }

    pub(crate) fn targets_for_source<'a>(
        &'a self,
        requirement: &'a TargetRequirement,
        controller: PlayerId,
        source: AbilitySource,
    ) -> impl Iterator<Item = Target> + 'a {
        self.target_candidates(&requirement.spec, controller)
            .into_iter()
            .filter(move |target| {
                self.target_meets_requirement(*target, requirement, controller, source)
            })
    }

    #[cfg(test)]
    pub(crate) fn legal_targets_for_spec(
        &self,
        spec: &TargetSpec,
        controller: PlayerId,
    ) -> Vec<Target> {
        self.target_candidates(spec, controller)
            .into_iter()
            .filter(|target| self.target_is_legal(*target, spec, controller))
            .collect()
    }

    fn target_candidates(&self, target_spec: &TargetSpec, controller: PlayerId) -> Vec<Target> {
        let mut candidates: Vec<Target> = match target_spec {
            TargetSpec::GraveyardCardYouOwn { .. }
            | TargetSpec::GraveyardCardYouOwnWithinSourcePower { .. } => self
                .state
                .zone_cards(ZoneType::Graveyard, controller)
                .filter_map(|card| self.state.live(self.state.object_of(card)))
                .map(Target::GraveyardCard)
                .collect(),
            TargetSpec::GraveyardCard { .. } | TargetSpec::GraveyardCardOpponentOwns { .. } => {
                [PlayerId(0), PlayerId(1)]
                    .into_iter()
                    .flat_map(|player| self.state.zone_cards(ZoneType::Graveyard, player))
                    .filter_map(|card| self.state.live(self.state.object_of(card)))
                    .map(Target::GraveyardCard)
                    .collect()
            }
            TargetSpec::Player
            | TargetSpec::Opponent
            | TargetSpec::AnyTarget
            | TargetSpec::PlayerOrPlaneswalker => {
                vec![Target::Player(PlayerId(0)), Target::Player(PlayerId(1))]
            }
            TargetSpec::Spell
            | TargetSpec::SpellMatching { .. }
            | TargetSpec::SpellOrPermanent { .. }
            | TargetSpec::CreatureOrSpell => self
                .state
                .stack_objects
                .iter()
                .rev()
                .filter_map(|object| match object {
                    StackObject::Spell(spell) => Some(Target::StackSpell(spell.object)),
                    _ => None,
                })
                .collect(),
            TargetSpec::SpellOrAbilityWithSingleTarget => self
                .state
                .stack_objects
                .iter()
                .rev()
                .map(StackObject::as_target)
                .collect(),
            TargetSpec::Ability => self
                .state
                .stack_objects
                .iter()
                .rev()
                .filter(|object| !matches!(object, StackObject::Spell(_)))
                .map(|object| Target::StackAbility(object.id()))
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
                | TargetSpec::AnyTarget
                | TargetSpec::PlayerOrPlaneswalker
                | TargetSpec::CreatureYouControl
                | TargetSpec::CreatureOpponentControls
                | TargetSpec::LandYouControl
                | TargetSpec::PermanentOpponentControls { .. }
                | TargetSpec::SpellOrPermanent { .. }
                | TargetSpec::CreatureOrSpell
        ) {
            for player in [PlayerId(0), PlayerId(1)] {
                for object_id in self
                    .state
                    .zones
                    .zone_objects(ZoneType::Battlefield, player)
                    .iter()
                {
                    if let Some(permanent_id) = self.state.permanent_of(*object_id) {
                        candidates.push(Target::Permanent(permanent_id));
                    }
                }
            }
        }
        candidates
    }

    /// Collect occurrences in order without placing them on the stack.
    pub(crate) fn collect_waiting_triggers(&mut self) {
        self.process_game_events();
        self.check_state_triggers();
    }

    /// Match pending committed events against the applicable exact sources.
    pub(crate) fn process_game_events(&mut self) {
        self.drain_pending_events(None);
    }

    /// Departures look back to the whole pre-batch battlefield. Entry and
    /// counter events see the completed battlefield, including every entrant.
    pub(crate) fn process_event_batch(&mut self, before: &[TriggerSourceSnapshot]) {
        self.drain_pending_events(Some(before));
    }

    fn drain_pending_events(&mut self, before: Option<&[TriggerSourceSnapshot]>) {
        while !self.state.pending_events.is_empty() {
            let events = std::mem::take(self.state.pending_events_mut());
            for event in events {
                let use_before = before.filter(|_| {
                    matches!(
                        event,
                        GameEvent::CardMoved {
                            from: Some(ZoneType::Battlefield),
                            ..
                        } | GameEvent::PermanentSacrificed { .. }
                    )
                });
                if let Some(sources) = use_before {
                    self.check_triggers_for_event(&event, sources);
                } else {
                    let sources = self.trigger_ability_sources(&event);
                    self.check_triggers_for_event(&event, &sources);
                }
            }
        }
    }

    /// Resolve printed and granted trigger matches before committing the event.
    fn firing_battlefield_triggers<'a>(
        &self,
        event: &GameEvent,
        sources: &'a [TriggerSourceSnapshot],
    ) -> Vec<FiredTrigger<'a>> {
        let mut fired = Vec::new();
        for source in sources {
            let lki = &source.lki;
            let face = self
                .source_face(lki.object, lki.definition_id)
                .expect("source face");
            let printed = face
                .abilities
                .iter()
                .enumerate()
                .filter(|_| !lki.abilities_lost)
                .map(|(index, ability)| (TriggerAbility::Printed(index), ability));
            let granted = lki
                .granted_abilities
                .iter()
                .map(|grant| (TriggerAbility::Granted(grant), &grant.ability));
            for (origin, ability) in printed.chain(granted) {
                let Ability::Triggered {
                    condition,
                    intervening_if,
                    ..
                } = ability;
                let Some((leaf, once)) =
                    self.match_trigger_wrappers(condition, event, lki.object, lki.controller)
                else {
                    continue;
                };
                if once {
                    let used = match origin {
                        TriggerAbility::Printed(index) => self
                            .state
                            .turn
                            .ability_triggers_this_turn
                            .contains_key(&(lki.object, lki.definition_id, index)),
                        TriggerAbility::Granted(grant) => self
                            .state
                            .turn
                            .used_trigger_grants
                            .contains(lki.object, grant.clock.expect("limited grant clock")),
                    };
                    if used {
                        continue;
                    }
                }
                let mut copies = once.then_some(1);
                let mut emitted = false;
                self.collect_occurrences(leaf, event, source, |member| {
                    if once && emitted {
                        return;
                    }
                    let context = self.trigger_context(ability, member, event);
                    if intervening_if.as_ref().is_some_and(|condition| {
                        !self.check_intervening_if(
                            condition,
                            lki.controller,
                            Some(lki.object),
                            context,
                        )
                    }) {
                        return;
                    }
                    // CR 603.2d doubles each occurrence; a once-per-turn limit caps it at one.
                    let count =
                        *copies.get_or_insert_with(|| 1 + self.additional_trigger_times(source));
                    fired.extend(std::iter::repeat_n(
                        FiredTrigger {
                            source,
                            ability: origin,
                            context,
                            player: self.event_player(leaf, event, member),
                            amount: event_amount(event, member),
                            once,
                        },
                        count,
                    ));
                    emitted = true;
                });
            }
        }

        fired
    }

    fn check_triggers_for_event(&mut self, event: &GameEvent, sources: &[TriggerSourceSnapshot]) {
        #[cfg(feature = "benchmark")]
        crate::benchmark_layers::record(|m| m.trigger_sources += sources.len() as u64);
        self.graveyard_triggers(event);
        self.cast_triggers(event);
        self.delayed_triggers_for_event(event);
        let fired = self.firing_battlefield_triggers(event, sources);

        #[cfg(feature = "benchmark")]
        crate::benchmark_layers::record(|m| m.triggers_found += fired.len() as u64);
        for FiredTrigger {
            context,
            player,
            amount,
            once,
            source,
            ability,
        } in fired
        {
            let source_lki = &source.lki;
            let controller = source_lki.controller;
            let source_object = source_lki.object;
            let (ability_index, granted, program) = match ability {
                TriggerAbility::Printed(index) => (index, None, false),
                TriggerAbility::Granted(grant) => {
                    let Ability::Triggered {
                        intervening_if,
                        effects,
                        ..
                    } = &grant.ability;
                    // A granted trigger retains its CR 603.4 resolution check with its program.
                    let inline = match intervening_if {
                        Some(condition) => vec![Effect::If {
                            condition: condition.clone(),
                            then: effects.clone(),
                            otherwise: Vec::new(),
                        }],
                        None => effects.clone(),
                    };
                    (0, Some(inline), grant.program)
                }
            };
            // CR 107.3m: an entering permanent's own trigger sees its spell's
            // X and the mana spent to cast it; a counters trigger's X is how
            // many were put ("that much").
            let (x, spent) = match event {
                GameEvent::CountersPut { count, .. } => {
                    (u8::try_from(*count).unwrap_or(u8::MAX), Default::default())
                }
                GameEvent::CardMoved {
                    object_id: object,
                    to: ZoneType::Battlefield,
                    ..
                } if *object == source_object => self
                    .state
                    .permanent_of(*object)
                    .and_then(|id| self.state.permanents[id].as_ref())
                    .map_or(Default::default(), |permanent| {
                        (permanent.cast_x, permanent.cast_spent.clone())
                    }),
                _ => Default::default(),
            };
            let attacker = match event {
                GameEvent::AttackersDeclared { attackers, .. } => attackers
                    .iter()
                    .copied()
                    .find(|id| self.permanent_object(*id) == Some(source_lki.object)),
                _ => None,
            };
            if once {
                match ability {
                    TriggerAbility::Printed(_) => self.spend_once_each_turn((
                        source_object,
                        source_lki.definition_id,
                        ability_index,
                    )),
                    TriggerAbility::Granted(grant) => self
                        .state
                        .turn
                        .used_trigger_grants
                        .record(source_object, grant.clock.expect("limited grant clock")),
                }
            }
            let id = self.next_trigger_id();
            let mut trigger = PendingTrigger::new(
                id,
                controller,
                AbilitySource::exact(source_object),
                Some(source_lki.clone()),
                ability_index,
            );
            trigger.triggering_amount = amount;
            trigger.context = context;
            trigger.player = player;
            trigger.x = x;
            trigger.spent = spent;
            trigger.attacker = attacker;
            trigger.inline_effects = granted;
            trigger.program = program;
            self.state.pending_triggers_mut().push(trigger);
        }
    }

    fn collect_occurrences(
        &self,
        condition: &TriggerCondition,
        event: &GameEvent,
        source: &TriggerSourceSnapshot,
        mut emit: impl FnMut(TriggerMember),
    ) {
        let lki = &source.lki;
        match (condition, event) {
            (
                TriggerCondition::Attacks {
                    subject,
                    per: Per::Object,
                    attackers: size,
                    nth,
                },
                GameEvent::AttackersDeclared { attackers, .. },
            ) => {
                if size.matches(attackers.len()) {
                    for &id in attackers {
                        if self.attack_matches(subject, *nth, lki.object, lki.controller, id) {
                            emit(TriggerMember::Attacker(id));
                        }
                    }
                }
            }
            (
                TriggerCondition::DealsCombatDamage {
                    subject,
                    to: Some(to),
                    per: Per::Object,
                },
                GameEvent::CombatDamageToPlayers { hits },
            ) => {
                for hit in hits {
                    if self.hit_matches(hit, subject, *to, lki.object, lki.controller) {
                        emit(TriggerMember::CombatHit(*hit));
                    }
                }
            }
            (
                TriggerCondition::DealsCombatDamage {
                    subject,
                    to: None,
                    per: Per::Object,
                },
                GameEvent::CombatDamageTotals { dealt },
            ) => {
                for &(dealer, amount) in dealt {
                    if self.subject_matches_damage_source(
                        subject,
                        lki.object,
                        lki.controller,
                        dealer,
                    ) {
                        emit(TriggerMember::CombatTotal(dealer, amount));
                    }
                }
            }
            _ if self.trigger_condition_fires(
                condition,
                lki.object,
                lki.controller,
                true,
                event,
            ) =>
            {
                debug_assert!(condition_event_kinds(condition).contains(event.kind()));
                emit(TriggerMember::Event);
            }
            _ => {}
        }
    }

    fn permanent_or_lki(&self, object: ObjectId) -> Option<PermanentId> {
        self.state
            .permanent_of(object)
            .or_else(|| self.object_lki(object).map(|lki| lki.permanent))
    }

    /// Bind the event object once so the intervening condition and the
    /// resulting stack ability read the same occurrence (CR 603.4).
    fn trigger_context(
        &self,
        ability: &Ability,
        member: TriggerMember,
        event: &GameEvent,
    ) -> Option<Target> {
        match member {
            TriggerMember::Attacker(id) => return Some(Target::Permanent(id)),
            TriggerMember::CombatHit(hit) if ability.reads_triggering_object() => {
                return self.permanent_or_lki(hit.source).map(Target::Permanent);
            }
            TriggerMember::CombatHit(_) => return None,
            TriggerMember::CombatTotal(source, _) => {
                return self.permanent_or_lki(source).map(Target::Permanent)
            }
            TriggerMember::Event => {}
        }
        match event {
            GameEvent::PermanentTargeted { by, .. } => Some(*by),
            GameEvent::AbilityTriggered {
                ability,
                attacker: Some(_),
                ..
            } => Some(Target::StackAbility(*ability)),
            _ if !ability.reads_triggering_object() => None,
            GameEvent::SpellCast { object_id, .. } => Some(Target::StackSpell(*object_id)),
            // "That creature" / "that source" of a damage trigger: the dealing
            // permanent live or as it last existed, else the dealing spell,
            // as `subject_matches_damage_source` matched it.
            GameEvent::DamageDealt {
                source: Some(source),
                ..
            } => Some(
                self.permanent_or_lki(*source)
                    .map_or(Target::StackSpell(*source), Target::Permanent),
            ),
            GameEvent::CardMoved {
                object_id,
                to: ZoneType::Battlefield,
                ..
            } => self
                .lookup_current_permanent(*object_id)
                .ok()
                .map(Target::Permanent),
            GameEvent::CardMoved {
                from: Some(ZoneType::Battlefield),
                ..
            } => event
                .departed_object()
                .and_then(|object| self.object_lki(object))
                .map(|lki| Target::Permanent(lki.permanent)),
            GameEvent::CardMoved {
                object_id,
                to: ZoneType::Graveyard,
                ..
            } => Some(Target::GraveyardCard(*object_id)),
            GameEvent::AttackersDeclared { attackers, .. } if attackers.len() == 1 => {
                Some(Target::Permanent(attackers[0]))
            }
            _ => None,
        }
    }

    /// "That player" for a trigger whose event `leaf` matched `event`: the
    /// recipient of a combat hit, the caster, drawer or performer, whose
    /// graveyard a card was put into, or whose step it is. A permanent that
    /// dies, leaves, enters or is sacrificed binds its controller as it
    /// last existed, which an owner's graveyard need not share (Massacre
    /// Wurm and a stolen creature).
    fn event_player(
        &self,
        leaf: &TriggerCondition,
        event: &GameEvent,
        member: TriggerMember,
    ) -> Option<PlayerId> {
        match event {
            GameEvent::CombatDamageToPlayers { .. } => match member {
                TriggerMember::CombatHit(hit) => Some(hit.recipient),
                _ => None,
            },
            GameEvent::SpellCast { caster, .. } => Some(*caster),
            GameEvent::CardDrawn { player, .. } => Some(*player),
            GameEvent::LifeChanged { player, .. } => Some(*player),
            GameEvent::DamageDealt {
                target: DamageTarget::Player(player),
                ..
            } => Some(*player),
            GameEvent::PermanentSacrificed { controller, .. } => Some(*controller),
            GameEvent::CardMoved { controller, .. }
                if matches!(
                    leaf,
                    TriggerCondition::Dies { .. }
                        | TriggerCondition::LeavesTheBattlefield { .. }
                        | TriggerCondition::EntersTheBattlefield { .. }
                ) =>
            {
                Some(*controller)
            }
            GameEvent::CardMoved {
                object_id: object,
                to: ZoneType::Graveyard,
                ..
            } => Some(self.state[*object].owner),
            GameEvent::StepStarted { .. } => Some(self.active_player()),
            _ => performed_action(event).map(|(player, _)| player),
        }
    }

    /// Extra occurrences of one of `source`'s triggered abilities: one for
    /// each permanent its controller controls whose "triggers an additional
    /// time" covers the source (CR 603.2d), as the source is or last was.
    fn additional_trigger_times(&self, source: &TriggerSourceSnapshot) -> usize {
        let current = self.lookup_current_permanent(source.lki.object).ok();
        self.static_sources()
            .filter(|(_, controller, _)| *controller == source.lki.controller)
            .map(|(_, _, grantor)| {
                grantor
                    .static_abilities
                    .iter()
                    .filter_map(|ability| match ability {
                        crate::state::card::StaticAbility::TriggersAdditionalTime(predicate) => {
                            Some(predicate)
                        }
                        _ => None,
                    })
                    .filter(|predicate| match current {
                        Some(id) => self.permanent_matches_predicate(id, predicate),
                        None => self.lki_matches_predicate(&source.lki, predicate),
                    })
                    .count()
            })
            .sum()
    }

    /// Wait for a delayed triggered ability's event (CR 603.7).
    pub(crate) fn add_delayed_trigger(&mut self, trigger: DelayedTrigger) {
        self.state.delayed_triggers_mut().push(trigger);
    }

    /// Match ordinary events against waiting watchers. `Leaves` matches at
    /// the exact object's battlefield departure instead.
    /// One-shot triggers are removed as they fire (CR 603.7c).
    fn delayed_triggers_for_event(&mut self, event: &GameEvent) {
        let mut fired = Vec::new();
        let mut spent = Vec::new();
        for (index, trigger) in self.state.delayed_triggers.iter().enumerate() {
            let context = match (&trigger.event, event) {
                (
                    DelayedTriggerEvent::CreatureDies,
                    GameEvent::CardMoved {
                        from: Some(ZoneType::Battlefield),
                        to: ZoneType::Graveyard,
                        ..
                    },
                ) => {
                    let Some(lki) = event
                        .departed_object()
                        .and_then(|object| self.object_lki(object))
                        .filter(|lki| lki.was_creature())
                    else {
                        continue;
                    };
                    Some(Target::Permanent(lki.permanent))
                }
                (
                    DelayedTriggerEvent::EndStep { your_turn },
                    GameEvent::StepStarted {
                        step: StepKind::End,
                    },
                ) if !your_turn || self.active_player() == trigger.controller => None,
                (
                    DelayedTriggerEvent::Upkeep,
                    GameEvent::StepStarted {
                        step: StepKind::Upkeep,
                    },
                ) => None,
                (
                    DelayedTriggerEvent::YouCastSpell(predicate),
                    GameEvent::SpellCast {
                        object_id: object,
                        caster,
                        ..
                    },
                ) if *caster == trigger.controller
                    && self.object_matches_predicate(*object, predicate) =>
                {
                    Some(Target::StackSpell(*object))
                }
                _ => continue,
            };
            if trigger.frequency == DelayedFrequency::Once {
                spent.push(index);
            }
            fired.push((trigger.clone(), context));
        }
        for index in spent.into_iter().rev() {
            self.state.delayed_triggers_mut().remove(index);
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
            trigger.source_object(),
            trigger.source_ref(),
            trigger.controller,
            trigger.effects.clone(),
            context,
        )
        .remembered = trigger.remembered.clone();
    }

    /// A fresh identity for the trigger occurrence about to be enqueued.
    fn next_trigger_id(&mut self) -> TriggerId {
        let id = TriggerId(self.state.trigger_enqueue_counter);
        *self.state.trigger_enqueue_counter_mut() =
            id.0.checked_add(1)
                .expect("trigger occurrence identity exhausted");
        id
    }

    /// Put an occurrence with inline effects (no `Object::abilities` entry to
    /// reference: a delayed, reflexive or granted ability) in the
    /// waiting list, with its source's last known information when that was
    /// a battlefield object.
    pub(crate) fn enqueue_inline_trigger(
        &mut self,
        source_object: ObjectId,
        source_ref: Option<ObjectId>,
        controller: PlayerId,
        effects: Vec<Effect>,
        context: Option<Target>,
    ) -> &mut PendingTrigger {
        let source_lki = source_ref.and_then(|source| {
            self.object_lki(source).cloned().or_else(|| {
                (source == source_object && self.state.live(source).is_some())
                    .then(|| self.snapshot_current_permanent(source))
                    .flatten()
            })
        });
        let id = self.next_trigger_id();
        let mut trigger = PendingTrigger::new(
            id,
            controller,
            AbilitySource::new(source_object, source_ref),
            source_lki,
            0,
        );
        trigger.context = context;
        trigger.inline_effects = Some(effects);
        self.state.pending_triggers_mut().push(trigger);
        self.state
            .pending_triggers_mut()
            .last_mut()
            .expect("enqueued trigger")
    }

    /// State triggers (CR 603.8) of battlefield sources whose state holds,
    /// unless that exact source's occurrence is still waiting or on the stack.
    fn check_state_triggers(&mut self) {
        let mut fired = Vec::new();
        for source in self.indexed_sources(EventKinds::STATE_TRIGGER) {
            let Some(permanent) = self
                .state
                .permanent_of(source.lki.object)
                .and_then(|id| self.state.permanents[id].as_ref())
            else {
                continue;
            };
            if source.lki.abilities_lost {
                continue;
            }
            let face = self
                .source_face(source.lki.object, source.lki.definition_id)
                .expect("source face");
            let object = Some(source.lki.object);
            for (ability_index, ability) in face.abilities.iter().enumerate() {
                let Ability::Triggered { condition, .. } = ability;
                let holds = match condition {
                    TriggerCondition::NoCounters { kind } => permanent.counter_count(*kind) == 0,
                    TriggerCondition::State(condition) => {
                        self.condition_holds(condition, source.lki.controller, None, object, None)
                    }
                    _ => continue,
                };
                if holds && !self.state_trigger_waiting(object, ability_index) {
                    fired.push((source.clone(), ability_index));
                }
            }
        }
        for (source, ability_index) in fired {
            let id = self.next_trigger_id();
            let trigger = PendingTrigger::new(
                id,
                source.lki.controller,
                AbilitySource::exact(source.lki.object),
                Some(source.lki),
                ability_index,
            );
            self.state.pending_triggers_mut().push(trigger);
        }
    }

    /// CR 603.8: a state trigger doesn't trigger again while that exact
    /// source's occurrence waits to be put on the stack or is on it.
    fn state_trigger_waiting(&self, source: Option<ObjectId>, ability_index: usize) -> bool {
        // A granted or delayed trigger carries its own effects and may share
        // the printed ability's index; it is never the state trigger.
        let same = |trigger: &PendingTrigger| {
            trigger.source_ref() == source
                && trigger.ability_index == ability_index
                && trigger.inline_effects.is_none()
        };
        let placing = self
            .state
            .trigger_placement
            .as_deref()
            .into_iter()
            .flat_map(|placement| placement.remaining.iter().chain(placement.placing.as_ref()));
        self.state.pending_triggers.iter().chain(placing).any(same)
            || self.state.stack_objects.iter().any(|item| {
                matches!(item, StackObject::TriggeredAbility(trigger)
                    if trigger.source_ref() == source
                        && trigger.ability_index == ability_index
                        && trigger.inline_effects.is_none())
            })
    }

    /// "When you cast this spell" abilities of the spell just cast; the spell
    /// is the source and its caster controls the trigger.
    fn cast_triggers(&mut self, event: &GameEvent) {
        let GameEvent::SpellCast {
            object_id: object,
            caster,
            ..
        } = event
        else {
            return;
        };
        let object = *object;
        let caster = *caster;
        let fired: Vec<Vec<Effect>> = self.state[object]
            .abilities
            .iter()
            .filter(|ability| {
                let Ability::Triggered { condition, .. } = ability;
                *condition == TriggerCondition::YouCastThis
            })
            .map(|ability| ability.effects().to_vec())
            .collect();
        let source_ref = self.state.live(object);
        for effects in fired {
            self.enqueue_inline_trigger(object, source_ref, caster, effects, None);
        }
        // "When that mana is spent to cast [a spell], copy that spell": one
        // trigger of the object that made each such unit.
        let Some(spell) = self.stack_spell(object) else {
            return;
        };
        let copy_sources: Vec<_> = spell.spent.copy_triggers().flatten().collect();
        let replicated = if self.state[object].replicate().is_some() {
            spell.x
        } else {
            0
        };
        for origin in copy_sources {
            self.enqueue_inline_trigger(
                origin.object,
                Some(origin.object),
                caster,
                vec![Effect::CopyTriggeringObject],
                Some(Target::StackSpell(object)),
            );
        }
        // CR 702.56a: replicate copies the spell once per payment, each copy
        // with a chance to choose new targets.
        if replicated > 0 {
            let copies = vec![Effect::CopyTriggeringObject; usize::from(replicated)];
            self.enqueue_inline_trigger(
                object,
                source_ref,
                caster,
                copies,
                Some(Target::StackSpell(object)),
            );
        }
    }

    /// Abilities that function from their owner's graveyard (CR 113.6k):
    /// the exact graveyard object is the source, and its owner controls it.
    fn graveyard_triggers(&mut self, event: &GameEvent) {
        let mut fired = Vec::new();
        for owner in [PlayerId(0), PlayerId(1)] {
            for &object in self
                .state
                .zones
                .zone_objects(ZoneType::Graveyard, owner)
                .iter()
            {
                for (ability_index, ability) in self.state[object].abilities.iter().enumerate() {
                    let Ability::Triggered {
                        condition: TriggerCondition::InGraveyard { condition },
                        ..
                    } = ability
                    else {
                        continue;
                    };
                    let Some((leaf, once)) =
                        self.match_trigger_wrappers(condition, event, object, owner)
                    else {
                        continue;
                    };
                    let key = (object, self.state[object].definition_id, ability_index);
                    if !(once
                        && self
                            .state
                            .turn
                            .ability_triggers_this_turn
                            .contains_key(&key))
                        && self.trigger_condition_fires(leaf, object, owner, false, event)
                    {
                        fired.push((object, ability_index, owner, once.then_some(key)));
                    }
                }
            }
        }
        for (object, ability_index, owner, key) in fired {
            if let Some(key) = key {
                self.spend_once_each_turn(key);
            }
            let source_ref = self.state.live(object);
            let id = self.next_trigger_id();
            let trigger = PendingTrigger::new(
                id,
                owner,
                AbilitySource::new(object, source_ref),
                None,
                ability_index,
            );
            self.state.pending_triggers_mut().push(trigger);
        }
    }

    /// Without a listed spend, a float still matters if it has an additional
    /// cost or effect, or its tap fires a battlefield trigger. Inspect only
    /// already legal activations and use the live event's trigger matching.
    pub(crate) fn usable_floats_without_spend(&self, player: PlayerId, floats: &mut Vec<Action>) {
        let listeners = self.indexed_sources(EventKinds::of(EventKind::PermanentTapped));
        if !listeners.is_empty() {
            self.append_mana_floats(player, floats);
        }
        floats.retain(|action| {
            let permanent = match *action {
                Action::TapForMana { permanent, .. } => permanent,
                Action::ActivateAbility {
                    permanent,
                    ability_index,
                    ..
                } => {
                    let view = self.characteristics(permanent);
                    let ability = &view.activated_abilities[ability_index];
                    if ability.mana_activation_has_side_effects() {
                        return true;
                    }
                    if !ability.tap_source {
                        return false;
                    }
                    permanent
                }
                _ => unreachable!("only mana production is appended after passing priority"),
            };
            !listeners.is_empty()
                && !self
                    .firing_battlefield_triggers(
                        &GameEvent::PermanentTapped {
                            permanent,
                            for_mana: true,
                        },
                        &listeners,
                    )
                    .is_empty()
        });
    }

    /// The sources whose abilities can see a battlefield departure or a
    /// sacrifice, as they exist before the event: the indexed permanents with a
    /// printed or granted ability of those kinds, including a departing
    /// object's own, so leave-the-battlefield abilities look back in time
    /// (CR 603.6c, 603.10a).
    pub(crate) fn snapshot_trigger_sources(&self) -> Vec<TriggerSourceSnapshot> {
        self.indexed_sources(
            EventKinds::of(EventKind::CardMoved) | EventKinds::of(EventKind::PermanentSacrificed),
        )
    }

    /// Current snapshots of the indexed permanents with an ability of any of
    /// `kinds`, in object order.
    fn indexed_sources(&self, kinds: EventKinds) -> Vec<TriggerSourceSnapshot> {
        let index = self.trigger_index();
        index
            .sources
            .iter()
            .filter(|entry| entry.kinds.intersects(kinds))
            .filter_map(|entry| self.snapshot_current_permanent(entry.object))
            .map(|lki| TriggerSourceSnapshot { lki })
            .collect()
    }

    /// Trigger candidates at the current source and characteristic revisions.
    /// Layered grants/conditions can change the event kinds without a zone move.
    fn trigger_index(&self) -> Arc<TriggerIndex> {
        let revisions = self.state.revisions();
        let cached = self
            .derived
            .borrow_mut()
            .at(revisions)
            .trigger_index
            .clone();
        if let Some(index) = cached {
            debug_assert_eq!(*index, self.build_trigger_index(), "stale trigger index");
            return index;
        }
        let index = Arc::new(self.build_trigger_index());
        self.derived.borrow_mut().at(revisions).trigger_index = Some(index.clone());
        index
    }

    fn build_trigger_index(&self) -> TriggerIndex {
        #[cfg(feature = "benchmark")]
        crate::benchmark_layers::record(|m| m.trigger_indexes += 1);
        let mut sources: Vec<TriggerSourceEntry> = self
            .layer_views(
                (0..self.state.permanents.len())
                    .map(PermanentId)
                    .filter(|&id| !self.is_phased_out(id)),
            )
            .map(|(id, view)| {
                let permanent = self.state.permanents[id].as_ref().unwrap();
                TriggerSourceEntry {
                    object: permanent.object,
                    kinds: view.triggered_abilities.iter().fold(
                        EventKinds::NONE,
                        |kinds, grant| {
                            let Ability::Triggered { condition, .. } = &grant.ability;
                            kinds | condition_event_kinds(condition)
                        },
                    ),
                }
            })
            .collect();
        // Object order (a battlefield card has one current object, so its
        // entity orders it): the order the fired triggers are numbered in.
        sources.sort_by_key(|entry| entry.object);
        TriggerIndex { sources }
    }

    /// The sources whose abilities can see `event`: the indexed permanents
    /// with a printed or grantable ability of its kind (the same facts
    /// `snapshot_trigger_sources` reads, less sources nothing could fire
    /// for), plus a moved card's last battlefield object.
    fn trigger_ability_sources(&self, event: &GameEvent) -> Vec<TriggerSourceSnapshot> {
        let mut sources = self.indexed_sources(EventKinds::of(event.kind()));

        if let GameEvent::CardMoved {
            from: Some(ZoneType::Battlefield),
            ..
        } = event
        {
            if let Some(lki) = event
                .departed_object()
                .and_then(|object| self.object_lki(object))
            {
                if let Err(index) =
                    sources.binary_search_by_key(&lki.object, |source| source.lki.object)
                {
                    sources.insert(index, TriggerSourceSnapshot { lki: lki.clone() });
                }
            }
        }

        sources
    }

    /// Peel and test wrappers at event time, retaining the leaf's cardinality and bindings.
    fn match_trigger_wrappers<'a>(
        &self,
        mut condition: &'a TriggerCondition,
        event: &GameEvent,
        object: ObjectId,
        controller: PlayerId,
    ) -> Option<(&'a TriggerCondition, bool)> {
        let mut once = false;
        let permanent = || {
            self.state
                .permanent_of(object)
                .and_then(|id| self.state.permanents[id].as_ref())
        };
        loop {
            let (inner, enabled) = match condition {
                TriggerCondition::WasCast { condition } => {
                    (condition, permanent().is_some_and(|p| p.was_cast))
                }
                TriggerCondition::FromZone { zone, condition } => {
                    (condition, self.event_origin(event) == Some(*zone))
                }
                TriggerCondition::Prepared { condition } => (
                    condition,
                    matches!(event, GameEvent::SpellCast { object_id, .. }
                        if self.stack_spell(*object_id).is_some_and(|spell| spell.from_prepared)),
                ),
                TriggerCondition::WhileSourceAttacking { condition } => {
                    (condition, permanent().is_some_and(|p| p.attacking))
                }
                TriggerCondition::ActiveIf {
                    active_if,
                    condition,
                } => (
                    condition,
                    self.check_intervening_if(active_if, controller, Some(object), None),
                ),
                TriggerCondition::OnceEachTurn { condition } => {
                    once = true;
                    (condition, true)
                }
                TriggerCondition::InGraveyard { .. } => return None,
                leaf => {
                    debug_assert!(
                        leaf.inner().is_none(),
                        "unhandled trigger wrapper: {leaf:?}"
                    );
                    return Some((leaf, once));
                }
            };
            if !enabled {
                return None;
            }
            condition = inner;
        }
    }

    fn trigger_condition_fires(
        &self,
        condition: &TriggerCondition,
        source_object: ObjectId,
        source_controller: PlayerId,
        source_was_on_battlefield: bool,
        event: &GameEvent,
    ) -> bool {
        match (condition, event) {
            (TriggerCondition::Sacrificed { subject }, GameEvent::PermanentSacrificed { .. }) => {
                self.subject_matches_moved_object(
                    subject,
                    source_object,
                    source_controller,
                    source_was_on_battlefield,
                    event,
                )
            }
            (
                TriggerCondition::EntersTheBattlefield { subject }
                | TriggerCondition::Dies { subject }
                | TriggerCondition::LeavesTheBattlefield { subject },
                GameEvent::CardMoved { from, to, .. },
            ) => {
                let matches_move = match condition {
                    TriggerCondition::EntersTheBattlefield { .. } => {
                        *to == ZoneType::Battlefield && *from != Some(ZoneType::Battlefield)
                    }
                    TriggerCondition::Dies { .. } => {
                        *from == Some(ZoneType::Battlefield) && *to == ZoneType::Graveyard
                    }
                    TriggerCondition::LeavesTheBattlefield { .. } => {
                        *from == Some(ZoneType::Battlefield) && *to != ZoneType::Battlefield
                    }
                    _ => unreachable!(),
                };
                matches_move
                    && self.subject_matches_moved_object(
                        subject,
                        source_object,
                        source_controller,
                        source_was_on_battlefield,
                        event,
                    )
            }
            (
                TriggerCondition::Expend(threshold),
                GameEvent::SpellCast {
                    caster,
                    spend_before,
                    spend_after,
                    ..
                },
            ) => {
                *caster == source_controller && spend_before < threshold && threshold <= spend_after
            }
            // `check_state_triggers` and `cast_triggers` own these; a
            // battlefield source never fires them from an event.
            (
                TriggerCondition::NoCounters { .. }
                | TriggerCondition::State(_)
                | TriggerCondition::YouCastThis,
                _,
            ) => false,
            (
                TriggerCondition::SpellTargetsOnlyThis { predicate },
                GameEvent::SpellCast {
                    object_id: object,
                    targets,
                    ..
                },
            ) => {
                self.object_matches_predicate(*object, predicate)
                    && self.state.permanent_of(source_object).is_some_and(|id| {
                        !targets.is_empty() && targets.iter().all(|t| *t == Target::Permanent(id))
                    })
            }
            (
                TriggerCondition::Attacks {
                    subject,
                    attackers: size,
                    nth,
                    ..
                },
                GameEvent::AttackersDeclared { attackers, .. },
            ) => {
                size.matches(attackers.len())
                    && attackers.iter().any(|&id| {
                        self.attack_matches(subject, *nth, source_object, source_controller, id)
                    })
            }
            (
                TriggerCondition::DealsCombatDamage {
                    subject,
                    to: Some(to),
                    ..
                },
                GameEvent::CombatDamageToPlayers { hits },
            ) => hits
                .iter()
                .any(|hit| self.hit_matches(hit, subject, *to, source_object, source_controller)),
            (
                TriggerCondition::DealsCombatDamage {
                    subject, to: None, ..
                },
                GameEvent::CombatDamageTotals { dealt },
            ) => dealt.iter().any(|(dealer, _)| {
                self.subject_matches_damage_source(
                    subject,
                    source_object,
                    source_controller,
                    *dealer,
                )
            }),
            (
                TriggerCondition::BecomesTapped { subject },
                GameEvent::PermanentTapped { permanent, .. },
            )
            | (
                TriggerCondition::BecomesUntapped { subject },
                GameEvent::PermanentUntapped { permanent },
            )
            | (
                TriggerCondition::TappedForMana { subject },
                GameEvent::PermanentTapped {
                    permanent,
                    for_mana: true,
                },
            ) => self.subject_matches_permanent(
                subject,
                source_object,
                source_controller,
                *permanent,
            ),
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
                        source_object,
                        source_controller,
                        *permanent,
                    )
            }
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
            (
                TriggerCondition::NthCounter { kind, nth },
                GameEvent::CountersPut {
                    permanent,
                    kind: put,
                    count,
                    total,
                    ..
                },
            ) if kind == put => {
                let threshold = i32::from(*nth);
                self.state.permanent_of(source_object) == Some(*permanent)
                    && total - count < threshold
                    && threshold <= *total
            }
            (
                TriggerCondition::Chapter { chapter },
                GameEvent::CountersPut {
                    permanent,
                    kind: CounterKind::Lore,
                    count,
                    total,
                    ..
                },
            ) => {
                let threshold = i32::from(*chapter);
                self.state.permanent_of(source_object) == Some(*permanent)
                    && total - count < threshold
                    && threshold <= *total
            }
            (
                TriggerCondition::PutsCounters { who, kind, on },
                GameEvent::CountersPut {
                    permanent,
                    kind: put,
                    by,
                    ..
                },
            ) => {
                put == kind
                    && who.includes(source_controller, *by)
                    && self.state.permanents[*permanent].is_some()
                    && self.permanent_matches_predicate(*permanent, on)
            }
            (
                TriggerCondition::PutsCountersOn { who, kind, subject },
                GameEvent::CountersPut {
                    permanent,
                    kind: put,
                    by,
                    ..
                },
            ) => {
                put == kind
                    && who.includes(source_controller, *by)
                    && self.state.permanents[*permanent].is_some()
                    && self.subject_matches_permanent(
                        subject,
                        source_object,
                        source_controller,
                        *permanent,
                    )
            }
            (
                TriggerCondition::Casts {
                    who,
                    predicate,
                    nth,
                    targets: filter,
                },
                GameEvent::SpellCast {
                    object_id: object,
                    caster,
                    targets,
                    ..
                },
            ) => {
                let caster = *caster;
                who.includes(source_controller, caster)
                    && self.object_matches_predicate(*object, predicate)
                    && match filter {
                        SpellTargetFilter::Any => true,
                        SpellTargetFilter::TargetsThis => {
                            let this = self.state.permanent_of(source_object);
                            this.is_some_and(|id| targets.contains(&Target::Permanent(id)))
                        }
                    }
                    && nth.is_none_or(|n| {
                        self.state.turn.spells_cast[caster.0]
                            .iter()
                            .position(|cast| cast.object == *object)
                            == Some(n as usize - 1)
                    })
            }
            (
                TriggerCondition::GainsLife { who, nth },
                GameEvent::LifeChanged {
                    player,
                    gain_ordinal,
                    ..
                },
            ) => {
                *gain_ordinal > 0
                    && who.includes(source_controller, *player)
                    && nth.is_none_or(|n| n == *gain_ordinal)
            }
            (
                TriggerCondition::DealsDamage { kind, source, to },
                GameEvent::DamageDealt {
                    source: Some(damage_source),
                    target,
                    combat,
                    ..
                },
            ) => {
                kind.includes(*combat)
                    && match (to, target) {
                        (DamageRecipientFilter::Any, _) => true,
                        (DamageRecipientFilter::Player(actor), DamageTarget::Player(player)) => {
                            actor.includes(source_controller, *player)
                        }
                        (DamageRecipientFilter::Player(_), _) => false,
                    }
                    && self.subject_matches_damage_source(
                        source,
                        source_object,
                        source_controller,
                        *damage_source,
                    )
            }
            (
                TriggerCondition::LosesLife { who },
                GameEvent::LifeChanged {
                    player, old, new, ..
                },
            ) => new < old && who.includes(source_controller, *player),
            (
                TriggerCondition::Draws { who, nth },
                GameEvent::CardDrawn {
                    player,
                    nth_this_turn,
                },
            ) => {
                who.includes(source_controller, *player) && nth.is_none_or(|n| n == *nth_this_turn)
            }
            (
                TriggerCondition::BeginningOf { step, whose },
                GameEvent::StepStarted { step: started },
            ) => {
                turn_step(*started) == Some(*step)
                    && whose.includes(source_controller, self.active_player())
            }
            (TriggerCondition::Performs { who, any_of }, _) => {
                performed_action(event).is_some_and(|(player, action)| {
                    who.includes(source_controller, player) && any_of.contains(&action)
                })
            }
            (
                TriggerCondition::PutIntoGraveyard { whose, predicate },
                GameEvent::CardMoved {
                    object_id: object,
                    to: ZoneType::Graveyard,
                    ..
                },
            ) => {
                let moved = &self.state[*object];
                moved.is_card()
                    && whose.includes(source_controller, moved.owner)
                    && self.object_matches_predicate(*object, predicate)
            }
            _ => false,
        }
    }

    /// The zone a cast spell was cast from, or a moved card came from.
    fn event_origin(&self, event: &GameEvent) -> Option<ZoneType> {
        match event {
            GameEvent::SpellCast { object_id, .. } => {
                self.stack_spell(*object_id).map(|spell| spell.cast_from)
            }
            GameEvent::CardMoved { from, .. } => *from,
            _ => None,
        }
    }

    fn attack_matches(
        &self,
        subject: &TriggerSubject,
        nth: Option<u32>,
        source: ObjectId,
        controller: PlayerId,
        attacker: PermanentId,
    ) -> bool {
        nth.is_none_or(|ordinal| {
            self.permanent_object(attacker)
                .is_some_and(|object| self.state.turn.attack_count(object) == ordinal)
        }) && self.subject_matches_permanent(subject, source, controller, attacker)
    }

    fn hit_matches(
        &self,
        hit: &CombatPlayerHit,
        subject: &TriggerSubject,
        to: Actor,
        source: ObjectId,
        controller: PlayerId,
    ) -> bool {
        to.includes(controller, hit.recipient)
            && self
                .state
                .permanent_of(hit.source)
                .is_some_and(|id| self.subject_matches_permanent(subject, source, controller, id))
    }

    fn subject_matches_moved_object(
        &self,
        subject: &TriggerSubject,
        source_object: ObjectId,
        source_controller: PlayerId,
        source_was_on_battlefield: bool,
        event: &GameEvent,
    ) -> bool {
        let (moved_object, moved_controller) = match event {
            GameEvent::CardMoved {
                object_id,
                controller,
                ..
            } => (*object_id, *controller),
            GameEvent::PermanentSacrificed {
                object_id,
                controller,
            } => (*object_id, *controller),
            _ => return false,
        };
        let departed = event.departed_object();
        let lki = departed.and_then(|object| self.object_lki(object));
        // A departed object is what it last was; an arriving one is what it
        // is now.
        let matches = |predicate: &crate::state::predicate::CardPredicate| match lki {
            Some(lki) => self.lki_matches_predicate(lki, predicate),
            None => self.object_matches_predicate(moved_object, predicate),
        };
        // The source is the moved object as it entered, or as it was before
        // it left (CR 603.10a looks back to the departed object).
        let is_source = moved_object == source_object || departed == Some(source_object);
        match subject {
            TriggerSubject::CreatureDamagedByThis => {
                let object = departed.unwrap_or(moved_object);
                source_was_on_battlefield
                    && self.state.turn.damage_pairs.contains(source_object, object)
                    && lki.map_or_else(
                        || {
                            self.lookup_current_permanent(object)
                                .is_ok_and(|id| self.characteristics(id).is_creature())
                        },
                        |lki| lki.was_creature(),
                    )
            }
            TriggerSubject::ThisOr(other) => {
                is_source
                    || self.subject_matches_moved_object(
                        other,
                        source_object,
                        source_controller,
                        source_was_on_battlefield,
                        event,
                    )
            }
            TriggerSubject::This => is_source,
            TriggerSubject::Matching {
                controlled_by,
                predicate,
                another,
            } => {
                source_was_on_battlefield
                    && controlled_by.includes(source_controller, moved_controller)
                    && !(*another && is_source)
                    && matches(predicate)
            }
            // The Aura is still attached to the departed object as the
            // event is matched (CR 603.10a).
            TriggerSubject::Attached => self
                .state
                .permanent_of(source_object)
                .and_then(|aura| self.state.permanents[aura].as_ref())
                .and_then(|aura| aura.attached_to)
                .is_some_and(|attached| {
                    Some(attached) == departed && self.state.object_lki.contains(attached)
                }),
        }
    }

    /// Whether the object that dealt damage is `subject`, as it is or last
    /// was: a live permanent, a departed one's last known information, or a
    /// spell on or off the stack. Its controller is
    /// [`Game::source_controller`]'s, so a stolen spell's damage is its
    /// controller's.
    fn subject_matches_damage_source(
        &self,
        subject: &TriggerSubject,
        source_object: ObjectId,
        source_controller: PlayerId,
        damage_source: ObjectId,
    ) -> bool {
        match subject {
            TriggerSubject::ThisOr(other) => {
                damage_source == source_object
                    || self.subject_matches_damage_source(
                        other,
                        source_object,
                        source_controller,
                        damage_source,
                    )
            }
            TriggerSubject::This => damage_source == source_object,
            TriggerSubject::Matching {
                controlled_by,
                predicate,
                another,
            } => {
                !(*another && damage_source == source_object)
                    && self
                        .source_controller(damage_source)
                        .is_some_and(|controller| {
                            controlled_by.includes(source_controller, controller)
                        })
                    && if let Some(permanent) = self.state.permanent_of(damage_source) {
                        self.permanent_matches_predicate(permanent, predicate)
                    } else if let Some(lki) = self.object_lki(damage_source) {
                        self.lki_matches_predicate(lki, predicate)
                    } else {
                        self.spell_matches_predicate(damage_source, predicate)
                    }
            }
            TriggerSubject::Attached | TriggerSubject::CreatureDamagedByThis => self
                .state
                .permanent_of(damage_source)
                .is_some_and(|permanent| {
                    self.subject_matches_permanent(
                        subject,
                        source_object,
                        source_controller,
                        permanent,
                    )
                }),
        }
    }

    fn subject_matches_permanent(
        &self,
        subject: &TriggerSubject,
        source_object: ObjectId,
        source_controller: PlayerId,
        permanent_id: PermanentId,
    ) -> bool {
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return false;
        };
        match subject {
            TriggerSubject::CreatureDamagedByThis => {
                self.state
                    .turn
                    .damage_pairs
                    .contains(source_object, permanent.object)
                    && self.characteristics(permanent_id).is_creature()
            }
            TriggerSubject::ThisOr(other) => {
                permanent.object == source_object
                    || self.subject_matches_permanent(
                        other,
                        source_object,
                        source_controller,
                        permanent_id,
                    )
            }
            TriggerSubject::This => permanent.object == source_object,
            TriggerSubject::Attached => {
                self.state
                    .permanent_of(source_object)
                    .and_then(|source| self.attached_permanent(source))
                    == Some(permanent_id)
            }
            TriggerSubject::Matching {
                controlled_by,
                predicate,
                another,
            } => {
                controlled_by.includes(source_controller, permanent.controller)
                    && !(*another && permanent.object == source_object)
                    && self.permanent_matches_predicate(permanent_id, predicate)
            }
        }
    }
}

#[cfg(test)]
#[path = "target_authority_tests.rs"]
mod target_authority_tests;
