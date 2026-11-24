// triggers.rs
// Triggered ability processing: event handling, trigger matching, and target selection.

use std::sync::Arc;

use crate::{
    agent::action::{Action, ActionSpace, ActionSpaceKind, AgentError},
    cards::content_pack::default_content_pack,
    flow::{
        event::{EventKind, GameEvent},
        game::Game,
        trigger::{
            DelayedTrigger, DelayedTriggerEvent, PendingTrigger, TriggerId, TriggerPlacement,
        },
        turn::{Bending, StepKind},
    },
    state::{
        ability::{
            Ability, AttackSize, DelayedFrequency, Effect, EffectValue, KeywordAction, Per,
            TargetRequirement, TargetSpec, TriggerCondition, TriggerSubject, TurnStep,
        },
        card::{CardDefId, PrintedCharacteristics},
        effect::{EffectDuration, Modification},
        game_object::{
            AbilitySource, ObjectId, ObjectKind, ObjectLki, PermanentId, PlayerId, Target,
        },
        mana::{Color, Mana},
        permanent::CounterKind,
        stack_object::{StackObject, TriggeredAbilityOnStack},
        zone::ZoneType,
    },
};

/// One occurrence of a matched triggered ability, before it waits to be
/// placed: a printed ability by index, or a granted one's effects.
#[derive(Clone)]
struct FiredTrigger {
    source: TriggerSourceSnapshot,
    ability_index: usize,
    granted: Option<Vec<Effect>>,
    program: bool,
}

/// Exact battlefield source facts used while matching one or more committed
/// events. SBA batches reuse one sorted snapshot so simultaneous departures
/// see the same pre-batch abilities (CR 603.6c, 704.3).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TriggerSourceSnapshot {
    pub(crate) controller: PlayerId,
    pub(crate) lki: ObjectLki,
}

impl serde::Serialize for TriggerSourceSnapshot {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        // V16 records the source slot under `card`; LKI owns the exact id.
        let mut snapshot = serializer.serialize_struct("TriggerSourceSnapshot", 3)?;
        snapshot.serialize_field("card", &self.lki.object)?;
        snapshot.serialize_field("controller", &self.controller)?;
        snapshot.serialize_field("lki", &self.lki)?;
        snapshot.end()
    }
}

/// A set of `EventKind`s, one bit per variant.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct EventKinds(u32);

impl EventKinds {
    const NONE: Self = Self(0);

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
    match condition {
        TriggerCondition::WasCast { condition }
        | TriggerCondition::OnceEachTurn { condition }
        | TriggerCondition::FromZone { condition, .. }
        | TriggerCondition::InGraveyard { condition }
        | TriggerCondition::WhileSourceAttacking { condition }
        | TriggerCondition::ActiveIf { condition, .. } => condition_event_kinds(condition),
        TriggerCondition::Sacrificed { .. } => EventKinds::of(EventKind::PermanentSacrificed),
        // Indexed with counter changes; `check_state_triggers` asks it, never
        // an event.
        TriggerCondition::NoCounters { .. } => EventKinds::of(EventKind::CountersPut),
        TriggerCondition::Casts { .. }
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
        TriggerCondition::CardsLeaveYourGraveyard => EventKinds::of(EventKind::CardsLeftGraveyard),
        TriggerCondition::SourceDealsDamage => EventKinds::of(EventKind::SourceDealtDamage),
        TriggerCondition::DealsCombatDamage { .. } => {
            EventKinds::of(EventKind::CombatDamageToPlayers)
        }
        TriggerCondition::BecomesTapped { .. } | TriggerCondition::TappedForMana { .. } => {
            EventKinds::of(EventKind::PermanentTapped)
        }
        TriggerCondition::BecomesTargeted { .. } => EventKinds::of(EventKind::PermanentTargeted),
        TriggerCondition::BeginningOf { .. } => EventKinds::of(EventKind::StepStarted),
        TriggerCondition::YourAttackingCreatureTriggers => {
            EventKinds::of(EventKind::AbilityTriggered)
        }
        TriggerCondition::GainsLife { .. } => EventKinds::of(EventKind::LifeChanged),
        TriggerCondition::Draws { .. } => EventKinds::of(EventKind::CardDrawn),
        TriggerCondition::NthCounter { .. }
        | TriggerCondition::PutsCounters { .. }
        | TriggerCondition::Chapter { .. } => EventKinds::of(EventKind::CountersPut),
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

fn ability_event_kinds(ability: &Ability) -> EventKinds {
    let Ability::Triggered { condition, .. } = ability;
    condition_event_kinds(condition)
}

/// A battlefield permanent and the event kinds its printed triggered
/// abilities can fire on.
#[derive(Debug, PartialEq, Eq)]
struct TriggerSourceEntry {
    permanent: PermanentId,
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
    /// A waiting trigger keeps the definition of the appearance that fired it.
    pub(crate) fn pending_trigger_definition(&self, trigger: &PendingTrigger) -> CardDefId {
        trigger.source_lki.as_ref().map_or_else(
            || self.state.source_base(trigger.source_object()).0,
            |lki| lki.definition_id,
        )
    }

    pub(crate) fn pending_trigger_face(
        &self,
        trigger: &PendingTrigger,
    ) -> Option<&PrintedCharacteristics> {
        self.source_face(
            trigger.source_object(),
            self.pending_trigger_definition(trigger),
        )
        .map(Arc::as_ref)
    }

    fn trigger_requirements(&self, trigger: &PendingTrigger) -> Option<Vec<TargetRequirement>> {
        let effects = match &trigger.inline_effects {
            Some(effects) => effects.as_slice(),
            None => self
                .pending_trigger_face(trigger)?
                .abilities
                .get(trigger.ability_index)?
                .effects(),
        };
        Some(crate::state::ability::effect_target_requirements(effects))
    }

    /// Distinctness and the announced X budget apply within one clause
    /// while choosing targets, including when replacing one chosen slot.
    pub(super) fn can_add_clause_target(
        &self,
        requirement: &TargetRequirement,
        x: u8,
        chosen: impl Iterator<Item = Target> + Clone,
        target: Target,
    ) -> bool {
        !chosen.clone().any(|previous| previous == target)
            && (!requirement.total_mana_value_x
                || self.target_mana_value(target)
                    + chosen
                        .map(|target| self.target_mana_value(target))
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
        if trigger.controller != player
            || !self.target_fits_clause(target, requirement, player, trigger.source_identity)
            || !self.can_add_clause_target(requirement, trigger.x, trigger.clause_targets(), target)
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
        if trigger.clause_targets().count() >= requirement.max {
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
        'placement: loop {
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
            if let Some(Effect::TriggerModes {
                modes,
                up_to,
                unchosen,
            }) = self.pending_mode_rule(&trigger)
            {
                let legal = self.legal_trigger_modes(&trigger, modes, *unchosen);
                let optional = up_to.is_some();
                if up_to.as_ref().is_some_and(|limit| {
                    let limit = usize::try_from(self.controller_value(limit, trigger.controller))
                        .unwrap_or(0);
                    legal.is_empty() || trigger.chosen_modes.len() >= limit
                }) {
                    trigger.inline_effects = Some(Self::chosen_mode_effects(&trigger, modes));
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
                            .chain(optional.then_some(Action::Decline { player: controller }))
                            .collect(),
                        focus: Vec::new(),
                    });
                }
            }
            let Some(requirements) = self.trigger_requirements(&trigger) else {
                continue;
            };
            while let Some(requirement) = requirements.get(trigger.requirement_index) {
                let count = trigger.clause_targets().count();
                let controller = trigger.controller;
                let mut actions: Vec<_> = self
                    .legal_clause_targets(requirement, controller, trigger.source_identity)
                    .filter(|target| {
                        self.can_add_clause_target(
                            requirement,
                            trigger.x,
                            trigger.clause_targets(),
                            *target,
                        )
                    })
                    .map(|target| Action::ChooseTarget {
                        player: controller,
                        target,
                    })
                    .collect();
                if actions.is_empty() {
                    if count < requirement.min {
                        continue 'placement;
                    }
                    trigger.requirement_index += 1;
                    continue;
                }
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
            self.place_triggered_ability_on_stack(trigger);
        }
    }

    /// The printed modal rule applies until the chosen effects have been bound.
    pub(crate) fn pending_mode_rule(&self, trigger: &PendingTrigger) -> Option<&Effect> {
        if trigger.inline_effects.is_some() {
            return None;
        }
        match self
            .pending_trigger_face(trigger)?
            .abilities
            .get(trigger.ability_index)?
            .effects()
        {
            [rule @ Effect::TriggerModes { .. }] => Some(rule),
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
        let source = trigger
            .source_ref()
            .and_then(|object| self.lookup_current_permanent(object).ok());
        let chosen_before = source
            .filter(|_| unchosen)
            .map(|id| self.chosen_modes(id))
            .unwrap_or_default();
        (0..modes.len())
            .filter(|mode| !trigger.chosen_modes.contains(mode) && !chosen_before.contains(mode))
            .filter(|mode| {
                crate::state::ability::effect_target_requirements(&modes[*mode])
                    .iter()
                    .filter(|requirement| requirement.min > 0)
                    .all(|requirement| {
                        self.legal_clause_targets(
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
            .state
            .trigger_placement
            .as_ref()
            .and_then(|p| p.placing.as_ref())
            .ok_or_else(|| AgentError("no pending trigger mode".into()))?;
        let Some(Effect::TriggerModes {
            modes,
            up_to,
            unchosen,
        }) = self.pending_mode_rule(trigger)
        else {
            return Err(AgentError("no pending trigger mode".into()));
        };
        if trigger.controller != player
            || !self
                .legal_trigger_modes(trigger, modes, *unchosen)
                .contains(&mode)
        {
            return Err(AgentError("illegal trigger mode".into()));
        }
        let effects = up_to.is_none().then(|| modes[mode].clone());
        let source = trigger.source_ref().filter(|_| *unchosen);
        let placing = self
            .state
            .trigger_placement
            .as_mut()
            .and_then(|p| p.placing.as_mut())
            .expect("validated trigger");
        if let Some(effects) = effects {
            placing.inline_effects = Some(effects);
            if let Some(permanent) =
                source.and_then(|object| self.lookup_current_permanent(object).ok())
            {
                self.apply_continuous(
                    source,
                    player,
                    self.affected_permanent(permanent),
                    Modification::ModeChosen {
                        mode: u8::try_from(mode).unwrap_or(u8::MAX),
                    },
                    EffectDuration::Indefinite,
                );
            }
        } else {
            placing.chosen_modes.push(mode);
        }
        Ok(())
    }

    /// The effects of the placing trigger's chosen modes, in printed order.
    fn chosen_mode_effects(trigger: &PendingTrigger, modes: &[Vec<Effect>]) -> Vec<Effect> {
        modes
            .iter()
            .enumerate()
            .filter(|(index, _)| trigger.chosen_modes.contains(index))
            .flat_map(|(_, effects)| effects.iter().cloned())
            .collect()
    }

    pub(crate) fn decline_trigger_target(&mut self, player: PlayerId) -> Result<(), AgentError> {
        let trigger = self
            .state
            .trigger_placement
            .as_ref()
            .and_then(|p| p.placing.as_ref())
            .ok_or_else(|| AgentError("no pending trigger".into()))?;
        if let Some(Effect::TriggerModes {
            up_to: Some(_),
            modes,
            ..
        }) = self.pending_mode_rule(trigger)
        {
            if trigger.controller != player {
                return Err(AgentError("not the trigger's controller".into()));
            }
            let effects = Self::chosen_mode_effects(trigger, modes);
            self.state
                .trigger_placement
                .as_mut()
                .and_then(|p| p.placing.as_mut())
                .expect("validated trigger")
                .inline_effects = Some(effects);
            return Ok(());
        }
        let requirements = self
            .trigger_requirements(trigger)
            .ok_or_else(|| AgentError("missing triggered ability".into()))?;
        let requirement = requirements
            .get(trigger.requirement_index)
            .ok_or_else(|| AgentError("no pending target requirement".into()))?;
        let count = trigger.clause_targets().count();
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
        let source_definition_id = self.pending_trigger_definition(&trigger);
        let source_face = self
            .source_face(trigger.source_object(), source_definition_id)
            .expect("placing trigger has a source face");
        let source_face = (!default_content_pack()
            .definition(source_definition_id)
            .is_some_and(|row| Arc::ptr_eq(source_face, &row.printed)))
        .then(|| source_face.clone());
        let id = self.state.id_gen_mut().next_id();
        let targets = trigger.targets;
        self.emit_permanents_targeted(&targets, Target::StackAbility(id), trigger.controller);
        self.push_to_stack(StackObject::TriggeredAbility(TriggeredAbilityOnStack {
            source_face,
            trigger_damage: trigger.trigger_damage,
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
            kicked: trigger.kicked,
            x: trigger.x,
            spent: trigger.spent,
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

    /// All currently legal targets for a spec, from `controller`'s
    /// perspective ("you").
    pub(crate) fn legal_targets_for_spec(
        &self,
        target_spec: &TargetSpec,
        controller: PlayerId,
    ) -> Vec<Target> {
        let mut candidates = self.target_candidates(target_spec, controller);
        candidates.retain(|target| self.target_is_legal(*target, target_spec, controller));
        candidates
    }

    pub(super) fn legal_clause_targets<'a>(
        &'a self,
        requirement: &'a TargetRequirement,
        controller: PlayerId,
        source: AbilitySource,
    ) -> impl Iterator<Item = Target> + 'a {
        self.target_candidates(&requirement.spec, controller)
            .into_iter()
            .filter(move |target| self.target_fits_clause(*target, requirement, controller, source))
    }

    fn target_candidates(&self, target_spec: &TargetSpec, controller: PlayerId) -> Vec<Target> {
        let graveyard = |owner| {
            self.state
                .zones
                .zone_objects(ZoneType::Graveyard, owner)
                .iter()
                .copied()
                .map(Target::GraveyardCard)
        };
        let mut candidates: Vec<Target> = match target_spec {
            TargetSpec::GraveyardCardYouOwn { .. }
            | TargetSpec::GraveyardCardYouOwnWithinSourcePower { .. } => {
                graveyard(controller).collect()
            }
            TargetSpec::GraveyardCardOpponentOwns { .. } => {
                graveyard(PlayerId(1 - controller.0)).collect()
            }
            TargetSpec::GraveyardCard { .. } => [PlayerId(0), PlayerId(1)]
                .into_iter()
                .flat_map(graveyard)
                .collect(),
            TargetSpec::Player | TargetSpec::Opponent | TargetSpec::AnyTarget => {
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
        while !self.state.pending_events.is_empty() {
            let events = std::mem::take(self.state.pending_events_mut());
            for event in events {
                let sources = self.trigger_ability_sources(&event);
                self.check_triggers_for_event(&event, &sources);
            }
        }
    }

    pub(super) fn graveyard_departure_sources(&self) -> Vec<TriggerSourceSnapshot> {
        self.indexed_sources(EventKinds::of(EventKind::CardsLeftGraveyard))
    }

    /// Departures look back to the whole pre-batch battlefield. Entry and
    /// counter events see the completed battlefield, including every entrant.
    pub(crate) fn process_event_batch(
        &mut self,
        before: &[TriggerSourceSnapshot],
        graveyard_sources: &[TriggerSourceSnapshot],
    ) {
        while !self.state.pending_events.is_empty() {
            let events = std::mem::take(self.state.pending_events_mut());
            self.check_source_damage_triggers(&events);
            if !graveyard_sources.is_empty() {
                let mut departed = [false; 2];
                for event in &events {
                    if let GameEvent::CardMoved {
                        object_id,
                        from: Some(ZoneType::Graveyard),
                        ..
                    } = event
                    {
                        if let Some((_, owner, ObjectKind::Card)) =
                            self.state.objects.provenance(*object_id)
                        {
                            departed[owner.0] = true;
                        }
                    }
                }
                for (seat, _) in departed.into_iter().enumerate().filter(|(_, left)| *left) {
                    self.check_triggers_for_event(
                        &GameEvent::CardsLeftGraveyard {
                            owner: PlayerId(seat),
                        },
                        graveyard_sources,
                    );
                }
            }
            for event in events {
                if matches!(
                    event,
                    GameEvent::CardMoved {
                        from: Some(ZoneType::Battlefield),
                        ..
                    } | GameEvent::PermanentSacrificed { .. }
                ) {
                    self.check_triggers_for_event(&event, before);
                } else {
                    let sources = self.trigger_ability_sources(&event);
                    self.check_triggers_for_event(&event, &sources);
                }
            }
        }
    }

    /// “Whenever this deals damage” counts one hit across simultaneous recipients.
    /// Per-recipient events remain available to history and recipient-specific rules.
    fn check_source_damage_triggers(&mut self, events: &[GameEvent]) {
        if !events
            .iter()
            .any(|event| matches!(event, GameEvent::DamageDealt { .. }))
        {
            return;
        }
        let sources = self.indexed_sources(EventKinds::of(EventKind::SourceDealtDamage));
        if sources.is_empty() {
            return;
        }
        let mut totals = std::collections::BTreeMap::<ObjectId, i32>::new();
        for event in events {
            if let GameEvent::DamageDealt {
                source: Some(source),
                amount,
                ..
            } = event
            {
                let total = totals.entry(*source).or_default();
                *total = total.saturating_add(*amount as i32);
            }
        }
        for source in sources {
            if let Some(&amount) = totals.get(&source.lki.object) {
                let event = GameEvent::SourceDealtDamage {
                    source: source.lki.object,
                    amount,
                };
                self.check_triggers_for_event(&event, std::slice::from_ref(&source));
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

    /// Resolve printed and granted trigger matches before committing the event.
    fn firing_battlefield_triggers(
        &self,
        event: &GameEvent,
        sources: &[TriggerSourceSnapshot],
    ) -> Vec<FiredTrigger> {
        let mut fired: Vec<FiredTrigger> = Vec::new();
        for source in sources {
            for (ability_index, ability) in self
                .source_face(source.lki.object, source.lki.definition_id)
                .expect("source face")
                .abilities
                .iter()
                .enumerate()
                .filter(|_| !source.lki.abilities_lost)
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
                        Some(source.lki.object),
                        self.triggering_permanent(event),
                    )
                }) && (!matches!(condition, TriggerCondition::OnceEachTurn { .. })
                    || !self
                        .state
                        .turn
                        .ability_triggers_this_turn
                        .contains_key(&(source.lki.object, ability_index)))
                {
                    let times = self.trigger_occurrences(condition, source, event);
                    fired.extend(std::iter::repeat_n(
                        FiredTrigger {
                            source: source.clone(),
                            ability_index,
                            granted: None,
                            program: false,
                        },
                        times,
                    ));
                }
            }
            // Departure triggers use the grants captured at the event boundary.
            for effective in &source.lki.granted_abilities {
                let Ability::Triggered {
                    condition,
                    effects,
                    intervening_if,
                } = &effective.ability;
                if intervening_if.as_ref().is_some_and(|condition| {
                    !self.check_intervening_if(
                        condition,
                        source.controller,
                        Some(source.lki.object),
                        self.triggering_permanent(event),
                    )
                }) {
                    continue;
                }
                // Keep CR 603.4's second check in the captured program when
                // there is no printed ability to recover at resolution.
                let effects = if let Some(condition) = intervening_if {
                    vec![Effect::If {
                        condition: condition.clone(),
                        then: effects.clone(),
                        otherwise: Vec::new(),
                    }]
                } else {
                    effects.clone()
                };
                let times = self.trigger_occurrences(condition, source, event);
                fired.extend(std::iter::repeat_n(
                    FiredTrigger {
                        source: source.clone(),
                        ability_index: 0,
                        granted: Some(effects),
                        program: effective.program,
                    },
                    times,
                ));
            }
        }

        fired
    }

    fn check_triggers_for_event(&mut self, event: &GameEvent, sources: &[TriggerSourceSnapshot]) {
        #[cfg(feature = "benchmark")]
        crate::benchmark_layers::record(|m| m.trigger_sources += sources.len() as u64);
        self.granted_firebending_triggers(event);
        self.graveyard_triggers(event);
        self.cast_triggers(event);
        self.delayed_triggers_for_event(event);
        let fired = self.firing_battlefield_triggers(event, sources);

        #[cfg(feature = "benchmark")]
        crate::benchmark_layers::record(|m| m.triggers_found += fired.len() as u64);
        for FiredTrigger {
            source,
            ability_index,
            granted,
            program,
        } in fired
        {
            let TriggerSourceSnapshot {
                controller,
                lki: source_lki,
            } = source;
            let source_object = source_lki.object;
            if granted.is_none()
                && matches!(
                    &self
                        .source_face(source_object, source_lki.definition_id)
                        .expect("source face")
                        .abilities[ability_index],
                    Ability::Triggered {
                        condition: TriggerCondition::OnceEachTurn { .. },
                        ..
                    }
                )
            {
                self.spend_once_each_turn((source_lki.object, ability_index));
            }
            let effects = match &granted {
                Some(effects) => effects.as_slice(),
                None => self
                    .source_face(source_object, source_lki.definition_id)
                    .expect("source face")
                    .abilities[ability_index]
                    .effects(),
            };
            let reads_triggering_permanent = effects.iter().any(Effect::reads_triggering_permanent);
            let asks_triggering_spell = effects.iter().any(Effect::asks_triggering_spell);
            let context = if reads_triggering_permanent {
                self.triggering_permanent(event)
            } else if let (
                true,
                GameEvent::SpellCast {
                    object_id: object, ..
                },
            ) = (asks_triggering_spell, event)
            {
                Some(Target::StackSpell(*object))
            } else {
                self.trigger_context_for_event(event)
            };
            let player = match event {
                GameEvent::SpellCast {
                    object_id: object, ..
                } => Some(self.spell_caster(*object)),
                GameEvent::CardDrawn { player, .. } | GameEvent::LifeChanged { player, .. } => {
                    Some(*player)
                }
                // "that player": whose graveyard the card was put into.
                GameEvent::CardMoved {
                    object_id: object,
                    to: ZoneType::Graveyard,
                    ..
                } => Some(self.state[*object].owner),
                // "that player": whose step it is.
                GameEvent::StepStarted { .. } => Some(self.active_player()),
                // "that player": who searched, scried, surveilled or bent.
                _ => performed_action(event).map(|(player, _)| player),
            };
            let attacker = match event {
                GameEvent::AttackersDeclared { attackers, .. } => attackers
                    .iter()
                    .copied()
                    .find(|id| self.permanent_object(*id) == Some(source_lki.object)),
                _ => None,
            };
            let id = self.next_trigger_id();
            let mut trigger = PendingTrigger {
                trigger_damage: match event {
                    GameEvent::SourceDealtDamage { amount, .. } => *amount,
                    _ => 0,
                },
                source_lki: Some(source_lki),
                context,
                player,
                attacker,
                inline_effects: granted,
                program,
                ..PendingTrigger::new(
                    id,
                    controller,
                    AbilitySource::new(source_object, Some(source_object)),
                    ability_index,
                )
            };
            match event {
                // A counters trigger's "that much" is the number just put.
                GameEvent::CountersPut { count, .. } => {
                    trigger.x = u8::try_from(*count).unwrap_or(u8::MAX);
                }
                // CR 107.3m, 400.7d: an entering permanent's own trigger
                // remembers the choices and payment from its spell.
                GameEvent::CardMoved {
                    object_id,
                    to: ZoneType::Battlefield,
                    ..
                } if *object_id == source_object => {
                    if let Some(spell) = self.state.spell_entry(source_object) {
                        trigger.kicked = spell.kicked;
                        trigger.x = spell.x;
                        trigger.spent = spell.payment.clone().unwrap_or_default();
                    }
                }
                _ => {}
            }
            self.state.pending_triggers_mut().push(trigger);
        }
    }

    /// How many times one of `source`'s triggered abilities triggers on
    /// `event`: zero unless its condition fires, else its own multiplicity
    /// plus the "triggers an additional time" statics that cover it.
    fn trigger_occurrences(
        &self,
        condition: &TriggerCondition,
        source: &TriggerSourceSnapshot,
        event: &GameEvent,
    ) -> usize {
        if !self.trigger_condition_fires(
            condition,
            source.lki.object,
            source.controller,
            true,
            event,
        ) {
            return 0;
        }
        debug_assert!(
            condition_event_kinds(condition).contains(event.kind()),
            "{condition:?} fired on an event kind it is not indexed by: {event:?}"
        );
        self.trigger_multiplicity(condition, source.lki.object, source.controller, event)
            + self.additional_trigger_times(source)
    }

    /// The permanent whose event fired a trigger: an entering one, a
    /// departed one (by the slot it occupied, read from its last known
    /// information) or a lone attacker.
    fn triggering_permanent(&self, event: &GameEvent) -> Option<Target> {
        match event {
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
            GameEvent::AttackersDeclared { attackers, .. } if attackers.len() == 1 => {
                Some(Target::Permanent(attackers[0]))
            }
            _ => None,
        }
    }

    /// Extra occurrences of one of `source`'s triggered abilities: one for
    /// each permanent its controller controls whose "triggers an additional
    /// time" covers the source (CR 603.2d), as the source is or last was.
    fn additional_trigger_times(&self, source: &TriggerSourceSnapshot) -> usize {
        self.static_sources()
            .filter(|(_, controller, _)| *controller == source.controller)
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
                    .filter(|predicate| super::layers::matches_departure(&source.lki, predicate))
                    .count()
            })
            .sum()
    }

    /// Granted firebending (CR 702.189): each attacking creature with
    /// firebending until end of turn triggers "add that much {R} until end
    /// of combat".
    fn granted_firebending_triggers(&mut self, event: &GameEvent) {
        let GameEvent::AttackersDeclared { attackers, .. } = event else {
            return;
        };
        for &attacker in attackers {
            let amount = self.characteristics(attacker).firebending;
            let Some(permanent) = self.state.permanents[attacker].as_ref() else {
                continue;
            };
            let controller = permanent.controller;
            if amount == 0 {
                continue;
            }
            let Some(source) = self.permanent_object(attacker) else {
                continue;
            };
            let mut mana = Mana::default();
            mana.mana[Color::Red as usize] = amount;
            let effect = Effect::AddMana {
                mana,
                amount: EffectValue::Fixed(1),
                until_end_of_combat: true,
            };
            self.enqueue_inline_trigger(
                AbilitySource::new(source, Some(source)),
                controller,
                vec![effect],
                None,
            )
            .attacker = Some(attacker);
        }
    }

    /// Wait for a delayed triggered ability's event (CR 603.7).
    pub(crate) fn add_delayed_trigger(&mut self, trigger: DelayedTrigger) {
        self.state.delayed_triggers_mut().push(trigger);
    }

    /// Match ordinary events against waiting watchers. `Leaves` matches at
    /// the exact object's battlefield departure instead.
    /// One-shot triggers are removed as they fire (CR 603.7c).
    fn delayed_triggers_for_event(&mut self, event: &GameEvent) {
        let dead_creature = match event {
            GameEvent::CardMoved {
                from: Some(ZoneType::Battlefield),
                to: ZoneType::Graveyard,
                ..
            } => event
                .departed_object()
                .and_then(|object| self.object_lki(object))
                .filter(|lki| lki.battlefield.types.is_creature())
                .map(|lki| Target::Permanent(lki.permanent)),
            _ => None,
        };
        let mut fired = Vec::new();
        let mut spent = Vec::new();
        for (index, trigger) in self.state.delayed_triggers.iter().enumerate() {
            let context = match (&trigger.event, event) {
                (DelayedTriggerEvent::CreatureDies, _) if dead_creature.is_some() => dead_creature,
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
                        object_id: object, ..
                    },
                ) if self.spell_caster(*object) == trigger.controller
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
        let pending = self.enqueue_inline_trigger(
            trigger.source_identity,
            trigger.controller,
            trigger.effects.clone(),
            context,
        );
        pending.remembered = trigger.remembered.clone();
        pending.program = true;
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
    /// reference: a delayed, reflexive or granted-firebending ability) in the
    /// waiting list, with its source's last known information when that was
    /// a battlefield object.
    pub(crate) fn enqueue_inline_trigger(
        &mut self,
        source: AbilitySource,
        controller: PlayerId,
        effects: Vec<Effect>,
        context: Option<Target>,
    ) -> &mut PendingTrigger {
        let source_lki = source.reference().and_then(|object| {
            self.object_lki(object)
                .cloned()
                .or_else(|| self.snapshot_current_permanent(object))
        });
        let id = self.next_trigger_id();
        self.state.pending_triggers_mut().push(PendingTrigger {
            source_lki,
            context,
            inline_effects: Some(effects),
            ..PendingTrigger::new(id, controller, source, 0)
        });
        self.state
            .pending_triggers_mut()
            .last_mut()
            .expect("enqueued trigger")
    }

    /// State triggers (CR 603.8) of battlefield sources whose state holds,
    /// unless that exact source's occurrence is still waiting or on the stack.
    fn check_state_triggers(&mut self) {
        let mut fired = Vec::new();
        for source in self.indexed_sources(EventKinds::of(EventKind::CountersPut)) {
            let Some(permanent) = self
                .state
                .permanent_of(source.lki.object)
                .and_then(|id| self.state.permanents[id].as_ref())
            else {
                continue;
            };
            let face = self
                .source_face(source.lki.object, source.lki.definition_id)
                .expect("source face");
            for (ability_index, ability) in face.abilities.iter().enumerate() {
                let Ability::Triggered {
                    condition: TriggerCondition::NoCounters { kind },
                    ..
                } = ability
                else {
                    continue;
                };
                let object = Some(source.lki.object);
                let waiting = self.state.pending_triggers.iter().any(|trigger| {
                    trigger.source_ref() == object && trigger.ability_index == ability_index
                }) || self.state.stack_objects.iter().any(|item| {
                    matches!(item, StackObject::TriggeredAbility(trigger)
                        if trigger.source_ref() == object && trigger.ability_index == ability_index)
                });
                if !source.lki.abilities_lost && permanent.counter_count(*kind) == 0 && !waiting {
                    fired.push((source.clone(), ability_index));
                }
            }
        }
        for (source, ability_index) in fired {
            let id = self.next_trigger_id();
            let object = source.lki.object;
            self.state.pending_triggers_mut().push(PendingTrigger {
                source_lki: Some(source.lki),
                ..PendingTrigger::new(
                    id,
                    source.controller,
                    AbilitySource::new(object, Some(object)),
                    ability_index,
                )
            });
        }
    }

    /// "When you cast this spell" abilities of the spell just cast; the spell
    /// is the source and its caster controls the trigger.
    fn cast_triggers(&mut self, event: &GameEvent) {
        let GameEvent::SpellCast {
            object_id: object, ..
        } = event
        else {
            return;
        };
        let object = *object;
        let caster = self.spell_caster(object);
        let fired: Vec<Vec<Effect>> = self.state[object]
            .abilities
            .iter()
            .filter(|ability| {
                let Ability::Triggered { condition, .. } = ability;
                *condition == TriggerCondition::YouCastThis
            })
            .map(|ability| ability.effects().to_vec())
            .collect();
        let source = AbilitySource::new(object, self.state.live(object));
        for effects in fired {
            self.enqueue_inline_trigger(source, caster, effects, None);
        }
        let Some(spell) = self.spell_on_stack(object) else {
            return;
        };
        let replicated = if self.state[object].replicate().is_some() {
            usize::from(spell.x)
        } else {
            0
        };
        // "When that mana is spent to cast [a spell], copy that spell": one
        // trigger of the object that made each such unit.
        let copy_sources: Vec<_> = spell.spent.copy_triggers().flatten().collect();
        for origin in copy_sources {
            self.enqueue_inline_trigger(
                AbilitySource::new(origin.object, Some(origin.object)),
                caster,
                vec![Effect::CopyTriggeringObject],
                Some(Target::StackSpell(object)),
            );
        }
        // CR 702.56a: replicate copies the spell once per payment, each copy
        // with a chance to choose new targets.
        if replicated > 0 {
            let copies = vec![Effect::CopyTriggeringObject; replicated];
            self.enqueue_inline_trigger(source, caster, copies, Some(Target::StackSpell(object)));
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
                    if self.trigger_condition_fires(condition, object, owner, false, event) {
                        fired.push((object, ability_index, owner));
                    }
                }
            }
        }
        for (object, ability_index, owner) in fired {
            let source_ref = self.state.live(object);
            let id = self.next_trigger_id();
            self.state.pending_triggers_mut().push(PendingTrigger::new(
                id,
                owner,
                AbilitySource::new(object, source_ref),
                ability_index,
            ));
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
        let mut sources: Vec<_> = index
            .sources
            .iter()
            .filter(|entry| entry.kinds.intersects(kinds))
            .filter_map(|entry| {
                let permanent = self.state.permanents[entry.permanent].as_ref()?;
                self.snapshot_current_permanent(entry.object)
                    .map(|lki| TriggerSourceSnapshot {
                        controller: permanent.controller,
                        lki,
                    })
            })
            .collect();
        sources.sort_by_key(|source| source.lki.object);
        sources
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
                    permanent: id,
                    object: permanent.object,
                    kinds: view
                        .triggered_abilities
                        .iter()
                        .fold(EventKinds::NONE, |kinds, ability| {
                            kinds | ability_event_kinds(&ability.ability)
                        }),
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
            controller,
            ..
        } = event
        {
            if let Some(lki) = event
                .departed_object()
                .and_then(|object| self.object_lki(object))
            {
                sources.push(TriggerSourceSnapshot {
                    controller: *controller,
                    lki: lki.clone(),
                });
            }
        }

        sources.sort_by_key(|source| source.lki.object);
        sources.dedup_by_key(|source| source.lki.object);
        sources
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
            (TriggerCondition::WasCast { condition }, _) => {
                self.state
                    .spell_entry(source_object)
                    .is_some_and(|entry| entry.payment.is_some())
                    && self.trigger_condition_fires(
                        condition,
                        source_object,
                        source_controller,
                        source_was_on_battlefield,
                        event,
                    )
            }
            (TriggerCondition::FromZone { zone, condition }, _) => {
                self.event_origin(event) == Some(*zone)
                    && self.trigger_condition_fires(
                        condition,
                        source_object,
                        source_controller,
                        source_was_on_battlefield,
                        event,
                    )
            }
            (TriggerCondition::OnceEachTurn { condition }, _) => self.trigger_condition_fires(
                condition,
                source_object,
                source_controller,
                source_was_on_battlefield,
                event,
            ),
            (TriggerCondition::Sacrificed { subject }, GameEvent::PermanentSacrificed { .. }) => {
                self.subject_matches_moved_object(
                    subject,
                    source_object,
                    source_controller,
                    source_was_on_battlefield,
                    event,
                )
            }
            (TriggerCondition::WhileSourceAttacking { condition }, _) => {
                self.state
                    .permanent_of(source_object)
                    .and_then(|id| self.state.permanents[id].as_ref())
                    .is_some_and(|source| source.attacking)
                    && self.trigger_condition_fires(
                        condition,
                        source_object,
                        source_controller,
                        source_was_on_battlefield,
                        event,
                    )
            }
            // `check_state_triggers` and `cast_triggers` own these; a
            // battlefield source never fires them from an event.
            (TriggerCondition::NoCounters { .. } | TriggerCondition::YouCastThis, _) => false,
            (
                TriggerCondition::SpellTargetsOnlyThis { predicate },
                GameEvent::SpellCast {
                    object_id: object, ..
                },
            ) => {
                let Some(id) = self.state.permanent_of(source_object) else {
                    return false;
                };
                self.object_matches_predicate(*object, predicate)
                    && self.spell_on_stack(*object).is_some_and(|spell| {
                        !spell.targets.is_empty()
                            && spell
                                .targets
                                .iter()
                                .all(|target| *target == Target::Permanent(id))
                    })
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
                // "While saddled" asks about the exact source.
                self.check_intervening_if(active_if, source_controller, Some(source_object), None)
                    && self.trigger_condition_fires(
                        condition,
                        source_object,
                        source_controller,
                        source_was_on_battlefield,
                        event,
                    )
            }
            (
                TriggerCondition::EntersTheBattlefield { subject },
                GameEvent::CardMoved { from, to, .. },
            ) => {
                let entered = *to == ZoneType::Battlefield && *from != Some(ZoneType::Battlefield);
                entered
                    && self.subject_matches_moved_object(
                        subject,
                        source_object,
                        source_controller,
                        source_was_on_battlefield,
                        event,
                    )
            }
            (
                TriggerCondition::CardsLeaveYourGraveyard,
                GameEvent::CardsLeftGraveyard { owner },
            ) => source_controller == *owner,
            (TriggerCondition::SourceDealsDamage, GameEvent::SourceDealtDamage { source, .. }) => {
                *source == source_object
            }
            (TriggerCondition::Dies { subject }, GameEvent::CardMoved { from, to, .. }) => {
                let died = *from == Some(ZoneType::Battlefield) && *to == ZoneType::Graveyard;
                died && self.subject_matches_moved_object(
                    subject,
                    source_object,
                    source_controller,
                    source_was_on_battlefield,
                    event,
                )
            }
            (
                TriggerCondition::LeavesTheBattlefield { subject },
                GameEvent::CardMoved {
                    from: Some(ZoneType::Battlefield),
                    to,
                    ..
                },
            ) if *to != ZoneType::Battlefield => self.subject_matches_moved_object(
                subject,
                source_object,
                source_controller,
                source_was_on_battlefield,
                event,
            ),
            (TriggerCondition::Attacks { .. }, GameEvent::AttackersDeclared { .. })
            | (
                TriggerCondition::DealsCombatDamage { .. },
                GameEvent::CombatDamageToPlayers { .. },
            ) => self.trigger_multiplicity(condition, source_object, source_controller, event) > 0,
            (
                TriggerCondition::BecomesTapped { subject },
                GameEvent::PermanentTapped { permanent, .. },
            ) => self.subject_matches_permanent(
                subject,
                source_object,
                source_controller,
                *permanent,
            ),
            (
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
            ) => {
                let threshold = i32::from(*nth);
                kind == put
                    && self.state.permanent_of(source_object) == Some(*permanent)
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
                let chapter = i32::from(*chapter);
                self.state.permanent_of(source_object) == Some(*permanent)
                    && total - count < chapter
                    && chapter <= *total
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
                TriggerCondition::Casts {
                    who,
                    predicate,
                    nth,
                },
                GameEvent::SpellCast {
                    object_id: object, ..
                },
            ) => {
                let caster = self.spell_caster(*object);
                who.includes(source_controller, caster)
                    && self.object_matches_predicate(*object, predicate)
                    && nth.is_none_or(|n| {
                        self.state.turn.spells_cast[caster.0]
                            .iter()
                            .position(|cast| cast == object)
                            == Some(n as usize - 1)
                    })
            }
            (TriggerCondition::GainsLife { who }, GameEvent::LifeChanged { player, old, new }) => {
                new > old && who.includes(source_controller, *player)
            }
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
                self.state
                    .stack_objects
                    .iter()
                    .find_map(|object| match object {
                        StackObject::Spell(spell) if spell.object == *object_id => {
                            Some(spell.cast_from)
                        }
                        _ => None,
                    })
            }
            GameEvent::CardMoved { from, .. } => *from,
            _ => None,
        }
    }

    /// How many times one event triggers a matching ability: once, except
    /// attacks and combat damage `per` object, which trigger for each
    /// matching creature (CR 603.2c).
    fn trigger_multiplicity(
        &self,
        condition: &TriggerCondition,
        source_object: ObjectId,
        source_controller: PlayerId,
        event: &GameEvent,
    ) -> usize {
        let (subject, per, creatures) = match (condition, event) {
            (
                TriggerCondition::DealsCombatDamage { subject, to, per },
                GameEvent::CombatDamageToPlayers { creatures },
            ) => {
                // In the two-seat combat model only the defending player, the
                // active player's opponent, receives combat damage.
                let defender = PlayerId(1 - self.active_player().0);
                if !to.includes(source_controller, defender) {
                    return 0;
                }
                (subject, per, creatures)
            }
            (
                TriggerCondition::Attacks {
                    subject,
                    per,
                    attackers: size,
                },
                GameEvent::AttackersDeclared { attackers, .. },
            ) => {
                let size_holds = match size {
                    AttackSize::Any => true,
                    AttackSize::Alone => attackers.len() == 1,
                    AttackSize::WithAnother => attackers.len() >= 2,
                };
                if !size_holds {
                    return 0;
                }
                (subject, per, attackers)
            }
            _ => return 1,
        };
        let matching = creatures
            .iter()
            .filter(|creature| {
                self.subject_matches_permanent(
                    subject,
                    source_object,
                    source_controller,
                    **creature,
                )
            })
            .count();
        match per {
            Per::Object => matching,
            Per::Batch => usize::from(matching > 0),
        }
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
        // The source is the moved object as it entered, or as it was before
        // it left (CR 603.10a looks back to the departed object).
        let is_source = moved_object == source_object || departed == Some(source_object);
        match subject {
            TriggerSubject::This => is_source,
            TriggerSubject::Matching {
                controlled_by,
                predicate,
                another,
            } => {
                source_was_on_battlefield
                    && controlled_by.includes(source_controller, moved_controller)
                    && !(*another && is_source)
                    && lki.map_or_else(
                        || self.object_matches_predicate(moved_object, predicate),
                        |lki| super::layers::matches_departure(lki, predicate),
                    )
            }
            // The Aura is still attached to the departed object as the
            // event is matched (CR 603.10a).
            TriggerSubject::EnchantedCreature => self
                .state
                .permanent_of(source_object)
                .and_then(|aura| self.state.permanents[aura].as_ref())
                .and_then(|aura| aura.attached_to)
                .or_else(|| self.object_lki(source_object)?.attached_to)
                .is_some_and(|attached| {
                    Some(attached) == departed && self.state.object_lki.contains(attached)
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
            TriggerSubject::This => permanent.object == source_object,
            TriggerSubject::EnchantedCreature => self.is_enchanted_by(permanent_id, source_object),
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

    /// Whether the Aura card `aura` is on the battlefield attached to the
    /// permanent.
    fn is_enchanted_by(&self, permanent_id: PermanentId, aura: ObjectId) -> bool {
        self.state
            .permanent_of(aura)
            .and_then(|aura| self.attached_permanent(aura))
            == Some(permanent_id)
    }
}

#[cfg(test)]
#[path = "target_authority_tests.rs"]
mod target_authority_tests;
