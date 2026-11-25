// resolution.rs
// Resolving stack objects. Effects execute through an EffectFrame so a
// resolution can suspend on a mid-resolution decision (see decision.rs).

use crate::{
    cards::content_pack::default_content_pack,
    flow::{
        cast_permission::PlayGrant,
        decision::{
            Decision, EffectFrame, FrameCompletion, ResolutionCastOffer, ResolutionTarget,
            TargetIteration,
        },
        event::GameEvent,
        game::Game,
        payment::{CostComponent, PaymentProgress},
        proposed_event::{
            EventBatch, EventBinding, ProposedDamageTarget, ProposedEvent, TokenSource, ZoneEntry,
        },
        trigger::{DelayedTrigger, DelayedTriggerEvent},
        turn::OnceEachTurnKey,
    },
    state::{
        ability::{
            Ability, DelayedEvent, DelayedExpiry, DelayedFrequency, Effect, EffectValue, Except,
            LinkedChooser, LinkedExile, ObjectProperty, ObjectSelector, PlayerSelector, Scope,
            TargetRequirement, TargetSpec,
        },
        card::{
            CardDefId, CardTypes, Keyword, PrintedCharacteristics, SourceQuality, StaticAbility,
        },
        effect::{Affected, EffectDuration, Modification},
        game_object::{
            AbilitySource, ObjectId, ObjectLki, PermanentId, PlayerId, PublicId, Target,
        },
        mana::{Colors, ManaCost},
        payment::How,
        permanent::{CounterKind, Counters, Permanent},
        predicate::CardPredicate,
        stack_object::{
            ActivatedAbilityOnStack, SpellOnStack, StackObject, TriggeredAbilityOnStack,
        },
        zone::ZoneType,
    },
};

/// Characteristics of the spell or ability source, with its decision's
/// controller. An ability keeps its controller even if its source changes hands.
pub(crate) struct TargetingSource {
    pub(super) controller: PlayerId,
    colors: Colors,
    types: CardTypes,
}

impl TargetingSource {
    pub(crate) fn printed(controller: PlayerId, face: &PrintedCharacteristics) -> Self {
        Self {
            controller,
            colors: face.colors(),
            types: face.types.clone(),
        }
    }

    pub(super) fn matches(&self, quality: SourceQuality) -> bool {
        quality.matches(&self.colors, &self.types)
    }
}

impl Game {
    pub(super) fn targeting_source(
        &self,
        controller: PlayerId,
        source: ObjectId,
    ) -> TargetingSource {
        if let Some(id) = self.state.permanent_of(source) {
            let view = self.layer_view(id);
            TargetingSource {
                controller,
                colors: view.colors.clone(),
                types: view.types.clone(),
            }
        } else if let Some(lki) = self.object_lki(source) {
            TargetingSource {
                controller,
                colors: lki.colors.clone(),
                types: lki.types.clone(),
            }
        } else if let Some(spell) = self.state.spell_lki.get(&source) {
            TargetingSource {
                controller,
                colors: spell.colors.clone(),
                types: self
                    .recorded_spell_printing(source, spell.profile)
                    .types
                    .clone(),
            }
        } else {
            TargetingSource::printed(controller, self.state.source_base(source).1)
        }
    }

    /// Protection on a permanent applies to damage, enchant/equip, blocks,
    /// and targeting alike. Resolve source qualities only for a protected object.
    pub(super) fn protected_from(
        &self,
        permanent: PermanentId,
        matches: impl Fn(SourceQuality) -> bool,
    ) -> bool {
        self.layer_view(permanent)
            .static_abilities
            .iter()
            .any(|ability| ability.protects_from(&matches))
    }

    pub(super) fn protected_from_permanent(
        &self,
        recipient: PermanentId,
        source: PermanentId,
    ) -> bool {
        self.protected_from(recipient, |quality| {
            let view = self.layer_view(source);
            quality.matches(&view.colors, &view.types)
        })
    }

    pub(crate) fn permanent_power_or_lki(&self, object: ObjectId) -> Option<i32> {
        self.object_property_or_lki(object, ObjectProperty::Power)
    }

    fn object_property_or_lki(&self, object: ObjectId, property: ObjectProperty) -> Option<i32> {
        self.state
            .permanent_of(object)
            .map(|id| self.live_property(id, property))
            .or_else(|| {
                self.object_lki(object)
                    .map(|lki| self.departed_property(lki, property))
            })
    }

    /// A target permanent's or spell's property as it is, or as it last
    /// existed (CR 608.2h). A spell's power and toughness are not read.
    fn target_property(&self, target: Option<Target>, property: ObjectProperty) -> i32 {
        match target {
            Some(Target::Permanent(id)) => self
                .with_permanent_or_lki(
                    id,
                    |_| self.live_property(id, property),
                    |lki| self.departed_property(lki, property),
                )
                .unwrap_or(0),
            Some(Target::StackSpell(object)) => {
                self.spell_info(object).map_or(0, |spell| match property {
                    ObjectProperty::ManaValue => i32::from(spell.mana_value),
                    ObjectProperty::Colors => spell.colors.len() as i32,
                    ObjectProperty::Power
                    | ObjectProperty::Toughness
                    | ObjectProperty::Counters(_) => 0,
                })
            }
            _ => 0,
        }
    }

    fn live_property(&self, id: PermanentId, property: ObjectProperty) -> i32 {
        match property {
            ObjectProperty::Power => self.layer_view(id).power,
            ObjectProperty::Toughness => self.layer_view(id).toughness,
            ObjectProperty::ManaValue => self.permanent_mana_value(id),
            ObjectProperty::Colors => self.layer_view(id).colors.len() as i32,
            ObjectProperty::Counters(kind) => self.state.permanents[id]
                .as_ref()
                .map_or(0, |permanent| permanent.counter_count(kind)),
        }
    }

    fn departed_property(&self, lki: &ObjectLki, property: ObjectProperty) -> i32 {
        match property {
            ObjectProperty::Power => lki.power,
            ObjectProperty::Toughness => lki.toughness,
            ObjectProperty::ManaValue => self
                .lki_mana_printing(lki)
                .mana_cost
                .as_ref()
                .map_or(0, |cost| i32::from(cost.mana_value)),
            ObjectProperty::Colors => lki.colors.len() as i32,
            ObjectProperty::Counters(kind) => lki.counters.get(kind),
        }
    }

    /// Resolve a quantity against the frame's targets, payments and instruction results.
    pub(crate) fn effect_value(&self, value: &EffectValue, frame: &EffectFrame) -> i32 {
        let controller = frame.controller;
        let paid_properties = |how, property| {
            frame
                .spent
                .paid_as(how)
                .filter_map(move |object| self.object_property_or_lki(object, property))
        };
        match value {
            EffectValue::Property {
                of,
                property,
                aggregate,
            } => match of.as_ref() {
                ObjectSelector::This => frame
                    .source_ref()
                    .and_then(|source| self.object_property_or_lki(source, *property))
                    .unwrap_or(0),
                ObjectSelector::Triggering => self.target_property(frame.context_target, *property),
                ObjectSelector::Target(_) => {
                    self.target_property(frame.primary_target(), *property)
                }
                ObjectSelector::Paid(how) => aggregate.evaluate(paid_properties(*how, *property)),
                _ => aggregate.evaluate(
                    self.selected_permanents(of, frame)
                        .into_iter()
                        .map(|id| self.live_property(id, *property)),
                ),
            },
            EffectValue::Objects {
                of,
                predicate: None,
                in_zone: None,
            } if **of == ObjectSelector::Moved => frame.moved.len() as i32,
            EffectValue::Objects {
                of,
                predicate,
                in_zone,
            } => self
                .select(of, frame)
                .into_iter()
                .filter(|object| {
                    in_zone.is_none_or(|zone| self.state.zones.zone_of(*object) == Some(zone))
                        && predicate.as_ref().is_none_or(|predicate| {
                            self.object_matches_predicate(*object, predicate)
                        })
                })
                .count() as i32,
            EffectValue::DistinctManaValues { of } => self
                .selected_permanents(of, frame)
                .into_iter()
                .map(|id| self.permanent_mana_value(id))
                .collect::<std::collections::BTreeSet<_>>()
                .len() as i32,
            EffectValue::LifeTotal(who)
            | EffectValue::LifeGainedThisTurn(who)
            | EffectValue::LifeLostThisTurn(who)
            | EffectValue::LifeAboveStarting(who) => self
                .selected_players(who, frame)
                .into_iter()
                .map(|player| match value {
                    EffectValue::LifeTotal(_) => self.state.players[player.0].life,
                    EffectValue::LifeAboveStarting(_) => {
                        let player = &self.state.players[player.0];
                        player.life.saturating_sub(player.starting_life)
                    }
                    EffectValue::LifeGainedThisTurn(_) => {
                        self.state.turn.history[player.0].life_gained as i32
                    }
                    _ => self.state.turn.history[player.0].life_lost as i32,
                })
                .sum(),
            EffectValue::SpellManaSpent(of) | EffectValue::SpellManaValue(of) => {
                // A chosen or triggering spell keeps its exact identity after a counter.
                let stack_spell = |target| match target {
                    Target::StackSpell(object) => Some(object),
                    _ => None,
                };
                let object = match of.as_ref() {
                    ObjectSelector::Target(_) => frame
                        .active_targets()
                        .next()
                        .and_then(|t| stack_spell(t.target)),
                    ObjectSelector::Triggering => frame.context_target.and_then(stack_spell),
                    _ => self.select(of, frame).first().copied(),
                };
                object
                    .and_then(|object| self.spell_info(object))
                    .map_or(0, |spell| match value {
                        EffectValue::SpellManaSpent(_) => i32::from(spell.spent),
                        _ => i32::from(spell.mana_value),
                    })
            }
            EffectValue::X => i32::from(frame.x),
            EffectValue::NegativeX => -i32::from(frame.x),
            EffectValue::TargetPower => {
                self.target_property(frame.primary_target(), ObjectProperty::Power)
            }
            EffectValue::TriggeringPower => {
                self.target_property(frame.context_target, ObjectProperty::Power)
            }
            EffectValue::TriggeringManaValue => {
                self.target_property(frame.context_target, ObjectProperty::ManaValue)
            }
            EffectValue::TargetToughness => {
                self.target_property(frame.primary_target(), ObjectProperty::Toughness)
            }
            EffectValue::TriggeringAmount => frame.triggering_amount,
            EffectValue::ExcessDamage => frame.excess_damage,
            EffectValue::LifeLostThisWay => frame.life_lost,
            EffectValue::PowerTappedToPay => paid_properties(How::Tapped, ObjectProperty::Power)
                .map(|power| power.max(0))
                .sum(),
            EffectValue::ManaSpentToCast => i32::from(frame.spent.total()),
            EffectValue::ColorsSpentToCast => i32::from(frame.spent.colors()),
            EffectValue::SourcePower => frame
                .source_ref()
                .and_then(|source| self.permanent_power_or_lki(source))
                .unwrap_or(0),
            EffectValue::AttachedCreaturePower => self
                .source_permanent_id(frame)
                .map_or(0, |source| self.attached_creature_power(source)),
            EffectValue::AttackingMatching { predicate } => self
                .iter_battlefield_permanents(controller)
                .filter(|id| {
                    self.state.permanents[*id]
                        .as_ref()
                        .is_some_and(|permanent| permanent.attacking)
                        && self.permanent_matches_predicate(*id, predicate)
                })
                .count() as i32,
            EffectValue::Half(inner) => self.effect_value(inner, frame).div_euclid(2),
            EffectValue::Scaled { factor, value } => {
                factor.saturating_mul(self.effect_value(value, frame))
            }
            EffectValue::Fixed(value) => *value,
            EffectValue::CreaturesWithCounters => self
                .layer_views(self.iter_battlefield_permanents(controller))
                .filter(|(id, view)| {
                    view.is_creature()
                        && self.state.permanents[*id]
                            .as_ref()
                            .is_some_and(|p| p.counter_count(CounterKind::PlusOnePlusOne) > 0)
                })
                .count() as i32,
            EffectValue::BattlefieldMatching { predicate } => {
                self.count_battlefield_matching(controller, predicate) as i32
            }
            EffectValue::GraveyardMatching { base, predicate } => {
                base.saturating_add(self.count_graveyard_matching(controller, predicate) as i32)
            }
            EffectValue::CardsDrawnMinus { subtract } => self.state.turn.history[controller.0]
                .cards_drawn
                .saturating_sub(*subtract)
                as i32,
            EffectValue::GreatestSharedCreatureType => {
                self.greatest_shared_creature_type(controller) as i32
            }
            EffectValue::GreatestPowerYouControl { predicate } => self
                .layer_views(self.iter_battlefield_permanents(controller))
                .filter(|(id, view)| {
                    view.is_creature()
                        && super::layers::matches_view(self, *id, predicate, view, None)
                })
                .map(|(_, view)| view.power)
                .max()
                .unwrap_or(0)
                .max(0),
            EffectValue::CardsInHand => self.state.zones.size(ZoneType::Hand, controller) as i32,
            EffectValue::PermanentsSacrificedThisTurn => {
                self.state.turn.history[controller.0].permanents_sacrificed as i32
            }
            EffectValue::SpellsCastThisTurn => {
                self.state.turn.spells_cast[controller.0].len() as i32
            }
            EffectValue::OwnedPermanentsOpponentsControl => self
                .iter_battlefield_permanents(PlayerId(1 - controller.0))
                .filter(|id| {
                    self.state.permanents[*id]
                        .as_ref()
                        .is_some_and(|permanent| self.state[permanent.object].owner == controller)
                })
                .count() as i32,
        }
    }

    /// Evaluate an amount a pending ability reads from what paid its cost.
    pub(crate) fn paid_cost_value(
        &self,
        value: &EffectValue,
        controller: PlayerId,
        paid: &crate::state::payment::PaymentReceipt,
    ) -> i32 {
        let frame = EffectFrame {
            spent: paid.clone(),
            ..EffectFrame::new(controller, Vec::new())
        };
        self.effect_value(value, &frame)
    }

    /// Evaluate a mana output or price with only its controller bound.
    pub(crate) fn controller_value(&self, value: &EffectValue, controller: PlayerId) -> i32 {
        self.effect_value(value, &EffectFrame::new(controller, Vec::new()))
    }

    pub(crate) fn resolve_top_of_stack(&mut self) {
        let Some(stack_object) = self.state.stack_objects.last().cloned() else {
            return;
        };

        match stack_object {
            // The spell's stack object stays on the stack while it resolves
            // (CR 608.2m — the card moves out as resolution's final step),
            // so a suspended resolution leaves a consistent stack.
            StackObject::Spell(spell) => self.resolve_spell_object(spell),
            StackObject::ActivatedAbility(ability) => {
                self.pop_stack();
                self.resolve_activated_ability(ability);
            }
            StackObject::TriggeredAbility(triggered) => {
                self.pop_stack();
                self.resolve_triggered_ability(triggered);
            }
        }
    }

    fn resolve_spell_object(&mut self, spell: SpellOnStack) {
        let object = spell.object;
        let card_ref = &self.state[object];
        let effects =
            crate::state::ability::bind_effect_clauses(card_ref.spell_effects_for_mode(spell.mode));
        let requirements = card_ref.target_requirements_for_mode(spell.mode);

        let Some(targets) = self.resolution_targets(
            &spell.targets,
            &spell.target_req_indices,
            &requirements,
            spell.controller,
            AbilitySource::new(object, None),
        ) else {
            self.move_object(object, ZoneType::Graveyard);
            self.emit(GameEvent::SpellCountered {
                object_id: object,
                by: None,
            });
            return;
        };

        let frame = EffectFrame {
            source_identity: crate::state::game_object::AbilitySource::optional(
                Some(object),
                self.state.live(object),
            ),
            kicked: spell.kicked,
            x: spell.x,
            spent: spell.spent.clone(),
            targets,
            completion: FrameCompletion::Spell {
                object,
                attach_to: None,
            },
            ..EffectFrame::new(spell.controller, effects)
        };
        self.run_frame(frame);
    }

    /// CR 725.1b: abilities cease to exist. All spells, including copies
    /// and the resolving spell, leave the stack in one zone-change batch.
    fn exile_stack(&mut self) -> EventBatch {
        self.state
            .stack_objects_mut()
            .retain(|item| matches!(item, StackObject::Spell(_)));
        let moves: Vec<_> = self
            .state
            .stack_objects
            .iter()
            .rev()
            .map(|spell| {
                self.zone_move_proposal(spell.source_object(), ZoneType::Exile, Default::default())
            })
            .collect();
        self.event_batch(EventBinding::None, moves)
    }

    /// Counter an activated or triggered ability: it is removed from the
    /// stack and does nothing (CR 701.6b).
    pub(crate) fn counter_ability(&mut self, id: PublicId, by: Option<ObjectId>) {
        let Some(index) = self.stack_index_of(Target::StackAbility(id)) else {
            return;
        };
        let ability = self.state.stack_objects_mut().remove(index);
        let source_object = match &ability {
            StackObject::ActivatedAbility(ability) => ability.source_object(),
            StackObject::TriggeredAbility(ability) => ability.source_object(),
            StackObject::Spell(_) => unreachable!("an ability index"),
        };
        self.emit(GameEvent::AbilityCountered {
            ability: id,
            source_object,
            controller: ability.controller(),
            by,
        });
    }

    /// Counter a spell by an effect (CR 701.6a), putting it into `to` (its
    /// owner's graveyard unless the effect says otherwise); a spell that
    /// can't be countered stays on the stack (CR 101.2).
    pub(crate) fn counter_spell(&mut self, object: ObjectId, by: Option<ObjectId>, to: ZoneType) {
        if !self.spell_can_be_countered(object) {
            return;
        }
        self.move_object(object, to);
        self.emit(GameEvent::SpellCountered {
            object_id: object,
            by,
        });
    }

    fn spell_can_be_countered(&self, object: ObjectId) -> bool {
        let Some(spell) = self.stack_spell(object) else {
            return false;
        };
        if self.state[object]
            .statics
            .contains(&StaticAbility::CantBeCountered)
        {
            return false;
        }
        !self.static_sources().any(|(id, owner, view)| {
            owner == spell.controller && !self.is_phased_out(id) && view.static_abilities.iter().any(|ability| {
                matches!(ability, StaticAbility::SpellsCantBeCountered(predicate) if self.object_matches_predicate(object, predicate))
            })
        })
    }

    /// The face an ability on the stack came from: the card as it stands
    /// while that face is up, else that face's row (the card has turned over
    /// or left since; the ability exists independently, CR 113.7a).
    pub(crate) fn source_face(
        &self,
        object: ObjectId,
        face: CardDefId,
    ) -> Option<&PrintedCharacteristics> {
        let (definition, current) = self.state.source_base(object);
        if definition == face {
            Some(current)
        } else {
            default_content_pack()
                .definition(face)
                .map(|row| &*row.printed)
        }
    }

    /// The effects a stacked triggered ability runs: its inline (delayed,
    /// granted or reflexive) effects, or its printed ability's with a modal
    /// trigger's chosen modes expanded. Resolution reads the printed ability
    /// itself, since it also checks the intervening "if".
    pub(crate) fn stacked_trigger_effects<'a>(
        &'a self,
        triggered: &'a TriggeredAbilityOnStack,
    ) -> Option<std::borrow::Cow<'a, [Effect]>> {
        if let Some(effects) = &triggered.inline_effects {
            return Some(std::borrow::Cow::Borrowed(effects.as_slice()));
        }
        let object = triggered
            .source_lki
            .as_ref()
            .map_or(triggered.source_object(), |lki| lki.object);
        let printed = self
            .source_face(object, triggered.source_definition_id)?
            .abilities
            .get(triggered.ability_index)?;
        Some(crate::state::ability::with_chosen_modes(
            printed.effects(),
            &triggered.chosen_modes,
        ))
    }

    fn resolve_activated_ability(&mut self, ability: ActivatedAbilityOnStack) {
        let effects = ability.effects;
        let program = if ability.program {
            effects.clone()
        } else {
            Vec::new()
        };
        let requirements = crate::state::ability::effect_target_requirements(&effects);
        let Some(targets) = self.resolution_targets(
            &ability.targets,
            &ability.target_req_indices,
            &requirements,
            ability.controller,
            AbilitySource::new(ability.source_object, Some(ability.source_object)),
        ) else {
            return;
        };

        // CR 113.7a: an activated ability exists independently of its source.
        // Source-relative effects still require the original battlefield object;
        // independent effects (such as drawing) resolve after that object leaves.
        let frame = EffectFrame {
            program,
            source_identity: crate::state::game_object::AbilitySource::optional(
                Some(ability.source_object),
                Some(ability.source_object),
            ),
            inherited_from: ability.inherited_from,
            source_transforms: ability.source_transforms,
            x: ability.x,
            spent: ability.paid,
            targets,
            grantor: ability.granted_by.map(|grant| grant.grantor),
            ..EffectFrame::new(
                ability.controller,
                crate::state::ability::bind_effect_clauses(effects),
            )
        };
        self.run_frame(frame);
    }

    fn resolve_triggered_ability(&mut self, triggered: TriggeredAbilityOnStack) {
        let program = if triggered.program {
            triggered.inline_effects.clone().unwrap_or_default()
        } else {
            Vec::new()
        };
        let source_object = triggered
            .source_lki
            .as_ref()
            .map(|lki| lki.object)
            .unwrap_or(triggered.source_object());
        let source_controller = triggered
            .source_lki
            .as_ref()
            .map(|lki| lki.controller)
            .unwrap_or(triggered.controller);
        // Inline (delayed, granted or reflexive) effects have no printed
        // ability; a printed modal trigger runs its chosen modes.
        let (effects, ability) = if let Some(effects) = triggered.inline_effects {
            (effects, None)
        } else {
            let Some(ability) = self
                .source_face(source_object, triggered.source_definition_id)
                .and_then(|face| face.abilities.get(triggered.ability_index))
                .cloned()
            else {
                return;
            };
            let Ability::Triggered {
                intervening_if,
                effects,
                ..
            } = ability;
            // CR 603.4: an intervening "if" that no longer holds removes the
            // ability from the stack doing nothing.
            if intervening_if.as_ref().is_some_and(|condition| {
                !self.check_intervening_if(
                    condition,
                    source_controller,
                    triggered.source_ref(),
                    triggered.context,
                )
            }) {
                return;
            }
            (
                crate::state::ability::with_chosen_modes(&effects, &triggered.chosen_modes)
                    .into_owned(),
                Some(triggered.ability_index),
            )
        };

        let requirements = crate::state::ability::effect_target_requirements(&effects);
        let Some(targets) = self.resolution_targets(
            &triggered.targets,
            &triggered.target_req_indices,
            &requirements,
            triggered.controller,
            triggered.source_identity,
        ) else {
            return;
        };

        let resolutions_this_turn = if effects.iter().any(Effect::reads_resolution_count) {
            let key = (
                triggered.source_identity.object(),
                triggered.source_definition_id,
                triggered.ability_index,
            );
            let count = self
                .state
                .turn
                .ability_resolutions_this_turn
                .entry(key)
                .or_default();
            *count += 1;
            *count
        } else {
            0
        };

        let frame = EffectFrame {
            program,
            remembered: triggered.remembered,
            triggering_amount: triggered.triggering_amount,
            source_identity: crate::state::game_object::AbilitySource::optional(
                Some(source_object),
                triggered.source_identity.reference(),
            ),
            inherited_from: triggered
                .source_lki
                .as_ref()
                .and_then(|lki| lki.inherited_from),
            source_transforms: triggered.source_transforms,
            resolutions_this_turn,
            x: triggered.x,
            spent: triggered.spent.clone(),
            targets,
            context_target: triggered.context,
            triggering_player: triggered.player,
            ability,
            ability_face: ability.map(|_| triggered.source_definition_id),
            ..EffectFrame::new(
                triggered.controller,
                crate::state::ability::bind_effect_clauses(effects),
            )
        };
        self.run_frame(frame);
    }

    /// CR 608.2b: check every chosen slot against its own clause once, before
    /// any instruction. Zero chosen optional targets still resolve; a nonempty
    /// all-illegal set suppresses the entire spell or ability, even untargeted
    /// instructions. The retained slots also enter clones and state hashes.
    fn resolution_targets(
        &self,
        targets: &[Target],
        indices: &[usize],
        requirements: &[TargetRequirement],
        controller: PlayerId,
        source: AbilitySource,
    ) -> Option<Vec<ResolutionTarget>> {
        if targets.is_empty() {
            return Some(Vec::new());
        }
        let targeting = self.targeting_source(controller, source.object());
        let mut slots: Vec<_> = targets
            .iter()
            .enumerate()
            .map(|(index, target)| {
                let requirement = indices.get(index).copied().unwrap_or(0);
                ResolutionTarget {
                    target: *target,
                    requirement,
                    legal: requirements.get(requirement).is_some_and(|req| {
                        self.target_meets_requirement(*target, req, &targeting, source)
                    }),
                }
            })
            .collect();
        for (index, requirement) in requirements.iter().enumerate() {
            if requirement.shares_creature_type {
                self.require_shared_creature_type(&mut slots, index);
            }
        }
        (slots.is_empty() || slots.iter().any(|slot| slot.legal)).then_some(slots)
    }

    /// "That share a creature type" (CR 608.2b): a target stays legal only
    /// while it shares a creature type with each other target of its clause.
    /// A departed target counts with its creature types as it last existed
    /// on the battlefield (CR 608.2h), and is itself illegal.
    fn require_shared_creature_type(&self, slots: &mut [ResolutionTarget], requirement: usize) {
        let members: Vec<usize> = (0..slots.len())
            .filter(|&index| slots[index].requirement == requirement)
            .collect();
        let types: Vec<Vec<String>> = members
            .iter()
            .map(|&index| self.last_known_creature_types(slots[index].target))
            .collect();
        for (member, &index) in members.iter().enumerate() {
            let own = &types[member];
            slots[index].legal &= types
                .iter()
                .enumerate()
                .all(|(other, theirs)| other == member || theirs.iter().any(|t| own.contains(t)));
        }
    }

    /// A permanent target's creature types: current ones while it remains,
    /// else as it last existed on the battlefield (CR 608.2h).
    fn last_known_creature_types(&self, target: Target) -> Vec<String> {
        match target {
            Target::Permanent(id) => self
                .with_permanent_or_lki(
                    id,
                    |_| self.layer_view(id).creature_types.clone(),
                    |lki| lki.creature_types(),
                )
                .unwrap_or_default(),
            _ => Vec::new(),
        }
    }

    /// Instructions cannot act on an object that has left its expected zone
    /// (CR 400.7). This is deliberately independent of TargetSpec: hexproof,
    /// controller and characteristic changes during resolution do not retarget.
    fn target_is_present(&self, target: Target) -> bool {
        match target {
            Target::Permanent(id) => self.permanent_object(id).is_some(),
            Target::GraveyardCard(object) => {
                self.state.card_of(object).is_some()
                    && self.state.zones.zone_of(object) == Some(ZoneType::Graveyard)
            }
            Target::StackSpell(_) | Target::StackAbility(_) => {
                self.stack_index_of(target).is_some()
            }
            Target::Player(player) => self.state.loss(player.0).is_none(),
        }
    }

    /// Read the attachment at resolution (CR 608.2h). If the source left,
    /// use its last known wearer, which may itself have left.
    fn frame_attached(&self, frame: &EffectFrame) -> Option<PermanentId> {
        let source = frame.source_ref()?;
        if let Ok(permanent) = self.lookup_current_permanent(source) {
            return self.attached_permanent(permanent);
        }
        self.lookup_current_permanent(self.object_lki(source)?.attached_to?)
            .ok()
    }

    /// Put library cards into their owners' graveyards as one simultaneous
    /// move (CR 701.17a), bound as the frame's `CardsThisWay`.
    fn mill_objects(&mut self, frame: &mut EffectFrame, object_ids: Vec<ObjectId>) {
        frame.this_way.clear();
        let events: Vec<_> = object_ids
            .into_iter()
            .map(|object_id| {
                self.zone_move_proposal(object_id, ZoneType::Graveyard, ZoneEntry::default())
            })
            .collect();
        frame.batch = Some(self.event_batch(EventBinding::ThisWay, events));
    }

    /// The players a selector names as the effect resolves; a target player
    /// that is no longer a legal target is skipped.
    fn selected_players(&self, selector: &PlayerSelector, frame: &EffectFrame) -> Vec<PlayerId> {
        match selector {
            PlayerSelector::You => vec![frame.controller],
            PlayerSelector::EachOpponent => vec![PlayerId(1 - frame.controller.0)],
            PlayerSelector::EachPlayer => vec![self.active_player(), self.non_active_player()],
            PlayerSelector::TriggeringPlayer => frame.triggering_player.into_iter().collect(),
            PlayerSelector::TriggeringPlayersOpponents => frame
                .triggering_player
                .map(|player| PlayerId(1 - player.0))
                .into_iter()
                .collect(),
            PlayerSelector::Target(_) => match self.present_primary_target(frame) {
                Some(Target::Player(player)) => vec![player],
                _ => Vec::new(),
            },
            PlayerSelector::TargetClause(index) => frame
                .targets
                .iter()
                .find_map(|slot| match slot.target {
                    Target::Player(player) if slot.requirement == *index && slot.legal => {
                        Some(player)
                    }
                    _ => None,
                })
                .into_iter()
                .collect(),
            PlayerSelector::Owner => Vec::new(),
            PlayerSelector::TargetController => match frame.primary_target() {
                Some(Target::Permanent(id)) => self.last_known_controller(id).into_iter().collect(),
                // A countered, uncounterable, copied or stolen spell alike.
                Some(Target::StackSpell(object)) => self
                    .spell_info(object)
                    .map(|spell| spell.controller)
                    .into_iter()
                    .collect(),
                _ => Vec::new(),
            },
        }
    }

    /// The owner and zone a zone selector reads: the resolving player's,
    /// or the clause's target player's.
    fn selection_zone(
        &self,
        selector: &ObjectSelector,
        frame: &EffectFrame,
    ) -> Option<(PlayerId, ZoneType)> {
        match selector {
            ObjectSelector::Zone(zone) => Some((frame.controller, *zone)),
            ObjectSelector::TargetPlayersZone { zone, .. } => {
                match self.present_primary_target(frame)? {
                    Target::Player(player) => Some((player, *zone)),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    pub(crate) fn present_primary_target(&self, frame: &EffectFrame) -> Option<Target> {
        frame
            .primary_target()
            .filter(|target| self.target_is_present(*target))
    }

    /// The players and battlefield permanents a selector names, fixed now
    /// (CR 611.2c): `You` the resolving player, a `Target` selector the
    /// chosen player when the clause targets one, else the objects
    /// [`Game::select`] names that are on the battlefield.
    fn resolve_affected(&self, what: &ObjectSelector, frame: &EffectFrame) -> Vec<Affected> {
        if *what == ObjectSelector::You {
            return vec![Affected::Player(frame.controller)];
        }
        if matches!(what, ObjectSelector::Targets(_)) {
            return frame
                .active_targets()
                .filter(|slot| self.target_is_present(slot.target))
                .filter_map(|slot| match slot.target {
                    Target::Player(player) => Some(Affected::Player(player)),
                    Target::Permanent(id) => self.affected_permanent(id),
                    _ => None,
                })
                .collect();
        }
        if matches!(what, ObjectSelector::Target(_)) {
            if let Some(Target::Player(player)) = self.present_primary_target(frame) {
                return vec![Affected::Player(player)];
            }
        }
        self.selected_permanents(what, frame)
            .into_iter()
            .filter_map(|id| self.affected_permanent(id))
            .collect()
    }

    /// Execute one effect. Returns a decision to suspend on, or None if the
    /// effect completed. Branching effects push their branch onto the front
    /// of the frame's queue. A verb over a `Chosen` selector first suspends
    /// on the choice (`Decision::ChooseObject`); the chosen object is bound
    /// to the frame and the verb runs over it.
    pub(crate) fn execute_frame_effect(
        &mut self,
        effect: &Effect,
        frame: &mut EffectFrame,
    ) -> Option<Decision> {
        if frame.chosen.is_none() {
            if let Some((predicate, except, optional)) = effect.chosen() {
                // What a sacrifice moves answers a following "if you do"; a
                // declined or impossible choice moves nothing.
                if matches!(effect, Effect::Sacrifice { .. } | Effect::Move { .. }) {
                    frame.moved.clear();
                }
                let decision = Decision::ChooseObject {
                    player: frame.controller,
                    predicate: predicate.clone(),
                    excluded: self.excepted_permanent(except, frame),
                    optional,
                    effect: Box::new(effect.clone()),
                };
                return (!self.choice_candidates(&decision).is_empty()).then_some(decision);
            }
        }
        match effect {
            Effect::ApplyContinuous {
                affected,
                modification,
                duration,
            } => {
                let mut affected = self.resolve_affected(affected, frame);
                let modification = modification.resolve(|value| self.effect_value(value, frame));
                if modification == Modification::PhasedOut {
                    affected = self.phase_out_set(affected);
                }
                // "For as long as you control [this]" with this gone or
                // already someone else's lasts no time at all.
                let source_object = frame.source_ref().filter(|source| {
                    self.lookup_current_permanent(*source)
                        .ok()
                        .and_then(|id| self.state.permanents[id].as_ref())
                        .is_some_and(|permanent| permanent.controller == frame.controller)
                });
                if *duration == EffectDuration::WhileYouControlSource && source_object.is_none() {
                    return None;
                }
                self.record_continuous(
                    frame.source(),
                    source_object,
                    frame.controller,
                    affected,
                    modification,
                    *duration,
                );
                None
            }
            Effect::PutCounters { on, kind, amount } => {
                let amount = self.effect_value(amount, frame);
                let events = self
                    .selected_permanents(on, frame)
                    .into_iter()
                    .filter_map(|id| self.permanent_object(id))
                    .map(|target| ProposedEvent::CounterChange {
                        provenance: crate::flow::proposed_event::CounterProvenance::new(
                            crate::flow::proposed_event::EventCause::Effect,
                        ),
                        target,
                        kind: *kind,
                        delta: amount,
                        by: Some(frame.controller),
                    });
                frame.batch = Some(self.event_batch(EventBinding::None, events));
                None
            }
            Effect::PutCountersFrom { from, on } => {
                let counters = self.selected_counters(from, frame);
                let mut events = Vec::new();
                for target in self
                    .selected_permanents(on, frame)
                    .into_iter()
                    .filter_map(|id| self.permanent_object(id))
                {
                    for (kind, delta) in counters.iter() {
                        events.push(ProposedEvent::CounterChange {
                            provenance: crate::flow::proposed_event::CounterProvenance::new(
                                crate::flow::proposed_event::EventCause::Effect,
                            ),
                            target,
                            kind,
                            delta,
                            by: Some(frame.controller),
                        });
                    }
                }
                frame.batch = Some(self.event_batch(EventBinding::None, events));
                None
            }
            Effect::Sacrifice { what, optional } => {
                debug_assert!(
                    !optional || what.is_chosen(),
                    "\"you may sacrifice\" chooses among permanents"
                );
                // A player sacrifices only what they control (CR 701.17a).
                let events = self
                    .selected_permanents(what, frame)
                    .into_iter()
                    .filter_map(|id| {
                        let permanent = self.state.permanents[id].as_ref()?;
                        (permanent.controller == frame.controller).then_some(
                            ProposedEvent::Sacrifice {
                                target: permanent.object,
                            },
                        )
                    });
                frame.batch = Some(self.event_batch(
                    EventBinding::Moved {
                        linked: None,
                        shuffle: false,
                        library_end: None,
                        reveal: false,
                    },
                    events,
                ));
                None
            }
            Effect::Move {
                unrevealed,
                library_end,
                what,
                to,
                tapped,
                transformed,
                under,
                with_counter,
                linked,
            } => {
                // Select before clearing: a `Moved` selector reads the objects
                // the previous instruction moved (flicker's return).
                let selected = self.select(what, frame);
                frame.moved.clear();
                let source = frame
                    .source_ref()
                    .filter(|source| self.lookup_current_permanent(*source).is_ok());
                // "Until this leaves the battlefield" (CR 603.6e pragmatics,
                // Banisher Priest ruling): the exact source must still be on
                // the battlefield; a later incarnation of the same card does
                // not keep the old link alive.
                if *linked == Some(LinkedExile::UntilSourceLeaves) && source.is_none() {
                    return None;
                }
                let controller = match under {
                    None => Some(frame.controller),
                    Some(PlayerSelector::Owner) => None,
                    Some(selector) => match self.selected_players(selector, frame)[..] {
                        [player] => Some(player),
                        _ => return None,
                    },
                };
                let mut counters = Counters::default();
                if let Some(kind) = with_counter {
                    counters.change(*kind, 1);
                }
                let mut events = Vec::new();
                for object in selected {
                    let from = self.state.zones.zone_of(object);
                    // "Put that card on top" of the library it is already in
                    // (Cruel Tutor, after its shuffle) only reorders it.
                    if let (Some(ZoneType::Library), ZoneType::Library, Some(end)) =
                        (from, *to, library_end)
                    {
                        self.reorder_in_library(object, *end);
                        if *end == crate::state::ability::LibraryEnd::Top {
                            self.remember_placed_top(object, frame);
                        }
                        continue;
                    }
                    // A token cannot return after leaving the battlefield (CR 111.7).
                    // Returning transformed requires a back face (CR 712.14a).
                    if *to == ZoneType::Battlefield
                        && ((!self.state[object].is_card() && from != Some(ZoneType::Battlefield))
                            || (*transformed
                                && default_content_pack()
                                    .back_face(self.state[object].definition_id)
                                    .is_none()))
                    {
                        continue;
                    }
                    let entry = if *to == ZoneType::Battlefield {
                        ZoneEntry {
                            tapped: *tapped,
                            transformed: *transformed,
                            counters: counters.clone(),
                            controller,
                            ..ZoneEntry::default()
                        }
                    } else {
                        // "exile it with a stash counter on it" (CR 122.1).
                        ZoneEntry {
                            counters: counters.clone(),
                            ..ZoneEntry::default()
                        }
                    };
                    events.push(self.zone_move_proposal(object, *to, entry));
                }
                frame.batch = Some(self.event_batch(
                    EventBinding::Moved {
                        linked: linked.and_then(|linked| {
                            source.map(|source| (source, linked == LinkedExile::UntilSourceLeaves))
                        }),
                        shuffle: *to == ZoneType::Library && library_end.is_none(),
                        library_end: *library_end,
                        reveal: *to == ZoneType::Hand && !unrevealed,
                    },
                    events,
                ));
                None
            }
            Effect::Prepare { what } | Effect::Unprepare { what } => {
                for id in self.selected_permanents(what, frame) {
                    let source = self.state.permanents[id].as_ref().unwrap().object;
                    if matches!(effect, Effect::Prepare { .. }) {
                        self.prepare_inset(source);
                    } else {
                        self.unprepare_inset(source);
                    }
                }
                None
            }
            Effect::Tap { what } => {
                for id in self.selected_permanents(what, frame) {
                    self.tap_permanent(id, false);
                }
                None
            }
            Effect::Untap { what } => {
                for id in self.selected_permanents(what, frame) {
                    self.untap_permanent(id);
                }
                None
            }
            Effect::RemoveAllCounters { from } => {
                for id in self.selected_permanents(from, frame) {
                    for kind in CounterKind::ALL {
                        let count = self.state.permanents[id]
                            .as_ref()
                            .map_or(0, |permanent| permanent.counter_count(kind));
                        if count > 0 {
                            self.change_counters(id, kind, -count);
                        }
                    }
                }
                None
            }
            Effect::RemoveCounters {
                from,
                up_to,
                chosen,
            } => {
                let permanent_id = *self.selected_permanents(from, frame).first()?;
                let mut up_to = *up_to;
                if let Some(kind) = chosen {
                    self.change_counters(permanent_id, *kind, -1);
                    up_to -= 1;
                }
                let permanent = self.state.permanents[permanent_id].as_ref()?;
                let mut modes: Vec<Vec<Effect>> = CounterKind::ALL
                    .into_iter()
                    .filter(|kind| up_to > 0 && permanent.counter_count(*kind) > 0)
                    .map(|kind| {
                        vec![Effect::RemoveCounters {
                            from: from.clone(),
                            up_to,
                            chosen: Some(kind),
                        }]
                    })
                    .collect();
                if modes.is_empty() {
                    return None;
                }
                // "Up to": stopping is always a legal choice.
                modes.push(Vec::new());
                Some(Decision::Modal {
                    player: frame.controller,
                    modes,
                })
            }
            Effect::Destroy { what } => {
                // One simultaneous destruction: leave-the-battlefield
                // abilities look back at the permanents as they all were
                // before it (CR 603.10a).
                let events = self
                    .selected_permanents(what, frame)
                    .into_iter()
                    .filter_map(|id| self.permanent_object(id))
                    .map(|target| ProposedEvent::Destroy { target });
                frame.batch = Some(self.event_batch(EventBinding::Destroyed, events));
                None
            }
            Effect::DistributeCountersToTappedOpponentCreatures { kind, count } => {
                self.distribute_counters(frame.controller, *kind, *count)
            }
            Effect::GivePlayerCounters {
                player,
                kind,
                count,
            } => {
                for seat in self.selected_players(player, frame) {
                    self.state.players[seat.0].counters.change(*kind, *count);
                }
                None
            }
            Effect::DealDamageToPlayer { player, amount } => {
                let amount = self.effect_value(amount, frame);
                frame.batch = Some(
                    self.event_batch(
                        EventBinding::None,
                        self.selected_players(player, frame)
                            .into_iter()
                            .map(|player| ProposedEvent::Damage {
                                source: frame.source_ref(),
                                target: ProposedDamageTarget::Player(player),
                                amount,
                                combat: false,
                            }),
                    ),
                );
                None
            }
            Effect::DrainLife { amount } => {
                let amount = self.effect_value(amount, frame).max(0);
                self.change_life(PlayerId(1 - frame.controller.0), -amount);
                self.gain_life(frame.controller, amount);
                None
            }
            Effect::MillCards {
                count,
                target,
                player,
            } => {
                let players = match (player, target) {
                    (Some(player), _) => self.selected_players(player, frame),
                    (None, Some(_)) => match self.present_primary_target(frame) {
                        Some(Target::Player(player)) => vec![player],
                        _ => return None,
                    },
                    (None, None) => vec![frame.controller],
                };
                let count = self.effect_value(count, frame).max(0) as usize;
                let object_ids: Vec<_> = players
                    .into_iter()
                    .flat_map(|player| self.library_top_objects(player, count))
                    .collect();
                self.mill_objects(frame, object_ids);
                None
            }
            Effect::MillUntil { player, predicate } => {
                let mut object_ids = Vec::new();
                for player in self.selected_players(player, frame) {
                    for object_id in self.library_top_objects(player, usize::MAX) {
                        let definition = self.state[object_id].definition_id;
                        self.emit(GameEvent::CardRevealed {
                            owner: player,
                            definition,
                        });
                        object_ids.push(object_id);
                        if self.object_matches_predicate(object_id, predicate) {
                            break;
                        }
                    }
                }
                self.mill_objects(frame, object_ids);
                None
            }
            Effect::SelectCards {
                from,
                predicate,
                min,
                max,
                effects,
                if_declined,
            } => {
                // A selection from a zone or from milled cards chooses cards, so
                // a transient token there is not a candidate. Other selectors
                // name objects: "create a token, then attach this to it" selects
                // the token through `CreatedThisWay`.
                let cards_only = matches!(
                    from,
                    ObjectSelector::Zone(_)
                        | ObjectSelector::TargetPlayersZone { .. }
                        | ObjectSelector::CardsThisWay
                );
                let candidates: Vec<_> = self
                    .select(from, frame)
                    .into_iter()
                    .filter(|object| {
                        let object = *object;
                        (!cards_only || self.state.card_of(object).is_some())
                            && self.object_matches_predicate(object, predicate)
                    })
                    .collect();
                let max = self.effect_value(max, frame).max(0) as usize;
                let min = (*min).min(max).min(candidates.len());
                if *from == ObjectSelector::Zone(ZoneType::Library) {
                    frame.complete_with_event(GameEvent::LibrarySearched {
                        player: frame.controller,
                    });
                }
                if max == 0 || candidates.is_empty() {
                    frame.select_scope(Vec::new(), None, if_declined.clone());
                    None
                } else {
                    Some(Decision::SelectCards {
                        origin: match from {
                            ObjectSelector::CardsThisWay => {
                                super::decision::SelectionOrigin::Milled
                            }
                            ObjectSelector::Zone(ZoneType::Library)
                            | ObjectSelector::TargetPlayersZone {
                                zone: ZoneType::Library,
                                ..
                            } => super::decision::SelectionOrigin::Library,
                            ObjectSelector::Zone(ZoneType::Hand) => {
                                super::decision::SelectionOrigin::Hand
                            }
                            _ => super::decision::SelectionOrigin::Other,
                        },
                        player: frame.controller,
                        from: self.selection_zone(from, frame).map(|(owner, zone)| {
                            crate::state::binding::ZoneSelection {
                                owner,
                                zone,
                                predicate: predicate.clone(),
                            }
                        }),
                        candidates,
                        selected: Vec::new(),
                        min,
                        max,
                        effects: effects.clone(),
                        if_declined: if_declined.clone(),
                    })
                }
            }
            Effect::SelectTargetRequirement { index } => {
                frame.active_requirement = *index;
                None
            }
            Effect::TargetClause { .. } => unreachable!("clauses are bound before resolution"),

            Effect::Discard { what, count } => {
                frame.this_way.clear();
                let [Affected::Player(player)] = self.resolve_affected(what, frame)[..] else {
                    return None;
                };
                (self
                    .state
                    .zone_cards(ZoneType::Hand, player)
                    .next()
                    .is_some()
                    && *count > 0)
                    .then_some(Decision::Discard {
                        player,
                        remaining: *count,
                    })
            }
            Effect::CopyTriggeringObject => {
                let copy = self.copy_stack_object(frame.context_target?, frame.controller)?;
                self.next_retarget(frame.controller, copy, 0, true)
            }
            Effect::ChangeTarget { .. } => {
                let object = self.present_primary_target(frame)?;
                self.next_retarget(frame.controller, object, 0, false)
            }
            Effect::AdditionalCombatPhase { attackers } => {
                self.state
                    .turn
                    .additional_combats
                    .push(crate::flow::turn::AdditionalCombat {
                        attackers: attackers.clone(),
                    });
                None
            }
            Effect::TakeExtraTurn => {
                self.state.turn.extra_turns.push(frame.controller);
                None
            }
            Effect::CopyTriggeringSpellForEach { each } => {
                let spell = frame.context_target?;
                let index = self.stack_index_of(spell)?;
                let slots = self.state.stack_objects[index].targets().len();
                let legal = self.retarget_candidates_for(spell, 0);
                let onto: Vec<Target> = self
                    .selected_permanents(each, frame)
                    .into_iter()
                    .map(Target::Permanent)
                    .filter(|target| legal.contains(target))
                    .collect();
                for target in onto {
                    let Some(copy) = self.copy_stack_object(spell, frame.controller) else {
                        break;
                    };
                    for slot in 0..slots {
                        self.change_stack_target(copy, slot, target);
                    }
                }
                None
            }
            Effect::WinGame => {
                self.win_game(frame.controller);
                None
            }
            Effect::LoseGame => {
                self.lose_game(frame.controller);
                None
            }
            Effect::EndTheTurn => {
                // CR 725.1a: collect occurrences from earlier instructions
                // before discarding them. Later triggers wait for cleanup.
                self.collect_waiting_triggers();
                self.state.pending_triggers_mut().clear();
                self.capture_resolution(frame);
                *frame = EffectFrame {
                    batch: Some(self.exile_stack()),
                    completion: FrameCompletion::EndTurn,
                    ..EffectFrame::new(frame.controller, Vec::new())
                };
                None
            }
            Effect::ControlPlayer { what, duration } => {
                for affected in self.resolve_affected(what, frame) {
                    if let Affected::Player(player) = affected {
                        self.state
                            .turn
                            .player_controls
                            .push(crate::flow::turn::PlayerControl {
                                player,
                                controller: frame.controller,
                                duration: *duration,
                                active: false,
                            });
                    }
                }
                None
            }
            Effect::TransformSource => {
                let id = self.source_permanent_id(frame)?;
                // CR 701.27f: only if it hasn't turned over since this ability
                // went on the stack.
                if frame.source_transforms == self.state.permanents[id].as_ref()?.transforms {
                    self.transform_permanent(id);
                }
                None
            }
            Effect::GainControl {
                player,
                what,
                duration,
            } => {
                let [player] = self.selected_players(player, frame)[..] else {
                    return None;
                };
                let affected: Vec<_> = self
                    .selected_permanents(what, frame)
                    .into_iter()
                    .filter_map(|id| self.affected_permanent(id))
                    .collect();
                self.apply_continuous(
                    frame.source(),
                    player,
                    affected,
                    Modification::Control,
                    *duration,
                );
                None
            }
            Effect::Airbend { what } => {
                let selected = self.select(what, frame);
                let mut bent = false;
                for object in selected {
                    if matches!(self.state.zones.zone_of(object), Some(ZoneType::Stack))
                        || self.lookup_current_permanent(object).is_ok()
                    {
                        self.airbend(object);
                        bent = true;
                    }
                }
                if bent {
                    self.record_bending(frame.controller, crate::flow::turn::Bending::Air);
                }
                None
            }
            Effect::GrantGraveyardCastThisTurn { flashback, .. } => {
                if let Target::GraveyardCard(object) = self.present_primary_target(frame)? {
                    let grant = PlayGrant {
                        flashback: *flashback,
                        origin: frame
                            .source_ref()
                            .map(|source| self.state.source_base(source).0),
                        this_turn: true,
                        ..PlayGrant::new(object, frame.controller)
                    };
                    self.state.cast_permissions.grant(grant);
                }
                None
            }
            Effect::OpponentsCastOnlyFromHand => {
                self.restrict_opponent_to_hand(frame.controller);
                None
            }
            Effect::ExileLinked {
                chooser,
                zone,
                predicate,
                up_to,
                grant,
                ..
            } => {
                let Some(Target::Player(owner)) = self.present_primary_target(frame) else {
                    return None;
                };
                let decision = Decision::ExileLinked {
                    player: match chooser {
                        LinkedChooser::Controller => frame.controller,
                        LinkedChooser::TargetPlayer => owner,
                    },
                    owner,
                    zone: *zone,
                    predicate: predicate.clone(),
                    up_to: *up_to,
                    grant: *grant,
                    grantee: frame.controller,
                    source: frame.source_ref(),
                };
                (!self.linked_exile_candidates(&decision).is_empty()).then_some(decision)
            }
            Effect::CastFromGraveyard {
                predicate,
                exile_instants_and_sorceries,
            } => {
                let decision = Decision::CastDuringResolution {
                    player: frame.controller,
                    paid: true,
                    spends: None,
                    offer: ResolutionCastOffer::Graveyard {
                        predicate: predicate.clone(),
                        exile_instants_and_sorceries: *exile_instants_and_sorceries,
                    },
                };
                (!self.resolution_cast_candidates(&decision).is_empty()).then_some(decision)
            }
            Effect::CastFromHandFree { predicate } => {
                let decision = Decision::CastDuringResolution {
                    player: frame.controller,
                    paid: false,
                    offer: ResolutionCastOffer::Hand {
                        predicate: predicate.clone(),
                    },
                    spends: None,
                };
                (!self.resolution_cast_candidates(&decision).is_empty()).then_some(decision)
            }
            Effect::ExileUntilNonlandCastFree { below } => {
                let player = frame.controller;
                let mut nonland = None;
                while let Some(&top) = self
                    .state
                    .zones
                    .zone_objects(ZoneType::Library, player)
                    .last()
                {
                    let top = self.move_object(top, ZoneType::Exile);
                    if !self.state[top].types.is_land() {
                        nonland = Some(top);
                        break;
                    }
                }
                let object_id = nonland?;
                let below = self.effect_value(below, frame);
                let decision = Decision::CastDuringResolution {
                    player,
                    paid: false,
                    offer: ResolutionCastOffer::Exiled {
                        object: object_id,
                        below,
                    },
                    spends: None,
                };
                if !self.resolution_cast_candidates(&decision).is_empty() {
                    return Some(decision);
                }
                self.reveal_object_to_hand(object_id, player, ZoneEntry::default());
                None
            }
            Effect::CumulativeUpkeep { each } => {
                let ages = self
                    .source_permanent_id(frame)
                    .and_then(|id| self.state.permanents[id].as_ref())?
                    .counter_count(CounterKind::Age);
                let cost = (0..ages).fold(ManaCost::default(), |cost, _| cost.plus(each));
                Some(Decision::Payment(PaymentProgress::new(
                    frame.controller,
                    CostComponent::mana(cost),
                    Vec::new(),
                    vec![Effect::sacrifice_this()],
                )))
            }
            Effect::ExileEachLibraryTopCastFree => {
                let mut objects = Vec::new();
                for owner in [PlayerId(0), PlayerId(1)] {
                    if let Some(card) = self.library_top_card(owner) {
                        objects.push(self.move_object(self.state.object_of(card), ZoneType::Exile));
                    }
                }
                let decision = Decision::CastDuringResolution {
                    player: frame.controller,
                    offer: ResolutionCastOffer::AmongExiled { objects },
                    paid: false,
                    spends: None,
                };
                (!self.resolution_cast_candidates(&decision).is_empty()).then_some(decision)
            }
            Effect::Cascade => {
                let player = frame.controller;
                let below = frame
                    .source()
                    .map_or(0, |object_id| self.object_mana_value(object_id));
                let mut exiled = Vec::new();
                let mut hit = None;
                while let Some(&top) = self
                    .state
                    .zones
                    .zone_objects(ZoneType::Library, player)
                    .last()
                {
                    let top = self.move_object(top, ZoneType::Exile);
                    exiled.push(top);
                    if !self.state[top].types.is_land() && self.object_mana_value(top) < below {
                        hit = Some(top);
                        break;
                    }
                }
                let decision = hit
                    .map(|object_id| Decision::CastDuringResolution {
                        player,
                        paid: false,
                        offer: ResolutionCastOffer::Cascade {
                            object: object_id,
                            exiled: exiled.clone(),
                            below,
                        },
                        spends: None,
                    })
                    .filter(|decision| !self.resolution_cast_candidates(decision).is_empty());
                if decision.is_none() {
                    self.put_exiled_on_bottom_in_random_order(exiled);
                }
                decision
            }
            Effect::Reveal { what } => {
                for object in self.select(what, frame) {
                    self.emit(GameEvent::CardRevealed {
                        owner: self.state[object].owner,
                        definition: self.state[object].definition_id,
                    });
                }
                frame.selected_revealed |= *what == ObjectSelector::Selected;
                None
            }
            Effect::RevealUntilToHand { predicate } => {
                self.reveal_until_to_hand(frame.controller, predicate);
                None
            }
            Effect::RevealHandDiscardChosen { predicate, .. } => {
                let Some(Target::Player(owner)) = self.present_primary_target(frame) else {
                    return None;
                };
                self.reveal_hand(owner);
                let decision = Decision::DiscardFromRevealedHand {
                    player: frame.controller,
                    owner,
                    predicate: predicate.clone(),
                };
                (!self.revealed_hand_candidates(&decision).is_empty()).then_some(decision)
            }
            Effect::SearchDistinctNames {
                predicate,
                count,
                end,
            } => {
                self.emit(GameEvent::LibrarySearched {
                    player: frame.controller,
                });
                Some(Decision::SearchDistinctNames {
                    player: frame.controller,
                    predicate: predicate.clone(),
                    remaining: *count,
                    selected: Vec::new(),
                    end: *end,
                })
            }
            Effect::OwnerPutsIntoLibrary { target, top } => {
                let Some(Target::Permanent(id)) = self.present_primary_target(frame) else {
                    return None;
                };
                let object_id = self.state.permanents[id].as_ref()?.object;
                let modes = if *top {
                    [
                        crate::state::ability::LibraryEnd::Top,
                        crate::state::ability::LibraryEnd::Bottom,
                    ]
                    .map(|end| {
                        vec![Effect::put_in_library(
                            ObjectSelector::Target(target.clone()),
                            end,
                        )]
                    })
                    .to_vec()
                } else {
                    [true, false]
                        .map(|second_from_top| {
                            vec![Effect::PlaceTargetInLibrary { second_from_top }]
                        })
                        .to_vec()
                };
                Some(Decision::Modal {
                    player: self.state[object_id].owner,
                    modes,
                })
            }
            Effect::PlaceTargetInLibrary { second_from_top } => {
                let Some(Target::Permanent(id)) = self.present_primary_target(frame) else {
                    return None;
                };
                let permanent = self.state.permanents[id].as_ref()?;
                let object_id = permanent.object;
                frame.batch = Some(self.event_batch(
                    EventBinding::PlaceInLibrary {
                        second_from_top: *second_from_top,
                    },
                    [self.zone_move_proposal(object_id, ZoneType::Library, Default::default())],
                ));
                None
            }
            Effect::ExileTopCardsToPlay {
                count,
                until,
                face_down,
                condition,
            } => {
                self.exile_top_cards_to_play(
                    frame.controller,
                    *count,
                    *until,
                    *face_down,
                    condition.as_ref(),
                );
                None
            }
            Effect::ShuffleGraveyardIntoLibrary => {
                let player = frame.controller;
                let graveyard = self
                    .state
                    .zones
                    .zone_objects(ZoneType::Graveyard, player)
                    .to_vec();
                for object_id in graveyard {
                    self.move_object(object_id, ZoneType::Library);
                }
                self.shuffle_library(player);
                None
            }
            Effect::DealDamage { amount, what, by } => {
                let amount = self.effect_value(amount, frame);
                let source = match by {
                    Some(by) => self.damage_source(by, frame),
                    None => frame.source_ref(),
                };
                let events = self
                    .resolve_affected(what, frame)
                    .into_iter()
                    .map(|affected| ProposedEvent::Damage {
                        source,
                        target: match affected {
                            Affected::Player(player) => ProposedDamageTarget::Player(player),
                            Affected::Object(object) => ProposedDamageTarget::Object(object),
                        },
                        amount,
                        combat: false,
                    });
                frame.batch = Some(self.event_batch(EventBinding::Damage { result: None }, events));
                None
            }
            Effect::AddMana {
                mana,
                amount,
                until_end_of_combat,
            } => {
                let count = self.effect_value(amount, frame).max(0);
                self.invalidate_mana_cache(frame.controller);
                let origin = frame.source().map(|source| self.mana_origin(source));
                let pool = if *until_end_of_combat {
                    self.state.combat_mana_pool_mut(frame.controller)
                } else {
                    self.state.mana_pool_mut(frame.controller)
                };
                for _ in 0..count {
                    pool.add(mana, origin);
                }
                if *until_end_of_combat {
                    self.record_bending(frame.controller, crate::flow::turn::Bending::Fire);
                }
                None
            }
            Effect::LoseLife { player, amount } => {
                let amount = self.effect_value(amount, frame).max(0);
                let events = self
                    .selected_players(player, frame)
                    .into_iter()
                    .map(|player| ProposedEvent::LifeChange {
                        player,
                        delta: -amount,
                    });
                frame.batch = Some(self.event_batch(EventBinding::LifeLost, events));
                None
            }
            Effect::FightLastTarget => {
                // CR 701.14b: if either creature is an illegal target, has
                // left the battlefield or is no longer a creature, neither
                // deals damage.
                let [first, .., last] = frame.targets.as_slice() else {
                    return None;
                };
                let (Target::Permanent(first), Target::Permanent(last)) =
                    (first.target, last.target)
                else {
                    return None;
                };
                if !frame.targets.iter().all(|slot| slot.legal)
                    || !self.permanent_is_battlefield_creature(first)
                    || !self.permanent_is_battlefield_creature(last)
                {
                    return None;
                }
                let first_ref = self.permanent_object(first)?;
                let last_ref = self.permanent_object(last)?;
                // Each deals damage equal to its power at once (CR 701.14a).
                let first_power = self.characteristics(first).power;
                let last_power = self.characteristics(last).power;
                frame.batch = Some(self.event_batch(
                    EventBinding::Damage { result: Some(0) },
                    [
                        ProposedEvent::Damage {
                            source: Some(first_ref),
                            target: ProposedDamageTarget::Object(last_ref),
                            amount: first_power,
                            combat: false,
                        },
                        ProposedEvent::Damage {
                            source: Some(last_ref),
                            target: ProposedDamageTarget::Object(first_ref),
                            amount: last_power,
                            combat: false,
                        },
                    ],
                ));
                None
            }
            Effect::ShuffleLibrary => {
                self.shuffle_library(frame.controller);
                None
            }
            Effect::CounterSpell { what, exile } => {
                let to = if *exile {
                    ZoneType::Exile
                } else {
                    ZoneType::Graveyard
                };
                for spell in self.select(what, frame) {
                    self.counter_spell(spell, frame.source(), to);
                }
                None
            }
            Effect::CounterAbility { what } => {
                debug_assert!(
                    matches!(what, ObjectSelector::Target(_)),
                    "a stack ability is reached only as the clause's target"
                );
                if let Target::StackAbility(ability) = self.present_primary_target(frame)? {
                    self.counter_ability(ability, frame.source());
                }
                None
            }
            Effect::CreateTokenCopy { of, exceptions } => {
                let exceptions: Vec<_> = exceptions
                    .iter()
                    .map(|exception| exception.resolve(|value| self.effect_value(value, frame)))
                    .collect();
                let copied: Vec<ObjectId> = match of {
                    ObjectSelector::This => vec![frame.source()?],
                    // The target as it was chosen, even once the resolution
                    // has moved it ("exile ..., create a copy of it").
                    ObjectSelector::Target(_) => vec![match frame.primary_target()? {
                        Target::Permanent(id) => {
                            self.present_primary_target(frame)?;
                            self.state.permanents[id].as_ref()?.object
                        }
                        // Copied once this resolution has exiled it: the
                        // exiled object the targeted card became (CR 400.7j).
                        Target::GraveyardCard(object) => {
                            self.state.zones.successor(object).filter(|exiled| {
                                self.state.zones.zone_of(*exiled) == Some(ZoneType::Exile)
                            })?
                        }
                        _ => return None,
                    }],
                    selector => self.select(selector, frame),
                };
                if copied.is_empty() {
                    return None;
                }
                // Every copiable value is read before any token exists, so
                // the copies are created simultaneously (CR 707.2).
                let events = copied
                    .into_iter()
                    .map(|copied| ProposedEvent::CreateTokens {
                        definition: Box::new(self.copy_source(copied, exceptions.clone())),
                        controller: frame.controller,
                        count: 1,
                        entry: ZoneEntry::default(),
                    });
                frame.batch =
                    Some(self.event_batch(EventBinding::Created { attacking: false }, events));
                None
            }
            Effect::OptionalPayment {
                cost,
                if_paid,
                if_declined,
            } => Some(Decision::Payment(PaymentProgress::new(
                frame.controller,
                CostComponent::resolving(cost),
                if_paid.clone(),
                if_declined.clone(),
            ))),
            Effect::CastLibraryTopFree => self.library_top_free_cast(frame.controller, None),
            Effect::WhenYouDo { effects } => {
                // A spell's reflexive trigger has its card as its source.
                let trigger = self.enqueue_inline_trigger(
                    frame.source()?,
                    frame.source_ref(),
                    frame.controller,
                    effects.clone(),
                    frame.context_target,
                );
                trigger.remembered = frame.remembered.clone();
                trigger.x = frame.x;
                trigger.spent = frame.spent.clone();
                trigger.player = frame.triggering_player;
                None
            }
            Effect::MayOnceEachTurn { effects } => {
                let key = (frame.source_ref()?, frame.ability_face?, frame.ability?);
                if self
                    .state
                    .turn
                    .ability_triggers_this_turn
                    .contains_key(&key)
                {
                    return None;
                }
                // An option that is itself a free cast is spent by casting.
                if let [Effect::CastLibraryTopFree] = effects.as_slice() {
                    return self.library_top_free_cast(frame.controller, Some(key));
                }
                let mut payment = PaymentProgress::new(
                    frame.controller,
                    CostComponent::mana(ManaCost::default()),
                    effects.clone(),
                    Vec::new(),
                );
                payment.spends = Some(key);
                Some(Decision::Payment(payment))
            }
            Effect::If {
                condition,
                then,
                otherwise,
            } => {
                let branch = if self.check_condition(condition, frame.controller, Some(frame)) {
                    then
                } else {
                    otherwise
                };
                frame.prepend_effects(branch.iter().cloned());
                None
            }
            Effect::DrawCards { count } => {
                // Drawing from an empty library sets `drew_when_empty`; the player
                // loses via state-based actions (CR 704.5c), same as the draw step.
                let count = self.effect_value(count, frame).max(0) as usize;
                self.draw_cards(frame.controller, count);
                None
            }
            Effect::CreateToken {
                token,
                count,
                tapped_and_attacking,
                tapped,
            } => {
                let count = self.effect_value(count, frame).max(0);
                frame.batch = Some(self.event_batch(
                    EventBinding::Created {
                        attacking: *tapped_and_attacking,
                    },
                    [ProposedEvent::CreateTokens {
                        definition: Box::new(TokenSource::Registered(token.clone())),
                        controller: frame.controller,
                        count,
                        entry: ZoneEntry {
                            tapped: *tapped || *tapped_and_attacking,
                            ..Default::default()
                        },
                    }],
                ));
                None
            }
            Effect::AttachSourceTo { what } => {
                for permanent_id in self.selected_permanents(what, frame) {
                    if let FrameCompletion::Spell { attach_to, .. } = &mut frame.completion {
                        *attach_to = self.permanent_object(permanent_id);
                    } else {
                        self.attach_source_to(frame, permanent_id);
                    }
                }
                None
            }
            Effect::ForPlayer { player, effects } => {
                // Build the queue backwards. Each player's effects restore the
                // next controller, including when a choice suspends the frame.
                for player in self.selected_players(player, frame).into_iter().rev() {
                    frame
                        .queue
                        .push_front(Effect::RestoreController(frame.controller));
                    frame.prepend_effects(effects.iter().cloned());
                    frame.controller = player;
                }
                None
            }
            Effect::RestoreController(player) => {
                frame.controller = *player;
                None
            }
            Effect::MayExileTriggeringWithSource => {
                // CR 400.7e: the card the dying creature became, while it is
                // still that graveyard object; `Move` exiles it as the frame's
                // target and retains the link.
                let Some(Target::Permanent(id)) = frame.context_target else {
                    return None;
                };
                let died = self.departed_permanent_lki(id)?.object;
                let became = self.state.zones.successor(died)?;
                if self.state.card_of(became).is_none()
                    || self.state.zones.zone_of(became) != Some(ZoneType::Graveyard)
                {
                    return None;
                }
                frame.targets = vec![crate::flow::decision::ResolutionTarget {
                    target: Target::GraveyardCard(became),
                    requirement: 0,
                    legal: true,
                }];
                Some(Decision::Modal {
                    player: frame.controller,
                    modes: vec![
                        vec![Effect::exile_linked(
                            ObjectSelector::Target(TargetSpec::GraveyardCard {
                                predicate: CardPredicate::default(),
                            }),
                            LinkedExile::WithSource,
                        )],
                        vec![],
                    ],
                })
            }
            Effect::ChooseExiledWithSource { predicate } => {
                let source = frame.source_ref()?;
                let modes: Vec<Vec<Effect>> = self
                    .state
                    .exile_links
                    .iter()
                    .filter(|link| link.source == source)
                    .filter(|link| self.state.object(link.exiled_object).is_ok())
                    .map(|link| link.exiled_object)
                    .filter(|object_id| {
                        self.state.zones.zone_of(*object_id) == Some(ZoneType::Exile)
                            && predicate.matches_object(&self.state[*object_id])
                    })
                    .map(|object_id| {
                        vec![Effect::HaveAbilitiesOf {
                            definition: self.state[object_id].definition_id,
                        }]
                    })
                    .collect();
                (!modes.is_empty()).then_some(Decision::Modal {
                    player: frame.controller,
                    modes,
                })
            }
            Effect::HaveAbilitiesOf { definition } => {
                let id = self.source_permanent_id(frame)?;
                let affected = self.affected_permanent(id)?;
                let timestamp = self.state.id_gen.next_id();
                self.state
                    .effects
                    .push(crate::state::effect::ContinuousEffect {
                        timestamp,
                        source: Some(*definition),
                        controller: frame.controller,
                        affected,
                        modification: Modification::HasAbilitiesOfSource,
                        duration: EffectDuration::Indefinite,
                        source_object: None,
                    });
                self.invalidate_mana_cache(frame.controller);
                None
            }
            Effect::NameAndExileFromTargetPlayer { count } => {
                let names = default_content_pack().name_candidates();
                Some(Decision::NameCard {
                    scope: names.root(),
                    names,
                    player: frame.controller,
                    count: *count,
                })
            }
            Effect::ExileNamed { name, count } => {
                let Some(Target::Player(player)) = self.present_primary_target(frame) else {
                    return None;
                };
                self.remember_hand(player);
                self.emit(GameEvent::LibrarySearched { player });
                Some(Decision::SearchNamed {
                    player: frame.controller,
                    owner: player,
                    name: name.clone(),
                    remaining: *count,
                })
            }
            Effect::OptionalManaPaymentX { if_paid } => Some(Decision::ChoosePaymentX {
                player: frame.controller,
                if_paid: if_paid.clone(),
            }),
            Effect::EachPlayerChoosesToExile { predicate } => self.choose_to_exile(
                frame.controller,
                frame.controller,
                predicate.clone(),
                Vec::new(),
                frame,
            ),
            Effect::EachPlayerKeepsCreatures { max_total_power } => self.keep_creatures(
                self.active_player(),
                *max_total_power,
                None,
                Vec::new(),
                frame,
            ),
            Effect::EachPlayerSacrificesCreatures { count } => self.keep_creatures(
                self.active_player(),
                i32::MAX,
                Some(*count),
                Vec::new(),
                frame,
            ),
            Effect::CreateDelayedTrigger {
                event,
                frequency,
                expiry,
                remember,
                effects,
            } => {
                let remembered = remember
                    .as_ref()
                    .map_or_else(Vec::new, |what| self.capture_bindings(what, frame));
                let events = match event {
                    DelayedEvent::Dies => remembered
                        .iter()
                        .map(|binding| binding.object)
                        .filter(|object| self.lookup_current_permanent(*object).is_ok())
                        .map(|watched| DelayedTriggerEvent::Leaves {
                            watched,
                            or_exiled: false,
                        })
                        .collect(),
                    DelayedEvent::CreatureDies => vec![DelayedTriggerEvent::CreatureDies],
                    DelayedEvent::EndStep { your_turn } => vec![DelayedTriggerEvent::EndStep {
                        your_turn: *your_turn,
                    }],
                    DelayedEvent::YouCastSpell(predicate) => {
                        vec![DelayedTriggerEvent::YouCastSpell(predicate.clone())]
                    }
                    DelayedEvent::Upkeep => vec![DelayedTriggerEvent::Upkeep],
                };
                for event in events {
                    self.add_delayed_trigger(DelayedTrigger {
                        remembered: remembered.clone(),
                        event,
                        controller: frame.controller,
                        source_identity: crate::state::game_object::AbilitySource::new(
                            frame.source()?,
                            frame.source_ref(),
                        ),
                        effects: effects.clone(),
                        frequency: *frequency,
                        expiry: *expiry,
                    });
                }
                None
            }
            Effect::GainLife { amount } => {
                let amount = self.effect_value(amount, frame);
                self.gain_life(frame.controller, amount);
                None
            }
            Effect::OnNthResolutionThisTurn { n, effect } => {
                if frame.resolutions_this_turn == *n {
                    frame.queue.push_front((**effect).clone());
                }
                None
            }
            Effect::Scry { count } => {
                if *count > 0 {
                    frame.complete_with_event(GameEvent::Scried {
                        player: frame.controller,
                    });
                }
                let remaining = self.library_top_objects(frame.controller, *count);
                if remaining.is_empty() {
                    return None;
                }
                self.remember_library_objects(frame.controller, &remaining);
                Some(Decision::Scry {
                    player: frame.controller,
                    looked: remaining.clone(),
                    remaining,
                    top: Vec::new(),
                    bottom: Vec::new(),
                })
            }
            Effect::Surveil { count } => {
                if *count > 0 {
                    frame.complete_with_event(GameEvent::Surveilled {
                        player: frame.controller,
                    });
                }
                let looked = self.library_top_objects(frame.controller, *count);
                self.remember_library_objects(frame.controller, &looked);
                (!looked.is_empty()).then_some(Decision::Surveil {
                    player: frame.controller,
                    looked,
                    graveyard: Vec::new(),
                })
            }
            Effect::LookAndSelect {
                random_order,
                reveal,
                to_battlefield,
                look,
                min_select,
                max_select,
                predicate,
                max_mana_value,
            } => {
                let look = self.effect_value(look, frame).max(0) as usize;
                let looked = self.library_top_objects(frame.controller, look);
                if looked.is_empty() {
                    return None;
                }
                self.remember_library_objects(frame.controller, &looked);
                // The decision carries the bound as resolved now.
                let mut predicate = predicate.clone();
                if let Some(value) = max_mana_value {
                    let bound =
                        u8::try_from(self.effect_value(value, frame).max(0)).unwrap_or(u8::MAX);
                    predicate.max_mana_value = Some(
                        predicate
                            .max_mana_value
                            .map_or(bound, |printed| printed.min(bound)),
                    );
                }
                Some(Decision::LookAndSelect {
                    random_order: *random_order,
                    reveal: *reveal,
                    to_battlefield: *to_battlefield,
                    player: frame.controller,
                    looked,
                    predicate,
                    selected: 0,
                    min_select: *min_select,
                    max_select: *max_select,
                })
            }
            Effect::PutTopCardsInHand { count } => {
                for object_id in self.library_top_objects(frame.controller, *count) {
                    self.move_object(object_id, ZoneType::Hand);
                }
                self.invalidate_mana_cache(frame.controller);
                None
            }
            Effect::Learn => Some(Decision::Learn {
                player: frame.controller,
            }),
            // A trigger's mode is chosen as it is put on the stack, which
            // replaces this effect with the chosen mode's effects.
            Effect::TriggerModes { .. } => {
                unreachable!("trigger modes are chosen at placement")
            }
            Effect::Modal { modes } => {
                if modes.is_empty() {
                    return None;
                }
                Some(Decision::Modal {
                    player: frame.controller,
                    modes: modes.clone(),
                })
            }
            Effect::CounterUnlessPays { cost } => {
                // The spell or ability may already have left the stack (e.g.
                // it was countered in response) — the effect does nothing.
                let index = self.stack_index_of(frame.primary_target()?)?;
                let spell_controller = self.state.stack_objects[index].controller();
                let mut payment = PaymentProgress::new(
                    spell_controller,
                    CostComponent::resolving(cost),
                    Vec::new(),
                    Vec::new(),
                );
                payment.counters_on_decline = frame.primary_target();
                Some(Decision::Payment(payment))
            }
            Effect::TargetCreaturesDealPowerDamageToLastTarget => {
                let (victim_slot, attackers) = frame.targets.split_last()?;
                if !victim_slot.legal {
                    return None;
                }
                let Target::Permanent(victim) = victim_slot.target else {
                    return None;
                };
                let Some(target) = self.permanent_object(victim) else {
                    return None;
                };
                let mut events = Vec::new();
                for attacker in attackers.iter().filter(|slot| slot.legal) {
                    let Target::Permanent(attacker_id) = attacker.target else {
                        continue;
                    };
                    // Eligibility was fixed at entry; damage still requires
                    // the original battlefield objects (CR 400.7).
                    let Some(attacker_ref) = self.permanent_object(attacker_id) else {
                        continue;
                    };
                    let power = self.layer_view(attacker_id).power;
                    events.push(ProposedEvent::Damage {
                        source: Some(attacker_ref),
                        target: ProposedDamageTarget::Object(target),
                        amount: power,
                        combat: false,
                    });
                }
                frame.batch = Some(self.event_batch(EventBinding::None, events));
                None
            }
            Effect::Earthbend { count, what } => {
                let count = self.effect_value(count, frame);
                let selected = self.selected_permanents(what, frame);
                let bent = !selected.is_empty();
                let mut events = Vec::new();
                for permanent_id in selected {
                    let Some(watched) = self.permanent_object(permanent_id) else {
                        continue;
                    };
                    let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
                        continue;
                    };
                    let controller = permanent.controller;
                    self.animate_permanent(permanent_id, frame.source(), controller);
                    // Earthbend 0 still animates the land and creates its return trigger.
                    events.push(ProposedEvent::CounterChange {
                        provenance: crate::flow::proposed_event::CounterProvenance::new(
                            crate::flow::proposed_event::EventCause::Effect,
                        ),
                        target: watched,
                        kind: CounterKind::PlusOnePlusOne,
                        delta: count,
                        by: Some(frame.controller),
                    });
                    // Delayed trigger: when it dies or is exiled, return it to
                    // the battlefield tapped (CR 603.7 one-shot).
                    self.add_delayed_trigger(DelayedTrigger {
                        remembered: Vec::new(),
                        event: DelayedTriggerEvent::Leaves {
                            watched,
                            or_exiled: true,
                        },
                        controller,
                        source_identity: crate::state::game_object::AbilitySource::new(
                            watched,
                            Some(watched),
                        ),
                        effects: vec![Effect::Move {
                            unrevealed: false,
                            library_end: None,
                            what: ObjectSelector::This,
                            to: ZoneType::Battlefield,
                            tapped: true,
                            transformed: false,
                            under: Some(PlayerSelector::Owner),
                            with_counter: None,
                            linked: None,
                        }],
                        frequency: DelayedFrequency::Once,
                        expiry: DelayedExpiry::Never,
                    });
                }
                let binding = if bent {
                    EventBinding::Bending {
                        player: frame.controller,
                        kind: crate::flow::turn::Bending::Earth,
                    }
                } else {
                    EventBinding::None
                };
                frame.batch = Some(self.event_batch(binding, events));
                None
            }
            Effect::CastExiledWithSource { predicate } => {
                let decision = Decision::CastDuringResolution {
                    player: frame.controller,
                    paid: true,
                    offer: ResolutionCastOffer::ExiledWith {
                        source: frame.source_ref()?,
                        predicate: predicate.clone(),
                    },
                    spends: None,
                };
                (!self.resolution_cast_candidates(&decision).is_empty()).then_some(decision)
            }
            Effect::AddOneColorManaValue { .. } => {
                frame.prepend_effects(
                    self.current_mana_output(effect, frame.controller)
                        .into_iter(),
                );
                None
            }
            Effect::ForEachTarget { effects } => {
                frame.iteration = Some(TargetIteration {
                    remaining: frame.active_targets().collect(),
                    effects: effects.clone(),
                });
                None
            }
        }
    }

    /// `player` looks at their library's top card and may cast it for free.
    fn library_top_free_cast(
        &mut self,
        player: PlayerId,
        spends: Option<OnceEachTurnKey>,
    ) -> Option<Decision> {
        let object = *self
            .state
            .zones
            .zone_objects(ZoneType::Library, player)
            .last()?;
        // Even declining this offer leaves evidence about the next draw.
        self.state.library_order_pins_mut(player).insert(object);
        let decision = Decision::CastDuringResolution {
            player,
            paid: false,
            offer: ResolutionCastOffer::LibraryTop { object },
            spends,
        };
        (!self.resolution_cast_candidates(&decision).is_empty()).then_some(decision)
    }

    // -----------------------------------------------------------------------
    // Object selection.

    /// The object a `DealDamage { by }` names as the damage's source: the
    /// exact current object, else a departed triggering permanent through its
    /// last known information (CR 608.2h).
    fn damage_source(&self, by: &ObjectSelector, frame: &EffectFrame) -> Option<ObjectId> {
        self.select(by, frame)
            .into_iter()
            .next()
            .or_else(|| match (by, frame.context_target) {
                (ObjectSelector::Triggering, Some(Target::Permanent(id))) => {
                    self.departed_permanent_lki(id).map(|lki| lki.object)
                }
                _ => None,
            })
    }

    /// The objects a selector names as the instruction resolves. A `Chosen`
    /// selector names the object the resolving player chose for this
    /// instruction (`execute_frame_effect` suspended on the choice first).
    pub(crate) fn select(&self, what: &ObjectSelector, frame: &EffectFrame) -> Vec<ObjectId> {
        let permanent = |id: Option<PermanentId>| {
            id.and_then(|id| self.permanent_object(id))
                .into_iter()
                .collect()
        };
        match what {
            ObjectSelector::ExactSource => frame
                .source_ref()
                .and_then(|source| self.state.live(source))
                .into_iter()
                .collect(),
            ObjectSelector::This => self.source_object(frame).into_iter().collect(),
            ObjectSelector::Target(_) => self
                .present_primary_target(frame)
                .and_then(|target| self.target_object(target))
                .into_iter()
                .collect(),
            ObjectSelector::Targets(_) => frame
                .active_targets()
                .filter(|slot| self.target_is_present(slot.target))
                .filter_map(|slot| self.target_object(slot.target))
                .collect(),
            ObjectSelector::TargetAndSameName(_) => {
                let Some(Target::Permanent(target)) = self.present_primary_target(frame) else {
                    return Vec::new();
                };
                let named = CardPredicate {
                    named: Some(self.layer_view(target).name.clone()),
                    ..Default::default()
                };
                std::iter::once(target)
                    .chain(
                        [PlayerId(0), PlayerId(1)]
                            .into_iter()
                            .flat_map(|player| self.iter_battlefield_permanents(player))
                            .filter(|id| *id != target)
                            .filter(|id| self.permanent_matches_predicate(*id, &named)),
                    )
                    .filter_map(|id| self.permanent_object(id))
                    .collect()
            }
            ObjectSelector::Triggering => match frame.context_target {
                Some(Target::Permanent(id)) => permanent(Some(id)),
                Some(Target::GraveyardCard(object)) => {
                    self.state.live(object).into_iter().collect()
                }
                _ => Vec::new(),
            },
            ObjectSelector::Attached => permanent(self.frame_attached(frame)),
            ObjectSelector::CreatedThisWay => frame
                .created
                .iter()
                .copied()
                .filter(|object| self.lookup_current_permanent(*object).is_ok())
                .collect(),
            ObjectSelector::Grantor => frame
                .grantor
                .filter(|grantor| self.lookup_current_permanent(*grantor).is_ok())
                .into_iter()
                .collect(),
            ObjectSelector::Paid(how) => frame
                .spent
                .paid_as(*how)
                .filter(|object| self.state.live(*object).is_some())
                .collect(),
            ObjectSelector::Remembered => frame
                .remembered
                .iter()
                .map(|binding| binding.object)
                .filter(|object| self.state.live(*object).is_some())
                .collect(),
            ObjectSelector::Matching { of, predicate } => self
                .select(of, frame)
                .into_iter()
                .filter(|object| self.object_matches_predicate(*object, predicate))
                .collect(),
            ObjectSelector::Moved | ObjectSelector::CardsThisWay | ObjectSelector::Selected => {
                (match what {
                    ObjectSelector::Moved => &frame.moved,
                    ObjectSelector::CardsThisWay => &frame.this_way,
                    ObjectSelector::Selected => &frame.selected,
                    _ => unreachable!(),
                })
                .iter()
                .copied()
                .filter(|object| self.state.live(*object).is_some())
                .collect()
            }
            ObjectSelector::Zone(_) | ObjectSelector::TargetPlayersZone { .. } => {
                let Some((owner, zone)) = self.selection_zone(what, frame) else {
                    return Vec::new();
                };
                self.selection_zone_cards(owner, zone)
                    .into_iter()
                    .filter_map(|card| self.state.live(self.state.object_of(card)))
                    .collect()
            }
            ObjectSelector::You => Vec::new(),
            ObjectSelector::Chosen { .. } => {
                debug_assert!(frame.chosen.is_some(), "a Chosen selector is chosen first");
                frame
                    .chosen
                    .filter(|object| self.lookup_current_permanent(*object).is_ok())
                    .into_iter()
                    .collect()
            }
            ObjectSelector::Each {
                of,
                predicate,
                except,
                max_mana_value,
                exact_mana_value,
            } => {
                let players = self.scope_players(
                    of,
                    frame.controller,
                    frame.source(),
                    self.present_primary_target(frame),
                );
                let excepted = self.excepted_permanent(*except, frame);
                let source = self.source_permanent_id(frame);
                let exact = exact_mana_value
                    .as_ref()
                    .map(|value| self.effect_value(value, frame));
                let max_mana_value = max_mana_value
                    .as_ref()
                    .map(|value| self.effect_value(value, frame));
                players
                    .into_iter()
                    .flat_map(|player| self.iter_battlefield_permanents(player))
                    .filter(|id| Some(*id) != excepted)
                    .filter(|id| self.permanent_matches_with_source(*id, predicate, source))
                    .filter(|id| {
                        exact.is_none_or(|value| self.permanent_mana_value(*id) == value)
                            && max_mana_value
                                .is_none_or(|max| self.permanent_mana_value(*id) <= max)
                    })
                    .filter_map(|id| self.permanent_object(id))
                    .collect()
            }
        }
    }

    pub(crate) fn scope_players(
        &self,
        scope: &Scope,
        controller: PlayerId,
        source: Option<ObjectId>,
        target: Option<Target>,
    ) -> Vec<PlayerId> {
        match scope {
            Scope::YouControl => vec![controller],
            Scope::OpponentsControl => vec![PlayerId(1 - controller.0)],
            Scope::DamagedBySourceInCombat => source
                .into_iter()
                .flat_map(|source| self.state.turn.damage_pairs.combat_players(source))
                .collect(),
            Scope::Battlefield => vec![PlayerId(0), PlayerId(1)],
            Scope::TargetPlayerControls => match target {
                Some(Target::Player(player)) => vec![player],
                _ => Vec::new(),
            },
        }
    }

    /// The permanent an `Except` leaves out: the source, or the clause's
    /// chosen target while it is a permanent.
    pub(crate) fn excepted_permanent(
        &self,
        except: Option<Except>,
        frame: &EffectFrame,
    ) -> Option<PermanentId> {
        match except? {
            Except::This => self.source_permanent_id(frame),
            Except::Target => match self.present_primary_target(frame) {
                Some(Target::Permanent(id)) => Some(id),
                _ => None,
            },
        }
    }

    /// A battlefield permanent's mana value (CR 202.3): that of its card;
    /// 0 without a mana cost.
    pub(super) fn permanent_mana_value(&self, id: PermanentId) -> i32 {
        self.state.permanents[id]
            .as_ref()
            .and_then(|permanent| self.state[permanent.object].mana_value_cost())
            .map_or(0, |cost| i32::from(cost.mana_value))
    }

    /// The battlefield permanents a selector names.
    fn selected_permanents(&self, what: &ObjectSelector, frame: &EffectFrame) -> Vec<PermanentId> {
        self.select(what, frame)
            .into_iter()
            .filter_map(|object| self.lookup_current_permanent(object).ok())
            .collect()
    }

    /// The counters on what a selector names, as it is or (for the source
    /// and the triggering permanent) as it last existed on the battlefield.
    fn selected_counters(&self, what: &ObjectSelector, frame: &EffectFrame) -> Counters {
        let departed = |source: ObjectId| self.object_lki(source).map(|lki| lki.counters.clone());
        let present = |id: PermanentId| {
            self.state.permanents[id]
                .as_ref()
                .map(|permanent| permanent.counters.clone())
        };
        match what {
            ObjectSelector::This => {
                frame
                    .source_ref()
                    .and_then(|source| match self.lookup_current_permanent(source) {
                        Ok(id) => present(id),
                        Err(_) => departed(source),
                    })
            }
            ObjectSelector::Triggering => match frame.context_target {
                Some(Target::Permanent(id)) => self.with_permanent_or_lki(
                    id,
                    |p| p.counters.clone(),
                    |lki| lki.counters.clone(),
                ),
                _ => None,
            },
            other => self
                .selected_permanents(other, frame)
                .first()
                .and_then(|id| present(*id)),
        }
        .unwrap_or_default()
    }

    /// [`ObjectSelector::This`]: the exact source object in any zone, or
    /// once it has left the battlefield the card it became in the graveyard
    /// or exile it went to (CR 400.7, 603.10a); a spell while on the stack.
    fn source_object(&self, frame: &EffectFrame) -> Option<ObjectId> {
        let object = frame.source()?;
        match frame.source_ref() {
            Some(source) => self.state.live(source).or_else(|| {
                // Only a departing battlefield source can refer to the card it
                // became this way. A graveyard activation keeps its exact source.
                self.object_lki(source)?;
                self.state.zones.successor(source).filter(|successor| {
                    matches!(
                        self.state.zones.zone_of(*successor),
                        Some(ZoneType::Graveyard | ZoneType::Exile)
                    )
                })
            }),
            None => self.stack_spell(object).is_some().then_some(object),
        }
    }

    /// The exact object a chosen target is, when it is a card object.
    fn target_object(&self, target: Target) -> Option<ObjectId> {
        match target {
            Target::Permanent(id) => self.permanent_object(id),
            Target::GraveyardCard(object) => Some(object),
            Target::StackSpell(object_id) => self.state.live(object_id),
            Target::Player(_) | Target::StackAbility(_) => None,
        }
    }

    pub(crate) fn source_permanent_id(&self, frame: &EffectFrame) -> Option<PermanentId> {
        frame
            .source_ref()
            .and_then(|source| self.lookup_current_permanent(source).ok())
    }

    /// Attach the source Equipment to a battlefield creature (CR 701.3). It
    /// does nothing once the Equipment has left the battlefield.
    pub(crate) fn attach_source_to(&mut self, frame: &EffectFrame, permanent_id: PermanentId) {
        let Some(object) = self.permanent_object(permanent_id) else {
            return;
        };
        if self
            .source_permanent_id(frame)
            .is_some_and(|source| self.protected_from_permanent(permanent_id, source))
        {
            return;
        }
        let timestamp = self.state.id_gen_mut().next_id();
        if let Some(equipment) = self.source_permanent_mut(frame) {
            equipment.attached_to = Some(object);
            // CR 613.7e: an attachment receives a new timestamp.
            equipment.timestamp = timestamp;
        }
    }

    fn source_permanent_mut(&mut self, frame: &EffectFrame) -> Option<&mut Permanent> {
        let permanent_id = self.source_permanent_id(frame)?;
        self.state.permanents[permanent_id].as_mut()
    }

    /// Copy the live appearance or its last battlefield values (CR 608.2h, 707.2).
    fn copy_source(&self, copied: ObjectId, exceptions: Vec<Modification>) -> TokenSource {
        if let Ok(row) = self.state.object(copied) {
            return TokenSource::Copy {
                definition_id: row.definition_id,
                definition: default_content_pack().object_row(row),
                front: row.front_face().cloned(),
                copy: row.copy.clone(),
                exceptions,
            };
        }
        let (definition_id, printed, kind) = self
            .state
            .objects
            .retained_definition(copied)
            .expect("copied object storage");
        let changed = self.object_lki(copied).and_then(|lki| {
            lki.copiable
                .as_deref()
                .map(|values| (lki.definition_id, values))
        });
        let definition_id = changed.map_or(definition_id, |(id, _)| id);
        TokenSource::Copy {
            definition_id,
            definition: crate::state::card::ObjectDefinition {
                key: "",
                scryfall_id: None,
                printed: changed
                    .map_or(printed, |(_, values)| &values.printed)
                    .clone(),
                is_token: kind != crate::state::game_object::ObjectKind::Card,
                back_face: default_content_pack()
                    .definition(definition_id)
                    .and_then(|row| row.back_face.clone()),
            },
            front: changed.and_then(|(_, values)| values.front.clone()),
            copy: changed.and_then(|(_, values)| values.copy.clone()),
            exceptions,
        }
    }

    #[cfg(test)]
    pub(crate) fn test_target_is_legal(
        &self,
        target: Target,
        spec: &TargetSpec,
        controller: PlayerId,
    ) -> bool {
        self.target_is_legal(
            target,
            spec,
            &TargetingSource {
                controller,
                colors: Default::default(),
                types: Default::default(),
            },
        )
    }

    /// Check whether a target is (still) legal for a given TargetSpec, from
    /// the perspective of `controller` ("you").
    pub(crate) fn target_is_legal(
        &self,
        target: Target,
        spec: &TargetSpec,
        source: &TargetingSource,
    ) -> bool {
        let controller = source.controller;
        match target {
            Target::Permanent(id) => {
                let Some(permanent) = self.state.permanents.get(id.0).and_then(|p| p.as_ref())
                else {
                    return false;
                };
                // CR 702.11b applies regardless of target wording. Recheck live
                // presence and hexproof at selection and resolution (608.2b).
                let view = self.layer_view(id);
                let opposing = permanent.controller != controller;
                let disallowed = (opposing && view.keywords.has(Keyword::Hexproof))
                    || view.static_abilities.iter().any(|ability| match ability {
                        StaticAbility::HexproofFrom(q) => opposing && source.matches(*q),
                        _ => ability.protects_from(|q| source.matches(q)),
                    });
                self.state.zones.zone_of(permanent.object) == Some(ZoneType::Battlefield)
                    && !self.is_phased_out(id)
                    && !disallowed
                    && self.permanent_matches_spec(id, spec, controller)
            }
            // Protection from everything: no spell or ability can target
            // the player, theirs included (CR 702.16b, 702.16j).
            Target::Player(player)
                if self.player_has(player, &Modification::ProtectionFromEverything)
                    || (player != controller
                        && self.player_has(player, &Modification::Hexproof)) =>
            {
                false
            }
            Target::Player(player) => match spec {
                TargetSpec::Player | TargetSpec::AnyTarget | TargetSpec::PlayerOrPlaneswalker => {
                    true
                }
                TargetSpec::Opponent => player != controller,
                _ => false,
            },
            Target::StackSpell(object) => {
                self.stack_spell(object).is_some_and(|spell| match spec {
                    TargetSpec::Spell | TargetSpec::CreatureOrSpell => true,
                    TargetSpec::SpellMatching { predicate } => {
                        self.object_matches_predicate(object, predicate)
                    }
                    TargetSpec::SpellOrPermanent { min_mana_value } => {
                        self.state[object].mana_value_with_x(spell.x) >= *min_mana_value
                    }
                    TargetSpec::SpellOrAbilityWithSingleTarget => spell.targets.len() == 1,
                    _ => false,
                })
            }
            Target::StackAbility(id) => {
                self.stack_index_of(Target::StackAbility(id))
                    .is_some_and(|index| match spec {
                        TargetSpec::Ability => true,
                        TargetSpec::SpellOrAbilityWithSingleTarget => {
                            self.state.stack_objects[index].targets().len() == 1
                        }
                        _ => false,
                    })
            }
            Target::GraveyardCard(object) => {
                let (predicate, you_own) = match spec {
                    TargetSpec::GraveyardCardYouOwn { predicate }
                    | TargetSpec::GraveyardCardYouOwnWithinSourcePower { predicate } => {
                        (predicate, true)
                    }
                    TargetSpec::GraveyardCard { predicate } => (predicate, false),
                    TargetSpec::GraveyardCardOpponentOwns { predicate } => {
                        if self
                            .state
                            .object(object)
                            .map_or(true, |row| row.owner == controller)
                        {
                            return false;
                        }
                        (predicate, false)
                    }
                    _ => return false,
                };
                self.state.card_of(object).is_some()
                    && self.state.zones.zone_of(object) == Some(ZoneType::Graveyard)
                    && (!you_own || self.state[object].owner == controller)
                    && self.object_matches_predicate(object, predicate)
            }
        }
    }

    /// Source-relative parts of a targeting clause, including a dying
    /// permanent's immediate graveyard successor for "another" (CR 400.7).
    pub(crate) fn target_meets_requirement(
        &self,
        target: Target,
        requirement: &TargetRequirement,
        targeting: &TargetingSource,
        source: AbilitySource,
    ) -> bool {
        if !self.target_is_legal(target, &requirement.spec, targeting) {
            return false;
        }
        if requirement.another
            && match target {
                Target::Permanent(id) => self.permanent_object(id) == source.reference(),
                Target::GraveyardCard(card) => {
                    card == source.object()
                        || self.state.zones.successor(source.object()) == Some(card)
                }
                _ => false,
            }
        {
            return false;
        }
        let TargetSpec::GraveyardCardYouOwnWithinSourcePower { .. } = requirement.spec else {
            return true;
        };
        let Target::GraveyardCard(object) = target else {
            return false;
        };
        let power = source
            .reference()
            .and_then(|object| self.permanent_power_or_lki(object))
            .unwrap_or_else(|| self.object_pt(source.object()).0);
        i32::from(self.object_mana_value(object)) <= power
    }

    /// Whether a battlefield permanent matches a spec's characteristics and
    /// controller relation, without targeting rules (hexproof): the
    /// targeting check, and an Aura's enchant legality (CR 704.5m).
    pub(crate) fn permanent_matches_spec(
        &self,
        id: PermanentId,
        spec: &TargetSpec,
        controller: PlayerId,
    ) -> bool {
        let Some(permanent) = self.state.permanents[id].as_ref() else {
            return false;
        };
        match spec {
            TargetSpec::Permanent { predicate } => self.permanent_matches_predicate(id, predicate),
            TargetSpec::PermanentYouControl { predicate } => {
                permanent.controller == controller
                    && self.permanent_matches_predicate(id, predicate)
            }
            TargetSpec::PermanentOpponentControls { predicate } => {
                permanent.controller != controller
                    && self.permanent_matches_predicate(id, predicate)
            }
            TargetSpec::AnyTarget => {
                let types = &self.characteristics(id).types;
                types.is_creature() || types.is_planeswalker() || types.is_battle()
            }
            TargetSpec::PlayerOrPlaneswalker => self.characteristics(id).types.is_planeswalker(),
            TargetSpec::Creature => self.characteristics(id).is_creature(),
            TargetSpec::CreatureYouControl => {
                permanent.controller == controller && self.characteristics(id).is_creature()
            }
            TargetSpec::CreatureOrSpell => self.characteristics(id).is_creature(),
            TargetSpec::CreatureOpponentControls => {
                permanent.controller != controller && self.characteristics(id).is_creature()
            }
            TargetSpec::LandYouControl => {
                permanent.controller == controller && self.characteristics(id).is_land()
            }
            TargetSpec::SpellOrPermanent { min_mana_value } => {
                self.object_mana_value(permanent.object) >= *min_mana_value
            }
            _ => false,
        }
    }

    /// The object's mana value; a spell on the stack counts its announced X
    /// (CR 202.3e).
    pub(crate) fn object_mana_value(&self, object: ObjectId) -> u8 {
        let x = self.stack_spell(object).map_or(0, |spell| spell.x);
        self.state[object].mana_value_with_x(x)
    }
}

#[cfg(test)]
#[path = "resolution_tests.rs"]
mod resolution_tests;
