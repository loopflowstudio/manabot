// resolution.rs
// Resolving stack objects. Effects execute through an EffectFrame so a
// resolution can suspend on a mid-resolution decision (see decision.rs).

use std::collections::VecDeque;

use crate::{
    flow::{
        decision::{
            Decision, EffectFrame, FrameFinalize, OptionalSacrifice, PaymentProgress,
            ResolutionCastOffer, ResolutionTarget,
        },
        event::GameEvent,
        game::Game,
        trigger::{DelayedTrigger, DelayedTriggerEvent, ExileLink},
        turn::OnceEachTurnKey,
    },
    state::{
        ability::{
            Ability, AffectedSpec, DelayedEvent, Effect, EffectValue, GraveyardReturn,
            LinkedChooser, PaymentCost, PlayerSelector, TargetRequirement, TargetSpec,
        },
        effect::{Affected, EffectDuration, Modification},
        game_object::{CardId, ObjectId, ObjectRef, PermanentId, PlayerId, Target},
        permanent::{CounterKind, Permanent},
        predicate::CardPredicate,
        stack_object::{
            ActivatedAbilityOnStack, SpellOnStack, StackObject, TriggeredAbilityOnStack,
        },
        zone::ZoneType,
    },
};

impl Game {
    /// A quantity read by a resolving instruction: it sees the frame's
    /// announced X and recorded excess damage.
    fn effect_value(&self, value: &EffectValue, frame: &EffectFrame) -> i32 {
        match value {
            EffectValue::X => i32::from(frame.x),
            EffectValue::TriggeringPower => match frame.context_target {
                Some(Target::Permanent(id)) if self.state.permanents[id].is_some() => {
                    self.effective_power(id)
                }
                Some(Target::Permanent(id)) => {
                    self.departed_permanent_lki(id).map_or(0, |lki| lki.power)
                }
                _ => 0,
            },
            EffectValue::ExcessDamage => frame.excess_damage,
            EffectValue::SourcePower => self
                .source_permanent_id(frame)
                .map_or(0, |id| self.effective_power(id)),
            EffectValue::AttackingMatching { predicate } => self
                .battlefield_permanents(frame.controller)
                .into_iter()
                .filter(|id| {
                    self.state.permanents[*id]
                        .as_ref()
                        .is_some_and(|permanent| permanent.attacking)
                        && self.permanent_matches_predicate(*id, predicate)
                })
                .count() as i32,
            EffectValue::Half(inner) => self.effect_value(inner, frame).div_euclid(2),
            _ => self.controller_value(value, frame.controller),
        }
    }

    /// A quantity relative to `controller` outside any resolution (a mana
    /// ability): X and excess damage are 0.
    pub(crate) fn controller_value(&self, value: &EffectValue, controller: PlayerId) -> i32 {
        match value {
            EffectValue::Fixed(value) => *value,
            EffectValue::X
            | EffectValue::ExcessDamage
            | EffectValue::SourcePower
            | EffectValue::TriggeringPower
            | EffectValue::AttackingMatching { .. } => 0,
            EffectValue::CreaturesWithCounters => self
                .battlefield_permanents(controller)
                .into_iter()
                .filter(|id| {
                    self.permanent_is_battlefield_creature(*id)
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
            EffectValue::PermanentsSacrificedThisTurn => {
                self.state.turn.history[controller.0].permanents_sacrificed as i32
            }
            EffectValue::OwnedPermanentsOpponentsControl => self
                .battlefield_permanents(PlayerId(1 - controller.0))
                .into_iter()
                .filter(|id| {
                    self.state.permanents[*id]
                        .as_ref()
                        .is_some_and(|permanent| {
                            self.state.cards[permanent.card].owner == controller
                        })
                })
                .count() as i32,
            EffectValue::Half(inner) => self.controller_value(inner, controller).div_euclid(2),
        }
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
        let card = spell.card;
        let card_ref = &self.state.cards[card];
        let effects =
            crate::state::ability::bind_effect_clauses(card_ref.spell_effects_for_mode(spell.mode));
        let requirements = card_ref.target_requirements_for_mode(spell.mode);

        let Some(targets) = self.resolution_targets(
            &spell.targets,
            &spell.target_req_indices,
            &requirements,
            spell.controller,
        ) else {
            self.move_card(card, ZoneType::Graveyard);
            self.emit(GameEvent::SpellCountered { card, by: None });
            return;
        };

        let frame = EffectFrame {
            source: Some(card),
            source_ref: self.current_object_ref(card),
            controller: spell.controller,
            resolutions_this_turn: 0,
            kicked: spell.kicked,
            x: spell.x,
            excess_damage: 0,
            created: None,
            this_way: Vec::new(),
            targets,
            active_requirement: 0,
            context_target: None,
            triggering_player: None,
            grantor: None,
            ability: None,
            queue: VecDeque::from(effects),
            finalize: FrameFinalize::Spell {
                card,
                attach_to: None,
            },
        };
        self.run_frame(frame);
    }

    /// Counter an activated or triggered ability: it is removed from the
    /// stack and does nothing (CR 701.6b).
    pub(crate) fn counter_ability(&mut self, id: ObjectId, by: Option<CardId>) {
        let Some(index) = self.find_ability_on_stack_index(id) else {
            return;
        };
        self.journal_stack();
        let ability = self.state.stack_objects.remove(index);
        let source_card = match &ability {
            StackObject::ActivatedAbility(ability) => ability.source_card,
            StackObject::TriggeredAbility(ability) => ability.source_card,
            StackObject::Spell(_) => unreachable!("an ability index"),
        };
        self.emit(GameEvent::AbilityCountered {
            ability: id,
            source_card,
            controller: ability.controller(),
            by,
        });
    }

    pub(crate) fn counter_spell(&mut self, card: CardId, by: Option<CardId>) {
        let Some(_) = self.find_spell_on_stack_index(card) else {
            return;
        };
        self.move_card(card, ZoneType::Graveyard);
        self.emit(GameEvent::SpellCountered { card, by });
    }

    fn resolve_activated_ability(&mut self, ability: ActivatedAbilityOnStack) {
        // A granted ability keeps its grantor's text after the grantor leaves.
        let definition = match ability.granted_by {
            Some(grant) => self.state.cards[CardId::from(grant.grantor.entity)]
                .attached_grants()
                .nth(grant.index),
            None => self.state.cards[ability.source_card]
                .activated_abilities
                .get(ability.ability_index),
        };
        let Some(definition) = definition.cloned() else {
            return;
        };

        let effects = definition.effects;
        let requirements = crate::state::ability::effect_target_requirements(&effects);
        // The object an Aura or Equipment was attached to as the ability was
        // activated (last known information once the source has left).
        let attached = ability
            .attached
            .and_then(|object| self.lookup_current_permanent(object).ok())
            .map(Target::Permanent);
        let Some(targets) =
            self.resolution_targets(&ability.targets, &[], &requirements, ability.controller)
        else {
            return;
        };

        // CR 113.7a: an activated ability exists independently of its source.
        // Source-relative effects still require the original battlefield object;
        // independent effects (such as drawing) resolve after that object leaves.
        let source_ref = Some(ability.source_ref);

        let frame = EffectFrame {
            source: Some(ability.source_card),
            source_ref,
            controller: ability.controller,
            resolutions_this_turn: 0,
            kicked: false,
            x: ability.x,
            excess_damage: 0,
            created: None,
            this_way: Vec::new(),
            targets,
            active_requirement: 0,
            context_target: attached,
            triggering_player: None,
            grantor: ability.granted_by.map(|grant| grant.grantor),
            ability: None,
            queue: VecDeque::from(crate::state::ability::bind_effect_clauses(effects)),
            finalize: FrameFinalize::None,
        };
        self.run_frame(frame);
    }

    fn resolve_triggered_ability(&mut self, triggered: TriggeredAbilityOnStack) {
        let source_card = triggered
            .source_lki
            .as_ref()
            .map(|lki| lki.card)
            .unwrap_or(triggered.source_card);
        let source_controller = triggered
            .source_lki
            .as_ref()
            .map(|lki| lki.controller)
            .unwrap_or(triggered.controller);
        // Inline (delayed or chosen-mode) effects have no printed ability.
        let (effects, ability) = if let Some(effects) = triggered.inline_effects {
            (effects, None)
        } else {
            let Some(ability) = self
                .state
                .cards
                .get(source_card.0)
                .and_then(|card| card.abilities.get(triggered.ability_index))
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
                    triggered.source_ref,
                    triggered.context,
                )
            }) {
                return;
            }
            (effects, Some(triggered.ability_index))
        };

        let requirements = crate::state::ability::effect_target_requirements(&effects);
        let Some(targets) = self.resolution_targets(
            &triggered.targets,
            &triggered.target_req_indices,
            &requirements,
            triggered.controller,
        ) else {
            return;
        };

        // Track per-turn resolutions for "the Nth time this ability has
        // resolved this turn" gating.
        self.journal_turn();
        // Inline effects count under a sentinel slot that never collides
        // with a printed ability.
        let key = (source_card.0, ability.unwrap_or(usize::MAX));
        let count = self
            .state
            .turn
            .ability_resolutions_this_turn
            .entry(key)
            .or_insert(0);
        *count += 1;
        let resolutions_this_turn = *count;

        let frame = EffectFrame {
            source: Some(source_card),
            source_ref: triggered.source_ref,
            controller: source_controller,
            resolutions_this_turn,
            kicked: false,
            x: triggered.x,
            excess_damage: 0,
            created: None,
            this_way: Vec::new(),
            targets,
            active_requirement: 0,
            context_target: triggered.context,
            triggering_player: triggered.player,
            grantor: None,
            ability,
            queue: VecDeque::from(crate::state::ability::bind_effect_clauses(effects)),
            finalize: FrameFinalize::None,
        };
        self.run_frame(frame);
    }

    /// CR 608.2b: check every chosen slot against its own clause once, before
    /// any instruction. Zero chosen optional targets still resolve; a nonempty
    /// all-illegal set suppresses the entire spell or ability, even untargeted
    /// instructions. The retained slots also enter clone/undo and state hashes.
    fn resolution_targets(
        &self,
        targets: &[Target],
        indices: &[usize],
        requirements: &[TargetRequirement],
        controller: PlayerId,
    ) -> Option<Vec<ResolutionTarget>> {
        let mut slots: Vec<_> = targets
            .iter()
            .enumerate()
            .map(|(index, target)| {
                let requirement = indices.get(index).copied().unwrap_or(0);
                ResolutionTarget {
                    target: *target,
                    requirement,
                    legal: requirements
                        .get(requirement)
                        .is_some_and(|req| self.target_is_legal(*target, &req.spec, controller)),
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
            Target::Permanent(id) if self.permanent_object_ref(id).is_some() => self
                .creature_types(id)
                .into_iter()
                .map(String::from)
                .collect(),
            Target::Permanent(id) => self
                .departed_permanent_lki(id)
                .map(|lki| lki.creature_types.clone())
                .unwrap_or_default(),
            _ => Vec::new(),
        }
    }

    /// Instructions cannot act on an object that has left its expected zone
    /// (CR 400.7). This is deliberately independent of TargetSpec: hexproof,
    /// controller and characteristic changes during resolution do not retarget.
    fn target_is_present(&self, target: Target) -> bool {
        match target {
            Target::Permanent(id) => self.permanent_object_ref(id).is_some(),
            Target::GraveyardCard(object) => {
                let card = CardId::from(object.entity);
                self.current_object_ref(card) == Some(object)
                    && self.state.zones.zone_of(card) == Some(ZoneType::Graveyard)
            }
            Target::StackSpell(card) => self.find_spell_on_stack_index(card).is_some(),
            Target::StackAbility(id) => self.find_ability_on_stack_index(id).is_some(),
            Target::Player(player) => self.state.players[player.0].alive,
        }
    }

    /// The object the resolving ability's Aura or Equipment is attached to:
    /// the one captured as an activated ability was activated, or else the
    /// source's current attachment (a triggered ability).
    fn frame_attached(&self, frame: &EffectFrame) -> Option<PermanentId> {
        match frame.context_target {
            Some(Target::Permanent(id)) => Some(id),
            Some(_) => None,
            None => frame
                .source_ref
                .and_then(|source| self.lookup_current_permanent(source).ok())
                .and_then(|source| self.attached_permanent(source)),
        }
    }

    /// The players a selector names as the effect resolves; a target player
    /// that is no longer a legal target is skipped.
    fn selected_players(&self, selector: &PlayerSelector, frame: &EffectFrame) -> Vec<PlayerId> {
        match selector {
            PlayerSelector::You => vec![frame.controller],
            PlayerSelector::EachOpponent => vec![PlayerId(1 - frame.controller.0)],
            PlayerSelector::TriggeringPlayer => frame.triggering_player.into_iter().collect(),
            PlayerSelector::Target(_) => match self.present_primary_target(frame) {
                Some(Target::Player(player)) => vec![player],
                _ => Vec::new(),
            },
            PlayerSelector::TargetController => match frame.primary_target() {
                Some(Target::Permanent(id)) => match self.state.permanents[id].as_ref() {
                    Some(permanent) => vec![permanent.controller],
                    None => self
                        .departed_permanent_lki(id)
                        .map(|lki| lki.controller)
                        .into_iter()
                        .collect(),
                },
                _ => Vec::new(),
            },
        }
    }

    pub(crate) fn present_primary_target(&self, frame: &EffectFrame) -> Option<Target> {
        frame
            .primary_target()
            .filter(|target| self.target_is_present(*target))
    }

    /// Whom an `ApplyContinuous` affects, fixed now (CR 611.2c).
    fn resolve_affected(&self, affected: &AffectedSpec, frame: &EffectFrame) -> Vec<Affected> {
        let permanent = |id: Option<PermanentId>| {
            id.and_then(|id| self.affected_permanent(id))
                .into_iter()
                .collect()
        };
        match affected {
            AffectedSpec::You => vec![Affected::Player(frame.controller)],
            AffectedSpec::Source => permanent(self.source_permanent_id(frame)),
            AffectedSpec::Target(_) => permanent(match self.present_primary_target(frame) {
                Some(Target::Permanent(id)) => Some(id),
                _ => None,
            }),
            AffectedSpec::TriggeringPermanent => permanent(match frame.context_target {
                Some(Target::Permanent(id)) => Some(id),
                _ => None,
            }),
            AffectedSpec::EachYouControl(predicate) => {
                self.each_controlled_matching(frame.controller, predicate)
            }
            AffectedSpec::EachOpponentControls(predicate) => {
                self.each_controlled_matching(PlayerId(1 - frame.controller.0), predicate)
            }
            AffectedSpec::EachMatching(predicate) => [PlayerId(0), PlayerId(1)]
                .into_iter()
                .flat_map(|player| self.each_controlled_matching(player, predicate))
                .collect(),
            AffectedSpec::EachTargetPlayerControls(predicate) => {
                match self.present_primary_target(frame) {
                    Some(Target::Player(player)) => {
                        self.each_controlled_matching(player, predicate)
                    }
                    _ => Vec::new(),
                }
            }
        }
    }

    fn each_controlled_matching(
        &self,
        controller: PlayerId,
        predicate: &CardPredicate,
    ) -> Vec<Affected> {
        self.battlefield_permanents(controller)
            .into_iter()
            .filter(|&id| self.permanent_matches_predicate(id, predicate))
            .filter_map(|id| self.affected_permanent(id))
            .collect()
    }

    /// Execute one effect. Returns a decision to suspend on, or None if the
    /// effect completed. Branching effects push their branch onto the front
    /// of the frame's queue.
    pub(crate) fn execute_frame_effect(
        &mut self,
        effect: &Effect,
        frame: &mut EffectFrame,
    ) -> Option<Decision> {
        match effect {
            Effect::ApplyContinuous {
                affected,
                modification,
                duration,
            } => {
                let affected = self.resolve_affected(affected, frame);
                let modification = modification.resolve(|value| self.effect_value(value, frame));
                // "For as long as you control [this]" with this gone or
                // already someone else's lasts no time at all.
                let source_object = frame.source_ref.filter(|source| {
                    self.lookup_current_permanent(*source)
                        .ok()
                        .and_then(|id| self.state.permanents[id].as_ref())
                        .is_some_and(|permanent| permanent.controller == frame.controller)
                });
                if *duration == EffectDuration::WhileYouControlSource && source_object.is_none() {
                    return None;
                }
                self.record_continuous(
                    frame.source,
                    source_object,
                    frame.controller,
                    affected,
                    modification,
                    *duration,
                );
                None
            }
            Effect::PutCountersOnTriggeringPermanent { count } => {
                if let Some(Target::Permanent(id)) = frame.context_target {
                    if self.permanent_object_ref(id).is_some() {
                        self.put_counters(
                            id,
                            CounterKind::PlusOnePlusOne,
                            *count,
                            frame.controller,
                        );
                    }
                }
                None
            }
            Effect::Sacrifice { predicate } => {
                let candidates = self.sacrifice_candidates(frame.controller, predicate);
                (!candidates.is_empty()).then_some(Decision::Sacrifice {
                    player: frame.controller,
                    predicate: predicate.clone(),
                    optional: None,
                })
            }
            Effect::MaySacrifice {
                predicate,
                another,
                then,
            } => {
                let excluded = another.then(|| self.source_permanent_id(frame)).flatten();
                let decision = Decision::Sacrifice {
                    player: frame.controller,
                    predicate: predicate.clone(),
                    optional: Some(OptionalSacrifice {
                        excluded,
                        then: then.clone(),
                    }),
                };
                (!self.sacrifice_decision_candidates(&decision).is_empty()).then_some(decision)
            }
            Effect::DistributeCountersToTappedOpponentCreatures { kind, count } => {
                self.distribute_counters(frame.controller, *kind, *count)
            }
            Effect::DealDamageToPlayer { player, amount } => {
                let amount = self.effect_value(amount, frame);
                for player in self.selected_players(player, frame) {
                    self.apply_player_damage(frame.source_ref, player, amount);
                }
                None
            }
            Effect::DrainLife { amount } => {
                let amount = self.effect_value(amount, frame).max(0);
                self.change_life(PlayerId(1 - frame.controller.0), -amount);
                self.gain_life(frame.controller, amount);
                None
            }
            Effect::ReturnGraveyardTarget { to, .. } => {
                let target = self.present_primary_target(frame)?;
                if let Target::GraveyardCard(object) = target {
                    let card = CardId::from(object.entity);
                    match to {
                        GraveyardReturn::Hand => self.reveal_card_to_hand(card, frame.controller),
                        GraveyardReturn::Battlefield | GraveyardReturn::BattlefieldTapped => {
                            self.move_card_with_entry(
                                card,
                                ZoneType::Battlefield,
                                crate::flow::proposed_event::ZoneEntry {
                                    tapped: *to == GraveyardReturn::BattlefieldTapped,
                                    controller: Some(frame.controller),
                                    ..crate::flow::proposed_event::ZoneEntry::default()
                                },
                            );
                        }
                    }
                }
                None
            }
            Effect::MoveSourceFromGraveyard { to, then } => {
                let card = frame.source?;
                let source = frame.source_ref?;
                // The card the dying object became (CR 400.7): its next
                // incarnation, still in the graveyard.
                let current = self.current_object_ref(card)?;
                if current.incarnation.0 == source.incarnation.0 + 1
                    && self.state.zones.zone_of(card) == Some(ZoneType::Graveyard)
                {
                    match to {
                        ZoneType::Hand => {
                            self.reveal_card_to_hand(card, self.state.cards[card].owner)
                        }
                        to => {
                            self.move_card(card, *to);
                        }
                    }
                    frame.prepend_effects(then.iter().cloned());
                }
                None
            }
            Effect::MillCards { count, target } => {
                let player = match target {
                    Some(_) => match self.present_primary_target(frame) {
                        Some(Target::Player(player)) => player,
                        _ => return None,
                    },
                    None => frame.controller,
                };
                let count = self.effect_value(count, frame).max(0) as usize;
                frame.this_way = self.library_top(player, count);
                for card in frame.this_way.clone() {
                    self.move_card(card, ZoneType::Graveyard);
                }
                None
            }
            Effect::ReturnMilledThisWay { predicate } => {
                let cards: Vec<_> = frame
                    .this_way
                    .iter()
                    .copied()
                    .filter(|card| {
                        self.state.zones.zone_of(*card) == Some(ZoneType::Graveyard)
                            && predicate.matches_card(&self.state.cards[*card])
                    })
                    .collect();
                (!cards.is_empty()).then_some(Decision::SelectMilled {
                    player: frame.controller,
                    cards,
                    if_declined: Vec::new(),
                    to_battlefield: true,
                })
            }
            Effect::MillAndSelect {
                count,
                predicate,
                if_declined,
                if_milled,
            } => {
                let cards = self.library_top(frame.controller, *count);
                for card in &cards {
                    self.move_card(*card, ZoneType::Graveyard);
                }
                if let Some(bonus) = if_milled.as_ref().filter(|bonus| {
                    cards
                        .iter()
                        .any(|card| bonus.predicate.matches_card(&self.state.cards[*card]))
                }) {
                    frame.prepend_effects(bonus.effects.iter().cloned());
                }
                let cards: Vec<_> = cards
                    .into_iter()
                    .filter(|card| predicate.matches_card(&self.state.cards[*card]))
                    .collect();
                if cards.is_empty() {
                    frame.prepend_effects(if_declined.iter().cloned());
                    None
                } else {
                    Some(Decision::SelectMilled {
                        player: frame.controller,
                        cards,
                        if_declined: if_declined.clone(),
                        to_battlefield: false,
                    })
                }
            }
            Effect::SelectTargetRequirement { index } => {
                frame.active_requirement = *index;
                None
            }
            Effect::TargetClause { .. } => unreachable!("clauses are bound before resolution"),
            Effect::DamageOpponentCreatures { amount } => {
                for id in self.battlefield_permanents(PlayerId(1 - frame.controller.0)) {
                    if self.permanent_is_battlefield_creature(id) {
                        self.apply_permanent_damage(frame.source_ref, id, *amount);
                    }
                }
                None
            }
            Effect::CreateTokensValue { token_name, count } => {
                let count = self.effect_value(count, frame).max(0) as usize;
                for _ in 0..count {
                    self.create_token(token_name, frame.controller, false);
                }
                None
            }
            Effect::ExileWithControllerToken { token_name, .. } => {
                let Some(Target::Permanent(id)) = self.present_primary_target(frame) else {
                    return None;
                };
                let permanent = self.state.permanents[id].as_ref()?;
                let controller = permanent.controller;
                let card = permanent.card;
                self.move_card(card, ZoneType::Exile);
                self.create_token(token_name, controller, false);
                None
            }
            Effect::SearchBasicLand {
                subtype,
                count,
                tapped,
                untap_if_lands,
                bottom_source,
                to_hand,
            } => {
                self.emit(GameEvent::LibrarySearched {
                    player: frame.controller,
                });
                Some(Decision::SearchBasicLand {
                    player: frame.controller,
                    subtype: subtype.clone(),
                    remaining: self.effect_value(count, frame).max(0) as usize,
                    tapped: *tapped,
                    untap_if_lands: *untap_if_lands,
                    bottom_source: *bottom_source,
                    to_hand: *to_hand,
                    selected: Vec::new(),
                })
            }
            Effect::OptionalDiscardDraw { count, up_to } => {
                (*up_to > 0).then_some(Decision::OptionalDiscardDraw {
                    player: frame.controller,
                    count: *count,
                    remaining: *up_to,
                    discarded: 0,
                })
            }
            Effect::DiscardTargetPlayer { count, .. } => {
                let Some(Target::Player(player)) = self.present_primary_target(frame) else {
                    return None;
                };
                (!self
                    .state
                    .zones
                    .zone_cards(ZoneType::Hand, player)
                    .is_empty()
                    && *count > 0)
                    .then_some(Decision::Discard {
                        player,
                        remaining: *count,
                        then: None,
                    })
            }
            Effect::Destroy { .. } | Effect::Exile { .. } => {
                let target = self.present_primary_target(frame);
                if let (Effect::Exile { .. }, Some(Target::GraveyardCard(object))) =
                    (effect, target)
                {
                    self.move_card(CardId::from(object.entity), ZoneType::Exile);
                    return None;
                }
                let Some(Target::Permanent(id)) = target else {
                    return None;
                };
                if matches!(effect, Effect::Destroy { .. }) {
                    if let Some(object) = self.permanent_object_ref(id) {
                        self.destroy_object(object);
                    }
                } else if let Some(permanent) = self.state.permanents[id].as_ref() {
                    self.move_card(permanent.card, ZoneType::Exile);
                }
                None
            }
            Effect::CopyTriggeringObject => {
                let copy = self.copy_stack_object(frame.context_target?, frame.controller)?;
                self.next_retarget(frame.controller, copy, 0, true)
            }
            Effect::ChangeTarget { .. } => {
                let object = self.present_primary_target(frame)?;
                self.next_retarget(frame.controller, object, 0, false)
            }
            Effect::ExileSourceAndReturn { player } => {
                let [player] = self.selected_players(player, frame)[..] else {
                    return None;
                };
                let id = self.source_permanent_id(frame)?;
                let card = self.state.permanents[id].as_ref()?.card;
                self.move_card(card, ZoneType::Exile);
                if self.state.zones.zone_of(card) == Some(ZoneType::Exile) {
                    self.move_card_with_entry(
                        card,
                        ZoneType::Battlefield,
                        crate::flow::proposed_event::ZoneEntry {
                            controller: Some(player),
                            ..Default::default()
                        },
                    );
                }
                None
            }
            Effect::GainControl {
                player, duration, ..
            } => {
                let Some(Target::Permanent(id)) = self.present_primary_target(frame) else {
                    return None;
                };
                let [player] = self.selected_players(player, frame)[..] else {
                    return None;
                };
                let previous = self.state.permanents[id].as_ref()?.controller;
                self.change_control(id, player);
                let affected = self.affected_permanent(id);
                self.apply_continuous(
                    frame.source,
                    player,
                    affected,
                    Modification::ControlChanged { previous },
                    *duration,
                );
                None
            }
            Effect::PutCountersOnCreatedToken { count } => {
                let count = self.effect_value(count, frame);
                if let Some(token) = frame.created {
                    if self.permanent_object_ref(token).is_some() {
                        self.put_counters(
                            token,
                            CounterKind::PlusOnePlusOne,
                            count,
                            frame.controller,
                        );
                    }
                }
                None
            }
            Effect::ExileAndReturn { .. } => {
                let Some(Target::Permanent(id)) = self.present_primary_target(frame) else {
                    return None;
                };
                let card = self.state.permanents[id].as_ref()?.card;
                self.move_card(card, ZoneType::Exile);
                if self.state.zones.zone_of(card) == Some(ZoneType::Exile) {
                    let owner = self.state.cards[card].owner;
                    self.move_card_with_entry(
                        card,
                        ZoneType::Battlefield,
                        crate::flow::proposed_event::ZoneEntry {
                            controller: Some(owner),
                            ..Default::default()
                        },
                    );
                }
                None
            }
            Effect::Airbend { .. } => {
                if let Some(Target::Permanent(id)) = self.present_primary_target(frame) {
                    self.airbend_permanent(id);
                }
                None
            }
            Effect::AirbendAll { predicate } => {
                let exempt = match self.present_primary_target(frame) {
                    Some(Target::Permanent(id)) => Some(id),
                    _ => None,
                };
                let chosen: Vec<_> = [PlayerId(0), PlayerId(1)]
                    .into_iter()
                    .flat_map(|player| self.battlefield_permanents(player))
                    .filter(|id| {
                        Some(*id) != exempt && self.permanent_matches_predicate(*id, predicate)
                    })
                    .collect();
                for id in chosen {
                    self.airbend_permanent(id);
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
                    source: frame.source_ref,
                };
                (!self.linked_exile_candidates(&decision).is_empty()).then_some(decision)
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
                    .zone_cards(ZoneType::Library, player)
                    .last()
                {
                    self.move_card(top, ZoneType::Exile);
                    if !self.state.cards[top].types.is_land() {
                        nonland = Some(top);
                        break;
                    }
                }
                let card = nonland?;
                let limit = self.effect_value(below, frame);
                let decision = Decision::CastDuringResolution {
                    player,
                    paid: false,
                    offer: ResolutionCastOffer::Exiled { card },
                    spends: None,
                };
                if i32::from(self.card_mana_value(card)) < limit
                    && !self.resolution_cast_candidates(&decision).is_empty()
                {
                    return Some(decision);
                }
                self.reveal_card_to_hand(card, player);
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
            Effect::SearchLandsOpponentSplits {
                count,
                opponent_picks,
            } => {
                self.emit(GameEvent::LibrarySearched {
                    player: frame.controller,
                });
                Some(Decision::SearchDistinctLands {
                    player: frame.controller,
                    remaining: *count,
                    selected: Vec::new(),
                    opponent_picks: *opponent_picks,
                })
            }
            Effect::OwnerPutsIntoLibrary { .. } => {
                let Some(Target::Permanent(id)) = self.present_primary_target(frame) else {
                    return None;
                };
                let card = self.state.permanents[id].as_ref()?.card;
                Some(Decision::Modal {
                    player: self.state.cards[card].owner,
                    modes: [true, false]
                        .map(|second_from_top| {
                            vec![Effect::PlaceTargetInLibrary { second_from_top }]
                        })
                        .to_vec(),
                })
            }
            Effect::PlaceTargetInLibrary { second_from_top } => {
                let Some(Target::Permanent(id)) = self.present_primary_target(frame) else {
                    return None;
                };
                let permanent = self.state.permanents[id].as_ref()?;
                let card = permanent.card;
                self.move_card(card, ZoneType::Library);
                if *second_from_top {
                    let owner = self.state.cards[card].owner;
                    self.journal_zones();
                    let library = self.state.zones.zone_cards_mut(ZoneType::Library, owner);
                    if let Some(index) = library.iter().position(|c| *c == card) {
                        library.remove(index);
                        let below_top = library.len().saturating_sub(1);
                        library.insert(below_top, card);
                    }
                } else {
                    self.put_on_bottom_of_library(card);
                }
                None
            }
            Effect::SacrificeSource { then } => {
                if let Some(permanent_id) = self.source_permanent_id(frame) {
                    self.sacrifice_permanent(permanent_id);
                    frame.prepend_effects(then.iter().cloned());
                }
                None
            }
            Effect::ExileTopCardPlayThisTurn => {
                self.exile_top_card_play_this_turn(frame.controller);
                None
            }
            Effect::ShuffleGraveyardIntoLibrary => {
                let player = frame.controller;
                let graveyard = self
                    .state
                    .zones
                    .zone_cards(ZoneType::Graveyard, player)
                    .to_vec();
                for card in graveyard {
                    self.move_card(card, ZoneType::Library);
                }
                self.journal_zones();
                self.state
                    .zones
                    .shuffle(ZoneType::Library, player, &mut self.state.rng);
                None
            }
            Effect::ExileSourceSpell => {
                if let Some(card) = frame
                    .source
                    .filter(|card| self.find_spell_on_stack_index(*card).is_some())
                {
                    self.move_card(card, ZoneType::Exile);
                }
                None
            }
            Effect::ReturnToHandDrawIfControlled { .. } => {
                let Some(Target::Permanent(id)) = self.present_primary_target(frame) else {
                    return None;
                };
                let controlled = self.state.permanents[id]
                    .as_ref()
                    .is_some_and(|p| p.controller == frame.controller);
                self.return_permanent_to_owner_hand(id);
                if controlled {
                    frame.queue.push_front(Effect::DrawCards {
                        count: EffectValue::Fixed(1),
                    });
                }
                None
            }
            Effect::ReturnToHand { .. } => {
                let chosen = self.present_primary_target(frame)?;
                match chosen {
                    Target::Permanent(permanent_id) => {
                        self.return_permanent_to_owner_hand(permanent_id);
                    }
                    Target::StackSpell(card) => {
                        // Bounce a spell: its stack object ceases and the
                        // card returns to its owner's hand.
                        if let Some(index) = self.find_spell_on_stack_index(card) {
                            self.journal_stack();
                            self.state.stack_objects.remove(index);
                            let owner = self.state.cards[card].owner;
                            self.move_card(card, ZoneType::Hand);
                            self.invalidate_mana_cache(owner);
                        }
                    }
                    Target::Player(_) | Target::StackAbility(_) | Target::GraveyardCard(_) => {}
                }
                None
            }
            Effect::DealDamage { amount, .. } => {
                let chosen = self.present_primary_target(frame)?;
                match chosen {
                    Target::Player(player) => {
                        self.apply_player_damage(frame.source_ref, player, *amount);
                    }
                    Target::Permanent(permanent_id) => {
                        frame.excess_damage = self.apply_creature_damage_measuring_excess(
                            frame.source_ref,
                            permanent_id,
                            *amount,
                        );
                    }
                    Target::StackSpell(_) | Target::StackAbility(_) | Target::GraveyardCard(_) => {}
                }
                None
            }
            Effect::GainLifeValue { amount } => {
                let amount = self.effect_value(amount, frame);
                self.gain_life(frame.controller, amount);
                None
            }
            Effect::AddManaValue {
                mana,
                amount,
                until_end_of_combat,
            } => {
                let count = self.effect_value(amount, frame).max(0);
                self.journal_player(frame.controller);
                self.invalidate_mana_cache(frame.controller);
                let player = &mut self.state.players[frame.controller.0];
                let pool = if *until_end_of_combat {
                    &mut player.combat_mana_pool
                } else {
                    &mut player.mana_pool
                };
                for _ in 0..count {
                    pool.add(mana);
                }
                None
            }
            Effect::LoseLife { player, amount } => {
                let amount = self.effect_value(amount, frame);
                for player in self.selected_players(player, frame) {
                    self.change_life(player, -amount);
                }
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
                let first_ref = self.permanent_object_ref(first)?;
                let last_ref = self.permanent_object_ref(last)?;
                // Each deals damage equal to its power at once (CR 701.14a).
                let first_power = self.effective_power(first);
                let last_power = self.effective_power(last);
                if first_power > 0 {
                    frame.excess_damage = self.apply_creature_damage_measuring_excess(
                        Some(first_ref),
                        last,
                        first_power,
                    );
                }
                if last_power > 0 {
                    self.apply_permanent_damage(Some(last_ref), first, last_power);
                }
                None
            }
            Effect::SearchLibraryToHand { predicate } => {
                self.emit(GameEvent::LibrarySearched {
                    player: frame.controller,
                });
                // With nothing to find, the search still shuffles.
                if self
                    .library_search_candidates(frame.controller, predicate)
                    .is_empty()
                {
                    self.shuffle_library(frame.controller);
                    return None;
                }
                Some(Decision::SearchLibraryToHand {
                    player: frame.controller,
                    predicate: predicate.clone(),
                })
            }
            Effect::CounterSpell { .. } => {
                // Spells only: countering an ability is a different effect.
                if let Target::StackSpell(spell) = self.present_primary_target(frame)? {
                    self.counter_spell(spell, frame.source);
                }
                None
            }
            Effect::CounterAbility { .. } => {
                if let Target::StackAbility(ability) = self.present_primary_target(frame)? {
                    self.counter_ability(ability, frame.source);
                }
                None
            }
            Effect::CreateNonlegendaryCopy => {
                let source = frame.source?;
                let original = &self.state.cards[source];
                let mut copy = original.clone();
                copy.id = self.state.id_gen.next_id();
                copy.owner = frame.controller;
                copy.is_token = true;
                copy.supertypes
                    .retain(|supertype| !supertype.eq_ignore_ascii_case("legendary"));
                let id = CardId(self.state.cards.len());
                self.state.cards.push(copy);
                self.state.card_to_permanent.push(None);
                self.state
                    .object_incarnations
                    .push(crate::state::game_object::Incarnation::INITIAL);
                self.move_card(id, ZoneType::Battlefield);
                None
            }
            Effect::OptionalPayment {
                cost,
                if_paid,
                if_declined,
            } => Some(Decision::Payment(PaymentProgress::new(
                frame.controller,
                cost.clone(),
                if_paid.clone(),
                if_declined.clone(),
            ))),
            Effect::Discard | Effect::MayDiscard { .. } => {
                frame.this_way.clear();
                (!self
                    .state
                    .zones
                    .zone_cards(ZoneType::Hand, frame.controller)
                    .is_empty())
                .then(|| Decision::Discard {
                    player: frame.controller,
                    remaining: 1,
                    then: match effect {
                        Effect::MayDiscard { then } => Some(then.clone()),
                        _ => None,
                    },
                })
            }
            Effect::CastLibraryTopFree => self.library_top_free_cast(frame.controller, None),
            Effect::WhenYouDo { effects } => {
                // A spell's reflexive trigger has its card as its source.
                self.enqueue_inline_trigger(
                    frame.source?,
                    frame.source_ref,
                    frame.controller,
                    effects.clone(),
                    None,
                );
                None
            }
            Effect::MayOnceEachTurn { effects } => {
                let key = (frame.source_ref?, frame.ability?);
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
                    PaymentCost::Mana(Default::default()),
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
            Effect::DealDamageValue { amount, target } => {
                let amount = self.effect_value(amount, frame);
                self.execute_frame_effect(
                    &Effect::DealDamage {
                        amount,
                        target: target.clone(),
                    },
                    frame,
                )
            }
            Effect::PutCountersValue { count, target } => {
                let count = self.effect_value(count, frame);
                self.execute_frame_effect(
                    &Effect::PutCounters {
                        count,
                        kind: CounterKind::PlusOnePlusOne,
                        target: target.clone(),
                    },
                    frame,
                )
            }
            Effect::DrawCards { count } => {
                // Drawing from an empty library sets `drew_when_empty`; the player
                // loses via state-based actions (CR 704.5c), same as the draw step.
                let count = self.effect_value(count, frame).max(0) as usize;
                self.draw_cards(frame.controller, count);
                None
            }
            Effect::DestroyCreaturesLosingAbilities { max_mana_value } => {
                let max = self.effect_value(max_mana_value, frame);
                let creatures: Vec<PermanentId> = [PlayerId(0), PlayerId(1)]
                    .into_iter()
                    .flat_map(|player| self.battlefield_permanents(player))
                    .filter(|id| {
                        self.permanent_is_creature(*id)
                            && self.state.permanents[*id].as_ref().is_some_and(|p| {
                                let card = &self.state.cards[p.card];
                                card.mana_cost
                                    .as_ref()
                                    .map_or(0, |c| i32::from(c.mana_value))
                                    <= max
                            })
                    })
                    .collect();
                let affected: Vec<_> = creatures
                    .iter()
                    .filter_map(|id| self.affected_permanent(*id))
                    .collect();
                self.apply_continuous(
                    frame.source,
                    frame.controller,
                    affected,
                    Modification::LoseAllAbilities,
                    EffectDuration::EndOfTurn,
                );
                // One simultaneous destruction: leave-the-battlefield abilities
                // look back at creatures that had already lost them (CR 603.10a).
                let objects: Vec<_> = creatures
                    .into_iter()
                    .filter_map(|id| self.permanent_object_ref(id))
                    .collect();
                let sources = self.snapshot_trigger_sources();
                for object in objects {
                    self.destroy_object(object);
                }
                self.process_game_events_with_sources(&sources);
                None
            }
            Effect::MassDamage { amount } => {
                for player in [PlayerId(0), PlayerId(1)] {
                    for permanent_id in self.battlefield_permanents(player) {
                        if self.permanent_is_creature(permanent_id) {
                            self.apply_permanent_damage(frame.source_ref, permanent_id, *amount);
                        }
                    }
                }
                // Lethal damage is cleaned up by state-based actions (CR 704.5g).
                None
            }
            Effect::CreateToken {
                token_name,
                count,
                tapped_and_attacking,
                attach_source,
            } => {
                for _ in 0..*count {
                    let token =
                        self.create_token(token_name, frame.controller, *tapped_and_attacking);
                    frame.created = token.or(frame.created);
                    if let Some(token) = token.filter(|_| *attach_source) {
                        self.attach_source_to(frame, token);
                    }
                }
                None
            }
            Effect::AttachSourceTo { .. } => {
                if let Target::Permanent(permanent_id) = self.present_primary_target(frame)? {
                    if let FrameFinalize::Spell { attach_to, .. } = &mut frame.finalize {
                        *attach_to = self.permanent_object_ref(permanent_id);
                    } else {
                        self.attach_source_to(frame, permanent_id);
                    }
                }
                None
            }
            Effect::ReturnGrantorToHand => {
                if let Some(grantor) = frame
                    .grantor
                    .and_then(|object| self.lookup_current_permanent(object).ok())
                {
                    self.return_permanent_to_owner_hand(grantor);
                }
                None
            }
            Effect::TapAttached => {
                let id = self.frame_attached(frame)?;
                self.tap_permanent(id, false);
                None
            }
            Effect::RemoveAllCountersFromAttached => {
                let id = self.frame_attached(frame)?;
                for kind in CounterKind::ALL {
                    let count = self.state.permanents[id].as_ref()?.counter_count(kind);
                    if count > 0 {
                        self.change_counters(id, kind, -count);
                    }
                }
                None
            }
            Effect::ExileAttached | Effect::ShuffleAttachedIntoLibrary => {
                let id = self.frame_attached(frame)?;
                let (card, owner) = self.state.permanents[id]
                    .as_ref()
                    .map(|permanent| (permanent.card, self.state.cards[permanent.card].owner))?;
                if matches!(effect, Effect::ExileAttached) {
                    self.move_card(card, ZoneType::Exile);
                } else {
                    self.move_card(card, ZoneType::Library);
                    self.journal_zones();
                    self.state
                        .zones
                        .shuffle(ZoneType::Library, owner, &mut self.state.rng);
                }
                None
            }
            Effect::PutCountersOnSource { count, kind } => {
                let count = self.effect_value(count, frame);
                if let Some(permanent_id) = self.source_permanent_id(frame) {
                    self.put_counters(permanent_id, *kind, count, frame.controller);
                }
                None
            }
            Effect::PutTriggeringCountersOnSource => {
                let Some(Target::Permanent(triggering)) = frame.context_target else {
                    return None;
                };
                let source = self.source_permanent_id(frame)?;
                let counters = match self.state.permanents[triggering].as_ref() {
                    Some(permanent) => permanent.counters,
                    None => self.departed_permanent_lki(triggering)?.counters,
                };
                for (kind, count) in counters.iter() {
                    self.put_counters(source, kind, count, frame.controller);
                }
                None
            }
            Effect::ForPlayer { player, effects } => {
                let [player] = self.selected_players(player, frame)[..] else {
                    return None;
                };
                frame
                    .queue
                    .push_front(Effect::RestoreController(frame.controller));
                frame.prepend_effects(effects.iter().cloned());
                frame.controller = player;
                None
            }
            Effect::RestoreController(player) => {
                frame.controller = *player;
                None
            }
            Effect::EachPlayerKeepsCreatures { max_total_power } => {
                self.keep_creatures(self.active_player(), *max_total_power, Vec::new())
            }
            Effect::PutSourceCountersOnTarget { .. } => {
                let Some(Target::Permanent(permanent_id)) = self.present_primary_target(frame)
                else {
                    return None;
                };
                let counters = self.object_lki(frame.source_ref?)?.counters;
                for (kind, count) in counters.iter() {
                    self.put_counters(permanent_id, kind, count, frame.controller);
                }
                None
            }
            Effect::PutCounters { count, kind, .. } => {
                let chosen = self.present_primary_target(frame)?;
                if let Target::Permanent(permanent_id) = chosen {
                    self.put_counters(permanent_id, *kind, *count, frame.controller);
                }
                None
            }
            Effect::CreateDelayedTrigger { event, effects } => {
                let (event, this_turn) = match event {
                    DelayedEvent::TargetDiesThisTurn => {
                        let Some(Target::Permanent(id)) = self.present_primary_target(frame) else {
                            return None;
                        };
                        let watched = self.permanent_object_ref(id)?;
                        (
                            DelayedTriggerEvent::Leaves {
                                watched,
                                or_exiled: false,
                            },
                            true,
                        )
                    }
                    DelayedEvent::CreatureDiesThisTurn => (DelayedTriggerEvent::CreatureDies, true),
                    DelayedEvent::NextEndStepForCreatedToken => (
                        DelayedTriggerEvent::NextEndStep {
                            context: frame.created.map(Target::Permanent),
                        },
                        false,
                    ),
                    DelayedEvent::YouNextCastThisTurn(predicate) => {
                        (DelayedTriggerEvent::YouCastSpell(predicate.clone()), true)
                    }
                };
                self.add_delayed_trigger(DelayedTrigger {
                    event,
                    controller: frame.controller,
                    source_card: frame.source?,
                    source_ref: frame.source_ref,
                    effects: effects.clone(),
                    this_turn,
                });
                None
            }
            Effect::SacrificeTriggeringPermanent => {
                if let Some(Target::Permanent(id)) = frame.context_target {
                    if self.state.permanents[id]
                        .as_ref()
                        .is_some_and(|permanent| permanent.controller == frame.controller)
                    {
                        self.sacrifice_permanent(id);
                    }
                }
                None
            }
            Effect::TapSource => {
                if let Some(permanent_id) = self.source_permanent_id(frame) {
                    self.tap_permanent(permanent_id, false);
                }
                None
            }
            Effect::UntapSource => {
                if let Some(permanent) = self.source_permanent_mut(frame) {
                    permanent.untap();
                    let controller = permanent.controller;
                    self.invalidate_mana_cache(controller);
                }
                None
            }
            Effect::ReturnSourceToHand => {
                let card = frame.source?;
                let owner = self.state.cards[card].owner;
                if frame.source_ref.is_some()
                    && self.current_object_ref(card) == frame.source_ref
                    && self.state.zones.contains(card, ZoneType::Graveyard, owner)
                {
                    self.reveal_card_to_hand(card, owner);
                }
                None
            }
            Effect::ReturnSourceToBattlefield { counter } => {
                let card = frame.source?;
                let owner = self.state.cards[card].owner;
                if frame.source_ref.is_some()
                    && self.current_object_ref(card) == frame.source_ref
                    && self.state.zones.contains(card, ZoneType::Graveyard, owner)
                {
                    let mut counters = crate::state::permanent::Counters::default();
                    counters.change(*counter, 1);
                    self.move_card_with_entry(
                        card,
                        ZoneType::Battlefield,
                        crate::flow::proposed_event::ZoneEntry {
                            counters,
                            ..Default::default()
                        },
                    );
                }
                None
            }
            Effect::GainLife { amount } => {
                self.gain_life(frame.controller, *amount);
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
                    self.emit(GameEvent::Scried {
                        player: frame.controller,
                    });
                }
                let remaining = self.library_top(frame.controller, *count);
                if remaining.is_empty() {
                    return None;
                }
                Some(Decision::Scry {
                    player: frame.controller,
                    looked: remaining.clone(),
                    remaining,
                    top: Vec::new(),
                    bottom: Vec::new(),
                })
            }
            Effect::LookAndSelect {
                reveal,
                look,
                min_select,
                max_select,
                predicate,
            } => {
                let look = self.effect_value(look, frame).max(0) as usize;
                let looked = self.library_top(frame.controller, look);
                if looked.is_empty() {
                    return None;
                }
                Some(Decision::LookAndSelect {
                    reveal: *reveal,
                    player: frame.controller,
                    looked,
                    predicate: predicate.clone(),
                    selected: 0,
                    min_select: *min_select,
                    max_select: *max_select,
                })
            }
            Effect::PutTopCardsInHand { count } => {
                for card in self.library_top(frame.controller, *count) {
                    self.move_card(card, ZoneType::Hand);
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
                let index = match frame.primary_target()? {
                    Target::StackSpell(spell) => self.find_spell_on_stack_index(spell)?,
                    Target::StackAbility(ability) => self.find_ability_on_stack_index(ability)?,
                    _ => return None,
                };
                let spell_controller = self.state.stack_objects[index].controller();
                let mut payment =
                    PaymentProgress::new(spell_controller, cost.clone(), Vec::new(), Vec::new());
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
                for attacker in attackers.iter().filter(|slot| slot.legal) {
                    let Target::Permanent(attacker_id) = attacker.target else {
                        continue;
                    };
                    // Eligibility was fixed at entry; damage still requires
                    // the original battlefield objects (CR 400.7).
                    if self.permanent_object_ref(victim).is_none() {
                        break;
                    }
                    let Some(attacker_ref) = self.permanent_object_ref(attacker_id) else {
                        continue;
                    };
                    let power = self.effective_power(attacker_id);
                    if power > 0 {
                        self.apply_permanent_damage(Some(attacker_ref), victim, power);
                    }
                }
                None
            }
            Effect::Earthbend { count, .. } => {
                let count = self.effect_value(count, frame);
                let chosen = self.present_primary_target(frame)?;
                let Target::Permanent(permanent_id) = chosen else {
                    return None;
                };
                let watched = self.permanent_object_ref(permanent_id)?;
                self.journal_permanent(permanent_id);
                let controller = {
                    let permanent = self.state.permanents[permanent_id].as_mut()?;
                    // The land becomes a 0/0 creature with haste that's
                    // still a land (base 0/0 by construction — lands print
                    // no P/T).
                    permanent.animated = true;
                    permanent.controller
                };
                if !self.put_counters(
                    permanent_id,
                    CounterKind::PlusOnePlusOne,
                    count,
                    frame.controller,
                ) {
                    return None;
                }
                // Delayed trigger: when it dies or is exiled, return it to
                // the battlefield tapped (CR 603.7 one-shot).
                self.add_delayed_trigger(DelayedTrigger {
                    event: DelayedTriggerEvent::Leaves {
                        watched,
                        or_exiled: true,
                    },
                    controller,
                    source_card: CardId::from(watched.entity),
                    source_ref: Some(watched),
                    effects: vec![Effect::ReturnSourceToBattlefieldTapped],
                    this_turn: false,
                });
                None
            }
            Effect::ReturnSourceToBattlefieldTapped => {
                let card = frame.source?;
                if matches!(
                    self.state.zones.zone_of(card),
                    Some(ZoneType::Graveyard) | Some(ZoneType::Exile)
                ) {
                    self.move_card_with_entry(
                        card,
                        ZoneType::Battlefield,
                        crate::flow::proposed_event::ZoneEntry {
                            tapped: true,
                            ..crate::flow::proposed_event::ZoneEntry::default()
                        },
                    );
                }
                None
            }
            Effect::ExileUntilSourceLeaves { .. } => {
                let chosen = self.present_primary_target(frame)?;
                let Target::Permanent(permanent_id) = chosen else {
                    return None;
                };
                // CR 603.6e pragmatics (Banisher Priest ruling): the exact
                // source object must still exist. A later incarnation of the
                // same physical card does not keep the old trigger alive.
                let source_ref = frame.source_ref?;
                self.lookup_current_permanent(source_ref).ok()?;
                let permanent = self.state.permanents[permanent_id].as_ref()?;
                let exiled_card = permanent.card;
                self.move_card(exiled_card, ZoneType::Exile);
                self.state.exile_links.push(ExileLink {
                    source: source_ref,
                    exiled_card,
                    returns: true,
                });
                None
            }
            Effect::ExileWithSource { .. } => {
                let card = match self.present_primary_target(frame)? {
                    Target::GraveyardCard(object) => CardId::from(object.entity),
                    Target::Permanent(id) => self.state.permanents[id].as_ref()?.card,
                    _ => return None,
                };
                self.move_card(card, ZoneType::Exile);
                let source = frame
                    .source_ref
                    .filter(|source| self.lookup_current_permanent(*source).is_ok());
                if let (Some(source), Some(ZoneType::Exile)) =
                    (source, self.state.zones.zone_of(card))
                {
                    self.journal_exile_links();
                    self.state.exile_links.push(ExileLink {
                        source,
                        exiled_card: card,
                        returns: false,
                    });
                }
                None
            }
            Effect::CastExiledWithSource { predicate } => {
                let decision = Decision::CastDuringResolution {
                    player: frame.controller,
                    paid: true,
                    offer: ResolutionCastOffer::ExiledWith {
                        source: frame.source_ref?,
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
            Effect::AddMana {
                mana,
                until_end_of_combat,
            } => {
                self.journal_player(frame.controller);
                self.invalidate_mana_cache(frame.controller);
                let player = &mut self.state.players[frame.controller.0];
                if *until_end_of_combat {
                    player.combat_mana_pool.add(mana);
                } else {
                    player.mana_pool.add(mana);
                }
                None
            }
            Effect::UntapTarget { .. } => {
                let chosen = self.present_primary_target(frame)?;
                if let Target::Permanent(permanent_id) = chosen {
                    self.journal_permanent(permanent_id);
                    if let Some(permanent) = self.state.permanents[permanent_id].as_mut() {
                        permanent.untap();
                        let controller = permanent.controller;
                        self.invalidate_mana_cache(controller);
                    }
                }
                None
            }
            Effect::RemoveCounters {
                up_to,
                chosen,
                target,
            } => {
                let Target::Permanent(permanent_id) = self.present_primary_target(frame)? else {
                    return None;
                };
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
                            up_to,
                            chosen: Some(kind),
                            target: target.clone(),
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
            Effect::TapTarget { .. } => {
                if let Some(Target::Permanent(permanent_id)) = self.present_primary_target(frame) {
                    self.tap_permanent(permanent_id, false);
                }
                None
            }
            Effect::ForEachTarget { effects } => {
                // Run the inner effects once per chosen target with that
                // target as the primary. Inner effects must not suspend.
                for target in frame.active_targets() {
                    let mut sub = EffectFrame {
                        source: frame.source,
                        source_ref: frame.source_ref,
                        controller: frame.controller,
                        resolutions_this_turn: frame.resolutions_this_turn,
                        kicked: frame.kicked,
                        x: frame.x,
                        excess_damage: 0,
                        created: None,
                        this_way: Vec::new(),
                        targets: vec![ResolutionTarget {
                            requirement: 0,
                            ..target
                        }],
                        active_requirement: 0,
                        context_target: None,
                        triggering_player: frame.triggering_player,
                        grantor: frame.grantor,
                        ability: frame.ability,
                        queue: VecDeque::from(effects.clone()),
                        finalize: FrameFinalize::None,
                    };
                    while let Some(effect) = sub.queue.pop_front() {
                        let decision = self.execute_frame_effect(&effect, &mut sub);
                        debug_assert!(
                            decision.is_none(),
                            "ForEachTarget sub-effects must not suspend"
                        );
                    }
                }
                None
            }
            Effect::PutCountersOnEachMatching {
                count,
                predicate,
                other,
            } => {
                let source_permanent = self.source_permanent_id(frame);
                let count = self.effect_value(count, frame);
                for permanent_id in self.battlefield_permanents(frame.controller) {
                    if *other && Some(permanent_id) == source_permanent {
                        continue;
                    }
                    if !self.permanent_matches_predicate(permanent_id, predicate) {
                        continue;
                    }
                    self.put_counters(
                        permanent_id,
                        CounterKind::PlusOnePlusOne,
                        count,
                        frame.controller,
                    );
                }
                None
            }
        }
    }

    pub(crate) fn permanent_is_battlefield_creature(&self, permanent_id: PermanentId) -> bool {
        let Some(permanent) = self
            .state
            .permanents
            .get(permanent_id.0)
            .and_then(|p| p.as_ref())
        else {
            return false;
        };
        self.state.zones.zone_of(permanent.card) == Some(ZoneType::Battlefield)
            && self.permanent_is_creature(permanent_id)
    }

    /// `player` looks at their library's top card and may cast it for free.
    fn library_top_free_cast(
        &self,
        player: PlayerId,
        spends: Option<OnceEachTurnKey>,
    ) -> Option<Decision> {
        let card = *self
            .state
            .zones
            .zone_cards(ZoneType::Library, player)
            .last()?;
        let decision = Decision::CastDuringResolution {
            player,
            paid: false,
            offer: ResolutionCastOffer::LibraryTop { card },
            spends,
        };
        (!self.resolution_cast_candidates(&decision).is_empty()).then_some(decision)
    }

    pub(crate) fn source_permanent_id(&self, frame: &EffectFrame) -> Option<PermanentId> {
        frame
            .source_ref
            .and_then(|source| self.lookup_current_permanent(source).ok())
    }

    /// Attach the source Equipment to a battlefield creature (CR 701.3). It
    /// does nothing once the Equipment has left the battlefield.
    fn attach_source_to(&mut self, frame: &EffectFrame, permanent_id: PermanentId) {
        let Some(object) = self.permanent_object_ref(permanent_id) else {
            return;
        };
        let timestamp = self.state.id_gen.next_id();
        if let Some(equipment) = self.source_permanent_mut(frame) {
            equipment.attached_to = Some(object);
            // CR 613.7e: an attachment receives a new timestamp.
            equipment.timestamp = timestamp;
        }
    }

    fn source_permanent_mut(&mut self, frame: &EffectFrame) -> Option<&mut Permanent> {
        let permanent_id = self.source_permanent_id(frame)?;
        self.journal_permanent(permanent_id);
        self.state.permanents[permanent_id].as_mut()
    }

    fn return_permanent_to_owner_hand(&mut self, permanent_id: PermanentId) {
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return;
        };
        let card = permanent.card;
        self.move_card(card, ZoneType::Hand);
    }

    /// Check whether a target is (still) legal for a given TargetSpec, from
    /// the perspective of `controller` ("you").
    pub(crate) fn target_is_legal(
        &self,
        target: Target,
        spec: &TargetSpec,
        controller: PlayerId,
    ) -> bool {
        match target {
            Target::Permanent(id) => {
                let Some(permanent) = self.state.permanents.get(id.0).and_then(|p| p.as_ref())
                else {
                    return false;
                };
                // CR 702.11b applies regardless of target wording. Recheck live
                // presence and hexproof at selection and resolution (608.2b).
                self.state.zones.zone_of(permanent.card) == Some(ZoneType::Battlefield)
                    && (permanent.controller == controller || !self.effective_keywords(id).hexproof)
                    && self.permanent_matches_spec(id, spec, controller)
            }
            Target::Player(player) => match spec {
                TargetSpec::Player | TargetSpec::CreatureOrPlayer => true,
                TargetSpec::Opponent => player != controller,
                _ => false,
            },
            Target::StackSpell(card) => {
                self.find_spell_on_stack_index(card).is_some()
                    && match spec {
                        TargetSpec::Spell => true,
                        TargetSpec::SpellOrPermanent { min_mana_value } => {
                            self.card_mana_value(card) >= *min_mana_value
                        }
                        TargetSpec::SpellOrAbilityWithSingleTarget => {
                            self.find_spell_on_stack_index(card).is_some_and(|index| {
                                self.state.stack_objects[index].targets().len() == 1
                            })
                        }
                        _ => false,
                    }
            }
            Target::StackAbility(id) => {
                self.find_ability_on_stack_index(id)
                    .is_some_and(|index| match spec {
                        TargetSpec::Ability => true,
                        TargetSpec::SpellOrAbilityWithSingleTarget => {
                            self.state.stack_objects[index].targets().len() == 1
                        }
                        _ => false,
                    })
            }
            Target::GraveyardCard(object) => match spec {
                TargetSpec::GraveyardCardYouOwn { predicate } => {
                    let card = CardId::from(object.entity);
                    self.current_object_ref(card) == Some(object)
                        && self.state.zones.zone_of(card) == Some(ZoneType::Graveyard)
                        && self.state.cards[card].owner == controller
                        && predicate.matches_card(&self.state.cards[card])
                }
                TargetSpec::GraveyardCardYouOwnWithinSourcePower { predicate } => self
                    .target_is_legal(
                        target,
                        &TargetSpec::GraveyardCardYouOwn {
                            predicate: predicate.clone(),
                        },
                        controller,
                    ),
                TargetSpec::GraveyardCard { predicate } => {
                    let card = CardId::from(object.entity);
                    self.current_object_ref(card) == Some(object)
                        && self.state.zones.zone_of(card) == Some(ZoneType::Graveyard)
                        && predicate.matches_card(&self.state.cards[card])
                }
                _ => false,
            },
        }
    }

    /// The bound a spec places with the ability's exact source (CR 608.2h:
    /// its last known information once it has left), beyond
    /// `target_is_legal`.
    pub(crate) fn within_source_bound(
        &self,
        target: Target,
        spec: &TargetSpec,
        source_card: CardId,
        source: Option<ObjectRef>,
    ) -> bool {
        let TargetSpec::GraveyardCardYouOwnWithinSourcePower { .. } = spec else {
            return true;
        };
        let Target::GraveyardCard(object) = target else {
            return false;
        };
        let power = source
            .and_then(|source| {
                self.object_lki(source).map(|lki| lki.power).or_else(|| {
                    self.lookup_current_permanent(source)
                        .ok()
                        .map(|id| self.effective_power(id))
                })
            })
            .unwrap_or_else(|| self.state.cards[source_card].power.unwrap_or(0));
        i32::from(self.card_mana_value(CardId::from(object.entity))) <= power
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
            TargetSpec::CreatureOrPlayer | TargetSpec::Creature => self.permanent_is_creature(id),
            TargetSpec::CreatureYouControl => {
                permanent.controller == controller && self.permanent_is_creature(id)
            }
            TargetSpec::CreatureOpponentControls => {
                permanent.controller != controller && self.permanent_is_creature(id)
            }
            TargetSpec::LandYouControl => {
                permanent.controller == controller && self.permanent_is_land(id)
            }
            TargetSpec::SpellOrPermanent { min_mana_value } => {
                self.card_mana_value(permanent.card) >= *min_mana_value
            }
            _ => false,
        }
    }

    /// The card's mana value; a spell on the stack counts its announced X
    /// (CR 202.3e).
    pub(crate) fn card_mana_value(&self, card: CardId) -> u8 {
        let Some(cost) = self.state.cards[card].mana_cost.as_ref() else {
            return 0;
        };
        let x = match self
            .find_spell_on_stack_index(card)
            .map(|index| &self.state.stack_objects[index])
        {
            Some(StackObject::Spell(spell)) => spell.x,
            _ => 0,
        };
        cost.mana_value.saturating_add(cost.x.saturating_mul(x))
    }
}

#[cfg(test)]
#[path = "resolution_tests.rs"]
mod resolution_tests;
