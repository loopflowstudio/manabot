// resolution.rs
// Resolving stack objects. Effects execute through an EffectFrame so a
// resolution can suspend on a mid-resolution decision (see decision.rs).

use std::collections::VecDeque;

use crate::{
    flow::{
        decision::{Decision, EffectFrame, FrameFinalize, PaymentProgress, ResolutionTarget},
        event::GameEvent,
        game::Game,
        trigger::{DelayedTrigger, DelayedTriggerKind, ExileLink},
    },
    state::{
        ability::{Ability, Effect, EffectValue, PaymentCost, TargetRequirement, TargetSpec},
        game_object::{CardId, PermanentId, PlayerId, Target},
        permanent::Permanent,
        stack_object::{
            ActivatedAbilityOnStack, SpellOnStack, StackObject, TriggeredAbilityOnStack,
        },
        zone::ZoneType,
    },
};

impl Game {
    fn effect_value(&self, value: &EffectValue, frame: &EffectFrame) -> i32 {
        let controller = frame.controller;
        match value {
            EffectValue::Fixed(value) => *value,
            EffectValue::X => i32::from(frame.x),
            EffectValue::CreaturesWithCounters => self
                .battlefield_permanents(controller)
                .into_iter()
                .filter(|id| {
                    self.permanent_is_battlefield_creature(*id)
                        && self.state.permanents[*id]
                            .as_ref()
                            .is_some_and(|p| p.plus1_counters > 0)
                })
                .count() as i32,
            EffectValue::BattlefieldMatching { predicate } => self
                .battlefield_permanents(controller)
                .into_iter()
                .filter(|id| self.permanent_matches_predicate(*id, predicate))
                .count() as i32,
            EffectValue::GraveyardMatching { base, predicate } => {
                base.saturating_add(self.count_graveyard_matching(controller, predicate) as i32)
            }
            EffectValue::CardsDrawnMinus { subtract } => {
                self.state.turn.cards_drawn_this_turn[controller.0].saturating_sub(*subtract) as i32
            }
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
            targets,
            active_requirement: 0,
            context_target: None,
            queue: VecDeque::from(effects),
            finalize: FrameFinalize::Spell { card },
        };
        self.run_frame(frame);
    }

    pub(crate) fn counter_spell(&mut self, card: CardId, by: Option<CardId>) {
        let Some(_) = self.find_spell_on_stack_index(card) else {
            return;
        };
        self.move_card(card, ZoneType::Graveyard);
        self.emit(GameEvent::SpellCountered { card, by });
    }

    fn resolve_activated_ability(&mut self, ability: ActivatedAbilityOnStack) {
        let Some(definition) = self.state.cards[ability.source_card]
            .activated_abilities
            .get(ability.ability_index)
            .cloned()
        else {
            return;
        };

        let effects = definition.effects;
        let requirements = crate::state::ability::effect_target_requirements(&effects);
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
            x: 0,
            targets,
            active_requirement: 0,
            context_target: None,
            queue: VecDeque::from(crate::state::ability::bind_effect_clauses(effects)),
            finalize: FrameFinalize::None,
        };
        self.run_frame(frame);
    }

    fn resolve_triggered_ability(&mut self, triggered: TriggeredAbilityOnStack) {
        let source_card = triggered
            .source_lki
            .map(|lki| lki.card)
            .unwrap_or(triggered.source_card);
        let source_controller = triggered
            .source_lki
            .map(|lki| lki.controller)
            .unwrap_or(triggered.controller);
        let (effects, ability_key) = if let Some(effects) = triggered.inline_effects {
            // Inline (delayed) triggers use a sentinel so their per-turn
            // resolution count never collides with a printed ability slot.
            (effects, usize::MAX)
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
            let Ability::Triggered { effects, .. } = ability;
            (effects, triggered.ability_index)
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
        let key = (source_card.0, ability_key);
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
            x: 0,
            targets,
            active_requirement: 0,
            context_target: triggered.context,
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
        let slots: Vec<_> = targets
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
        (slots.is_empty() || slots.iter().any(|slot| slot.legal)).then_some(slots)
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
            Target::Player(player) => self.state.players[player.0].alive,
        }
    }

    fn present_primary_target(&self, frame: &EffectFrame) -> Option<Target> {
        frame
            .primary_target()
            .filter(|target| self.target_is_present(*target))
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
            Effect::Sacrifice { predicate } => {
                let candidates = self.sacrifice_candidates(frame.controller, predicate);
                (!candidates.is_empty()).then_some(Decision::Sacrifice {
                    player: frame.controller,
                    predicate: predicate.clone(),
                })
            }
            Effect::DealDamageToController { amount } => {
                self.apply_player_damage(frame.source_ref, frame.controller, *amount);
                None
            }
            Effect::DrainLife { amount } => {
                self.change_life(PlayerId(1 - frame.controller.0), -*amount);
                self.gain_life(frame.controller, *amount);
                None
            }
            Effect::CantBeBlockedTarget { .. } => {
                let target = self.present_primary_target(frame)?;
                if let Target::Permanent(id) = target {
                    self.journal_permanent(id);
                    self.state.permanents[id]
                        .as_mut()?
                        .cant_be_blocked_this_turn = true;
                }
                None
            }
            Effect::ReturnGraveyardTarget {
                tapped_onto_battlefield,
                ..
            } => {
                let target = self.present_primary_target(frame)?;
                if let Target::GraveyardCard(object) = target {
                    let card = CardId::from(object.entity);
                    if *tapped_onto_battlefield {
                        self.move_card_with_entry(
                            card,
                            ZoneType::Battlefield,
                            crate::flow::proposed_event::BattlefieldEntry {
                                tapped: true,
                                ..crate::flow::proposed_event::BattlefieldEntry::default()
                            },
                        );
                        self.invalidate_mana_cache(frame.controller);
                    } else {
                        self.reveal_card_to_hand(card, frame.controller);
                    }
                }
                None
            }
            Effect::MillCards { count } => {
                for card in self.library_top(frame.controller, *count) {
                    self.move_card(card, ZoneType::Graveyard);
                }
                None
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
                    })
                }
            }
            Effect::DrainLifeValue { amount } => {
                let amount = self.effect_value(amount, frame).max(0);
                self.change_life(PlayerId(1 - frame.controller.0), -amount);
                self.gain_life(frame.controller, amount);
                None
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
            } => Some(Decision::SearchBasicLand {
                player: frame.controller,
                subtype: subtype.clone(),
                remaining: *count,
                tapped: *tapped,
                untap_if_lands: *untap_if_lands,
                bottom_source: *bottom_source,
                to_hand: *to_hand,
                selected: Vec::new(),
            }),
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
                    })
            }
            Effect::Destroy { .. } | Effect::Exile { .. } => {
                let Some(Target::Permanent(id)) = self.present_primary_target(frame) else {
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
            } => Some(Decision::SearchDistinctLands {
                player: frame.controller,
                remaining: *count,
                selected: Vec::new(),
                opponent_picks: *opponent_picks,
            }),
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
                let (card, controller) = (permanent.card, permanent.controller);
                self.move_card(card, ZoneType::Library);
                self.invalidate_mana_cache(controller);
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
            Effect::NoMaximumHandSize => {
                self.journal_player(frame.controller);
                self.state.players[frame.controller.0].no_maximum_hand_size = true;
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
            Effect::BuffTargetValue {
                power,
                toughness,
                target,
            } => self.execute_frame_effect(
                &Effect::BuffTarget {
                    power: self.effect_value(power, frame),
                    toughness: *toughness,
                    target: target.clone(),
                },
                frame,
            ),
            Effect::ReturnToHandDrawIfControlled { .. } => {
                let Some(Target::Permanent(id)) = self.present_primary_target(frame) else {
                    return None;
                };
                let controlled = self.state.permanents[id]
                    .as_ref()
                    .is_some_and(|p| p.controller == frame.controller);
                self.return_permanent_to_owner_hand(id);
                if controlled {
                    frame.queue.push_front(Effect::DrawCards { count: 1 });
                }
                None
            }
            Effect::BuffTriggeringPermanentValue { power, toughness } => {
                let Some(Target::Permanent(id)) = frame.context_target else {
                    return None;
                };
                self.permanent_object_ref(id)?;
                let power = self.effect_value(power, frame);
                let toughness = self.effect_value(toughness, frame);
                self.journal_permanent(id);
                let permanent = self.state.permanents[id].as_mut()?;
                permanent.temp_power += power;
                permanent.temp_toughness += toughness;
                None
            }
            Effect::ModifyTriggeringPermanent { counters, keywords } => {
                let Some(Target::Permanent(id)) = frame.context_target else {
                    return None;
                };
                self.permanent_object_ref(id)?;
                self.change_plus_one_counters(id, *counters);
                self.journal_permanent(id);
                if let Some(permanent) = self.state.permanents[id].as_mut() {
                    permanent.temp_keywords = permanent.temp_keywords.union(keywords);
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
                    Target::Player(_) | Target::GraveyardCard(_) => {}
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
                        self.apply_permanent_damage(frame.source_ref, permanent_id, *amount);
                    }
                    Target::StackSpell(_) | Target::GraveyardCard(_) => {}
                }
                None
            }
            Effect::CounterSpell { .. } => {
                let Some(Target::StackSpell(target_spell)) = self.present_primary_target(frame)
                else {
                    return None;
                };
                self.counter_spell(target_spell, frame.source);
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
            Effect::ModifyControlled {
                predicate,
                power,
                toughness,
            } => {
                for id in self.battlefield_permanents(frame.controller) {
                    if self.permanent_matches_predicate(id, predicate) {
                        self.journal_permanent(id);
                        let permanent = self.state.permanents[id].as_mut()?;
                        permanent.temp_power += power;
                        permanent.temp_toughness += toughness;
                    }
                }
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
            Effect::ModifyUntilEot {
                power_delta,
                toughness_delta,
            } => {
                if let Some(permanent) = self.source_permanent_mut(frame) {
                    permanent.temp_power += power_delta;
                    permanent.temp_toughness += toughness_delta;
                }
                None
            }
            Effect::Discard => (!self
                .state
                .zones
                .zone_cards(ZoneType::Hand, frame.controller)
                .is_empty())
            .then_some(Decision::Discard {
                player: frame.controller,
                remaining: 1,
            }),
            Effect::IfCondition { condition, effects } => {
                if self.check_static_condition(condition, frame.controller) {
                    frame.prepend_effects(effects.iter().cloned());
                }
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
                        target: target.clone(),
                    },
                    frame,
                )
            }
            Effect::AddSubtypeToTarget { subtype, .. } => {
                let chosen = self.present_primary_target(frame)?;
                if let Target::Permanent(id) = chosen {
                    self.journal_permanent(id);
                    let permanent = self.state.permanents[id].as_mut()?;
                    if !permanent.temp_subtypes.contains(subtype) {
                        permanent.temp_subtypes.push(subtype.clone());
                    }
                }
                None
            }
            Effect::ExileIfDiesThisTurn { .. } => {
                let chosen = self.present_primary_target(frame)?;
                if let Target::Permanent(id) = chosen {
                    self.journal_permanent(id);
                    self.state.permanents[id].as_mut()?.exile_if_dies_this_turn = true;
                }
                None
            }
            Effect::DrawCards { count } => {
                // Drawing from an empty library sets `drew_when_empty`; the player
                // loses via state-based actions (CR 704.5c), same as the draw step.
                self.draw_cards(frame.controller, *count);
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
            } => {
                for _ in 0..*count {
                    self.create_token(token_name, frame.controller, *tapped_and_attacking);
                }
                None
            }
            Effect::PutCountersOnSource { count } => {
                if let Some(permanent_id) = self.source_permanent_id(frame) {
                    self.change_plus_one_counters(permanent_id, *count);
                }
                None
            }
            Effect::PutCounters { count, .. } => {
                let chosen = self.present_primary_target(frame)?;
                if let Target::Permanent(permanent_id) = chosen {
                    self.change_plus_one_counters(permanent_id, *count);
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
            Effect::GrantKeywordsToSource { keywords } => {
                if let Some(permanent) = self.source_permanent_mut(frame) {
                    permanent.temp_keywords = permanent.temp_keywords.union(keywords);
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
            Effect::CantBeBlockedThisTurnSource => {
                if let Some(permanent) = self.source_permanent_mut(frame) {
                    permanent.cant_be_blocked_this_turn = true;
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
                let Some(Target::StackSpell(target_spell)) = frame.primary_target() else {
                    return None;
                };
                // The spell may already have left the stack (e.g. it was
                // countered in response) — the effect does nothing.
                let index = self.find_spell_on_stack_index(target_spell)?;
                let spell_controller = self.state.stack_objects[index].controller();
                Some(Decision::Payment(PaymentProgress::new(
                    spell_controller,
                    PaymentCost::Mana(cost.clone()),
                    Vec::new(),
                    vec![Effect::CounterSpell {
                        target: TargetSpec::Spell,
                    }],
                )))
            }
            Effect::IfKicked { then, otherwise } => {
                let branch = if frame.kicked { then } else { otherwise };
                frame.prepend_effects(branch.iter().cloned());
                None
            }
            Effect::IfGraveyardAtLeast {
                count,
                predicate,
                then,
                otherwise,
            } => {
                let matches = self.count_graveyard_matching(frame.controller, predicate);
                let branch = if matches >= *count { then } else { otherwise };
                frame.prepend_effects(branch.iter().cloned());
                None
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
                if !self.change_plus_one_counters_on_object(watched, count) {
                    return None;
                }
                // Delayed trigger: when it dies or is exiled, return it to
                // the battlefield tapped (CR 603.7 one-shot).
                self.state.delayed_triggers.push(DelayedTrigger {
                    watched,
                    controller,
                    kind: DelayedTriggerKind::ReturnToBattlefieldTapped,
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
                        crate::flow::proposed_event::BattlefieldEntry {
                            tapped: true,
                            ..crate::flow::proposed_event::BattlefieldEntry::default()
                        },
                    );
                    let owner = self.state.cards[card].owner;
                    self.invalidate_mana_cache(owner);
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
                let exiled_controller = permanent.controller;
                self.move_card(exiled_card, ZoneType::Exile);
                self.invalidate_mana_cache(exiled_controller);
                self.state.exile_links.push(ExileLink {
                    source: source_ref,
                    exiled_card,
                });
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
            Effect::BuffTarget {
                power, toughness, ..
            } => {
                let chosen = self.present_primary_target(frame)?;
                if let Target::Permanent(permanent_id) = chosen {
                    self.journal_permanent(permanent_id);
                    if let Some(permanent) = self.state.permanents[permanent_id].as_mut() {
                        permanent.temp_power += power;
                        permanent.temp_toughness += toughness;
                    }
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
            Effect::TapTarget { .. } => {
                if let Some(Target::Permanent(permanent_id)) = self.present_primary_target(frame) {
                    self.tap_permanent(permanent_id, false);
                }
                None
            }
            Effect::GrantKeywordsToTarget { keywords, .. } => {
                let chosen = self.present_primary_target(frame)?;
                if let Target::Permanent(permanent_id) = chosen {
                    self.journal_permanent(permanent_id);
                    if let Some(permanent) = self.state.permanents[permanent_id].as_mut() {
                        permanent.temp_keywords = permanent.temp_keywords.union(keywords);
                    }
                }
                None
            }
            Effect::IfTargetMatches { predicate, then } => {
                let Some(Target::Permanent(permanent_id)) = self.present_primary_target(frame)
                else {
                    return None;
                };
                if self.permanent_matches_predicate(permanent_id, predicate) {
                    frame.prepend_effects(then.iter().cloned());
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
                        targets: vec![ResolutionTarget {
                            requirement: 0,
                            ..target
                        }],
                        active_requirement: 0,
                        context_target: None,
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
                for permanent_id in self.battlefield_permanents(frame.controller) {
                    if *other && Some(permanent_id) == source_permanent {
                        continue;
                    }
                    if !self.permanent_matches_predicate(permanent_id, predicate) {
                        continue;
                    }
                    self.change_plus_one_counters(permanent_id, *count);
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

    pub(crate) fn source_permanent_id(&self, frame: &EffectFrame) -> Option<PermanentId> {
        frame
            .source_ref
            .and_then(|source| self.lookup_current_permanent(source).ok())
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
        let controller = permanent.controller;
        self.move_card(card, ZoneType::Hand);
        self.invalidate_mana_cache(controller);
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
                if self.state.zones.zone_of(permanent.card) != Some(ZoneType::Battlefield)
                    || (permanent.controller != controller && self.effective_keywords(id).hexproof)
                {
                    return false;
                }
                match spec {
                    TargetSpec::Permanent { predicate } => {
                        self.permanent_matches_predicate(id, predicate)
                    }
                    TargetSpec::PermanentYouControl { predicate } => {
                        permanent.controller == controller
                            && self.permanent_matches_predicate(id, predicate)
                    }
                    TargetSpec::PermanentOpponentControls { predicate } => {
                        permanent.controller != controller
                            && self.permanent_matches_predicate(id, predicate)
                    }
                    TargetSpec::CreatureOrPlayer | TargetSpec::Creature => {
                        self.permanent_is_creature(id)
                    }
                    TargetSpec::CreatureYouControl => {
                        permanent.controller == controller && self.permanent_is_creature(id)
                    }
                    TargetSpec::CreatureOpponentControls => {
                        permanent.controller != controller && self.permanent_is_creature(id)
                    }
                    TargetSpec::LandYouControl => {
                        permanent.controller == controller
                            && self.state.cards[permanent.card].types.is_land()
                    }
                    TargetSpec::SpellOrPermanent { min_mana_value } => {
                        self.card_mana_value(permanent.card) >= *min_mana_value
                    }
                    _ => false,
                }
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
                        _ => false,
                    }
            }
            Target::GraveyardCard(object) => match spec {
                TargetSpec::GraveyardCardYouOwn { predicate } => {
                    let card = CardId::from(object.entity);
                    self.current_object_ref(card) == Some(object)
                        && self.state.zones.zone_of(card) == Some(ZoneType::Graveyard)
                        && self.state.cards[card].owner == controller
                        && predicate.matches_card(&self.state.cards[card])
                }
                _ => false,
            },
        }
    }

    /// The card's mana value; a spell on the stack counts its announced X
    /// (CR 202.3e).
    pub(crate) fn card_mana_value(&self, card: CardId) -> u8 {
        let printed = self.state.cards[card]
            .mana_cost
            .as_ref()
            .map_or(0, |cost| cost.mana_value);
        let x = self
            .find_spell_on_stack_index(card)
            .and_then(|index| match &self.state.stack_objects[index] {
                StackObject::Spell(spell) => {
                    let symbols = self.state.cards[card].mana_cost.as_ref().map_or(0, |c| c.x);
                    Some(spell.x.saturating_mul(symbols))
                }
                _ => None,
            })
            .unwrap_or(0);
        printed.saturating_add(x)
    }
}

#[cfg(test)]
#[path = "resolution_tests.rs"]
mod resolution_tests;
