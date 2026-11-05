// resolution.rs
// Resolving stack objects. Effects execute through an EffectFrame so a
// resolution can suspend on a mid-resolution decision (see decision.rs).

use crate::{
    cards::content_pack::default_content_pack,
    flow::{
        decision::{
            Decision, EffectFrame, FrameCompletion, PaymentProgress, ResolutionCastOffer,
            ResolutionTarget, TargetIteration,
        },
        event::GameEvent,
        game::Game,
        proposed_event::{
            EventBinding, ProposedDamageTarget, ProposedEvent, TokenDefinition, ZoneEntry,
        },
        trigger::{DelayedTrigger, DelayedTriggerEvent},
        turn::OnceEachTurnKey,
    },
    state::{
        ability::{
            Ability, DelayedEvent, DelayedExpiry, DelayedFrequency, Effect, EffectValue, Except,
            LinkedChooser, LinkedExile, ObjectSelector, PaymentCost, PlayerSelector, Scope,
            TargetRequirement, TargetSpec,
        },
        card::{CardDefId, CardDefinition, Keyword},
        effect::{Affected, EffectDuration, Modification},
        game_object::{CardId, ObjectId, ObjectRef, PermanentId, PlayerId, Target},
        permanent::{CounterKind, Counters, Permanent},
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
            EffectValue::Objects(of) => self.select(of, frame).len() as i32,
            EffectValue::X => i32::from(frame.x),
            EffectValue::TriggeringPower => match frame.context_target {
                Some(Target::Permanent(id)) if self.state.permanents[id].is_some() => {
                    self.characteristics(id).power
                }
                Some(Target::Permanent(id)) => {
                    self.departed_permanent_lki(id).map_or(0, |lki| lki.power)
                }
                _ => 0,
            },
            EffectValue::ExcessDamage => frame.excess_damage,
            EffectValue::SourcePower => self
                .source_permanent_id(frame)
                .map(|id| self.characteristics(id).power)
                .or_else(|| {
                    frame
                        .source_ref
                        .and_then(|source| self.object_lki(source))
                        .map(|lki| lki.power)
                })
                .unwrap_or(0),
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
            EffectValue::Objects(_)
            | EffectValue::X
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
            EffectValue::GreatestPowerYouControl => self
                .battlefield_permanents(controller)
                .into_iter()
                .filter(|id| self.permanent_is_battlefield_creature(*id))
                .map(|id| self.effective_pt(id).0)
                .max()
                .unwrap_or(0)
                .max(0),
            EffectValue::CardsInHand => self.state.zones.size(ZoneType::Hand, controller) as i32,
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
            kicked: spell.kicked,
            x: spell.x,
            targets,
            completion: FrameCompletion::Spell {
                card,
                attach_to: None,
            },
            ..EffectFrame::new(spell.controller, effects)
        };
        self.run_frame(frame);
    }

    /// Counter an activated or triggered ability: it is removed from the
    /// stack and does nothing (CR 701.6b).
    pub(crate) fn counter_ability(&mut self, id: ObjectId, by: Option<CardId>) {
        let Some(index) = self.find_ability_on_stack_index(id) else {
            return;
        };
        let ability = self.state.stack_objects_mut().remove(index);
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

    /// The face an ability on the stack came from: the card as it stands
    /// while that face is up, else that face's row (the card has turned over
    /// or left since; the ability exists independently, CR 113.7a).
    pub(crate) fn source_face(&self, card: CardId, face: CardDefId) -> Option<&CardDefinition> {
        let current = self.state.cards.get(card.0)?;
        if current.definition_id == face {
            Some(current)
        } else {
            default_content_pack().definition(face)
        }
    }

    fn resolve_activated_ability(&mut self, ability: ActivatedAbilityOnStack) {
        let effects = ability.effects;
        let program = if ability.program {
            effects.clone()
        } else {
            Vec::new()
        };
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
        let frame = EffectFrame {
            program,
            source: Some(ability.source_card),
            source_ref: Some(ability.source_ref),
            inherited_from: ability.inherited_from,
            source_transforms: ability.source_transforms,
            x: ability.x,
            targets,
            context_target: attached,
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
                .source_face(source_card, triggered.source_definition_id)
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
            program,
            remembered: triggered.remembered,
            source: Some(source_card),
            source_ref: triggered.source_ref,
            inherited_from: triggered
                .source_lki
                .as_ref()
                .and_then(|lki| lki.inherited_from),
            source_transforms: triggered.source_transforms,
            resolutions_this_turn,
            x: triggered.x,
            targets,
            context_target: triggered.context,
            triggering_player: triggered.player,
            ability,
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
            Target::Permanent(id) if self.permanent_object_ref(id).is_some() => {
                self.characteristics(id).creature_types.clone()
            }
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
            PlayerSelector::Owner => Vec::new(),
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
            Effect::PutCounters { on, kind, amount } => {
                let amount = self.effect_value(amount, frame);
                let events = self
                    .selected_permanents(on, frame)
                    .into_iter()
                    .filter_map(|id| self.permanent_object_ref(id))
                    .map(|target| ProposedEvent::CounterChange {
                        provenance: crate::flow::proposed_event::CounterProvenance::new(
                            crate::flow::proposed_event::EventCause::Effect,
                        ),
                        target,
                        kind: *kind,
                        delta: amount,
                        by: Some(frame.controller),
                    })
                    .collect();
                frame.batch = Some(self.event_batch(events));
                None
            }
            Effect::PutCountersFrom { from, on } => {
                let counters = self.selected_counters(from, frame);
                let mut events = Vec::new();
                for target in self
                    .selected_permanents(on, frame)
                    .into_iter()
                    .filter_map(|id| self.permanent_object_ref(id))
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
                frame.batch = Some(self.event_batch(events));
                None
            }
            Effect::Sacrifice { what, optional } => {
                debug_assert!(
                    !optional || what.is_chosen(),
                    "\"you may sacrifice\" chooses among permanents"
                );
                // A player sacrifices only what they control (CR 701.17a).
                let controlled: Vec<_> = self
                    .selected_permanents(what, frame)
                    .into_iter()
                    .filter(|id| {
                        self.state.permanents[*id]
                            .as_ref()
                            .is_some_and(|permanent| permanent.controller == frame.controller)
                    })
                    .collect();
                // Select before replacing the result: `Moved` may name a
                // permanent the preceding instruction put on the battlefield.
                // What this sacrifice moves answers a following "if you do".
                let events = controlled
                    .into_iter()
                    .filter_map(|id| self.permanent_object_ref(id))
                    .map(|target| ProposedEvent::Sacrifice { target })
                    .collect();
                let mut batch = self.event_batch(events);
                batch.binding = EventBinding::Moved {
                    linked: None,
                    shuffle: false,
                    library_end: None,
                    reveal: false,
                };
                frame.batch = Some(batch);
                None
            }
            Effect::Move {
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
                    .source_ref
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
                    let card = CardId::from(object.entity);
                    let from = self.state.zones.zone_of(card);
                    // A token cannot return after leaving the battlefield (CR 111.7).
                    // Returning transformed requires a back face (CR 712.14a).
                    if *to == ZoneType::Battlefield
                        && ((self.state.cards[card].is_token
                            && from != Some(ZoneType::Battlefield))
                            || (*transformed
                                && default_content_pack()
                                    .back_face(self.state.cards[card].definition_id)
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
                        ZoneEntry::default()
                    };
                    events.push(self.zone_move_proposal(card, *to, entry));
                }
                let mut batch = self.event_batch(events);
                batch.binding = EventBinding::Moved {
                    linked: linked.and_then(|linked| {
                        source.map(|source| (source, linked == LinkedExile::UntilSourceLeaves))
                    }),
                    shuffle: *to == ZoneType::Library && library_end.is_none(),
                    library_end: *library_end,
                    reveal: *to == ZoneType::Hand,
                };
                frame.batch = Some(batch);
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
                    .filter_map(|id| self.permanent_object_ref(id))
                    .map(|target| ProposedEvent::Destroy { target })
                    .collect();
                frame.batch = Some(self.event_batch(events));
                None
            }
            Effect::DistributeCountersToTappedOpponentCreatures { kind, count } => {
                self.distribute_counters(frame.controller, *kind, *count)
            }
            Effect::DealDamageToPlayer { player, amount } => {
                let amount = self.effect_value(amount, frame);
                frame.batch = Some(
                    self.event_batch(
                        self.selected_players(player, frame)
                            .into_iter()
                            .map(|player| ProposedEvent::Damage {
                                source: frame.source_ref,
                                target: ProposedDamageTarget::Player(player),
                                amount,
                                combat: false,
                            })
                            .collect(),
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
            Effect::MillCards { count, target } => {
                let player = match target {
                    Some(_) => match self.present_primary_target(frame) {
                        Some(Target::Player(player)) => player,
                        _ => return None,
                    },
                    None => frame.controller,
                };
                let count = self.effect_value(count, frame).max(0) as usize;
                frame.this_way.clear();
                let cards = self.library_top(player, count);
                let events = cards
                    .into_iter()
                    .map(|card| {
                        self.zone_move_proposal(card, ZoneType::Graveyard, ZoneEntry::default())
                    })
                    .collect();
                let mut batch = self.event_batch(events);
                batch.binding = EventBinding::ThisWay;
                frame.batch = Some(batch);
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
                let candidates: Vec<_> = self
                    .select(from, frame)
                    .into_iter()
                    .filter(|object| {
                        self.card_matches_predicate(CardId::from(object.entity), predicate)
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
                            ObjectSelector::Zone(ZoneType::Library) => {
                                super::decision::SelectionOrigin::Library
                            }
                            ObjectSelector::Zone(ZoneType::Hand) => {
                                super::decision::SelectionOrigin::Hand
                            }
                            _ => super::decision::SelectionOrigin::Other,
                        },
                        player: frame.controller,
                        from: if let ObjectSelector::Zone(zone) = from {
                            Some(crate::state::binding::ZoneSelection {
                                owner: frame.controller,
                                zone: *zone,
                                predicate: predicate.clone(),
                            })
                        } else {
                            None
                        },
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
                for id in self.selected_permanents(what, frame) {
                    let Some(previous) = self.state.permanents[id].as_ref().map(|p| p.controller)
                    else {
                        continue;
                    };
                    self.change_control(id, player);
                    let affected = self.affected_permanent(id);
                    self.apply_continuous(
                        frame.source,
                        player,
                        affected,
                        Modification::ControlChanged { previous },
                        *duration,
                    );
                }
                None
            }
            Effect::Airbend { what } => {
                let selected = self.select(what, frame);
                let mut bent = false;
                for object in selected {
                    let card = CardId::from(object.entity);
                    match self.state.zones.zone_of(card) {
                        Some(ZoneType::Battlefield) => {
                            if let Ok(id) = self.lookup_current_permanent(object) {
                                self.airbend_permanent(id);
                                bent = true;
                            }
                        }
                        Some(ZoneType::Stack) => {
                            self.airbend_spell(card);
                            bent = true;
                        }
                        _ => {}
                    }
                }
                if bent {
                    self.record_bending(frame.controller, crate::flow::turn::Bending::Air);
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
                let mut batch = self.event_batch(vec![self.zone_move_proposal(
                    card,
                    ZoneType::Library,
                    Default::default(),
                )]);
                batch.binding = EventBinding::PlaceInLibrary {
                    second_from_top: *second_from_top,
                };
                frame.batch = Some(batch);
                None
            }
            Effect::ExileTopCardsToPlay { count, until } => {
                self.exile_top_cards_to_play(frame.controller, *count, *until);
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
                self.shuffle_library(player);
                None
            }
            Effect::DealDamage { amount, what } => {
                let amount = self.effect_value(amount, frame);
                let events = self
                    .resolve_affected(what, frame)
                    .into_iter()
                    .map(|affected| ProposedEvent::Damage {
                        source: frame.source_ref,
                        target: match affected {
                            Affected::Player(player) => ProposedDamageTarget::Player(player),
                            Affected::Object(object) => ProposedDamageTarget::Object(object),
                        },
                        amount,
                        combat: false,
                    })
                    .collect();
                let mut batch = self.event_batch(events);
                batch.binding = EventBinding::Damage { result: None };
                frame.batch = Some(batch);
                None
            }
            Effect::AddMana {
                mana,
                amount,
                until_end_of_combat,
            } => {
                let count = self.effect_value(amount, frame).max(0);
                self.invalidate_mana_cache(frame.controller);
                let pool = if *until_end_of_combat {
                    self.state.combat_mana_pool_mut(frame.controller)
                } else {
                    self.state.mana_pool_mut(frame.controller)
                };
                for _ in 0..count {
                    pool.add(mana);
                }
                if *until_end_of_combat {
                    self.record_bending(frame.controller, crate::flow::turn::Bending::Fire);
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
                let first_power = self.characteristics(first).power;
                let last_power = self.characteristics(last).power;
                let mut batch = self.event_batch(vec![
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
                ]);
                batch.binding = EventBinding::Damage { result: Some(0) };
                frame.batch = Some(batch);
                None
            }
            Effect::ShuffleLibrary => {
                self.shuffle_library(frame.controller);
                None
            }
            Effect::CounterSpell { what } => {
                // Spells only: countering an ability is a different effect.
                let spells: Vec<_> = self
                    .select(what, frame)
                    .into_iter()
                    .map(|object| CardId::from(object.entity))
                    .filter(|card| self.find_spell_on_stack_index(*card).is_some())
                    .collect();
                for spell in spells {
                    self.counter_spell(spell, frame.source);
                }
                None
            }
            Effect::CounterAbility { what } => {
                debug_assert!(
                    matches!(what, ObjectSelector::Target(_)),
                    "a stack ability is reached only as the clause's target"
                );
                if let Target::StackAbility(ability) = self.present_primary_target(frame)? {
                    self.counter_ability(ability, frame.source);
                }
                None
            }
            Effect::CreateTokenCopy { of, exceptions } => {
                let exceptions = exceptions
                    .iter()
                    .map(|exception| exception.resolve(|value| self.effect_value(value, frame)))
                    .collect();
                let copied = match of {
                    None => frame.source?,
                    // The target as it was chosen, even once the resolution
                    // has moved it ("exile ..., create a copy of it").
                    Some(_) => match frame.primary_target()? {
                        Target::Permanent(id) => {
                            self.present_primary_target(frame)?;
                            self.state.permanents[id].as_ref()?.card
                        }
                        // Copied once this resolution has exiled it.
                        Target::GraveyardCard(object) => {
                            let card = CardId::from(object.entity);
                            if self.state.zones.zone_of(card) != Some(ZoneType::Exile) {
                                return None;
                            }
                            card
                        }
                        _ => return None,
                    },
                };
                let mut batch = self.event_batch(vec![ProposedEvent::CreateTokens {
                    definition: Box::new(TokenDefinition::Copy {
                        definition_id: self.state.cards[copied].definition_id,
                        definition: self.state.cards[copied].copiable_definition(),
                        front: self.state.cards[copied].front_face().cloned(),
                        copy: self.state.cards[copied].copy.clone(),
                        exceptions,
                    }),
                    controller: frame.controller,
                    count: 1,
                    entry: ZoneEntry::default(),
                }]);
                batch.binding = EventBinding::Created { attacking: false };
                frame.batch = Some(batch);
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
            Effect::CastLibraryTopFree => self.library_top_free_cast(frame.controller, None),
            Effect::WhenYouDo { effects } => {
                // A spell's reflexive trigger has its card as its source.
                let trigger = self.enqueue_inline_trigger(
                    frame.source?,
                    frame.source_ref,
                    frame.controller,
                    effects.clone(),
                    frame.context_target,
                );
                trigger.remembered = frame.remembered.clone();
                trigger.x = frame.x;
                trigger.player = frame.triggering_player;
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
            Effect::DrawCards { count } => {
                // Drawing from an empty library sets `drew_when_empty`; the player
                // loses via state-based actions (CR 704.5c), same as the draw step.
                let count = self.effect_value(count, frame).max(0) as usize;
                self.draw_cards(frame.controller, count);
                None
            }
            Effect::CreateToken {
                token_name,
                count,
                tapped_and_attacking,
            } => {
                let count = self.effect_value(count, frame).max(0);
                let mut batch = self.event_batch(vec![ProposedEvent::CreateTokens {
                    definition: Box::new(TokenDefinition::Registered(token_name.clone())),
                    controller: frame.controller,
                    count,
                    entry: ZoneEntry {
                        tapped: *tapped_and_attacking,
                        ..Default::default()
                    },
                }]);
                batch.binding = EventBinding::Created {
                    attacking: *tapped_and_attacking,
                };
                frame.batch = Some(batch);
                None
            }
            Effect::AttachSourceTo { what } => {
                for permanent_id in self.selected_permanents(what, frame) {
                    if let FrameCompletion::Spell { attach_to, .. } = &mut frame.completion {
                        *attach_to = self.permanent_object_ref(permanent_id);
                    } else {
                        self.attach_source_to(frame, permanent_id);
                    }
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
            Effect::MayExileTriggeringWithSource => {
                // The card the dying creature became (CR 400.7), still in its
                // graveyard; `Move` exiles it as the frame's target and retains the link.
                let Some(Target::Permanent(id)) = frame.context_target else {
                    return None;
                };
                let card = self.departed_permanent_lki(id)?.card;
                if self.state.zones.zone_of(card) != Some(ZoneType::Graveyard) {
                    return None;
                }
                frame.targets = vec![crate::flow::decision::ResolutionTarget {
                    target: Target::GraveyardCard(self.current_object_ref(card)?),
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
                let source = frame.source_ref?;
                let modes: Vec<Vec<Effect>> = self
                    .state
                    .exile_links
                    .iter()
                    .filter(|link| link.source == source)
                    .map(|link| link.exiled_card)
                    .filter(|card| {
                        self.state.zones.zone_of(*card) == Some(ZoneType::Exile)
                            && predicate.matches_card(&self.state.cards[*card])
                    })
                    .map(|card| {
                        vec![Effect::HaveAbilitiesOf {
                            definition: self.state.cards[card].definition_id,
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
                // Every name a deck may hold; names are public knowledge.
                let names: std::collections::BTreeSet<String> = default_content_pack()
                    .definition_entries()
                    .filter(|(id, definition)| {
                        !definition.is_token
                            && (!definition.key.is_empty()
                                || default_content_pack().front_face(*id).is_some())
                    })
                    .map(|(_, definition)| definition.name.clone())
                    .collect();
                Some(Decision::Modal {
                    player: frame.controller,
                    modes: names
                        .into_iter()
                        .map(|name| {
                            vec![Effect::ExileNamed {
                                name,
                                count: *count,
                            }]
                        })
                        .collect(),
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
            Effect::EachPlayerKeepsCreatures { max_total_power } => {
                self.keep_creatures(self.active_player(), *max_total_power, Vec::new(), frame)
            }
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
                        source_card: frame.source?,
                        source_ref: frame.source_ref,
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
            Effect::Surveil { count } => {
                if *count > 0 {
                    frame.complete_with_event(GameEvent::Surveilled {
                        player: frame.controller,
                    });
                }
                let looked = self.library_top(frame.controller, *count);
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
            } => {
                let look = self.effect_value(look, frame).max(0) as usize;
                let looked = self.library_top(frame.controller, look);
                if looked.is_empty() {
                    return None;
                }
                Some(Decision::LookAndSelect {
                    random_order: *random_order,
                    reveal: *reveal,
                    to_battlefield: *to_battlefield,
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
                let mut events = Vec::new();
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
                    let power = self.characteristics(attacker_id).power;
                    if let Some(target) = self.permanent_object_ref(victim) {
                        events.push(ProposedEvent::Damage {
                            source: Some(attacker_ref),
                            target: ProposedDamageTarget::Object(target),
                            amount: power,
                            combat: false,
                        });
                    }
                }
                frame.batch = Some(self.event_batch(events));
                None
            }
            Effect::Earthbend { count, what } => {
                let count = self.effect_value(count, frame);
                let selected = self.selected_permanents(what, frame);
                let bent = !selected.is_empty();
                let mut events = Vec::new();
                for permanent_id in selected {
                    let Some(watched) = self.permanent_object_ref(permanent_id) else {
                        continue;
                    };
                    let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
                        continue;
                    };
                    let controller = permanent.controller;
                    self.animate_permanent(permanent_id, frame.source, controller);
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
                        source_card: CardId::from(watched.entity),
                        source_ref: Some(watched),
                        effects: vec![Effect::Move {
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
                let mut batch = self.event_batch(events);
                if bent {
                    batch.binding = EventBinding::Bending {
                        player: frame.controller,
                        kind: crate::flow::turn::Bending::Earth,
                    };
                }
                frame.batch = Some(batch);
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

    // -----------------------------------------------------------------------
    // Object selection.

    /// The objects a selector names as the instruction resolves. A `Chosen`
    /// selector names the object the resolving player chose for this
    /// instruction (`execute_frame_effect` suspended on the choice first).
    pub(crate) fn select(&self, what: &ObjectSelector, frame: &EffectFrame) -> Vec<ObjectRef> {
        let permanent = |id: Option<PermanentId>| {
            id.and_then(|id| self.permanent_object_ref(id))
                .into_iter()
                .collect()
        };
        match what {
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
            ObjectSelector::Triggering => permanent(match frame.context_target {
                Some(Target::Permanent(id)) => Some(id),
                _ => None,
            }),
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
            ObjectSelector::Remembered => frame
                .remembered
                .iter()
                .map(|binding| binding.object)
                .filter(|object| self.current_object_ref(object.entity.into()) == Some(*object))
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
                .filter(|object| {
                    self.current_object_ref(CardId::from(object.entity)) == Some(*object)
                })
                .collect()
            }
            ObjectSelector::Zone(zone) => self
                .selection_zone_cards(frame.controller, *zone)
                .into_iter()
                .filter_map(|card| self.current_object_ref(card))
                .collect(),
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
            } => {
                let players: Vec<PlayerId> = match of {
                    Scope::YouControl => vec![frame.controller],
                    Scope::OpponentsControl => vec![PlayerId(1 - frame.controller.0)],
                    Scope::Battlefield => vec![PlayerId(0), PlayerId(1)],
                    Scope::TargetPlayerControls => match self.present_primary_target(frame) {
                        Some(Target::Player(player)) => vec![player],
                        _ => Vec::new(),
                    },
                };
                let excepted = self.excepted_permanent(*except, frame);
                let max_mana_value = max_mana_value
                    .as_ref()
                    .map(|value| self.effect_value(value, frame));
                players
                    .into_iter()
                    .flat_map(|player| self.battlefield_permanents(player))
                    .filter(|id| Some(*id) != excepted)
                    .filter(|id| self.permanent_matches_predicate(*id, predicate))
                    .filter(|id| {
                        max_mana_value.is_none_or(|max| self.permanent_mana_value(*id) <= max)
                    })
                    .filter_map(|id| self.permanent_object_ref(id))
                    .collect()
            }
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
    fn permanent_mana_value(&self, id: PermanentId) -> i32 {
        self.state.permanents[id]
            .as_ref()
            .and_then(|permanent| self.state.cards[permanent.card].mana_cost.as_ref())
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
        let departed = |source: ObjectRef| self.object_lki(source).map(|lki| lki.counters.clone());
        let present = |id: PermanentId| {
            self.state.permanents[id]
                .as_ref()
                .map(|permanent| permanent.counters.clone())
        };
        match what {
            ObjectSelector::This => {
                frame
                    .source_ref
                    .and_then(|source| match self.lookup_current_permanent(source) {
                        Ok(id) => present(id),
                        Err(_) => departed(source),
                    })
            }
            ObjectSelector::Triggering => match frame.context_target {
                Some(Target::Permanent(id)) => present(id).or_else(|| {
                    self.departed_permanent_lki(id)
                        .map(|lki| lki.counters.clone())
                }),
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
    fn source_object(&self, frame: &EffectFrame) -> Option<ObjectRef> {
        let card = frame.source?;
        let current = self.current_object_ref(card)?;
        match frame.source_ref {
            Some(source) if current == source => Some(current),
            Some(source) => (Some(current.incarnation) == source.incarnation.checked_next()
                && matches!(
                    self.state.zones.zone_of(card),
                    Some(ZoneType::Graveyard | ZoneType::Exile)
                ))
            .then_some(current),
            None => self
                .find_spell_on_stack_index(card)
                .is_some()
                .then_some(current),
        }
    }

    /// The exact object a chosen target is, when it is a card object.
    fn target_object(&self, target: Target) -> Option<ObjectRef> {
        match target {
            Target::Permanent(id) => self.permanent_object_ref(id),
            Target::GraveyardCard(object) => Some(object),
            Target::StackSpell(card) => self.current_object_ref(card),
            Target::Player(_) | Target::StackAbility(_) => None,
        }
    }

    pub(crate) fn source_permanent_id(&self, frame: &EffectFrame) -> Option<PermanentId> {
        frame
            .source_ref
            .and_then(|source| self.lookup_current_permanent(source).ok())
    }

    /// Attach the source Equipment to a battlefield creature (CR 701.3). It
    /// does nothing once the Equipment has left the battlefield.
    pub(crate) fn attach_source_to(&mut self, frame: &EffectFrame, permanent_id: PermanentId) {
        let Some(object) = self.permanent_object_ref(permanent_id) else {
            return;
        };
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
                    && (permanent.controller == controller
                        || !self.characteristics(id).keywords.has(Keyword::Hexproof))
                    && self.permanent_matches_spec(id, spec, controller)
            }
            Target::Player(player) => match spec {
                TargetSpec::Player | TargetSpec::AnyTarget => true,
                TargetSpec::Opponent => player != controller,
                _ => false,
            },
            Target::StackSpell(card) => {
                self.find_spell_on_stack_index(card).is_some()
                    && match spec {
                        TargetSpec::Spell | TargetSpec::CreatureOrSpell => true,
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
            Target::GraveyardCard(object) => {
                let (predicate, you_own) = match spec {
                    TargetSpec::GraveyardCardYouOwn { predicate }
                    | TargetSpec::GraveyardCardYouOwnWithinSourcePower { predicate } => {
                        (predicate, true)
                    }
                    TargetSpec::GraveyardCard { predicate } => (predicate, false),
                    TargetSpec::GraveyardCardOpponentOwns { predicate } => {
                        let card = CardId::from(object.entity);
                        if self.state.cards[card].owner == controller {
                            return false;
                        }
                        (predicate, false)
                    }
                    _ => return false,
                };
                let card = CardId::from(object.entity);
                self.current_object_ref(card) == Some(object)
                    && self.state.zones.zone_of(card) == Some(ZoneType::Graveyard)
                    && (!you_own || self.state.cards[card].owner == controller)
                    && self.card_matches_predicate(card, predicate)
            }
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
                        .map(|id| self.characteristics(id).power)
                })
            })
            .unwrap_or_else(|| self.card_power(source_card));
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
            TargetSpec::AnyTarget => {
                let types = &self.characteristics(id).types;
                types.is_creature() || types.is_planeswalker() || types.is_battle()
            }
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
                self.card_mana_value(permanent.card) >= *min_mana_value
            }
            _ => false,
        }
    }

    /// The card's mana value; a spell on the stack counts its announced X
    /// (CR 202.3e).
    pub(crate) fn card_mana_value(&self, card: CardId) -> u8 {
        let Some(cost) = self.state.cards[card].mana_value_cost() else {
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
