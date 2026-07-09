// resolution.rs
// Resolving stack objects. Effects execute through an EffectFrame so a
// resolution can suspend on a mid-resolution decision (see decision.rs).

use std::collections::VecDeque;

use crate::{
    flow::{
        decision::{Decision, EffectFrame, FrameFinalize},
        event::GameEvent,
        game::Game,
    },
    state::{
        ability::{Ability, Effect, TargetSpec},
        game_object::{CardId, PermanentId, PlayerId, Target},
        permanent::Permanent,
        stack_object::{ActivatedAbilityOnStack, SpellOnStack, StackObject, TriggeredAbilityOnStack},
        zone::ZoneType,
    },
};

impl Game {
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
        let effects = card_ref.spell_effects.clone();
        let requirements = card_ref.target_requirements();

        // CR 608.2b — A spell whose targets are all illegal on resolution
        // doesn't resolve and is put into its owner's graveyard.
        if !spell.targets.is_empty() {
            let controller = spell.controller;
            let any_legal = spell.targets.iter().enumerate().any(|(i, target)| {
                let req_index = spell.target_req_indices.get(i).copied().unwrap_or(0);
                requirements
                    .get(req_index)
                    .is_some_and(|req| self.target_is_legal(*target, &req.spec, controller))
            });
            if !any_legal {
                if let Some(index) = self.find_spell_on_stack_index(card) {
                    self.state.stack_objects.remove(index);
                }
                self.move_card(card, ZoneType::Graveyard);
                self.emit(GameEvent::SpellCountered { card, by: None });
                return;
            }
        }

        let frame = EffectFrame {
            source: Some(card),
            controller: spell.controller,
            resolutions_this_turn: 0,
            kicked: spell.kicked,
            targets: spell.targets.clone(),
            target_req_indices: spell.target_req_indices.clone(),
            context_target: None,
            queue: VecDeque::from(effects),
            finalize: FrameFinalize::Spell { card },
        };
        self.run_frame(frame);
    }

    pub(crate) fn counter_spell(&mut self, card: CardId, by: Option<CardId>) {
        let Some(index) = self.find_spell_on_stack_index(card) else {
            return;
        };
        self.state.stack_objects.remove(index);
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

        // The resolving ability must still belong to the permanent that
        // activated it — a new permanent from the same card re-entering the
        // battlefield must not receive its effects. When the source was
        // sacrificed as an activation cost it is expected to be gone and the
        // ability resolves anyway (CR 113.7a).
        if !definition.sacrifice_source {
            let Some(source_permanent_id) = self.state.card_to_permanent[ability.source_card]
            else {
                return;
            };
            let Some(source_permanent) = self.state.permanents[source_permanent_id].as_ref()
            else {
                return;
            };
            if source_permanent.id != ability.source_permanent_object_id {
                return;
            }
        }

        let frame = EffectFrame {
            source: Some(ability.source_card),
            controller: ability.controller,
            resolutions_this_turn: 0,
            kicked: false,
            targets: ability.targets.clone(),
            target_req_indices: Vec::new(),
            context_target: None,
            queue: VecDeque::from(vec![definition.effect]),
            finalize: FrameFinalize::None,
        };
        self.run_frame(frame);
    }

    fn resolve_triggered_ability(&mut self, triggered: TriggeredAbilityOnStack) {
        let Some(ability) = self
            .state
            .cards
            .get(triggered.source_card.0)
            .and_then(|card| card.abilities.get(triggered.ability_index))
            .cloned()
        else {
            return;
        };

        let Ability::Triggered { effects, .. } = ability;

        // Track per-turn resolutions for "the Nth time this ability has
        // resolved this turn" gating.
        let key = (triggered.source_card.0, triggered.ability_index);
        let count = self
            .state
            .turn
            .ability_resolutions_this_turn
            .entry(key)
            .or_insert(0);
        *count += 1;
        let resolutions_this_turn = *count;

        let frame = EffectFrame {
            source: Some(triggered.source_card),
            controller: triggered.controller,
            resolutions_this_turn,
            kicked: false,
            targets: triggered.targets.clone(),
            target_req_indices: Vec::new(),
            context_target: triggered.context,
            queue: VecDeque::from(effects),
            finalize: FrameFinalize::None,
        };
        self.run_frame(frame);
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
            Effect::ReturnToHand { target: spec } => {
                let chosen = frame.primary_target()?;
                if !self.target_is_legal(chosen, spec, frame.controller) {
                    return None;
                }
                match chosen {
                    Target::Permanent(permanent_id) => {
                        self.return_permanent_to_owner_hand(permanent_id);
                    }
                    Target::StackSpell(card) => {
                        // Bounce a spell: its stack object ceases and the
                        // card returns to its owner's hand.
                        if let Some(index) = self.find_spell_on_stack_index(card) {
                            self.state.stack_objects.remove(index);
                            let owner = self.state.cards[card].owner;
                            self.move_card(card, ZoneType::Hand);
                            self.invalidate_mana_cache(owner);
                        }
                    }
                    Target::Player(_) => {}
                }
                None
            }
            Effect::DealDamage { amount, target: spec } => {
                let chosen = frame.primary_target()?;
                if !self.target_is_legal(chosen, spec, frame.controller) {
                    return None;
                }
                match chosen {
                    Target::Player(player) => {
                        self.apply_player_damage(frame.source, player, *amount);
                    }
                    Target::Permanent(permanent_id) => {
                        self.apply_permanent_damage(frame.source, permanent_id, *amount);
                    }
                    Target::StackSpell(_) => {}
                }
                None
            }
            Effect::CounterSpell { .. } => {
                let Some(Target::StackSpell(target_spell)) = frame.primary_target() else {
                    return None;
                };
                self.counter_spell(target_spell, frame.source);
                None
            }
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
            Effect::DrawCards { count } => {
                // Drawing from an empty library sets `drew_when_empty`; the player
                // loses via state-based actions (CR 704.5c), same as the draw step.
                self.draw_cards(frame.controller, *count);
                None
            }
            Effect::MassDamage { amount } => {
                for player in [PlayerId(0), PlayerId(1)] {
                    for permanent_id in self.battlefield_permanents(player) {
                        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
                            continue;
                        };
                        if self.state.cards[permanent.card].types.is_creature() {
                            self.apply_permanent_damage(frame.source, permanent_id, *amount);
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
                if let Some(permanent) = self.source_permanent_mut(frame) {
                    permanent.plus1_counters += count;
                }
                None
            }
            Effect::PutCounters { count, target: spec } => {
                let chosen = frame.primary_target()?;
                if !self.target_is_legal(chosen, spec, frame.controller) {
                    return None;
                }
                if let Target::Permanent(permanent_id) = chosen {
                    if let Some(permanent) = self.state.permanents[permanent_id].as_mut() {
                        permanent.plus1_counters += count;
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
                    remaining,
                })
            }
            Effect::LookAndSelect {
                look,
                min_select,
                max_select,
                predicate,
            } => {
                let looked = self.library_top(frame.controller, *look);
                if looked.is_empty() {
                    return None;
                }
                Some(Decision::LookAndSelect {
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
            Effect::Learn => Some(Decision::DiscardThenDraw {
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
                self.find_spell_on_stack_index(target_spell)?;
                let spell_controller = self.spell_controller(target_spell)?;
                Some(Decision::PayOrNot {
                    player: spell_controller,
                    cost: cost.clone(),
                    if_paid: Vec::new(),
                    if_declined: vec![Effect::CounterSpell {
                        target: TargetSpec::Spell,
                    }],
                })
            }
            Effect::IfKicked { then, otherwise } => {
                let branch = if frame.kicked { then } else { otherwise };
                for effect in branch.iter().rev() {
                    frame.queue.push_front(effect.clone());
                }
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
                for effect in branch.iter().rev() {
                    frame.queue.push_front(effect.clone());
                }
                None
            }
            Effect::TargetCreaturesDealPowerDamageToLastTarget => {
                let targets = frame.targets.clone();
                if targets.len() < 2 {
                    return None;
                }
                let Some(Target::Permanent(victim)) = targets.last().copied() else {
                    return None;
                };
                for attacker in &targets[..targets.len() - 1] {
                    let Target::Permanent(attacker_id) = attacker else {
                        continue;
                    };
                    // Both creatures must still be legal (on the
                    // battlefield) when damage is dealt.
                    if !self.permanent_is_battlefield_creature(victim) {
                        break;
                    }
                    if !self.permanent_is_battlefield_creature(*attacker_id) {
                        continue;
                    }
                    let Some(attacker_perm) = self.state.permanents[*attacker_id].as_ref() else {
                        continue;
                    };
                    let attacker_card = attacker_perm.card;
                    let power = attacker_perm
                        .effective_power(&self.state.cards[attacker_card]);
                    if power > 0 {
                        self.apply_permanent_damage(Some(attacker_card), victim, power);
                    }
                }
                None
            }
        }
    }

    fn spell_controller(&self, card: CardId) -> Option<PlayerId> {
        self.state.stack_objects.iter().find_map(|object| match object {
            StackObject::Spell(spell) if spell.card == card => Some(spell.controller),
            _ => None,
        })
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
            && self.state.cards[permanent.card].types.is_creature()
    }

    fn source_permanent_id(&self, frame: &EffectFrame) -> Option<PermanentId> {
        let source_card = frame.source?;
        self.state.card_to_permanent[source_card]
    }

    fn source_permanent_mut(&mut self, frame: &EffectFrame) -> Option<&mut Permanent> {
        let permanent_id = self.source_permanent_id(frame)?;
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
        match (target, spec) {
            (Target::Player(_), TargetSpec::CreatureOrPlayer) => true,
            (Target::Permanent(perm_id), TargetSpec::CreatureOrPlayer | TargetSpec::Creature) => {
                self.permanent_is_battlefield_creature(perm_id)
            }
            (Target::Permanent(perm_id), TargetSpec::CreatureYouControl) => {
                self.permanent_is_battlefield_creature(perm_id)
                    && self.state.permanents[perm_id]
                        .as_ref()
                        .is_some_and(|p| p.controller == controller)
            }
            (Target::Permanent(perm_id), TargetSpec::CreatureOpponentControls) => {
                self.permanent_is_battlefield_creature(perm_id)
                    && self.state.permanents[perm_id]
                        .as_ref()
                        .is_some_and(|p| p.controller != controller)
            }
            (Target::StackSpell(card_id), TargetSpec::Spell) => {
                self.find_spell_on_stack_index(card_id).is_some()
            }
            (Target::StackSpell(card_id), TargetSpec::SpellOrPermanent { min_mana_value }) => {
                self.find_spell_on_stack_index(card_id).is_some()
                    && self.card_mana_value(card_id) >= *min_mana_value
            }
            (Target::Permanent(perm_id), TargetSpec::SpellOrPermanent { min_mana_value }) => {
                let Some(permanent) = self
                    .state
                    .permanents
                    .get(perm_id.0)
                    .and_then(|p| p.as_ref())
                else {
                    return false;
                };
                self.state.zones.zone_of(permanent.card) == Some(ZoneType::Battlefield)
                    && self.card_mana_value(permanent.card) >= *min_mana_value
            }
            _ => false,
        }
    }

    pub(crate) fn card_mana_value(&self, card: CardId) -> u8 {
        self.state.cards[card]
            .mana_cost
            .as_ref()
            .map(|cost| cost.mana_value)
            .unwrap_or(0)
    }
}
