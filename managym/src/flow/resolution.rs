use crate::{
    flow::{event::GameEvent, game::Game},
    state::{
        ability::{Ability, Effect, TargetSpec},
        game_object::{CardId, PermanentId, PlayerId, Target},
        permanent::Permanent,
        stack_object::{
            ActivatedAbilityOnStack, SpellOnStack, StackObject, TriggeredAbilityOnStack,
        },
        target::Target as ActionTarget,
        zone::ZoneType,
    },
};

/// Everything an effect needs to know about what is resolving it.
pub(crate) struct EffectContext {
    /// The card whose spell or ability is resolving.
    pub source: Option<CardId>,
    /// The player the effect resolves for ("you").
    pub controller: PlayerId,
    /// For triggered abilities: how many times this ability has resolved
    /// this turn, counting this resolution. Zero for spells and activated
    /// abilities.
    pub resolutions_this_turn: u32,
}

impl Game {
    pub(crate) fn resolve_top_of_stack(&mut self) {
        let Some(stack_object) = self.pop_stack() else {
            return;
        };

        match stack_object {
            StackObject::Spell(spell) => self.resolve_spell_object(spell),
            StackObject::ActivatedAbility(ability) => self.resolve_activated_ability(ability),
            StackObject::TriggeredAbility(triggered) => self.resolve_triggered_ability(triggered),
        }
    }

    fn resolve_spell_object(&mut self, spell: SpellOnStack) {
        let card = spell.card;
        let spell_effect = self.state.cards[card].spell_effect.clone();

        if let Some(effect) = spell_effect {
            let target = spell.targets.first().copied();
            if let Some(target_spec) = effect.target_spec() {
                // CR 608.2b — A spell whose targets are all illegal on
                // resolution doesn't resolve and is put into its owner's
                // graveyard. The stack object was already popped, so
                // counter_spell (which searches the stack) would no-op and
                // strand the card in the stack zone; move it directly.
                let fizzles = match target {
                    None => true,
                    Some(target) => !self.is_legal_target(target, target_spec),
                };
                if fizzles {
                    self.move_card(card, ZoneType::Graveyard);
                    self.emit(GameEvent::SpellCountered { card, by: None });
                    return;
                }
            }

            let ctx = EffectContext {
                source: Some(card),
                controller: spell.controller,
                resolutions_this_turn: 0,
            };
            self.execute_spell_effect(&effect, target, &ctx);
            self.move_card(card, ZoneType::Graveyard);
            self.emit(GameEvent::SpellResolved { card });
            return;
        }

        let is_permanent = self.state.cards[card].types.is_permanent();
        if is_permanent {
            self.move_card(card, ZoneType::Battlefield);
            let owner = self.state.cards[card].owner;
            self.invalidate_mana_cache(owner);
        } else {
            self.move_card(card, ZoneType::Graveyard);
        }
        self.emit(GameEvent::SpellResolved { card });
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

        let action_target = ability.targets.first().and_then(|t| match t {
            Target::Player(p) => Some(ActionTarget::Player(*p)),
            Target::Permanent(p) => Some(ActionTarget::Permanent(*p)),
            Target::StackSpell(_) => None,
        });
        let ctx = EffectContext {
            source: Some(ability.source_card),
            controller: ability.controller,
            resolutions_this_turn: 0,
        };
        self.execute_effect(&definition.effect, action_target, &ctx);
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

        let action_target = triggered.targets.first().and_then(|t| match t {
            Target::Player(p) => Some(ActionTarget::Player(*p)),
            Target::Permanent(p) => Some(ActionTarget::Permanent(*p)),
            Target::StackSpell(_) => None,
        });
        let ctx = EffectContext {
            source: Some(triggered.source_card),
            controller: triggered.controller,
            resolutions_this_turn,
        };
        for effect in &effects {
            self.execute_effect(effect, action_target, &ctx);
        }
    }

    /// Execute a spell effect, with access to the raw Target (needed for StackSpell targets).
    fn execute_spell_effect(
        &mut self,
        effect: &Effect,
        target: Option<Target>,
        ctx: &EffectContext,
    ) {
        match effect {
            Effect::CounterSpell { .. } => {
                let Some(Target::StackSpell(target_spell)) = target else {
                    return;
                };
                self.counter_spell(target_spell, ctx.source);
            }
            _ => {
                let action_target = target.map(|t| match t {
                    Target::Player(p) => ActionTarget::Player(p),
                    Target::Permanent(p) => ActionTarget::Permanent(p),
                    Target::StackSpell(c) => ActionTarget::StackSpell(c),
                });
                self.execute_effect(effect, action_target, ctx);
            }
        }
    }

    fn execute_effect(
        &mut self,
        effect: &Effect,
        target: Option<ActionTarget>,
        ctx: &EffectContext,
    ) {
        match effect {
            Effect::ReturnToHand { target: spec } => {
                let Some(chosen) = target else { return };
                if !self.is_valid_target_for_spec(chosen, spec) {
                    return;
                }
                if let ActionTarget::Permanent(permanent_id) = chosen {
                    self.return_permanent_to_owner_hand(permanent_id);
                }
            }
            Effect::DealDamage { amount, .. } => {
                let Some(chosen) = target else { return };
                match chosen {
                    ActionTarget::Player(player) => {
                        self.apply_player_damage(ctx.source, player, *amount);
                    }
                    ActionTarget::Permanent(permanent_id) => {
                        self.apply_permanent_damage(ctx.source, permanent_id, *amount);
                    }
                    ActionTarget::StackSpell(_) => {}
                }
            }
            Effect::CounterSpell { .. } => {
                // Handled by execute_spell_effect — should not reach here.
            }
            Effect::ModifyUntilEot {
                power_delta,
                toughness_delta,
            } => {
                if let Some(permanent) = self.source_permanent_mut(ctx) {
                    permanent.temp_power += power_delta;
                    permanent.temp_toughness += toughness_delta;
                }
            }
            Effect::DrawCards { count } => {
                // Drawing from an empty library sets `drew_when_empty`; the player
                // loses via state-based actions (CR 704.5c), same as the draw step.
                self.draw_cards(ctx.controller, *count);
            }
            Effect::MassDamage { amount } => {
                for player in [PlayerId(0), PlayerId(1)] {
                    for permanent_id in self.battlefield_permanents(player) {
                        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
                            continue;
                        };
                        if self.state.cards[permanent.card].types.is_creature() {
                            self.apply_permanent_damage(ctx.source, permanent_id, *amount);
                        }
                    }
                }
                // Lethal damage is cleaned up by state-based actions (CR 704.5g).
            }
            Effect::CreateToken {
                token_name,
                count,
                tapped_and_attacking,
            } => {
                for _ in 0..*count {
                    self.create_token(token_name, ctx.controller, *tapped_and_attacking);
                }
            }
            Effect::PutCountersOnSource { count } => {
                if let Some(permanent) = self.source_permanent_mut(ctx) {
                    permanent.plus1_counters += count;
                }
            }
            Effect::PutCounters {
                count,
                target: spec,
            } => {
                let Some(chosen) = target else { return };
                if !self.is_valid_target_for_spec(chosen, spec) {
                    return;
                }
                if let ActionTarget::Permanent(permanent_id) = chosen {
                    if let Some(permanent) = self.state.permanents[permanent_id].as_mut() {
                        permanent.plus1_counters += count;
                    }
                }
            }
            Effect::TapSource => {
                if let Some(permanent_id) = self.source_permanent_id(ctx) {
                    self.tap_permanent(permanent_id, false);
                }
            }
            Effect::UntapSource => {
                if let Some(permanent) = self.source_permanent_mut(ctx) {
                    permanent.untap();
                    let controller = permanent.controller;
                    self.invalidate_mana_cache(controller);
                }
            }
            Effect::CantBeBlockedThisTurnSource => {
                if let Some(permanent) = self.source_permanent_mut(ctx) {
                    permanent.cant_be_blocked_this_turn = true;
                }
            }
            Effect::GainLife { amount } => {
                self.gain_life(ctx.controller, *amount);
            }
            Effect::OnNthResolutionThisTurn { n, effect } => {
                if ctx.resolutions_this_turn == *n {
                    self.execute_effect(effect, target, ctx);
                }
            }
        }
    }

    fn source_permanent_id(&self, ctx: &EffectContext) -> Option<PermanentId> {
        let source_card = ctx.source?;
        self.state.card_to_permanent[source_card]
    }

    fn source_permanent_mut(&mut self, ctx: &EffectContext) -> Option<&mut Permanent> {
        let permanent_id = self.source_permanent_id(ctx)?;
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

    /// Check whether a game-object Target is still legal for a given TargetSpec.
    fn is_legal_target(&self, target: Target, spec: &TargetSpec) -> bool {
        match (target, spec) {
            (Target::Player(_), TargetSpec::CreatureOrPlayer) => true,
            (Target::Permanent(perm_id), TargetSpec::CreatureOrPlayer | TargetSpec::Creature) => {
                let Some(permanent) = self
                    .state
                    .permanents
                    .get(perm_id.0)
                    .and_then(|p| p.as_ref())
                else {
                    return false;
                };
                self.state.zones.zone_of(permanent.card) == Some(ZoneType::Battlefield)
                    && self.state.cards[permanent.card].types.is_creature()
            }
            (Target::StackSpell(card_id), TargetSpec::Spell) => {
                self.find_spell_on_stack_index(card_id).is_some()
            }
            _ => false,
        }
    }
}
