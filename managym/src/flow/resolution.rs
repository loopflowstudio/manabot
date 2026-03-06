use crate::{
    flow::{event::GameEvent, game::Game},
    state::{
        ability::{Ability, Effect, TargetSpec},
        game_object::{CardId, PermanentId, Target},
        stack_object::{ActivatedAbilityOnStack, SpellOnStack, StackObject},
        target::Target as ActionTarget,
        zone::ZoneType,
    },
};

impl Game {
    pub(crate) fn resolve_top_of_stack(&mut self) {
        let Some(stack_object) = self.state.stack_objects.pop() else {
            return;
        };

        match stack_object {
            StackObject::Spell(spell) => self.resolve_spell_object(spell),
            StackObject::ActivatedAbility(ability) => self.resolve_activated_ability(ability),
            StackObject::TriggeredAbility(triggered) => self.resolve_triggered_ability(
                triggered.source_card,
                triggered.ability_index,
                triggered.targets.first().copied(),
            ),
        }
    }

    fn resolve_spell_object(&mut self, spell: SpellOnStack) {
        let card = spell.card;
        let spell_effect = self.state.cards[card].spell_effect.clone();

        if let Some(effect) = spell_effect {
            let target = spell.targets.first().copied();
            if let Some(target_spec) = effect.target_spec() {
                let Some(target) = target else {
                    // CR 608.2b — Targeted spells with no target are countered.
                    self.counter_spell(card, None);
                    return;
                };
                if !self.is_legal_target(target, target_spec) {
                    // CR 608.2b — Targeted spells with illegal targets are countered.
                    self.counter_spell(card, None);
                    return;
                }
            }

            self.execute_spell_effect(&effect, target, card);
            self.move_card(card, ZoneType::Graveyard);
            self.emit(GameEvent::SpellResolved { card });
            return;
        }

        let is_permanent = self.state.cards[card].types.is_permanent();
        if is_permanent {
            // CR 608.3 — A resolving permanent spell enters the battlefield.
            self.move_card(card, ZoneType::Battlefield);
        } else {
            // CR 608.2k — Nonpermanent spells resolve then go to graveyard.
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
        let Some(source_permanent_id) = self.state.card_to_permanent[ability.source_card] else {
            return;
        };
        let Some(source_permanent) = self.state.permanents[source_permanent_id].as_ref() else {
            return;
        };
        if source_permanent.id != ability.source_permanent_object_id {
            return;
        }

        let Some(effect) = self.state.cards[ability.source_card]
            .activated_abilities
            .get(ability.ability_index)
            .map(|def| def.effect.clone())
        else {
            return;
        };

        let action_target = ability.targets.first().and_then(|t| match t {
            Target::Player(p) => Some(ActionTarget::Player(*p)),
            Target::Permanent(p) => Some(ActionTarget::Permanent(*p)),
            Target::StackSpell(_) => None,
        });
        self.execute_effect(&effect, action_target, Some(ability.source_card));
    }

    fn resolve_triggered_ability(
        &mut self,
        source_card: CardId,
        ability_index: usize,
        target: Option<Target>,
    ) {
        let Some(ability) = self
            .state
            .cards
            .get(source_card.0)
            .and_then(|card| card.abilities.get(ability_index))
            .cloned()
        else {
            return;
        };

        let Ability::Triggered {
            effect,
            intervening_if,
            ..
        } = ability;

        if let Some(condition) = intervening_if.as_ref() {
            // CR 603.4 — Intervening "if" clauses are checked on resolution.
            if !self.check_trigger_condition(condition, source_card) {
                return;
            }
        }

        let action_target = target.and_then(|t| match t {
            Target::Player(p) => Some(ActionTarget::Player(p)),
            Target::Permanent(p) => Some(ActionTarget::Permanent(p)),
            Target::StackSpell(_) => None,
        });
        self.execute_effect(&effect, action_target, Some(source_card));
    }

    /// Execute a spell effect, with access to the raw Target (needed for StackSpell targets).
    fn execute_spell_effect(&mut self, effect: &Effect, target: Option<Target>, source: CardId) {
        match effect {
            Effect::CounterSpell { .. } => {
                let Some(Target::StackSpell(target_spell)) = target else {
                    return;
                };
                self.counter_spell(target_spell, Some(source));
            }
            _ => {
                let action_target = target.map(|t| match t {
                    Target::Player(p) => ActionTarget::Player(p),
                    Target::Permanent(p) => ActionTarget::Permanent(p),
                    Target::StackSpell(c) => ActionTarget::StackSpell(c),
                });
                self.execute_effect(effect, action_target, Some(source));
            }
        }
    }

    fn execute_effect(
        &mut self,
        effect: &Effect,
        target: Option<ActionTarget>,
        source: Option<CardId>,
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
                        self.apply_player_damage(source, player, *amount);
                    }
                    ActionTarget::Permanent(permanent_id) => {
                        self.apply_permanent_damage(source, permanent_id, *amount);
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
                let Some(source_card) = source else { return };
                let Some(perm_id) = self.state.card_to_permanent[source_card] else {
                    return;
                };
                let Some(permanent) = self.state.permanents[perm_id].as_mut() else {
                    return;
                };
                permanent.temp_power += power_delta;
                permanent.temp_toughness += toughness_delta;
            }
        }
    }

    fn return_permanent_to_owner_hand(&mut self, permanent_id: PermanentId) {
        let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
            return;
        };
        let card = permanent.card;
        self.move_card(card, ZoneType::Hand);
    }

    pub(crate) fn find_spell_on_stack_index(&self, card: CardId) -> Option<usize> {
        self.state
            .stack_objects
            .iter()
            .position(|object| matches!(object, StackObject::Spell(spell) if spell.card == card))
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
