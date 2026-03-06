use crate::{
    flow::{event::GameEvent, game::Game},
    state::{
        ability::{Ability, Effect},
        card::ActivatedAbilityEffect,
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
        match self.state.cards[card].name.as_str() {
            "Lightning Bolt" => {
                self.resolve_lightning_bolt(&spell);
                return;
            }
            "Counterspell" => {
                self.resolve_counterspell(&spell);
                return;
            }
            _ => {}
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

    fn resolve_lightning_bolt(&mut self, spell: &SpellOnStack) {
        let card = spell.card;
        let Some(target) = spell.targets.first().copied() else {
            self.counter_spell(card, None);
            return;
        };

        if !self.is_legal_target_for_bolt(target) {
            // CR 608.2b — Spells with illegal targets are countered by game rules.
            self.counter_spell(card, None);
            return;
        }

        match target {
            Target::Player(player) => self.apply_player_damage(Some(card), player, 3),
            Target::Permanent(permanent) => self.apply_permanent_damage(Some(card), permanent, 3),
            Target::StackSpell(_) => {
                self.counter_spell(card, None);
                return;
            }
        }

        self.move_card(card, ZoneType::Graveyard);
        self.emit(GameEvent::SpellResolved { card });
    }

    fn resolve_counterspell(&mut self, spell: &SpellOnStack) {
        let card = spell.card;
        let Some(Target::StackSpell(target_spell)) = spell.targets.first().copied() else {
            self.counter_spell(card, None);
            return;
        };

        if self.find_spell_on_stack_index(target_spell).is_none() {
            self.counter_spell(card, None);
            return;
        }

        self.counter_spell(target_spell, Some(card));
        self.move_card(card, ZoneType::Graveyard);
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
        let Some(source_permanent) = self.state.permanents[source_permanent_id].as_mut() else {
            return;
        };
        if source_permanent.id != ability.source_permanent_object_id {
            return;
        }

        let Some(ability_definition) = self.state.cards[ability.source_card]
            .activated_abilities
            .get(ability.ability_index)
        else {
            return;
        };

        match ability_definition.effect {
            ActivatedAbilityEffect::SelfGetsUntilEot {
                power_delta,
                toughness_delta,
            } => {
                source_permanent.temp_power += power_delta;
                source_permanent.temp_toughness += toughness_delta;
            }
        }
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
        self.execute_effect(&effect, action_target);
    }

    fn execute_effect(&mut self, effect: &Effect, target: Option<ActionTarget>) {
        match effect {
            Effect::ReturnToHand {
                target: target_spec,
            } => {
                let Some(chosen_target) = target else {
                    return;
                };
                // CR 608.2b — Revalidate target legality on resolution.
                if !self.is_valid_target_for_spec(chosen_target, target_spec) {
                    return;
                }

                if let ActionTarget::Permanent(permanent_id) = chosen_target {
                    self.return_permanent_to_owner_hand(permanent_id);
                }
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

    fn is_legal_target_for_bolt(&self, target: Target) -> bool {
        match target {
            Target::Player(player) => self.state.players.get(player.0).is_some(),
            Target::Permanent(permanent_id) => {
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
            Target::StackSpell(_) => false,
        }
    }
}
