use crate::{
    flow::game::Game,
    state::{
        ability::{Ability, Effect},
        game_object::PermanentId,
        stack::StackObject,
        target::Target,
        zone::ZoneType,
    },
};

impl Game {
    pub(crate) fn resolve_top_of_stack(&mut self) {
        let Some(stack_object) = self.state.stack.pop() else {
            return;
        };

        match stack_object {
            StackObject::Spell { card } => {
                let is_permanent = self.state.cards[card].types.is_permanent();
                if is_permanent {
                    // CR 608.3 — A resolving permanent spell enters the battlefield.
                    self.move_card(card, ZoneType::Battlefield);
                    let owner = self.state.cards[card].owner;
                    self.invalidate_mana_cache(owner);
                } else {
                    // CR 608.2k — Nonpermanent spells resolve then go to graveyard.
                    self.move_card(card, ZoneType::Graveyard);
                }
            }
            StackObject::TriggeredAbility {
                source_card,
                ability_index,
                target,
                ..
            } => self.resolve_triggered_ability(source_card, ability_index, target),
        }
    }

    fn resolve_triggered_ability(
        &mut self,
        source_card: crate::state::game_object::CardId,
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

        self.execute_effect(&effect, target);
    }

    fn execute_effect(&mut self, effect: &Effect, target: Option<Target>) {
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

                if let Target::Permanent(permanent_id) = chosen_target {
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
        let controller = permanent.controller;
        self.move_card(card, ZoneType::Hand);
        self.invalidate_mana_cache(controller);
    }
}
