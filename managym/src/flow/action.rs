// action.rs
// Player action computation and action space construction.

use crate::{
    agent::action::Action,
    flow::game::Game,
    state::{game_object::PlayerId, mana::Mana, zone::ZoneType},
};

impl Game {
    pub(crate) fn compute_player_actions(&mut self, player: PlayerId) -> Vec<Action> {
        let hand = self.state.zones.zone_cards(ZoneType::Hand, player).to_vec();

        let can_play_land = self.can_play_land(player);
        let can_cast_sorcery = self.can_cast_sorceries(player);

        let mut actions = Vec::new();
        let mut producible: Option<Mana> = None;

        for card_id in hand {
            let (is_land, is_castable, is_instant_speed, mana_cost) = {
                let card = &self.state.cards[card_id];
                (
                    card.types.is_land(),
                    card.types.is_castable(),
                    card.types.is_instant_speed(),
                    card.mana_cost.clone(),
                )
            };
            if is_land {
                if can_play_land {
                    actions.push(Action::PlayLand {
                        player,
                        card: card_id,
                    });
                }
                continue;
            }

            if !is_castable {
                continue;
            }
            // CR 117.1a — Instants can be cast any time a player has priority;
            // sorcery-speed spells only during the active player's main phase with an empty stack.
            if !is_instant_speed && !can_cast_sorcery {
                continue;
            }

            match mana_cost.as_ref() {
                Some(cost) => {
                    if producible.is_none() {
                        producible = Some(self.cached_producible_mana(player));
                    }
                    if producible.as_ref().is_some_and(|m| m.can_pay(cost)) {
                        actions.push(Action::CastSpell {
                            player,
                            card: card_id,
                        });
                    }
                }
                None => actions.push(Action::CastSpell {
                    player,
                    card: card_id,
                }),
            }
        }

        actions.push(Action::PassPriority { player });
        actions
    }

    pub(crate) fn execute_action(
        &mut self,
        action: &Action,
    ) -> Result<(), crate::agent::action::AgentError> {
        match action {
            Action::PlayLand { player, card } => self.play_land(*player, *card),
            Action::CastSpell { player, card } => self.cast_spell_action(*player, *card),
            Action::PassPriority { .. } => {
                self.state.priority.pass_priority();
                Ok(())
            }
            Action::DeclareAttacker {
                permanent, attack, ..
            } => self.declare_attacker(*permanent, *attack),
            Action::DeclareBlocker {
                blocker, attacker, ..
            } => self.declare_blocker(*blocker, *attacker),
            Action::ChooseTarget { player, target } => self.choose_target(*player, *target),
        }
    }
}
