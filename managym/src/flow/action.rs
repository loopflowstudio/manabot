// action.rs
// Player action computation and action space construction.

use crate::{
    agent::action::{Action, ActionSpace, ActionSpaceKind, AgentError},
    flow::game::{Game, PendingChoice},
    state::{
        game_object::{CardId, PlayerId, Target},
        mana::Mana,
        target::Target as ActionTarget,
        zone::ZoneType,
    },
};

impl Game {
    pub(crate) fn can_player_act(&mut self, player: PlayerId) -> bool {
        let mut producible = None;
        self.state
            .zones
            .zone_cards(ZoneType::Hand, player)
            .to_vec()
            .into_iter()
            .any(|card| {
                self.priority_action_for_card(player, card, &mut producible)
                    .is_some()
            })
            || self
                .priority_activate_ability_actions(player, &mut producible)
                .into_iter()
                .next()
                .is_some()
    }

    pub(crate) fn compute_player_actions(&mut self, player: PlayerId) -> Vec<Action> {
        let mut actions = Vec::new();
        let mut producible = None;

        for card in self.state.zones.zone_cards(ZoneType::Hand, player).to_vec() {
            if let Some(action) = self.priority_action_for_card(player, card, &mut producible) {
                actions.push(action);
            }
        }

        actions.extend(self.priority_activate_ability_actions(player, &mut producible));
        actions.push(Action::PassPriority { player });
        actions
    }

    fn priority_activate_ability_actions(
        &mut self,
        player: PlayerId,
        producible: &mut Option<Mana>,
    ) -> Vec<Action> {
        let mut actions = Vec::new();
        for permanent_id in self.battlefield_permanents(player) {
            let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
                continue;
            };
            let card = &self.state.cards[permanent.card];
            for (ability_index, ability) in card.activated_abilities.iter().enumerate() {
                if producible.is_none() {
                    *producible = Some(self.producible_mana(player));
                }
                if producible
                    .as_ref()
                    .is_some_and(|mana| mana.can_pay(&ability.mana_cost))
                {
                    actions.push(Action::ActivateAbility {
                        player,
                        permanent: permanent_id,
                        ability_index,
                    });
                }
            }
        }
        actions
    }

    fn priority_action_for_card(
        &mut self,
        player: PlayerId,
        card_id: CardId,
        producible: &mut Option<Mana>,
    ) -> Option<Action> {
        let card = &self.state.cards[card_id];
        if card.types.is_land() {
            return self.can_play_land(player).then_some(Action::PlayLand {
                player,
                card: card_id,
            });
        }
        if !card.types.is_castable() {
            return None;
        }

        let can_cast_now = if card.types.is_instant_speed() {
            self.can_cast_instants(player)
        } else {
            self.can_cast_sorceries(player)
        };
        if !can_cast_now {
            return None;
        }

        if self
            .legal_targets_for_spell(card_id)
            .is_some_and(|targets| targets.is_empty())
        {
            return None;
        }

        let mana_cost = card.mana_cost.clone();
        match mana_cost.as_ref() {
            Some(cost) => {
                if producible.is_none() {
                    *producible = Some(self.producible_mana(player));
                }
                producible
                    .as_ref()
                    .is_some_and(|m| m.can_pay(cost))
                    .then_some(Action::CastSpell {
                        player,
                        card: card_id,
                    })
            }
            None => Some(Action::CastSpell {
                player,
                card: card_id,
            }),
        }
    }

    pub(crate) fn pending_choice_action_space(&self) -> Option<ActionSpace> {
        let choice = self.pending_choice.as_ref()?;
        match choice {
            PendingChoice::ChooseTarget {
                player,
                card,
                legal_targets,
            } => Some(ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::ChooseTarget,
                actions: legal_targets
                    .iter()
                    .copied()
                    .map(|target| Action::ChooseTarget {
                        player: *player,
                        target: match target {
                            Target::Player(p) => ActionTarget::Player(p),
                            Target::Permanent(p) => ActionTarget::Permanent(p),
                            Target::StackSpell(c) => ActionTarget::StackSpell(c),
                        },
                    })
                    .collect(),
                focus: vec![self.state.cards[card].id],
            }),
        }
    }

    pub(crate) fn execute_action(&mut self, action: &Action) -> Result<(), AgentError> {
        match action {
            Action::PlayLand { player, card } => {
                self.play_land(*player, *card)?;
                self.state.priority.on_non_pass_action(self.active_player());
                Ok(())
            }
            Action::CastSpell { player, card } => self.cast_spell_action(*player, *card),
            Action::ActivateAbility {
                player,
                permanent,
                ability_index,
            } => self.activate_ability_action(*player, *permanent, *ability_index),
            Action::ChooseTarget { player, target } => {
                let target = match *target {
                    ActionTarget::Player(p) => Target::Player(p),
                    ActionTarget::Permanent(p) => Target::Permanent(p),
                    ActionTarget::StackSpell(c) => Target::StackSpell(c),
                };
                if self.pending_choice.is_some() {
                    self.choose_target_action(*player, target)
                } else if self.state.pending_trigger_choice.is_some() {
                    self.choose_trigger_target(*player, target)
                } else {
                    Err(AgentError("no pending target choice".to_string()))
                }
            }
            Action::PassPriority { player } => {
                if self.state.priority.holder != *player {
                    return Err(AgentError("player does not have priority".to_string()));
                }
                let next = self.next_player(*player);
                self.state.priority.on_pass(next);
                Ok(())
            }
            Action::DeclareAttacker {
                permanent, attack, ..
            } => self.declare_attacker(*permanent, *attack),
            Action::DeclareBlocker {
                blocker, attacker, ..
            } => self.declare_blocker(*blocker, *attacker),
        }
    }
}
