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

fn to_action_target(target: Target) -> ActionTarget {
    match target {
        Target::Player(p) => ActionTarget::Player(p),
        Target::Permanent(p) => ActionTarget::Permanent(p),
        Target::StackSpell(c) => ActionTarget::StackSpell(c),
    }
}

fn to_game_target(target: ActionTarget) -> Target {
    match target {
        ActionTarget::Player(p) => Target::Player(p),
        ActionTarget::Permanent(p) => Target::Permanent(p),
        ActionTarget::StackSpell(c) => Target::StackSpell(c),
    }
}

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
        let waterbend_candidates = self.waterbend_candidates(player).len();
        for permanent_id in self.battlefield_permanents(player) {
            let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
                continue;
            };
            let card = &self.state.cards[permanent.card];
            for (ability_index, ability) in card.activated_abilities.iter().enumerate() {
                let affordable = if ability.waterbend {
                    self.can_pay_with_waterbend(player, &ability.mana_cost, waterbend_candidates)
                } else {
                    if producible.is_none() {
                        *producible = Some(self.producible_mana(player));
                    }
                    producible
                        .as_ref()
                        .is_some_and(|mana| mana.can_pay(&ability.mana_cost))
                };
                if affordable {
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

        let can_cast_now = if card.is_instant_speed() {
            self.can_cast_instants(player)
        } else {
            self.can_cast_sorceries(player)
        };
        if !can_cast_now {
            return None;
        }

        // Every mandatory targeting requirement needs a legal target.
        for requirement in self.state.cards[card_id].target_requirements() {
            if requirement.min > 0
                && self
                    .legal_targets_for_requirement(player, card_id, &requirement)
                    .is_empty()
            {
                return None;
            }
        }

        match self.effective_spell_cost(player, card_id, false) {
            Some(cost) => {
                if producible.is_none() {
                    *producible = Some(self.producible_mana(player));
                }
                producible
                    .as_ref()
                    .is_some_and(|m| m.can_pay(&cost))
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
            PendingChoice::KickerChoice { player, card } => Some(ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::PayOrNot,
                actions: vec![
                    Action::PayCost { player: *player },
                    Action::Decline { player: *player },
                ],
                focus: vec![self.state.cards[card].id],
            }),
            PendingChoice::ChooseTargets {
                player,
                card,
                requirement_index,
                chosen_req_indices,
                legal_targets,
                ..
            } => {
                let mut actions: Vec<Action> = legal_targets
                    .iter()
                    .copied()
                    .map(|target| Action::ChooseTarget {
                        player: *player,
                        target: to_action_target(target),
                    })
                    .collect();
                let requirements = self.state.cards[card].target_requirements();
                let chosen_in_req = chosen_req_indices
                    .iter()
                    .filter(|i| **i == *requirement_index)
                    .count();
                let min_met = requirements
                    .get(*requirement_index)
                    .is_some_and(|req| chosen_in_req >= req.min && req.max > req.min);
                if min_met {
                    actions.push(Action::Decline { player: *player });
                }
                Some(ActionSpace {
                    player: Some(*player),
                    kind: ActionSpaceKind::ChooseTarget,
                    actions,
                    focus: vec![self.state.cards[card].id],
                })
            }
            PendingChoice::Waterbend {
                player,
                permanent,
                ability_index,
                remaining_generic,
            } => {
                let cost = self.state.permanents[*permanent]
                    .as_ref()
                    .and_then(|perm| {
                        self.state.cards[perm.card]
                            .activated_abilities
                            .get(*ability_index)
                    })
                    .map(|ability| ability.mana_cost.clone())?;
                let remainder = cost.with_generic(*remaining_generic);
                let candidates = self.waterbend_candidates(*player);

                let mut actions = Vec::new();
                for candidate in &candidates {
                    // Only offer taps that keep the rest of the cost payable.
                    let after_tap = remainder.reduced_generic(1);
                    if self.can_pay_with_waterbend(*player, &after_tap, candidates.len() - 1) {
                        actions.push(Action::WaterbendTap {
                            player: *player,
                            permanent: *candidate,
                        });
                    }
                }
                if self.available_mana(*player).can_pay(&remainder) {
                    actions.push(Action::PayCost { player: *player });
                }
                let focus = self.state.permanents[*permanent]
                    .as_ref()
                    .map(|perm| vec![perm.id])
                    .unwrap_or_default();
                Some(ActionSpace {
                    player: Some(*player),
                    kind: ActionSpaceKind::Waterbend,
                    actions,
                    focus,
                })
            }
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
                let target = to_game_target(*target);
                if matches!(
                    self.pending_choice,
                    Some(PendingChoice::ChooseTargets { .. })
                ) {
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
            Action::ScryCard { .. } | Action::SelectCard { .. } | Action::ChooseMode { .. } => {
                self.execute_decision_action(action)
            }
            Action::Decline { player } => {
                if self.state.suspended_decision.is_some() {
                    return self.execute_decision_action(action);
                }
                match self.pending_choice {
                    Some(PendingChoice::KickerChoice { .. }) => {
                        self.kicker_choice_action(*player, false)
                    }
                    Some(PendingChoice::ChooseTargets { .. }) => {
                        self.finish_targets_action(*player)
                    }
                    _ => Err(AgentError("nothing to decline".to_string())),
                }
            }
            Action::PayCost { player } => {
                if self.state.suspended_decision.is_some() {
                    return self.execute_decision_action(action);
                }
                match self.pending_choice {
                    Some(PendingChoice::KickerChoice { .. }) => {
                        self.kicker_choice_action(*player, true)
                    }
                    Some(PendingChoice::Waterbend { .. }) => {
                        self.waterbend_pay_remainder_action(*player)
                    }
                    _ => Err(AgentError("no cost payment is pending".to_string())),
                }
            }
            Action::WaterbendTap { player, permanent } => {
                self.waterbend_tap_action(*player, *permanent)
            }
        }
    }
}
