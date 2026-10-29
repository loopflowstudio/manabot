// action.rs
// Player action computation and action space construction.

use crate::agent::ability_description::{mana_description, trigger_description};

use crate::{
    agent::action::{Action, ActionSpace, ActionSpaceKind, AgentError},
    flow::game::{ActivationSource, Game, PendingChoice},
    state::{
        game_object::{CardId, PlayerId},
        mana::Mana,
        zone::ZoneType,
    },
};

impl Game {
    pub(crate) fn action_description(&self, action: &Action) -> Option<String> {
        use crate::flow::decision::Decision;
        use crate::state::ability::Effect;
        match action {
            Action::ChooseTrigger { trigger_id, player } => {
                let trigger = self
                    .state
                    .trigger_placement
                    .as_ref()?
                    .remaining
                    .iter()
                    .find(|t| t.id == *trigger_id)?;
                let card = &self.state.cards[trigger.source_card];
                let mut label = format!(
                    "Put on stack next — {}",
                    trigger_description(
                        card,
                        trigger.ability_index,
                        trigger.inline_effects.as_deref()
                    )
                );
                if let Some(context) = trigger.context {
                    let reference = self.target_reference(context, *player);
                    if let Some(seat) = reference.player {
                        label.push_str(&format!(" — event: Seat {}", seat + 1));
                    } else if let Some(definition) = reference.definition.and_then(|id| {
                        self.state
                            .content
                            .definition(crate::state::card::CardDefId(id))
                    }) {
                        label.push_str(&format!(" — event: {}", definition.name));
                        if let Some(object) = reference.object {
                            label.push_str(&format!(" (object {object})"));
                        }
                    }
                }
                Some(label)
            }
            Action::ChooseMode { mode, .. } => {
                match &self.state.suspended_decision.as_ref()?.decision {
                    Decision::ManaChoice { choices, .. } => {
                        Some(format!("Add {}", mana_description(choices.get(*mode)?)))
                    }
                    Decision::Modal { modes, .. } => match modes.get(*mode)?.as_slice() {
                        [] => Some("Decline".into()),
                        [Effect::AddMana { mana, .. }] => {
                            Some(format!("Add {}", mana_description(mana)))
                        }
                        [Effect::CreateToken { token_name, .. }] => {
                            Some(format!("Create a {token_name} token"))
                        }
                        [Effect::PlaceTargetInLibrary { second_from_top }] => {
                            Some(if *second_from_top {
                                "Second from the top".into()
                            } else {
                                "On the bottom".into()
                            })
                        }
                        [Effect::SearchBasicLand { .. }] => {
                            Some("Search your library for a basic land".into())
                        }
                        _ => None,
                    },
                    _ => None,
                }
            }
            Action::PayCost { .. } => match &self.pending_choice {
                Some(PendingChoice::SacrificePayment { cost, .. }) => {
                    cost.or_mana.as_ref().map(|alternative| {
                        let mut label = String::new();
                        if alternative.generic() > 0 {
                            label.push_str(&format!("{{{}}}", alternative.generic()));
                        }
                        let colored = Mana {
                            mana: alternative.cost[..6]
                                .try_into()
                                .expect("six colored mana slots"),
                            ..Default::default()
                        };
                        label.push_str(&mana_description(&colored));
                        format!("Pay {label} instead of sacrificing")
                    })
                }
                _ => None,
            },
            Action::SelectCard { card, .. } => {
                let sacrifice = matches!(
                    self.pending_choice,
                    Some(PendingChoice::SacrificePayment { .. })
                ) || self
                    .state
                    .suspended_decision
                    .as_ref()
                    .is_some_and(|s| matches!(s.decision, Decision::Sacrifice { .. }));
                if sacrifice {
                    Some(format!("Sacrifice {}", self.state.cards[*card].name))
                } else if let Some(suspended) = &self.state.suspended_decision {
                    match &suspended.decision {
                        Decision::SearchBasicLand { to_hand: true, .. } => Some(format!(
                            "Reveal {} and put it into your hand",
                            self.state.cards[*card].name
                        )),
                        Decision::SearchBasicLand { tapped, .. } => Some(format!(
                            "Put {} onto the battlefield{}",
                            self.state.cards[*card].name,
                            if *tapped { " tapped" } else { "" }
                        )),
                        _ => None,
                    }
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    pub(crate) fn can_player_act(&self, player: PlayerId) -> bool {
        self.state
            .zones
            .zone_cards(ZoneType::Hand, player)
            .iter()
            .chain(
                self.state
                    .zones
                    .zone_cards(ZoneType::Graveyard, player)
                    .iter(),
            )
            .copied()
            .chain(self.exile_granted_cards(player))
            .any(|card| self.priority_action_for_card(player, card).is_some())
            || self
                .priority_activate_ability_actions(player)
                .into_iter()
                .next()
                .is_some()
    }

    pub(crate) fn compute_player_actions(&self, player: PlayerId) -> Vec<Action> {
        let mut actions = Vec::new();

        for card in self
            .state
            .zones
            .zone_cards(ZoneType::Hand, player)
            .iter()
            .chain(
                self.state
                    .zones
                    .zone_cards(ZoneType::Graveyard, player)
                    .iter(),
            )
            .copied()
            .chain(self.exile_granted_cards(player))
        {
            if let Some(action) = self.priority_action_for_card(player, card) {
                actions.push(action);
            }
        }

        actions.extend(self.priority_activate_ability_actions(player));
        actions.push(Action::PassPriority { player });
        actions
    }

    fn priority_activate_ability_actions(&self, player: PlayerId) -> Vec<Action> {
        let mut actions = Vec::new();
        for permanent_id in self.battlefield_permanents(player) {
            let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
                continue;
            };
            let card = &self.state.cards[permanent.card];
            for (ability_index, ability) in self
                .activated_abilities_for(permanent_id)
                .iter()
                .enumerate()
            {
                if ability.zone != crate::state::card::ActivationZone::Battlefield
                    || (ability.requires_target()
                        && self
                            .activated_ability_targets(
                                player,
                                ActivationSource::Permanent(permanent_id),
                                ability_index,
                            )
                            .is_empty())
                {
                    continue;
                }
                if (ability.own_turn_only && self.active_player() != player)
                    || permanent.activation_used(ability_index, ability)
                    || self.state.players[player.0].life < ability.life_cost as i32
                    || (ability.discard_card
                        && self
                            .state
                            .zones
                            .zone_cards(ZoneType::Hand, player)
                            .is_empty())
                {
                    continue;
                }
                if ability.mana_choices().is_some() {
                    if self.can_activate_mana_ability(player, permanent_id, ability_index) {
                        actions.push(Action::ActivateAbility {
                            player,
                            permanent: permanent_id,
                            ability_index,
                        });
                    }
                    continue;
                }
                if let Some(cost) = &ability.sacrifice {
                    let payment = crate::flow::game::Payment::Activation {
                        source: ActivationSource::Permanent(permanent_id),
                        ability_index,
                        targets: Vec::new(),
                    };
                    if !self
                        .payment_candidates(player, &payment, cost)
                        .into_iter()
                        .any(|id| {
                            self.can_pay_announced_activation(
                                player,
                                &ability.mana_cost,
                                &payment,
                                Some(id),
                            )
                        })
                        && !cost.or_mana.as_ref().is_some_and(|extra| {
                            self.can_pay_announced_activation(
                                player,
                                &ability.mana_cost.plus(extra),
                                &payment,
                                None,
                            )
                        })
                    {
                        continue;
                    }
                }
                if ability.tap_source && !permanent.can_tap(card) {
                    continue;
                }
                let affordable = if ability.waterbend {
                    self.can_pay_waterbend_activation(player, permanent_id, &ability)
                } else {
                    self.can_pay_announced_activation(
                        player,
                        &ability.mana_cost,
                        &crate::flow::game::Payment::Activation {
                            source: ActivationSource::Permanent(permanent_id),
                            ability_index,
                            targets: Vec::new(),
                        },
                        None,
                    )
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
        actions.extend(self.priority_card_ability_actions(player));
        actions
    }

    /// Abilities that function from `player`'s hand or graveyard. Their
    /// costs are mana, life and discarding the card itself.
    fn priority_card_ability_actions(&self, player: PlayerId) -> Vec<Action> {
        let mut actions = Vec::new();
        for zone in [ZoneType::Hand, ZoneType::Graveyard] {
            for &card in self.state.zones.zone_cards(zone, player) {
                for ability_index in 0..self.state.cards[card].activated_abilities.len() {
                    let Some(ability) = self.card_ability_in_zone(player, card, ability_index)
                    else {
                        continue;
                    };
                    let source = ActivationSource::Card(card);
                    if (ability.own_turn_only && self.active_player() != player)
                        || (ability.requires_target()
                            && self
                                .activated_ability_targets(player, source, ability_index)
                                .is_empty())
                    {
                        continue;
                    }
                    let payment = crate::flow::game::Payment::Activation {
                        source,
                        ability_index,
                        targets: Vec::new(),
                    };
                    if self.can_pay_announced_activation(player, &ability.mana_cost, &payment, None)
                    {
                        actions.push(Action::ActivateCardAbility {
                            player,
                            card,
                            ability_index,
                        });
                    }
                }
            }
        }
        actions
    }

    fn priority_action_for_card(&self, player: PlayerId, card_id: CardId) -> Option<Action> {
        let card = &self.state.cards[card_id];
        if !self.may_play_from_zone(player, card_id) {
            return None;
        }
        if card.types.is_land() {
            return self.can_play_land(player).then_some(Action::PlayLand {
                player,
                card: card_id,
            });
        }
        if !card.types.is_castable() {
            return None;
        }

        let can_cast_now = if self.has_instant_timing(player, card_id) {
            self.can_cast_instants(player)
        } else {
            self.can_cast_sorceries(player)
        };
        if !can_cast_now {
            return None;
        }

        if card.spell_modes().is_some() && self.legal_spell_modes(player, card_id).is_empty() {
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
                // Castability gating sees pooled mana too — until-end-of-
                // combat mana (firebending) must make spells castable.
                let payable = if let Some(additional) = &self.state.cards[card_id].sacrifice_cost {
                    let payment = crate::flow::game::Payment::Spell {
                        card: card_id,
                        kicked: false,
                        mode: None,
                        targets: Vec::new(),
                        target_req_indices: Vec::new(),
                    };
                    self.payment_candidates(player, &payment, additional)
                        .into_iter()
                        .any(|id| self.can_pay_with_mana_abilities(player, &cost, Some(id)))
                        || additional.or_mana.as_ref().is_some_and(|extra| {
                            self.can_pay_with_mana_abilities(player, &cost.plus(extra), None)
                        })
                } else {
                    self.can_pay_with_mana_abilities(player, &cost, None)
                };
                payable.then_some(Action::CastSpell {
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
            PendingChoice::ManaPayment {
                player,
                payment,
                sacrifice,
                alternative,
            } => {
                let mut actions = self.mana_ability_actions(*player, *sacrifice);
                let cost = self.payment_mana_cost(*player, payment, alternative.as_ref());
                if self.can_pay_mana_cost(*player, &cost) {
                    actions.push(Action::PayCost { player: *player });
                }
                Some(ActionSpace {
                    player: Some(*player),
                    kind: ActionSpaceKind::PayOrNot,
                    actions,
                    focus: Vec::new(),
                })
            }
            PendingChoice::SacrificePayment {
                player,
                payment,
                cost,
            } => {
                let mut actions: Vec<_> = self
                    .payment_candidates(*player, payment, cost)
                    .into_iter()
                    .filter(|id| {
                        self.can_pay_with_mana_abilities(
                            *player,
                            &self.payment_mana_cost(*player, payment, None),
                            Some(*id),
                        )
                    })
                    .map(|id| Action::SelectCard {
                        player: *player,
                        card: self.state.permanents[id].as_ref().unwrap().card,
                    })
                    .collect();
                if cost.or_mana.as_ref().is_some_and(|extra| {
                    self.can_pay_with_mana_abilities(
                        *player,
                        &self.payment_mana_cost(*player, payment, Some(extra)),
                        None,
                    )
                }) {
                    actions.push(Action::PayCost { player: *player });
                }
                Some(ActionSpace {
                    player: Some(*player),
                    kind: ActionSpaceKind::LookAndSelect,
                    actions,
                    focus: Vec::new(),
                })
            }
            PendingChoice::DiscardPayment { player, .. } => Some(ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::Discard,
                actions: self
                    .state
                    .zones
                    .zone_cards(ZoneType::Hand, *player)
                    .iter()
                    .map(|card| Action::SelectCard {
                        player: *player,
                        card: *card,
                    })
                    .collect(),
                focus: Vec::new(),
            }),
            PendingChoice::ActivatedTarget {
                player,
                source,
                ability_index,
            } => {
                let mut actions: Vec<_> = self
                    .activated_ability_targets(*player, *source, *ability_index)
                    .into_iter()
                    .map(|target| Action::ChooseTarget {
                        player: *player,
                        target,
                    })
                    .collect();
                let ability = self.activation_ability(*source, *ability_index)?;
                if ability.target_is_optional() {
                    actions.push(Action::Decline { player: *player });
                }
                Some(ActionSpace {
                    player: Some(*player),
                    kind: ActionSpaceKind::ChooseTarget,
                    actions,
                    focus: Vec::new(),
                })
            }
            PendingChoice::SpellMode { player, card, .. } => Some(ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::Modal,
                actions: self
                    .legal_spell_modes(*player, *card)
                    .into_iter()
                    .map(|mode| Action::ChooseMode {
                        player: *player,
                        mode,
                    })
                    .collect(),
                focus: vec![self.state.cards[card].id],
            }),
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
                mode,
                chosen_req_indices,
                legal_targets,
                ..
            } => {
                let mut actions: Vec<Action> = legal_targets
                    .iter()
                    .copied()
                    .map(|target| Action::ChooseTarget {
                        player: *player,
                        target,
                    })
                    .collect();
                let requirements = self.state.cards[card].target_requirements_for_mode(*mode);
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
                ..
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
                let candidates =
                    self.waterbend_activation_candidates(*player, *permanent, *ability_index);
                let taps_source = self
                    .activated_abilities_for(*permanent)
                    .get(*ability_index)
                    .is_some_and(|ability| ability.tap_source);

                let mut actions = self.mana_ability_actions(*player, None);
                for candidate in &candidates {
                    // Only offer taps that keep the rest of the cost payable.
                    let after_tap = remainder.reduced_generic(1);
                    let mut probe = self.clone();
                    probe.tap_permanent(*candidate, false);
                    if taps_source {
                        probe.tap_permanent(*permanent, false);
                    }
                    if probe.can_pay_with_waterbend(*player, &after_tap) {
                        actions.push(Action::WaterbendTap {
                            player: *player,
                            permanent: *candidate,
                        });
                    }
                }
                if self.can_pay_mana_cost(*player, &remainder) {
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
        self.journal_priority();
        match action {
            Action::ChooseTrigger { player, trigger_id } => {
                self.choose_trigger(*player, *trigger_id)
            }
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
            } => {
                if self.state.suspended_decision.is_some() {
                    self.activate_mana_ability(*player, *permanent, *ability_index)
                } else {
                    self.activate_ability_action(*player, *permanent, *ability_index)
                }
            }
            Action::ActivateCardAbility {
                player,
                card,
                ability_index,
            } => self.activate_card_ability_action(*player, *card, *ability_index),
            Action::ChooseTarget { player, target } => {
                let target = *target;
                if matches!(
                    self.pending_choice,
                    Some(PendingChoice::ChooseTargets { .. })
                ) {
                    self.choose_target_action(*player, target)
                } else if matches!(
                    self.pending_choice,
                    Some(PendingChoice::ActivatedTarget { .. })
                ) {
                    self.choose_activated_target(*player, target)
                } else if self
                    .state
                    .trigger_placement
                    .as_ref()
                    .is_some_and(|p| p.placing.is_some())
                {
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
            Action::ChooseMode { player, mode }
                if matches!(self.pending_choice, Some(PendingChoice::SpellMode { .. })) =>
            {
                self.spell_mode_action(*player, *mode)
            }
            Action::SelectCard { player, card }
                if matches!(
                    self.pending_choice,
                    Some(PendingChoice::SacrificePayment { .. })
                ) =>
            {
                self.sacrifice_payment_action(*player, Some(*card))
            }
            Action::SelectCard { player, card }
                if matches!(
                    self.pending_choice,
                    Some(PendingChoice::DiscardPayment { .. })
                ) =>
            {
                self.discard_payment_action(*player, *card)
            }
            Action::ScryCard { .. }
            | Action::SelectCard { .. }
            | Action::ChooseMode { .. }
            | Action::LearnDiscard { .. }
            | Action::LearnTakeLesson { .. } => self.execute_decision_action(action),
            Action::Decline { player } => {
                if self.state.suspended_decision.is_some() {
                    return self.execute_decision_action(action);
                }
                match self.pending_choice {
                    Some(PendingChoice::ActivatedTarget { .. }) => {
                        self.decline_activated_target(*player)
                    }
                    Some(PendingChoice::KickerChoice { .. }) => {
                        self.kicker_choice_action(*player, false)
                    }
                    Some(PendingChoice::ChooseTargets { .. }) => {
                        self.finish_targets_action(*player)
                    }
                    _ if self
                        .state
                        .trigger_placement
                        .as_ref()
                        .is_some_and(|p| p.placing.is_some()) =>
                    {
                        // "Up to one target" triggered ability: decline
                        // choosing a target.
                        self.decline_trigger_target(*player)
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
                    Some(PendingChoice::ManaPayment { .. }) => self.finish_mana_payment(*player),
                    Some(PendingChoice::SacrificePayment { .. }) => {
                        self.sacrifice_payment_action(*player, None)
                    }
                    Some(PendingChoice::Waterbend { .. }) => {
                        self.waterbend_pay_remainder_action(*player)
                    }
                    _ => Err(AgentError("no cost payment is pending".to_string())),
                }
            }
            Action::WaterbendTap { player, permanent } => {
                if self.state.suspended_decision.is_some() {
                    return self.execute_decision_action(action);
                }
                self.waterbend_tap_action(*player, *permanent)
            }
        }
    }
}
