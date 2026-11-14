// action.rs
// Player action computation and action space construction.

use crate::agent::ability_description::{
    counter_description, mana_description, trigger_description,
};
use crate::state::ability::LibraryEnd;

use crate::cards::names::NameOption;
use crate::{
    agent::action::{Action, ActionSpace, ActionSpaceKind, AgentError},
    cards::content_pack::default_content_pack,
    flow::game::{ActivationSource, CastChoices, Game, Payment, PendingChoice},
    state::{
        ability::Effect,
        card::ActivationZone,
        game_object::{CardId, ObjectId, PlayerId, PublicId},
        mana::Mana,
        zone::ZoneType,
    },
};

impl Game {
    pub(crate) fn action_description(&self, action: &Action) -> Option<String> {
        use crate::flow::decision::Decision;
        match action {
            Action::ChooseTrigger { trigger_id, player } => {
                let trigger = self
                    .state
                    .trigger_placement
                    .as_ref()?
                    .remaining
                    .iter()
                    .find(|t| t.id == *trigger_id)?;
                let record = &self.state.objects[trigger.source_object];
                let mut label = format!(
                    "Put on stack next — {}",
                    trigger_description(
                        record,
                        trigger.ability_index,
                        trigger.inline_effects.as_deref()
                    )
                );
                if let Some(context) = trigger.context {
                    let reference = self.target_reference(context, *player);
                    if let Some(seat) = reference.player {
                        label.push_str(&format!(" — event: Seat {}", seat + 1));
                    } else if let Some(definition) = reference.definition.and_then(|id| {
                        default_content_pack().definition(crate::state::card::CardDefId(id))
                    }) {
                        label.push_str(&format!(" — event: {}", definition.name));
                        if let Some(object) = reference.object {
                            label.push_str(&format!(" (object {object})"));
                        }
                    }
                }
                Some(label)
            }
            Action::Foretell {
                object: object_id, ..
            } => Some(format!("Foretell {}", self.state.objects[*object_id].name)),
            Action::ChooseMode { mode, .. }
                if matches!(
                    self.pending_choice,
                    Some(PendingChoice::AnnounceX { .. } | PendingChoice::ActivationX { .. })
                ) =>
            {
                Some(format!("X = {mode}"))
            }
            Action::ChooseMode { mode, .. } if self.state.suspended_decision.is_none() => {
                let trigger = self.state.trigger_placement.as_ref()?.placing.as_ref()?;
                let modes = self.unchosen_trigger_modes(trigger)?;
                Some(crate::agent::ability_description::effects_description(
                    modes.get(*mode)?,
                ))
            }
            Action::ChooseMode { mode, .. } => {
                match &self.state.suspended_decision.as_ref()?.decision {
                    Decision::OrderReplacement { candidates, .. } => {
                        let candidate = candidates.get(*mode)?;
                        let object_id = candidate.source().entity;
                        Some(format!(
                            "{}: {}",
                            self.state.objects[object_id].name,
                            candidate.description()
                        ))
                    }
                    Decision::NameCard { names, scope, .. } => match scope.option(*mode)? {
                        NameOption::Group(group) => Some(format!(
                            "Names {} through {}",
                            names.get(group.start)?.name,
                            names.get(group.end - 1)?.name
                        )),
                        NameOption::Name(row) => Some(format!("Name {}", names.get(row)?.name)),
                    },
                    Decision::ChoosePaymentX { .. } => Some(format!("X = {mode}")),
                    Decision::ManaSpend { player, cost, .. } => {
                        let choices = self.mana_spend_choices(*player, cost);
                        let unit = &choices.get(*mode)?.unit;
                        let expiry = if unit.classes().iter().any(|(class, _)| class.combat) {
                            ""
                        } else {
                            " (expires at step end)"
                        };
                        Some(format!("Spend {}{expiry}", mana_description(unit)))
                    }
                    Decision::ManaChoice { choices, .. } => {
                        Some(format!("Add {}", mana_description(choices.get(*mode)?)))
                    }
                    Decision::Modal { modes, .. } => match modes.get(*mode)?.as_slice() {
                        [] => Some("Decline".into()),
                        [Effect::AddMana { mana, .. }] => {
                            Some(format!("Add {}", mana_description(mana)))
                        }
                        [Effect::CreateToken { token, .. }] => Some(format!(
                            "Create a {} token",
                            default_content_pack().token_name(token)
                        )),
                        [Effect::PlaceTargetInLibrary { second_from_top }] => {
                            Some(if *second_from_top {
                                "Second from the top".into()
                            } else {
                                "On the bottom".into()
                            })
                        }
                        [Effect::SelectCards {
                            from: crate::state::ability::ObjectSelector::Zone(ZoneType::Library),
                            ..
                        }] => Some("Search your library".into()),
                        [Effect::RemoveCounters {
                            chosen: Some(kind), ..
                        }] => Some(format!("Remove a {} counter", counter_description(*kind))),
                        _ => None,
                    },
                    _ => None,
                }
            }
            Action::PayCost { .. } => match &self.pending_choice {
                Some(PendingChoice::KickerChoice {
                    object: object_id, ..
                }) if self.state.objects[*object_id].exile_alternative().is_some() => {
                    Some("Exile a card from your hand instead of paying the mana cost".into())
                }
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
            Action::SelectCard {
                object: object_id, ..
            } => {
                if matches!(
                    self.pending_choice,
                    Some(PendingChoice::ExileFromHandPayment { .. })
                ) {
                    Some(format!(
                        "Exile {} to pay",
                        self.state.objects[*object_id].name
                    ))
                } else if matches!(self.pending_choice, Some(PendingChoice::Crew { .. })) {
                    Some(format!(
                        "Tap {} to crew",
                        self.state.objects[*object_id].name
                    ))
                } else if matches!(self.pending_choice, Some(PendingChoice::TapPayment { .. })) {
                    Some(format!(
                        "Tap {} to pay",
                        self.state.objects[*object_id].name
                    ))
                } else if matches!(
                    self.pending_choice,
                    Some(PendingChoice::SacrificePayment { .. })
                ) {
                    Some(format!("Sacrifice {}", self.state.objects[*object_id].name))
                } else if let Some(suspended) = &self.state.suspended_decision {
                    match &suspended.decision {
                        Decision::ManaUpfront { tap_remaining, .. } if *tap_remaining > 0 => Some(
                            format!("Tap {} to pay", self.state.objects[*object_id].name),
                        ),
                        Decision::ChooseObject { effect, .. } => Some(format!(
                            "{} {}",
                            choice_verb(effect),
                            self.state.objects[*object_id].name
                        )),

                        Decision::SelectCards { .. } => {
                            Some(format!("Select {}", self.state.objects[*object_id].name))
                        }
                        Decision::OrderLibrary { end, .. } => Some(format!(
                            "Put {} next on the {}",
                            self.state.objects[*object_id].name,
                            match end {
                                LibraryEnd::Top => "top",
                                LibraryEnd::Bottom => "bottom",
                            }
                        )),
                        Decision::Surveil { .. } => Some(format!(
                            "Put {} into your graveyard",
                            self.state.objects[*object_id].name
                        )),
                        Decision::SearchNamed { .. } => Some(format!(
                            "Exile {} from {}",
                            self.state.objects[*object_id].name,
                            self.state
                                .zones
                                .zone_of(*object_id)
                                .map_or("unknown zone".to_string(), |zone| format!("{zone:?}"))
                        )),
                        Decision::DistributeCounters { kind, .. } => Some(format!(
                            "Put a {} counter on {}",
                            counter_description(*kind),
                            self.state.objects[*object_id].name
                        )),
                        _ => None,
                    }
                } else {
                    None
                }
            }
            Action::Decline { .. }
                if matches!(
                    &self.pending_choice,
                    Some(PendingChoice::KickerChoice { object: object_id, .. })
                        if self.state.objects[*object_id].exile_alternative().is_some()
                ) =>
            {
                Some("Pay the mana cost".into())
            }
            Action::Decline { .. } => match &self.state.suspended_decision.as_ref()?.decision {
                Decision::SearchNamed { .. } => Some("Stop searching and shuffle".into()),
                Decision::Surveil { .. } => Some("Keep on top".into()),
                _ => None,
            },
            _ => None,
        }
    }

    /// Cards `player` might play at priority: their hand and graveyard, and
    /// exiled cards they hold grants for, and a library top a permanent lets
    /// them cast. `priority_action_for_card` decides.
    fn priority_candidate_cards(&self, player: PlayerId) -> impl Iterator<Item = CardId> + '_ {
        self.state
            .zone_cards(ZoneType::Hand, player)
            .chain(self.state.zone_cards(ZoneType::Graveyard, player))
            .chain(self.exile_granted_cards(player))
            .chain(self.library_top_castable(player))
    }

    pub(crate) fn compute_player_actions(&self, player: PlayerId) -> Vec<Action> {
        let _scan = self.board_scan();
        let _payments = self.payment_scan();
        let mut actions: Vec<Action> = self
            .priority_candidate_cards(player)
            .filter_map(|card| self.priority_action_for_card(player, card))
            .collect();
        actions.extend(self.priority_activate_ability_actions(player));
        actions.push(Action::PassPriority { player });
        actions
    }

    fn priority_activate_ability_actions(&self, player: PlayerId) -> Vec<Action> {
        let mut actions = Vec::new();
        for permanent_id in self.iter_battlefield_permanents(player) {
            let Some(permanent) = self.state.permanents[permanent_id].as_ref() else {
                continue;
            };
            if self.activations_prohibited(permanent_id) {
                continue;
            }
            for (ability_index, ability) in self
                .characteristics(permanent_id)
                .activated_abilities
                .iter()
                .enumerate()
            {
                let source = ActivationSource::Permanent(permanent_id);
                if ability.zone != ActivationZone::Battlefield
                    || !self.activation_targets_available(player, source, ability_index)
                {
                    continue;
                }
                if self.check_activation_terms(player, ability).is_err()
                    || permanent.activation_used(ability_index, ability)
                    || (ability.discard_card
                        && self
                            .state
                            .zone_cards(ZoneType::Hand, player)
                            .next()
                            .is_none())
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
                let payment = Payment::Activation {
                    source,
                    ability_index,
                    targets: Vec::new(),
                    x: 0,
                };
                if let Some(cost) = &ability.sacrifice {
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
                if ability.tap_source && !self.permanent_can_tap(permanent_id) {
                    continue;
                }
                if ability
                    .crew
                    .is_some_and(|power| self.crew_power_available(player, permanent_id) < power)
                {
                    continue;
                }
                let affordable = if let Some(cost) = &ability.tap_cost {
                    self.tap_activation_payment_possible(player, &payment, cost.count, None)
                } else if ability.waterbend {
                    self.can_pay_waterbend_activation(player, permanent_id, ability)
                } else {
                    self.can_pay_announced_activation(player, &ability.mana_cost, &payment, None)
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
        actions.extend(
            self.state
                .zone_cards(ZoneType::Hand, player)
                .filter(|card| self.can_foretell(player, *card))
                .map(|card| Action::Foretell {
                    player,
                    object: card.into(),
                }),
        );
        actions
    }

    /// Abilities that function from `player`'s hand or graveyard. Their
    /// costs are mana, life and discarding the card itself.
    fn priority_card_ability_actions(&self, player: PlayerId) -> Vec<Action> {
        let mut actions = Vec::new();
        for zone in [ZoneType::Hand, ZoneType::Graveyard] {
            for card in self.state.zone_cards(zone, player) {
                for ability_index in 0..self.state.objects[card].activated_abilities.len() {
                    let Some(ability) = self.card_ability_in_zone(player, card, ability_index)
                    else {
                        continue;
                    };
                    let source = ActivationSource::Card(card);
                    if self.check_activation_terms(player, &ability).is_err()
                        || !self.activation_targets_available(player, source, ability_index)
                    {
                        continue;
                    }
                    let payment = Payment::Activation {
                        source,
                        ability_index,
                        targets: Vec::new(),
                        x: 0,
                    };
                    if self.can_pay_announced_activation(player, &ability.mana_cost, &payment, None)
                    {
                        actions.push(Action::ActivateCardAbility {
                            player,
                            object: card.into(),
                            ability_index,
                        });
                    }
                }
            }
        }
        actions
    }

    fn priority_action_for_card(&self, player: PlayerId, card: CardId) -> Option<Action> {
        let object = &self.state.objects[card];
        if !self.may_play_from_zone(player, card) {
            return None;
        }
        if object.types.is_land() {
            return self.can_play_land(player).then_some(Action::PlayLand {
                player,
                object: card.into(),
            });
        }
        if !object.types.is_castable() {
            return None;
        }

        let can_cast_now = if self.has_instant_timing(player, card) {
            self.can_cast_instants(player)
        } else {
            self.can_cast_sorceries(player)
        };
        if !can_cast_now {
            return None;
        }
        self.can_cast_with(player, card, Default::default())
            .then_some(Action::CastSpell {
                player,
                object: card.into(),
            })
    }

    /// Whether `card_id`'s modes and mandatory targets admit casting it and
    /// `player` can pay its total cost with `cast`'s choices. Timing and zone
    /// permission are the caller's.
    pub(crate) fn can_cast_with(&self, player: PlayerId, card: CardId, cast: CastChoices) -> bool {
        let object = &self.state.objects[card];
        if object.spell_modes().is_some() && self.legal_spell_modes(player, card).is_empty() {
            return false;
        }
        // Every mandatory targeting requirement needs a legal target.
        for requirement in object.target_requirements() {
            if requirement.min > 0
                && self
                    .legal_targets_for_requirement(player, card, &requirement)
                    .is_empty()
            {
                return false;
            }
        }

        let Some(cost) = self.effective_spell_cost(player, card, cast, None) else {
            return true;
        };
        // Castability gating sees pooled mana too — until-end-of-combat mana
        // (firebending) must make spells castable.
        let Some(additional) = object.sacrifice_cost() else {
            return self.can_pay_cast(player, card, cast)
                || (object.exile_alternative().is_some()
                    && self.can_pay_cast(
                        player,
                        card,
                        CastChoices {
                            kicked: true,
                            ..cast
                        },
                    ));
        };
        let payment = Payment::Spell {
            object: card.into(),
            cast,
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
    }

    pub(crate) fn pending_choice_action_space(&self) -> Option<ActionSpace> {
        let _payments = self.payment_scan();
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
                        object: self.state.permanents[id].as_ref().unwrap().object,
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
            PendingChoice::TapPayment {
                player,
                source,
                ability_index,
                targets,
                remaining,
            } => Some(ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::LookAndSelect,
                actions: self
                    .activation_ability(*source, *ability_index)
                    .map(|ability| self.tap_cost_candidates(*player, &ability))
                    .unwrap_or_default()
                    .into_iter()
                    .filter(|id| {
                        self.tap_activation_payment_possible(
                            *player,
                            &Payment::Activation {
                                source: *source,
                                ability_index: *ability_index,
                                targets: targets.clone(),
                                x: 0,
                            },
                            *remaining,
                            Some(*id),
                        )
                    })
                    .filter_map(|id| self.state.permanents[id].as_ref())
                    .map(|permanent| Action::SelectCard {
                        player: *player,
                        object: permanent.object,
                    })
                    .collect(),
                focus: Vec::new(),
            }),
            PendingChoice::ExileFromHandPayment { player, payment } => Some(ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::LookAndSelect,
                actions: self
                    .payment_source_object(payment)
                    .map(|object| self.exile_alternative_cards(*player, object))
                    .unwrap_or_default()
                    .into_iter()
                    .map(|card| Action::SelectCard {
                        player: *player,
                        object: card.into(),
                    })
                    .collect(),
                focus: Vec::new(),
            }),
            PendingChoice::DiscardPayment { player, .. } => Some(ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::Discard,
                actions: self
                    .state
                    .zone_cards(ZoneType::Hand, *player)
                    .map(|card| Action::SelectCard {
                        player: *player,
                        object: card.into(),
                    })
                    .collect(),
                focus: Vec::new(),
            }),
            PendingChoice::Crew {
                player, permanent, ..
            } => Some(ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::LookAndSelect,
                actions: self
                    .crew_candidates(*player, *permanent)
                    .into_iter()
                    .filter_map(|id| self.state.permanents[id].as_ref())
                    .map(|crew| Action::SelectCard {
                        player: *player,
                        object: crew.object,
                    })
                    .collect(),
                focus: Vec::new(),
            }),
            PendingChoice::ActivatedTarget {
                player,
                source,
                ability_index,
                chosen,
                requirement,
            } => {
                let chooser =
                    self.activation_target_chooser(*player, *source, *ability_index, *requirement);
                let mut actions: Vec<_> = self
                    .activated_ability_targets(
                        *player,
                        *source,
                        *ability_index,
                        *requirement,
                        chosen,
                    )
                    .into_iter()
                    .map(|target| Action::ChooseTarget {
                        player: chooser,
                        target,
                    })
                    .collect();
                let clause =
                    self.activation_target_requirement(*source, *ability_index, *requirement)?;
                let in_clause = self
                    .activation_clause_chosen(*source, *ability_index, *requirement, chosen)
                    .len();
                if in_clause >= clause.min {
                    actions.push(Action::Decline { player: chooser });
                }
                Some(ActionSpace {
                    player: Some(chooser),
                    kind: ActionSpaceKind::ChooseTarget,
                    actions,
                    focus: Vec::new(),
                })
            }
            PendingChoice::SpellMode { player, object, .. } => Some(ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::Modal,
                actions: self
                    .legal_spell_modes(*player, *object)
                    .into_iter()
                    .map(|mode| Action::ChooseMode {
                        player: *player,
                        mode,
                    })
                    .collect(),
                focus: vec![self.state.objects[object].id],
            }),
            PendingChoice::AnnounceX {
                player,
                object,
                cast,
            } => Some(x_action_space(
                *player,
                self.legal_x_values(*player, *object, *cast),
                vec![self.state.objects[object].id],
            )),
            PendingChoice::ActivationX {
                player,
                permanent,
                ability_index,
            } => Some(x_action_space(
                *player,
                self.legal_activation_x_values(*player, *permanent, *ability_index),
                self.state.permanents[*permanent]
                    .as_ref()
                    .map(|perm| vec![perm.id])
                    .unwrap_or_default(),
            )),
            PendingChoice::KickerChoice { player, object, .. } => Some(ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::PayOrNot,
                actions: vec![
                    Action::PayCost { player: *player },
                    Action::Decline { player: *player },
                ],
                focus: vec![self.state.objects[object].id],
            }),
            PendingChoice::ChooseTargets {
                player,
                object,
                cast,
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
                let requirements = self.state.objects[object].target_requirements_for_mode(*mode);
                let chosen_in_req = chosen_req_indices
                    .iter()
                    .filter(|i| **i == *requirement_index)
                    .count();
                let min_met = requirements.get(*requirement_index).is_some_and(|req| {
                    chosen_in_req >= req.min_for(cast.x)
                        && req.max_for(cast.x) > req.min_for(cast.x)
                });
                if min_met {
                    actions.push(Action::Decline { player: *player });
                }
                Some(ActionSpace {
                    player: Some(*player),
                    kind: ActionSpaceKind::ChooseTarget,
                    actions,
                    focus: vec![self.state.objects[object].id],
                })
            }
            PendingChoice::Waterbend {
                player,
                payment,
                remaining_generic,
                tapped,
            } => {
                let (owed, waterbend) =
                    self.waterbend_owed(*player, payment, *remaining_generic, *tapped);
                let tapped_source = self.payment_tapped_source(payment);
                let mut actions = self.mana_ability_actions(*player, None);
                if waterbend > 0 {
                    // Only offer taps that keep the rest of the cost payable.
                    for candidate in self.waterbend_taps_keeping_payable(
                        *player,
                        &owed.reduced_generic(1),
                        waterbend - 1,
                        self.waterbend_payment_candidates(*player, payment),
                        tapped_source,
                    ) {
                        actions.push(Action::WaterbendTap {
                            player: *player,
                            permanent: candidate,
                        });
                    }
                }
                if self.can_pay_mana_cost(*player, &owed) {
                    actions.push(Action::PayCost { player: *player });
                }
                let focus = match payment.as_ref() {
                    Payment::Activation {
                        source: ActivationSource::Permanent(permanent),
                        ..
                    } => self.state.permanents[*permanent]
                        .as_ref()
                        .map(|perm| vec![perm.id])
                        .unwrap_or_default(),
                    _ => self
                        .payment_source_object(payment)
                        .map(|object| vec![self.state.objects[object].id])
                        .unwrap_or_default(),
                };
                Some(ActionSpace {
                    player: Some(*player),
                    kind: ActionSpaceKind::Waterbend,
                    actions,
                    focus,
                })
            }
        }
    }

    /// Narrow untrusted action ids at card-only entry points.
    fn chosen_card(&self, object: ObjectId) -> Result<CardId, AgentError> {
        self.state
            .card_id(object)
            .ok_or_else(|| AgentError("choice must name a card".into()))
    }

    pub(crate) fn execute_action(&mut self, action: &Action) -> Result<(), AgentError> {
        match action {
            Action::ChooseTrigger { player, trigger_id } => {
                self.choose_trigger(*player, *trigger_id)
            }
            Action::PlayLand { player, object } => {
                self.play_land(*player, self.chosen_card(*object)?)?;
                self.restart_priority_round();
                Ok(())
            }
            Action::CastSpell { player, object } => {
                self.cast_spell_action(*player, self.chosen_card(*object)?)
            }
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
                object,
                ability_index,
            } => self.activate_card_ability_action(*player, *object, *ability_index),
            Action::Foretell { player, object } => {
                if self.state.priority.holder != *player {
                    return Err(AgentError("player does not have priority".to_string()));
                }
                self.foretell(*player, self.chosen_card(*object)?)
            }
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
                } else if self.state.suspended_decision.is_some() {
                    self.execute_decision_action(action)
                } else {
                    Err(AgentError("no pending target choice".to_string()))
                }
            }
            Action::PassPriority { player } => {
                if self.state.priority.holder != *player {
                    return Err(AgentError("player does not have priority".to_string()));
                }
                let next = self.next_player(*player);
                self.state.priority_mut().on_pass(next);
                Ok(())
            }
            Action::DeclareAttacker {
                permanent, attack, ..
            } => self.declare_attacker(*permanent, *attack),
            Action::DeclareBlocker {
                blocker, attacker, ..
            } => self.declare_blocker(*blocker, *attacker),
            Action::ChooseMode { player, mode }
                if self.state.suspended_decision.is_none()
                    && self.pending_choice.is_none()
                    && self
                        .state
                        .trigger_placement
                        .as_ref()
                        .is_some_and(|p| p.placing.is_some()) =>
            {
                self.choose_trigger_mode(*player, *mode)
            }
            Action::ChooseMode { player, mode }
                if matches!(self.pending_choice, Some(PendingChoice::SpellMode { .. })) =>
            {
                self.spell_mode_action(*player, *mode)
            }
            Action::ChooseMode { player, mode }
                if matches!(self.pending_choice, Some(PendingChoice::AnnounceX { .. })) =>
            {
                self.announce_x_action(*player, *mode)
            }
            Action::ChooseMode { player, mode }
                if matches!(self.pending_choice, Some(PendingChoice::ActivationX { .. })) =>
            {
                self.choose_activation_x_action(*player, *mode)
            }
            Action::SelectCard { player, object }
                if matches!(
                    self.pending_choice,
                    Some(PendingChoice::SacrificePayment { .. })
                ) =>
            {
                self.sacrifice_payment_action(*player, Some(*object))
            }
            Action::SelectCard { player, object }
                if matches!(
                    self.pending_choice,
                    Some(PendingChoice::ExileFromHandPayment { .. })
                ) =>
            {
                self.exile_payment_action(*player, self.chosen_card(*object)?)
            }
            Action::SelectCard { player, object }
                if matches!(
                    self.pending_choice,
                    Some(PendingChoice::DiscardPayment { .. })
                ) =>
            {
                self.discard_payment_action(*player, self.chosen_card(*object)?)
            }
            Action::SelectCard { player, object }
                if matches!(self.pending_choice, Some(PendingChoice::TapPayment { .. })) =>
            {
                self.tap_payment_action(*player, *object)
            }
            Action::SelectCard { player, object }
                if matches!(self.pending_choice, Some(PendingChoice::Crew { .. })) =>
            {
                self.crew_tap_action(*player, *object)
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

/// Announcing X offers each payable value as a `ChooseMode` whose mode is X.
fn x_action_space(player: PlayerId, values: Vec<u8>, focus: Vec<PublicId>) -> ActionSpace {
    ActionSpace {
        player: Some(player),
        kind: ActionSpaceKind::Modal,
        actions: values
            .into_iter()
            .map(|x| Action::ChooseMode {
                player,
                mode: usize::from(x),
            })
            .collect(),
        focus,
    }
}

/// The word a `ChooseObject` action starts with: what the instruction does
/// to the chosen permanent.
fn choice_verb(effect: &Effect) -> &'static str {
    match effect {
        Effect::Sacrifice { .. } => "Sacrifice",
        Effect::Tap { .. } => "Tap",
        Effect::Untap { .. } => "Untap",
        Effect::Destroy { .. } => "Destroy",
        Effect::Move {
            to: ZoneType::Exile,
            ..
        } => "Exile",
        Effect::Move {
            to: ZoneType::Hand, ..
        } => "Return",
        _ => "Choose",
    }
}
