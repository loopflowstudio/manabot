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
    flow::{
        game::{ActivationSource, CastChoices, Game, PendingChoice},
        payment::{CostComponent, Operation, Payment, TapPayment},
    },
    state::{
        ability::Effect,
        card::{ActivationZone, CastOption},
        game_object::{CardId, ObjectId, PermanentId, PlayerId, PublicId},
        mana::{Color, Mana, ManaCost},
        payment::Toward,
        player::ManaFloats,
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
                let face = trigger.source_lki.as_ref().map_or_else(
                    || self.state.source_base(trigger.source_object()).0,
                    |lki| lki.definition_id,
                );
                let record = self.source_face(trigger.source_object(), face)?;
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
            Action::WaterbendTap { player, permanent } => {
                // A tap says what it pays unless it only waterbends, which
                // keeps its type name.
                let (_, payment, _) = self
                    .paying()
                    .filter(|_| self.state.suspended_decision.is_none())?;
                let options = self.help_tap_options(*player, payment, *permanent);
                if options.iter().all(|paid| paid.toward == Toward::Waterbend) {
                    return None;
                }
                let name = &self.state[self.permanent_object(*permanent)?].name;
                Some(format!("Tap {name} to pay {}", tap_payments_text(&options)))
            }
            // A fixed source tapped while an announced cost is paid: its mana
            // pays toward that cost as it is made (`commit_produced_mana`).
            Action::TapForMana { permanent, .. }
                if self.state.suspended_decision.is_none()
                    && matches!(self.paying(), Some((_, _, CostComponent::Mana { .. }))) =>
            {
                let name = &self.state[self.permanent_object(*permanent)?].name;
                let output = crate::flow::mana::fixed_output(&self.characteristics(*permanent));
                Some(format!("Tap {name} for {}", mana_description(&output)))
            }
            Action::Foretell {
                object: object_id, ..
            } => Some(format!("Foretell {}", self.state[*object_id].name)),
            Action::ChooseMode { mode, .. }
                if matches!(
                    self.pending_choice,
                    Some(PendingChoice::AnnounceX { .. } | PendingChoice::ActivationX { .. })
                ) =>
            {
                Some(format!("X = {mode}"))
            }
            Action::ChooseMode { mode, .. }
                if self.state.suspended_decision.is_none()
                    && matches!(self.paying(), Some((_, _, CostComponent::TapPays { .. }))) =>
            {
                let (_, _, CostComponent::TapPays { tapped, options }) = self.paying()? else {
                    unreachable!()
                };
                Some(format!(
                    "{} pays {}",
                    self.state[*tapped].name,
                    tap_payments_text(std::slice::from_ref(
                        options.iter().find(|paid| paid.mode() == *mode)?
                    ))
                ))
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
                        let object_id = candidate.source();
                        Some(format!(
                            "{}: {}",
                            self.state[object_id].name,
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
                }) => match self.state[*object_id].cast_option()? {
                    CastOption::ExileFromHand => {
                        Some("Exile a card from your hand instead of paying the mana cost".into())
                    }
                    CastOption::AlternativeMana(cost) => {
                        Some(format!("Pay {} instead of the mana cost", cost_label(cost)))
                    }
                    CastOption::Bargain => {
                        Some("Bargain: sacrifice an artifact, enchantment or token".into())
                    }
                    CastOption::Kicker => None,
                },
                Some(PendingChoice::Pay { payment, .. }) => {
                    let (or_mana, instead) = match &payment.owed[0] {
                        CostComponent::Sacrifice(cost, _) => {
                            (cost.or_mana.as_ref()?, "sacrificing")
                        }
                        CostComponent::Forage { or_mana, .. } => (or_mana, "foraging"),
                        _ => return None,
                    };
                    Some(format!("Pay {} instead of {instead}", cost_label(or_mana)))
                }
                _ => None,
            },
            Action::SelectCard {
                object: object_id, ..
            } => {
                let name = &self.state[*object_id].name;
                let paying = self.paying().and_then(|(_, _, component)| match component {
                    CostComponent::ExileFromHand => Some(format!("Exile {name} to pay")),
                    CostComponent::TapPower {
                        toward: Toward::Saddle,
                        ..
                    } => Some(format!("Tap {name} to saddle")),
                    CostComponent::TapPower { .. } => Some(format!("Tap {name} to crew")),
                    CostComponent::Tap(_, Toward::Station) => {
                        Some(format!("Tap {name} to station"))
                    }
                    CostComponent::Tap(..) => Some(format!("Tap {name} to pay")),
                    CostComponent::Sacrifice(_, Toward::Bargain) => {
                        Some(format!("Sacrifice {name} (bargain)"))
                    }
                    CostComponent::Sacrifice(..) => Some(format!("Sacrifice {name}")),
                    CostComponent::Forage { .. } => {
                        Some(if self.state.permanent_of(*object_id).is_some() {
                            format!("Sacrifice {name} (forage)")
                        } else {
                            format!("Exile {name} (forage)")
                        })
                    }
                    CostComponent::Mana { help, .. }
                        if help.delve.is_some() && self.state.suspended_decision.is_none() =>
                    {
                        Some(format!("Exile {name} to pay {{1}} (delve)"))
                    }
                    CostComponent::Discard
                    | CostComponent::Mana { .. }
                    | CostComponent::TapPays { .. }
                    | CostComponent::Life(_)
                    | CostComponent::Source(_) => None,
                });
                if paying.is_some() {
                    paying
                } else if let Some(suspended) = &self.state.suspended_decision {
                    match &suspended.decision {
                        Decision::ManaUpfront { tap_remaining, .. } if *tap_remaining > 0 => {
                            Some(format!("Tap {} to pay", self.state[*object_id].name))
                        }
                        Decision::ChooseObject { effect, .. } => Some(format!(
                            "{} {}",
                            choice_verb(effect),
                            self.state[*object_id].name
                        )),

                        Decision::SelectCards { .. } => {
                            Some(format!("Select {}", self.state[*object_id].name))
                        }
                        Decision::OrderLibrary { end, .. } => Some(format!(
                            "Put {} next on the {}",
                            self.state[*object_id].name,
                            match end {
                                LibraryEnd::Top => "top",
                                LibraryEnd::Bottom => "bottom",
                            }
                        )),
                        Decision::Surveil { .. } => Some(format!(
                            "Put {} into your graveyard",
                            self.state[*object_id].name
                        )),
                        Decision::SearchNamed { .. } => Some(format!(
                            "Exile {} from {}",
                            self.state[*object_id].name,
                            self.state
                                .zones
                                .zone_of(*object_id)
                                .map_or("unknown zone".to_string(), |zone| format!("{zone:?}"))
                        )),
                        Decision::DistributeCounters { kind, .. } => Some(format!(
                            "Put a {} counter on {}",
                            counter_description(*kind),
                            self.state[*object_id].name
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
                        if self.state[*object_id].cast_option() == Some(CastOption::ExileFromHand)
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
        let mode = self.mana_float_mode(player);
        let mut spell_spends = false;
        let mut actions = Vec::new();
        for card in self.priority_candidate_cards(player) {
            if let Some((action, mana_value)) = self.priority_action_for_card(player, card) {
                if mode == ManaFloats::Usable && !spell_spends {
                    if let Action::CastSpell { object, .. } = action {
                        spell_spends = mana_value > 0
                            || self.cast_needs_x_announcement(
                                player,
                                object,
                                CastChoices::default(),
                            )
                            || self.state[object]
                                .optional_cost()
                                .is_some_and(|cost| cost.mana_value > 0)
                            || self.state[object]
                                .sacrifice_cost()
                                .and_then(|sac| sac.or_mana.as_ref())
                                .is_some_and(|cost| cost.mana_value > 0);
                    }
                }
                actions.push(action);
            }
        }
        let mut floats = Vec::new();
        actions.extend(self.priority_activate_ability_actions(player, mode, &mut floats));
        actions.push(Action::PassPriority { player });
        match mode {
            ManaFloats::Off => {}
            ManaFloats::Usable
                if !spell_spends
                    && !actions
                        .iter()
                        .chain(&floats)
                        .any(|action| self.activation_spends_mana(action)) =>
            {
                self.usable_floats_without_spend(player, &mut floats);
            }
            _ => self.append_mana_floats(player, &mut floats),
        }
        actions.extend(floats);
        actions
    }

    /// Inspect costs only for actions already admitted in this priority window.
    fn activation_spends_mana(&self, action: &Action) -> bool {
        let ability = match *action {
            Action::ActivateAbility {
                permanent,
                ability_index,
                ..
            } => self.activation_ability(ActivationSource::Permanent(permanent), ability_index),
            Action::ActivateCardAbility {
                object,
                ability_index,
                ..
            } => self.state.card_id(object).and_then(|card| {
                self.activation_ability(ActivationSource::Card(card), ability_index)
            }),
            Action::Foretell { .. } => return true,
            _ => return false,
        };
        ability.is_some_and(|ability| {
            ability.mana_cost.mana_value > 0
                || ability.mana_cost.x > 0
                || ability
                    .sacrifice
                    .as_ref()
                    .and_then(|sac| sac.or_mana.as_ref())
                    .is_some_and(|cost| cost.mana_value > 0)
        })
    }

    fn priority_activate_ability_actions(
        &self,
        player: PlayerId,
        mode: ManaFloats,
        floats: &mut Vec<Action>,
    ) -> Vec<Action> {
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
                    if !mode.is_off()
                        && self.can_activate_mana_ability(player, permanent_id, ability_index)
                    {
                        floats.push(Action::ActivateAbility {
                            player,
                            permanent: permanent_id,
                            ability_index,
                        });
                    }
                    continue;
                }
                let operation = Operation::Activation {
                    source,
                    ability_index,
                    targets: Vec::new(),
                    x: 0,
                };
                if let Some(cost) = &ability.sacrifice {
                    if !self
                        .payment_candidates(player, &operation, cost)
                        .into_iter()
                        .any(|id| {
                            self.can_pay_announced_activation(
                                player,
                                &ability.mana_cost,
                                &operation,
                                Some(id),
                            )
                        })
                        && !cost.or_mana.as_ref().is_some_and(|extra| {
                            self.can_pay_announced_activation(
                                player,
                                &ability.mana_cost.plus(extra),
                                &operation,
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
                if ability.tap_power().is_some_and(|(power, _)| {
                    self.crew_power_available(player, permanent_id) < power
                }) {
                    continue;
                }
                let affordable = if ability.chosen_tap().is_some() {
                    let payment = Payment {
                        owed: self.cost_components(player, &operation),
                        operation,
                        receipt: Default::default(),
                    };
                    self.tap_activation_payment_possible(player, &payment, None)
                } else if ability.waterbend {
                    self.can_pay_waterbend_activation(player, permanent_id, ability)
                } else {
                    self.can_pay_announced_activation(player, &ability.mana_cost, &operation, None)
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
                    object: self.state.object_of(card),
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
                for ability_index in 0..self.state[self.state.object_of(card)]
                    .activated_abilities
                    .len()
                {
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
                    let operation = Operation::Activation {
                        source,
                        ability_index,
                        targets: Vec::new(),
                        x: 0,
                    };
                    if self.can_pay_announced_activation(
                        player,
                        &ability.mana_cost,
                        &operation,
                        None,
                    ) {
                        actions.push(Action::ActivateCardAbility {
                            player,
                            object: self.state.object_of(card),
                            ability_index,
                        });
                    }
                }
            }
        }
        actions
    }

    fn priority_action_for_card(&self, player: PlayerId, card: CardId) -> Option<(Action, u8)> {
        let object = &self.state[self.state.object_of(card)];
        if !self.may_play_from_zone(player, self.state.object_of(card)) {
            return None;
        }
        if object.types.is_land() {
            return self.can_play_land(player).then_some((
                Action::PlayLand {
                    player,
                    object: self.state.object_of(card),
                },
                0,
            ));
        }
        if !object.types.is_castable() {
            return None;
        }

        let can_cast_now = if self.has_instant_timing(player, self.state.object_of(card)) {
            self.can_cast_instants(player)
        } else {
            self.can_cast_sorceries(player)
        };
        if !can_cast_now {
            return None;
        }
        self.castable_mana_value(player, card, Default::default())
            .map(|value| {
                (
                    Action::CastSpell {
                        player,
                        object: self.state.object_of(card),
                    },
                    value,
                )
            })
    }

    /// Mana charged by a legal cast at the given choices, or None when modes,
    /// targets or resources prevent it. Timing and zone permission are the caller's.
    pub(crate) fn castable_mana_value(
        &self,
        player: PlayerId,
        card: CardId,
        cast: CastChoices,
    ) -> Option<u8> {
        let object = &self.state[self.state.object_of(card)];
        if object.spell_modes().is_some()
            && self
                .legal_spell_modes(player, self.state.object_of(card))
                .is_empty()
        {
            return None;
        }
        // Every mandatory targeting requirement needs a legal target.
        for requirement in object.target_requirements() {
            if requirement.min > 0
                && self
                    .legal_targets_for_requirement(player, self.state.object_of(card), &requirement)
                    .is_empty()
            {
                return None;
            }
        }

        let Some(cost) = self.effective_spell_cost(player, self.state.object_of(card), cast, None)
        else {
            return Some(0);
        };
        // Castability gating sees pooled mana too — until-end-of-combat mana
        // (firebending) must make spells castable.
        let Some(additional) = object.sacrifice_cost() else {
            if self.can_pay_cast(player, self.state.object_of(card), cast) {
                return Some(cost.mana_value);
            }
            return (self.alternative_cost_available(player, self.state.object_of(card))
                && self.can_pay_cast(
                    player,
                    self.state.object_of(card),
                    CastChoices {
                        kicked: true,
                        ..cast
                    },
                ))
            .then_some(0);
        };
        let operation = Operation::Spell {
            object: self.state.object_of(card),
            cast,
            mode: None,
            targets: Vec::new(),
            target_req_indices: Vec::new(),
        };
        (self
            .payment_candidates(player, &operation, additional)
            .into_iter()
            .any(|id| self.can_pay_with_mana_abilities(player, &cost, Some(id)))
            || additional.or_mana.as_ref().is_some_and(|extra| {
                self.can_pay_with_mana_abilities(player, &cost.plus(extra), None)
            }))
        .then_some(cost.mana_value)
    }

    pub(crate) fn pending_choice_action_space(&self) -> Option<ActionSpace> {
        let _payments = self.payment_scan();
        let choice = self.pending_choice.as_ref()?;
        match choice {
            PendingChoice::Pay { player, payment } => {
                Some(self.payment_action_space(*player, payment))
            }
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
                focus: vec![self.state[object].id],
            }),
            PendingChoice::AnnounceX {
                player,
                object,
                cast,
            } => Some(x_action_space(
                *player,
                self.legal_x_values(*player, *object, *cast),
                vec![self.state[object].id],
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
                focus: vec![self.state[object].id],
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
                let requirements = self.state[object].target_requirements_for_mode(*mode);
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
                    focus: vec![self.state[object].id],
                })
            }
        }
    }

    /// The offers for the cost component being paid. Every legal way to pay
    /// it is offered; an object whose choice would leave the rest of the
    /// cost unpayable is not.
    fn payment_action_space(&self, player: PlayerId, payment: &Payment) -> ActionSpace {
        let select = |objects: Vec<ObjectId>| -> Vec<Action> {
            objects
                .into_iter()
                .map(|object| Action::SelectCard { player, object })
                .collect()
        };
        let permanents = |ids: Vec<PermanentId>| -> Vec<ObjectId> {
            ids.into_iter()
                .filter_map(|id| self.state.permanents[id].as_ref())
                .map(|permanent| permanent.object)
                .collect()
        };
        let source_focus = || -> Vec<PublicId> {
            match &payment.operation {
                Operation::Activation {
                    source: ActivationSource::Permanent(permanent),
                    ..
                } => self.state.permanents[*permanent]
                    .as_ref()
                    .map(|perm| vec![perm.id])
                    .unwrap_or_default(),
                operation => self
                    .payment_source_object(operation)
                    .map(|object| vec![self.state[object].id])
                    .unwrap_or_default(),
            }
        };
        let (kind, actions, focus) = match &payment.owed[0] {
            CostComponent::TapPower { .. } => {
                let Operation::Activation {
                    source: ActivationSource::Permanent(vehicle),
                    ..
                } = payment.operation
                else {
                    unreachable!("a permanent's ability taps creatures for power")
                };
                (
                    ActionSpaceKind::LookAndSelect,
                    select(permanents(self.crew_candidates(player, vehicle))),
                    Vec::new(),
                )
            }
            CostComponent::Tap(cost, toward) => (
                ActionSpaceKind::LookAndSelect,
                select(permanents(
                    self.tap_component_candidates(player, &payment.operation, cost, *toward)
                        .into_iter()
                        .filter(|id| {
                            self.tap_activation_payment_possible(player, payment, Some(*id))
                        })
                        .collect(),
                )),
                Vec::new(),
            ),
            CostComponent::Discard => (
                ActionSpaceKind::Discard,
                select(self.discard_candidates(player, payment.operation.spell())),
                Vec::new(),
            ),
            CostComponent::ExileFromHand => (
                ActionSpaceKind::LookAndSelect,
                select(
                    self.payment_source_object(&payment.operation)
                        .map(|object| self.exile_alternative_cards(player, object))
                        .unwrap_or_default()
                        .into_iter()
                        .map(|card| self.state.object_of(card))
                        .collect(),
                ),
                Vec::new(),
            ),
            CostComponent::Forage { or_mana, exiled } => {
                let mut actions = select(self.forage_objects(player, payment));
                if *exiled == 0
                    && self.can_pay_with_mana_abilities(
                        player,
                        &payment.mana().0.plus(or_mana),
                        None,
                    )
                {
                    actions.push(Action::PayCost { player });
                }
                (ActionSpaceKind::LookAndSelect, actions, Vec::new())
            }
            CostComponent::Sacrifice(cost, _) => {
                let mana = payment.mana().0;
                let mut actions = select(permanents(
                    self.payment_candidates(player, &payment.operation, cost)
                        .into_iter()
                        .filter(|id| self.can_pay_with_mana_abilities(player, mana, Some(*id)))
                        .collect(),
                ));
                if cost.or_mana.as_ref().is_some_and(|extra| {
                    self.can_pay_with_mana_abilities(player, &mana.plus(extra), None)
                }) {
                    actions.push(Action::PayCost { player });
                }
                (ActionSpaceKind::LookAndSelect, actions, Vec::new())
            }
            CostComponent::Mana { .. } => {
                let owed = payment.mana().0;
                let mut actions = self.mana_ability_actions(player);
                actions.extend(self.useful_payment_taps(player));
                actions.extend(self.help_actions(player));
                if self.can_pay_mana_cost(player, owed) {
                    actions.push(Action::PayCost { player });
                }
                self.append_mana_floats(player, &mut actions);
                if payment.waterbends() {
                    (ActionSpaceKind::Waterbend, actions, source_focus())
                } else {
                    (ActionSpaceKind::PayOrNot, actions, Vec::new())
                }
            }
            // The offers are about the permanent just tapped, so it is the
            // focus; each mode names its payment (`TapPayment::mode`).
            CostComponent::TapPays { tapped, options } => (
                ActionSpaceKind::Modal,
                options
                    .iter()
                    .map(|paid| Action::ChooseMode {
                        player,
                        mode: paid.mode(),
                    })
                    .collect(),
                vec![self.state[*tapped].id],
            ),
            CostComponent::Life(_) | CostComponent::Source(_) => {
                unreachable!("paid as the payment completes, after its mana")
            }
        };
        ActionSpace {
            player: Some(player),
            kind,
            actions,
            focus,
        }
    }

    /// Narrow untrusted action ids at card-only entry points.
    fn chosen_card(&self, object: ObjectId) -> Result<CardId, AgentError> {
        self.state
            .card_of(object)
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
                if self.state.suspended_decision.is_none()
                    && matches!(self.paying(), Some((_, _, CostComponent::TapPays { .. }))) =>
            {
                self.tap_pays_action(*player, *mode)
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
                if self.state.suspended_decision.is_none() && self.paying().is_some() =>
            {
                self.pay_with_object(*player, *object)
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
                    Some(PendingChoice::Pay { .. }) => self.pay_cost_action(*player),
                    _ => Err(AgentError("no cost payment is pending".to_string())),
                }
            }
            Action::WaterbendTap { player, permanent } => {
                if self.state.suspended_decision.is_some() {
                    return self.execute_decision_action(action);
                }
                self.help_tap_action(*player, *permanent)
            }
            Action::TapForMana { player, permanent } => {
                self.tap_for_mana_action(*player, *permanent)
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

/// What a direct tap pays, as "{W} or {1} (convoke)", or "{1} (convoke)
/// or {1} (waterbend)" when its reasons differ.
fn tap_payments_text(options: &[TapPayment]) -> String {
    let mut text = String::new();
    for (index, paid) in options.iter().enumerate() {
        if index > 0 {
            text.push_str(" or ");
        }
        let symbol = match paid.color {
            Color::Generic => "1",
            color => color.symbol(),
        };
        text.push_str(&format!("{{{symbol}}}"));
        if options
            .get(index + 1)
            .is_none_or(|next| next.toward != paid.toward)
        {
            text.push_str(&format!(" ({})", toward_keyword(paid.toward)));
        }
    }
    text
}

/// The keyword a direct tap pays by.
fn toward_keyword(toward: Toward) -> &'static str {
    match toward {
        Toward::Convoke => "convoke",
        Toward::Improvise => "improvise",
        _ => "waterbend",
    }
}

/// A mana cost as its symbols: generic first, then colors.
pub(crate) fn cost_label(cost: &ManaCost) -> String {
    let mut label = String::new();
    if cost.generic() > 0 {
        label.push_str(&format!("{{{}}}", cost.generic()));
    }
    let colored = Mana {
        mana: cost.cost[..6].try_into().expect("six colored mana slots"),
        ..Default::default()
    };
    label.push_str(&mana_description(&colored));
    label
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

#[cfg(test)]
mod identity_tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::{search_state::witness, state::player::PlayerConfig};

    #[test]
    fn stale_card_action_cannot_follow_a_card_back_to_hand() {
        let mut game = Game::new(
            vec![
                PlayerConfig::new("p0", BTreeMap::from([("Mountain".into(), 40)])),
                PlayerConfig::new("p1", BTreeMap::from([("Island".into(), 40)])),
            ],
            400_715,
            false,
        );
        let player = game.active_player();
        game.state.turn.current_phase = 1;
        game.state.turn.current_step = 0;
        let old = game.state.zones.zone_objects(ZoneType::Hand, player)[0];
        let card = game.state.card_of(old).unwrap();
        let exiled = game.move_object(old, ZoneType::Exile);
        let returned = game.move_object(exiled, ZoneType::Hand);
        let before = witness(&game);
        assert!(game
            .execute_action(&Action::PlayLand {
                player,
                object: old
            })
            .is_err());
        assert_eq!(witness(&game), before);
        assert_eq!(game.chosen_card(returned).unwrap(), card);
        assert!(game
            .execute_action(&Action::PlayLand {
                player,
                object: returned
            })
            .is_ok());
        assert_eq!(
            game.state.zones.zone_of(game.state.object_of(card)),
            Some(ZoneType::Battlefield)
        );
    }
}
