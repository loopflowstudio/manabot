//! Read-only explanation of the production witness. No announcement, target,
//! output or mana unit is committed by this query.
use super::*;
use crate::agent::ability_description::cost_description;
use crate::experience::{ObjectRenderId, PooledManaView, SubjectRef};
use crate::flow::game::CastChoices;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PaymentPreviewState {
    Exact,
    NeedsChoice,
    NotApplicable,
    Unavailable,
}

/// The cost boundary explained, never a promise about resolution effects.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PaymentScope {
    SpellCost,
    ActivationFee,
    CurrentPayment,
}

impl PaymentScope {
    fn for_payment(payment: &Payment) -> Self {
        match payment {
            Payment::Spell { .. } => Self::SpellCost,
            Payment::Activation { .. } => Self::ActivationFee,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct PaymentContinuation {
    /// Native output alternatives follow the fee; legality can narrow them.
    pub ability_output: bool,
    /// Another payment remains outside this fee, including optional payments.
    pub outer_payment: bool,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct PaymentSourcePreview {
    pub source: SubjectRef,
    pub label: String,
    pub output: PreviewPool,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct PaymentPreview {
    pub state: PaymentPreviewState,
    pub scope: Option<PaymentScope>,
    pub continuation: PaymentContinuation,
    pub explanation: String,
    pub cost: Option<String>,
    pub sources: Vec<PaymentSourcePreview>,
    pub pool_after_production: Option<PreviewPool>,
    /// Absent if more than one legal unit can be spent at any point.
    pub pool_after_spending: Option<PreviewPool>,
}

/// Public quantities and class rules, without internal spending snapshots.
#[derive(Clone, Debug, serde::Serialize)]
pub struct PreviewPool {
    pub amounts: [u16; 6],
    pub classes: Vec<PooledManaView>,
}

impl From<&Mana> for PreviewPool {
    fn from(pool: &Mana) -> Self {
        let mut amounts = pool.mana.map(u16::from);
        for (_, class_amounts) in pool.classes() {
            for (total, amount) in amounts.iter_mut().zip(class_amounts) {
                *total += u16::from(*amount);
            }
        }
        Self {
            amounts,
            classes: pool
                .classes()
                .iter()
                .map(|(class, amounts)| PooledManaView {
                    mana: amounts.map(i32::from),
                    combat: class.combat,
                    restriction: class.restriction.as_ref().map(|r| r.describe()),
                    effect: class.effect.as_ref().map(|e| e.describe()),
                })
                .collect(),
        }
    }
}

impl PaymentPreview {
    fn message(
        state: PaymentPreviewState,
        explanation: &str,
        scope: Option<PaymentScope>,
        continuation: PaymentContinuation,
    ) -> Self {
        Self {
            state,
            scope,
            continuation,
            explanation: explanation.into(),
            cost: None,
            sources: Vec::new(),
            pool_after_production: None,
            pool_after_spending: None,
        }
    }
}

impl Game {
    fn preview_activation_continuation(
        &self,
        permanent: PermanentId,
        ability_index: usize,
        outer_payment: bool,
    ) -> PaymentContinuation {
        PaymentContinuation {
            ability_output: self.characteristics(permanent).activated_abilities[ability_index]
                .mana_choices()
                .is_some_and(|choices| choices.len() > 1),
            outer_payment,
        }
    }

    fn preview_boundary(&self, action: &Action) -> (Option<PaymentScope>, PaymentContinuation) {
        let mut continuation = PaymentContinuation::default();
        let pending_payment = matches!(
            self.pending_choice,
            Some(PendingChoice::ManaPayment { .. } | PendingChoice::Waterbend { .. })
        );
        let scope = match action {
            Action::CastSpell { .. } => Some(PaymentScope::SpellCost),
            Action::ChooseTarget { .. }
                if matches!(
                    self.pending_choice,
                    Some(PendingChoice::ChooseTargets { .. })
                ) =>
            {
                Some(PaymentScope::SpellCost)
            }
            Action::ActivateAbility {
                permanent,
                ability_index,
                ..
            } => {
                let outer_payment = pending_payment
                    || self.state.suspended_decision.as_ref().is_some_and(|s| {
                        matches!(
                            s.decision,
                            Decision::Payment(_)
                                | Decision::ManaSpend { .. }
                                | Decision::ManaUpfront { .. }
                        )
                    });
                continuation =
                    self.preview_activation_continuation(*permanent, *ability_index, outer_payment);
                Some(PaymentScope::ActivationFee)
            }
            Action::PayCost { .. } => Some(match &self.pending_choice {
                Some(
                    PendingChoice::ManaPayment { payment, .. }
                    | PendingChoice::Waterbend { payment, .. },
                ) => PaymentScope::for_payment(payment),
                _ => PaymentScope::CurrentPayment,
            }),
            // A spending offer is one unit of the same pending fee/cost. It
            // has no automatic source plan, but keeps its continuation clear.
            Action::ChooseMode { .. } => {
                match self.state.suspended_decision.as_ref().map(|s| &s.decision) {
                    Some(Decision::ManaSpend { completion, .. }) => match completion.as_ref() {
                        SpendCompletion::Upfront(next) => {
                            if let Decision::ManaUpfront {
                                permanent,
                                ability_index,
                                resume,
                                ..
                            } = next.as_ref()
                            {
                                continuation = self.preview_activation_continuation(
                                    *permanent,
                                    *ability_index,
                                    resume.is_some() || pending_payment,
                                );
                            }
                            Some(PaymentScope::ActivationFee)
                        }
                        SpendCompletion::Announced { payment, .. } => {
                            Some(PaymentScope::for_payment(payment))
                        }
                        _ => Some(PaymentScope::CurrentPayment),
                    },
                    _ => None,
                }
            }
            _ => None,
        };
        (scope, continuation)
    }

    /// Query one legal action at its exact semantic authority address.
    pub fn payment_preview(
        &self,
        viewer: PlayerId,
        revision: u64,
        fingerprint: &str,
        action_index: usize,
    ) -> Result<PaymentPreview, AgentError> {
        let _scan = self.board_scan();
        let frame = self
            .semantic_decision_frame()
            .map_err(|e| AgentError(e.to_string()))?;
        if frame.actor as usize != viewer.0
            || frame.revision != revision
            || frame.fingerprint != fingerprint
        {
            return Err(AgentError("stale or unauthorized payment preview".into()));
        }
        let space = self
            .action_space()
            .ok_or_else(|| AgentError("no decision".into()))?;
        let action = space
            .actions
            .get(action_index)
            .ok_or_else(|| AgentError("unknown payment action".into()))?;
        let (scope, continuation) = self.preview_boundary(action);
        let player = action.player();
        let mut reservation = None;
        let mut upfront_obligations = None;
        let cost = match action {
            Action::CastSpell { object, .. } => {
                let card = &self.state[*object];
                if self.cast_needs_x_announcement(player, *object, CastChoices::default()) {
                    return Ok(PaymentPreview::message(
                        PaymentPreviewState::NeedsChoice,
                        "Choose X or replicate count before previewing payment.",
                        scope,
                        continuation,
                    ));
                }
                let Some(cast) =
                    self.fixed_cast_cost_choices(player, *object, CastChoices::default())
                else {
                    return Ok(PaymentPreview::message(
                        PaymentPreviewState::NeedsChoice,
                        "Choose the optional or alternative cost before previewing payment.",
                        scope,
                        continuation,
                    ));
                };
                if card.sacrifice_cost().is_some()
                    || (cast.kicked && card.exile_alternative().is_some())
                {
                    return Ok(PaymentPreview::message(
                        PaymentPreviewState::NeedsChoice,
                        "Choose the additional or alternative cost before previewing payment.",
                        scope,
                        continuation,
                    ));
                }
                if card.cost_reduction_if_targets().is_some() {
                    return Ok(PaymentPreview::message(
                        PaymentPreviewState::NeedsChoice,
                        "Choose targets before previewing this target-dependent cost.",
                        scope,
                        continuation,
                    ));
                }
                self.effective_spell_cost(player, *object, cast, None)
            }
            Action::ChooseTarget { target, .. } => match &self.pending_choice {
                Some(PendingChoice::ChooseTargets {
                    object,
                    cast,
                    mode,
                    requirement_index,
                    chosen,
                    chosen_req_indices,
                    ..
                }) => {
                    let card = &self.state[*object];
                    if card.sacrifice_cost().is_some() || card.exile_alternative().is_some() {
                        return Ok(PaymentPreview::message(
                            PaymentPreviewState::NeedsChoice,
                            "Choose the remaining additional costs before previewing payment.",
                            scope,
                            continuation,
                        ));
                    }
                    let requirements = card.target_requirements_for_mode(*mode);
                    let completes = *requirement_index + 1 == requirements.len()
                        && chosen_req_indices
                            .iter()
                            .filter(|index| **index == *requirement_index)
                            .count()
                            + 1
                            >= requirements[*requirement_index].max_for(cast.x);
                    if card.cost_reduction_if_targets().is_some() && !completes {
                        return Ok(PaymentPreview::message(PaymentPreviewState::NeedsChoice,
                            "Choose the remaining targets before previewing this target-dependent cost.", scope, continuation));
                    }
                    let mut targets = chosen.clone();
                    targets.push(*target);
                    self.effective_spell_cost(player, *object, *cast, Some(&targets))
                }
                _ => None,
            },
            Action::PayCost { .. } => match &self.pending_choice {
                Some(PendingChoice::ManaPayment {
                    payment,
                    alternative,
                    ..
                }) => Some(self.payment_mana_cost(player, payment, alternative.as_ref())),
                Some(PendingChoice::Waterbend {
                    payment,
                    remaining_generic,
                    tapped,
                    ..
                }) => Some(
                    self.waterbend_owed(player, payment, *remaining_generic, *tapped)
                        .0,
                ),
                Some(_) => {
                    return Ok(PaymentPreview::message(
                        PaymentPreviewState::NeedsChoice,
                        "Finish announcing costs and targets before previewing payment.",
                        scope,
                        continuation,
                    ))
                }
                None => self
                    .state
                    .suspended_decision
                    .as_ref()
                    .and_then(|s| match &s.decision {
                        Decision::Payment(payment) => Some(payment.remaining.mana_cost()),
                        _ => None,
                    }),
            },
            Action::ChooseMode { .. } => {
                if let Some(Decision::ManaSpend { cost, .. }) =
                    self.state.suspended_decision.as_ref().map(|s| &s.decision)
                {
                    let mut preview = PaymentPreview::message(
                        PaymentPreviewState::NeedsChoice,
                        "Choose mana to finish paying this cost; the spending remainder is not yet determined.",
                        scope,
                        continuation,
                    );
                    preview.cost = Some(cost_description(cost));
                    return Ok(preview);
                }
                None
            }
            Action::ActivateAbility {
                permanent,
                ability_index,
                ..
            } => {
                let ability = &self.characteristics(*permanent).activated_abilities[*ability_index];
                if ability.mana_cost.x > 0
                    || ability.sacrifice.is_some()
                    || ability.tap_cost.is_some()
                    || ability.waterbend
                    || ability.discard_card
                {
                    return Ok(PaymentPreview::message(
                        PaymentPreviewState::NeedsChoice,
                        "Choose the activation's additional costs before previewing payment.",
                        scope,
                        continuation,
                    ));
                }
                reservation = Some(SourceReservation {
                    permanent: *permanent,
                    tap: ability.tap_source,
                    sacrifice: ability.tap_source || ability.sacrifice_source,
                    used_ability: (ability.once_each_turn || ability.exhaust)
                        .then_some(*ability_index),
                });
                if ability.mana_choices().is_some() {
                    let model = self.payment_model(player, None);
                    let activation = model
                        .sources
                        .iter()
                        .find(|s| s.permanent == *permanent)
                        .and_then(|s| {
                            s.abilities
                                .iter()
                                .find(|a| a.ability_index == *ability_index)
                        })
                        .expect("legal mana activation");
                    let mut obligations = PaymentModel::activation_obligations(activation);
                    obligations.extend(self.payment_obligations(player));
                    upfront_obligations = Some(obligations);
                }
                Some(ability.mana_cost.clone())
            }
            _ => None,
        };
        let Some(cost) = cost else {
            return Ok(PaymentPreview::message(
                PaymentPreviewState::NotApplicable,
                "This action does not have an automatic payment preview.",
                scope,
                continuation,
            ));
        };
        let policy = if upfront_obligations.is_some() {
            SourcePolicy::AutomaticPrefix
        } else {
            SourcePolicy::FixedOnly
        };
        let obligations = upfront_obligations.unwrap_or_else(|| {
            vec![Obligation::Pay {
                cost: cost.clone(),
                waterbend: false,
            }]
        });
        let path = self.automatic_payment_plan(player, obligations, policy, reservation);
        let Some(path) = path.filter(|path| {
            path.iter()
                .take_while(|step| !matches!(step, PaymentStep::Spend))
                .all(|step| matches!(step, PaymentStep::Fixed(_)))
        }) else {
            let mut preview = PaymentPreview::message(
                PaymentPreviewState::Unavailable,
                "No automatic route. Use a legal mana activation and choose its output first.",
                scope,
                continuation,
            );
            preview.cost = Some(cost_description(&cost));
            return Ok(preview);
        };
        let triggers = self.mana_triggers();
        let mut pool = self.pooled_mana(player);
        let mut sources = Vec::new();
        for step in path
            .into_iter()
            .take_while(|step| !matches!(step, PaymentStep::Spend))
        {
            let PaymentStep::Fixed(permanent) = step else {
                continue;
            };
            let view = self.characteristics(permanent);
            let output = self
                .fixed_source_mana(player, permanent, &view, &triggers)
                .expect("planned fixed source");
            pool.add(&output);
            let object = self
                .permanent_object(permanent)
                .expect("planned source on battlefield");
            sources.push(PaymentSourcePreview {
                source: SubjectRef::Object {
                    id: ObjectRenderId {
                        entity: self.state.permanents[permanent].as_ref().unwrap().id.0,
                        incarnation: object.generation(),
                    },
                },
                label: view.name.clone(),
                output: PreviewPool::from(&output),
            });
        }
        let mut remaining_pool = pool.clone();
        let mut remaining_cost = cost.clone();
        let final_pool = loop {
            if remaining_cost.mana_value == 0 {
                break Some(remaining_pool);
            }
            let choices = remaining_pool.spending_choices(&remaining_cost);
            let [only] = choices.as_slice() else {
                break None;
            };
            remaining_pool.spend(&only.unit, &remaining_cost.purpose);
            remaining_cost = only.remaining.clone();
        };
        Ok(PaymentPreview {
            state: PaymentPreviewState::Exact,
            scope,
            continuation,
            explanation: if final_pool.is_some() {
                "Exact automatic source plan."
            } else {
                "Exact automatic source plan; mana spending still requires a choice."
            }
            .into(),
            cost: Some(cost_description(&cost)),
            sources,
            pool_after_production: Some(PreviewPool::from(&pool)),
            pool_after_spending: final_pool.as_ref().map(PreviewPool::from),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        agent::observation::Observation,
        search_state::witness,
        state::{player::PlayerConfig, zone::ZoneType},
    };
    use std::collections::BTreeMap;

    fn preview_game() -> Game {
        let mut game = Game::new(
            vec![
                PlayerConfig::new(
                    "payer",
                    BTreeMap::from([
                        ("Island".into(), 24),
                        ("Ancestral Recall".into(), 8),
                        ("Prophetic Prism".into(), 8),
                    ]),
                ),
                PlayerConfig::new("other", BTreeMap::from([("Plains".into(), 40)])),
            ],
            165,
            true,
        );
        game.scenario_force_card_in_hand(PlayerId(0), "Ancestral Recall")
            .unwrap();
        for _ in 0..2 {
            game.scenario_force_battlefield(PlayerId(0), "Island", true)
                .unwrap();
        }
        game.scenario_refresh_priority().unwrap();
        game
    }

    fn cast_index(game: &Game) -> usize {
        game.action_space()
            .unwrap()
            .actions
            .iter()
            .position(|action| {
                matches!(action,
            Action::CastSpell { object, .. } if game.state[*object].name == "Ancestral Recall")
            })
            .unwrap()
    }

    fn preview(game: &Game, index: usize) -> PaymentPreview {
        let frame = game.semantic_decision_frame().unwrap();
        game.payment_preview(PlayerId(0), frame.revision, &frame.fingerprint, index)
            .unwrap()
    }

    #[test]
    fn payment_preview_is_read_only_and_matches_committed_sources() {
        let mut game = preview_game();
        let before = witness(&game);
        let result = preview(&game, cast_index(&game));
        assert_eq!(result.state, PaymentPreviewState::Exact);
        assert!(!result.sources.is_empty());
        assert_eq!(before, witness(&game));
        let cost = ManaCost::parse("U");
        game.produce_mana(PlayerId(0), &cost).unwrap();
        let actual = Observation::for_player(&game, PlayerId(0));
        for source in result.sources {
            let SubjectRef::Object { id } = source.source else {
                panic!("source object")
            };
            assert!(actual
                .agent_permanents
                .iter()
                .any(|p| p.id == id.entity as i32 && p.tapped));
        }
        assert_eq!(
            result.pool_after_production.unwrap().amounts,
            PreviewPool::from(&game.pooled_mana(PlayerId(0))).amounts
        );
    }

    #[test]
    fn payment_preview_deliberate_source_changes_plan_and_invalidates_address() {
        let mut game = preview_game();
        let frame = game.semantic_decision_frame().unwrap();
        let first = preview(&game, cast_index(&game));
        let chosen = game.action_space().unwrap().actions.iter().position(|action| matches!(action,
            Action::TapForMana { permanent, .. } if first.sources.iter().all(|s|
                !matches!(&s.source, SubjectRef::Object { id } if id.entity == game.state.permanents[*permanent].as_ref().unwrap().id.0)))).unwrap();
        game.step(chosen).unwrap();
        assert_eq!(game.pooled_mana(PlayerId(0)).total(), 1);
        assert!(game
            .payment_preview(
                PlayerId(0),
                frame.revision,
                &frame.fingerprint,
                cast_index(&game)
            )
            .is_err());
        assert!(preview(&game, cast_index(&game)).sources.is_empty());
        assert!(game
            .payment_preview(PlayerId(1), frame.revision, &frame.fingerprint, 0)
            .is_err());
        assert!(game.state.zone_cards(ZoneType::Hand, PlayerId(0)).count() > 0);
    }

    #[test]
    fn payment_preview_does_not_guess_a_spending_choice() {
        let mut game = preview_game();
        game.state
            .mana_pool_mut(PlayerId(0))
            .add(&Mana::parse("UR"));
        let exact = preview(&game, cast_index(&game));
        assert!(exact.sources.is_empty());
        assert_eq!(
            exact.pool_after_spending.unwrap().amounts,
            [0, 0, 0, 1, 0, 0]
        );
        game.run_frame(EffectFrame::new(
            PlayerId(0),
            vec![Effect::OptionalPayment {
                cost: crate::state::ability::PaymentCost::Mana(ManaCost::parse("1")),
                if_paid: Vec::new(),
                if_declined: Vec::new(),
            }],
        ));
        game.publish_action_space(game.suspended_decision_action_space().unwrap());
        let pay = game
            .action_space()
            .unwrap()
            .actions
            .iter()
            .position(|a| matches!(a, Action::PayCost { .. }))
            .unwrap();
        assert!(preview(&game, pay).pool_after_spending.is_none());
    }
    #[test]
    fn payment_preview_zero_cost_needs_no_sources_or_spending() {
        let mut game = preview_game();
        game.run_frame(EffectFrame::new(
            PlayerId(0),
            vec![Effect::OptionalPayment {
                cost: crate::state::ability::PaymentCost::Mana(ManaCost::default()),
                if_paid: Vec::new(),
                if_declined: Vec::new(),
            }],
        ));
        game.publish_action_space(game.suspended_decision_action_space().unwrap());
        let pay = game
            .action_space()
            .unwrap()
            .actions
            .iter()
            .position(|a| matches!(a, Action::PayCost { .. }))
            .unwrap();
        let result = preview(&game, pay);
        assert_eq!(result.state, PaymentPreviewState::Exact);
        assert_eq!(result.cost.as_deref(), Some("{0}"));
        assert!(result.sources.is_empty());
        assert_eq!(result.pool_after_spending.unwrap().amounts, [0; 6]);
    }

    #[test]
    fn payment_preview_keeps_restrictions_and_combat_expiry() {
        use crate::state::mana::ManaRestriction;
        let mut game = preview_game();
        game.state.mana_pool_mut(PlayerId(0)).add(&Mana::parse("U"));
        game.state
            .mana_pool_mut(PlayerId(0))
            .add(&Mana::restricted_to(
                ManaRestriction::spell(["Lesson"]),
                [0, 1, 0, 0, 0, 0],
            ));
        game.state
            .combat_mana_pool_mut(PlayerId(0))
            .add(&Mana::parse("R"));
        let result = preview(&game, cast_index(&game));
        assert!(result.sources.is_empty());
        let final_pool = result.pool_after_spending.unwrap();
        assert_eq!(final_pool.amounts, [0, 1, 0, 1, 0, 0]);
        assert!(final_pool.classes.iter().any(|class| class.combat));
        assert!(final_pool.classes.iter().any(|class| class
            .restriction
            .as_ref()
            .is_some_and(|r| r.contains("Lesson"))));
    }

    #[test]
    fn payment_preview_upfront_fee_uses_reserved_production_prefix() {
        let mut game = preview_game();
        let prism = game
            .scenario_force_battlefield(PlayerId(0), "Prophetic Prism", true)
            .unwrap();
        game.scenario_refresh_priority().unwrap();
        let index = game
            .action_space()
            .unwrap()
            .actions
            .iter()
            .position(
                |a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == prism),
            )
            .unwrap();
        let result = preview(&game, index);
        assert_eq!(result.state, PaymentPreviewState::Exact);
        assert_eq!(result.cost.as_deref(), Some("{1}"));
        assert_eq!(result.scope, Some(PaymentScope::ActivationFee));
        assert!(result.continuation.ability_output);
        assert!(!result.continuation.outer_payment);
        assert_eq!(result.pool_after_spending.as_ref().unwrap().amounts, [0; 6]);
        assert!(!result.sources.is_empty());
        game.step(index).unwrap();
        for source in result.sources {
            let SubjectRef::Object { id } = source.source else {
                panic!("object source")
            };
            assert!(game
                .state
                .permanents
                .iter()
                .flatten()
                .any(|p| p.id.0 == id.entity && p.tapped));
        }
        assert!(matches!(
            game.state.suspended_decision.as_ref().unwrap().decision,
            Decision::ManaChoice { .. }
        ));
    }

    #[test]
    fn payment_preview_nested_fee_keeps_boundary_during_spending() {
        let mut game = preview_game();
        let prism = game
            .scenario_force_battlefield(PlayerId(0), "Prophetic Prism", true)
            .unwrap();
        game.state
            .mana_pool_mut(PlayerId(0))
            .add(&Mana::parse("UR"));
        game.scenario_refresh_priority().unwrap();
        let cast = cast_index(&game);
        assert_eq!(preview(&game, cast).scope, Some(PaymentScope::SpellCost));
        game.step(cast).unwrap();
        assert!(matches!(
            game.pending_choice,
            Some(PendingChoice::ManaPayment { .. })
        ));
        let activate = game
            .action_space()
            .unwrap()
            .actions
            .iter()
            .position(
                |a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == prism),
            )
            .unwrap();
        let before = witness(&game);
        let fee = preview(&game, activate);
        assert_eq!(fee.scope, Some(PaymentScope::ActivationFee));
        assert!(fee.continuation.ability_output);
        assert!(fee.continuation.outer_payment);
        assert!(fee.pool_after_spending.is_none());
        assert_eq!(before, witness(&game));
        game.step(activate).unwrap();
        assert!(matches!(
            game.state.suspended_decision.as_ref().unwrap().decision,
            Decision::ManaSpend { .. }
        ));
        let spending = preview(&game, 0);
        assert_eq!(spending.scope, fee.scope);
        assert_eq!(spending.cost.as_deref(), Some("{1}"));
        assert_eq!(spending.state, PaymentPreviewState::NeedsChoice);
        assert_eq!(spending.continuation, fee.continuation);
        // No prediction of the chosen unit or the ability's output.
        assert!(spending.pool_after_spending.is_none());
        game.step(0).unwrap();
        assert!(matches!(
            game.state.suspended_decision.as_ref().unwrap().decision,
            Decision::ManaChoice { .. }
        ));
        assert_eq!(game.pooled_mana(PlayerId(0)).total(), 1);
        game.step(0).unwrap();
        assert!(matches!(
            game.pending_choice,
            Some(PendingChoice::ManaPayment { .. })
        ));
        assert_eq!(game.pooled_mana(PlayerId(0)).total(), 2);
    }

    #[test]
    fn payment_preview_preserves_fixed_and_indexed_choices_in_policy_input() {
        use crate::agent::observation_encoder::{encode, ObservationEncoderConfig, ACTION_DIM};
        let mut game = Game::new(
            vec![
                PlayerConfig::new("payer", BTreeMap::from([("Tarnished Citadel".into(), 40)])),
                PlayerConfig::new("other", BTreeMap::from([("Plains".into(), 40)])),
            ],
            165,
            true,
        );
        let source = game
            .scenario_force_battlefield(PlayerId(0), "Tarnished Citadel", true)
            .unwrap();
        game.scenario_refresh_priority().unwrap();
        let actions = &game.action_space().unwrap().actions;
        let fixed = actions
            .iter()
            .position(|a| matches!(a, Action::TapForMana { permanent, .. } if *permanent == source))
            .unwrap();
        let modal = actions
            .iter()
            .position(
                |a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == source),
            )
            .unwrap();
        let obs = Observation::for_player(&game, PlayerId(0));
        assert_eq!(obs.action_space.actions[fixed].ability_index, None);
        assert_eq!(obs.action_space.actions[modal].ability_index, Some(0));
        assert_eq!(
            obs.action_space.actions[fixed].focus,
            obs.action_space.actions[modal].focus
        );
        let encoded = encode(&obs, &ObservationEncoderConfig::default()).unwrap();
        assert_ne!(
            &encoded.actions[fixed * ACTION_DIM..(fixed + 1) * ACTION_DIM],
            &encoded.actions[modal * ACTION_DIM..(modal + 1) * ACTION_DIM]
        );
    }
}
