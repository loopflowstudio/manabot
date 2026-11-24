//! Read-only explanation of the production witness. No announcement, target,
//! output or mana unit is committed by this query.
use super::*;
use crate::agent::ability_description::cost_description;
use crate::experience::{ObjectRenderId, PooledManaView, SubjectRef};
use crate::flow::game::CastChoices;
use crate::flow::payment::CostComponent;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PaymentPreviewState {
    Exact,
    NeedsChoice,
    NotApplicable,
    Unavailable,
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
                    effect: class
                        .effect
                        .as_ref()
                        .map(crate::agent::ability_description::spend_effect_description),
                })
                .collect(),
        }
    }
}

/// Whether `owed` still asks its payer for something other than mana: a
/// component ahead of the mana (a sacrifice, discard, exile or forage). The
/// preview does not make those choices for the player.
fn owes_a_choice(owed: &[CostComponent]) -> bool {
    !matches!(owed[0], CostComponent::Mana { .. })
}

/// Whether a cost not yet being paid would ask its payer for more than mana.
/// Before the payment begins a waterbend cost's taps are such a choice too;
/// once it is being paid, the preview shows its mana route.
fn will_owe_a_choice(owed: &[CostComponent]) -> bool {
    owes_a_choice(owed)
        || matches!(&owed[0], CostComponent::Mana { help, .. } if help.waterbend > 0)
}

/// Whether convoke, improvise or delve may pay part of the mana `owed`.
fn keyword_help(owed: &[CostComponent]) -> bool {
    matches!(&owed[0], CostComponent::Mana { help, .. } if help.keyworded())
}

impl PaymentPreview {
    fn needs_additional_cost() -> Self {
        Self::message(
            PaymentPreviewState::NeedsChoice,
            "Choose the additional or alternative cost before previewing payment.",
        )
    }

    fn message(state: PaymentPreviewState, explanation: &str) -> Self {
        Self {
            state,
            explanation: explanation.into(),
            cost: None,
            sources: Vec::new(),
            pool_after_production: None,
            pool_after_spending: None,
        }
    }
}

impl Game {
    /// What casting `object` with `cast` would owe.
    fn spell_owed(
        &self,
        player: PlayerId,
        object: ObjectId,
        cast: CastChoices,
    ) -> Vec<CostComponent> {
        self.cost_components(
            player,
            &Operation::Spell {
                object,
                cast,
                mode: None,
                targets: Vec::new(),
                target_req_indices: Vec::new(),
            },
        )
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
            .map_err(|error| AgentError(error.to_string()))?;
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
        let player = action.player();
        // Execute the production prefix and forced spends on a copy, stopping
        // before cost completion can advance the spell or ability.
        let mut position = std::borrow::Cow::Borrowed(self);
        let mut sources = Vec::new();
        let mut resolution = None;
        // Whether convoke, improvise or delve may pay where mana cannot.
        let mut helped = false;
        // A deliberate tap in a payment previews its own production first.
        let offered_tap = match action {
            Action::TapForMana { permanent, .. } => Some(AutomaticTap::fixed(*permanent)),
            _ => None,
        };
        let cost = match action {
            Action::CastSpell {
                object,
                route,
                profile,
                ..
            } => {
                let cast = if let Some(cast) =
                    self.state.suspended_decision.as_ref().and_then(|s| {
                        s.decision
                            .resolution_cast_choices(&self.state[*object], *profile)
                    }) {
                    cast
                } else {
                    CastChoices::from_route(
                        self.selected_play_route(player, *object, *route)?,
                        *profile,
                    )
                };
                let card = cast.spell(&self.state[*object]);
                if self.cast_needs_x_announcement(*object, cast) {
                    return Ok(PaymentPreview::message(
                        PaymentPreviewState::NeedsChoice,
                        "Choose X or replicate count before previewing payment.",
                    ));
                }
                let Some(cast) = self.fixed_cast_cost_choices(player, *object, cast) else {
                    return Ok(PaymentPreview::message(
                        PaymentPreviewState::NeedsChoice,
                        "Choose the optional or alternative cost before previewing payment.",
                    ));
                };
                let owed = self.spell_owed(player, *object, cast);
                if will_owe_a_choice(&owed) {
                    return Ok(PaymentPreview::needs_additional_cost());
                }
                helped = keyword_help(&owed);
                if card.cost_reduction_if_targets().is_some() {
                    return Ok(PaymentPreview::message(
                        PaymentPreviewState::NeedsChoice,
                        "Choose targets before previewing this target-dependent cost.",
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
                    let card = cast.spell(&self.state[*object]);
                    let owed = self.spell_owed(player, *object, *cast);
                    if will_owe_a_choice(&owed) {
                        return Ok(PaymentPreview::needs_additional_cost());
                    }
                    helped = keyword_help(&owed);
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
                            "Choose the remaining targets before previewing this target-dependent cost."));
                    }
                    let mut targets = chosen.clone();
                    targets.push(*target);
                    self.effective_spell_cost(player, *object, *cast, Some(&targets))
                }
                _ => None,
            },
            Action::PayCost { .. } | Action::TapForMana { .. }
                if self.state.suspended_decision.is_some() =>
            {
                let suspended = position
                    .to_mut()
                    .state
                    .suspended_decision_mut()
                    .take()
                    .unwrap();
                // A cost paid in life has no mana route: `None` answers
                // `not_applicable` below, as for any other decision.
                match suspended.decision {
                    Decision::Payment(payment) => match payment.owed.clone() {
                        CostComponent::Mana { cost, .. } => {
                            sources.extend(
                                offered_tap.map(|tap| position.to_mut().preview_tap(player, tap)),
                            );
                            resolution = Some((payment, suspended.frame));
                            Some(cost)
                        }
                        _ => None,
                    },
                    _ => None,
                }
            }
            Action::PayCost { .. } | Action::TapForMana { .. } => match &self.pending_choice {
                Some(PendingChoice::Pay { payment, .. }) if owes_a_choice(&payment.owed) => {
                    return Ok(PaymentPreview::needs_additional_cost());
                }
                Some(PendingChoice::Pay { payment, .. }) => {
                    helped = keyword_help(&payment.owed);
                    sources
                        .extend(offered_tap.map(|tap| position.to_mut().preview_tap(player, tap)));
                    if payment.waterbends() {
                        position.to_mut().pending_choice = None;
                    }
                    Some(payment.mana().0.clone())
                }
                Some(_) => {
                    return Ok(PaymentPreview::message(
                        PaymentPreviewState::NeedsChoice,
                        "Finish announcing costs and targets before previewing payment.",
                    ))
                }
                None => None,
            },
            Action::ActivateAbility {
                permanent,
                ability_index,
                ..
            } => {
                let ability = &self.characteristics(*permanent).activated_abilities[*ability_index];
                if ability.mana_choices().is_some()
                    || ability.mana_cost.mana_value == 0 && ability.mana_cost.x == 0
                {
                    return Ok(PaymentPreview::message(
                        PaymentPreviewState::NotApplicable,
                        "No automatic activation fee preview.",
                    ));
                }
                let operation = Operation::Activation {
                    source: ActivationSource::Permanent(*permanent),
                    ability_index: *ability_index,
                    targets: Vec::new(),
                    x: 0,
                };
                let owed = self.cost_components(player, &operation);
                if ability.mana_cost.x > 0 || will_owe_a_choice(&owed) {
                    return Ok(PaymentPreview::message(
                        PaymentPreviewState::NeedsChoice,
                        "Choose the activation's additional costs before previewing payment.",
                    ));
                }
                let cost = ability.mana_cost.clone();
                position.to_mut().pending_choice = Some(PendingChoice::Pay {
                    player,
                    payment: Box::new(Payment {
                        owed,
                        operation,
                        receipt: Default::default(),
                    }),
                });
                Some(cost)
            }
            _ => None,
        };
        let Some(cost) = cost else {
            return Ok(PaymentPreview::message(
                PaymentPreviewState::NotApplicable,
                "This action does not have an automatic payment preview.",
            ));
        };
        let Ok(taps) = position.automatic_taps(player, &cost) else {
            let mut preview = if helped {
                PaymentPreview::message(
                    PaymentPreviewState::NeedsChoice,
                    "Mana alone cannot pay this cost. Choose what to tap or exile.",
                )
            } else {
                PaymentPreview::message(
                    PaymentPreviewState::Unavailable,
                    "No automatic route. Use a legal mana activation and choose its output first.",
                )
            };
            preview.cost = Some(cost_description(&cost));
            return Ok(preview);
        };
        let position = position.to_mut();
        sources.extend(
            taps.into_iter()
                .map(|tap| position.preview_tap(player, tap)),
        );
        let pool = position.pooled_mana(player);
        if let Some((payment, frame)) = resolution {
            *position.state.suspended_decision_mut() = Some(SuspendedResolution {
                frame,
                decision: Decision::ManaSpend {
                    player,
                    cost: cost.clone(),
                    spent: Default::default(),
                    completion: Box::new(SpendCompletion::Resolution(payment)),
                },
            });
        } else {
            position.pending_choice = None;
        }
        let mut remaining = cost.clone();
        let final_pool = loop {
            if remaining.mana_value == 0 {
                break Some(position.pooled_mana(player));
            }
            let choices = position.mana_spend_choices(player, &remaining);
            let [only] = choices.as_slice() else {
                break None;
            };
            position.spend_mana_unit(player, &only.unit, &remaining);
            remaining = only.remaining.clone();
        };
        Ok(PaymentPreview {
            state: PaymentPreviewState::Exact,
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
    /// Run only the selected production on the isolated preview position. Its
    /// pool delta includes triggered production, spend riders and expiry.
    fn preview_tap(&mut self, player: PlayerId, tap: AutomaticTap) -> PaymentSourcePreview {
        let permanent = tap.permanent;
        let card = self.state.permanents[permanent]
            .as_ref()
            .expect("planned source");
        let source = SubjectRef::Object {
            id: ObjectRenderId {
                entity: card.id.0,
                incarnation: card.object.generation(),
            },
        };
        let label = self.characteristics(permanent).name.clone();
        let before = self.pooled_mana(player);
        self.perform_automatic_tap(player, tap);
        let after = self.pooled_mana(player);
        let mut output = Mana {
            mana: std::array::from_fn(|i| after.mana[i] - before.mana[i]),
            ..Mana::default()
        };
        for (class, amounts) in after.classes() {
            let previous = before.class_of(class);
            output.add(&Mana::of_class(
                class.clone(),
                std::array::from_fn(|i| amounts[i] - previous[i]),
            ));
        }
        PaymentSourcePreview {
            source,
            label,
            output: PreviewPool::from(&output),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        agent::observation::Observation, search_state::witness, state::player::PlayerConfig,
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
        game.keep_opening_hands();
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
        let frame = game.semantic_decision_frame().unwrap();
        assert!(game
            .payment_preview(
                PlayerId(1),
                frame.revision,
                &frame.fingerprint,
                cast_index(&game)
            )
            .is_err());
        game.produce_mana(PlayerId(0), &ManaCost::parse("U"))
            .unwrap();
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

    /// A convoke spell previews its mana route like any other cost. Short of
    /// mana the convoker is the player's pick, never the preview's.
    #[test]
    fn payment_preview_pays_a_convoke_cost_with_mana_or_asks_for_the_pick() {
        for (plains, state) in [
            (true, PaymentPreviewState::Exact),
            (false, PaymentPreviewState::NeedsChoice),
        ] {
            let lands = ["Mountain", "Forest", "Plains"];
            let mut deck: BTreeMap<String, usize> =
                lands.iter().map(|land| (land.to_string(), 10)).collect();
            deck.insert("Guardian of Vitu-Ghazi".into(), 10);
            let mut game = Game::new(
                vec![
                    PlayerConfig::new("payer", deck),
                    PlayerConfig::new("other", BTreeMap::from([("Plains".into(), 40)])),
                ],
                165,
                true,
            );
            game.keep_opening_hands();
            let payer = PlayerId(0);
            game.scenario_force_card_in_hand(payer, "Guardian of Vitu-Ghazi")
                .unwrap();
            game.scenario_force_battlefield(payer, "Guardian of Vitu-Ghazi", true)
                .unwrap();
            for land in lands.iter().take(if plains { 3 } else { 2 }) {
                for _ in 0..if *land == "Mountain" { 6 } else { 1 } {
                    game.scenario_force_battlefield(payer, land, true).unwrap();
                }
            }
            game.scenario_refresh_priority().unwrap();
            let cast = game
                .action_space()
                .unwrap()
                .actions
                .iter()
                .position(|action| matches!(action, Action::CastSpell { .. }))
                .unwrap();
            let before = witness(&game);
            let result = preview(&game, cast);
            assert_eq!(before, witness(&game));
            assert_eq!(result.state, state);
            assert_eq!(result.cost.as_deref(), Some("{6}{W}{G}"));
            if plains {
                assert_eq!(result.sources.len(), 8);
                assert_eq!(result.pool_after_spending.unwrap().amounts, [0; 6]);
            } else {
                assert_eq!(
                    result.explanation,
                    "Mana alone cannot pay this cost. Choose what to tap or exile."
                );
                assert!(result.sources.is_empty());
            }
        }
    }

    #[test]
    fn payment_preview_does_not_guess_a_spending_choice() {
        let mut game = preview_game();
        game.state
            .mana_pool_mut(PlayerId(0))
            .add(&Mana::parse("UR"), None);
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
        game.step(pay).unwrap();
        let choices = &game.action_space().unwrap().actions;
        assert_eq!(choices.len(), 2);
        for (index, choice) in choices.iter().enumerate() {
            assert!(matches!(choice, Action::ChooseMode { .. }));
            assert_eq!(
                preview(&game, index).state,
                PaymentPreviewState::NotApplicable
            );
        }
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
    fn payment_preview_matches_committed_restricted_mana_and_spend_effect() {
        use crate::state::{
            mana::{ManaClass, ManaRestriction, SpendEffect},
            stack_object::StackObject,
        };
        let payer = PlayerId(0);
        let mut game = Game::new(
            vec![
                PlayerConfig::new(
                    "payer",
                    BTreeMap::from([("Island".into(), 24), ("Spectral Sailor".into(), 16)]),
                ),
                PlayerConfig::new("other", BTreeMap::from([("Plains".into(), 40)])),
            ],
            165,
            true,
        );
        game.keep_opening_hands();
        game.scenario_force_card_in_hand(payer, "Spectral Sailor")
            .unwrap();
        let blue = [0, 1, 0, 0, 0, 0];
        game.state.mana_pool_mut(payer).add(
            &Mana::restricted_to(ManaRestriction::spell(["Lesson"]), blue),
            None,
        );
        game.state.combat_mana_pool_mut(payer).add(
            &Mana::of_class(
                ManaClass {
                    restriction: Some(ManaRestriction::spell(["Pirate"])),
                    effect: Some(SpendEffect::CreatureEntersWithPlusOneCounter {
                        without_subtype: "Human".into(),
                    }),
                    combat: false,
                },
                blue,
            ),
            None,
        );
        game.scenario_refresh_priority().unwrap();
        let index = game
            .action_space()
            .unwrap()
            .actions
            .iter()
            .position(|action| matches!(action, Action::CastSpell { .. }))
            .unwrap();
        let before = witness(&game);
        let proposal = preview(&game, index);
        assert_eq!(witness(&game), before);
        assert_eq!(proposal.state, PaymentPreviewState::Exact);
        assert!(proposal.sources.is_empty());
        let production = proposal.pool_after_production.unwrap();
        assert_eq!(production.amounts, [0, 2, 0, 0, 0, 0]);
        assert!(production.classes.iter().any(|class| class.combat
            && class.effect.is_some()
            && class
                .restriction
                .as_ref()
                .is_some_and(|s| s.contains("Pirate"))));

        // Inspect the committed spell before priority passes can resolve it.
        let cast = game.action_space().unwrap().actions[index].clone();
        game.execute_action(&cast).unwrap();
        let Some(StackObject::Spell(spell)) = game.state.stack_objects.last() else {
            panic!("expected the paid spell on the stack");
        };
        assert_eq!(spell.spent.amounts(), blue);
        assert_eq!(spell.spent.entry_counters(), 1);
        let leftover = proposal.pool_after_spending.unwrap();
        assert_eq!(leftover.amounts, blue.map(u16::from));
        assert_eq!(
            serde_json::to_value(leftover).unwrap(),
            serde_json::to_value(PreviewPool::from(&game.pooled_mana(payer))).unwrap(),
        );
    }

    #[test]
    fn payment_preview_keeps_restrictions_and_combat_expiry() {
        use crate::state::mana::ManaRestriction;
        let mut game = preview_game();
        game.state
            .mana_pool_mut(PlayerId(0))
            .add(&Mana::parse("U"), None);
        game.state.mana_pool_mut(PlayerId(0)).add(
            &Mana::restricted_to(ManaRestriction::spell(["Lesson"]), [0, 1, 0, 0, 0, 0]),
            None,
        );
        let rider = crate::state::mana::SpendEffect::CreatureEntersWithPlusOneCounter {
            without_subtype: "Human".into(),
        };
        game.state
            .combat_mana_pool_mut(PlayerId(0))
            .add(&Mana::of_class(rider.into(), [0, 0, 0, 1, 0, 0]), None);
        let result = preview(&game, cast_index(&game));
        assert!(result.sources.is_empty());
        let final_pool = result.pool_after_spending.unwrap();
        assert_eq!(final_pool.amounts, [0, 1, 0, 1, 0, 0]);
        assert!(final_pool
            .classes
            .iter()
            .any(|class| class.combat && class.effect.is_some()));
        assert!(final_pool.classes.iter().any(|class| class
            .restriction
            .as_ref()
            .is_some_and(|r| r.contains("Lesson"))));
    }
}
