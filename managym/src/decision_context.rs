//! Viewer-safe explanation of the existing one-action decision seam.
//!
//! `Game::decision_context` reads pending native continuations; it never stores
//! a second payment transaction or changes legality. These facts summarize the
//! acting player's already-observed actions and current legal decision. The
//! tensor encoder and frozen semantic observation/digest contracts stay intact.

use crate::{
    experience::{
        DecisionContextView, DecisionObjectView, DecisionTargetView, ObjectRenderId,
        OfferSubjectView, RequiredNullable, SubjectRef,
    },
    flow::{
        decision::Decision,
        game::{Game, Payment, PendingChoice},
    },
    state::game_object::{CardId, PlayerId, Target},
};

impl Game {
    fn decision_card(&self, card: CardId) -> Option<DecisionObjectView> {
        let reference = self.current_object_ref(card)?;
        let display_id = self
            .state
            .card_to_permanent
            .get(card.0)
            .and_then(|permanent| *permanent)
            .and_then(|permanent| self.state.permanents[permanent].as_ref())
            .map_or(self.state.cards[card].id.0, |permanent| permanent.id.0);
        let card = &self.state.cards[card];
        Some(DecisionObjectView {
            entity_id: reference.entity.0 as u32,
            subject: SubjectRef::Object {
                id: ObjectRenderId {
                    entity: display_id,
                    incarnation: reference.incarnation.0,
                },
            },
            display_id,
            label: card.name.clone(),
        })
    }

    fn decision_target(&self, target: &Target) -> Option<DecisionObjectView> {
        match target {
            Target::Player(player) => Some(DecisionObjectView {
                entity_id: self.state.players[player.0].id.0,
                subject: SubjectRef::Player {
                    id: crate::experience::PlayerId(player.0 as u8),
                },
                display_id: self.state.players[player.0].id.0,
                label: format!("Seat {}", player.0 + 1),
            }),
            Target::Permanent(permanent) => {
                let permanent = self.state.permanents[*permanent].as_ref()?;
                let mut view = self.decision_card(permanent.card)?;
                view.display_id = permanent.id.0;
                if let SubjectRef::Object { ref mut id } = view.subject {
                    id.entity = permanent.id.0;
                }
                Some(view)
            }
            Target::StackSpell(card) => self.decision_card(*card),
            Target::GraveyardCard(reference) => {
                let mut view = self.decision_card(CardId::from(reference.entity))?;
                // Keep the announced incarnation even if the card later moved.
                view.subject = SubjectRef::Object {
                    id: ObjectRenderId {
                        entity: view.display_id,
                        incarnation: reference.incarnation.0,
                    },
                };
                Some(view)
            }
        }
    }

    /// Exact subjects for each one-action offer, in native offer order.
    /// Only already-visible focus objects are eligible for this projection.
    pub fn decision_offer_subjects(&self, viewer: PlayerId) -> Vec<Vec<OfferSubjectView>> {
        use crate::agent::{action::Action, observation::Observation};
        let Some(space) = self.action_space() else {
            return Vec::new();
        };
        if space.player != Some(viewer) {
            return Vec::new();
        }
        let observation = Observation::for_player(self, viewer);
        space
            .actions
            .iter()
            .map(|action| {
                Observation::action_focus(self, action)
                    .iter()
                    .enumerate()
                    .filter_map(|(index, id)| {
                        let object = if let Some(player) = self
                            .state
                            .players
                            .iter()
                            .position(|player| player.id == *id)
                        {
                            self.decision_target(&Target::Player(PlayerId(player)))
                        } else if let Some((permanent, _)) = self
                            .state
                            .permanents
                            .iter()
                            .enumerate()
                            .find(|(_, permanent)| {
                                permanent
                                    .as_ref()
                                    .is_some_and(|permanent| permanent.id == *id)
                            })
                        {
                            self.decision_target(&Target::Permanent(
                                crate::state::game_object::PermanentId(permanent),
                            ))
                        } else {
                            observation
                                .agent_cards
                                .iter()
                                .chain(observation.opponent_cards.iter())
                                .find(|card| card.id == id.0 as i32)
                                .and_then(|_| {
                                    self.state
                                        .cards
                                        .iter()
                                        .position(|card| card.id == *id)
                                        .and_then(|card| self.decision_card(CardId(card)))
                                })
                        }?;
                        let role = match action {
                            Action::PlayLand { .. }
                            | Action::CastSpell { .. }
                            | Action::ActivateAbility { .. } => "source",
                            Action::DeclareBlocker { .. } if index > 0 => "target",
                            Action::ChooseTarget { .. } => "target",
                            _ => "choice",
                        };
                        Some(OfferSubjectView {
                            role: role.into(),
                            object,
                        })
                    })
                    .collect()
            })
            .collect()
    }

    /// Public event identities captured before a transition so a permanent's
    /// later departure cannot turn a committed damage target into a guess.
    pub fn presentation_subjects(&self) -> std::collections::BTreeMap<String, SubjectRef> {
        let mut subjects = std::collections::BTreeMap::new();
        for (index, _) in self.state.players.iter().enumerate() {
            subjects.insert(
                format!("player:{index}"),
                SubjectRef::Player {
                    id: crate::experience::PlayerId(index as u8),
                },
            );
        }
        for (index, permanent) in self.state.permanents.iter().enumerate() {
            if let Some(permanent) = permanent {
                if let Some(reference) =
                    self.permanent_object_ref(crate::state::game_object::PermanentId(index))
                {
                    subjects.insert(
                        format!("permanent:{index}"),
                        SubjectRef::Object {
                            id: ObjectRenderId {
                                entity: permanent.id.0,
                                incarnation: reference.incarnation.0,
                            },
                        },
                    );
                }
            }
        }
        subjects
    }

    /// Only the decision owner receives announcement/private continuation
    /// details. Other viewers receive the public owner/kind and committed facts.
    pub fn decision_context(&self, viewer: PlayerId) -> Option<DecisionContextView> {
        let space = self.action_space()?;
        if self.is_game_over() {
            return None;
        }
        let actor = space.player?;
        let mut view = DecisionContextView {
            actor: actor.0 as u8,
            stage: "priority".into(),
            instruction: "Choose an explicit legal offer to commit one decision.".into(),
            source: RequiredNullable(None),
            targets: Vec::new(),
            reserved_sacrifice: RequiredNullable(None),
            mana_cost: RequiredNullable(None),
            chosen_mode: RequiredNullable(None),
            recent_facts: self
                .state
                .events
                .iter()
                .skip(self.state.events.len().saturating_sub(64))
                .filter_map(|event| {
                    use crate::flow::event::GameEvent;
                    match event {
                        GameEvent::PermanentSacrificed { card, controller } => Some(format!(
                            "Seat {} sacrificed {}.",
                            controller.0 + 1,
                            self.state.cards[*card].name
                        )),
                        GameEvent::SpellCast { card, .. } => Some(format!(
                            "{} was cast; its announced costs are committed.",
                            self.state.cards[*card].name
                        )),
                        GameEvent::SpellResolved { card } => Some(format!(
                            "{} finished resolving.",
                            self.state.cards[*card].name
                        )),
                        GameEvent::SpellCountered { card, .. } => {
                            Some(format!("{} was countered.", self.state.cards[*card].name))
                        }
                        _ => None,
                    }
                })
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .take(12)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect(),
        };
        if actor != viewer {
            view.stage = format!("{:?}", space.kind).to_lowercase();
            view.instruction = format!("Waiting for Seat {}. Private continuation details are available only to that player.", actor.0 + 1);
            return Some(view);
        }
        let mut payment = None;
        match &self.pending_choice {
            Some(PendingChoice::ManaPayment {
                payment: pending,
                sacrifice,
                alternative,
                ..
            }) => {
                view.stage = "payment".into();
                view.instruction = "Produce mana or finish the announced payment. Reserved sacrifices are not paid yet; activated mana abilities commit their own costs immediately.".into();
                payment = Some(pending.as_ref());
                view.reserved_sacrifice = RequiredNullable(
                    sacrifice.and_then(|id| self.decision_target(&Target::Permanent(id))),
                );
                let cost = self.payment_mana_cost(viewer, pending, alternative.as_ref());
                let mut symbols = String::new();
                if cost.cost[6] > 0 {
                    symbols.push_str(&format!("{{{}}}", cost.cost[6]));
                }
                for (index, symbol) in ["W", "U", "B", "R", "G", "C"].iter().enumerate() {
                    for _ in 0..cost.cost[index] {
                        symbols.push_str(&format!("{{{symbol}}}"));
                    }
                }
                for pip in cost.hybrid {
                    symbols.push_str(&format!("{{{}/{}}}", pip[0].symbol(), pip[1].symbol()));
                }
                if symbols.is_empty() {
                    symbols.push_str("{0}");
                }
                view.mana_cost = RequiredNullable(Some(symbols));
            }
            Some(PendingChoice::SacrificePayment {
                payment: pending, ..
            }) => {
                view.stage = "additional_cost".into();
                view.instruction = "Choose an additional cost. A sacrifice selection is reserved until the announced payment completes.".into();
                payment = Some(pending.as_ref());
            }
            Some(PendingChoice::ChooseTargets {
                card,
                chosen,
                chosen_req_indices,
                mode,
                ..
            }) => {
                view.stage = "targets".into();
                view.instruction = "Choose a target for the current clause. Previously selected targets remain bound to their clauses.".into();
                view.source = RequiredNullable(self.decision_card(*card));
                view.targets = chosen
                    .iter()
                    .enumerate()
                    .filter_map(|(index, target)| {
                        self.decision_target(target)
                            .map(|object| DecisionTargetView {
                                object,
                                clause: chosen_req_indices.get(index).copied().unwrap_or(0) as u32,
                            })
                    })
                    .collect();
                view.chosen_mode = RequiredNullable(mode.map(|mode| mode as u32));
            }
            Some(PendingChoice::SpellMode { card, .. }) => {
                view.stage = "mode".into();
                view.instruction = "Choose a spell mode.".into();
                view.source = RequiredNullable(self.decision_card(*card));
            }
            Some(PendingChoice::KickerChoice { card, .. }) => {
                view.stage = "additional_cost".into();
                view.instruction = "Choose whether to include the kicker cost.".into();
                view.source = RequiredNullable(self.decision_card(*card));
            }
            Some(PendingChoice::ActivatedTarget { permanent, .. }) => {
                view.stage = "targets".into();
                view.instruction = "Choose the ability's target.".into();
                view.source =
                    RequiredNullable(self.decision_target(&Target::Permanent(*permanent)));
            }
            Some(PendingChoice::Waterbend {
                permanent,
                remaining_generic,
                ..
            }) => {
                view.stage = "payment".into();
                view.source =
                    RequiredNullable(self.decision_target(&Target::Permanent(*permanent)));
                view.instruction = format!("Pay the activation cost; {remaining_generic} generic remains before colored payment. Submitted taps are committed.");
            }
            None => {
                use crate::agent::action::ActionSpaceKind;
                let (stage, instruction) = match space.kind {
                    ActionSpaceKind::DeclareAttacker => ("declare_attackers", "Choose each attack declaration, including the legal nonparticipation offers. Submitted declarations are committed."),
                    ActionSpaceKind::DeclareBlocker => ("declare_blockers", "Choose each block declaration, including the legal nonparticipation offers. Submitted declarations are committed."),
                    _ => ("priority", "Choose an explicit legal offer to commit one decision."),
                };
                view.stage = stage.into();
                view.instruction = instruction.into();
            }
        }
        if let Some(payment) = payment {
            let (card, targets, clauses, mode) = match payment {
                Payment::Spell {
                    card,
                    targets,
                    mode,
                    target_req_indices,
                    ..
                } => (*card, targets, target_req_indices.clone(), *mode),
                Payment::Activation {
                    permanent, targets, ..
                } => (
                    self.state.permanents[*permanent].as_ref()?.card,
                    targets,
                    vec![0; targets.len()],
                    None,
                ),
            };
            view.source = RequiredNullable(self.decision_card(card));
            view.targets = targets
                .iter()
                .enumerate()
                .filter_map(|(index, target)| {
                    self.decision_target(target)
                        .map(|object| DecisionTargetView {
                            object,
                            clause: clauses.get(index).copied().unwrap_or(0) as u32,
                        })
                })
                .collect();
            view.chosen_mode = RequiredNullable(mode.map(|mode| mode as u32));
        }
        if let Some(suspended) = &self.state.suspended_decision {
            // A mana ability may interrupt an announced payment. Keep the
            // payment's source/reservation while explaining the actual choice.
            if view.source.0.is_none() {
                view.source = RequiredNullable(
                    suspended
                        .frame
                        .source
                        .and_then(|card| self.decision_card(card)),
                );
            }
            let (stage, instruction) = match &suspended.decision {
                Decision::ManaUpfront { .. } => ("mana_payment", "Pay the mana ability activation cost before choosing its output."),
                Decision::ManaChoice { .. } => ("mana_color", "Choose the mana to add. The mana ability's activation costs have already committed; this does not open priority."),
                Decision::Sacrifice { .. } => ("resolution_sacrifice", "Choose a permanent to sacrifice as a resolving effect, not as an announcement cost."),
                Decision::SearchBasicLand { .. } => ("private_search", "Inspect the permitted library candidates. Choose a land or the legal failure-to-find offer; no library order is disclosed."),
                Decision::Learn { .. } => ("learn", "Take a Lesson, discard and draw, or decline Learn. Navigation does not commit a choice."),
                Decision::LookAndSelect { .. } => ("private_inspection", "Choose from the inspected cards using the complete legal offers."),
                Decision::Scry { .. } => ("scry", "Choose the placement of the inspected cards."),
                Decision::Payment(_) => ("resolution_payment", "Choose a legal payment for the resolving effect, or decline when offered. Submitted costs remain committed."),
                Decision::Modal { .. } => ("resolution_mode", "Choose a mode for the resolving effect."),
                Decision::Discard { .. } | Decision::OptionalDiscardDraw { .. } => ("discard", "Choose a card to discard, or decline when offered."),
                Decision::SelectMilled { .. } => ("select_milled", "Choose from the milled cards, or decline when offered."),
                Decision::LegendRule { .. } => ("legend_rule", "Choose which legendary permanent to keep."),
            };
            view.stage = stage.into();
            view.instruction = instruction.into();
        }
        Some(view)
    }
}
