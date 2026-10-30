//! Viewer-safe explanation of the existing one-action decision seam.
//!
//! `Game::decision_context` reads pending native continuations; it never stores
//! a second payment transaction or changes legality. These facts summarize the
//! acting player's already-observed actions and current legal decision. The
//! typed continuation projection is shared with tensor and text inputs; the
//! frozen semantic observation/digest contract stays intact.

use crate::{
    agent::continuation::ContinuationFrame,
    experience::{
        DecisionContextView, DecisionObjectView, DecisionTargetView, ObjectRenderId,
        OfferSubjectView, RequiredNullable, SubjectRef,
    },
    flow::{
        decision::Decision,
        game::{Game, PendingChoice},
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
                            | Action::ActivateAbility { .. }
                            | Action::ActivateCardAbility { .. }
                            | Action::Foretell { .. } => "source",
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
        let frames = self.continuation_frames(viewer);
        let mut view = self.decision_context_from_frames(viewer, &frames)?;
        view.recent_facts = self
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
            .collect();
        let excess = view.recent_facts.len().saturating_sub(12);
        view.recent_facts.drain(..excess);
        Some(view)
    }

    /// Project current facts without the presentation-only event suffix. Capture
    /// supplies its observation's frames and separately admits policy history.
    pub(crate) fn decision_context_from_frames(
        &self,
        viewer: PlayerId,
        frames: &[ContinuationFrame],
    ) -> Option<DecisionContextView> {
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
            recent_facts: Vec::new(),
        };
        if actor != viewer {
            view.stage = format!("{:?}", space.kind).to_lowercase();
            view.instruction = format!("Waiting for Seat {}. Private continuation details are available only to that player.", actor.0 + 1);
            return Some(view);
        }
        // Presentation consumes the same typed current facts as the native
        // observation and policy tensors. Only labels and instructions live here.
        if let Some(frame) = frames
            .iter()
            .find(|frame| frame.kind == 8)
            .or_else(|| frames.iter().find(|frame| frame.kind == 4))
            .or_else(|| frames.iter().find(|frame| frame.kind == 5))
        {
            let object = |reference: crate::agent::continuation::VisibleReference| {
                if let Some(player) = reference.player {
                    return self.decision_target(&Target::Player(PlayerId(player)));
                }
                let display_id = reference.object? as u32;
                self.state.cards.iter().enumerate().find_map(|(index, _)| {
                    self.decision_card(CardId(index))
                        .filter(|view| view.display_id == display_id)
                })
            };
            view.source = RequiredNullable(object(frame.source));
            view.targets = frame
                .targets
                .iter()
                .filter_map(|target| {
                    object(target.reference).map(|object| DecisionTargetView {
                        object,
                        clause: target.clause as u32,
                    })
                })
                .collect();
            view.reserved_sacrifice = RequiredNullable(frame.sacrifice.and_then(object));
            view.chosen_mode = RequiredNullable(frame.mode.map(|mode| mode as u32));
            if let Some(cost) = frame.cost {
                let mut symbols = String::new();
                if cost[6] > 0 {
                    symbols.push_str(&format!("{{{}}}", cost[6]));
                }
                for (amount, symbol) in cost[..6].iter().zip(["W", "U", "B", "R", "G", "C"]) {
                    for _ in 0..*amount {
                        symbols.push_str(&format!("{{{symbol}}}"));
                    }
                }
                for pair in &frame.hybrid {
                    let colors = ["W", "U", "B", "R", "G", "C"];
                    symbols.push_str(&format!(
                        "{{{}/{}}}",
                        colors[pair[0] as usize], colors[pair[1] as usize]
                    ));
                }
                if symbols.is_empty() {
                    symbols.push_str("{0}");
                }
                view.mana_cost = RequiredNullable(Some(symbols));
            }
        }
        let (stage, instruction) = match &self.pending_choice {
            Some(PendingChoice::ManaPayment { .. } | PendingChoice::Waterbend { .. }) =>
                ("payment", "Produce mana or finish the announced payment. Reserved sacrifices are not paid yet; activated mana abilities commit their own costs immediately."),
            Some(PendingChoice::DiscardPayment { .. }) =>
                ("additional_cost", "Choose a card to discard as an activation cost."),
            Some(PendingChoice::SacrificePayment { .. }) =>
                ("additional_cost", "Choose an additional cost. A sacrifice selection is reserved until the announced payment completes."),
            Some(PendingChoice::ChooseTargets { .. } | PendingChoice::ActivatedTarget { .. }) =>
                ("targets", "Choose a target for the current clause. Previously selected targets remain bound to their clauses."),
            Some(PendingChoice::Crew { .. }) =>
                ("additional_cost", "Choose a creature to tap for crew. Each tap commits immediately; the cost completes once the tapped creatures' total power reaches the crew number."),
            Some(PendingChoice::SpellMode { .. }) => ("mode", "Choose a spell mode."),
            Some(PendingChoice::ActivationX { .. }) => ("announcement", "Choose the value of X. Each choice is a value the announced total cost can pay."),
            Some(PendingChoice::KickerChoice { .. }) => ("additional_cost", "Choose whether to include the kicker cost."),
            Some(PendingChoice::AnnounceX { .. }) => ("mode", "Announce the value of X; each offer's mode index is that value."),
            None => match space.kind {
                crate::agent::action::ActionSpaceKind::DeclareAttacker => ("declare_attackers", "Choose each attack declaration, including the legal nonparticipation offers. Submitted declarations are committed."),
                crate::agent::action::ActionSpaceKind::DeclareBlocker => ("declare_blockers", "Choose each block declaration, including the legal nonparticipation offers. Submitted declarations are committed."),
                _ => ("priority", "Choose an explicit legal offer to commit one decision."),
            },
        };
        view.stage = stage.into();
        view.instruction = instruction.into();
        if let Some(suspended) = &self.state.suspended_decision {
            let (stage, instruction) = match &suspended.decision {
                Decision::ManaUpfront { .. } => ("mana_payment", "Pay the mana ability activation cost before choosing its output."),
                Decision::ManaChoice { .. } => ("mana_color", "Choose the mana to add. The mana ability's activation costs have already committed; this does not open priority."),
                Decision::Sacrifice { .. } => ("resolution_sacrifice", "Choose a permanent to sacrifice as a resolving effect, not as an announcement cost."),
                Decision::SearchBasicLand { .. } | Decision::SearchLibraryToHand { .. } => ("private_search", "Inspect the permitted library candidates. Choose a card or the legal failure-to-find offer; no library order is disclosed."),
                Decision::Learn { .. } => ("learn", "Take a Lesson, discard and draw, or decline Learn. Navigation does not commit a choice."),
                Decision::LookAndSelect { .. } => ("private_inspection", "Choose from the inspected cards using the complete legal offers."),
                Decision::Scry { .. } => ("scry", "Choose the placement of the inspected cards."),
                Decision::Surveil { .. } => ("surveil", "Choose inspected cards to put into your graveyard; declining keeps the rest on top."),
                Decision::Payment(_) => ("resolution_payment", "Choose a legal payment for the resolving effect, or decline when offered. Submitted costs remain committed."),
                Decision::Modal { .. } => ("resolution_mode", "Choose a mode for the resolving effect."),
                Decision::Discard { .. } | Decision::OptionalDiscardDraw { .. } => ("discard", "Choose a card to discard, or decline when offered."),
                Decision::SelectMilled { .. } => ("select_milled", "Choose from the milled cards, or decline when offered."),
                Decision::LegendRule { .. } => ("legend_rule", "Choose which legendary permanent to keep."),
                Decision::SearchDistinctLands { .. } => ("private_search", "Inspect the permitted library candidates. Choose a land with a new name, or stop searching; chosen cards are revealed."),
                Decision::DiscardFromRevealedHand { .. } => ("revealed_hand", "Choose a card from the revealed hand for its owner to discard."),
                Decision::CastWithoutPaying { .. } => ("free_cast", "Choose a spell to cast without paying its mana cost, or decline."),
                Decision::DistributeCounters { .. } => ("distribute_counters", "Place the next counter on a tapped creature an opponent controls; each placement is one choice."),
                Decision::ExileLinked { .. } => ("linked_exile", "Choose a card to exile; the resolving ability's controller may later cast it as that ability allows."),
                Decision::OpponentSplitsRevealed { .. } => ("split_revealed", "Choose revealed cards for their owner's graveyard; the rest enter the battlefield tapped."),
            };
            view.stage = stage.into();
            view.instruction = instruction.into();
        }
        if let Some(placement) = &self.state.trigger_placement {
            view.stage = if placement.placing.is_some() {
                "trigger_targets"
            } else {
                "order_triggers"
            }
            .into();
            view.instruction = if let Some(trigger) = &placement.placing {
                let description = crate::agent::ability_description::trigger_description(
                    &self.state.cards[trigger.source_card],
                    trigger.ability_index,
                    trigger.inline_effects.as_deref(),
                );
                format!("{description}. Choose targets for this ability before placing the next one. Earlier placements and paid costs remain committed.")
            } else {
                "Choose the next ability to put on the stack. Later placements resolve first. Each selection commits immediately.".into()
            };
        }
        Some(view)
    }
}
