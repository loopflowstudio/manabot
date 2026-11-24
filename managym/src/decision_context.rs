//! Viewer-safe explanation of the existing one-action decision seam.
//!
//! `Game::decision_context` reads pending native continuations; it never stores
//! a second payment transaction or changes legality. These facts summarize the
//! acting player's already-observed actions and current legal decision. The
//! typed continuation projection is shared with tensor and text inputs; the
//! frozen semantic observation/digest contract stays intact.

use crate::state::ability::LibraryEnd;
use crate::{
    agent::{
        continuation::ContinuationFrame,
        effect_execution::{contribution_phrase, spent_mana_phrase},
        observation::spell_profile_code,
    },
    experience::{
        DecisionContextView, DecisionObjectView, DecisionTargetView, ObjectRenderId,
        OfferSubjectView, PaidView, RequiredNullable, SubjectRef,
    },
    flow::{
        action::cost_label,
        decision::Decision,
        game::{Game, PendingChoice},
        payment::{CostComponent, Help, PaymentProgress},
    },
    state::{
        card::CastOption,
        game_object::{ObjectId, PlayerId, Target},
        payment::{PaidWith, PaymentReceipt, Toward},
    },
};

impl Game {
    fn decision_object(&self, object: ObjectId) -> Option<DecisionObjectView> {
        self.state.live(object)?;
        let display_id = self
            .state
            .object_to_permanent
            .get(object.slot())
            .and_then(|permanent| *permanent)
            .and_then(|permanent| self.state.permanents[permanent].as_ref())
            .map_or(self.state[object].id.0, |permanent| permanent.id.0);
        Some(DecisionObjectView {
            entity_id: object.slot() as u32,
            subject: SubjectRef::Object {
                id: ObjectRenderId {
                    entity: display_id,
                    incarnation: object.generation(),
                },
            },
            display_id,
            label: self.state[object].name.clone(),
        })
    }

    /// What `viewer` can see now of an object that paid toward a cost: the
    /// appearance that paid, or the one it became by its one move since (the
    /// exiled or discarded card).
    fn paid_object(&self, paid: ObjectId, viewer: PlayerId) -> Option<DecisionObjectView> {
        let now = if self.state.live(paid).is_some() {
            paid
        } else {
            self.state.zones.successor(paid)?
        };
        self.visible_reference(now, viewer).object?;
        self.decision_object(now)
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
                let mut view = self.decision_object(permanent.object)?;
                view.display_id = permanent.id.0;
                if let SubjectRef::Object { ref mut id } = view.subject {
                    id.entity = permanent.id.0;
                }
                Some(view)
            }
            Target::StackSpell(object) => self.decision_object(*object),
            Target::StackAbility(id) => self.decision_object(self.stack_ability_source(*id)?),
            Target::GraveyardCard(reference) => {
                let mut view = self.decision_object(*reference)?;
                // Keep the announced incarnation even if the card later moved.
                view.subject = SubjectRef::Object {
                    id: ObjectRenderId {
                        entity: view.display_id,
                        incarnation: reference.generation(),
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
                                .object_records()
                                .find(|card| card.id == id.0 as i32)
                                .and_then(|_| {
                                    self.state
                                        .latest_objects()
                                        .find(|(_, object)| object.id == *id)
                                        .and_then(|(object, _)| self.decision_object(object))
                                })
                        }?;
                        let role = match action {
                            Action::PlayLand { .. }
                            | Action::CastSpell { .. }
                            | Action::ActivateAbility { .. }
                            | Action::ActivateCardAbility { .. }
                            | Action::Foretell { .. }
                            // The creature an attack declaration sends in acts;
                            // its decline names the same creature as a choice.
                            | Action::DeclareAttacker { attack: true, .. } => "source",
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
    /// `object:<slot>` names each battlefield and graveyard appearance by the
    /// storage slot a committed `CardMoved` event reports, so a land played or
    /// a card discarded is named as the public object it became. Both zones
    /// are public to every viewer.
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
                    self.permanent_object(crate::state::game_object::PermanentId(index))
                {
                    subjects.insert(
                        format!("permanent:{index}"),
                        SubjectRef::Object {
                            id: ObjectRenderId {
                                entity: permanent.id.0,
                                incarnation: reference.generation(),
                            },
                        },
                    );
                }
            }
        }
        for (index, _) in self.state.players.iter().enumerate() {
            let player = PlayerId(index);
            for zone in [
                crate::state::zone::ZoneType::Battlefield,
                crate::state::zone::ZoneType::Graveyard,
            ] {
                for object in self.state.zones.zone_objects(zone, player).iter() {
                    if let Some(view) = self.decision_object(*object) {
                        subjects.insert(format!("object:{}", object.slot()), view.subject);
                    }
                }
            }
        }
        subjects
    }

    /// Only the decision owner receives announcement/private continuation
    /// details. Other viewers receive the public owner/kind and committed facts.
    pub fn decision_context(&self, viewer: PlayerId) -> Option<DecisionContextView> {
        let mut frames = self.continuation_frames(viewer);
        self.describe_effect_execution(viewer, &mut frames);
        let mut view = self.decision_context_from_frames(viewer, &frames)?;
        view.recent_facts = self
            .state
            .events
            .iter()
            .skip(self.state.events.len().saturating_sub(64))
            .filter_map(|event| {
                use crate::flow::event::GameEvent;
                match event {
                    GameEvent::PermanentSacrificed {
                        object_id: object,
                        controller,
                    } => Some(format!(
                        "Seat {} sacrificed {}.",
                        controller.0 + 1,
                        self.state.source_base(*object).1.name
                    )),
                    GameEvent::SpellCast {
                        object_id: object, ..
                    } => Some(format!(
                        "{} was cast; its announced costs are committed.",
                        self.state.source_base(*object).1.name
                    )),
                    GameEvent::SpellResolved { object_id: object } => Some(format!(
                        "{} finished resolving.",
                        self.state.source_base(*object).1.name
                    )),
                    GameEvent::SpellCountered {
                        object_id: object, ..
                    } => Some(format!(
                        "{} was countered.",
                        self.state.source_base(*object).1.name
                    )),
                    GameEvent::AbilityCountered { source_object, .. } => Some(format!(
                        "An ability of {} was countered.",
                        self.state.source_base(*source_object).1.name
                    )),
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
        let player = space.actions.first().map(|action| action.player())?;
        let actor = space.player?;
        let mut view = DecisionContextView {
            pending: Vec::new(),
            actor: actor.0 as u8,
            stage: "priority".into(),
            instruction: "Choose an explicit legal offer to commit one decision.".into(),
            source: RequiredNullable(None),
            targets: Vec::new(),
            reserved_sacrifice: RequiredNullable(None),
            mana_cost: RequiredNullable(None),
            chosen_mode: RequiredNullable(None),
            recent_facts: Vec::new(),
            announcing: None,
            paid: Vec::new(),
        };
        if let Some(opening) = self.state.opening() {
            let (stage, instruction) = opening.instruction();
            view.stage = stage.into();
            view.instruction = if viewer == actor {
                instruction.into()
            } else {
                format!("Waiting for Seat {}.", actor.0 + 1)
            };
            return Some(view);
        }
        for frame in frames.iter().filter(|frame| frame.triggering_amount > 0) {
            let name = frame
                .source
                .definition
                .and_then(|id| {
                    crate::cards::content_pack::default_content_pack()
                        .definition(crate::state::card::CardDefId(id))
                })
                .map_or("This ability", |card| card.name.as_str());
            view.pending.push(format!(
                "{name} triggered on an event amount of {}.",
                frame.triggering_amount
            ));
        }
        if actor != viewer {
            view.stage = format!("{:?}", space.kind).to_lowercase();
            view.instruction = format!("Waiting for Seat {}. Private continuation details are available only to that player.", actor.0 + 1);
            return Some(view);
        }
        view.pending.extend(
            frames
                .iter()
                .flat_map(|frame| frame.effect_execution_summary.iter().cloned()),
        );
        // Presentation consumes the same typed current facts as the native
        // observation and policy tensors. Only labels and instructions live here.
        if let Some(frame) = frames
            .iter()
            .find(|frame| frame.kind == 8)
            .or_else(|| frames.iter().find(|frame| frame.kind == 4))
            .or_else(|| frames.iter().find(|frame| frame.kind == 5))
        {
            let live = |reference: crate::agent::continuation::VisibleReference| {
                let display_id = reference.object? as u32;
                self.state.latest_objects().find_map(|(object, _)| {
                    self.decision_object(object)
                        .filter(|view| view.display_id == display_id)
                        .map(|view| (object, view))
                })
            };
            let object = |reference: crate::agent::continuation::VisibleReference| {
                if let Some(player) = reference.player {
                    return self.decision_target(&Target::Player(PlayerId(player)));
                }
                live(reference).map(|(_, view)| view)
            };
            view.source = RequiredNullable(
                live(frame.source)
                    .map(|(source, mut view)| {
                        // An announced inset spell is named for itself, not its card.
                        if let Some(spell) = self.state[source]
                            .spell_profiles()
                            .find(|profile| {
                                !profile.is_front()
                                    && spell_profile_code(*profile) == frame.spell_profile
                            })
                            .and_then(|profile| {
                                crate::cards::content_pack::default_content_pack()
                                    .spell_definition(&self.state[source], profile)
                            })
                        {
                            view.label = spell.name.clone();
                        }
                        view
                    })
                    .or_else(|| object(frame.source)),
            );
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
            // An announcement frame names the spell or ability being put on the stack.
            if frame.kind == 4 && view.source.0.is_some() {
                view.announcing = Some(
                    if frame.ability.is_some() {
                        "activated_ability"
                    } else {
                        "spell"
                    }
                    .into(),
                );
            }
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
        // What is committed to the pending cost, for a cast, an activation
        // and a resolving payment alike: each object and each unit of mana
        // spent toward it as it was produced, in the order paid.
        view.paid = self
            .pending_receipt()
            .iter()
            .flat_map(PaymentReceipt::in_order)
            .map(|paid| match paid {
                PaidWith::Object(paid) => PaidView {
                    phrase: contribution_phrase(self, viewer, paid),
                    object: self.paid_object(paid.object, viewer),
                },
                PaidWith::Mana(unit) => PaidView {
                    phrase: spent_mana_phrase(self, viewer, unit),
                    object: unit
                        .source
                        .and_then(|origin| self.paid_object(origin.object, viewer)),
                },
            })
            .collect();
        // The one yes-or-no cost choice of a cast, named for what the card
        // offers: `KickerChoice` asks bargain and alternative costs too.
        let cast_option = match &self.pending_choice {
            Some(PendingChoice::KickerChoice { object, cast, .. }) => match cast.spell(&self.state[*object]).cast_option() {
                Some(CastOption::FlashSurcharge(cost)) => format!("Choose whether to pay {} more to cast as though it had flash.", cost_label(cost)),
                Some(CastOption::ExileFromHand) => "Choose whether to exile a card from hand rather than pay the mana cost: paying the cost uses the alternative.".into(),
                Some(CastOption::AlternativeMana(cost)) => format!("Choose whether to pay {} rather than the mana cost.", cost_label(cost)),
                Some(CastOption::Bargain) => "Choose whether to bargain: sacrifice an artifact, enchantment or token as you cast this spell.".into(),
                Some(CastOption::Kicker) | None => "Choose whether to pay the kicker cost.".into(),
            },
            _ => String::new(),
        };
        let (stage, instruction) = match &self.pending_choice {
            Some(PendingChoice::Pay { payment, .. }) => match &payment.owed[0] {
                CostComponent::Mana { help, .. } if *help != Help::default() =>
                    ("payment", "Produce mana, pay part of the cost another way, or finish the announced payment. Mana produced now pays toward the cost as it is made; mana that cannot pay it, or is surplus, stays in the pool, and so does new mana when pooled mana of another class or color could pay the same pip, for the spend choice. A chosen sacrifice leaves when the payment completes; activated mana abilities commit their own costs immediately. A creature tapped for convoke pays {1} or one mana of its color; an artifact tapped for improvise, a creature or artifact tapped for waterbend, or a graveyard card exiled for delve pays {1}. Each such payment commits immediately. The payment ends once no mana is owed."),
                CostComponent::Mana { .. } =>
                    ("payment", "Produce mana or finish the announced payment. Mana produced now pays toward the cost as it is made; mana that cannot pay it, or is surplus, stays in the pool, and so does new mana when pooled mana of another class or color could pay the same pip, for the spend choice. A chosen sacrifice leaves when the payment completes; activated mana abilities commit their own costs immediately. The payment ends once no mana is owed."),
                CostComponent::ExileFromHand =>
                    ("additional_cost", "Choose a card in hand to exile instead of paying the spell's mana cost."),
                CostComponent::Discard if payment.operation.spell().is_some() =>
                    ("additional_cost", "Choose another card in hand to discard as an additional cost to cast this spell."),
                CostComponent::Discard =>
                    ("additional_cost", "Choose a card to discard as an activation cost."),
                CostComponent::Tap(_, Toward::Station) =>
                    ("additional_cost", "Choose another untapped creature to tap to station. The ability puts charge counters equal to that creature's power on its source as it resolves."),
                CostComponent::Tap(..) =>
                    ("additional_cost", "Choose an untapped permanent to tap as an activation cost."),
                CostComponent::Sacrifice(..) =>
                    ("additional_cost", "Choose an additional cost. A sacrifice selection is reserved until the announced payment completes."),
                CostComponent::Forage { .. } =>
                    ("additional_cost", "Forage or pay the stated mana instead. To forage, sacrifice a Food, which is reserved until the announced payment completes, or exile three cards from your graveyard one at a time; the first card exiled commits to exiling three."),
                CostComponent::RemoveCounters { choosing_from: None, .. } =>
                    ("additional_cost", "Choose a creature you control to remove a counter from. Each removal commits immediately; the cost completes once the stated number of counters, of any kinds, are removed from among your creatures."),
                CostComponent::RemoveCounters { .. } =>
                    ("additional_cost", "Choose which kind of counter to remove from the chosen creature. Each mode is one counter kind on it."),
                CostComponent::TapPays { .. } =>
                    ("payment", "Choose what the permanent just tapped pays: an owed color or {1} by convoke, {1} by improvise, or {1} of the waterbend cost. Each mode names the same payment in every decision; only payable ones are offered."),
                CostComponent::Life(_) | CostComponent::RemoveSourceCounters(..) | CostComponent::AddCounters(..) | CostComponent::Source(_) | CostComponent::ExileGrantor(_) | CostComponent::TapGrantor(_) =>
                    unreachable!("paid as the payment completes, after its mana"),
                CostComponent::TapPower { toward: Toward::Saddle, .. } =>
                    ("additional_cost", "Choose a creature to tap to saddle. Each tap commits immediately; the cost completes once the tapped creatures' total power reaches the saddle number."),
                CostComponent::TapPower { .. } =>
                    ("additional_cost", "Choose a creature to tap for crew. Each tap commits immediately; the cost completes once the tapped creatures' total power reaches the crew number."),
            },
            Some(PendingChoice::ChooseTargets { .. } | PendingChoice::ActivatedTarget { .. }) =>
                ("targets", "Choose a target for the current clause. Previously selected targets remain bound to their clauses."),
            Some(PendingChoice::SpellMode { .. }) => ("mode", "Choose a spell mode."),
            Some(PendingChoice::ActivationX { .. }) => ("announcement", "Choose the value of X. Each choice is a value the announced total cost can pay."),
            Some(PendingChoice::KickerChoice { .. }) => ("additional_cost", cast_option.as_str()),
            Some(PendingChoice::AnnounceX { object: object_id, cast, .. }) if cast.spell(&self.state[*object_id]).replicate().is_some() => ("mode", "Announce how many times to pay the replicate cost; each offer's mode index is that count, and the spell is copied once per payment."),
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
                Decision::EnterAsCopy { .. } => ("resolution_choice", "Choose a creature for the entering permanent to enter as a copy of, or decline so it enters as itself."),
                Decision::ManaUpfront { tap_remaining, .. } if *tap_remaining > 0 => ("tap_payment", "Choose an untapped permanent to pay the ability's tap cost. Each selection commits a tap; the continuation records the unpaid count."),
                Decision::ManaUpfront { .. } => ("mana_payment", "Pay the mana ability activation cost before choosing its output."),
                Decision::ManaChoice { production: crate::flow::decision::ManaProduction::Bonus { .. }, .. } => ("mana_color", "Choose the type of mana the source's triggered mana ability adds, one of the types the tapped permanent's mana ability produced. The tapping player chooses; the produced mana is already in the pool and this does not open priority."),
                Decision::ManaSpend { .. } => ("mana_spend", "Choose one unit of mana to spend. Its color, restriction, spend effect and expiry are retained. Each offered choice can complete every saved payment; spent units cannot be undone."),
                Decision::ManaChoice { .. } => ("mana_color", "Choose the mana to add. The mana ability's activation costs have already committed; this does not open priority."),
                Decision::ChooseObject { .. } => ("resolution_choice", "Choose a permanent you control for the resolving instruction (sacrifice, tap, ...), not as an announcement cost."),
                Decision::Learn { .. } => ("learn", "Take a Lesson, discard and draw, or decline Learn. Navigation does not commit a choice."),
                Decision::LookAndSelect { .. } => ("private_inspection", "Choose from the inspected cards using the complete legal offers."),
                Decision::Scry { .. } => ("scry", "Choose the placement of the inspected cards."),
                Decision::OrderLibrary { end: LibraryEnd::Bottom, .. } => ("order_library_bottom", "Choose which remaining card to draw first. It goes above the other remaining cards on the bottom. Each choice commits immediately; the last card is placed automatically."),
                Decision::OrderLibrary { end: LibraryEnd::Top, .. } => ("order_library_top", "Choose which remaining card to draw next. Earlier choices stay above later choices; the last card is placed automatically."),
                Decision::Surveil { .. } => ("surveil", "Choose inspected cards to put into your graveyard, then keep and order the rest on top."),
                Decision::AssignCombatDamage { .. } => ("assign_combat_damage", "Assign combat damage from the attacker."),
                Decision::OrderReplacement { .. } => ("order_replacement", "Choose the next replacement to apply to the proposed event."),
                Decision::ChoosePaymentX { .. } => ("resolution_payment", "Choose X for the optional payment, or decline. Targets are chosen only after payment."),
                Decision::SearchNamed { .. } => ("private_search", "Inspect the searched zones. Choose a matching card to exile, or stop and shuffle."),
                Decision::Payment(PaymentProgress { owed: CostComponent::Discard, .. }) => ("resolution_payment", "Choose a card from your hand to discard, or decline."),
                Decision::Payment(PaymentProgress { owed: CostComponent::Life(_), .. }) => ("resolution_payment", "Pay the stated life for the resolving effect, or decline. Paying is offered only when your life total is at least that amount, and the life is lost at once."),
                Decision::Payment(_) => ("resolution_payment", "Choose a legal payment for the resolving effect, or decline when offered. Submitted costs remain committed."),
                Decision::NameCard { scope, .. } if !scope.is_leaf() => ("name_card", "Choose the group of names containing the card name you want. Choosing a group names nothing and commits nothing; the next choice offers the group's names or smaller groups."),
                Decision::NameCard { .. } => ("name_card", "Choose a card name from this group of the installed naming vocabulary."),
                Decision::Modal { .. } => ("resolution_mode", "Choose a mode for the resolving effect."),
                Decision::Discard { .. } => ("discard", "Choose a card to discard, or decline when offered."),
                Decision::SelectCards { origin: crate::flow::decision::SelectionOrigin::Library, .. } => ("private_search", "Inspect the permitted library candidates. Choose a card, then finish searching when offered."),
                Decision::SelectCards { .. } => ("select_cards", "Choose from the eligible cards, then finish selection when offered."),
                Decision::LegendRule { .. } => ("legend_rule", "Choose which legendary permanent to keep."),
                Decision::SearchDistinctNames { .. } => ("private_search", "Inspect the permitted library candidates. Choose a permitted card with a new name, or stop searching; chosen cards are revealed."),
                Decision::DiscardFromRevealedHand { .. } => ("revealed_hand", "Choose a card from the revealed hand for its owner to discard."),
                Decision::CastDuringResolution { paid: false, .. } => ("free_cast", "Choose a spell to cast without paying its mana cost, or decline."),
                Decision::CastDuringResolution { paid: true, .. } => ("resolution_cast", "Choose a spell to cast now, paying its costs, or decline."),
                Decision::DistributeCounters { .. } => ("distribute_counters", "Place the next counter on a tapped creature an opponent controls; each placement is one choice."),
                Decision::Retarget { .. } => ("retarget", "Choose a new target for the spell or ability on the stack, or keep its target when offered."),
                Decision::ChooseToExile { .. } => ("choose_to_exile", "Choose up to one permanent from among those the resolving ability's controller's opponents control; every chosen permanent is then exiled."),
                Decision::KeepCreatures { sacrifice: Some(_), .. } => ("keep_creatures", "Choose another creature you control to keep; you keep all but the stated number, and every creature not kept is then sacrificed."),
                Decision::KeepCreatures { .. } => ("keep_creatures", "Choose another creature you control to keep within the total power, or stop; every creature not kept is then sacrificed."),
                Decision::ExileLinked { .. } => ("linked_exile", "Choose a card to exile; the resolving ability's controller may later cast it as that ability allows."),
                Decision::OpponentSplitsRevealed { .. } => ("split_revealed", "Choose revealed cards for their owner's graveyard; the rest enter the battlefield tapped."),
            };
            view.stage = stage.into();
            view.instruction = instruction.into();
        }
        if let Some(placement) = &self.state.trigger_placement {
            let choosing_mode = placement
                .placing
                .as_ref()
                .is_some_and(|trigger| self.unchosen_trigger_modes(trigger).is_some());
            view.stage = if choosing_mode {
                "trigger_mode"
            } else if placement.placing.is_some() {
                "trigger_targets"
            } else {
                "order_triggers"
            }
            .into();
            view.instruction = if let Some(trigger) = &placement.placing {
                let description = self.pending_trigger_description(trigger)?;
                if choosing_mode {
                    format!("{description}. Choose this ability's mode, then its targets, before placing the next one.")
                } else {
                    format!("{description}. Choose targets for this ability before placing the next one. Earlier placements and paid costs remain committed.")
                }
            } else {
                "Choose the next ability to put on the stack. Later placements resolve first. Each selection commits immediately.".into()
            };
        }
        if actor != player {
            view.instruction = format!(
                "Seat {} chooses for Seat {} using that player's resources. {}",
                actor.0 + 1,
                player.0 + 1,
                view.instruction
            );
        }
        Some(view)
    }
}
