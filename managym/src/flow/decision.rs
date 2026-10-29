// decision.rs
// Mid-resolution decisions: a resolving spell or ability can suspend, surface
// a choice to a specific player as an ActionSpace, and resume when the choice
// action arrives. The suspended state (remaining effects + the pending
// decision) lives on GameState so clones/forks carry it.

use std::collections::VecDeque;

use rand::seq::SliceRandom;

use crate::{
    agent::action::{Action, ActionSpace, ActionSpaceKind, AgentError},
    flow::{event::GameEvent, game::Game},
    state::{
        ability::{Effect, PaymentCost},
        game_object::{CardId, ObjectRef, PlayerId, Target},
        predicate::CardPredicate,
        zone::ZoneType,
    },
};

/// What to do with the resolving object once its effects finish (CR 608.2m:
/// a spell moves out of the stack as the final part of its resolution).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum FrameFinalize {
    /// Abilities: nothing to clean up.
    None,
    /// Spells: remove the stack object, then put the card into the
    /// battlefield (permanents) or graveyard. A resolving Aura spell enters
    /// attached to `attach_to`, the object its attach instruction chose
    /// (CR 303.4f).
    Spell {
        card: CardId,
        #[serde(skip_serializing_if = "Option::is_none")]
        attach_to: Option<ObjectRef>,
    },
}

/// A chosen target's clause and eligibility at resolution start (CR 608.2b).
/// Keep illegal slots: positional instructions must not reinterpret another
/// chosen object as the final target. Presence is checked separately on use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ResolutionTarget {
    pub target: Target,
    pub requirement: usize,
    pub legal: bool,
}

/// Everything a resolving spell/ability needs while its effects execute,
/// including what's left to do if it suspends for a decision.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct EffectFrame {
    /// The card whose spell or ability is resolving.
    pub source: Option<CardId>,
    /// Exact battlefield source when the effect belongs to an ability of a
    /// permanent. This intentionally does not follow `source` across zones.
    pub source_ref: Option<ObjectRef>,
    /// The player the effects resolve for ("you").
    pub controller: PlayerId,
    /// For triggered abilities: how many times this ability has resolved
    /// this turn, counting this resolution. Zero for spells and activated
    /// abilities.
    pub resolutions_this_turn: u32,
    /// Whether the resolving spell was kicked.
    pub kicked: bool,
    /// The resolving spell's announced X (CR 107.3m); 0 otherwise.
    #[serde(skip_serializing_if = "crate::state::is_zero")]
    pub x: u8,
    /// Excess damage (CR 120.4a) the latest damage instruction dealt to its
    /// creature, read by a later "the excess damage dealt this way".
    #[serde(skip_serializing_if = "crate::state::is_zero")]
    pub excess_damage: i32,
    /// Chosen slots, with eligibility retained across instructions and choices.
    pub targets: Vec<ResolutionTarget>,
    pub active_requirement: usize,
    /// Trigger-context object (e.g. the spell that targeted a warded
    /// permanent) — used as the primary target when none was chosen. For an
    /// activated ability of an Aura or Equipment, the object it was attached to.
    pub context_target: Option<Target>,
    /// The exact object that granted the resolving ability (CR 400.7: once
    /// it leaves the battlefield, references to it find nothing).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grantor: Option<ObjectRef>,
    /// Effects still to execute.
    pub queue: VecDeque<Effect>,
    pub finalize: FrameFinalize,
}

impl EffectFrame {
    /// Run a selected branch before the remaining effects, in authored order.
    pub fn prepend_effects(&mut self, effects: impl DoubleEndedIterator<Item = Effect>) {
        for effect in effects.rev() {
            self.queue.push_front(effect);
        }
    }

    /// Eligible chosen slots for the active clause, in original order.
    pub fn active_targets(&self) -> impl Iterator<Item = ResolutionTarget> + '_ {
        self.targets
            .iter()
            .copied()
            .filter(|slot| slot.legal && slot.requirement == self.active_requirement)
    }

    /// Trigger context is untargeted; it cannot replace an illegal chosen slot.
    pub fn primary_target(&self) -> Option<Target> {
        self.active_targets()
            .next()
            .map(|slot| slot.target)
            .or_else(|| self.context_target.filter(|_| self.targets.is_empty()))
    }
}

/// Remaining obligation and outcome branches of one optional payment.
/// Mana activation costs commit immediately through the mana authority. Only
/// a waterbend contribution closes the decline branch (the existing contract).
/// Exact contributed objects are retained as evidence, never followed or untapped.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct PaymentProgress {
    pub player: PlayerId,
    pub remaining: PaymentCost,
    pub waterbend_paid: Vec<ObjectRef>,
    pub if_paid: Vec<Effect>,
    pub if_declined: Vec<Effect>,
}

impl PaymentProgress {
    pub fn new(
        player: PlayerId,
        cost: PaymentCost,
        if_paid: Vec<Effect>,
        if_declined: Vec<Effect>,
    ) -> Self {
        Self {
            player,
            remaining: cost,
            waterbend_paid: Vec::new(),
            if_paid,
            if_declined,
        }
    }

    pub fn can_decline(&self) -> bool {
        self.waterbend_paid.is_empty()
    }
}

/// A pending mid-resolution decision, including its own bookkeeping so a
/// multi-step choice (scry card-by-card, pick-by-pick selection) re-suspends
/// with updated state.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Decision {
    Payment(PaymentProgress),
    /// A mana ability's printed upfront cost, before its output exists.
    ManaUpfront {
        player: PlayerId,
        permanent: crate::state::game_object::PermanentId,
        ability_index: usize,
        resume: Option<Box<Decision>>,
    },
    ManaChoice {
        player: PlayerId,
        choices: Vec<crate::state::mana::Mana>,
        resume: Option<Box<Decision>>,
    },
    Sacrifice {
        player: PlayerId,
        predicate: CardPredicate,
    },
    SelectMilled {
        player: PlayerId,
        cards: Vec<CardId>,
        if_declined: Vec<Effect>,
    },
    /// Candidates come from the current library, not stored or pinned cards.
    SearchBasicLand {
        player: PlayerId,
        subtype: Option<String>,
        #[serde(skip_serializing_if = "is_one")]
        remaining: usize,
        #[serde(skip_serializing_if = "is_true")]
        tapped: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        untap_if_lands: Option<usize>,
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        bottom_source: bool,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        selected: Vec<CardId>,
    },
    /// "Search your library for a [predicate] card, reveal it, put it into
    /// your hand, then shuffle." Candidates come from the current library.
    SearchLibraryToHand {
        player: PlayerId,
        predicate: CardPredicate,
    },
    OptionalDiscardDraw {
        player: PlayerId,
        count: usize,
    },
    /// Choose cards in their desired top/bottom order (CR 701.26).
    Scry {
        player: PlayerId,
        /// All inspected cards stay pinned until the whole reorder commits.
        looked: Vec<CardId>,
        remaining: Vec<CardId>,
        top: Vec<CardId>,
        bottom: Vec<CardId>,
    },
    /// Look at top N, select up to K matching cards to hand, rest to the
    /// bottom of the library in random order.
    LookAndSelect {
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        reveal: bool,
        player: PlayerId,
        /// Looked-at cards still in the library, top first.
        looked: Vec<CardId>,
        predicate: CardPredicate,
        selected: usize,
        min_select: usize,
        max_select: usize,
    },
    /// "Choose one —".
    Modal {
        player: PlayerId,
        modes: Vec<Vec<Effect>>,
    },
    /// Learn: retrieve a Lesson, discard then draw, or do neither.
    Learn {
        player: PlayerId,
    },
    Discard {
        player: PlayerId,
        remaining: usize,
    },
    /// All legend selections precede the simultaneous SBA batch (CR 704.5j).
    LegendRule {
        player: PlayerId,
        groups: Vec<Vec<CardId>>,
        keep: Vec<CardId>,
    },
}

fn is_one(value: &usize) -> bool {
    *value == 1
}
fn is_true(value: &bool) -> bool {
    *value
}

impl Decision {
    pub fn player(&self) -> PlayerId {
        match self {
            Decision::Payment(payment) => payment.player,
            Decision::ManaUpfront { player, .. }
            | Decision::ManaChoice { player, .. }
            | Decision::Sacrifice { player, .. } => *player,
            Decision::SelectMilled { player, .. }
            | Decision::SearchBasicLand { player, .. }
            | Decision::SearchLibraryToHand { player, .. }
            | Decision::OptionalDiscardDraw { player, .. }
            | Decision::Scry { player, .. }
            | Decision::LookAndSelect { player, .. }
            | Decision::Modal { player, .. }
            | Decision::Learn { player }
            | Decision::Discard { player, .. }
            | Decision::LegendRule { player, .. } => *player,
        }
    }

    /// Exact top-of-library cards inspected by scry/look decisions, top first.
    /// These enter the deciding player's observation and stay pinned during
    /// `Game::determinize`. Search candidates are derived separately.
    pub fn revealed_cards(&self) -> &[CardId] {
        match self {
            Decision::ManaUpfront { .. }
            | Decision::Payment(_)
            | Decision::ManaChoice { .. }
            | Decision::Sacrifice { .. } => &[],
            Decision::Scry { looked, .. } => looked,
            Decision::LookAndSelect { looked, .. } => looked,
            Decision::SelectMilled { .. }
            | Decision::SearchBasicLand { .. }
            | Decision::SearchLibraryToHand { .. }
            | Decision::OptionalDiscardDraw { .. }
            | Decision::Modal { .. }
            | Decision::Learn { .. }
            | Decision::Discard { .. }
            | Decision::LegendRule { .. } => &[],
        }
    }
}

/// A resolution paused on a decision.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct SuspendedResolution {
    pub frame: EffectFrame,
    pub decision: Decision,
}

impl Game {
    pub(crate) fn suspend_rule_decision(&mut self, decision: Decision) {
        self.state.suspended_decision = Some(SuspendedResolution {
            frame: EffectFrame {
                source: None,
                source_ref: None,
                controller: decision.player(),
                resolutions_this_turn: 0,
                kicked: false,
                x: 0,
                excess_damage: 0,
                targets: Vec::new(),
                active_requirement: 0,
                context_target: None,
                grantor: None,
                queue: VecDeque::new(),
                finalize: FrameFinalize::None,
            },
            decision,
        });
    }

    /// Execute a frame's remaining effects. If an effect suspends for a
    /// decision, park the frame; otherwise finalize the resolution.
    pub(crate) fn run_frame(&mut self, mut frame: EffectFrame) {
        while let Some(effect) = frame.queue.pop_front() {
            if let Some(decision) = self.execute_frame_effect(&effect, &mut frame) {
                self.state.suspended_decision = Some(SuspendedResolution { frame, decision });
                return;
            }
        }
        self.finalize_frame(frame);
    }

    fn finalize_frame(&mut self, frame: EffectFrame) {
        if let FrameFinalize::Spell { card, attach_to } = frame.finalize {
            let is_permanent = self.state.cards[card].types.is_permanent();
            if is_permanent {
                self.move_card_with_entry(
                    card,
                    ZoneType::Battlefield,
                    crate::flow::proposed_event::BattlefieldEntry {
                        x: frame.x,
                        ..Default::default()
                    },
                );
                if let Some(permanent_id) = self.state.card_to_permanent[card] {
                    self.journal_permanent(permanent_id);
                    if let Some(permanent) = self.state.permanents[permanent_id].as_mut() {
                        permanent.attached_to = attach_to;
                    }
                }
                let owner = self.state.cards[card].owner;
                self.invalidate_mana_cache(owner);
            } else {
                self.move_card(card, ZoneType::Graveyard);
            }
            self.emit(GameEvent::SpellResolved { card });
        }
        // A resolution completed: restart the priority round (CR 117.3b).
        self.state.priority.on_non_pass_action(self.active_player());
    }

    /// The action space for the pending mid-resolution decision, if any.
    pub(crate) fn suspended_decision_action_space(&self) -> Option<ActionSpace> {
        let suspended = self.state.suspended_decision.as_ref()?;
        let source_focus = suspended
            .frame
            .source
            .map(|card| vec![self.state.cards[card].id])
            .unwrap_or_default();
        let space = match &suspended.decision {
            Decision::ManaUpfront { player, .. } => ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::PayOrNot,
                actions: self.mana_ability_actions(*player, None),
                focus: source_focus,
            },
            Decision::Payment(payment) => {
                let player = payment.player;
                let mut actions = self.mana_ability_actions(player, None);
                if let PaymentCost::Waterbend { generic } = payment.remaining {
                    if generic > 0 {
                        for permanent in self.waterbend_candidates(player) {
                            let mut probe = self.clone();
                            probe.tap_permanent(permanent, false);
                            if probe.can_pay_with_waterbend(
                                player,
                                &crate::state::mana::ManaCost::default().with_generic(generic - 1),
                            ) {
                                actions.push(Action::WaterbendTap { player, permanent });
                            }
                        }
                    }
                }
                if self.can_pay_mana_cost(player, &payment.remaining.mana_cost()) {
                    actions.push(Action::PayCost { player });
                }
                if payment.can_decline() {
                    actions.push(Action::Decline { player });
                }
                ActionSpace {
                    player: Some(player),
                    kind: match payment.remaining {
                        PaymentCost::Mana(_) => ActionSpaceKind::PayOrNot,
                        PaymentCost::Waterbend { .. } => ActionSpaceKind::Waterbend,
                    },
                    actions,
                    focus: source_focus,
                }
            }
            Decision::ManaChoice {
                player, choices, ..
            } => ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::Modal,
                focus: source_focus,
                actions: (0..choices.len())
                    .filter(|mode| self.mana_choice_completes_payment(*player, &choices[*mode]))
                    .map(|mode| Action::ChooseMode {
                        player: *player,
                        mode,
                    })
                    .collect(),
            },
            Decision::Sacrifice { player, predicate } => ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::LookAndSelect,
                actions: self
                    .sacrifice_candidates(*player, predicate)
                    .into_iter()
                    .map(|id| Action::SelectCard {
                        player: *player,
                        card: self.state.permanents[id].as_ref().unwrap().card,
                    })
                    .collect(),
                focus: source_focus,
            },
            Decision::SearchBasicLand {
                player, subtype, ..
            } => {
                let mut actions: Vec<Action> = self
                    .basic_land_search_candidates(*player, subtype)
                    .into_iter()
                    .map(|card| Action::SelectCard {
                        player: *player,
                        card,
                    })
                    .collect();
                actions.push(Action::Decline { player: *player });
                ActionSpace {
                    player: Some(*player),
                    kind: ActionSpaceKind::LookAndSelect,
                    actions,
                    focus: source_focus,
                }
            }
            Decision::SearchLibraryToHand { player, predicate } => {
                let mut actions: Vec<Action> = self
                    .library_search_candidates(*player, predicate)
                    .into_iter()
                    .map(|card| Action::SelectCard {
                        player: *player,
                        card,
                    })
                    .collect();
                actions.push(Action::Decline { player: *player });
                ActionSpace {
                    player: Some(*player),
                    kind: ActionSpaceKind::LookAndSelect,
                    actions,
                    focus: source_focus,
                }
            }
            Decision::Discard { player, .. } | Decision::OptionalDiscardDraw { player, .. } => {
                let mut actions: Vec<Action> = self
                    .state
                    .zones
                    .zone_cards(ZoneType::Hand, *player)
                    .iter()
                    .map(|card| Action::SelectCard {
                        player: *player,
                        card: *card,
                    })
                    .collect();
                if matches!(suspended.decision, Decision::OptionalDiscardDraw { .. }) {
                    actions.push(Action::Decline { player: *player });
                }
                ActionSpace {
                    player: Some(*player),
                    kind: ActionSpaceKind::Discard,
                    actions,
                    focus: source_focus,
                }
            }

            Decision::SelectMilled { player, cards, .. } => ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::LookAndSelect,
                actions: cards
                    .iter()
                    .map(|card| Action::SelectCard {
                        player: *player,
                        card: *card,
                    })
                    .chain(std::iter::once(Action::Decline { player: *player }))
                    .collect(),
                focus: cards
                    .iter()
                    .map(|card| self.state.cards[*card].id)
                    .collect(),
            },
            Decision::Scry {
                player, remaining, ..
            } => ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::Scry,
                actions: remaining
                    .iter()
                    .flat_map(|card| {
                        [false, true].map(|to_bottom| Action::ScryCard {
                            player: *player,
                            card: *card,
                            to_bottom,
                        })
                    })
                    .collect(),
                focus: remaining
                    .iter()
                    .map(|card| self.state.cards[card].id)
                    .collect(),
            },
            Decision::LookAndSelect {
                player,
                looked,
                predicate,
                selected,
                min_select,
                ..
            } => {
                let mut actions: Vec<Action> = looked
                    .iter()
                    .filter(|card| predicate.matches_card(&self.state.cards[**card]))
                    .map(|card| Action::SelectCard {
                        player: *player,
                        card: *card,
                    })
                    .collect();
                if *selected >= *min_select || actions.is_empty() {
                    actions.push(Action::Decline { player: *player });
                }
                ActionSpace {
                    player: Some(*player),
                    kind: ActionSpaceKind::LookAndSelect,
                    actions,
                    focus: looked
                        .iter()
                        .map(|card| self.state.cards[card].id)
                        .collect(),
                }
            }
            Decision::Modal { player, modes } => ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::Modal,
                actions: (0..modes.len())
                    .map(|mode| Action::ChooseMode {
                        player: *player,
                        mode,
                    })
                    .collect(),
                focus: source_focus,
            },
            Decision::LegendRule { player, groups, .. } => ActionSpace {
                player: Some(*player),
                kind: ActionSpaceKind::LegendRule,
                actions: groups[0]
                    .iter()
                    .map(|card| Action::SelectCard {
                        player: *player,
                        card: *card,
                    })
                    .collect(),
                focus: groups[0]
                    .iter()
                    .map(|card| self.state.cards[*card].id)
                    .collect(),
            },
            Decision::Learn { player } => {
                let mut actions: Vec<Action> = self
                    .state
                    .zones
                    .zone_cards(ZoneType::Hand, *player)
                    .iter()
                    .map(|card| Action::LearnDiscard {
                        player: *player,
                        card: *card,
                    })
                    .collect();
                actions.extend(
                    self.state.players[player.0]
                        .sideboard
                        .iter()
                        .filter(|card| self.eligible_lesson(*player, **card))
                        .map(|card| Action::LearnTakeLesson {
                            player: *player,
                            card: *card,
                        }),
                );
                actions.push(Action::Decline { player: *player });
                ActionSpace {
                    player: Some(*player),
                    kind: ActionSpaceKind::Learn,
                    actions,
                    focus: source_focus,
                }
            }
        };
        Some(space)
    }

    /// Feed a player action into the pending mid-resolution decision.
    pub(crate) fn execute_decision_action(&mut self, action: &Action) -> Result<(), AgentError> {
        let Some(suspended) = &self.state.suspended_decision else {
            return Err(AgentError("no pending decision".to_string()));
        };

        if let Some(player) = action_player(action) {
            if player != suspended.decision.player() {
                return Err(AgentError("wrong player for pending decision".to_string()));
            }
        }

        // Validate while the full continuation is still attached, so the
        // payment model sees every reserved source and remaining obligation.
        if let (
            Decision::ManaChoice {
                player, choices, ..
            },
            Action::ChooseMode { mode, .. },
        ) = (&suspended.decision, action)
        {
            let mana = choices
                .get(*mode)
                .ok_or_else(|| AgentError("illegal mana color".into()))?;
            if !self.mana_choice_completes_payment(*player, mana) {
                return Err(AgentError("mana color cannot complete the payment".into()));
            }
        }
        let mut suspended = self.state.suspended_decision.take().unwrap();
        match (&mut suspended.decision, action) {
            (
                Decision::Payment(PaymentProgress {
                    player,
                    remaining: PaymentCost::Waterbend { generic },
                    waterbend_paid,
                    ..
                }),
                Action::WaterbendTap { permanent, .. },
            ) if *generic > 0 && self.waterbend_candidates(*player).contains(permanent) => {
                let object = self
                    .permanent_object_ref(*permanent)
                    .expect("waterbend candidate");
                self.tap_permanent(*permanent, false);
                *generic -= 1;
                waterbend_paid.push(object);
                self.state.suspended_decision = Some(suspended);
                Ok(())
            }
            (Decision::Payment(payment), Action::PayCost { .. }) => {
                let cost = payment.remaining.mana_cost();
                if let Err(error) = self
                    .produce_mana(payment.player, &cost)
                    .and_then(|()| self.spend_mana(payment.player, &cost))
                {
                    self.state.suspended_decision = Some(suspended);
                    return Err(error);
                }
                let effects = std::mem::take(&mut payment.if_paid);
                suspended.frame.prepend_effects(effects.into_iter());
                self.run_frame(suspended.frame);
                Ok(())
            }
            (Decision::Payment(payment), Action::Decline { .. }) if payment.can_decline() => {
                let effects = std::mem::take(&mut payment.if_declined);
                suspended.frame.prepend_effects(effects.into_iter());
                self.run_frame(suspended.frame);
                Ok(())
            }
            (
                Decision::ManaChoice {
                    player,
                    choices,
                    resume,
                },
                Action::ChooseMode { mode, .. },
            ) => {
                let mana = &choices[*mode];
                self.journal_player(*player);
                self.state.players[player.0].mana_pool.add(mana);
                if let Some(resume) = resume.take() {
                    suspended.decision = *resume;
                    self.state.suspended_decision = Some(suspended);
                    self.advance_mana_upfront()?;
                } else {
                    // An activated mana ability retains its payer's priority;
                    // its empty rule frame is not a resolving stack object.
                    self.state.priority.on_non_pass_action(*player);
                }
                Ok(())
            }
            (Decision::Sacrifice { player, predicate }, Action::SelectCard { card, .. }) => {
                let selected = self
                    .sacrifice_candidates(*player, predicate)
                    .into_iter()
                    .find(|id| {
                        self.state.permanents[*id]
                            .as_ref()
                            .is_some_and(|p| p.card == *card)
                    });
                let Some(selected) = selected else {
                    self.state.suspended_decision = Some(suspended);
                    return Err(AgentError("illegal sacrifice selection".into()));
                };
                self.sacrifice_permanent(selected);
                self.run_frame(suspended.frame);
                Ok(())
            }
            (
                Decision::SearchBasicLand {
                    player,
                    subtype,
                    remaining,
                    tapped,
                    untap_if_lands,
                    bottom_source,
                    selected,
                },
                Action::SelectCard { .. } | Action::Decline { .. },
            ) => {
                let player = *player;
                if let Action::SelectCard { card, .. } = action {
                    if !self
                        .basic_land_search_candidates(player, subtype)
                        .contains(card)
                    {
                        self.state.suspended_decision = Some(suspended);
                        return Err(AgentError("not a matching basic land in library".into()));
                    }
                    self.move_card_with_entry(
                        *card,
                        ZoneType::Battlefield,
                        crate::flow::proposed_event::BattlefieldEntry {
                            tapped: *tapped,
                            ..Default::default()
                        },
                    );
                    selected.push(*card);
                    *remaining -= 1;
                    if *remaining > 0 {
                        self.state.suspended_decision = Some(suspended);
                        return Ok(());
                    }
                }
                if *bottom_source {
                    if let Some(source) = self.source_permanent_id(&suspended.frame) {
                        let card = self.state.permanents[source].as_ref().unwrap().card;
                        self.move_card(card, ZoneType::Library);
                        self.put_on_bottom_of_library(card);
                    }
                }
                self.shuffle_library(player);
                if untap_if_lands.is_some_and(|minimum| {
                    self.battlefield_permanents(player)
                        .into_iter()
                        .filter(|id| {
                            self.permanent_matches_predicate(
                                *id,
                                &CardPredicate {
                                    card_type: Some(crate::state::card::CardType::Land),
                                    ..Default::default()
                                },
                            )
                        })
                        .count()
                        >= minimum
                }) {
                    for card in selected {
                        if let Some(id) = self.state.card_to_permanent[*card] {
                            self.journal_permanent(id);
                            if let Some(permanent) = self.state.permanents[id].as_mut() {
                                permanent.untap();
                            }
                        }
                    }
                    self.invalidate_mana_cache(player);
                }
                self.run_frame(suspended.frame);
                Ok(())
            }
            (
                Decision::SearchLibraryToHand { player, predicate },
                Action::SelectCard { .. } | Action::Decline { .. },
            ) => {
                let player = *player;
                if let Action::SelectCard { card, .. } = action {
                    if !self
                        .library_search_candidates(player, predicate)
                        .contains(card)
                    {
                        self.state.suspended_decision = Some(suspended);
                        return Err(AgentError("not a matching card in library".into()));
                    }
                    self.reveal_card_to_hand(*card, player);
                }
                self.shuffle_library(player);
                self.run_frame(suspended.frame);
                Ok(())
            }
            (Decision::OptionalDiscardDraw { player, count }, Action::SelectCard { card, .. }) => {
                if !self.state.zones.contains(*card, ZoneType::Hand, *player) {
                    self.state.suspended_decision = Some(suspended);
                    return Err(AgentError("discard must be from hand".into()));
                }
                let count = *count;
                self.move_card(*card, ZoneType::Graveyard);
                suspended
                    .frame
                    .queue
                    .push_front(Effect::DrawCards { count });
                self.run_frame(suspended.frame);
                Ok(())
            }
            (Decision::OptionalDiscardDraw { .. }, Action::Decline { .. }) => {
                self.run_frame(suspended.frame);
                Ok(())
            }
            (Decision::Discard { player, remaining }, Action::SelectCard { card, .. }) => {
                if !self.state.zones.contains(*card, ZoneType::Hand, *player) {
                    self.state.suspended_decision = Some(suspended);
                    return Err(AgentError("discard must be from hand".into()));
                }
                self.move_card(*card, ZoneType::Graveyard);
                *remaining -= 1;
                if *remaining == 0
                    || self
                        .state
                        .zones
                        .zone_cards(ZoneType::Hand, *player)
                        .is_empty()
                {
                    self.run_frame(suspended.frame);
                } else {
                    self.state.suspended_decision = Some(suspended);
                }
                Ok(())
            }

            (
                Decision::Scry {
                    player,
                    looked,
                    remaining,
                    top,
                    bottom,
                },
                Action::ScryCard {
                    card, to_bottom, ..
                },
            ) => {
                let Some(index) = remaining.iter().position(|candidate| candidate == card) else {
                    self.state.suspended_decision = Some(suspended);
                    return Err(AgentError("card is not in the pending scry".into()));
                };
                remaining.remove(index);
                if *to_bottom {
                    bottom.push(*card);
                } else {
                    top.push(*card);
                }
                if remaining.is_empty() {
                    self.journal_zones();
                    let library = self.state.zones.zone_cards_mut(ZoneType::Library, *player);
                    library.retain(|card| !looked.contains(card));
                    for card in bottom.iter().rev() {
                        library.insert(0, *card);
                    }
                    library.extend(top.iter().rev());
                    self.run_frame(suspended.frame);
                } else {
                    self.state.suspended_decision = Some(suspended);
                }
                Ok(())
            }
            (
                Decision::LookAndSelect {
                    player,
                    looked,
                    predicate,
                    selected,
                    reveal,
                    max_select,
                    ..
                },
                Action::SelectCard { card, .. },
            ) => {
                let selectable =
                    looked.contains(card) && predicate.matches_card(&self.state.cards[*card]);
                if !selectable {
                    let err = AgentError("card is not selectable".to_string());
                    self.state.suspended_decision = Some(suspended);
                    return Err(err);
                }
                let player = *player;
                looked.retain(|c| c != card);
                *selected += 1;
                if *reveal {
                    self.reveal_card_to_hand(*card, player);
                } else {
                    self.move_card(*card, ZoneType::Hand);
                }
                self.invalidate_mana_cache(player);

                let more_selectable = looked
                    .iter()
                    .any(|c| predicate.matches_card(&self.state.cards[*c]));
                if *selected >= *max_select || !more_selectable {
                    let rest = std::mem::take(looked);
                    self.bottom_in_random_order(rest);
                    self.run_frame(suspended.frame);
                } else {
                    self.state.suspended_decision = Some(suspended);
                }
                Ok(())
            }
            (
                Decision::LookAndSelect {
                    looked,
                    selected,
                    min_select,
                    predicate,
                    ..
                },
                Action::Decline { .. },
            ) => {
                let any_selectable = looked
                    .iter()
                    .any(|c| predicate.matches_card(&self.state.cards[*c]));
                if *selected < *min_select && any_selectable {
                    let err = AgentError("selection is not optional yet".to_string());
                    self.state.suspended_decision = Some(suspended);
                    return Err(err);
                }
                let rest = std::mem::take(looked);
                self.bottom_in_random_order(rest);
                self.run_frame(suspended.frame);
                Ok(())
            }
            (Decision::SelectMilled { player, cards, .. }, Action::SelectCard { card, .. })
                if cards.contains(card) =>
            {
                self.reveal_card_to_hand(*card, *player);
                self.run_frame(suspended.frame);
                Ok(())
            }
            (Decision::SelectMilled { if_declined, .. }, Action::Decline { .. }) => {
                suspended.frame.prepend_effects(if_declined.iter().cloned());
                self.run_frame(suspended.frame);
                Ok(())
            }
            (Decision::Modal { modes, .. }, Action::ChooseMode { mode, .. }) => {
                if *mode >= modes.len() {
                    let err = AgentError("modal mode out of range".to_string());
                    self.state.suspended_decision = Some(suspended);
                    return Err(err);
                }
                let effects = std::mem::take(&mut modes[*mode]);
                suspended.frame.prepend_effects(effects.into_iter());
                self.run_frame(suspended.frame);
                Ok(())
            }
            (
                Decision::LegendRule {
                    player,
                    groups,
                    keep,
                },
                Action::SelectCard { card, .. },
            ) => {
                if !groups[0].contains(card) {
                    self.state.suspended_decision = Some(suspended);
                    return Err(AgentError("not a legend in this group".into()));
                }
                keep.push(*card);
                groups.remove(0);
                if let Some(group) = groups.first() {
                    let id = self.state.card_to_permanent[group[0]].expect("legend on battlefield");
                    *player = self.state.permanents[id]
                        .as_ref()
                        .expect("legend")
                        .controller;
                    self.state.suspended_decision = Some(suspended);
                } else {
                    self.perform_state_based_actions_with_legends(keep);
                }
                Ok(())
            }
            (Decision::Learn { player }, Action::LearnDiscard { card, .. }) => {
                let player = *player;
                if !self.state.zones.contains(*card, ZoneType::Hand, player) {
                    let err = AgentError("card to discard is not in hand".to_string());
                    self.state.suspended_decision = Some(suspended);
                    return Err(err);
                }
                self.move_card(*card, ZoneType::Graveyard);
                self.draw_cards(player, 1);
                self.run_frame(suspended.frame);
                Ok(())
            }
            (Decision::Learn { player }, Action::LearnTakeLesson { card, .. }) => {
                if !self.eligible_lesson(*player, *card) {
                    self.state.suspended_decision = Some(suspended);
                    return Err(AgentError(
                        "card is not an available owned sideboard Lesson".to_string(),
                    ));
                }
                self.reveal_card_to_hand(*card, *player);
                self.run_frame(suspended.frame);
                Ok(())
            }
            (Decision::Learn { .. }, Action::Decline { .. }) => {
                self.run_frame(suspended.frame);
                Ok(())
            }
            _ => {
                let err = AgentError("action does not match pending decision".to_string());
                self.state.suspended_decision = Some(suspended);
                Err(err)
            }
        }
    }

    fn eligible_lesson(&self, player: PlayerId, card: CardId) -> bool {
        self.state.players[player.0].sideboard.contains(&card)
            && self.state.cards[card].owner == player
            && self.state.zones.zone_of(card).is_none()
            && self.state.cards[card]
                .subtypes
                .iter()
                .any(|subtype| subtype == "Lesson")
    }

    /// The top `count` cards of `player`'s library, top first.
    pub(crate) fn library_top(&self, player: PlayerId, count: usize) -> Vec<CardId> {
        self.state
            .zones
            .zone_cards(ZoneType::Library, player)
            .iter()
            .rev()
            .take(count)
            .copied()
            .collect()
    }

    /// The top card of `player`'s library.
    pub(crate) fn library_top_card(&self, player: PlayerId) -> Option<CardId> {
        self.state
            .zones
            .zone_cards(ZoneType::Library, player)
            .last()
            .copied()
    }

    /// Move a library card to the bottom of its owner's library. Not a zone
    /// change (CR 701.26b-style reorder), so no events fire.
    pub(crate) fn put_on_bottom_of_library(&mut self, card: CardId) {
        let owner = self.state.cards[card].owner;
        // A reorder within one zone, so the zone-move inverse does not apply.
        self.journal_zones();
        let library = self.state.zones.zone_cards_mut(ZoneType::Library, owner);
        if let Some(index) = library.iter().position(|c| *c == card) {
            library.remove(index);
            library.insert(0, card);
        }
    }

    /// Put the given library cards on the bottom of their owners' libraries
    /// in a random order.
    pub(crate) fn bottom_in_random_order(&mut self, mut cards: Vec<CardId>) {
        cards.shuffle(&mut self.state.rng);
        for card in cards {
            self.put_on_bottom_of_library(card);
        }
    }
}

fn action_player(action: &Action) -> Option<PlayerId> {
    match action {
        Action::ScryCard { player, .. }
        | Action::SelectCard { player, .. }
        | Action::LearnDiscard { player, .. }
        | Action::LearnTakeLesson { player, .. }
        | Action::Decline { player }
        | Action::WaterbendTap { player, .. }
        | Action::PayCost { player }
        | Action::ChooseMode { player, .. } => Some(*player),
        _ => None,
    }
}

#[cfg(test)]
mod learn_tests {
    use super::*;
    use crate::{
        decision::Command,
        search_state::{
            witness, BenchCommand, BranchDriver, ClonePlusUndoDriver, DensePageCowUndoDriver,
            FullCloneDriver,
        },
        state::player::PlayerConfig,
    };
    use std::collections::BTreeMap;

    fn counts(entries: &[(&str, usize)]) -> BTreeMap<String, usize> {
        entries
            .iter()
            .map(|(name, count)| (name.to_string(), *count))
            .collect()
    }

    fn game(sideboard: &[(&str, usize)]) -> Game {
        Game::new(
            vec![
                PlayerConfig::new("learner", counts(&[("Island", 40)]))
                    .with_sideboard(counts(sideboard)),
                PlayerConfig::new("opponent", counts(&[("Plains", 40)]))
                    .with_sideboard(counts(&[("Fancy Footwork", 1)])),
            ],
            17,
            false,
        )
    }

    fn learn(game: &mut Game) {
        game.run_frame(EffectFrame {
            source: None,
            source_ref: None,
            controller: PlayerId(0),
            resolutions_this_turn: 0,
            kicked: false,
            x: 0,
            excess_damage: 0,
            targets: vec![],
            active_requirement: 0,
            context_target: None,
            grantor: None,
            queue: VecDeque::from([Effect::Learn]),
            finalize: FrameFinalize::None,
        });
        game.publish_action_space(game.suspended_decision_action_space().unwrap());
    }

    fn take(game: &mut Game, card: CardId) {
        game.execute_decision_action(&Action::LearnTakeLesson {
            player: PlayerId(0),
            card,
        })
        .unwrap();
    }

    #[test]
    fn learn_retrieves_each_supported_lesson_without_drawing_or_allocating() {
        for name in [
            "Firebending Lesson",
            "It'll Quench Ya!",
            "Accumulate Wisdom",
            "Yip Yip!",
            "Fancy Footwork",
        ] {
            let mut game = game(&[(name, 1)]);
            let card = game.state.players[0].sideboard[0];
            assert!(game.current_object_ref(card).is_none());
            let deck = game.state.players[0].deck.clone();
            assert!(!deck.contains(&card));
            learn(&mut game);
            let before = witness(&game);
            let library = game
                .state
                .zones
                .zone_cards(ZoneType::Library, PlayerId(0))
                .clone();
            let hand_count = game.state.zones.size(ZoneType::Hand, PlayerId(0));
            let card_count = game.state.cards.len();
            let event_count = game.state.events.len();
            take(&mut game, card);
            assert_eq!(game.state.cards.len(), card_count);
            assert_eq!(
                game.state.zones.size(ZoneType::Hand, PlayerId(0)),
                hand_count + 1
            );
            assert!(game.state.zones.contains(card, ZoneType::Hand, PlayerId(0)));
            assert_eq!(game.state.cards[card].name, name);
            assert!(game.current_object_ref(card).is_some());
            assert_eq!(game.state.players[0].deck, deck);
            assert_eq!(
                game.state.zones.zone_cards(ZoneType::Library, PlayerId(0)),
                &library
            );
            assert_eq!(
                witness(&game).diagnostics.rng_probe,
                before.diagnostics.rng_probe
            );
            assert!(game.state.suspended_decision.is_none());
            let events: Vec<_> = game.state.events.iter().skip(event_count).collect();
            assert!(events.iter().any(|event| matches!(event, GameEvent::CardMoved { card: moved, from: None, to: ZoneType::Hand, .. } if *moved == card)));
            assert!(!events
                .iter()
                .any(|event| matches!(event, GameEvent::CardDrawn { .. })));
        }
    }

    #[test]
    fn learn_consumes_each_physical_copy_once_in_every_destination() {
        for destination in [ZoneType::Hand, ZoneType::Graveyard, ZoneType::Exile] {
            let mut game = game(&[("Firebending Lesson", 2)]);
            let roster = game.state.players[0].sideboard.clone();
            learn(&mut game);
            take(&mut game, roster[0]);
            if destination != ZoneType::Hand {
                game.move_card(roster[0], destination);
            }
            learn(&mut game);
            let before = witness(&game);
            assert!(game
                .execute_decision_action(&Action::LearnTakeLesson {
                    player: PlayerId(0),
                    card: roster[0]
                })
                .is_err());
            assert_eq!(witness(&game), before);
            take(&mut game, roster[1]);
            learn(&mut game);
            assert!(!game
                .action_space()
                .unwrap()
                .actions
                .iter()
                .any(|action| matches!(action, Action::LearnTakeLesson { .. })));
            assert_eq!(game.state.players[0].sideboard, roster);
        }
    }

    #[test]
    fn learn_rejects_wrong_owner_nonlesson_unowned_and_wrong_actor_atomically() {
        let mut game = game(&[("Firebending Lesson", 1), ("Island", 1)]);
        learn(&mut game);
        let opponent_card = game.state.players[1].sideboard[0];
        let nonlesson = *game.state.players[0]
            .sideboard
            .iter()
            .find(|&&card| game.state.cards[card].name == "Island")
            .unwrap();
        let lesson = *game.state.players[0]
            .sideboard
            .iter()
            .find(|&&card| game.state.cards[card].name == "Firebending Lesson")
            .unwrap();
        for action in [
            Action::LearnTakeLesson {
                player: PlayerId(0),
                card: opponent_card,
            },
            Action::LearnTakeLesson {
                player: PlayerId(0),
                card: nonlesson,
            },
            Action::LearnTakeLesson {
                player: PlayerId(0),
                card: CardId(usize::MAX),
            },
            Action::LearnTakeLesson {
                player: PlayerId(1),
                card: lesson,
            },
            Action::LearnDiscard {
                player: PlayerId(0),
                card: lesson,
            },
        ] {
            let before = witness(&game);
            assert!(game.execute_decision_action(&action).is_err());
            assert_eq!(witness(&game), before);
        }
    }

    #[test]
    fn learn_empty_hand_and_empty_sideboard_have_exact_choices() {
        for sideboard in [vec![], vec![("Firebending Lesson", 1)], vec![("Island", 1)]] {
            let mut game = game(&sideboard);
            let hand = game
                .state
                .zones
                .zone_cards(ZoneType::Hand, PlayerId(0))
                .clone();
            for card in hand {
                game.move_card(card, ZoneType::Graveyard);
            }
            learn(&mut game);
            let actions = &game.action_space().unwrap().actions;
            assert!(!actions
                .iter()
                .any(|action| matches!(action, Action::LearnDiscard { .. })));
            assert_eq!(
                actions.len(),
                if sideboard
                    .first()
                    .is_some_and(|(name, _)| *name == "Firebending Lesson")
                {
                    2
                } else {
                    1
                }
            );
            let before = game.state.deterministic_hash();
            let zones = game.state.zones.clone();
            game.execute_decision_action(&Action::Decline {
                player: PlayerId(0),
            })
            .unwrap();
            assert_eq!(
                serde_json::to_value(&game.state.zones).unwrap(),
                serde_json::to_value(&zones).unwrap()
            );
            assert_ne!(
                game.state.deterministic_hash(),
                before,
                "suspended continuation completes"
            );
        }
    }

    #[test]
    fn learn_sideboard_projection_is_private_complete_and_consumed_once() {
        use crate::agent::observation::Observation;

        let mut game = game(&[("Firebending Lesson", 40), ("Island", 1)]);
        let initial = Observation::for_player(&game, PlayerId(0));
        assert_eq!(initial.agent_sideboard.len(), 41);
        assert_eq!(initial.agent.sideboard_counts.values().sum::<u32>(), 41);
        assert_eq!(
            initial.agent.sideboard_counts,
            initial.agent.remaining_sideboard_counts
        );
        assert!(initial.agent_sideboard.iter().all(|outside| {
            initial
                .agent_cards
                .iter()
                .all(|card| card.id != outside.candidate_id)
        }));
        let other = Observation::for_player(&game, PlayerId(1));
        assert_eq!(other.agent_sideboard.len(), 1);
        assert_eq!(other.agent_sideboard[0].name, "Fancy Footwork");
        assert_eq!(
            other.opponent.sideboard_counts,
            initial.agent.sideboard_counts
        );
        assert!(other.action_space.actions.is_empty());

        learn(&mut game);
        let deciding = Observation::for_player(&game, PlayerId(0));
        let retrievals: Vec<_> = deciding
            .action_space
            .actions
            .iter()
            .filter(|action| {
                action.action_type == crate::agent::action::ActionType::LearnTakeLesson
            })
            .collect();
        assert_eq!(retrievals.len(), 40);
        for action in &retrievals {
            assert_eq!(action.focus.len(), 1);
            let outside = deciding
                .agent_sideboard
                .iter()
                .find(|card| card.candidate_id == action.focus[0])
                .unwrap();
            assert_eq!(outside.name, "Firebending Lesson");
        }
        let last = retrievals.last().unwrap().focus[0];
        let index = deciding
            .action_space
            .actions
            .iter()
            .rposition(|action| {
                action.action_type == crate::agent::action::ActionType::LearnTakeLesson
            })
            .unwrap();
        assert!(index > 32);
        let before = witness(&game);
        let mut branch = game.clone();
        let frame = branch.semantic_decision_frame().unwrap();
        branch
            .execute_semantic_command(&Command {
                command_id: "outside-projection".into(),
                expected_revision: frame.revision,
                offer_id: frame.offers[index].id.0,
                answers: vec![],
                object_preconditions: vec![],
            })
            .unwrap();
        let after = Observation::for_player(&branch, PlayerId(0));
        assert_eq!(after.agent_sideboard.len(), 40);
        assert!(after
            .agent_sideboard
            .iter()
            .all(|card| card.candidate_id != last));
        assert!(after
            .agent_cards
            .iter()
            .any(|card| card.id == last && card.zone == ZoneType::Hand));
        assert_eq!(after.agent.sideboard_counts, initial.agent.sideboard_counts);
        assert_eq!(
            after.agent.remaining_sideboard_counts.values().sum::<u32>(),
            40
        );
        let other_after = Observation::for_player(&branch, PlayerId(1));
        assert_eq!(
            other_after.opponent.remaining_sideboard_counts,
            after.agent.remaining_sideboard_counts
        );
        assert_eq!(other_after.opponent.known_hand.values().sum::<u32>(), 1);
        assert_eq!(other_after.agent_sideboard, other.agent_sideboard);
        assert_eq!(witness(&game), before);
    }

    #[test]
    fn learn_complete_offers_execute_beyond_thirty_two_and_reject_stale_reuse() {
        let mut game = game(&[("Firebending Lesson", 40)]);
        learn(&mut game);
        let source = game.clone();
        let before = witness(&source);
        let frame = game.semantic_decision_frame().unwrap();
        assert_eq!(
            frame.offers.len(),
            game.action_space().unwrap().actions.len()
        );
        assert!(frame.offers.len() > 40);
        let index = game
            .action_space()
            .unwrap()
            .actions
            .iter()
            .rposition(|action| matches!(action, Action::LearnTakeLesson { .. }))
            .unwrap();
        assert!(index > 32);
        let command = Command {
            command_id: "take-last-lesson".into(),
            expected_revision: frame.revision,
            offer_id: frame.offers[index].id.0,
            answers: vec![],
            object_preconditions: vec![],
        };
        game.execute_semantic_command(&command).unwrap();
        let committed = witness(&game);
        assert!(game.execute_semantic_command(&command).is_err());
        assert_eq!(witness(&game), committed);
        assert_eq!(witness(&source), before);
    }

    fn rollback_retrieval<D: BranchDriver<State = Game>>(driver: D) {
        let mut source = game(&[("Firebending Lesson", 2)]);
        learn(&mut source);
        let source_witness = witness(&source);
        let mut branch = driver.fork_exact(&source);
        let sibling = driver.fork_exact(&source);
        let mark = driver.mark(&mut branch);
        let action_index = branch
            .action_space()
            .unwrap()
            .actions
            .iter()
            .position(|action| matches!(action, Action::LearnTakeLesson { .. }))
            .unwrap();
        driver
            .apply(
                &mut branch,
                BenchCommand {
                    action_index,
                    expected_state_hash: None,
                    expected_action_hash: None,
                },
            )
            .unwrap();
        let retrieved = witness(&branch);
        let nested = driver.mark(&mut branch);
        driver
            .apply(
                &mut branch,
                BenchCommand {
                    action_index: 0,
                    expected_state_hash: None,
                    expected_action_hash: None,
                },
            )
            .unwrap();
        driver.rollback(&mut branch, nested);
        assert_eq!(witness(&branch), retrieved);
        driver.rollback(&mut branch, mark);
        assert_eq!(witness(&branch), source_witness);
        assert_eq!(witness(&source), source_witness);
        assert_eq!(witness(&sibling), source_witness);
    }

    #[test]
    fn learn_retrieval_forks_and_rolls_back_outside_inventory() {
        rollback_retrieval(FullCloneDriver);
        rollback_retrieval(ClonePlusUndoDriver::default());
        rollback_retrieval(DensePageCowUndoDriver::default());
    }
    #[test]
    fn learn_reveal_is_ordered_public_definition_and_known_hand_is_not_a_hand_leak() {
        use crate::agent::observation::{EventEntityKind, EventType, Observation};
        let mut game = game(&[("Firebending Lesson", 1)]);
        let card = game.state.players[0].sideboard[0];
        let definition = game.state.cards[card].definition_id;
        learn(&mut game);
        let before = game.semantic_observation(PlayerId(1)).unwrap();
        let event_start = game.state.events.len();
        take(&mut game, card);
        let events: Vec<_> = game.state.events.iter().skip(event_start).collect();
        assert!(
            matches!(events[0], GameEvent::CardRevealed { owner: PlayerId(0), definition: d } if *d == definition)
        );
        assert!(matches!(
            events[1],
            GameEvent::CardMoved {
                from: None,
                to: ZoneType::Hand,
                ..
            }
        ));
        let reveal = Observation::event_data(events[0]);
        assert_eq!(reveal[0].event_type, EventType::CardRevealed as i32);
        assert_eq!(reveal[0].source_kind, EventEntityKind::Definition as i32);
        assert_eq!(reveal[0].source_id, definition.0 as i32);
        assert_eq!(reveal[0].source_incarnation, -1);
        let owner = Observation::for_player(&game, PlayerId(0));
        let other = Observation::for_player(&game, PlayerId(1));
        assert_eq!(owner.agent.known_hand, other.opponent.known_hand);
        assert_eq!(other.opponent.known_hand.get(&definition.0), Some(&1));
        assert!(!other
            .opponent_cards
            .iter()
            .any(|card| card.zone == ZoneType::Hand));
        assert_ne!(
            before.identity.viewer_state_hash,
            game.semantic_observation(PlayerId(1))
                .unwrap()
                .identity
                .viewer_state_hash
        );
        game.draw_cards(PlayerId(0), 1);
        assert_eq!(game.state.players[0].known_hand.get(&definition), Some(&1));
        // Rebuilding a viewer projection needs no event queue or client memory.
        game.take_observation_events();
        assert_eq!(
            Observation::for_player(&game, PlayerId(1))
                .opponent
                .known_hand
                .get(&definition.0),
            Some(&1)
        );
        let with_knowledge = game.state.deterministic_hash();
        let mut without = game.clone();
        without.state.players[0].known_hand.clear();
        assert_ne!(without.state.deterministic_hash(), with_knowledge);
    }

    fn small_known_hand_game() -> Game {
        let mut game = Game::new(
            vec![
                PlayerConfig::new(
                    "learner",
                    counts(&[("Firebending Lesson", 2), ("Island", 2)]),
                )
                .with_sideboard(counts(&[("Firebending Lesson", 2)])),
                PlayerConfig::new("opponent", counts(&[("Plains", 40)])),
            ],
            17,
            false,
        );
        // One unknown hand slot, a residual pool of two Lessons and two Islands.
        let deck = game.state.players[0].deck.clone();
        for card in deck {
            game.move_card(card, ZoneType::Library);
        }
        let island = game.state.players[0]
            .deck
            .iter()
            .copied()
            .find(|card| game.state.cards[*card].name == "Island")
            .unwrap();
        game.move_card(island, ZoneType::Hand);
        learn(&mut game);
        let card = game.state.players[0].sideboard[0];
        take(&mut game, card);
        game
    }

    #[test]
    fn hidden_hand_constraints_match_exact_domain_without_truth() {
        use crate::possible_worlds::{HiddenHandConstraints, PossibleWorldSpace};
        let mut game = small_known_hand_game();
        let before = HiddenHandConstraints::for_viewer(&game, PlayerId(1));
        let exact = PossibleWorldSpace::for_viewer(&game, PlayerId(1)).projection();
        assert_eq!(before.pool, exact.pool);
        assert_eq!(before.known_hand, exact.known_hand);
        assert_eq!(before.hand_size, exact.hand_size);
        game.determinize(PlayerId(1), 781);
        let after = HiddenHandConstraints::for_viewer(&game, PlayerId(1));
        assert_eq!(before, after);
    }

    #[test]
    fn learn_possible_worlds_use_residual_copy_weights_and_reject_missing_knowledge() {
        use crate::possible_worlds::{PossibleWorldSpace, WorldQuery};
        let game = small_known_hand_game();
        let space = PossibleWorldSpace::for_viewer(&game, PlayerId(1));
        assert_eq!(
            space.projection().known_hand,
            BTreeMap::from([("Firebending Lesson".to_string(), 1)])
        );
        assert_eq!(space.pool().values().sum::<u32>(), 5); // unchosen sideboard excluded
        assert_eq!(space.hand_size(), 2);
        assert_eq!(space.total_weight(), 4); // C(4 residual copies, 1 unknown slot)
        assert_eq!(space.worlds().len(), 2);
        for world in space.worlds() {
            assert_eq!(world.weight, 2); // no extra multiplicity for the known Lesson
            assert!(world.hand["Firebending Lesson"] >= 1);
            let branch = space.materialize(&game, world, 42).unwrap();
            assert_eq!(
                branch.state.players[0].known_hand,
                game.state.players[0].known_hand
            );
            assert_eq!(
                crate::possible_worlds::viewer_observation(&branch, PlayerId(1)),
                crate::possible_worlds::viewer_observation(&game, PlayerId(1))
            );
        }
        let excludes = WorldQuery::Lacks {
            card: "Firebending Lesson".to_string(),
            fewer_than: 1,
        };
        assert_eq!(space.support_receipt(&excludes).total_weight, 0);
        assert!(space.condition(&excludes).is_err());
        let mut forged = space.worlds()[0].clone();
        forged.hand = BTreeMap::from([("Island".to_string(), 2)]);
        assert!(space.materialize(&game, &forged, 42).is_err());
        let mut stale = game.clone();
        stale.state.players[0].known_hand.clear();
        assert!(space.materialize(&stale, &space.worlds()[0], 42).is_err());
    }

    #[test]
    fn learn_determinization_preserves_known_counts_refreshes_choices_and_ignores_copy_identity() {
        use crate::possible_worlds::PossibleWorldSpace;
        let mut source = small_known_hand_game();
        learn(&mut source);
        let before = witness(&source);
        let space = PossibleWorldSpace::for_viewer(&source, PlayerId(1));
        for index in 0..space.worlds().len() {
            let mut branch = space
                .materialize_index(
                    &source,
                    index,
                    42,
                    crate::possible_worlds::MaterializeMode::RefreshOpponentCommitment,
                )
                .unwrap();
            let hand = branch.state.zones.zone_cards(ZoneType::Hand, PlayerId(0));
            for action in &branch.action_space().unwrap().actions {
                if let Action::LearnDiscard { card, .. } = action {
                    assert!(hand.contains(card));
                }
            }
            let action_index = branch
                .action_space()
                .unwrap()
                .actions
                .iter()
                .position(|action| matches!(action, Action::LearnTakeLesson { .. }))
                .unwrap();
            branch.step(action_index).unwrap();
            assert_eq!(branch.state.players[0].known_hand.values().sum::<u32>(), 2);
        }
        let mut seen = std::collections::BTreeSet::new();
        for seed in 0..32 {
            let mut branch = source.clone();
            branch.determinize(PlayerId(1), seed);
            let hand = branch.state.zones.zone_cards(ZoneType::Hand, PlayerId(0));
            let lesson_count = hand
                .iter()
                .filter(|card| branch.state.cards[**card].name == "Firebending Lesson")
                .count();
            assert!(lesson_count >= 1);
            seen.insert(lesson_count);
            let actions = &branch.action_space().unwrap().actions;
            assert_eq!(
                actions
                    .iter()
                    .filter(|action| matches!(action, Action::LearnDiscard { .. }))
                    .count(),
                hand.len()
            );
            for action in actions {
                if let Action::LearnDiscard { card, .. } = action {
                    assert!(hand.contains(card));
                }
            }
            assert_eq!(
                branch.state.players[0].known_hand,
                source.state.players[0].known_hand
            );
            assert_eq!(
                PossibleWorldSpace::for_viewer(&branch, PlayerId(1)).worlds(),
                space.worlds()
            );
            let index = actions
                .iter()
                .position(|action| matches!(action, Action::LearnDiscard { .. }))
                .unwrap();
            branch.step(index).unwrap();
        }
        assert_eq!(seen, std::collections::BTreeSet::from([1, 2]));
        assert_eq!(witness(&source), before);
        let own_choices = source.semantic_decision_frame().unwrap();
        let mut own = source.clone();
        own.determinize(PlayerId(0), 42);
        assert_eq!(
            serde_json::to_value(own.semantic_decision_frame().unwrap()).unwrap(),
            serde_json::to_value(own_choices).unwrap()
        );

        let mut swapped = source.clone();
        let retrieved = source.state.players[0].sideboard[0];
        let duplicate = source.state.players[0]
            .deck
            .iter()
            .copied()
            .find(|card| source.state.cards[*card].name == "Firebending Lesson")
            .unwrap();
        // An identical main-deck physical copy can witness the known card.
        swapped
            .state
            .zones
            .move_card(retrieved, PlayerId(0), ZoneType::Library);
        swapped
            .state
            .zones
            .move_card(duplicate, PlayerId(0), ZoneType::Hand);
        assert_eq!(
            crate::possible_worlds::viewer_observation(&swapped, PlayerId(1)),
            crate::possible_worlds::viewer_observation(&source, PlayerId(1))
        );
        assert_eq!(
            PossibleWorldSpace::for_viewer(&swapped, PlayerId(1)).projection(),
            space.projection()
        );
        let swapped_branch = space.materialize(&swapped, &space.worlds()[0], 42).unwrap();
        let source_branch = space.materialize(&source, &space.worlds()[0], 42).unwrap();
        assert_eq!(
            serde_json::to_value(&swapped_branch.state.zones).unwrap(),
            serde_json::to_value(&source_branch.state.zones).unwrap()
        );
    }

    #[test]
    fn learn_public_duplicate_departure_decrements_known_minimum_without_copy_identity() {
        let mut game = small_known_hand_game();
        let retrieved = game.state.players[0].sideboard[0];
        let definition = game.state.cards[retrieved].definition_id;
        let duplicate = game.state.players[0]
            .deck
            .iter()
            .copied()
            .find(|card| game.state.cards[*card].definition_id == definition)
            .unwrap();
        game.move_card(duplicate, ZoneType::Hand);
        assert_eq!(game.state.players[0].known_hand.get(&definition), Some(&1));
        game.move_card(duplicate, ZoneType::Graveyard);
        assert!(game
            .state
            .zones
            .contains(retrieved, ZoneType::Hand, PlayerId(0)));
        assert!(game.state.players[0].known_hand.is_empty());
    }

    fn rollback_knowledge<D: BranchDriver<State = Game>>(driver: D) {
        let mut source = small_known_hand_game();
        learn(&mut source);
        source
            .state
            .suspended_decision
            .as_mut()
            .unwrap()
            .frame
            .queue
            .push_back(Effect::Learn);
        let before = witness(&source);
        let mut branch = driver.fork_exact(&source);
        let sibling = driver.fork_exact(&source);
        let outer = driver.mark(&mut branch);
        let card = branch.state.players[0].sideboard[0];
        let index = branch.action_space().unwrap().actions.iter().position(|action|
            matches!(action, Action::LearnDiscard { card: chosen, .. } if *chosen == card)).unwrap();
        driver
            .apply(
                &mut branch,
                BenchCommand {
                    action_index: index,
                    expected_state_hash: None,
                    expected_action_hash: None,
                },
            )
            .unwrap();
        assert!(branch.state.players[0].known_hand.is_empty());
        let departed = witness(&branch);
        let inner = driver.mark(&mut branch);
        let index = branch
            .action_space()
            .unwrap()
            .actions
            .iter()
            .position(|action| matches!(action, Action::LearnTakeLesson { .. }))
            .unwrap();
        driver
            .apply(
                &mut branch,
                BenchCommand {
                    action_index: index,
                    expected_state_hash: None,
                    expected_action_hash: None,
                },
            )
            .unwrap();
        assert_eq!(branch.state.players[0].known_hand.values().sum::<u32>(), 1);
        branch.determinize(PlayerId(1), 22);
        driver.rollback(&mut branch, inner);
        assert!(
            witness(&branch) == departed,
            "nested rollback: {}",
            std::any::type_name::<D>()
        );
        driver.rollback(&mut branch, outer);
        assert!(
            witness(&branch) == before,
            "outer rollback: {}",
            std::any::type_name::<D>()
        );
        assert!(witness(&source) == before);
        assert!(witness(&sibling) == before);
    }

    #[test]
    fn learn_nested_knowledge_and_search_rollback_preserve_retained_roots() {
        rollback_knowledge(FullCloneDriver);
        rollback_knowledge(ClonePlusUndoDriver::default());
        rollback_knowledge(DensePageCowUndoDriver::default());
    }
}

#[cfg(test)]
mod payment_tests {
    use super::*;
    use crate::{
        search_state::{
            witness, BenchCommand, BranchDriver, ClonePlusUndoDriver, DensePageCowUndoDriver,
            FullCloneDriver,
        },
        state::{
            mana::{Mana, ManaCost},
            player::PlayerConfig,
        },
    };
    use std::collections::BTreeMap;

    fn payment_game(cost: PaymentCost) -> Game {
        payment_game_with_deck(cost, BTreeMap::from([("Island".into(), 40)]))
    }

    fn payment_game_with_deck(cost: PaymentCost, deck: BTreeMap<String, usize>) -> Game {
        let mut game = Game::new(
            vec![
                PlayerConfig::new("payer", deck),
                PlayerConfig::new("other", BTreeMap::from([("Plains".into(), 40)])),
            ],
            165,
            false,
        );
        game.run_frame(EffectFrame {
            source: None,
            source_ref: None,
            controller: PlayerId(0),
            resolutions_this_turn: 0,
            kicked: false,
            x: 0,
            excess_damage: 0,
            targets: vec![],
            active_requirement: 0,
            context_target: None,
            grantor: None,
            queue: VecDeque::from([
                Effect::OptionalPayment {
                    cost,
                    if_paid: vec![Effect::DrawCards { count: 2 }],
                    if_declined: vec![Effect::DrawCards { count: 1 }],
                },
                Effect::DrawCards { count: 3 },
            ]),
            finalize: FrameFinalize::None,
        });
        game.publish_action_space(game.suspended_decision_action_space().unwrap());
        game
    }

    #[test]
    fn either_method_runs_exactly_one_selected_continuation_before_the_suffix() {
        for cost in [
            PaymentCost::Mana(ManaCost::parse("{0}")),
            PaymentCost::Waterbend { generic: 0 },
        ] {
            for paid in [false, true] {
                let mut game = payment_game(cost.clone());
                let before = game.state.zones.size(ZoneType::Hand, PlayerId(0));
                let action = if paid {
                    Action::PayCost {
                        player: PlayerId(0),
                    }
                } else {
                    Action::Decline {
                        player: PlayerId(0),
                    }
                };
                game.execute_decision_action(&action).unwrap();
                assert!(game.state.suspended_decision.is_none());
                assert_eq!(
                    game.state.zones.size(ZoneType::Hand, PlayerId(0)),
                    before + if paid { 5 } else { 4 }
                );
                assert!(game.execute_decision_action(&action).is_err());
            }
        }
    }

    #[test]
    fn failed_payment_keeps_the_pending_obligation_and_continuations() {
        for cost in [
            PaymentCost::Mana(ManaCost::parse("{2}")),
            PaymentCost::Waterbend { generic: 2 },
        ] {
            let mut game = payment_game(cost);
            let before = witness(&game);
            assert!(game
                .execute_decision_action(&Action::PayCost {
                    player: PlayerId(0)
                })
                .is_err());
            assert_eq!(witness(&game), before);
            let actions = &game.suspended_decision_action_space().unwrap().actions;
            assert_eq!(
                actions,
                &[Action::Decline {
                    player: PlayerId(0)
                }]
            );
        }
    }

    #[test]
    fn shared_payment_and_both_continuations_restore_through_nested_command_rollback() {
        for cost in [
            PaymentCost::Mana(ManaCost::parse("{0}")),
            PaymentCost::Waterbend { generic: 0 },
        ] {
            let game = payment_game(cost);
            let root = witness(&game);
            let driver = ClonePlusUndoDriver::default();
            for index in 0..game.action_space().unwrap().actions.len() {
                let mut branch = driver.fork_exact(&game);
                let outer = driver.mark(&mut branch);
                driver
                    .apply(
                        &mut branch,
                        BenchCommand {
                            action_index: index,
                            expected_action_hash: None,
                            expected_state_hash: None,
                        },
                    )
                    .unwrap();
                let after_payment = driver.witness(&branch);
                let inner = driver.mark(&mut branch);
                driver
                    .apply(
                        &mut branch,
                        BenchCommand {
                            action_index: 0,
                            expected_action_hash: None,
                            expected_state_hash: None,
                        },
                    )
                    .unwrap();
                driver.rollback(&mut branch, inner);
                assert_eq!(driver.witness(&branch), after_payment);
                driver.rollback(&mut branch, outer);
                assert_eq!(driver.witness(&branch), root);
                assert_eq!(witness(&game), root);
            }
        }
    }

    fn apply_payment_action<D: BranchDriver<State = Game>>(
        driver: &D,
        game: &mut Game,
        predicate: impl Fn(&Action) -> bool,
    ) {
        let before = witness(game);
        let action_index = game
            .action_space()
            .unwrap()
            .actions
            .iter()
            .position(predicate)
            .expect("payment action is legal");
        driver
            .apply(
                game,
                BenchCommand {
                    action_index,
                    expected_action_hash: Some(before.legal_surface.hash),
                    expected_state_hash: Some(before.authority.hash),
                },
            )
            .unwrap();
    }

    fn rollback_suspended_waterbend<D: BranchDriver<State = Game>>(driver: D) {
        let player = PlayerId(0);
        let mut game = payment_game_with_deck(
            PaymentCost::Waterbend { generic: 2 },
            BTreeMap::from([
                ("Island".into(), 40),
                ("Hermitic Herbalist".into(), 1),
                ("Prophetic Prism".into(), 1),
            ]),
        );
        let mut sources = Vec::new();
        for name in ["Hermitic Herbalist", "Prophetic Prism"] {
            let card = game.state.players[0]
                .deck
                .iter()
                .copied()
                .find(|card| game.state.cards[*card].name == name)
                .unwrap();
            game.move_card(card, ZoneType::Battlefield);
            sources.push(game.current_object_ref(card).unwrap());
        }
        let waterbend = game.lookup_current_permanent(sources[0]).unwrap();
        let prism = game.lookup_current_permanent(sources[1]).unwrap();
        game.state.players[0].mana_pool = Mana::parse("1");
        game.invalidate_mana_cache(player);
        game.publish_action_space(game.suspended_decision_action_space().unwrap());
        let root = witness(&game);
        let before_hand = game.state.zones.size(ZoneType::Hand, player);

        // Keep a divergent sibling alive throughout all nested rollbacks.
        let mut sibling = driver.fork_exact(&game);
        apply_payment_action(&driver, &mut sibling, |a| {
            matches!(a, Action::Decline { .. })
        });
        let sibling_witness = witness(&sibling);
        assert_eq!(
            sibling.state.zones.size(ZoneType::Hand, player),
            before_hand + 4
        );

        let mut branch = driver.fork_exact(&game);
        let outer = driver.mark(&mut branch);
        apply_payment_action(
            &driver,
            &mut branch,
            |a| matches!(a, Action::WaterbendTap { permanent, .. } if *permanent == waterbend),
        );
        let partial = witness(&branch);
        let partial_resolution = branch.state.suspended_decision.clone().unwrap();
        let Decision::Payment(payment) = &partial_resolution.decision else {
            panic!("partial waterbend must retain its payment");
        };
        assert_eq!(payment.remaining, PaymentCost::Waterbend { generic: 1 });
        assert_eq!(payment.waterbend_paid, vec![sources[0]]);
        assert!(!payment.can_decline());
        assert!(!branch
            .action_space()
            .unwrap()
            .actions
            .iter()
            .any(|a| matches!(a, Action::Decline { .. })));

        let inner = driver.mark(&mut branch);
        let activate_prism = |a: &Action| matches!(a, Action::ActivateAbility { permanent, ability_index: 0, .. } if *permanent == prism);
        apply_payment_action(&driver, &mut branch, activate_prism);
        assert_eq!(branch.state.players[0].mana_pool.total(), 0);
        assert!(branch.state.permanents[prism].as_ref().unwrap().tapped);
        let modal_resolution = branch.state.suspended_decision.clone().unwrap();
        let Decision::ManaChoice {
            resume: Some(resume),
            ..
        } = &modal_resolution.decision
        else {
            panic!("Prism must suspend the committed payment for its immediate color choice");
        };
        assert_eq!(**resume, partial_resolution.decision);
        assert_eq!(modal_resolution.frame, partial_resolution.frame);
        assert_eq!(branch.action_space().unwrap().kind, ActionSpaceKind::Modal);
        let modal = witness(&branch);

        let color_mark = driver.mark(&mut branch);
        let choose_blue = |a: &Action| matches!(a, Action::ChooseMode { mode: 1, .. });
        apply_payment_action(&driver, &mut branch, choose_blue);
        assert_eq!(
            branch.state.suspended_decision,
            Some(partial_resolution.clone())
        );
        assert_eq!(branch.state.players[0].mana_pool, Mana::parse("U"));
        for source in [waterbend, prism] {
            assert!(branch.state.permanents[source].as_ref().unwrap().tapped);
        }
        assert!(!branch
            .action_space()
            .unwrap()
            .actions
            .iter()
            .any(|a| matches!(a, Action::Decline { .. })));
        let resumed = witness(&branch);
        apply_payment_action(&driver, &mut branch, |a| {
            matches!(a, Action::PayCost { .. })
        });
        assert!(branch.state.suspended_decision.is_none());
        assert_eq!(
            branch.state.zones.size(ZoneType::Hand, player),
            before_hand + 5
        );
        assert_eq!(branch.state.players[0].mana_pool.total(), 0);
        let completed = witness(&branch);

        driver.rollback(&mut branch, color_mark);
        assert_eq!(witness(&branch), modal);
        assert_eq!(branch.state.suspended_decision, Some(modal_resolution));
        apply_payment_action(&driver, &mut branch, choose_blue);
        assert_eq!(witness(&branch), resumed);
        apply_payment_action(&driver, &mut branch, |a| {
            matches!(a, Action::PayCost { .. })
        });
        assert_eq!(witness(&branch), completed);

        driver.rollback(&mut branch, inner);
        assert_eq!(witness(&branch), partial);
        assert_eq!(branch.state.suspended_decision, Some(partial_resolution));
        assert!(!branch.state.permanents[prism].as_ref().unwrap().tapped);
        assert_eq!(branch.state.players[0].mana_pool, Mana::parse("1"));
        apply_payment_action(&driver, &mut branch, activate_prism);
        assert_eq!(witness(&branch), modal);
        driver.rollback(&mut branch, outer);
        assert_eq!(witness(&branch), root);
        assert_eq!(witness(&game), root);
        assert_eq!(witness(&sibling), sibling_witness);
    }

    #[test]
    fn partial_waterbend_and_paid_mana_choice_restore_through_nested_command_rollback() {
        rollback_suspended_waterbend(FullCloneDriver);
        rollback_suspended_waterbend(ClonePlusUndoDriver::default());
        rollback_suspended_waterbend(DensePageCowUndoDriver::default());
    }
}
