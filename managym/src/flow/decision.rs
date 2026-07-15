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
        ability::Effect,
        game_object::{CardId, ObjectRef, PlayerId, Target},
        mana::ManaCost,
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
    /// battlefield (permanents) or graveyard.
    Spell { card: CardId },
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
    /// Chosen targets, in requirement order.
    pub targets: Vec<Target>,
    /// Requirement index for each chosen target (multi-target spells).
    pub target_req_indices: Vec<usize>,
    /// Trigger-context object (e.g. the spell that targeted a warded
    /// permanent) — used as the primary target when none was chosen.
    pub context_target: Option<Target>,
    /// Effects still to execute.
    pub queue: VecDeque<Effect>,
    pub finalize: FrameFinalize,
}

impl EffectFrame {
    /// The single target most effects act on: the first chosen target, or
    /// the trigger-context object.
    pub fn primary_target(&self) -> Option<Target> {
        self.targets.first().copied().or(self.context_target)
    }
}

/// A pending mid-resolution decision, including its own bookkeeping so a
/// multi-step choice (scry card-by-card, pick-by-pick selection) re-suspends
/// with updated state.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Decision {
    /// Scry: decide keep-or-bottom for `remaining[0]`, then continue.
    Scry {
        player: PlayerId,
        /// Undecided revealed cards, top of library first.
        remaining: Vec<CardId>,
    },
    /// Look at top N, select up to K matching cards to hand, rest to the
    /// bottom of the library in random order.
    LookAndSelect {
        player: PlayerId,
        /// Looked-at cards still in the library, top first.
        looked: Vec<CardId>,
        predicate: CardPredicate,
        selected: usize,
        min_select: usize,
        max_select: usize,
    },
    /// "You may pay [cost]" / "unless [player] pays [cost]".
    PayOrNot {
        player: PlayerId,
        cost: ManaCost,
        if_paid: Vec<Effect>,
        if_declined: Vec<Effect>,
    },
    /// "Choose one —".
    Modal {
        player: PlayerId,
        modes: Vec<Vec<Effect>>,
    },
    /// Learn: "you may discard a card; if you do, draw a card".
    DiscardThenDraw { player: PlayerId },
}

impl Decision {
    pub fn player(&self) -> PlayerId {
        match self {
            Decision::Scry { player, .. }
            | Decision::LookAndSelect { player, .. }
            | Decision::PayOrNot { player, .. }
            | Decision::Modal { player, .. }
            | Decision::DiscardThenDraw { player } => *player,
        }
    }

    /// Library cards revealed to the deciding player (empty for decisions
    /// that only concern public objects). Top of library first. These are
    /// added to the deciding agent's observation and pinned in place by
    /// `Game::determinize`.
    pub fn revealed_cards(&self) -> &[CardId] {
        match self {
            Decision::Scry { remaining, .. } => remaining,
            Decision::LookAndSelect { looked, .. } => looked,
            Decision::PayOrNot { .. }
            | Decision::Modal { .. }
            | Decision::DiscardThenDraw { .. } => &[],
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
        if let FrameFinalize::Spell { card } = frame.finalize {
            if let Some(index) = self.find_spell_on_stack_index(card) {
                self.state.stack_objects.remove(index);
            }
            let is_permanent = self.state.cards[card].types.is_permanent();
            if is_permanent {
                self.move_card(card, ZoneType::Battlefield);
                let owner = self.state.cards[card].owner;
                self.invalidate_mana_cache(owner);
            } else {
                self.move_card(card, ZoneType::Graveyard);
            }
            self.emit(GameEvent::SpellResolved { card });
        }
        // A resolution completed: restart the priority round (CR 117.3b).
        self.state
            .priority
            .on_non_pass_action(self.active_player());
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
            Decision::Scry { player, remaining } => {
                let card = *remaining.first()?;
                ActionSpace {
                    player: Some(*player),
                    kind: ActionSpaceKind::Scry,
                    actions: vec![
                        Action::ScryCard {
                            player: *player,
                            card,
                            to_bottom: false,
                        },
                        Action::ScryCard {
                            player: *player,
                            card,
                            to_bottom: true,
                        },
                    ],
                    focus: vec![self.state.cards[card].id],
                }
            }
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
            Decision::PayOrNot { player, cost, .. } => {
                let mut actions = Vec::with_capacity(2);
                if self.available_mana(*player).can_pay(cost) {
                    actions.push(Action::PayCost { player: *player });
                }
                actions.push(Action::Decline { player: *player });
                ActionSpace {
                    player: Some(*player),
                    kind: ActionSpaceKind::PayOrNot,
                    actions,
                    focus: source_focus,
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
            Decision::DiscardThenDraw { player } => {
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
                actions.push(Action::Decline { player: *player });
                ActionSpace {
                    player: Some(*player),
                    kind: ActionSpaceKind::DiscardThenDraw,
                    actions,
                    focus: source_focus,
                }
            }
        };
        Some(space)
    }

    /// Feed a player action into the pending mid-resolution decision.
    pub(crate) fn execute_decision_action(&mut self, action: &Action) -> Result<(), AgentError> {
        let Some(mut suspended) = self.state.suspended_decision.take() else {
            return Err(AgentError("no pending decision".to_string()));
        };

        if let Some(player) = action_player(action) {
            if player != suspended.decision.player() {
                self.state.suspended_decision = Some(suspended);
                return Err(AgentError("wrong player for pending decision".to_string()));
            }
        }

        match (&mut suspended.decision, action) {
            (Decision::Scry { remaining, .. }, Action::ScryCard { card, to_bottom, .. }) => {
                if remaining.first() != Some(card) {
                    let err = AgentError("scry decision is for a different card".to_string());
                    self.state.suspended_decision = Some(suspended);
                    return Err(err);
                }
                remaining.remove(0);
                if *to_bottom {
                    self.put_on_bottom_of_library(*card);
                }
                if remaining.is_empty() {
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
                    max_select,
                    ..
                },
                Action::SelectCard { card, .. },
            ) => {
                let selectable = looked.contains(card)
                    && predicate.matches_card(&self.state.cards[*card]);
                if !selectable {
                    let err = AgentError("card is not selectable".to_string());
                    self.state.suspended_decision = Some(suspended);
                    return Err(err);
                }
                let player = *player;
                looked.retain(|c| c != card);
                *selected += 1;
                self.move_card(*card, ZoneType::Hand);
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
            (
                Decision::PayOrNot {
                    player,
                    cost,
                    if_paid,
                    ..
                },
                Action::PayCost { .. },
            ) => {
                let player = *player;
                let cost = cost.clone();
                let effects = std::mem::take(if_paid);
                if let Err(err) = self.produce_mana(player, &cost) {
                    self.state.suspended_decision = Some(suspended);
                    return Err(err);
                }
                if let Err(err) = self.spend_mana(player, &cost) {
                    self.state.suspended_decision = Some(suspended);
                    return Err(err);
                }
                push_front(&mut suspended.frame.queue, effects);
                self.run_frame(suspended.frame);
                Ok(())
            }
            (Decision::PayOrNot { if_declined, .. }, Action::Decline { .. }) => {
                let effects = std::mem::take(if_declined);
                push_front(&mut suspended.frame.queue, effects);
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
                push_front(&mut suspended.frame.queue, effects);
                self.run_frame(suspended.frame);
                Ok(())
            }
            (Decision::DiscardThenDraw { player }, Action::SelectCard { card, .. }) => {
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
            (Decision::DiscardThenDraw { .. }, Action::Decline { .. }) => {
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

    /// Move a library card to the bottom of its owner's library. Not a zone
    /// change (CR 701.26b-style reorder), so no events fire.
    pub(crate) fn put_on_bottom_of_library(&mut self, card: CardId) {
        let owner = self.state.cards[card].owner;
        let library = self
            .state
            .zones
            .zone_cards_mut(ZoneType::Library, owner);
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

fn push_front(queue: &mut VecDeque<Effect>, effects: Vec<Effect>) {
    for effect in effects.into_iter().rev() {
        queue.push_front(effect);
    }
}

fn action_player(action: &Action) -> Option<PlayerId> {
    match action {
        Action::ScryCard { player, .. }
        | Action::SelectCard { player, .. }
        | Action::Decline { player }
        | Action::PayCost { player }
        | Action::ChooseMode { player, .. } => Some(*player),
        _ => None,
    }
}
