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
    /// Learn: retrieve a Lesson, discard then draw, or do neither.
    Learn { player: PlayerId },
}

impl Decision {
    pub fn player(&self) -> PlayerId {
        match self {
            Decision::Scry { player, .. }
            | Decision::LookAndSelect { player, .. }
            | Decision::PayOrNot { player, .. }
            | Decision::Modal { player, .. }
            | Decision::Learn { player } => *player,
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
            Decision::PayOrNot { .. } | Decision::Modal { .. } | Decision::Learn { .. } => &[],
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
                self.journal_stack();
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
            (
                Decision::Scry { remaining, .. },
                Action::ScryCard {
                    card, to_bottom, ..
                },
            ) => {
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
                let definition = self.state.cards[*card].definition_id;
                self.emit(crate::flow::event::GameEvent::CardRevealed {
                    owner: *player,
                    definition,
                });
                self.move_card(*card, ZoneType::Hand);
                if self.state.zones.contains(*card, ZoneType::Hand, *player) {
                    let count = self.state.players[player.0]
                        .known_hand
                        .get(&definition)
                        .copied()
                        .unwrap_or(0);
                    self.set_known_hand_count(*player, definition, count + 1);
                }
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

fn push_front(queue: &mut VecDeque<Effect>, effects: Vec<Effect>) {
    for effect in effects.into_iter().rev() {
        queue.push_front(effect);
    }
}

fn action_player(action: &Action) -> Option<PlayerId> {
    match action {
        Action::ScryCard { player, .. }
        | Action::SelectCard { player, .. }
        | Action::LearnDiscard { player, .. }
        | Action::LearnTakeLesson { player, .. }
        | Action::Decline { player }
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
            targets: vec![],
            target_req_indices: vec![],
            context_target: None,
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
