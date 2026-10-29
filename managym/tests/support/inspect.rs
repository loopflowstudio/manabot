// inspect.rs
// Read a scenario: turn structure, life and zones, cards and permanents,
// offered actions, and the observation/command contract checks.

use managym::{
    agent::action::{Action, ActionSpace, ActionType},
    flow::event::GameEvent,
    flow::turn::{PhaseKind, StepKind},
    state::{
        card::Keywords,
        game_object::{CardId, PermanentId, PlayerId},
        permanent::{CounterKind, Permanent},
        stack_object::StackObject,
        target::Target,
        zone::ZoneType,
    },
};

use crate::support::game::Scenario;

// ---------------------------------------------------------------------------
// Game state
// ---------------------------------------------------------------------------

impl Scenario {
    pub fn drain_events(&mut self) -> Vec<GameEvent> {
        self.game.take_observation_events()
    }

    pub fn action_space(&self) -> &ActionSpace {
        self.game
            .action_space()
            .expect("scenario should have an active action space")
    }

    pub fn action_index_by_type(&self, action_type: ActionType) -> Option<usize> {
        self.action_space()
            .actions
            .iter()
            .position(|action| action.action_type() == action_type)
    }

    pub fn current_step(&self) -> StepKind {
        self.game.state.turn.current_step_kind()
    }

    pub fn current_phase(&self) -> PhaseKind {
        self.game.state.turn.current_phase_kind()
    }

    pub fn active_player(&self) -> usize {
        self.game.active_player().0
    }

    pub fn life(&self, player: usize) -> i32 {
        self.game.state.players[player].life
    }

    pub fn zone_size(&self, player: usize, zone: ZoneType) -> usize {
        self.game.state.zones.size(zone, PlayerId(player))
    }

    pub fn battlefield_permanents_named(&self, player: usize, card_name: &str) -> Vec<PermanentId> {
        self.game
            .state
            .zones
            .zone_cards(ZoneType::Battlefield, PlayerId(player))
            .iter()
            .filter_map(|card_id| {
                let card = &self.game.state.cards[*card_id];
                if card.name != card_name {
                    return None;
                }
                self.game.state.card_to_permanent[*card_id]
            })
            .collect()
    }

    /// Number of untapped battlefield permanents with the given name.
    pub fn untapped_permanents_named(&self, player: usize, card_name: &str) -> usize {
        self.battlefield_permanents_named(player, card_name)
            .into_iter()
            .filter(|id| {
                self.game.state.permanents[*id]
                    .as_ref()
                    .is_some_and(|permanent| !permanent.tapped)
            })
            .count()
    }

    pub fn assert_life(&self, player: usize, expected: i32) {
        assert_eq!(
            self.life(player),
            expected,
            "unexpected life for player {player}"
        );
    }

    pub fn assert_zone_size(&self, player: usize, zone: ZoneType, expected: usize) {
        assert_eq!(
            self.zone_size(player, zone),
            expected,
            "unexpected zone size for player {player} in {zone:?}"
        );
    }

    pub fn assert_action_available(&self, action_type: ActionType) {
        assert!(
            self.action_index_by_type(action_type).is_some(),
            "expected action {action_type:?} in {:?}",
            self.action_space().actions
        );
    }

    pub fn assert_action_not_available(&self, action_type: ActionType) {
        assert!(
            self.action_index_by_type(action_type).is_none(),
            "did not expect action {action_type:?} in {:?}",
            self.action_space().actions
        );
    }

    pub fn assert_game_over(&self) {
        assert!(self.game.is_game_over(), "game should be over");
    }

    pub fn assert_winner(&self, player: usize) {
        assert_eq!(self.game.winner_index(), Some(player));
    }
}

// ---------------------------------------------------------------------------
// Finding cards
// ---------------------------------------------------------------------------

/// The first copy owned by `player` in any zone, in card-id order.
pub fn owned_card(s: &Scenario, player: usize, name: &str) -> CardId {
    let Some(index) = s
        .game()
        .state
        .cards
        .iter()
        .position(|card| card.owner == PlayerId(player) && card.name == name)
    else {
        panic!("card {name} not found for player {player}");
    };
    CardId(index)
}

/// The first library copy owned by `player`, in card-id order.
pub fn library_card(s: &Scenario, player: usize, name: &str) -> CardId {
    s.game()
        .state
        .cards
        .iter()
        .enumerate()
        .find_map(|(i, c)| {
            (c.name == name
                && c.owner == PlayerId(player)
                && s.game().state.zones.zone_of(CardId(i)) == Some(ZoneType::Library))
            .then_some(CardId(i))
        })
        .unwrap_or_else(|| panic!("no {name}"))
}

/// The first hand or library copy owned by `player`, in card-id order.
pub fn find_owned_card(s: &Scenario, player: usize, card_name: &str) -> CardId {
    s.game()
        .state
        .cards
        .iter()
        .enumerate()
        .find_map(|(index, card)| {
            (card.owner == PlayerId(player)
                && card.name == card_name
                && matches!(
                    s.game().state.zones.zone_of(CardId(index)),
                    Some(ZoneType::Hand) | Some(ZoneType::Library)
                ))
            .then_some(CardId(index))
        })
        .unwrap_or_else(|| panic!("card {card_name} not found for player {player}"))
}

/// The first copy owned by `player` in any zone but the battlefield.
pub fn find_owned_card_off_battlefield(s: &Scenario, player: usize, name: &str) -> CardId {
    let game = s.game();
    for (index, card) in game.state.cards.iter().enumerate() {
        if card.owner != PlayerId(player) || card.name != name {
            continue;
        }
        let card_id = CardId(index);
        if game.state.zones.zone_of(card_id) != Some(ZoneType::Battlefield) {
            return card_id;
        }
    }
    panic!("no off-battlefield {name} found for player {player}");
}

pub fn hand(s: &Scenario, player: usize) -> Vec<CardId> {
    s.game()
        .state
        .zones
        .zone_cards(ZoneType::Hand, PlayerId(player))
        .clone()
}

/// The library, bottom first.
pub fn library(s: &Scenario, player: usize) -> Vec<CardId> {
    s.game()
        .state
        .zones
        .zone_cards(ZoneType::Library, PlayerId(player))
        .clone()
}

pub fn library_top(s: &Scenario, player: usize) -> CardId {
    *library(s, player)
        .last()
        .expect("library should be nonempty")
}

pub fn stack_spell_card(s: &Scenario, name: &str) -> CardId {
    s.game()
        .state
        .stack_objects
        .iter()
        .find_map(|object| match object {
            StackObject::Spell(spell) if s.game().state.cards[spell.card].name == name => {
                Some(spell.card)
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("no {name} spell on the stack"))
}

// ---------------------------------------------------------------------------
// Permanents
// ---------------------------------------------------------------------------

/// Player 0's first battlefield permanent with this name.
pub fn permanent_named(s: &Scenario, name: &str) -> PermanentId {
    s.battlefield_permanents_named(0, name)[0]
}

pub fn permanent(s: &Scenario, id: PermanentId) -> &Permanent {
    s.game().state.permanents[id]
        .as_ref()
        .expect("permanent should exist")
}

pub fn card_of(s: &Scenario, id: PermanentId) -> CardId {
    permanent(s, id).card
}

pub fn counters(s: &Scenario, id: PermanentId) -> i32 {
    permanent(s, id).plus1_counters
}

pub fn counters_of(s: &Scenario, id: PermanentId, kind: CounterKind) -> i32 {
    permanent(s, id).counter_count(kind)
}

pub fn tapped(s: &Scenario, id: PermanentId) -> bool {
    permanent(s, id).tapped
}

pub fn keywords(s: &Scenario, id: PermanentId) -> Keywords {
    let permanent = permanent(s, id);
    permanent.effective_keywords(&s.game().state.cards[permanent.card])
}

// ---------------------------------------------------------------------------
// Offered actions and the command contract
// ---------------------------------------------------------------------------

pub fn has_action(s: &Scenario, predicate: impl Fn(&Action) -> bool) -> bool {
    s.action_space().actions.iter().any(predicate)
}

pub fn offered_targets(s: &Scenario) -> Vec<Target> {
    s.action_space()
        .actions
        .iter()
        .filter_map(|a| match a {
            Action::ChooseTarget { target, .. } => Some(*target),
            _ => None,
        })
        .collect()
}

pub fn can_cast(s: &Scenario, id: CardId) -> bool {
    s.action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::CastSpell { card, .. } if *card == id))
}

/// Exercise every offered atomic continuation through the real Command
/// authority, with an independent replay and a nested exact-fork rollback.
pub fn assert_choice_contract(s: &Scenario) {
    use managym::{
        decision::Command,
        search_state::{witness, BenchCommand, BranchDriver, ClonePlusUndoDriver},
    };
    let root = witness(s.game());
    let frame = s.game().semantic_decision_frame().unwrap();
    assert_eq!(frame.offers.len(), s.action_space().actions.len());
    for offer in &frame.offers {
        assert!(offer.choices.is_empty());
        let command = Command {
            command_id: format!("elemental-choice-{}", offer.id.0),
            expected_revision: frame.revision,
            offer_id: offer.id.0,
            answers: vec![],
            object_preconditions: vec![],
        };
        let mut branch = s.game().clone();
        let mut replay = s.game().clone();
        branch.execute_semantic_command(&command).unwrap();
        replay.execute_semantic_command(&command).unwrap();
        assert_eq!(witness(&branch), witness(&replay));
        let after = witness(&branch);
        assert!(branch.execute_semantic_command(&command).is_err());
        assert_eq!(witness(&branch), after);
    }
    let driver = ClonePlusUndoDriver::default();
    for index in 0..s.action_space().actions.len() {
        let mut branch = driver.fork_exact(s.game());
        let mark = driver.mark(&mut branch);
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
        let inner = driver.mark(&mut branch);
        if !branch.action_space().unwrap().actions.is_empty() {
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
        }
        driver.rollback(&mut branch, inner);
        driver.rollback(&mut branch, mark);
        assert_eq!(driver.witness(&branch), root);
    }
    assert_eq!(witness(s.game()), root);
}
