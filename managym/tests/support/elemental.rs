use crate::support::helpers::Scenario;
use managym::{
    agent::action::{Action, ActionSpaceKind},
    flow::turn::StepKind,
    state::{
        game_object::{CardId, PermanentId, PlayerId},
        zone::ZoneType,
    },
};
use std::collections::BTreeMap;

pub(crate) fn setup(names: &[&str]) -> Scenario {
    let mut deck: BTreeMap<String, usize> = ["Plains", "Island", "Swamp", "Mountain", "Forest"]
        .into_iter()
        .map(|name| (name.into(), 16))
        .collect();
    for name in names {
        deck.insert((*name).into(), 4);
    }
    let mut s = Scenario::new(deck.clone(), deck, 161);
    s.advance_to_active_step(0, StepKind::Main);
    s.game_mut().scenario_clear_hand(PlayerId(0));
    s.game_mut().scenario_clear_hand(PlayerId(1));
    s.set_player_mana_pool(0, "20WWWWUUUUBBBBRRRRGGGG");
    s.set_player_mana_pool(1, "20WWWWUUUUBBBBRRRRGGGG");
    s
}

pub(crate) fn card(s: &Scenario, player: usize, name: &str) -> CardId {
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

pub(crate) fn cast(s: &mut Scenario, name: &str) -> CardId {
    let id = card(s, 0, name);
    s.game_mut().move_card(id, ZoneType::Hand);
    s.game_mut().scenario_refresh_priority().unwrap();
    let index = s
        .action_space()
        .actions
        .iter()
        .position(|a| matches!(a,Action::CastSpell { card, .. } if *card == id))
        .unwrap_or_else(|| panic!("cannot cast {name}: {:?}", s.action_space()));
    s.step_action(index);
    id
}

pub(crate) fn settle(s: &mut Scenario) {
    for _ in 0..100 {
        if s.action_space().kind == ActionSpaceKind::Priority
            && s.game().state.stack_objects.is_empty()
            && s.game().state.suspended_decision.is_none()
        {
            return;
        }
        let index = s
            .action_space()
            .actions
            .iter()
            .position(|a| matches!(a, Action::PassPriority { .. } | Action::Decline { .. }))
            .unwrap_or(0);
        s.step_action(index);
    }
    panic!("resolution did not settle")
}

pub(crate) fn pass_until(s: &mut Scenario, kind: ActionSpaceKind) {
    for _ in 0..10 {
        if s.action_space().kind == kind {
            return;
        }
        if s.action_space().kind == ActionSpaceKind::OrderTriggers {
            // This fixture driver deliberately chooses the first offered trigger.
            s.step_action(0);
            continue;
        }
        s.pass_priority();
    }
    assert_eq!(s.action_space().kind, kind);
}

pub(crate) fn permanent(s: &Scenario, name: &str) -> PermanentId {
    s.battlefield_permanents_named(0, name)[0]
}

pub(crate) fn counters(s: &Scenario, id: PermanentId) -> i32 {
    s.game().state.permanents[id]
        .as_ref()
        .unwrap()
        .plus1_counters
}

pub(crate) fn mode(s: &mut Scenario, wanted: usize) {
    assert_eq!(s.action_space().kind, ActionSpaceKind::Modal);
    let index = s
        .action_space()
        .actions
        .iter()
        .position(|a| matches!(a, Action::ChooseMode { mode, .. } if *mode == wanted))
        .unwrap();
    s.step_action(index);
}

/// Arrange a known library prefix without generating unrelated zone-change events.
pub(crate) fn pin_top(s: &mut Scenario, names: &[&str]) -> Vec<CardId> {
    let mut cards = Vec::new();
    for name in names {
        let next = s
            .game()
            .state
            .zones
            .zone_cards(ZoneType::Library, PlayerId(0))
            .iter()
            .copied()
            .find(|id| s.game().state.cards[*id].name == *name && !cards.contains(id))
            .unwrap();
        cards.push(next);
    }
    let library = s
        .game_mut()
        .state
        .zones
        .zone_cards_mut(ZoneType::Library, PlayerId(0));
    library.retain(|id| !cards.contains(id));
    library.extend(cards.iter().rev());
    cards
}

/// Exercise every offered atomic continuation through the real Command
/// authority, with an independent replay and a nested exact-fork rollback.
pub(crate) fn assert_choice_contract(s: &Scenario) {
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
