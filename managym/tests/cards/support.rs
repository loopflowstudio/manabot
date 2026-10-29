//! Shared drivers for per-card tests: a five-color fixture, with or without
//! ample floating mana, and helpers that take legal actions rather than
//! mutating game state. `tests/rules/` mounts this module too, so it reaches
//! `Scenario` through `super::helpers`.
use super::helpers::Scenario;
use managym::{
    agent::action::{Action, ActionSpaceKind, ActionType},
    flow::turn::StepKind,
    state::{
        game_object::{CardId, PermanentId, PlayerId, Target},
        permanent::CounterKind,
        zone::ZoneType,
    },
};
use std::collections::BTreeMap;

/// Both players hold sixteen of each basic plus four of each named card, start
/// in player 0's main phase with empty hands and ample floating mana.
pub fn setup(names: &[&str]) -> Scenario {
    let mut s = setup_unfunded(names);
    s.set_player_mana_pool(0, "20WWWWUUUUBBBBRRRRGGGG");
    s.set_player_mana_pool(1, "20WWWWUUUUBBBBRRRRGGGG");
    s
}

/// [`setup`] without floating mana: tests pay costs from lands they play.
pub fn setup_unfunded(names: &[&str]) -> Scenario {
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
    s
}

/// A library copy of `name` owned by `player`.
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
        .unwrap_or_else(|| panic!("no {name} in library"))
}

pub fn has_action(s: &Scenario, predicate: impl Fn(&Action) -> bool) -> bool {
    s.action_space().actions.iter().any(predicate)
}

pub fn choose(s: &mut Scenario, predicate: impl Fn(&Action) -> bool) {
    let i = s
        .action_space()
        .actions
        .iter()
        .position(predicate)
        .unwrap_or_else(|| panic!("missing action: {:?}", s.action_space()));
    s.step_action(i);
}

pub fn mode(s: &mut Scenario, wanted: usize) {
    assert_eq!(s.action_space().kind, ActionSpaceKind::Modal);
    choose(
        s,
        |a| matches!(a, Action::ChooseMode { mode, .. } if *mode == wanted),
    );
}

/// Move a library copy of `name` to player 0's hand and refresh priority.
pub fn to_hand(s: &mut Scenario, name: &str) -> CardId {
    let id = library_card(s, 0, name);
    s.game_mut().move_card(id, ZoneType::Hand);
    s.game_mut().scenario_refresh_priority().unwrap();
    id
}

/// Play a library copy of `name` with the ordinary land action.
pub fn play_land(s: &mut Scenario, name: &str) -> PermanentId {
    let id = to_hand(s, name);
    choose(
        s,
        |a| matches!(a, Action::PlayLand { card, .. } if *card == id),
    );
    s.game().state.card_to_permanent[id].expect("land entered")
}

/// Put a library copy of `name` into player 0's hand and cast it, leaving
/// it on the stack.
pub fn cast(s: &mut Scenario, name: &str) -> CardId {
    let id = to_hand(s, name);
    cast_card(s, id);
    id
}

/// Cast `id` from whichever zone currently permits it.
pub fn cast_card(s: &mut Scenario, id: CardId) {
    s.game_mut().scenario_refresh_priority().unwrap();
    choose(
        s,
        |a| matches!(a, Action::CastSpell { card, .. } if *card == id),
    );
}

pub fn can_cast(s: &mut Scenario, id: CardId) -> bool {
    s.game_mut().scenario_refresh_priority().unwrap();
    can_cast_now(s, id)
}

/// Whether ability `index` of `source` is offered now.
pub fn can_activate(s: &mut Scenario, source: PermanentId, index: usize) -> bool {
    s.game_mut().scenario_refresh_priority().unwrap();
    has_action(s, |a| {
        matches!(a, Action::ActivateAbility { permanent, ability_index, .. }
            if *permanent == source && *ability_index == index)
    })
}

/// Whether any ability of `source` is offered now.
pub fn can_activate_any(s: &mut Scenario, source: PermanentId) -> bool {
    s.game_mut().scenario_refresh_priority().unwrap();
    has_action(
        s,
        |a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == source),
    )
}

pub fn activate(s: &mut Scenario, source: PermanentId, index: usize) {
    s.game_mut().scenario_refresh_priority().unwrap();
    choose(s, |a| {
        matches!(a, Action::ActivateAbility { permanent, ability_index, .. }
            if *permanent == source && *ability_index == index)
    });
}

/// Abilities of `card` that function from its current hand or graveyard.
pub fn can_activate_card(s: &mut Scenario, source: CardId) -> bool {
    s.game_mut().scenario_refresh_priority().unwrap();
    has_action(
        s,
        |a| matches!(a, Action::ActivateCardAbility { card, .. } if *card == source),
    )
}

pub fn activate_card(s: &mut Scenario, source: CardId, index: usize) {
    s.game_mut().scenario_refresh_priority().unwrap();
    choose(s, |a| {
        matches!(a, Action::ActivateCardAbility { card, ability_index, .. }
            if *card == source && *ability_index == index)
    });
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

/// Pass or decline until priority returns with an empty stack.
pub fn settle(s: &mut Scenario) {
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

/// Pass priority until the action space is `kind`.
pub fn pass_until(s: &mut Scenario, kind: ActionSpaceKind) {
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

/// The card a battlefield permanent represents.
pub fn card_of(s: &Scenario, id: PermanentId) -> CardId {
    permanent_state(s, id).card
}

pub fn tapped(s: &Scenario, id: PermanentId) -> bool {
    permanent_state(s, id).tapped
}

pub fn counters(s: &Scenario, id: PermanentId) -> i32 {
    counters_of(s, id, CounterKind::PlusOnePlusOne)
}

pub fn counters_of(s: &Scenario, id: PermanentId, kind: CounterKind) -> i32 {
    permanent_state(s, id).counter_count(kind)
}

pub fn ready(s: &mut Scenario, id: PermanentId) {
    s.game_mut().state.permanents[id]
        .as_mut()
        .unwrap()
        .summoning_sick = false;
}

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

/// Advance with default actions until the stack, pending events, and pending
/// triggers are empty (all triggered abilities have resolved).
pub fn resolve_all(s: &mut Scenario) {
    s.advance_until(
        |s| {
            s.game().state.stack_objects.is_empty()
                && s.game().state.pending_events.is_empty()
                && s.game().state.pending_triggers.is_empty()
                && s.action_space().kind == ActionSpaceKind::Priority
        },
        "stack should empty out".to_string(),
    );
}

pub fn deck(entries: &[(&str, usize)]) -> BTreeMap<String, usize> {
    entries
        .iter()
        .map(|(name, qty)| ((*name).to_string(), *qty))
        .collect()
}

pub fn force_lands(s: &mut Scenario, player: usize, land: &str, count: usize) {
    for _ in 0..count {
        s.force_permanent_on_battlefield(player, land);
    }
}

pub fn permanent_state(s: &Scenario, id: PermanentId) -> &managym::state::permanent::Permanent {
    s.game().state.permanents[id]
        .as_ref()
        .expect("permanent should exist")
}

pub fn hand_of(s: &Scenario, player: usize) -> Vec<CardId> {
    s.game()
        .state
        .zones
        .zone_cards(ZoneType::Hand, PlayerId(player))
        .clone()
}

/// Move every card in a player's hand back to their library (test setup —
/// removes castable noise so `PriorityCastSpell` is unambiguous).
pub fn clear_hand(s: &mut Scenario, player: usize) {
    for card in hand_of(s, player) {
        s.game_mut()
            .state
            .zones
            .move_card(card, PlayerId(player), ZoneType::Library);
    }
}

pub fn cast_only(s: &mut Scenario) {
    assert!(
        s.take_action_by_type(ActionType::PriorityCastSpell),
        "cast action should be available"
    );
}

pub fn cast_and_resolve(s: &mut Scenario) {
    cast_only(s);
    resolve(s);
}

pub fn enter_settled(s: &mut Scenario, name: &str) -> CardId {
    let id = library_card(s, 0, name);
    s.game_mut().move_card(id, ZoneType::Battlefield);
    s.game_mut().scenario_refresh_priority().unwrap();
    if !s.game().state.pending_triggers.is_empty() {
        s.pass_priority();
    }
    settle(s);
    id
}

pub fn activate_named(s: &mut Scenario, name: &str, index: usize) {
    let source = permanent(s, name);
    if s.action_space().kind == ActionSpaceKind::Priority {
        s.game_mut().scenario_refresh_priority().unwrap();
    }
    choose(
        s,
        |action| matches!(action, Action::ActivateAbility { permanent, ability_index, .. } if *permanent == source && *ability_index == index),
    );
}

pub fn ready_named(s: &mut Scenario, name: &str) {
    ready(s, permanent(s, name));
    s.game_mut().scenario_refresh_priority().unwrap();
}

/// [`can_cast`] against the current action space, without refreshing priority.
pub fn can_cast_now(s: &Scenario, id: CardId) -> bool {
    has_action(
        s,
        |a| matches!(a, Action::CastSpell { card, .. } if *card == id),
    )
}

pub fn permanent(s: &Scenario, name: &str) -> PermanentId {
    s.battlefield_permanents_named(0, name)[0]
}

/// Arrange a known library prefix without generating unrelated zone-change events.
pub fn pin_top(s: &mut Scenario, names: &[&str]) -> Vec<CardId> {
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

pub fn enter_or_cast(s: &mut Scenario, name: &str) -> CardId {
    let id = library_card(s, 0, name);
    if s.game().state.cards[id].types.is_land() {
        s.game_mut().move_card(id, ZoneType::Battlefield);
        s.game_mut().scenario_refresh_priority().unwrap();
    } else {
        cast(s, name);
    }
    settle(s);
    id
}

pub fn activate_any(s: &mut Scenario, id: PermanentId) {
    if s.action_space().kind == ActionSpaceKind::Priority {
        s.game_mut().scenario_refresh_priority().unwrap();
    }
    choose(
        s,
        |a| matches!(a,Action::ActivateAbility {permanent,..} if *permanent==id),
    );
}

pub fn target_opponent(s: &mut Scenario) {
    choose(s, |a| {
        matches!(
            a,
            Action::ChooseTarget {
                target: Target::Player(PlayerId(1)),
                ..
            }
        )
    });
}

/// Resolve damage triggers with a fixed target; fail if the stack never settles.
pub fn settle_targeting_opponent(s: &mut Scenario) {
    for _ in 0..20 {
        if s.action_space().kind == ActionSpaceKind::ChooseTarget {
            target_opponent(s);
        } else if s.action_space().kind == ActionSpaceKind::OrderTriggers {
            s.step_action(0);
        } else if s.action_space().kind == ActionSpaceKind::Priority
            && s.game().state.stack_objects.is_empty()
        {
            return;
        } else {
            s.pass_priority();
        }
    }
    panic!("damage triggers did not settle");
}

pub fn select(s: &mut Scenario, card: CardId) {
    choose(
        s,
        |a| matches!(a,Action::SelectCard {card: selected,..} if *selected==card),
    );
}

pub fn ur_lessons_deck() -> BTreeMap<String, usize> {
    [
        ("Island", 16),
        ("Mountain", 8),
        ("Gran-Gran", 3),
        ("Proft's Eidetic Memory", 3),
        ("Combustion Technique", 4),
        ("Accumulate Wisdom", 4),
        ("Pop Quiz", 2),
        ("Otter-Penguin", 2),
        ("Water Tribe Rallier", 2),
        ("Lightning Bolt", 2),
    ]
    .into_iter()
    .map(|(name, n)| (name.into(), n))
    .collect()
}

pub fn ur_lessons_setup() -> Scenario {
    let mut s = Scenario::new(ur_lessons_deck(), ur_lessons_deck(), 88);
    s.advance_to_active_step(0, StepKind::Main);
    s
}

pub fn available_card(s: &Scenario, player: usize, name: &str) -> CardId {
    s.game()
        .state
        .cards
        .iter()
        .enumerate()
        .find_map(|(i, card)| {
            (card.owner == PlayerId(player)
                && card.name == name
                && matches!(
                    s.game().state.zones.zone_of(CardId(i)),
                    Some(ZoneType::Hand | ZoneType::Library)
                ))
            .then_some(CardId(i))
        })
        .unwrap_or_else(|| panic!("no available {name}"))
}

pub fn put_permanent(s: &mut Scenario, player: usize, name: &str) -> PermanentId {
    let card = available_card(s, player, name);
    s.game_mut().move_card(card, ZoneType::Battlefield);
    s.game().state.card_to_permanent[card].unwrap()
}

pub fn cast_available(s: &mut Scenario, name: &str) -> CardId {
    let card = available_card(s, 0, name);
    s.game_mut().move_card(card, ZoneType::Hand);
    cast_card(s, card);
    card
}

pub fn resolve(s: &mut Scenario) {
    s.pass_priority();
    s.pass_priority();
}
