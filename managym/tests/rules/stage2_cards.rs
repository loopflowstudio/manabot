// stage2_cards.rs
// Trace tests for the Stage-2 machinery (mid-resolution decisions, cast-time
// choices, cost mechanics) and the cards that exercise it: Glider Kids,
// Firebending Lesson, It'll Quench Ya!, Accumulate Wisdom, Water Tribe
// Rallier, Allies at Last, Badgermole Cub, Pop Quiz, Igneous Inspiration,
// Divide by Zero, Waterfall Aerialist (ward), Crossroads of Destiny (modal).

use std::collections::BTreeMap;

use managym::{
    agent::{
        action::{Action, ActionSpaceKind, ActionType},
        observation::Observation,
    },
    flow::turn::StepKind,
    state::{
        game_object::{CardId, PermanentId, PlayerId, Target},
        zone::ZoneType,
    },
};

use super::helpers::*;

fn deck(entries: &[(&str, usize)]) -> BTreeMap<String, usize> {
    entries
        .iter()
        .map(|(name, qty)| ((*name).to_string(), *qty))
        .collect()
}

fn force_lands(s: &mut Scenario, player: usize, land: &str, count: usize) {
    for _ in 0..count {
        s.force_permanent_on_battlefield(player, land);
    }
}

fn permanent(s: &Scenario, id: PermanentId) -> &managym::state::permanent::Permanent {
    s.game().state.permanents[id]
        .as_ref()
        .expect("permanent should exist")
}

fn library(s: &Scenario, player: usize) -> Vec<CardId> {
    s.game()
        .state
        .zones
        .zone_cards(ZoneType::Library, PlayerId(player))
        .clone()
}

fn library_top(s: &Scenario, player: usize) -> CardId {
    *library(s, player).last().expect("library should be nonempty")
}

fn hand(s: &Scenario, player: usize) -> Vec<CardId> {
    s.game()
        .state
        .zones
        .zone_cards(ZoneType::Hand, PlayerId(player))
        .clone()
}

/// Move every card in a player's hand back to their library (test setup —
/// removes castable noise so `PriorityCastSpell` is unambiguous).
fn clear_hand(s: &mut Scenario, player: usize) {
    for card in hand(s, player) {
        s.game_mut()
            .state
            .zones
            .move_card(card, PlayerId(player), ZoneType::Library);
    }
}

fn cast_only(s: &mut Scenario) {
    assert!(
        s.take_action_by_type(ActionType::PriorityCastSpell),
        "cast action should be available"
    );
}

fn cast_and_resolve(s: &mut Scenario) {
    cast_only(s);
    s.pass_priority();
    s.pass_priority();
}

fn stack_spell_card(s: &Scenario, name: &str) -> CardId {
    s.game()
        .state
        .stack_objects
        .iter()
        .find_map(|object| match object {
            managym::state::stack_object::StackObject::Spell(spell)
                if s.game().state.cards[spell.card].name == name =>
            {
                Some(spell.card)
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("no {name} spell on the stack"))
}

// ---------------------------------------------------------------------------
// Glider Kids — ETB scry 1 (SCRY decision).
// ---------------------------------------------------------------------------

fn glider_kids_scry_setup(seed: u64) -> Scenario {
    let mut s = Scenario::new(
        deck(&[("Plains", 24), ("Glider Kids", 16)]),
        deck(&[("Island", 40)]),
        seed,
    );
    force_lands(&mut s, 0, "Plains", 3);
    clear_hand(&mut s, 0);
    s.force_card_in_hand(0, "Glider Kids");
    s.advance_to_active_step(0, StepKind::Main);
    cast_and_resolve(&mut s); // creature resolves, ETB trigger on stack
    s.pass_priority();
    s.pass_priority(); // trigger resolves -> scry decision
    s
}

#[test]
fn glider_kids_scry_surfaces_binary_decision_to_controller() {
    let s = glider_kids_scry_setup(21);
    let space = s.action_space();
    assert_eq!(space.kind, ActionSpaceKind::Scry);
    assert_eq!(space.player, Some(PlayerId(0)));
    assert_eq!(space.actions.len(), 2);
    assert_eq!(space.actions[0].action_type(), ActionType::ScryKeep);
    assert_eq!(space.actions[1].action_type(), ActionType::ScryBottom);
}

#[test]
fn glider_kids_scry_bottom_moves_top_card_to_bottom() {
    let mut s = glider_kids_scry_setup(22);
    let top = library_top(&s, 0);
    assert!(s.take_action_by_type(ActionType::ScryBottom));
    let lib = library(&s, 0);
    assert_eq!(lib[0], top, "scried card should be on the bottom");
    assert_ne!(library_top(&s, 0), top);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
}

#[test]
fn glider_kids_scry_keep_leaves_top_card() {
    let mut s = glider_kids_scry_setup(23);
    let top = library_top(&s, 0);
    assert!(s.take_action_by_type(ActionType::ScryKeep));
    assert_eq!(library_top(&s, 0), top);
}

/// Glider Kids has flying (oracle: {2}{W} 2/3 "Flying / When this creature
/// enters, scry 1.") — a ground creature can't block it.
#[test]
fn glider_kids_flies_over_ground_blockers() {
    let mut s = Scenario::new(
        deck(&[("Plains", 24), ("Glider Kids", 16)]),
        deck(&[("Forest", 24), ("Llanowar Elves", 16)]),
        25,
    );
    let kids = s.force_permanent_on_battlefield(0, "Glider Kids");
    let _elf = s.force_permanent_on_battlefield(1, "Llanowar Elves");
    s.game_mut().state.permanents[kids]
        .as_mut()
        .expect("permanent should exist")
        .summoning_sick = false;

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::DeclareBlockers);

    assert!(
        !s.action_space().actions.iter().any(|action| {
            matches!(
                action,
                Action::DeclareBlocker {
                    attacker: Some(_),
                    ..
                }
            )
        }),
        "ground creature should have no legal block against the flying Glider Kids"
    );
}

/// The deciding agent sees the revealed card in its observation (zone
/// Library) and the scry actions focus it.
#[test]
fn scry_revealed_card_is_observation_encodable() {
    let mut s = glider_kids_scry_setup(24);
    let top = library_top(&s, 0);
    let top_object_id = s.game().state.cards[top].id.0 as i32;

    let events = s.drain_events();
    let obs = Observation::new(s.game(), &events);
    assert!(obs.validate());
    let revealed: Vec<_> = obs
        .agent_cards
        .iter()
        .filter(|card| card.zone == ZoneType::Library)
        .collect();
    assert_eq!(revealed.len(), 1, "exactly the revealed card is visible");
    assert_eq!(revealed[0].id, top_object_id);
    // Both scry actions focus the revealed card.
    for action in &obs.action_space.actions {
        assert_eq!(action.focus, vec![top_object_id]);
    }
}

/// Determinize during a scry pins the revealed card back on top — it is
/// known information for the deciding player.
#[test]
fn determinize_pins_scry_revealed_cards_on_top() {
    let mut s = glider_kids_scry_setup(25);
    let top = library_top(&s, 0);
    s.game_mut().determinize(PlayerId(0), 12345);
    assert_eq!(library_top(&s, 0), top);
    // The decision is still answerable.
    assert!(s.take_action_by_type(ActionType::ScryBottom));
    assert_eq!(library(&s, 0)[0], top);
}

// ---------------------------------------------------------------------------
// Firebending Lesson — kicker (cast-time PAY_OR_NOT): 2 vs 5 damage.
// ---------------------------------------------------------------------------

fn firebending_setup(seed: u64, mountains: usize) -> (Scenario, PermanentId) {
    let mut s = Scenario::new(
        deck(&[("Mountain", 24), ("Firebending Lesson", 16)]),
        deck(&[("Plains", 24), ("Wall of Stone", 16)]),
        seed,
    );
    let wall = s.force_permanent_on_battlefield(1, "Wall of Stone");
    force_lands(&mut s, 0, "Mountain", mountains);
    clear_hand(&mut s, 0);
    s.force_card_in_hand(0, "Firebending Lesson");
    s.advance_to_active_step(0, StepKind::Main);
    (s, wall)
}

#[test]
fn firebending_lesson_kicked_deals_five() {
    let (mut s, wall) = firebending_setup(31, 5);
    cast_only(&mut s);
    let space = s.action_space();
    assert_eq!(space.kind, ActionSpaceKind::PayOrNot);
    assert_eq!(space.player, Some(PlayerId(0)));
    assert!(s.take_action_by_type(ActionType::PayCost));
    assert_eq!(s.action_space().kind, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(wall)));
    // All five mountains paid for {R} + kicker {4}.
    assert_eq!(s.untapped_permanents_named(0, "Mountain"), 0);
    s.pass_priority();
    s.pass_priority();
    assert_eq!(permanent(&s, wall).damage, 5);
}

#[test]
fn firebending_lesson_unkicked_deals_two() {
    let (mut s, wall) = firebending_setup(32, 5);
    cast_only(&mut s);
    assert_eq!(s.action_space().kind, ActionSpaceKind::PayOrNot);
    assert!(s.take_action_by_type(ActionType::DeclineChoice));
    assert!(s.choose_target(Target::Permanent(wall)));
    s.pass_priority();
    s.pass_priority();
    assert_eq!(permanent(&s, wall).damage, 2);
}

/// With only enough mana for the base cost, the kicker choice is not
/// offered - casting goes straight to targeting.
#[test]
fn firebending_lesson_kicker_not_offered_when_unaffordable() {
    let (mut s, wall) = firebending_setup(33, 2);
    cast_only(&mut s);
    assert_eq!(s.action_space().kind, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(wall)));
    s.pass_priority();
    s.pass_priority();
    assert_eq!(permanent(&s, wall).damage, 2);
}

// ---------------------------------------------------------------------------
// It'll Quench Ya! — counter unless controller pays {2} (opponent-facing
// PAY_OR_NOT at resolution).
// ---------------------------------------------------------------------------

fn quench_setup(seed: u64, p1_mountains: usize) -> Scenario {
    let mut s = Scenario::new(
        deck(&[("Island", 24), ("It'll Quench Ya!", 16)]),
        deck(&[("Mountain", 24), ("Gray Ogre", 16)]),
        seed,
    );
    force_lands(&mut s, 0, "Island", 2);
    force_lands(&mut s, 1, "Mountain", p1_mountains);
    clear_hand(&mut s, 0);
    clear_hand(&mut s, 1);
    s.force_card_in_hand(0, "It'll Quench Ya!");
    s.force_card_in_hand(1, "Gray Ogre");
    s.advance_to_active_step(1, StepKind::Main);
    // P1 casts Gray Ogre, then passes; P0 responds with the counter.
    cast_only(&mut s);
    s.pass_priority(); // P1 passes
    cast_only(&mut s); // P0 casts It'll Quench Ya!
    assert_eq!(s.action_space().kind, ActionSpaceKind::ChooseTarget);
    let ogre = stack_spell_card(&s, "Gray Ogre");
    assert!(s.choose_target(Target::StackSpell(ogre)));
    s.pass_priority();
    s.pass_priority(); // It'll Quench Ya! resolves -> pay-or-not for P1
    s
}

#[test]
fn quench_controller_declines_and_spell_is_countered() {
    let mut s = quench_setup(41, 3);
    let space = s.action_space();
    assert_eq!(space.kind, ActionSpaceKind::PayOrNot);
    assert_eq!(space.player, Some(PlayerId(1)), "the spell's controller decides");
    assert!(s.take_action_by_type(ActionType::DeclineChoice));

    // Gray Ogre countered; both spells in graveyards; stack empty.
    assert_eq!(s.game().state.stack_objects.len(), 0);
    assert_eq!(s.zone_size(1, ZoneType::Graveyard), 1);
    assert_eq!(s.zone_size(0, ZoneType::Graveyard), 1);
    assert_eq!(s.battlefield_permanents_named(1, "Gray Ogre").len(), 0);
}

#[test]
fn quench_controller_pays_and_spell_resolves() {
    let mut s = quench_setup(42, 5);
    assert_eq!(s.action_space().kind, ActionSpaceKind::PayOrNot);
    assert!(s.take_action_by_type(ActionType::PayCost));
    // Ogre still on the stack; let it resolve.
    assert_eq!(s.game().state.stack_objects.len(), 1);
    s.pass_priority();
    s.pass_priority();
    assert_eq!(s.battlefield_permanents_named(1, "Gray Ogre").len(), 1);
}

#[test]
fn quench_pay_option_absent_when_controller_cannot_pay() {
    let mut s = quench_setup(43, 3);
    // All three of P1's mountains were tapped casting the Ogre.
    let space = s.action_space();
    assert_eq!(space.kind, ActionSpaceKind::PayOrNot);
    assert_eq!(space.actions.len(), 1);
    assert_eq!(space.actions[0].action_type(), ActionType::DeclineChoice);
    assert!(s.take_action_by_type(ActionType::DeclineChoice));
    assert_eq!(s.battlefield_permanents_named(1, "Gray Ogre").len(), 0);
}

// ---------------------------------------------------------------------------
// Learn (Pop Quiz / Igneous Inspiration) — OPTIONAL_DISCARD_THEN_DRAW.
// ---------------------------------------------------------------------------

#[test]
fn pop_quiz_draws_then_learn_discard_draws() {
    let mut s = Scenario::new(
        deck(&[("Island", 24), ("Pop Quiz", 16)]),
        deck(&[("Plains", 40)]),
        51,
    );
    force_lands(&mut s, 0, "Island", 3);
    clear_hand(&mut s, 0);
    s.force_card_in_hand(0, "Pop Quiz");
    s.advance_to_active_step(0, StepKind::Main);
    cast_and_resolve(&mut s);

    // "Draw a card" resolved, then the learn decision surfaces.
    let space = s.action_space();
    assert_eq!(space.kind, ActionSpaceKind::DiscardThenDraw);
    assert_eq!(space.player, Some(PlayerId(0)));
    // One action per hand card plus a decline.
    let hand_size = hand(&s, 0).len();
    assert_eq!(space.actions.len(), hand_size + 1);
    assert_eq!(
        space.actions.last().map(Action::action_type),
        Some(ActionType::DeclineChoice)
    );

    assert!(s.take_action_by_type(ActionType::SelectCard));
    // Discarded one, drew one: hand size unchanged; graveyard has the
    // discarded card + Pop Quiz.
    assert_eq!(hand(&s, 0).len(), hand_size);
    assert_eq!(s.zone_size(0, ZoneType::Graveyard), 2);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
}

#[test]
fn pop_quiz_learn_can_be_declined() {
    let mut s = Scenario::new(
        deck(&[("Island", 24), ("Pop Quiz", 16)]),
        deck(&[("Plains", 40)]),
        52,
    );
    force_lands(&mut s, 0, "Island", 3);
    clear_hand(&mut s, 0);
    s.force_card_in_hand(0, "Pop Quiz");
    s.advance_to_active_step(0, StepKind::Main);
    cast_and_resolve(&mut s);
    assert_eq!(s.action_space().kind, ActionSpaceKind::DiscardThenDraw);
    let hand_size = hand(&s, 0).len();
    assert!(s.take_action_by_type(ActionType::DeclineChoice));
    assert_eq!(hand(&s, 0).len(), hand_size, "no discard, no draw");
    assert_eq!(s.zone_size(0, ZoneType::Graveyard), 1); // just Pop Quiz
}

#[test]
fn igneous_inspiration_deals_three_then_learns() {
    let mut s = Scenario::new(
        deck(&[("Mountain", 24), ("Igneous Inspiration", 16)]),
        deck(&[("Plains", 40)]),
        53,
    );
    force_lands(&mut s, 0, "Mountain", 3);
    clear_hand(&mut s, 0);
    s.force_card_in_hand(0, "Igneous Inspiration");
    s.advance_to_active_step(0, StepKind::Main);
    cast_only(&mut s);
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    s.pass_priority();
    s.pass_priority();
    s.assert_life(1, 17);
    assert_eq!(s.action_space().kind, ActionSpaceKind::DiscardThenDraw);
    assert!(s.take_action_by_type(ActionType::DeclineChoice));
}

// ---------------------------------------------------------------------------
// Divide by Zero — bounce a SPELL or a permanent (mv >= 1), then learn.
// ---------------------------------------------------------------------------

#[test]
fn divide_by_zero_bounces_spell_from_stack() {
    let mut s = Scenario::new(
        deck(&[("Island", 24), ("Divide by Zero", 16)]),
        deck(&[("Mountain", 24), ("Gray Ogre", 16)]),
        61,
    );
    force_lands(&mut s, 0, "Island", 3);
    force_lands(&mut s, 1, "Mountain", 3);
    clear_hand(&mut s, 0);
    clear_hand(&mut s, 1);
    s.force_card_in_hand(0, "Divide by Zero");
    s.force_card_in_hand(1, "Gray Ogre");
    s.advance_to_active_step(1, StepKind::Main);
    cast_only(&mut s); // P1 casts Gray Ogre
    s.pass_priority();
    cast_only(&mut s); // P0 responds with Divide by Zero
    let ogre = stack_spell_card(&s, "Gray Ogre");
    assert!(s.choose_target(Target::StackSpell(ogre)));
    s.pass_priority();
    s.pass_priority();

    // The Ogre spell was bounced to its owner's hand, not countered.
    assert_eq!(s.game().state.zones.zone_of(ogre), Some(ZoneType::Hand));
    assert_eq!(s.zone_size(1, ZoneType::Graveyard), 0);
    // Learn rider.
    assert_eq!(s.action_space().kind, ActionSpaceKind::DiscardThenDraw);
    assert!(s.take_action_by_type(ActionType::DeclineChoice));
    assert_eq!(s.game().state.stack_objects.len(), 0);
}

#[test]
fn divide_by_zero_bounces_permanent_but_not_lands() {
    let mut s = Scenario::new(
        deck(&[("Island", 24), ("Divide by Zero", 16)]),
        deck(&[("Mountain", 24), ("Gray Ogre", 16)]),
        62,
    );
    force_lands(&mut s, 0, "Island", 3);
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    clear_hand(&mut s, 0);
    clear_hand(&mut s, 1);
    s.force_card_in_hand(0, "Divide by Zero");
    s.advance_to_active_step(0, StepKind::Main);
    cast_only(&mut s);

    // Lands (mana value 0) are not legal targets; the Ogre is.
    let space = s.action_space().clone();
    assert_eq!(space.kind, ActionSpaceKind::ChooseTarget);
    assert_eq!(space.actions.len(), 1, "only the Gray Ogre is targetable");
    assert!(s.choose_target(Target::Permanent(ogre)));
    s.pass_priority();
    s.pass_priority();
    assert_eq!(s.battlefield_permanents_named(1, "Gray Ogre").len(), 0);
    assert_eq!(s.zone_size(1, ZoneType::Hand), 1);
    assert!(s.take_action_by_type(ActionType::DeclineChoice));
}

// ---------------------------------------------------------------------------
// Accumulate Wisdom — LOOK_AND_SELECT (look 3, pick 1); threshold puts all
// three into hand with no decision.
// ---------------------------------------------------------------------------

#[test]
fn accumulate_wisdom_look_three_pick_one_rest_to_bottom() {
    let mut s = Scenario::new(
        deck(&[("Island", 24), ("Accumulate Wisdom", 16)]),
        deck(&[("Plains", 40)]),
        71,
    );
    force_lands(&mut s, 0, "Island", 2);
    clear_hand(&mut s, 0);
    s.force_card_in_hand(0, "Accumulate Wisdom");
    s.advance_to_active_step(0, StepKind::Main);
    let top3: Vec<CardId> = library(&s, 0).iter().rev().take(3).copied().collect();
    cast_and_resolve(&mut s);

    let space = s.action_space().clone();
    assert_eq!(space.kind, ActionSpaceKind::LookAndSelect);
    // Mandatory pick: three selectable cards, no decline.
    assert_eq!(space.actions.len(), 3);
    assert!(space
        .actions
        .iter()
        .all(|action| action.action_type() == ActionType::SelectCard));

    // Pick the middle looked card.
    let picked = top3[1];
    let index = space
        .actions
        .iter()
        .position(|action| matches!(action, Action::SelectCard { card, .. } if *card == picked))
        .expect("picked card should be selectable");
    s.step_action(index);

    assert_eq!(s.game().state.zones.zone_of(picked), Some(ZoneType::Hand));
    // The other two are on the bottom of the library (random order).
    let lib = library(&s, 0);
    assert!(lib[..2].contains(&top3[0]));
    assert!(lib[..2].contains(&top3[2]));
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
}

#[test]
fn accumulate_wisdom_with_three_lessons_takes_all_three_without_decision() {
    let mut s = Scenario::new(
        deck(&[("Island", 24), ("Accumulate Wisdom", 16)]),
        deck(&[("Plains", 40)]),
        72,
    );
    force_lands(&mut s, 0, "Island", 2);
    clear_hand(&mut s, 0);
    // Three Lesson cards in the graveyard (Accumulate Wisdom is a Lesson).
    for _ in 0..3 {
        let card = s
            .game()
            .state
            .zones
            .zone_cards(ZoneType::Library, PlayerId(0))
            .iter()
            .copied()
            .find(|card| s.game().state.cards[card].name == "Accumulate Wisdom")
            .expect("library should contain Accumulate Wisdom");
        s.game_mut()
            .state
            .zones
            .move_card(card, PlayerId(0), ZoneType::Graveyard);
    }
    s.force_card_in_hand(0, "Accumulate Wisdom");
    s.advance_to_active_step(0, StepKind::Main);
    let top3: Vec<CardId> = library(&s, 0).iter().rev().take(3).copied().collect();
    let hand_before = hand(&s, 0).len();
    cast_and_resolve(&mut s);

    // No decision — all three go to hand ("put all three into your hand").
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
    assert_eq!(hand(&s, 0).len(), hand_before - 1 + 3);
    for card in top3 {
        assert_eq!(s.game().state.zones.zone_of(card), Some(ZoneType::Hand));
    }
    // The resolving Lesson did not count itself (3 in graveyard, not 4,
    // at check time); it is in the graveyard afterwards.
    assert_eq!(s.zone_size(0, ZoneType::Graveyard), 4);
}

// ---------------------------------------------------------------------------
// Water Tribe Rallier — waterbend {5} activation, LOOK_AND_SELECT payload.
// ---------------------------------------------------------------------------

fn rallier_deck() -> BTreeMap<String, usize> {
    deck(&[
        ("Forest", 20),
        ("Water Tribe Rallier", 12),
        ("Badgermole Cub", 8),
    ])
}

#[test]
fn waterbend_pays_entirely_with_creature_taps() {
    let mut s = Scenario::new(rallier_deck(), deck(&[("Plains", 40)]), 81);
    let rallier = s.force_permanent_on_battlefield(0, "Water Tribe Rallier");
    let mut creatures = vec![rallier];
    for _ in 0..4 {
        creatures.push(s.force_permanent_on_battlefield(0, "Water Tribe Rallier"));
    }
    clear_hand(&mut s, 0);
    s.advance_to_active_step(0, StepKind::Main);

    // No lands — the ability is affordable only through waterbend.
    assert!(s.take_action_by_type(ActionType::PriorityActivateAbility));
    for expected_remaining in (0..5).rev() {
        let space = s.action_space().clone();
        assert_eq!(space.kind, ActionSpaceKind::Waterbend);
        // No mana available: only tap actions.
        assert!(space
            .actions
            .iter()
            .all(|action| action.action_type() == ActionType::TapForCost));
        assert!(s.take_action_by_type(ActionType::TapForCost));
        if expected_remaining > 0 {
            assert_eq!(s.action_space().kind, ActionSpaceKind::Waterbend);
        }
    }
    // All five creatures tapped; ability on the stack.
    for creature in &creatures {
        assert!(permanent(&s, *creature).tapped);
    }
    assert_eq!(s.game().state.stack_objects.len(), 1);
    s.pass_priority();
    s.pass_priority();

    // Look at the top four, may put a creature with power <= 3 into hand.
    let space = s.action_space().clone();
    assert_eq!(space.kind, ActionSpaceKind::LookAndSelect);
    assert_eq!(
        space.actions.last().map(Action::action_type),
        Some(ActionType::DeclineChoice),
        "the reveal is optional"
    );
    let top4: Vec<CardId> = library(&s, 0).iter().rev().take(4).copied().collect();
    // Selectable cards are exactly the creatures with power <= 3 among them.
    let selectable = space
        .actions
        .iter()
        .filter(|action| action.action_type() == ActionType::SelectCard)
        .count();
    let matching = top4
        .iter()
        .filter(|card| {
            let card = &s.game().state.cards[**card];
            card.types.is_creature() && card.power.unwrap_or(0) <= 3
        })
        .count();
    assert_eq!(selectable, matching);

    if selectable > 0 {
        let hand_before = hand(&s, 0).len();
        assert!(s.take_action_by_type(ActionType::SelectCard));
        assert_eq!(hand(&s, 0).len(), hand_before + 1);
    } else {
        assert!(s.take_action_by_type(ActionType::DeclineChoice));
    }
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
}

/// Waterbend taps count as tapping for mana: Badgermole Cub adds {G} for
/// each creature tapped, and that mana helps pay the remainder — with two
/// taps and the Cub bonus, only one of three forests is needed.
#[test]
fn badgermole_cub_composes_with_waterbend_payment() {
    let mut s = Scenario::new(rallier_deck(), deck(&[("Plains", 40)]), 82);
    let rallier = s.force_permanent_on_battlefield(0, "Water Tribe Rallier");
    let cub = s.force_permanent_on_battlefield(0, "Badgermole Cub");
    force_lands(&mut s, 0, "Forest", 3);
    clear_hand(&mut s, 0);
    s.advance_to_active_step(0, StepKind::Main);

    assert!(s.take_action_by_type(ActionType::PriorityActivateAbility));
    assert_eq!(s.action_space().kind, ActionSpaceKind::Waterbend);
    // Tap both creatures: pays {2} and the Cub adds {G}{G} to the pool.
    assert!(s.choose_waterbend_tap(rallier));
    assert!(s.choose_waterbend_tap(cub));
    assert_eq!(
        s.game().state.players[0].mana_pool.total(),
        2,
        "Cub added {{G}} per creature tapped for mana"
    );
    // Pay the remaining {3}: {2} from the pool + one forest.
    assert!(s.take_action_by_type(ActionType::PayCost));
    let untapped_forests = s
        .battlefield_permanents_named(0, "Forest")
        .iter()
        .filter(|id| !permanent(&s, **id).tapped)
        .count();
    assert_eq!(untapped_forests, 2, "only one forest was needed");
    assert_eq!(s.game().state.stack_objects.len(), 1);
}

// ---------------------------------------------------------------------------
// Allies at Last — affinity cost reduction + up-to-2 + 1 multi-targeting.
// ---------------------------------------------------------------------------

#[test]
fn allies_at_last_affinity_and_two_attackers() {
    let mut s = Scenario::new(
        deck(&[("Forest", 20), ("Allies at Last", 8), ("Kyoshi Warriors", 12)]),
        deck(&[("Mountain", 24), ("Gray Ogre", 16)]),
        91,
    );
    let w1 = s.force_permanent_on_battlefield(0, "Kyoshi Warriors");
    let w2 = s.force_permanent_on_battlefield(0, "Kyoshi Warriors");
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    // Two Allies -> {2}{G} costs {G}: one forest suffices.
    force_lands(&mut s, 0, "Forest", 1);
    clear_hand(&mut s, 0);
    s.force_card_in_hand(0, "Allies at Last");
    s.advance_to_active_step(0, StepKind::Main);

    cast_only(&mut s);
    // Requirement 0: up to two creatures you control (+ Decline).
    let space = s.action_space().clone();
    assert_eq!(space.kind, ActionSpaceKind::ChooseTarget);
    assert_eq!(space.actions.len(), 3, "two warriors + decline");
    assert!(s.choose_target(Target::Permanent(w1)));
    assert!(s.choose_target(Target::Permanent(w2)));
    // Requirement 1: exactly one creature an opponent controls, no decline.
    let space = s.action_space().clone();
    assert_eq!(space.actions.len(), 1);
    assert!(s.choose_target(Target::Permanent(ogre)));
    // Affinity was applied: the cast succeeded on a single forest.
    assert_eq!(s.game().state.stack_objects.len(), 1);

    s.pass_priority();
    s.pass_priority();
    // Each 3/3 warrior dealt 3 damage to the 2/2 Ogre; it died.
    assert_eq!(s.battlefield_permanents_named(1, "Gray Ogre").len(), 0);
    assert_eq!(s.zone_size(1, ZoneType::Graveyard), 1);
    s.assert_life(1, 20);
}

#[test]
fn allies_at_last_up_to_zero_attackers_is_legal() {
    let mut s = Scenario::new(
        deck(&[("Forest", 20), ("Allies at Last", 8), ("Kyoshi Warriors", 12)]),
        deck(&[("Mountain", 24), ("Gray Ogre", 16)]),
        92,
    );
    let _w1 = s.force_permanent_on_battlefield(0, "Kyoshi Warriors");
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    force_lands(&mut s, 0, "Forest", 4);
    clear_hand(&mut s, 0);
    s.force_card_in_hand(0, "Allies at Last");
    s.advance_to_active_step(0, StepKind::Main);

    cast_only(&mut s);
    // Decline immediately: zero attackers chosen.
    assert!(s.take_action_by_type(ActionType::DeclineChoice));
    assert!(s.choose_target(Target::Permanent(ogre)));
    s.pass_priority();
    s.pass_priority();
    // Nothing happened; the Ogre survives.
    assert_eq!(s.battlefield_permanents_named(1, "Gray Ogre").len(), 1);
}

// ---------------------------------------------------------------------------
// Ward (Waterfall Aerialist) — triggered PAY_OR_NOT on being targeted.
// ---------------------------------------------------------------------------

fn ward_setup(seed: u64, p1_mountains: usize) -> (Scenario, PermanentId) {
    let mut s = Scenario::new(
        deck(&[
            ("Island", 12),
            ("Mountain", 12),
            ("Waterfall Aerialist", 8),
            ("Lightning Bolt", 8),
        ]),
        deck(&[("Mountain", 24), ("Lightning Bolt", 16)]),
        seed,
    );
    let aerialist = s.force_permanent_on_battlefield(0, "Waterfall Aerialist");
    force_lands(&mut s, 1, "Mountain", p1_mountains);
    clear_hand(&mut s, 0);
    clear_hand(&mut s, 1);
    s.force_card_in_hand(1, "Lightning Bolt");
    s.advance_to_active_step(1, StepKind::Main);
    cast_only(&mut s); // P1 casts Lightning Bolt
    s.choose_target_named("Waterfall Aerialist");
    // Ward trigger goes on the stack above the bolt; let it resolve.
    s.pass_priority();
    s.pass_priority();
    (s, aerialist)
}

#[test]
fn ward_asks_spell_controller_and_counters_on_decline() {
    let (mut s, aerialist) = ward_setup(101, 3);
    let space = s.action_space().clone();
    assert_eq!(space.kind, ActionSpaceKind::PayOrNot);
    assert_eq!(space.player, Some(PlayerId(1)), "the bolt's controller pays");
    assert!(s.take_action_by_type(ActionType::DeclineChoice));

    // The bolt was countered.
    assert_eq!(s.game().state.stack_objects.len(), 0);
    assert_eq!(s.zone_size(1, ZoneType::Graveyard), 1);
    assert_eq!(permanent(&s, aerialist).damage, 0);
}

#[test]
fn ward_paid_lets_the_spell_resolve() {
    let (mut s, _aerialist) = ward_setup(102, 3);
    assert_eq!(s.action_space().kind, ActionSpaceKind::PayOrNot);
    assert!(s.take_action_by_type(ActionType::PayCost));
    s.pass_priority();
    s.pass_priority();
    // Bolt resolved: 3 damage kills the 3/1 Aerialist (lethal SBA).
    assert_eq!(s.battlefield_permanents_named(0, "Waterfall Aerialist").len(), 0);
    assert_eq!(s.zone_size(0, ZoneType::Graveyard), 1);
}

#[test]
fn ward_does_not_trigger_for_controllers_own_spell() {
    let mut s = Scenario::new(
        deck(&[
            ("Island", 12),
            ("Mountain", 12),
            ("Waterfall Aerialist", 8),
            ("Lightning Bolt", 8),
        ]),
        deck(&[("Plains", 40)]),
        103,
    );
    let _aerialist = s.force_permanent_on_battlefield(0, "Waterfall Aerialist");
    force_lands(&mut s, 0, "Mountain", 1);
    clear_hand(&mut s, 0);
    s.force_card_in_hand(0, "Lightning Bolt");
    s.advance_to_active_step(0, StepKind::Main);
    cast_only(&mut s);
    s.choose_target_named("Waterfall Aerialist");
    s.pass_priority();
    s.pass_priority();
    // No ward pay-or-not; the bolt resolved directly and killed the 3/1.
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
    assert_eq!(s.battlefield_permanents_named(0, "Waterfall Aerialist").len(), 0);
}

// ---------------------------------------------------------------------------
// Crossroads of Destiny — MODAL machinery proof.
// ---------------------------------------------------------------------------

fn crossroads_setup(seed: u64) -> Scenario {
    let mut s = Scenario::new(
        deck(&[("Island", 24), ("Crossroads of Destiny", 16)]),
        deck(&[("Plains", 40)]),
        seed,
    );
    force_lands(&mut s, 0, "Island", 1);
    clear_hand(&mut s, 0);
    s.force_card_in_hand(0, "Crossroads of Destiny");
    s.advance_to_active_step(0, StepKind::Main);
    cast_and_resolve(&mut s);
    s
}

#[test]
fn modal_mode_one_gains_life() {
    let mut s = crossroads_setup(111);
    let space = s.action_space().clone();
    assert_eq!(space.kind, ActionSpaceKind::Modal);
    assert_eq!(space.actions.len(), 2);
    assert!(space
        .actions
        .iter()
        .all(|action| action.action_type() == ActionType::ChooseMode));
    s.step_action(0);
    s.assert_life(0, 23);
}

#[test]
fn modal_mode_two_draws_a_card() {
    let mut s = crossroads_setup(112);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Modal);
    let hand_before = hand(&s, 0).len();
    s.step_action(1);
    assert_eq!(hand(&s, 0).len(), hand_before + 1);
}
