use crate::support::helpers::*;
use managym::{
    agent::action::{ActionSpaceKind, ActionType},
    flow::turn::StepKind,
    state::{
        game_object::{CardId, PermanentId, PlayerId, Target},
        zone::ZoneType,
    },
};
use std::collections::BTreeMap;

pub(crate) fn deck(entries: &[(&str, usize)]) -> BTreeMap<String, usize> {
    entries
        .iter()
        .map(|(name, qty)| ((*name).to_string(), *qty))
        .collect()
}

pub(crate) fn force_lands(s: &mut Scenario, player: usize, land: &str, count: usize) {
    for _ in 0..count {
        s.force_permanent_on_battlefield(player, land);
    }
}

pub(crate) fn permanent(s: &Scenario, id: PermanentId) -> &managym::state::permanent::Permanent {
    s.game().state.permanents[id]
        .as_ref()
        .expect("permanent should exist")
}

pub(crate) fn library(s: &Scenario, player: usize) -> Vec<CardId> {
    s.game()
        .state
        .zones
        .zone_cards(ZoneType::Library, PlayerId(player))
        .clone()
}

pub(crate) fn library_top(s: &Scenario, player: usize) -> CardId {
    *library(s, player)
        .last()
        .expect("library should be nonempty")
}

pub(crate) fn hand(s: &Scenario, player: usize) -> Vec<CardId> {
    s.game()
        .state
        .zones
        .zone_cards(ZoneType::Hand, PlayerId(player))
        .clone()
}

/// Move every card in a player's hand back to their library (test setup —
/// removes castable noise so `PriorityCastSpell` is unambiguous).
pub(crate) fn clear_hand(s: &mut Scenario, player: usize) {
    for card in hand(s, player) {
        s.game_mut()
            .state
            .zones
            .move_card(card, PlayerId(player), ZoneType::Library);
    }
}

pub(crate) fn cast_only(s: &mut Scenario) {
    assert!(
        s.take_action_by_type(ActionType::PriorityCastSpell),
        "cast action should be available"
    );
}

pub(crate) fn cast_and_resolve(s: &mut Scenario) {
    cast_only(s);
    s.pass_priority();
    s.pass_priority();
}

pub(crate) fn stack_spell_card(s: &Scenario, name: &str) -> CardId {
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

pub(crate) fn glider_kids_scry_setup(seed: u64) -> Scenario {
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

// ---------------------------------------------------------------------------
// Firebending Lesson — kicker (cast-time PAY_OR_NOT): 2 vs 5 damage.
// ---------------------------------------------------------------------------

pub(crate) fn firebending_setup(seed: u64, mountains: usize) -> (Scenario, PermanentId) {
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

// ---------------------------------------------------------------------------
// It'll Quench Ya! — counter unless controller pays {2} (opponent-facing
// PAY_OR_NOT at resolution).
// ---------------------------------------------------------------------------

pub(crate) fn quench_setup(seed: u64, p1_mountains: usize) -> Scenario {
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

// ---------------------------------------------------------------------------
// Water Tribe Rallier — waterbend {5} activation, LOOK_AND_SELECT payload.
// ---------------------------------------------------------------------------

pub(crate) fn rallier_deck() -> BTreeMap<String, usize> {
    deck(&[
        ("Forest", 20),
        ("Water Tribe Rallier", 12),
        ("Badgermole Cub", 8),
    ])
}

// ---------------------------------------------------------------------------
// Ward (Waterfall Aerialist) — triggered PAY_OR_NOT on being targeted.
// ---------------------------------------------------------------------------

pub(crate) fn ward_setup(seed: u64, p1_mountains: usize) -> (Scenario, PermanentId) {
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

// ---------------------------------------------------------------------------
// Crossroads of Destiny — MODAL machinery proof.
// ---------------------------------------------------------------------------

pub(crate) fn crossroads_setup(seed: u64) -> Scenario {
    let mut s = Scenario::new(
        deck(&[("Island", 24), ("Crossroads of Destiny", 16)]),
        deck(&[("Plains", 40)]),
        seed,
    );
    force_lands(&mut s, 0, "Island", 1);
    clear_hand(&mut s, 0);
    s.force_card_in_hand(0, "Crossroads of Destiny");
    s.advance_to_active_step(0, StepKind::Main);
    cast_only(&mut s);
    s
}
