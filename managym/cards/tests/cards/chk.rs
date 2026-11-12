//! Tests for cards filed in `src/chk.rs`, in card order.

use crate::support::{
    game::{clear_summoning_sickness, five_color_unfunded_setup, move_to_hand, Scenario},
    inspect::{card_of, tapped},
    play::{activate, cast_card, castable, choose, choose_mode},
};
use managym::{
    agent::action::{Action, ActionSpaceKind},
    state::{
        ability::Effect,
        card::ActivatedAbilityDefinition,
        game_object::{PermanentId, PlayerId},
        mana::Mana,
        zone::ZoneType,
    },
};

fn pool(s: &Scenario, player: usize) -> &Mana {
    &s.game().state.players[player].mana_pool
}

/// Heartbeat of Spring with only `lands` producing mana for player 0.
fn heartbeat_setup(heartbeat_controller: usize, lands: &[&str]) -> (Scenario, Vec<PermanentId>) {
    let mut names = vec![
        "Heartbeat of Spring",
        "Badgermole Cub",
        "Hermitic Herbalist",
    ];
    names.extend(lands.iter().filter(|name| !is_basic(name)));
    let mut s = five_color_unfunded_setup(&names);
    for permanent in s.game_mut().state.permanents.iter_mut().flatten() {
        permanent.tapped = true;
    }
    s.force_permanent_on_battlefield(heartbeat_controller, "Heartbeat of Spring");
    let lands = lands
        .iter()
        .map(|name| s.force_permanent_on_battlefield(0, name))
        .collect();
    (s, lands)
}

fn is_basic(name: &str) -> bool {
    ["Plains", "Island", "Swamp", "Mountain", "Forest"].contains(&name)
}

/// Give `land` an extra "{T}: Add [mana]" (sacrificing it when asked), a
/// shape no printed land has, to exercise several produced types.
fn add_tap_ability(s: &mut Scenario, land: PermanentId, mana: &str, sacrifice: bool) -> usize {
    let card = card_of(s, land);
    let abilities = &mut s.game_mut().state.cards[card].activated_abilities;
    abilities.push(ActivatedAbilityDefinition {
        tap_source: true,
        sacrifice_source: sacrifice,
        effects: vec![Effect::add_mana(Mana::parse(mana))],
        ..Default::default()
    });
    abilities.len() - 1
}

/// A single output is chosen automatically when skipping is on.
fn take_single_output(s: &mut Scenario) {
    if s.action_space().kind == ActionSpaceKind::Modal && s.action_space().actions.len() == 1 {
        choose_mode(s, 0);
    }
}

/// One Forest pays {1}{G}: the Heartbeat's {G} joins the Forest's while the
/// spell's cost is paid, either player's Heartbeat adds to the tapping
/// player's pool, and the bonus is not itself a tap.
#[test]
fn heartbeat_of_spring_adds_one_mana_of_a_produced_type_to_the_tapping_player() {
    for controller in [0, 1] {
        let (mut s, lands) = heartbeat_setup(controller, &["Forest"]);
        assert_eq!(
            s.game_mut().cached_producible_mana(PlayerId(0)),
            Mana::parse("GG"),
            "controller {controller}"
        );
        let cub = move_to_hand(&mut s, "Badgermole Cub");
        assert!(castable(&mut s, cub), "controller {controller}");
        cast_card(&mut s, cub);
        if s.action_space().kind == ActionSpaceKind::PayOrNot {
            choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
        }
        assert_eq!(
            s.game().state.zones.zone_of(cub),
            Some(ZoneType::Stack),
            "controller {controller}"
        );
        assert!(tapped(&s, lands[0]));
        assert_eq!(pool(&s, 0).total(), 0);
        assert_eq!(
            pool(&s, 1).total(),
            0,
            "the bonus never goes to its controller"
        );
    }
}

/// The bonus copies the type the ability actually produced: the chosen half
/// of a dual land, and colorless.
#[test]
fn heartbeat_of_spring_copies_the_chosen_or_colorless_type() {
    let (mut s, lands) = heartbeat_setup(1, &["North Pole Gates"]);
    activate(&mut s, lands[0], 1);
    choose_mode(&mut s, 1);
    assert_eq!(*pool(&s, 0), Mana::parse("UU"));

    let (mut s, _) = heartbeat_setup(0, &["Jasmine Dragon Tea Shop"]);
    assert_eq!(
        s.game_mut().cached_producible_mana(PlayerId(0)),
        Mana::parse("CC")
    );
}

/// Each Heartbeat is its own triggered mana ability: after a land produces
/// two types, the tapping player chooses each one's type separately, and a
/// choice may differ from the other's.
#[test]
fn two_heartbeats_choose_their_types_separately_after_two_types_are_produced() {
    let (mut s, lands) = heartbeat_setup(0, &["Forest"]);
    s.force_permanent_on_battlefield(1, "Heartbeat of Spring");
    let ability = add_tap_ability(&mut s, lands[0], "WU", false);
    activate(&mut s, lands[0], ability);
    take_single_output(&mut s);
    assert_eq!(*pool(&s, 0), Mana::parse("WU"));
    for (pick, expected) in [(0, "WUW"), (1, "WUWU")] {
        assert_eq!(s.action_space().kind, ActionSpaceKind::Modal);
        assert_eq!(s.action_space().actions.len(), 2, "W or U");
        choose_mode(&mut s, pick);
        assert_eq!(*pool(&s, 0), Mana::parse(expected));
    }
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
}

/// A land sacrificed to its own mana ability was still tapped for mana: the
/// Heartbeat reads it as it was when tapped.
#[test]
fn heartbeat_of_spring_triggers_for_a_land_sacrificed_in_its_mana_ability() {
    let (mut s, lands) = heartbeat_setup(0, &["Forest"]);
    let ability = add_tap_ability(&mut s, lands[0], "G", true);
    activate(&mut s, lands[0], ability);
    take_single_output(&mut s);
    assert!(s.game().state.permanents[lands[0]].is_none());
    assert_eq!(*pool(&s, 0), Mana::parse("GG"));
}

/// Tapping a nonland for mana triggers nothing; Badgermole Cub's bonus stays
/// with its own controller's creatures.
#[test]
fn heartbeat_of_spring_ignores_nonlands_and_badgermole_stays_with_its_controller() {
    let (mut s, _) = heartbeat_setup(0, &[]);
    s.force_permanent_on_battlefield(1, "Badgermole Cub");
    let herbalist = s.force_permanent_on_battlefield(0, "Hermitic Herbalist");
    clear_summoning_sickness(&mut s, herbalist);
    activate(&mut s, herbalist, 0);
    choose_mode(&mut s, 1);
    assert_eq!(*pool(&s, 0), Mana::parse("U"));
}
