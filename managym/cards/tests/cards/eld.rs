//! Tests for cards filed in `src/eld.rs`, in card order.

use crate::support::{
    game::{five_color_setup, five_color_unfunded_setup, move_to_hand, Scenario},
    inspect::{counters_of, has_action, library_card, permanent_named, zone_of},
    play::{
        activate, activate_first, cast, cast_card, castable, choose, choose_mode, pass_until,
        play_and_settle, select_card, settle,
    },
};
use managym::{
    agent::action::{Action, ActionSpaceKind},
    state::{game_object::PlayerId, mana::Mana, permanent::CounterKind, zone::ZoneType},
};

#[test]
fn fetch_lands_have_distinct_entry_and_life_rules() {
    for (name, tapped, life) in [("Fabled Passage", true, 20), ("Prismatic Vista", false, 19)] {
        let mut s = five_color_setup(&[name]);
        play_and_settle(&mut s, name);
        let source = permanent_named(&s, name);
        activate_first(&mut s, source);
        pass_until(&mut s, ActionSpaceKind::LookAndSelect);
        let found = library_card(&s, 0, "Forest");
        select_card(&mut s, found);
        settle(&mut s);
        let id = s.game().state.card_to_permanent[found].unwrap();
        assert_eq!(
            s.game().state.permanents[id].as_ref().unwrap().tapped,
            tapped
        );
        assert_eq!(s.game().state.players[0].life, life);
        assert!(s.battlefield_permanents_named(0, name).is_empty());
    }
}

#[test]
fn return_of_the_wildspeaker_reads_and_pumps_only_non_humans() {
    use crate::support::{
        game::until_end_of_turn,
        play::{cast, choose_mode},
    };
    use managym::state::{effect::Modification, zone::ZoneType};
    let setup = || {
        let mut s = five_color_setup(&[
            "Return of the Wildspeaker",
            "Gray Ogre",
            "Agent of Treachery",
        ]);
        let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
        let agent = s.force_permanent_on_battlefield(0, "Agent of Treachery");
        // The Human is the strongest creature, and must not count.
        until_end_of_turn(
            &mut s,
            agent,
            Modification::ModifyPt {
                power: 5,
                toughness: 0,
            },
        );
        (s, ogre, agent)
    };
    let (mut s, _, _) = setup();
    let hand = s.zone_size(0, ZoneType::Hand);
    cast(&mut s, "Return of the Wildspeaker");
    choose_mode(&mut s, 0);
    settle(&mut s);
    s.assert_zone_size(0, ZoneType::Hand, hand + 2);

    let (mut s, ogre, agent) = setup();
    cast(&mut s, "Return of the Wildspeaker");
    choose_mode(&mut s, 1);
    settle(&mut s);
    let ogre_view = s.game().characteristics(ogre);
    assert_eq!((ogre_view.power, ogre_view.toughness), (5, 5));
    let agent_view = s.game().characteristics(agent);
    assert_eq!((agent_view.power, agent_view.toughness), (7, 3));
}

/// Take the one offered output of a single-output mana ability when the
/// scenario shows it.
fn take_single_output(s: &mut Scenario) {
    if s.action_space().kind == ActionSpaceKind::Modal {
        assert_eq!(s.action_space().actions.len(), 1);
        choose_mode(s, 0);
    }
}

/// The greatest power among your creatures comes off the generic cost; tokens
/// and the Henge's own entry do not trigger it, a nontoken creature gets a
/// counter and draws.
#[test]
fn the_great_henge_costs_less_by_greatest_power_and_rewards_nontoken_creatures() {
    let mut s = five_color_unfunded_setup(&["The Great Henge", "Hill Giant", "Lotus Cobra"]);
    let henge = move_to_hand(&mut s, "The Great Henge");
    s.set_player_mana_pool(0, "3GG");
    assert!(!castable(&mut s, henge));
    s.force_permanent_on_battlefield(0, "Hill Giant");
    s.set_player_mana_pool(0, "4GG");
    assert!(
        castable(&mut s, henge),
        "power 3 takes {{3}} off {{7}}{{G}}{{G}}"
    );
    s.set_player_mana_pool(0, "3GG");
    assert!(!castable(&mut s, henge));
    s.set_player_mana_pool(0, "4GG");
    cast_card(&mut s, henge);
    settle(&mut s);
    assert_eq!(
        s.battlefield_permanents_named(0, "The Great Henge").len(),
        1
    );

    let hand = s.zone_size(0, ZoneType::Hand);
    s.set_player_mana_pool(0, "1G");
    cast(&mut s, "Lotus Cobra");
    settle(&mut s);
    let cobra = permanent_named(&s, "Lotus Cobra");
    assert_eq!(counters_of(&s, cobra, CounterKind::PlusOnePlusOne), 1);
    assert_eq!(s.zone_size(0, ZoneType::Hand), hand + 1);

    let hand = s.zone_size(0, ZoneType::Hand);
    let spirit = s
        .game_mut()
        .create_token("Spirit", PlayerId(0), false)
        .unwrap();
    settle(&mut s);
    assert_eq!(counters_of(&s, spirit, CounterKind::PlusOnePlusOne), 0);
    assert_eq!(s.zone_size(0, ZoneType::Hand), hand);
}

/// "{T}: Add {G}{G}. You gain 2 life." is a mana ability: it never uses the
/// stack, and the life arrives once with the mana, at priority and while
/// paying a spell.
#[test]
fn the_great_henge_adds_gg_and_gains_two_life_once_per_activation() {
    let mut s = five_color_unfunded_setup(&["The Great Henge", "Heroic Intervention"]);
    let henge = s.force_permanent_on_battlefield(0, "The Great Henge");
    activate(&mut s, henge, 0);
    take_single_output(&mut s);
    assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("GG"));
    assert_eq!(s.life(0), 22);
    assert!(s.game().state.stack_objects.is_empty(), "no stack");
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);

    s.game_mut().state.permanents[henge]
        .as_mut()
        .unwrap()
        .tapped = false;
    s.set_player_mana_pool(0, "");
    let intervention = move_to_hand(&mut s, "Heroic Intervention");
    assert!(castable(&mut s, intervention), "the Henge pays {{1}}{{G}}");
    cast_card(&mut s, intervention);
    assert_ne!(s.action_space().kind, ActionSpaceKind::Priority);
    assert_eq!(s.life(0), 22, "nothing is produced before the choice");
    activate(&mut s, henge, 0);
    take_single_output(&mut s);
    assert_eq!(s.life(0), 24);
    if has_action(&s, |a| matches!(a, Action::PayCost { .. })) {
        choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    }
    assert_eq!(zone_of(&s, intervention), Some(ZoneType::Stack));
    assert_eq!(s.life(0), 24, "one more activation, one more gain");
    settle(&mut s);
    assert_eq!(s.life(0), 24);
}
