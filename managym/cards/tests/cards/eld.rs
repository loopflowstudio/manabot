//! Tests for cards filed in `src/eld.rs`, in card order.

use crate::support::inspect::permanent_of;
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
    for (name, forests, tapped, life) in [
        ("Fabled Passage", 2, true, 20),
        ("Fabled Passage", 3, false, 20),
        ("Prismatic Vista", 0, false, 19),
    ] {
        let mut s = five_color_setup(&[name]);
        for _ in 0..forests {
            s.force_permanent_on_battlefield(0, "Forest");
        }
        play_and_settle(&mut s, name);
        let source = permanent_named(&s, name);
        activate_first(&mut s, source);
        pass_until(&mut s, ActionSpaceKind::LookAndSelect);
        let found = library_card(&s, 0, "Forest");
        select_card(&mut s, found);
        settle(&mut s);
        let id = permanent_of(&s, found).unwrap();
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
        .create_token("token.spirit", PlayerId(0), false)
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
    s.allow_priority_floats();
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

/// Wildborn Preserver's trigger chooses X only as it resolves (CR 107.3),
/// so its stacked text keeps X symbolic rather than reading an unbound 0.
#[test]
fn wildborn_preserver_trigger_keeps_x_unchosen_on_the_stack() {
    use managym::agent::observation::Observation;
    let mut s = five_color_setup(&["Wildborn Preserver", "Gray Ogre"]);
    s.force_permanent_on_battlefield(0, "Wildborn Preserver");
    crate::support::game::enters_battlefield_with_triggers(&mut s, 0, "Gray Ogre");
    s.game_mut().scenario_refresh_priority().unwrap();
    s.advance_until(
        |sc| !sc.game().state.stack_objects.is_empty(),
        "the Preserver's trigger is stacked".into(),
    );
    for viewer in [PlayerId(0), PlayerId(1)] {
        let obs = Observation::for_player(s.game(), viewer);
        let [entry] = obs.stack_objects.as_slice() else {
            panic!("one trigger: {:?}", obs.stack_objects);
        };
        let text = entry.description.clone().unwrap();
        assert!(
            text.contains("Choose X. You may pay {X}")
                && text.contains("the announced value of X")
                && !text.contains("(0)")
                && !text.contains("that many"),
            "{text}"
        );
    }
}

/// Wildborn Preserver's trigger asks for X as it resolves; paying puts X
/// counters on it through the reflexive trigger. A Human does not trigger it.
#[test]
fn wildborn_preserver_pays_x_for_x_counters_when_a_non_human_creature_enters() {
    use crate::support::{inspect::tapped, play::fork_choosing};
    use managym::state::player::ManaPayment;
    for auto in [false, true] {
        let mut s = five_color_unfunded_setup(&[
            "Wildborn Preserver",
            "Gray Ogre",
            "Aang, at the Crossroads",
        ]);
        let preserver = s.force_permanent_on_battlefield(0, "Wildborn Preserver");
        let forests = [0; 3].map(|_| s.force_permanent_on_battlefield(0, "Forest"));
        if auto {
            s.game_mut().mana_payment[0] = ManaPayment::Auto;
        }
        crate::support::game::enters_battlefield_with_triggers(
            &mut s,
            0,
            "Aang, at the Crossroads",
        );
        settle(&mut s);
        assert_eq!(counters_of(&s, preserver, CounterKind::PlusOnePlusOne), 0);

        crate::support::game::enters_battlefield_with_triggers(&mut s, 0, "Gray Ogre");
        s.game_mut().scenario_refresh_priority().unwrap();
        s.advance_until(
            |sc| sc.action_space().kind == ActionSpaceKind::Modal,
            "the Preserver asks for X".into(),
        );
        // "You may pay" is one confirmation under either setting; the lands
        // then pay without a further choice.
        let pay = |s: &mut Scenario| {
            assert_eq!(s.action_space().kind, ActionSpaceKind::PayOrNot);
            choose(s, |a| matches!(a, Action::PayCost { .. }));
            settle(s);
        };
        // A fork at the choice of X: the lands tapped and the counters follow it.
        for x in [1, 2] {
            let mut fork = fork_choosing(
                &s,
                |a| matches!(a, Action::ChooseMode { mode, .. } if *mode == x),
            );
            pay(&mut fork);
            assert_eq!(
                counters_of(&fork, preserver, CounterKind::PlusOnePlusOne),
                x as i32,
                "auto={auto}"
            );
            let spent = forests.iter().filter(|land| tapped(&fork, **land)).count();
            assert_eq!(spent, x, "auto={auto}");
        }
        choose_mode(&mut s, 3);
        pay(&mut s);
        assert_eq!(
            counters_of(&s, preserver, CounterKind::PlusOnePlusOne),
            3,
            "auto={auto}"
        );
        assert!(forests.iter().all(|land| tapped(&s, *land)), "auto={auto}");
    }
}

/// Thrill of Possibility's discard is paid as it is cast, from the other cards
/// in hand; the receipt names the discarded card, and it then draws two.
#[test]
fn thrill_of_possibility_discards_a_chosen_card_as_it_is_cast() {
    for auto in [false, true] {
        let s =
            crate::support::play::cast_discarding_a_chosen_card("Thrill of Possibility", 2, auto);
        // The Ogre and two drawn cards.
        assert_eq!(s.zone_size(0, ZoneType::Hand), 3);
    }
}

#[test]
fn printed_creature_raging_redcap() {
    use managym::state::card::Keyword;
    crate::printed_creature_combat("Raging Redcap", 1, 2, &[Keyword::DoubleStrike]);
}

#[test]
fn foundations_5e_bake_into_a_pie() {
    use crate::support::game::five_color_setup;
    use crate::support::inspect::{card_of, zone_of};
    use crate::support::play::{cast, settle};
    use managym::state::game_object::Target;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Bake into a Pie", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let card = card_of(&s, ogre);
    cast(&mut s, "Bake into a Pie");
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    assert_eq!(zone_of(&s, card), Some(ZoneType::Graveyard));
    assert_eq!(s.battlefield_permanents_named(0, "Food").len(), 1);
    assert!(s.battlefield_permanents_named(1, "Food").is_empty());
}

#[test]
fn foundations_5f_syr_alin_the_lions_claw() {
    crate::attack_pumps_the_team("Syr Alin, the Lion's Claw", (4, 4), false);
}

/// "Scorching Dragonfire deals 3 damage to target creature or planeswalker.
/// If that creature or planeswalker would die this turn, exile it instead."
#[test]
fn foundations_5g_scorching_dragonfire() {
    crate::assert_damage_exiles_creature_or_planeswalker("Scorching Dragonfire", 3);
}

#[test]
fn entry_choice_heraldic_banner() {
    use crate::support::play::{choose_entry_value, resolve_top, spend_colorless};
    let mut s = five_color_setup(&["Heraldic Banner", "Gray Ogre", "Giant Koi"]);
    let red = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let blue = s.force_permanent_on_battlefield(0, "Giant Koi");
    cast(&mut s, "Heraldic Banner");
    spend_colorless(&mut s);
    resolve_top(&mut s);
    assert_eq!(s.action_space().actions.len(), 5);
    choose_entry_value(&mut s, "Red");
    assert_eq!(s.game().characteristics(red).power, 3);
    assert_eq!(s.game().characteristics(blue).power, 5);
    let banner = permanent_named(&s, "Heraldic Banner");
    assert_eq!(
        s.game().characteristics(banner).activated_abilities[0].mana_choices(),
        Some(vec![Mana::parse("R")])
    );
}

#[test]
fn charming_prince_can_blink_its_owners_stolen_creature_until_the_next_end_step() {
    use crate::support::{
        game::enters_battlefield_with_triggers,
        inspect::{card_of, offered_targets, permanent},
        play::resolve_top,
    };
    use managym::{flow::turn::StepKind, state::game_object::Target};
    let mut s = five_color_setup(&["Charming Prince", "Agent of Treachery", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let card = card_of(&s, ogre);
    enters_battlefield_with_triggers(&mut s, 1, "Agent of Treachery");
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    assert_eq!(permanent(&s, ogre).controller, PlayerId(1));
    enters_battlefield_with_triggers(&mut s, 0, "Charming Prince");
    pass_until(&mut s, ActionSpaceKind::Modal);
    let root = s.snapshot();
    choose_mode(&mut s, 0);
    pass_until(&mut s, ActionSpaceKind::Scry);
    s.restore(&root);
    choose_mode(&mut s, 1);
    settle(&mut s);
    s.assert_life(0, 23);
    s.restore(&root);
    choose_mode(&mut s, 2);
    assert_eq!(offered_targets(&s), [Target::Permanent(ogre)]);
    assert!(s.choose_target(Target::Permanent(ogre)));
    resolve_top(&mut s);
    assert_eq!(zone_of(&s, card), Some(ZoneType::Exile));
    s.advance_to_active_step(0, StepKind::End);
    settle(&mut s);
    let returned = permanent_of(&s, card).unwrap();
    assert_ne!(returned, ogre);
    assert_eq!(permanent(&s, returned).controller, PlayerId(0));
}

/// Run Away Together needs two creatures with different controllers: with
/// only one player's creatures it cannot be cast, and the second target
/// is offered only among the other player's creatures. A control change in
/// response leaves both under one controller, so both targets are illegal
/// and neither returns. A target that leaves in response is compared by
/// its last known controller, so the other still returns.
#[test]
fn run_away_together_returns_two_creatures_controlled_by_different_players() {
    use crate::support::{
        game::{move_card, move_from_library},
        inspect::offered_targets,
        play::spend_colorless,
    };
    use managym::state::{game_object::Target, player::ManaPayment};
    #[derive(Clone, Copy, PartialEq)]
    enum Response {
        None,
        ControlChange,
        TheirsLeaves,
    }
    let mut s = five_color_setup(&["Run Away Together", "Gray Ogre"]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let spell = move_from_library(&mut s, 0, "Run Away Together", ZoneType::Hand);
    let mine = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let other_mine = s.force_permanent_on_battlefield(0, "Gray Ogre");
    assert!(!castable(&mut s, spell));
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let root = s.snapshot();
    for response in [
        Response::None,
        Response::ControlChange,
        Response::TheirsLeaves,
    ] {
        s.restore(&root);
        cast_card(&mut s, spell);
        let mut first = offered_targets(&s);
        first.sort();
        let mut all = [mine, other_mine, theirs].map(Target::Permanent).to_vec();
        all.sort();
        assert_eq!(first, all);
        assert!(s.choose_target(Target::Permanent(mine)));
        assert_eq!(offered_targets(&s), [Target::Permanent(theirs)]);
        assert!(s.choose_target(Target::Permanent(theirs)));
        spend_colorless(&mut s);
        match response {
            Response::None => {}
            // The scenario seam models a completed control change in response.
            Response::ControlChange => {
                s.game_mut().state.permanents[theirs]
                    .as_mut()
                    .unwrap()
                    .controller = PlayerId(0);
            }
            Response::TheirsLeaves => {
                let object = s.game().permanent_object(theirs).unwrap();
                move_card(&mut s, object, ZoneType::Graveyard);
            }
        }
        let hands = [0, 1].map(|player| s.zone_size(player, ZoneType::Hand));
        settle(&mut s);
        let mine_returns = response != Response::ControlChange;
        let theirs_returns = response == Response::None;
        assert_eq!(
            s.zone_size(0, ZoneType::Hand),
            hands[0] + usize::from(mine_returns)
        );
        assert_eq!(
            s.zone_size(1, ZoneType::Hand),
            hands[1] + usize::from(theirs_returns)
        );
        assert_eq!(s.game().state.permanents[mine].is_none(), mine_returns);
        assert!(s.game().state.permanents[other_mine].is_some());
    }
}
