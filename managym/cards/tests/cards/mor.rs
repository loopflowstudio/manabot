//! Morningtide card behavior.

#[test]
fn foundations_spell_negate() {
    crate::check_counterspell("Negate", "Sol Ring", "Gray Ogre");
}

#[test]
fn foundations_5e_kindled_fury() {
    use managym::state::card::Keyword;
    crate::assert_targeted_pump(
        "Kindled Fury",
        "Gray Ogre",
        0,
        (3, 2),
        &[Keyword::FirstStrike],
        0,
        0,
    );
}

#[test]
fn ballyrush_banneret_reduces_kithkin_or_soldier_once() {
    use crate::support::game::{five_color_unfunded_setup, move_to_hand};
    use crate::support::inspect::can_cast;
    use crate::support::play::{cast_card, settle};

    let mut s = five_color_unfunded_setup(&["Ballyrush Banneret", "Gray Ogre"]);
    s.force_permanent_on_battlefield(0, "Ballyrush Banneret");
    // Second Ballyrush Banneret is both Kithkin and Soldier ({1}{W}): costs {W}, not free!
    let second = move_to_hand(&mut s, "Ballyrush Banneret");
    let ogre = move_to_hand(&mut s, "Gray Ogre");

    s.set_player_mana_pool(0, "");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_cast(&s, second), "Kithkin Soldier is not free");

    s.set_player_mana_pool(0, "W");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_cast(&s, second), "Kithkin Soldier costs 1 less");
    assert!(!can_cast(&s, ogre), "Gray Ogre gets no discount");

    cast_card(&mut s, second);
    settle(&mut s);
    assert_eq!(
        s.battlefield_permanents_named(0, "Ballyrush Banneret")
            .len(),
        2
    );
}

#[test]
fn taurean_mauler_is_revealed_as_an_ally_and_can_grow_on_an_opposing_cast() {
    use crate::support::{
        game::{clear_summoning_sickness, five_color_setup, pin_top},
        inspect::{counters, zone_of},
        play::{cast_card, pass_until, select_card, settle},
    };
    use managym::{
        agent::action::{ActionSpaceKind, ActionType},
        flow::turn::StepKind,
        state::zone::ZoneType,
    };
    let mut s = five_color_setup(&["Taurean Mauler", "Master Piandao", "Sol Ring"]);
    let master = s.force_permanent_on_battlefield(0, "Master Piandao");
    clear_summoning_sickness(&mut s, master);
    let top = pin_top(&mut s, &["Island", "Taurean Mauler", "Plains", "Swamp"]);
    pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
    s.declare_attack();
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    select_card(&mut s, top[1]);
    settle(&mut s);
    assert_eq!(zone_of(&s, top[1]), Some(ZoneType::Hand));

    let mauler = s.force_permanent_on_battlefield(0, "Taurean Mauler");
    s.advance_to_active_step(1, StepKind::Main);
    s.set_player_mana_pool(1, "1");
    let ring = crate::support::game::move_from_library(&mut s, 1, "Sol Ring", ZoneType::Hand);
    cast_card(&mut s, ring);
    pass_until(&mut s, ActionSpaceKind::PayOrNot);
    let mut decline = s.clone();
    settle(&mut decline);
    assert_eq!(counters(&decline, mauler), 0);
    assert!(s.take_action_by_type(ActionType::PayCost));
    settle(&mut s);
    assert_eq!(counters(&s, mauler), 1);
}

#[test]
fn changeling_defines_types_before_later_type_and_ability_changes() {
    use crate::support::game::{five_color_setup, until_end_of_turn};
    use managym::state::{card::Keyword, effect::Modification};
    let mut s = five_color_setup(&["Taurean Mauler"]);
    let mauler = s.force_permanent_on_battlefield(0, "Taurean Mauler");
    until_end_of_turn(&mut s, mauler, Modification::LoseAllAbilities);
    let face = s.game().characteristics(mauler);
    assert!(!face.keywords.has(Keyword::Changeling));
    assert!(face.has_subtype("Time Lord"));
    assert!(!face.has_subtype("Forest"));
    until_end_of_turn(
        &mut s,
        mauler,
        Modification::SetCreatureTypes(vec!["Elf".into()]),
    );
    let face = s.game().characteristics(mauler);
    assert!(face.has_subtype("Elf"));
    assert!(!face.has_subtype("Vampire"));
}

#[test]
fn mauler_counts_as_an_angel_elf_advisor_and_attacking_vampire() {
    use crate::support::{
        game::{clear_summoning_sickness, five_color_setup},
        inspect::{counters, counters_of, mode_labels, permanent_named},
        play::{attack_with, cast, choose_mode, pass_until, settle},
    };
    use managym::{
        agent::action::ActionSpaceKind,
        flow::turn::StepKind,
        state::{card::Keyword, permanent::CounterKind, player::ManaPayment},
    };

    let mut s = five_color_setup(&[
        "Giada, Font of Hope",
        "Dwynen, Gilt-Leaf Daen",
        "Crossway Troublemakers",
        "Taurean Mauler",
        "Banner of Kinship",
    ]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    for name in [
        "Giada, Font of Hope",
        "Dwynen, Gilt-Leaf Daen",
        "Crossway Troublemakers",
    ] {
        s.force_permanent_on_battlefield(0, name);
    }
    cast(&mut s, "Taurean Mauler");
    settle(&mut s);
    let mauler = permanent_named(&s, "Taurean Mauler");
    assert_eq!(counters(&s, mauler), 1);
    assert_eq!(s.game().characteristics(mauler).pt(), (4, 4));

    cast(&mut s, "Banner of Kinship");
    pass_until(&mut s, ActionSpaceKind::Modal);
    choose_mode(&mut s, 0);
    let advisor = mode_labels(&s)
        .iter()
        .position(|name| name == "Choose Advisor")
        .unwrap();
    choose_mode(&mut s, advisor);
    settle(&mut s);
    let banner = permanent_named(&s, "Banner of Kinship");
    assert_eq!(counters_of(&s, banner, CounterKind::Fellowship), 1);
    assert_eq!(s.game().characteristics(mauler).pt(), (5, 5));
    assert!(!s
        .game()
        .characteristics(mauler)
        .keywords
        .has(Keyword::Lifelink));

    clear_summoning_sickness(&mut s, mauler);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, mauler);
    for keyword in [Keyword::Lifelink, Keyword::Deathtouch] {
        assert!(s.game().characteristics(mauler).keywords.has(keyword));
    }
    s.advance_to_active_step(0, StepKind::PostcombatMain);
    s.assert_life(0, 25);
    s.assert_life(1, 15);
}
