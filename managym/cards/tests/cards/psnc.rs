use crate::support::{
    game::five_color_setup,
    inspect::{counters, permanent_named},
    play::{cast, settle},
};

#[test]
fn giada_counts_existing_angels_for_an_entering_changeling() {
    let mut s = five_color_setup(&[
        "Giada, Font of Hope",
        "Taurean Mauler",
        "Angel of Vitality",
        "Doubling Season",
    ]);
    s.force_permanent_on_battlefield(0, "Giada, Font of Hope");
    s.force_permanent_on_battlefield(0, "Angel of Vitality");
    s.force_permanent_on_battlefield(0, "Doubling Season");
    cast(&mut s, "Taurean Mauler");
    settle(&mut s);
    assert_eq!(counters(&s, permanent_named(&s, "Taurean Mauler")), 4);
}

#[test]
fn confiscated_giada_counts_and_funds_angels_for_her_new_controller() {
    use crate::support::{
        game::move_to_hand,
        inspect::{permanent, permanent_of},
        play::{cast_card, castable},
    };
    use managym::{
        flow::turn::StepKind,
        state::{
            game_object::{PlayerId, Target},
            player::ManaPayment,
        },
    };

    let mut s = five_color_setup(&[
        "Giada, Font of Hope",
        "Confiscate",
        "Angel of Vitality",
        "Crusader of Odric",
    ]);
    let giada = s.force_permanent_on_battlefield(1, "Giada, Font of Hope");
    for _ in 0..2 {
        s.force_permanent_on_battlefield(0, "Angel of Vitality");
    }
    cast(&mut s, "Confiscate");
    assert!(s.choose_target(Target::Permanent(giada)));
    settle(&mut s);
    assert_eq!(permanent(&s, giada).controller, PlayerId(0));

    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    // Two colorless mana and Giada can pay for the Angel, but not a non-Angel.
    s.set_player_mana_pool(0, "2");
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let angel = move_to_hand(&mut s, "Angel of Vitality");
    let other = move_to_hand(&mut s, "Crusader of Odric");
    assert!(!castable(&mut s, other));
    cast_card(&mut s, angel);
    settle(&mut s);
    assert!(permanent(&s, giada).tapped);
    assert_eq!(counters(&s, permanent_of(&s, angel).unwrap()), 3);
}
