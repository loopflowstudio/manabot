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
