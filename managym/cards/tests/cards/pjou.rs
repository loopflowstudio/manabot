use crate::support::{
    game::five_color_setup,
    inspect::{counters, permanent_named},
    play::{activate, enter_and_settle, settle},
};

#[test]
fn heroes_bane_adds_power_including_the_anthem() {
    let mut s = five_color_setup(&["Heroes' Bane", "Wilt-Leaf Liege"]);
    s.force_permanent_on_battlefield(0, "Wilt-Leaf Liege");
    enter_and_settle(&mut s, "Heroes' Bane");
    let hydra = permanent_named(&s, "Heroes' Bane");
    assert_eq!(counters(&s, hydra), 4);
    activate(&mut s, hydra, 0);
    settle(&mut s);
    assert_eq!(counters(&s, hydra), 9);
}
