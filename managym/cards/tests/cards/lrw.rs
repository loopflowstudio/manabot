#[test]
fn token_producer_imperious_perfect() {
    use crate::support::{
        game::five_color_setup,
        play::{activate, enter_and_settle, settle},
    };
    let mut s = five_color_setup(&["Imperious Perfect"]);
    enter_and_settle(&mut s, "Imperious Perfect");
    let source = crate::support::inspect::permanent_named(&s, "Imperious Perfect");
    crate::support::game::clear_summoning_sickness(&mut s, source);
    activate(&mut s, source, 0);
    settle(&mut s);
    crate::assert_creature_tokens(
        &s,
        "Elf Warrior",
        1,
        (2, 2),
        &[managym::state::mana::Color::Green],
        &[],
    );
}
