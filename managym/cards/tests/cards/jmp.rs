#[test]
fn token_producer_release_the_dogs() {
    use crate::support::{game::five_color_setup, play::play_and_settle};
    let mut s = five_color_setup(&["Release the Dogs"]);
    play_and_settle(&mut s, "Release the Dogs");
    crate::assert_creature_tokens(
        &s,
        "Dog",
        4,
        (1, 1),
        &[managym::state::mana::Color::White],
        &[],
    );
}

#[test]
fn token_producer_corsair_captain() {
    use crate::support::{game::five_color_setup, play::enter_and_settle};
    let mut s = five_color_setup(&["Corsair Captain", "Swab Goblin"]);
    let pirate = s.force_permanent_on_battlefield(0, "Swab Goblin");
    enter_and_settle(&mut s, "Corsair Captain");
    assert_eq!(s.game().characteristics(pirate).pt(), (3, 3));
    assert_eq!(s.battlefield_permanents_named(0, "Treasure").len(), 1);
}
