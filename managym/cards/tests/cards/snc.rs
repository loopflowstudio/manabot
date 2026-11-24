//! Tests for cards filed in `src/snc.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    play::{cast, settle},
};
use managym::state::{game_object::Target, mana::Color};

/// The Citizen keeps no other creature type, color, ability or name: a
/// flying Drake becomes a green and white 1/1 Legitimate Businessperson.
#[test]
fn witness_protection_renames_a_creature_a_green_white_citizen() {
    let mut s = five_color_setup(&["Witness Protection", "Wind Drake"]);
    let drake = s.force_permanent_on_battlefield(1, "Wind Drake");
    cast(&mut s, "Witness Protection");
    assert!(s.choose_target(Target::Permanent(drake)));
    settle(&mut s);
    let view = s.game().characteristics(drake);
    assert_eq!(&*view.name, "Legitimate Businessperson");
    assert_eq!(view.subtypes, vec!["Citizen".to_string()]);
    assert_eq!(
        view.colors.iter().copied().collect::<Vec<_>>(),
        vec![Color::White, Color::Green]
    );
    assert_eq!(view.pt(), (1, 1));
    assert!(
        !view.keywords.has(managym::state::card::Keyword::Flying),
        "no flying"
    );
}
