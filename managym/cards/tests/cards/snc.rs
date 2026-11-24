//! Tests for cards filed in `src/snc.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    inspect::observed,
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
    let row = observed(&s, drake);
    assert_eq!(row.name.as_deref(), Some("Legitimate Businessperson"));
    assert_eq!(row.creature_types, Some(vec!["Citizen".to_string()]));
    assert!(row.temp_subtypes.is_empty());
}

/// A copy takes copiable values only (CR 707.2): Clone copying a protected
/// Drake enters as a flying Wind Drake, not a Legitimate Businessperson.
#[test]
fn clone_copies_the_drake_under_witness_protection_not_the_citizen() {
    use crate::support::{
        inspect::permanent_of,
        play::{resolve_top, spend_colorless},
    };
    use managym::state::card::Keyword;
    let mut s = five_color_setup(&["Witness Protection", "Wind Drake", "Clone"]);
    let drake = s.force_permanent_on_battlefield(1, "Wind Drake");
    cast(&mut s, "Witness Protection");
    assert!(s.choose_target(Target::Permanent(drake)));
    settle(&mut s);
    let clone = cast(&mut s, "Clone");
    spend_colorless(&mut s);
    resolve_top(&mut s);
    assert!(s.choose_target(Target::Permanent(drake)));
    settle(&mut s);
    let copy = permanent_of(&s, clone).expect("entered");
    let view = s.game().characteristics(copy);
    assert_eq!(&*view.name, "Wind Drake");
    assert!(view.keywords.has(Keyword::Flying));
    assert_eq!(view.pt(), (2, 2));
    assert_eq!(observed(&s, copy).name, None);
    assert_eq!(
        &*s.game().characteristics(drake).name,
        "Legitimate Businessperson"
    );
}
