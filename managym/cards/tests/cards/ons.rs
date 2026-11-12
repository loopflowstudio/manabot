//! Tests for cards filed in `src/ons.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    inspect::{keywords, permanent, tapped},
    play::{cast_owned, settle},
};
use managym::{
    flow::turn::StepKind,
    state::{card::Keyword, game_object::PlayerId},
};

#[test]
fn insurrection_untaps_and_steals_every_creature_with_haste_this_turn() {
    let mut s = five_color_setup(&["Insurrection", "Gray Ogre"]);
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let mine = s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.game_mut().state.permanents[theirs]
        .as_mut()
        .unwrap()
        .tapped = true;
    s.game_mut().scenario_refresh_priority().unwrap();
    cast_owned(&mut s, "Insurrection");
    settle(&mut s);
    assert!(!tapped(&s, theirs));
    assert_eq!(permanent(&s, theirs).controller, PlayerId(0));
    assert_eq!(permanent(&s, mine).controller, PlayerId(0));
    assert!(keywords(&s, theirs).has(Keyword::Haste));
    s.advance_to_active_step(1, StepKind::Upkeep);
    assert_eq!(permanent(&s, theirs).controller, PlayerId(1));
}
