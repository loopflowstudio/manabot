//! Tests for cards filed in `src/m14.rs`, in card order.

use crate::support::inspect::{permanent_of, zone_of};
use managym::state::{game_object::PlayerId, zone::ZoneType};

#[test]
fn rise_of_the_dark_realms_takes_every_graveyard_creature_under_the_caster() {
    let (s, moved) = crate::return_from_two_graveyards("Rise of the Dark Realms");
    let state = &s.game().state;
    for object in [moved.momo, moved.sailor, moved.drake, moved.their_sailor] {
        let permanent = permanent_of(&s, object).unwrap();
        assert_eq!(
            state.permanents[permanent].as_ref().unwrap().controller,
            PlayerId(0)
        );
    }
    assert_eq!(state[moved.their_sailor].owner, PlayerId(1));
    assert_eq!(zone_of(&s, moved.bolt), Some(ZoneType::Graveyard));
    let momo = permanent_of(&s, moved.momo).unwrap();
    assert_eq!(
        s.game().characteristics(momo).power,
        4,
        "both Sailors and the Drake entered with Momo under one controller"
    );
}
