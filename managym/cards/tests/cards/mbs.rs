//! Tests for cards filed in `src/mbs.rs`, in card order.

use crate::support::{
    game::{five_color_setup, set_counters},
    inspect::{counters_of, owned_card, zone_of},
    play::{cast, choose_mode, settle},
};
use managym::state::{permanent::CounterKind, zone::ZoneType};

/// CR 122.1a, 704.5f, 704.5q: -1/-1 counters shrink every creature, a 0/0
/// dies, opposing counters annihilate in pairs, and the spell shuffles
/// itself into its owner's library instead of reaching the graveyard.
#[test]
fn black_sun_s_zenith_shrinks_every_creature_and_shuffles_itself_away() {
    let mut s = five_color_setup(&["Black Sun's Zenith", "Gray Ogre", "Craw Wurm"]);
    let plain = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let grown = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let partial = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let wurm = s.force_permanent_on_battlefield(0, "Craw Wurm");
    set_counters(&mut s, grown, CounterKind::PlusOnePlusOne, 2);
    set_counters(&mut s, partial, CounterKind::PlusOnePlusOne, 1);
    s.game_mut().scenario_refresh_priority().unwrap();
    let zenith = owned_card(&s, 0, "Black Sun's Zenith");
    let library = s.zone_size(0, ZoneType::Library);

    s.set_player_mana_pool(0, "BB2");
    cast(&mut s, "Black Sun's Zenith");
    choose_mode(&mut s, 2);
    settle(&mut s);

    assert!(s.game().state.permanents[plain].is_none(), "0/0 dies");
    for (id, plus, minus, pt) in [
        (grown, 0, 0, (2, 2)),
        (partial, 0, 1, (1, 1)),
        (wurm, 0, 2, (4, 2)),
    ] {
        assert_eq!(counters_of(&s, id, CounterKind::PlusOnePlusOne), plus);
        assert_eq!(counters_of(&s, id, CounterKind::MinusOneMinusOne), minus);
        let view = s.game().characteristics(id);
        assert_eq!((view.power, view.toughness), pt);
    }
    assert_eq!(zone_of(&s, zenith), Some(ZoneType::Library));
    s.assert_zone_size(0, ZoneType::Library, library);
    s.assert_zone_size(0, ZoneType::Graveyard, 0);
}
