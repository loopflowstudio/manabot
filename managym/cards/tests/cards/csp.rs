//! Tests for cards filed in `src/csp.rs`, in card order.

use crate::support::{
    game::{five_color_setup, set_counters},
    inspect::{counters_of, keywords, permanent_named},
    play::{activate, settle},
};
use managym::state::{card::Keyword, permanent::CounterKind, zone::ZoneType};

/// Dark Depths enters with ten ice counters; removing the last one triggers
/// once, sacrificing it for Marit Lage.
#[test]
fn dark_depths_becomes_marit_lage_when_its_last_ice_counter_is_removed() {
    let mut s = five_color_setup(&["Dark Depths"]);
    let card = crate::support::inspect::library_card(&s, 0, "Dark Depths");
    s.game_mut().move_card(card, ZoneType::Battlefield);
    s.game_mut().scenario_refresh_priority().unwrap();
    let depths = s.game().state.card_to_permanent[card].unwrap();
    assert_eq!(counters_of(&s, depths, CounterKind::Ice), 10);
    activate(&mut s, depths, 0);
    settle(&mut s);
    assert_eq!(counters_of(&s, depths, CounterKind::Ice), 9);
    set_counters(&mut s, depths, CounterKind::Ice, 1);
    activate(&mut s, depths, 0);
    settle(&mut s);
    assert_eq!(
        s.game().state.zones.zone_of(card),
        Some(ZoneType::Graveyard)
    );
    let lage = permanent_named(&s, "Marit Lage");
    let view = s.game().characteristics(lage);
    assert_eq!(view.pt(), (20, 20));
    assert!(keywords(&s, lage).has(Keyword::Flying));
    assert!(keywords(&s, lage).has(Keyword::Indestructible));
    assert_eq!(s.battlefield_permanents_named(0, "Marit Lage").len(), 1);
}
