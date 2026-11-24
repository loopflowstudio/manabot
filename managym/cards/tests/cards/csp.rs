//! Tests for cards filed in `src/csp.rs`, in card order.

use crate::support::game::move_card;
use crate::support::inspect::{permanent_of, zone_of};
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
    let object = crate::support::inspect::library_card(&s, 0, "Dark Depths");
    let object = move_card(&mut s, object, ZoneType::Battlefield);
    s.game_mut().scenario_refresh_priority().unwrap();
    let depths = permanent_of(&s, object).unwrap();
    assert_eq!(counters_of(&s, depths, CounterKind::Ice), 10);
    activate(&mut s, depths, 0);
    settle(&mut s);
    assert_eq!(counters_of(&s, depths, CounterKind::Ice), 9);
    set_counters(&mut s, depths, CounterKind::Ice, 1);
    activate(&mut s, depths, 0);
    settle(&mut s);
    assert_eq!(zone_of(&s, object), Some(ZoneType::Graveyard));
    let lage = permanent_named(&s, "Marit Lage");
    let view = s.game().characteristics(lage);
    assert_eq!(view.pt(), (20, 20));
    assert!(keywords(&s, lage).has(Keyword::Flying));
    assert!(keywords(&s, lage).has(Keyword::Indestructible));
    assert_eq!(s.battlefield_permanents_named(0, "Marit Lage").len(), 1);
}

#[test]
fn foundations_spell_flashfreeze() {
    crate::check_counterspell("Flashfreeze", "Gray Ogre", "Sol Ring");
}

#[test]
fn foundations_5e_deathmark() {
    use crate::support::game::five_color_setup;
    use crate::support::inspect::{card_of, offered_targets, zone_of};
    use crate::support::play::{cast, settle};
    use managym::state::game_object::Target;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Deathmark", "Savannah Lions", "Llanowar Elves", "Gray Ogre"]);
    let lions = s.force_permanent_on_battlefield(1, "Savannah Lions");
    let elves = s.force_permanent_on_battlefield(1, "Llanowar Elves");
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let card = card_of(&s, lions);
    cast(&mut s, "Deathmark");
    let offers = offered_targets(&s);
    assert!(offers.contains(&Target::Permanent(lions)));
    assert!(offers.contains(&Target::Permanent(elves)));
    assert!(!offers.contains(&Target::Permanent(ogre)));
    assert!(s.choose_target(Target::Permanent(lions)));
    settle(&mut s);
    assert_eq!(zone_of(&s, card), Some(ZoneType::Graveyard));
}
