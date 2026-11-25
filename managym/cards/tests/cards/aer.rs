//! Tests for cards filed in `src/aer.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    inspect::keywords,
    play::{cast_owned, settle},
};
use managym::state::card::Keyword;

#[test]
fn heroic_intervention_protects_only_your_permanents() {
    let mut s = five_color_setup(&["Heroic Intervention", "Gray Ogre", "Sol Ring"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let ring = s.force_permanent_on_battlefield(0, "Sol Ring");
    let foe = s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.game_mut().scenario_refresh_priority().unwrap();
    cast_owned(&mut s, "Heroic Intervention");
    settle(&mut s);
    for id in [ogre, ring] {
        assert!(keywords(&s, id).has(Keyword::Hexproof));
        assert!(keywords(&s, id).has(Keyword::Indestructible));
    }
    assert!(!keywords(&s, foe).has(Keyword::Hexproof));
}

/// "{T}: Add {G}": a 1/3 that taps for green once it is no longer
/// summoning-sick.
#[test]
fn foundations_5f_druid_of_the_cowl() {
    use crate::support::{
        game::{clear_summoning_sickness, five_color_unfunded_setup},
        inspect::tapped,
        play::tap_for_mana,
    };
    use managym::{
        flow::turn::StepKind,
        state::{mana::Mana, player::ManaFloats},
    };
    let mut s = five_color_unfunded_setup(&["Druid of the Cowl"]);
    s.game_mut().mana_floats[0] = ManaFloats::All;
    let druid = s.force_permanent_on_battlefield(0, "Druid of the Cowl");
    assert_eq!(s.game().characteristics(druid).pt(), (1, 3));
    clear_summoning_sickness(&mut s, druid);
    s.advance_to_active_step(0, StepKind::Upkeep);
    tap_for_mana(&mut s, druid);
    assert!(tapped(&s, druid));
    assert_eq!(
        s.game().state.players[0].mana_pool.mana(),
        &Mana::parse("G")
    );
}
