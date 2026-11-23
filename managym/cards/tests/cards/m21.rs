//! Core Set 2021 card behavior.
use crate::support::{
    game::{five_color_setup, move_card},
    inspect::{card_of, permanent, zone_of},
    play::{activate, cast, settle},
};
use managym::state::{game_object::Target, zone::ZoneType};

#[test]
fn heartfire_immolator_deals_its_power_at_sacrifice_after_returning() {
    for expected in [2, 3] {
        let mut s = five_color_setup(&["Heartfire Immolator", "Craw Wurm", "Lightning Bolt"]);
        let source = s.force_permanent_on_battlefield(0, "Heartfire Immolator");
        let card = card_of(&s, source);
        let wurm = s.force_permanent_on_battlefield(1, "Craw Wurm");
        if expected == 3 {
            cast(&mut s, "Lightning Bolt");
            assert!(s.choose_target(Target::Player(managym::state::game_object::PlayerId(1))));
            settle(&mut s);
        }
        assert_eq!(s.game().characteristics(source).power, expected);
        activate(&mut s, source, 0);
        assert!(crate::support::inspect::offered_targets(&s).contains(&Target::Permanent(source)));
        assert!(s.choose_target(Target::Permanent(wurm)));
        assert_eq!(zone_of(&s, card), Some(ZoneType::Graveyard));
        move_card(&mut s, card, ZoneType::Battlefield);
        let obs = managym::agent::observation::Observation::for_player(
            s.game(),
            managym::state::game_object::PlayerId(1),
        );
        let [ability] = obs.stack_objects.as_slice() else {
            panic!("expected pending Immolator activation")
        };
        assert_eq!(ability.source_power, Some(expected));
        assert!(ability
            .description
            .as_ref()
            .unwrap()
            .contains(&format!("source power {expected}")));
        settle(&mut s);
        assert_eq!(permanent(&s, wurm).damage, expected);
    }
}
