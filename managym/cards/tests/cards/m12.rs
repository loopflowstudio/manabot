//! Tests for cards filed in `src/m12.rs`, in card order.

use crate::support::{
    game::{five_color_setup, plant_in_graveyard, until_end_of_turn},
    inspect::zone_of,
    play::{activate, cast, cast_owned, offers_ability, settle, target_opponent},
};
use managym::{
    flow::turn::StepKind,
    state::{effect::Modification, zone::ZoneType},
};

/// Ending the turn exiles the spell beneath the ability, skips straight to
/// cleanup, where until-end-of-turn effects end; the next turn is the
/// opponent's, on whose turn the Sundial is not offered.
#[test]
fn sundial_of_the_infinite_ends_the_turn_exiling_the_stack() {
    let mut s = five_color_setup(&["Sundial of the Infinite", "Lightning Bolt", "Gray Ogre"]);
    let sundial = s.force_permanent_on_battlefield(0, "Sundial of the Infinite");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    until_end_of_turn(
        &mut s,
        ogre,
        Modification::ModifyPt {
            power: 3,
            toughness: 3,
        },
    );
    let turn = s.game().state.turn.turn_number;
    let bolt = cast(&mut s, "Lightning Bolt");
    target_opponent(&mut s);
    assert!(offers_ability(&mut s, sundial, 0));
    activate(&mut s, sundial, 0);
    s.pass_priority();
    s.pass_priority();
    assert_eq!(zone_of(&s, bolt), Some(ZoneType::Exile));
    assert!(s.game().state.stack_objects.is_empty());
    s.assert_life(1, 20);
    assert_eq!(s.game().state.turn.turn_number, turn + 1);
    assert_eq!(s.active_player(), 1);
    assert_eq!(s.game().characteristics(ogre).power, 2);

    s.advance_to_active_step(1, StepKind::Main);
    assert!(!offers_ability(&mut s, sundial, 0));
}

/// Either player's graveyard counts, read before drawing.
#[test]
fn visions_of_beyond_draws_three_when_any_graveyard_has_twenty_cards() {
    for (planted, draws) in [(19, 1), (20, 3)] {
        let mut s = five_color_setup(&["Visions of Beyond"]);
        for index in 0..planted {
            plant_in_graveyard(&mut s, 1, ["Island", "Swamp"][index % 2]);
        }
        cast_owned(&mut s, "Visions of Beyond");
        let hand = s.zone_size(0, ZoneType::Hand);
        settle(&mut s);
        s.assert_zone_size(0, ZoneType::Hand, hand + draws);
    }
}

/// The entering creature deals the damage, so its lifelink applies.
#[test]
fn warstorm_surge_has_the_entering_creature_deal_its_power() {
    use crate::support::play::{cast, settle_targeting_opponent};
    let mut s = five_color_setup(&["Warstorm Surge", "Aardvark Sloth"]);
    s.force_permanent_on_battlefield(0, "Warstorm Surge");
    cast(&mut s, "Aardvark Sloth");
    settle_targeting_opponent(&mut s);
    s.assert_life(1, 17);
    s.assert_life(0, 23);
}
