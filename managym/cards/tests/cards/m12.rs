//! Tests for cards filed in `src/m12.rs`, in card order.

use crate::support::{
    game::{five_color_setup, plant_in_graveyard, until_end_of_turn},
    inspect::zone_of,
    play::{activate, cast, cast_owned, offers_ability, settle, spend_colorless, target_opponent},
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
    spend_colorless(&mut s);
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
    spend_colorless(&mut s);
    settle_targeting_opponent(&mut s);
    s.assert_life(1, 17);
    s.assert_life(0, 23);
}

#[test]
fn foundations_grant_swiftfoot_boots() {
    use crate::support::game::{five_color_setup, move_card};
    use crate::support::inspect::{card_of, keywords, offered_targets};
    use crate::support::play::{activate, settle};
    use managym::state::card::Keyword;
    use managym::state::game_object::Target;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Swiftfoot Boots", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let enemy = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let source = s.force_permanent_on_battlefield(0, "Swiftfoot Boots");
    s.set_player_mana_pool(0, "1");
    activate(&mut s, source, 0);
    assert!(!offered_targets(&s).contains(&Target::Permanent(enemy)));
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    assert!(keywords(&s, ogre).has(Keyword::Hexproof));
    assert!(!keywords(&s, enemy).has(Keyword::Hexproof));
    assert!(keywords(&s, ogre).has(Keyword::Haste));
    assert!(!keywords(&s, enemy).has(Keyword::Haste));
    let card = card_of(&s, source);
    move_card(&mut s, card, ZoneType::Graveyard);
    assert!(!keywords(&s, ogre).has(Keyword::Hexproof));
    assert!(!keywords(&s, ogre).has(Keyword::Haste));
}

#[test]
fn foundations_5d_rune_scarred_demon() {
    use managym::state::zone::ZoneType;
    crate::assert_required_library_search("Rune-Scarred Demon", ZoneType::Hand);
}
