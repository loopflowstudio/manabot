//! Tests for cards filed in `src/ody.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    inspect::{library_card, offered_cards, zone_of},
    play::{cast_owned, select_card, settle},
};
use managym::state::zone::ZoneType;

#[test]
fn entomb_puts_any_chosen_library_card_into_the_graveyard() {
    let mut s = five_color_setup(&["Entomb", "Gray Ogre"]);
    let ogre = library_card(&s, 0, "Gray Ogre");
    cast_owned(&mut s, "Entomb");
    for _ in 0..4 {
        if offered_cards(&s).contains(&ogre) {
            break;
        }
        s.pass_priority();
    }
    assert!(offered_cards(&s).contains(&ogre));
    let library = s.zone_size(0, ZoneType::Library);
    select_card(&mut s, ogre);
    settle(&mut s);
    assert_eq!(zone_of(&s, ogre), Some(ZoneType::Graveyard));
    s.assert_zone_size(0, ZoneType::Library, library - 1);
}

/// The caster's opponent draws: the controller when the opponent casts, the
/// opponent when the controller casts.
#[test]
fn standstill_sacrifices_and_the_casters_opponent_draws_three() {
    for caster in [0usize, 1] {
        let mut s = five_color_setup(&["Standstill", "Lightning Bolt"]);
        let standstill = s.force_permanent_on_battlefield(0, "Standstill");
        if caster == 1 {
            s.advance_to_active_step(1, managym::flow::turn::StepKind::Main);
            s.set_player_mana_pool(1, "R");
        }
        let bolt = crate::support::game::force_new_card_in_hand(&mut s, caster, "Lightning Bolt");
        s.game_mut().scenario_refresh_priority().unwrap();
        let hands = [
            s.zone_size(0, ZoneType::Hand),
            s.zone_size(1, ZoneType::Hand),
        ];
        crate::support::play::cast_card(&mut s, bolt);
        crate::support::play::target_opponent(&mut s);
        settle(&mut s);
        assert!(s.game().state.permanents[standstill].is_none());
        let opponent = 1 - caster;
        s.assert_zone_size(opponent, ZoneType::Hand, hands[opponent] + 3);
        s.assert_zone_size(caster, ZoneType::Hand, hands[caster] - 1);
    }
}
