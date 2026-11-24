//! Tests for cards filed in `src/m14.rs`, in card order.

use crate::support::{
    game::{five_color_setup, force_new_card_in_hand, Scenario},
    play::{cast_card, settle},
};
use managym::{
    flow::turn::StepKind,
    state::{
        game_object::{PlayerId, Target},
        player::ManaPayment,
    },
};

/// Mindsparker deals 2 damage to an opponent who casts a white or blue
/// instant or sorcery; a red instant, a blue creature and its controller's
/// own spells do nothing.
#[test]
fn mindsparker_burns_opponents_casting_white_or_blue_instants_and_sorceries() {
    let cast_by = |s: &mut Scenario, player: usize, name: &str, target: Option<Target>| {
        let card = force_new_card_in_hand(s, player, name);
        cast_card(s, card);
        if let Some(target) = target {
            assert!(s.choose_target(target));
        }
        settle(s);
    };
    let mut s = five_color_setup(&[
        "Mindsparker",
        "Consider",
        "Scout's Warning",
        "Lightning Bolt",
        "Spectral Sailor",
    ]);
    s.game_mut().mana_payment = [ManaPayment::Auto; 2];
    s.force_permanent_on_battlefield(0, "Mindsparker");
    cast_by(&mut s, 0, "Consider", None);
    s.assert_life(0, 20);
    s.advance_to_active_step(1, StepKind::Main);
    s.set_player_mana_pool(1, "20WWWWUUUUBBBBRRRRGGGG");
    cast_by(&mut s, 1, "Consider", None);
    s.assert_life(1, 18);
    cast_by(&mut s, 1, "Scout's Warning", None);
    s.assert_life(1, 16);
    cast_by(
        &mut s,
        1,
        "Lightning Bolt",
        Some(Target::Player(PlayerId(0))),
    );
    cast_by(&mut s, 1, "Spectral Sailor", None);
    s.assert_life(1, 16);
    s.assert_life(0, 17);
}
