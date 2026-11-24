//! Tests for cards filed in `src/m10.rs`, in card order.

use crate::support::{
    game::{five_color_setup, force_new_card_in_hand},
    inspect::counters_of,
    play::{cast_card, pass_until, settle},
};
use managym::{
    agent::action::{ActionSpaceKind, ActionType},
    flow::turn::StepKind,
    state::{
        game_object::{PlayerId, Target},
        permanent::CounterKind,
        player::ManaPayment,
    },
};

/// Mold Adder may grow when an opponent casts a blue or black spell: the
/// controller accepts or declines, a blue-black spell triggers it once, and a
/// red spell or its controller's own blue spell not at all.
#[test]
fn mold_adder_may_grow_when_an_opponent_casts_a_blue_or_black_spell() {
    let mut s = five_color_setup(&[
        "Mold Adder",
        "Consider",
        "Foggy Swamp Spirit Keeper",
        "Lightning Bolt",
    ]);
    s.game_mut().mana_payment = [ManaPayment::Auto; 2];
    let adder = s.force_permanent_on_battlefield(0, "Mold Adder");
    let own = force_new_card_in_hand(&mut s, 0, "Consider");
    cast_card(&mut s, own);
    assert_eq!(s.game().state.stack_objects.len(), 1);
    settle(&mut s);
    s.advance_to_active_step(1, StepKind::Main);
    s.set_player_mana_pool(1, "20WWWWUUUUBBBBRRRRGGGG");
    let consider = force_new_card_in_hand(&mut s, 1, "Consider");
    cast_card(&mut s, consider);
    pass_until(&mut s, ActionSpaceKind::PayOrNot);
    let pending = s.game().clone();
    for accept in [false, true] {
        *s.game_mut() = pending.clone();
        assert!(s.take_action_by_type(if accept {
            ActionType::PayCost
        } else {
            ActionType::DeclineChoice
        }));
        settle(&mut s);
        assert_eq!(
            counters_of(&s, adder, CounterKind::PlusOnePlusOne),
            i32::from(accept)
        );
    }
    let keeper = force_new_card_in_hand(&mut s, 1, "Foggy Swamp Spirit Keeper");
    cast_card(&mut s, keeper);
    assert_eq!(s.game().state.stack_objects.len(), 2, "one trigger");
    pass_until(&mut s, ActionSpaceKind::PayOrNot);
    assert!(s.take_action_by_type(ActionType::PayCost));
    settle(&mut s);
    assert_eq!(counters_of(&s, adder, CounterKind::PlusOnePlusOne), 2);
    let bolt = force_new_card_in_hand(&mut s, 1, "Lightning Bolt");
    cast_card(&mut s, bolt);
    assert!(s.choose_target(Target::Player(PlayerId(0))));
    assert_eq!(s.game().state.stack_objects.len(), 1);
    settle(&mut s);
}
