//! Tests for cards filed in `src/grn.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    inspect::{card_of, hand},
    play::{cast, settle},
};
use managym::state::{game_object::PlayerId, player::ManaPayment};

/// Midnight Reaper triggers for each nontoken creature its controller
/// controlled that died, itself and a land copying a creature included, and not
/// for a creature token dying beside them.
#[test]
fn midnight_reaper_pings_and_draws_for_each_nontoken_death() {
    let mut s = five_color_setup(&["Midnight Reaper", "Gray Ogre", "Fumigate"]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    s.force_permanent_on_battlefield(0, "Midnight Reaper");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    // A noncreature card copying a creature is a nontoken creature.
    let island = s.force_permanent_on_battlefield(0, "Island");
    let original = s.game().state[card_of(&s, ogre)].clone();
    let copy = card_of(&s, island);
    s.game_mut().state[copy].become_copy_of(&original, &[]);
    s.game_mut()
        .create_token("token.cat", PlayerId(0), false)
        .unwrap();
    let before = hand(&s, 0).len();
    cast(&mut s, "Fumigate");
    s.advance_until(
        |s| s.game().state.stack_objects.len() == 3,
        "three Reaper triggers".into(),
    );
    settle(&mut s);
    // Fumigate gains one for each of four creatures; each trigger deals one.
    s.assert_life(0, 21);
    assert_eq!(hand(&s, 0).len(), before + 3);
}
