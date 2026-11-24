//! Tests for cards filed in `src/akh.rs`, in card order.

use crate::support::game::{five_color_setup, force_new_card_in_hand, plant_in_graveyard};

/// Enigma Drake's power counts instant and sorcery cards in its
/// controller's graveyard only, on the battlefield and in hand; its
/// toughness stays the printed 4.
#[test]
fn enigma_drake_counts_instants_and_sorceries_in_your_graveyard() {
    let mut s = five_color_setup(&[
        "Enigma Drake",
        "Stroke of Midnight",
        "Feed the Swarm",
        "Gray Ogre",
    ]);
    let drake = s.force_permanent_on_battlefield(0, "Enigma Drake");
    assert_eq!(s.game().characteristics(drake).pt(), (0, 4));
    plant_in_graveyard(&mut s, 0, "Stroke of Midnight");
    plant_in_graveyard(&mut s, 0, "Feed the Swarm");
    plant_in_graveyard(&mut s, 0, "Gray Ogre");
    plant_in_graveyard(&mut s, 1, "Stroke of Midnight");
    assert_eq!(s.game().characteristics(drake).pt(), (2, 4));
    let in_hand = force_new_card_in_hand(&mut s, 0, "Enigma Drake");
    assert_eq!(s.game().object_pt(in_hand), (2, 4));
}
