//! Tests for cards filed in `src/akh.rs`, in card order.
use crate::support::game::{
    five_color_setup, force_new_card_in_hand, move_card, plant_in_graveyard,
};
use crate::support::inspect::permanent_named;
use crate::support::play::enter_and_settle;
use managym::state::card::Keyword;
use managym::state::game_object::PlayerId;
use managym::state::zone::ZoneType;

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

#[test]
fn regal_caracal_tokens_keep_their_lifelink_after_the_lord_leaves() {
    let mut s = five_color_setup(&["Regal Caracal"]);
    let ordinary = s
        .game_mut()
        .create_token("token.cat_1_1_w", PlayerId(0))
        .unwrap();
    let opponent = s
        .game_mut()
        .create_token("token.cat_1_1_w", PlayerId(1))
        .unwrap();
    let source = enter_and_settle(&mut s, "Regal Caracal");
    let lord = s
        .game()
        .characteristics(permanent_named(&s, "Regal Caracal"));
    assert_eq!(lord.pt(), (3, 3));
    assert!(!lord.keywords.has(Keyword::Lifelink));
    assert!(!s
        .game()
        .characteristics(opponent)
        .keywords
        .has(Keyword::Lifelink));
    let cats = s.battlefield_permanents_named(0, "Cat");
    assert_eq!(cats.len(), 3);
    assert!(cats.contains(&ordinary));
    for &cat in &cats {
        let view = s.game().characteristics(cat);
        assert_eq!(view.pt(), (2, 2));
        assert!(view.keywords.has(Keyword::Lifelink));
    }
    move_card(&mut s, source, ZoneType::Graveyard);
    for cat in cats {
        let view = s.game().characteristics(cat);
        assert_eq!(view.pt(), (1, 1));
        assert_eq!(view.keywords.has(Keyword::Lifelink), cat != ordinary);
    }
}
