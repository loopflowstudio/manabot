//! Amonkhet card behavior.
use crate::support::{
    game::{five_color_setup, move_card},
    inspect::permanent_named,
    play::enter_and_settle,
};
use managym::state::{card::Keyword, game_object::PlayerId, zone::ZoneType};

#[test]
fn regal_caracal_tokens_keep_their_lifelink_after_the_lord_leaves() {
    let mut s = five_color_setup(&["Regal Caracal"]);
    let ordinary = s
        .game_mut()
        .create_token("token.cat_1_1_w", PlayerId(0), false)
        .unwrap();
    let opponent = s
        .game_mut()
        .create_token("token.cat_1_1_w", PlayerId(1), false)
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
