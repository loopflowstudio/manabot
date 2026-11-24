//! Worldwake card behavior.

#[test]
fn foundations_grant_basilisk_collar() {
    use crate::support::game::{five_color_setup, move_card};
    use crate::support::inspect::{card_of, keywords, offered_targets};
    use crate::support::play::{activate, settle};
    use managym::state::card::Keyword;
    use managym::state::game_object::Target;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Basilisk Collar", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let enemy = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let source = s.force_permanent_on_battlefield(0, "Basilisk Collar");
    s.set_player_mana_pool(0, "2");
    activate(&mut s, source, 0);
    assert!(!offered_targets(&s).contains(&Target::Permanent(enemy)));
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    assert!(keywords(&s, ogre).has(Keyword::Deathtouch));
    assert!(!keywords(&s, enemy).has(Keyword::Deathtouch));
    assert!(keywords(&s, ogre).has(Keyword::Lifelink));
    assert!(!keywords(&s, enemy).has(Keyword::Lifelink));
    let card = card_of(&s, source);
    move_card(&mut s, card, ZoneType::Graveyard);
    assert!(!keywords(&s, ogre).has(Keyword::Deathtouch));
    assert!(!keywords(&s, ogre).has(Keyword::Lifelink));
}
