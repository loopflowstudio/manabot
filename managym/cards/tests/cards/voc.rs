use crate::support::{
    game::{clear_summoning_sickness, five_color_setup, move_card},
    inspect::card_of,
    play::{attack_with, pass_until, settle},
};
use managym::{
    agent::action::{ActionSpaceKind, ActionType},
    state::{card::Keyword, zone::ZoneType},
};

#[test]
fn crossway_troublemakers_grants_attacking_vampires_keywords_and_offers_life_for_a_card() {
    let mut s = five_color_setup(&["Crossway Troublemakers", "Taurean Mauler"]);
    let source = s.force_permanent_on_battlefield(0, "Crossway Troublemakers");
    let mauler = s.force_permanent_on_battlefield(0, "Taurean Mauler");
    clear_summoning_sickness(&mut s, mauler);
    assert!(!s
        .game()
        .characteristics(mauler)
        .keywords
        .has(Keyword::Deathtouch));
    pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
    attack_with(&mut s, mauler);
    for kw in [Keyword::Deathtouch, Keyword::Lifelink] {
        assert!(s.game().characteristics(mauler).keywords.has(kw));
        assert!(!s.game().characteristics(source).keywords.has(kw));
    }
    let card = card_of(&s, mauler);
    move_card(&mut s, card, ZoneType::Graveyard);
    s.game_mut().scenario_refresh_priority().unwrap();
    pass_until(&mut s, ActionSpaceKind::PayOrNot);
    let hand = s.zone_size(0, ZoneType::Hand);
    let life = s.life(0);
    assert!(s.take_action_by_type(ActionType::PayCost));
    settle(&mut s);
    assert_eq!(s.zone_size(0, ZoneType::Hand), hand + 1);
    assert_eq!(s.life(0), life - 2);
}
