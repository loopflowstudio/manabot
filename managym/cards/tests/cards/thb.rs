//! Theros Beyond Death card behavior.

#[test]
fn foundations_permanent_soul_guide_lantern() {
    use crate::support::game::{five_color_setup, plant_in_graveyard};
    use crate::support::inspect::{card_of, permanent_named, zone_of};
    use crate::support::play::{activate, cast, pass_until, settle, spend_colorless};
    use managym::agent::action::ActionSpaceKind;
    use managym::state::game_object::Target;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Soul-Guide Lantern", "Giant Growth"]);
    let card = plant_in_graveyard(&mut s, 0, "Giant Growth");
    cast(&mut s, "Soul-Guide Lantern");
    spend_colorless(&mut s);
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::GraveyardCard(card)));
    settle(&mut s);
    assert_eq!(zone_of(&s, card), Some(ZoneType::Exile));
    let lantern = permanent_named(&s, "Soul-Guide Lantern");
    let source = card_of(&s, lantern);
    let own = plant_in_graveyard(&mut s, 0, "Forest");
    let other = plant_in_graveyard(&mut s, 1, "Island");
    let mut fork = s.clone();
    activate(&mut s, lantern, 0);
    assert_eq!(zone_of(&s, source), Some(ZoneType::Graveyard));
    settle(&mut s);
    assert_eq!(zone_of(&s, other), Some(ZoneType::Exile));
    assert_eq!(zone_of(&s, own), Some(ZoneType::Graveyard));
    let hand = fork.zone_size(0, ZoneType::Hand);
    fork.set_player_mana_pool(0, "1");
    activate(&mut fork, lantern, 1);
    assert_eq!(zone_of(&fork, source), Some(ZoneType::Graveyard));
    settle(&mut fork);
    assert_eq!(fork.zone_size(0, ZoneType::Hand), hand + 1);
    assert_eq!(zone_of(&fork, other), Some(ZoneType::Graveyard));
}
