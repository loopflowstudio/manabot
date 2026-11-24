//! Commander 2013 card behavior.

#[test]
fn foundations_permanent_angel_of_finality() {
    use crate::support::game::{five_color_setup, plant_in_graveyard};
    use crate::support::inspect::zone_of;
    use crate::support::play::{cast, pass_until, settle, spend_colorless};
    use managym::agent::action::ActionSpaceKind;
    use managym::state::game_object::{PlayerId, Target};
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Angel of Finality", "Gray Ogre"]);
    let own = plant_in_graveyard(&mut s, 0, "Forest");
    let card = plant_in_graveyard(&mut s, 1, "Gray Ogre");
    let land = plant_in_graveyard(&mut s, 1, "Forest");
    cast(&mut s, "Angel of Finality");
    spend_colorless(&mut s);
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    let mut self_target = s.clone();
    assert!(self_target.choose_target(Target::Player(PlayerId(0))));
    settle(&mut self_target);
    assert_eq!(zone_of(&self_target, own), Some(ZoneType::Exile));
    assert_eq!(zone_of(&self_target, card), Some(ZoneType::Graveyard));
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    settle(&mut s);
    assert_eq!(zone_of(&s, card), Some(ZoneType::Exile));
    assert_eq!(zone_of(&s, own), Some(ZoneType::Graveyard));
    assert_eq!(zone_of(&s, land), Some(ZoneType::Exile));
}
