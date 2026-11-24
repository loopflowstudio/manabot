//! Core Set 2021 card behavior.

#[test]
fn foundations_permanent_shipwreck_dowser() {
    use crate::support::game::{five_color_setup, plant_in_graveyard};
    use crate::support::inspect::{permanent_named, zone_of};
    use crate::support::play::{cast, pass_until, play_and_settle, settle, spend_colorless};
    use managym::agent::action::ActionSpaceKind;
    use managym::state::game_object::Target;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Shipwreck Dowser", "Gray Ogre", "Giant Growth", "Sol Ring"]);
    let wrong_type = plant_in_graveyard(&mut s, 0, "Gray Ogre");
    let theirs = plant_in_graveyard(&mut s, 1, "Giant Growth");
    let card = plant_in_graveyard(&mut s, 0, "Giant Growth");
    cast(&mut s, "Shipwreck Dowser");
    spend_colorless(&mut s);
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(!s.choose_target(Target::GraveyardCard(wrong_type)));
    assert!(!s.choose_target(Target::GraveyardCard(theirs)));
    assert!(s.choose_target(Target::GraveyardCard(card)));
    settle(&mut s);
    assert_eq!(zone_of(&s, card), Some(ZoneType::Hand));
    play_and_settle(&mut s, "Sol Ring");
    assert_eq!(
        s.game()
            .characteristics(permanent_named(&s, "Shipwreck Dowser"))
            .pt(),
        (4, 4)
    );
}
