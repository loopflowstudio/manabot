//! Core Set 2021 card behavior.

use crate::support::{
    game::{five_color_setup, move_card},
    inspect::{card_of, offered_targets, permanent_of, zone_of},
    play::{activate, cast, settle},
};
use managym::state::{
    game_object::{PlayerId, Target},
    player::ManaPayment,
    zone::ZoneType,
};

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

/// Prowess pumps the Immolator to 3/3; sacrificing it deals 3 to a 3/3 Hill
/// Giant. Returning the card to the battlefield before the ability resolves
/// makes a new 2/2 appearance, and the damage still reads the sacrificed one.
#[test]
fn heartfire_immolator_deals_its_sacrificed_appearances_pumped_power() {
    let mut s = five_color_setup(&["Heartfire Immolator", "Lightning Strike", "Hill Giant"]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let immolator = s.force_permanent_on_battlefield(0, "Heartfire Immolator");
    let immolator_card = card_of(&s, immolator);
    let giant = s.force_permanent_on_battlefield(1, "Hill Giant");
    let giant_card = card_of(&s, giant);

    cast(&mut s, "Lightning Strike");
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    settle(&mut s);
    assert_eq!(s.life(1), 17);
    assert_eq!(s.game().characteristics(immolator).pt(), (3, 3), "prowess");

    activate(&mut s, immolator, 0);
    let targets = offered_targets(&s);
    assert!(targets.contains(&Target::Permanent(giant)));
    assert!(!targets.iter().any(|t| matches!(t, Target::Player(_))));
    assert!(s.choose_target(Target::Permanent(giant)));
    assert_eq!(zone_of(&s, immolator_card), Some(ZoneType::Graveyard));
    let stacked = s.snapshot();
    settle(&mut s);
    assert_eq!(zone_of(&s, giant_card), Some(ZoneType::Graveyard));

    s.restore(&stacked);
    let returned = move_card(&mut s, immolator_card, ZoneType::Battlefield);
    let returned = permanent_of(&s, returned).expect("a new appearance");
    assert_eq!(s.game().characteristics(returned).pt(), (2, 2));
    settle(&mut s);
    assert_eq!(
        zone_of(&s, giant_card),
        Some(ZoneType::Graveyard),
        "3 damage from the sacrificed 3/3, not the returned 2/2"
    );
    assert!(s.game().state.permanents[returned].is_some());
}
