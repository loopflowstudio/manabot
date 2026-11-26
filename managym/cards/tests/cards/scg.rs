//! Scourge card behavior.

#[test]
fn foundations_5d_fierce_empath() {
    crate::assert_optional_revealed_search(
        "Fierce Empath",
        &["Shivan Dragon", "Rune-Scarred Demon"],
        &["Gray Ogre", "Blasphemous Edict"],
    );
}

#[test]
fn dragon_mage_combat_damage_discards_both_hands_and_draws_seven() {
    use crate::support::{
        game::{clear_summoning_sickness, five_color_setup, force_new_card_in_hand},
        inspect::{hand, zone_of},
        play::{attack_with, settle},
    };
    use managym::{flow::turn::StepKind, state::zone::ZoneType};

    let mut s = five_color_setup(&["Dragon Mage", "Gray Ogre", "Hill Giant"]);
    let dragon = s.force_permanent_on_battlefield(0, "Dragon Mage");
    clear_summoning_sickness(&mut s, dragon);
    let p0_old = force_new_card_in_hand(&mut s, 0, "Gray Ogre");
    let p1_old = force_new_card_in_hand(&mut s, 1, "Hill Giant");
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, dragon);
    s.advance_to_active_step(0, StepKind::PostcombatMain);
    settle(&mut s);
    assert_eq!(s.life(1), 15);
    assert_eq!(zone_of(&s, p0_old), Some(ZoneType::Graveyard));
    assert_eq!(zone_of(&s, p1_old), Some(ZoneType::Graveyard));
    assert_eq!(hand(&s, 0).len(), 7);
    assert_eq!(hand(&s, 1).len(), 7);
}
