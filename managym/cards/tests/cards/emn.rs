//! Tests for cards filed in `src/emn.rs`, in card order.

use crate::support::{
    game::{clear_summoning_sickness, five_color_setup},
    play::{cast, settle, target_opponent},
};
use managym::state::{card::CardType, game_object::Target};

#[test]
fn imprisoned_in_the_moon_makes_a_colorless_land_that_taps_for_colorless() {
    let mut s = five_color_setup(&["Imprisoned in the Moon", "Agent of Treachery"]);
    let agent = s.force_permanent_on_battlefield(1, "Agent of Treachery");
    clear_summoning_sickness(&mut s, agent);
    cast(&mut s, "Imprisoned in the Moon");
    assert!(s.choose_target(Target::Permanent(agent)));
    settle(&mut s);
    let view = s.game().characteristics(agent);
    assert!(view.types.types.contains(&CardType::Land));
    assert!(!view.types.types.contains(&CardType::Creature));
    assert!(view.colors.is_empty());
    assert!(view.subtypes.is_empty());
    assert!(view.triggered_abilities.is_empty());
    let [ability] = view.activated_abilities.as_slice() else {
        panic!("one granted ability: {:?}", view.activated_abilities);
    };
    assert_eq!(
        ability.mana_choices(),
        Some(vec![managym::state::mana::Mana::parse("C")])
    );
}

/// A spell aimed only at the Dragon is copied onto each other creature its
/// caster controls; the Dragon's controller's other creatures are spared,
/// and a spell aimed at a player is not copied.
#[test]
fn mirrorwing_dragon_redirects_copies_onto_the_casters_creatures() {
    let mut s = five_color_setup(&["Mirrorwing Dragon", "Lightning Bolt", "Gray Ogre"]);
    let dragon = s.force_permanent_on_battlefield(1, "Mirrorwing Dragon");
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let mine = s.force_permanent_on_battlefield(0, "Gray Ogre");
    cast(&mut s, "Lightning Bolt");
    assert!(s.choose_target(Target::Permanent(dragon)));
    settle(&mut s);
    assert!(s.game().state.permanents[mine].is_none());
    assert!(s.game().state.permanents[theirs].is_some());
    assert_eq!(
        s.game().state.permanents[dragon].as_ref().unwrap().damage,
        3
    );

    cast(&mut s, "Lightning Bolt");
    target_opponent(&mut s);
    settle(&mut s);
    s.assert_life(1, 17);
}

/// Cemetery Recruitment returns a creature card and draws only when the
/// card it returned to the hand is a Zombie.
#[test]
fn cemetery_recruitment_draws_only_for_a_zombie() {
    use crate::support::game::plant_in_graveyard;
    use managym::state::{game_object::PlayerId, zone::ZoneType};
    for (returned, drawn) in [("Gray Ogre", 0), ("Crypt Feaster", 1)] {
        let mut s = five_color_setup(&["Cemetery Recruitment", "Gray Ogre", "Crypt Feaster"]);
        let card = plant_in_graveyard(&mut s, 0, returned);
        cast(&mut s, "Cemetery Recruitment");
        assert!(s.choose_target(Target::GraveyardCard(card)));
        settle(&mut s);
        let hand = s
            .game()
            .state
            .zones
            .zone_objects(ZoneType::Hand, PlayerId(0));
        assert_eq!(hand.len(), 1 + drawn, "{returned}");
        assert!(hand
            .iter()
            .any(|object| s.game().state[*object].name == returned));
    }
}

#[test]
fn harmless_offering_requires_both_targets_to_remain_legal() {
    use crate::support::{
        game::move_card,
        inspect::{card_of, offered_targets, permanent},
        play::spend_colorless,
    };
    use managym::state::{
        effect::{Affected, EffectDuration, Modification},
        game_object::PlayerId,
        zone::ZoneType,
    };
    let mut s = five_color_setup(&["Harmless Offering", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let land = s.force_permanent_on_battlefield(0, "Island");
    s.force_permanent_on_battlefield(1, "Forest");
    cast(&mut s, "Harmless Offering");
    assert_eq!(offered_targets(&s), [Target::Player(PlayerId(1))]);
    target_opponent(&mut s);
    assert_eq!(
        offered_targets(&s),
        [Target::Permanent(ogre), Target::Permanent(land)]
    );
    assert!(s.choose_target(Target::Permanent(ogre)));
    spend_colorless(&mut s);
    let mut gone = s.clone();
    let card = card_of(&gone, ogre);
    move_card(&mut gone, card, ZoneType::Graveyard);
    settle(&mut gone);
    assert_eq!(permanent(&gone, land).controller, PlayerId(0));

    let mut protected = s.clone();
    protected.game_mut().apply_continuous(
        None,
        PlayerId(1),
        [Affected::Player(PlayerId(1))],
        Modification::ProtectionFromEverything,
        EffectDuration::EndOfTurn,
    );
    settle(&mut protected);
    assert_eq!(permanent(&protected, ogre).controller, PlayerId(0));
    settle(&mut s);
    assert_eq!(permanent(&s, ogre).controller, PlayerId(1));
    assert_eq!(permanent(&s, land).controller, PlayerId(0));
}
