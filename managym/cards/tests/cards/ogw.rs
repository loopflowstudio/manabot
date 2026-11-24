//! Tests for cards filed in `src/ogw.rs`.
use crate::support::{
    game::{five_color_setup, until_end_of_turn, Scenario},
    inspect::{card_of, offered_cards, zone_of},
    play::{activate, offers_ability, select_card, settle},
};
use managym::agent::observation::Observation;
use managym::state::{
    effect::Modification, game_object::PlayerId, player::ManaPayment, zone::ZoneType,
};

fn ayli_with(names: &[&str]) -> (Scenario, managym::state::game_object::PermanentId) {
    let mut all = vec!["Ayli, Eternal Pilgrim"];
    all.extend_from_slice(names);
    let mut s = five_color_setup(&all);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let ayli = s.force_permanent_on_battlefield(0, "Ayli, Eternal Pilgrim");
    (s, ayli)
}

/// The gain reads the sacrificed creature's toughness as it last existed
/// (CR 608.2h); a countered ability leaves the creature sacrificed and gains
/// nothing. Ayli cannot sacrifice itself.
#[test]
fn ayli_gains_the_last_known_toughness_of_the_sacrificed_creature() {
    for (buff, countered, gained) in [(0, false, 2), (3, false, 5), (3, true, 0)] {
        let (mut s, ayli) = ayli_with(&["Gray Ogre"]);
        let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
        if buff > 0 {
            until_end_of_turn(
                &mut s,
                ogre,
                Modification::ModifyPt {
                    power: 0,
                    toughness: buff,
                },
            );
        }
        let card = card_of(&s, ogre);
        let ayli_card = card_of(&s, ayli);
        activate(&mut s, ayli, 0);
        assert!(!offered_cards(&s).contains(&ayli_card), "another creature");
        select_card(&mut s, card);
        // Both viewers see the toughness the pending gain will read.
        for viewer in [0, 1] {
            let view = Observation::for_player(s.game(), PlayerId(viewer));
            let ability = &view.stack_objects[0];
            assert_eq!(ability.paid_amount, Some(2 + buff));
            let description = ability.description.as_deref().unwrap();
            assert!(description.contains(&format!("paid its cost: {}", 2 + buff)));
        }
        if countered {
            s.game_mut().state.stack_objects.pop();
        }
        settle(&mut s);
        s.assert_life(0, 20 + gained);
        assert_eq!(
            s.battlefield_permanents_named(0, "Gray Ogre").len(),
            0,
            "sacrificed as a cost"
        );
    }
}

/// "Activate only if you have at least 10 life more than your starting life
/// total" is checked as it is activated (CR 602.5b), against the player's
/// own starting total; losing life afterward does not stop it resolving.
#[test]
fn ayli_exiles_only_at_ten_life_above_starting_life() {
    for (starting, life, offered) in [
        (20, 20, false),
        (20, 29, false),
        (20, 30, true),
        (25, 30, false),
        (25, 35, true),
    ] {
        let (mut s, ayli) = ayli_with(&["Gray Ogre", "Hill Giant"]);
        s.force_permanent_on_battlefield(0, "Gray Ogre");
        s.game_mut().state.players[0].starting_life = starting;
        s.game_mut().scenario_set_life(PlayerId(0), life);
        assert_eq!(
            offers_ability(&mut s, ayli, 1),
            offered,
            "{starting}/{life}"
        );
    }

    let (mut s, ayli) = ayli_with(&["Gray Ogre", "Hill Giant"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let giant = s.force_permanent_on_battlefield(1, "Hill Giant");
    let giant_card = card_of(&s, giant);
    s.game_mut().scenario_set_life(PlayerId(0), 30);
    activate(&mut s, ayli, 1);
    s.choose_target_named("Hill Giant");
    let ogre_card = card_of(&s, ogre);
    select_card(&mut s, ogre_card);
    s.game_mut().scenario_set_life(PlayerId(0), 20);
    settle(&mut s);
    assert_eq!(
        zone_of(&s, crate::support::inspect::now(&s, giant_card)),
        Some(ZoneType::Exile)
    );
    assert!(s.battlefield_permanents_named(1, "Hill Giant").is_empty());
}
