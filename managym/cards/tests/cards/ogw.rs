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

#[test]
fn printed_creature_tajuru_pathwarden() {
    use managym::state::card::Keyword;
    crate::printed_creature_combat(
        "Tajuru Pathwarden",
        5,
        4,
        &[Keyword::Vigilance, Keyword::Trample],
    );
}

#[test]
fn foundations_spell_make_a_stand() {
    use crate::support::game::five_color_setup;
    use crate::support::play::{cast, settle};
    use managym::flow::turn::StepKind;
    use managym::state::card::Keyword;
    let mut s = five_color_setup(&["Make a Stand", "Hill Giant"]);
    let subject = s.force_permanent_on_battlefield(0, "Hill Giant");
    cast(&mut s, "Make a Stand");
    settle(&mut s);
    assert_eq!(s.game().characteristics(subject).pt(), (4, 3));
    assert!(s
        .game()
        .characteristics(subject)
        .keywords
        .has(Keyword::Indestructible));
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.game().characteristics(subject).pt(), (3, 3));
    assert!(!s
        .game()
        .characteristics(subject)
        .keywords
        .has(Keyword::Indestructible));
}

#[test]
fn foundations_grant_untamed_hunger() {
    use managym::state::card::Keyword;
    crate::check_attachment_grant("Untamed Hunger", None, (4, 3), &[Keyword::Menace]);
}

#[test]
fn sphinx_keeps_its_controllers_instant_on_the_stack() {
    use crate::support::{
        inspect::now,
        play::{cast, choose, target_opponent},
    };
    use managym::{agent::action::Action, state::game_object::Target};
    for (shield, spell) in [
        (false, "Lightning Bolt"),
        (true, "Lightning Bolt"),
        (false, "Sphinx of the Final Word"),
    ] {
        let mut s =
            five_color_setup(&["Sphinx of the Final Word", "Lightning Bolt", "Counterspell"]);
        s.game_mut().mana_payment = [managym::state::player::ManaPayment::Auto; 2];
        if shield {
            s.force_permanent_on_battlefield(0, "Sphinx of the Final Word");
        }
        s.force_card_in_hand(1, "Counterspell");
        let object = cast(&mut s, spell);
        if spell == "Lightning Bolt" {
            target_opponent(&mut s);
        }
        s.pass_priority();
        choose(&mut s, |a| {
            matches!(
                a,
                Action::CastSpell {
                    player: PlayerId(1),
                    ..
                }
            )
        });
        assert!(s.choose_target(Target::StackSpell(now(&s, object))));
        settle(&mut s);
        if spell == "Lightning Bolt" {
            s.assert_life(1, if shield { 17 } else { 20 });
        } else {
            assert_eq!(s.battlefield_permanents_named(0, spell).len(), 1);
        }
    }
}
