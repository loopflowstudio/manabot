//! Tests for cards filed in `src/znr.rs`, in card order.
use crate::support::game::{clear_summoning_sickness, five_color_setup, until_end_of_turn};
use crate::support::inspect::{hand, offered_targets};
use crate::support::play::{attack_with, cast, pass_until, resolve_top, settle};
use managym::agent::action::ActionSpaceKind;
use managym::agent::observation::Observation;
use managym::cards::content_pack::default_content_pack;
use managym::cards::tokens::{definition_tokens, name_value, TokenKind};
use managym::flow::turn::StepKind;
use managym::state::card::{Keyword, Keywords};
use managym::state::effect::Modification;
use managym::state::game_object::{PlayerId, Target};

/// Feed the Swarm targets only an opponent's creature or enchantment, and
/// its caster loses life equal to that permanent's mana value as it last
/// existed, or as it still is when it survives.
#[test]
fn feed_the_swarm_loses_life_equal_to_the_targets_mana_value() {
    for indestructible in [false, true] {
        let mut s = five_color_setup(&["Feed the Swarm", "Gray Ogre"]);
        let own = s.force_permanent_on_battlefield(0, "Gray Ogre");
        let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
        let land = s.force_permanent_on_battlefield(1, "Plains");
        if indestructible {
            until_end_of_turn(
                &mut s,
                ogre,
                Modification::AddKeywords(Keywords::of(&[Keyword::Indestructible])),
            );
        }
        cast(&mut s, "Feed the Swarm");
        let offered = offered_targets(&s);
        assert!(offered.contains(&Target::Permanent(ogre)));
        assert!(!offered.contains(&Target::Permanent(own)));
        assert!(!offered.contains(&Target::Permanent(land)));
        assert!(s.choose_target(Target::Permanent(ogre)));
        settle(&mut s);
        assert_eq!(s.game().state.permanents[ogre].is_some(), indestructible);
        s.assert_life(0, 17);
    }
}

#[test]
fn blight_priest_drains_once_per_reaver_hit() {
    let priest = "Marauding Blight-Priest";
    let pack = default_content_pack();
    let definition = pack
        .definition(pack.definition_id(priest).unwrap())
        .unwrap();
    let tokens = definition_tokens(definition, pack).unwrap();
    assert!(tokens.contains(&(
        TokenKind::Variant,
        name_value("TriggerCondition::GainsLife")
    )));

    let mut s = five_color_setup(&[priest, "Drogskol Reaver"]);
    s.force_permanent_on_battlefield(0, priest);
    let reaver = s.force_permanent_on_battlefield(0, "Drogskol Reaver");
    clear_summoning_sickness(&mut s, reaver);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    let opening = hand(&s, 0).len();
    attack_with(&mut s, reaver);
    for (hit, attacker, defender) in [(1, 23, 16), (2, 26, 12)] {
        pass_until(&mut s, ActionSpaceKind::OrderTriggers);
        assert_eq!(s.current_step(), StepKind::CombatDamage);
        pass_until(&mut s, ActionSpaceKind::Priority);
        let expected_step = if hit == 1 {
            StepKind::FirstStrikeCombatDamage
        } else {
            StepKind::CombatDamage
        };
        for seat in 0..2 {
            let obs = Observation::for_player(s.game(), PlayerId(seat));
            assert_eq!(obs.turn.step, expected_step);
            assert_eq!(obs.stack_objects.len(), 2);
            assert!(obs.stack_objects.iter().any(|object| object.description.as_deref() == Some("Marauding Blight-Priest — Whenever you gain life: each opponent loses 1 life")));
        }
        assert_eq!((s.life(0), s.life(1)), (attacker, defender + 1));
        resolve_top(&mut s);
        resolve_top(&mut s);
        assert!(s.game().state.stack_objects.is_empty());
        assert_eq!((s.life(0), s.life(1)), (attacker, defender));
        assert_eq!(hand(&s, 0).len(), opening + hit);
    }
}

#[test]
fn printed_creature_highborn_vampire() {
    crate::printed_creature_combat("Highborn Vampire", 4, 3, &[]);
}

#[test]
fn foundations_spell_broken_wings() {
    use managym::state::zone::ZoneType;
    crate::check_targeted_removal(
        "Broken Wings",
        "Serra Angel",
        "Gray Ogre",
        ZoneType::Graveyard,
    );
}

#[test]
fn foundations_5d_crawling_barrens() {
    use crate::support::game::five_color_setup;
    use crate::support::inspect::counters;
    use crate::support::play::{activate, choose_mode, pass_until, settle};
    use managym::agent::action::ActionSpaceKind;
    use managym::flow::turn::StepKind;

    let mut s = five_color_setup(&["Crawling Barrens"]);
    let id = s.force_permanent_on_battlefield(0, "Crawling Barrens");
    s.set_player_mana_pool(0, "4");
    activate(&mut s, id, 0);
    pass_until(&mut s, ActionSpaceKind::Modal);
    assert_eq!(counters(&s, id), 2);
    assert!(crate::support::inspect::mode_labels(&s)[0].contains("becomes a creature"));
    let mut decline = s.clone();
    choose_mode(&mut decline, 1);
    settle(&mut decline);
    assert!(!decline.game().characteristics(id).is_creature());
    choose_mode(&mut s, 0);
    settle(&mut s);
    let face = s.game().characteristics(id);
    assert!(face.is_creature() && face.has_subtype("Elemental"));
    assert_eq!(face.pt(), (2, 2));
    s.advance_to_active_step(1, StepKind::Main);
    assert!(!s.game().characteristics(id).is_creature());
    assert_eq!(counters(&s, id), 2);
}

#[test]
fn foundations_5f_spitfire_lagac() {
    use crate::support::inspect::library_card;
    use crate::support::play::{enter_and_settle, move_and_settle};
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Spitfire Lagac"]);
    s.force_permanent_on_battlefield(0, "Spitfire Lagac");
    enter_and_settle(&mut s, "Forest");
    assert_eq!((s.life(0), s.life(1)), (20, 19));
    let theirs = library_card(&s, 1, "Forest");
    move_and_settle(&mut s, theirs, ZoneType::Battlefield);
    assert_eq!((s.life(0), s.life(1)), (20, 19));
}
