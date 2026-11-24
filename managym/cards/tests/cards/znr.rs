//! Zendikar Rising card behavior.
use crate::support::{
    game::{clear_summoning_sickness, five_color_setup},
    inspect::hand,
    play::{attack_with, pass_until, resolve_top},
};
use managym::{
    agent::{action::ActionSpaceKind, observation::Observation},
    cards::{
        content_pack::default_content_pack,
        tokens::{definition_tokens, name_value, TokenKind},
    },
    flow::turn::StepKind,
    state::game_object::PlayerId,
};

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
