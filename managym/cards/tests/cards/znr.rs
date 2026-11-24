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
        for seat in 0..2 {
            let obs = Observation::for_player(s.game(), PlayerId(seat));
            assert_eq!(obs.turn.another_damage_step, hit == 1);
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
