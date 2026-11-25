//! Conflux.
use crate::support::{
    game::{clear_summoning_sickness, five_color_setup},
    inspect::{card_of, offered_targets, permanent, zone_of},
    play::{attack_with, cast, settle},
};
use managym::{
    agent::action::Action,
    flow::turn::StepKind,
    state::{game_object::Target, zone::ZoneType},
};

#[test]
fn progenitus_protection_covers_targets_blocks_damage_and_attachments() {
    let mut s = five_color_setup(&[
        "Progenitus",
        "Hill Giant",
        "Pyroclasm",
        "Fumigate",
        "Confiscate",
        "Kyoshi Battle Fan",
    ]);
    let hydra = s.force_permanent_on_battlefield(0, "Progenitus");
    let giant = s.force_permanent_on_battlefield(1, "Hill Giant");
    let mut target = s.clone();
    cast(&mut target, "Confiscate");
    assert!(!offered_targets(&target).contains(&Target::Permanent(hydra)));
    assert!(offered_targets(&target).contains(&Target::Permanent(giant)));

    let mut attached = s.clone();
    let aura = attached.force_permanent_on_battlefield(1, "Confiscate");
    let equipment = attached.force_permanent_on_battlefield(1, "Kyoshi Battle Fan");
    let object = card_of(&attached, hydra);
    let aura_card = card_of(&attached, aura);
    for id in [aura, equipment] {
        attached.game_mut().state.permanents[id]
            .as_mut()
            .unwrap()
            .attached_to = Some(object);
    }
    attached.game_mut().scenario_refresh_priority().unwrap();
    attached.pass_priority();
    assert_eq!(zone_of(&attached, aura_card), Some(ZoneType::Graveyard));
    assert!(permanent(&attached, equipment).attached_to.is_none());

    cast(&mut s, "Pyroclasm");
    settle(&mut s);
    assert_eq!(permanent(&s, hydra).damage, 0);
    assert_eq!(permanent(&s, giant).damage, 2);
    let mut wrath = s.clone();
    cast(&mut wrath, "Fumigate");
    settle(&mut wrath);
    assert!(wrath.game().state.permanents[hydra].is_none());
    clear_summoning_sickness(&mut s, hydra);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, hydra);
    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    assert!(s.action_space().actions.iter().all(|a| !matches!(
        a,
        Action::DeclareBlocker {
            attacker: Some(_),
            ..
        }
    )));
}
