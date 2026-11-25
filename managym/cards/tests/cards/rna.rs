//! Tests for cards filed in `src/rna.rs`, in card order.

use crate::support::{
    game::{clear_summoning_sickness, five_color_setup},
    inspect::hand,
    play::{attack_with, enter_and_settle, pass_until},
};
use managym::{
    agent::action::{Action, ActionSpaceKind},
    flow::turn::StepKind,
};

/// Gateway Sneak's two triggers are separate: a Gate entering under its
/// controller's control makes it unblockable this turn, and combat damage to
/// a player draws a card whether or not a Gate entered. A blocked Sneak
/// deals no damage to a player and draws nothing.
#[test]
fn gateway_sneak_slips_past_blockers_after_a_gate_and_draws_on_combat_damage() {
    for (gate, block) in [(false, true), (false, false), (true, true)] {
        let mut s = five_color_setup(&["Gateway Sneak", "Azorius Guildgate", "Hill Giant"]);
        let sneak = s.force_permanent_on_battlefield(0, "Gateway Sneak");
        let giant = s.force_permanent_on_battlefield(1, "Hill Giant");
        clear_summoning_sickness(&mut s, sneak);
        if gate {
            enter_and_settle(&mut s, "Azorius Guildgate");
        }
        let cards = hand(&s, 0).len();
        pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
        attack_with(&mut s, sneak);
        let mut offered = false;
        s.advance_until(
            |s| {
                s.action_space().kind == ActionSpaceKind::DeclareBlocker
                    || s.current_step() == StepKind::PostcombatMain
            },
            "blocks or the second main phase".into(),
        );
        if s.action_space().kind == ActionSpaceKind::DeclareBlocker {
            offered = s.action_space().actions.iter().any(|action| {
                matches!(action, Action::DeclareBlocker { blocker, attacker: Some(attacker), .. }
                    if *blocker == giant && *attacker == sneak)
            });
            if offered && block {
                let index = s
                    .action_space()
                    .actions
                    .iter()
                    .position(|action| {
                        matches!(
                            action,
                            Action::DeclareBlocker {
                                attacker: Some(_),
                                ..
                            }
                        )
                    })
                    .unwrap();
                s.step_action(index);
            }
        }
        assert_eq!(offered, !gate, "gate {gate}");
        s.advance_to_active_step(0, StepKind::PostcombatMain);
        let unblocked = gate || !block;
        s.assert_life(1, if unblocked { 19 } else { 20 });
        assert_eq!(
            hand(&s, 0).len(),
            cards + usize::from(unblocked),
            "gate {gate}, block {block}"
        );
    }
}

/// Archway Angel gains two life for each Gate as one life gain (CR 119.9):
/// three Gates make one gain of six.
#[test]
fn archway_angel_gains_two_life_per_gate_in_one_event() {
    use crate::support::game::force_lands;
    let mut s = five_color_setup(&["Archway Angel", "Azorius Guildgate"]);
    force_lands(&mut s, 0, "Azorius Guildgate", 3);
    force_lands(&mut s, 1, "Azorius Guildgate", 1);
    enter_and_settle(&mut s, "Archway Angel");
    s.assert_life(0, 26);
    assert_eq!(s.game().state.turn.history[0].life_gains, 1);
}

#[test]
fn biogenic_upgrade_announces_shares_and_loses_only_an_illegal_targets_share() {
    use crate::support::{
        game::move_card,
        inspect::{card_of, counters, mode_labels, top_spell},
        play::{cast, choose_mode, settle},
    };
    use managym::{
        agent::action::ActionType,
        state::{game_object::Target, player::ManaPayment, zone::ZoneType},
    };
    for (shares, remove) in [
        (&[3][..], false),
        (&[2, 1], false),
        (&[1, 2], true),
        (&[1, 1, 1], false),
    ] {
        let mut s = five_color_setup(&["Gray Ogre", "Hill Giant", "Biogenic Upgrade"]);
        s.game_mut().mana_payment[0] = ManaPayment::Auto;
        let creatures = [
            s.force_permanent_on_battlefield(0, "Gray Ogre"),
            s.force_permanent_on_battlefield(1, "Hill Giant"),
            s.force_permanent_on_battlefield(0, "Hill Giant"),
        ];
        cast(&mut s, "Biogenic Upgrade");
        for id in &creatures[..shares.len()] {
            assert!(s.choose_target(Target::Permanent(*id)));
        }
        if shares.len() < 3 {
            assert!(s.take_action_by_type(ActionType::DeclineChoice));
        }
        for (i, share) in shares.iter().enumerate() {
            let labels = mode_labels(&s);
            assert!(labels
                .iter()
                .any(|label| label.contains(&format!("Assign {share} to "))
                    && label.contains(&format!("target {}", i + 1))));
            assert!(labels.len() <= 2);
            choose_mode(&mut s, usize::from(*share));
        }
        for viewer in [0, 1] {
            let observation = managym::agent::observation::Observation::for_player(
                s.game(),
                managym::state::game_object::PlayerId(viewer),
            );
            let label = observation.stack_objects[0].description.as_deref().unwrap();
            for (i, share) in shares.iter().enumerate() {
                let name = if i == 0 { "Gray Ogre" } else { "Hill Giant" };
                assert!(label.contains(&format!("{share} assigned to {name}")));
            }
        }
        assert_eq!(top_spell(&s).target_shares, shares);
        if remove {
            let gone = card_of(&s, creatures[0]);
            move_card(&mut s, gone, ZoneType::Hand);
        }
        settle(&mut s);
        for (i, share) in shares.iter().enumerate().skip(usize::from(remove)) {
            assert_eq!(counters(&s, creatures[i]), 2 * i32::from(*share));
        }
    }
}

#[test]
fn biogenic_upgrade_copy_keeps_the_division_when_retargeted() {
    use crate::support::{
        inspect::{counters, top_spell},
        play::{cast, choose_mode, settle},
    };
    use managym::{
        agent::action::ActionType,
        state::{game_object::Target, player::ManaPayment},
    };
    let mut s = five_color_setup(&[
        "Gray Ogre",
        "Hill Giant",
        "Biogenic Upgrade",
        "Teach by Example",
    ]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let a = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let b = s.force_permanent_on_battlefield(0, "Hill Giant");
    let c = s.force_permanent_on_battlefield(1, "Gray Ogre");
    cast(&mut s, "Teach by Example");
    settle(&mut s);
    cast(&mut s, "Biogenic Upgrade");
    for id in [a, b] {
        assert!(s.choose_target(Target::Permanent(id)));
    }
    assert!(s.take_action_by_type(ActionType::DeclineChoice));
    choose_mode(&mut s, 2);
    choose_mode(&mut s, 1);
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(c)));
    assert!(s.take_action_by_type(ActionType::DeclineChoice));
    assert_eq!(top_spell(&s).target_shares, [2, 1]);
    settle(&mut s);
    assert_eq!([a, b, c].map(|id| counters(&s, id)), [4, 6, 4]);
}

#[test]
fn retarget_projects_earlier_chosen_targets_before_final_commit() {
    use crate::support::{
        inspect::top_spell,
        play::{cast, choose_mode, settle},
    };
    use managym::{
        agent::{
            action::ActionType,
            observation::Observation,
            observation_encoder::{encode, ObservationEncoderConfig},
        },
        state::{
            game_object::{PlayerId, Target},
            player::ManaPayment,
        },
    };
    let mut s = five_color_setup(&[
        "Gray Ogre",
        "Hill Giant",
        "Wind Drake",
        "Biogenic Upgrade",
        "Teach by Example",
    ]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let a = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let b = s.force_permanent_on_battlefield(0, "Hill Giant");
    let c = s.force_permanent_on_battlefield(1, "Wind Drake");
    let d = s.force_permanent_on_battlefield(1, "Gray Ogre");
    cast(&mut s, "Teach by Example");
    settle(&mut s);
    cast(&mut s, "Biogenic Upgrade");
    for id in [a, b] {
        assert!(s.choose_target(Target::Permanent(id)));
    }
    assert!(s.take_action_by_type(ActionType::DeclineChoice));
    choose_mode(&mut s, 2);
    choose_mode(&mut s, 1);
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);

    let mut fork_c = s.clone();
    let mut fork_d = s.clone();
    assert!(fork_c.choose_target(Target::Permanent(c)));
    assert!(fork_d.choose_target(Target::Permanent(d)));

    // Whole-assignment retargeting keeps the stack object's original targets until slot 1 commits.
    let expected_original = vec![Target::Permanent(a), Target::Permanent(b)];
    assert_eq!(top_spell(&fork_c).targets, expected_original);
    assert_eq!(top_spell(&fork_d).targets, expected_original);

    let obs_c = Observation::for_player(fork_c.game(), PlayerId(0));
    let obs_d = Observation::for_player(fork_d.game(), PlayerId(0));
    let enc_c = encode(&obs_c, &ObservationEncoderConfig::default()).unwrap();
    let enc_d = encode(&obs_d, &ObservationEncoderConfig::default()).unwrap();
    assert_ne!(
        (enc_c.facts, enc_c.relations),
        (enc_d.facts, enc_d.relations),
        "slot 1 encoded observations must distinguish earlier retarget choices"
    );

    let ctx_c = fork_c.game().decision_context(PlayerId(0)).unwrap();
    let ctx_d = fork_d.game().decision_context(PlayerId(0)).unwrap();
    assert!(
        ctx_c.pending.iter().any(|line| line.contains("Wind Drake")),
        "expected slot-0 new target in pending summary: {:?}",
        ctx_c.pending
    );
    assert!(
        ctx_d.pending.iter().any(|line| line.contains("Gray Ogre")),
        "expected slot-0 new target in pending summary: {:?}",
        ctx_d.pending
    );
    assert_ne!(ctx_c.pending, ctx_d.pending);

    let capture_c = fork_c.game().capture_decision(PlayerId(0), 0).unwrap();
    assert!(
        capture_c.frames.iter().any(|frame| frame
            .effect_execution_description
            .as_deref()
            .is_some_and(|desc| desc.contains("new target:"))),
        "expected 'new target:' in effect_execution_description: {:?}",
        capture_c
            .frames
            .iter()
            .filter_map(|frame| frame.effect_execution_description.as_deref())
            .collect::<Vec<_>>()
    );
}

#[test]
fn gate_colossus_affinity_evasion_and_graveyard_gate_trigger() {
    use crate::support::{
        game::{force_lands, plant_in_graveyard},
        inspect::{library_top, now, top_spell, zone_of},
        play::{cast, choose, settle},
    };
    use managym::state::{player::ManaPayment, zone::ZoneType};

    // Affinity for Gates + evasion against power <= 2.
    let mut s = five_color_setup(&[
        "Gate Colossus",
        "Azorius Guildgate",
        "Gray Ogre",
        "Hill Giant",
    ]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    force_lands(&mut s, 0, "Azorius Guildgate", 2);
    cast(&mut s, "Gate Colossus");
    assert_eq!(top_spell(&s).spent.total(), 6);
    settle(&mut s);

    let colossus = s.battlefield_permanents_named(0, "Gate Colossus")[0];
    let small = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let big = s.force_permanent_on_battlefield(1, "Hill Giant");
    clear_summoning_sickness(&mut s, colossus);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, colossus);
    pass_until(&mut s, ActionSpaceKind::DeclareBlocker);
    let can_block = |blocker| {
        s.action_space().actions.iter().any(|a| {
            matches!(a, Action::DeclareBlocker { blocker: b, attacker: Some(atk), .. } if *b == blocker && *atk == colossus)
        })
    };
    assert!(!can_block(small));
    assert!(can_block(big));

    // Graveyard trigger on Gate entry puts Gate Colossus on top of library.
    let mut s = five_color_setup(&["Gate Colossus", "Azorius Guildgate"]);
    let colossus = plant_in_graveyard(&mut s, 0, "Gate Colossus");
    crate::support::game::enters_battlefield_with_triggers(&mut s, 0, "Azorius Guildgate");
    s.game_mut().scenario_refresh_priority().unwrap();
    pass_until(&mut s, ActionSpaceKind::PayOrNot);
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    settle(&mut s);
    assert_eq!(zone_of(&s, colossus), Some(ZoneType::Library));
    assert_eq!(library_top(&s, 0), now(&s, colossus));
}
