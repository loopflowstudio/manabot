//! Native grouped-input tests covering combat declarations and library
//! arrangements (Scry, Surveil, OrderLibrary Top/Bottom), purity boundary
//! behavior, mid-group raw steps, and digest parity between `apply_group`
//! and step-by-step execution.
use managym::{
    agent::action::{Action, ActionSpaceKind},
    experience::ObjectRenderId,
    flow::{
        decision::Decision,
        group::{
            Group, GroupOutcome, GroupPlan, GroupPreview, GroupSpec, LibraryEnd, PlannedBlock,
        },
        turn::StepKind,
    },
    search_state::witness,
    state::{
        ability::{Effect, EffectValue},
        card::ActivatedAbilityDefinition,
        game_object::{ObjectId, PermanentId, PlayerId},
        predicate::CardPredicate,
        target::Target,
        zone::ZoneType,
    },
    Game,
};

use crate::support::{
    game::*,
    play::{activate, cast, settle},
};

fn perm_render(game: &Game, permanent: PermanentId) -> ObjectRenderId {
    let permanent = game.state.permanents[permanent]
        .as_ref()
        .expect("on the battlefield");
    ObjectRenderId {
        entity: permanent.id.0,
        incarnation: permanent.object.generation(),
    }
}

fn card_render(game: &Game, object: ObjectId) -> ObjectRenderId {
    let public_id = game.state.objects.provenance(object).expect("provenance").0;
    ObjectRenderId {
        entity: public_id.0,
        incarnation: object.generation(),
    }
}

fn follow_steps(game: &mut Game, preview: &GroupPreview) {
    for step in &preview.steps {
        let frame = game.semantic_decision_frame().unwrap();
        assert_eq!(
            (frame.revision, &frame.fingerprint),
            (step.revision, &step.fingerprint)
        );
        game.step(step.action_index).unwrap();
    }
}

/// Assert that `preview_group` is read-only and that `apply_group` reaches the
/// exact same canonical state digest as stepping `preview.steps` manually.
fn assert_apply_matches_manual(game: &mut Game, viewer: usize, plan: &GroupPlan) -> GroupPreview {
    let before_witness = witness(game);
    let before_digest = game.state.deterministic_hash();
    let frame = game.semantic_decision_frame().unwrap();

    let preview = game
        .preview_group(PlayerId(viewer), frame.revision, &frame.fingerprint, plan)
        .unwrap();
    assert_eq!(witness(game), before_witness, "preview must be read-only");
    assert_eq!(game.state.deterministic_hash(), before_digest);
    assert!(preview.complete, "expected complete preview: {preview:?}");
    assert!(preview.refused.is_none(), "{:?}", preview.refused);

    let mut manual = game.clone();
    follow_steps(&mut manual, &preview);

    let applied = game
        .apply_group(PlayerId(viewer), frame.revision, &frame.fingerprint, plan)
        .unwrap();
    assert_eq!(applied, preview);
    assert_eq!(
        game.state.deterministic_hash(),
        manual.state.deterministic_hash(),
        "apply_group and manual step execution must produce identical digests"
    );
    applied
}

fn combat_three_vs_three() -> (Scenario, Vec<PermanentId>, Vec<PermanentId>) {
    let mut s = Scenario::new(
        deck(&[("Mountain", 30), ("Boggart Brute", 4), ("Gray Ogre", 6)]),
        deck(&[("Forest", 30), ("Llanowar Elves", 10)]),
        7,
    );
    s.advance_to_active_step(0, StepKind::Main);
    let attackers: Vec<_> = ["Boggart Brute", "Gray Ogre", "Gray Ogre"]
        .iter()
        .map(|name| {
            let id = s.force_permanent_on_battlefield(0, name);
            clear_summoning_sickness(&mut s, id);
            id
        })
        .collect();
    let blockers: Vec<_> = (0..3)
        .map(|_| s.force_permanent_on_battlefield(1, "Llanowar Elves"))
        .collect();
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    assert_eq!(s.action_space().kind, ActionSpaceKind::DeclareAttacker);
    (s, attackers, blockers)
}

#[test]
fn combat_three_attackers_one_attacker_multi_block_and_refusals() {
    // 1. Three-creature attack + multi-block + menace refusal.
    let (mut s, attackers, blockers) = combat_three_vs_three();
    let active = s.game().group(PlayerId(0)).expect("attack group active");
    assert!(s.game().group(PlayerId(1)).is_none(), "non-actor gets None");
    match &active.spec {
        GroupSpec::DeclareAttackers {
            declarable,
            committed,
        } => {
            assert_eq!(declarable.len(), 3);
            assert!(committed.is_empty());
        }
        other => panic!("expected DeclareAttackers, got {other:?}"),
    }

    // Wrong plan kind errors and leaves state untouched.
    let before = witness(s.game());
    assert!(s
        .game()
        .preview_group(
            PlayerId(0),
            active.revision,
            &active.fingerprint,
            &GroupPlan::Scry {
                top: vec![],
                bottom: vec![],
            },
        )
        .is_err());
    assert!(s
        .game_mut()
        .apply_group(
            PlayerId(0),
            active.revision + 1,
            &active.fingerprint,
            &GroupPlan::DeclareAttackers { attackers: vec![] },
        )
        .is_err());
    assert_eq!(witness(s.game()), before);

    // Attack with all three so we can test blockers next.
    let all_attackers: Vec<_> = attackers
        .iter()
        .map(|&id| perm_render(s.game(), id))
        .collect();
    let attack_plan = GroupPlan::DeclareAttackers {
        attackers: all_attackers.clone(),
    };
    let attack_result = assert_apply_matches_manual(s.game_mut(), 0, &attack_plan);
    assert_eq!(attack_result.steps.len(), 3);
    assert!(matches!(
        attack_result.outcome,
        Some(GroupOutcome::DeclareAttackers { ref attackers }) if attackers.len() == 3
    ));

    // Advance to blockers.
    s.advance_until(
        |s| s.action_space().kind == ActionSpaceKind::DeclareBlocker,
        "failed to reach blockers".into(),
    );
    let blocker_group = s.game().group(PlayerId(1)).expect("blocker group active");
    assert!(matches!(
        blocker_group.spec,
        GroupSpec::DeclareBlockers { ref declarable, .. } if declarable.len() == 3
    ));

    // Menace refusal: lone Elf on Boggart Brute refuses without mutating.
    let lone_menace = GroupPlan::DeclareBlockers {
        blocks: vec![PlannedBlock {
            blocker: perm_render(s.game(), blockers[0]),
            attacker: perm_render(s.game(), attackers[0]),
        }],
    };
    let before_menace = witness(s.game());
    let menace_preview = s
        .game()
        .preview_group(
            PlayerId(1),
            blocker_group.revision,
            &blocker_group.fingerprint,
            &lone_menace,
        )
        .unwrap();
    assert!(!menace_preview.complete);
    let refusal = menace_preview.refused.as_ref().expect("menace refusal");
    assert!(
        refusal.reason.contains("block") && refusal.reason.contains("Boggart Brute"),
        "{}",
        refusal.reason
    );
    assert!(menace_preview.outcome.is_none());
    assert!(s
        .game_mut()
        .apply_group(
            PlayerId(1),
            blocker_group.revision,
            &blocker_group.fingerprint,
            &lone_menace,
        )
        .is_err());
    assert_eq!(witness(s.game()), before_menace);

    // Legal multi-block: two Elves on Brute, third on Ogre.
    let multi_block = GroupPlan::DeclareBlockers {
        blocks: vec![
            PlannedBlock {
                blocker: perm_render(s.game(), blockers[0]),
                attacker: perm_render(s.game(), attackers[0]),
            },
            PlannedBlock {
                blocker: perm_render(s.game(), blockers[1]),
                attacker: perm_render(s.game(), attackers[0]),
            },
            PlannedBlock {
                blocker: perm_render(s.game(), blockers[2]),
                attacker: perm_render(s.game(), attackers[1]),
            },
        ],
    };
    assert_apply_matches_manual(s.game_mut(), 1, &multi_block);

    // 2. Single-creature attack still activates the combat group (>= 1 threshold).
    let mut single = Scenario::new(
        deck(&[("Mountain", 30), ("Gray Ogre", 10)]),
        deck(&[("Forest", 40)]),
        11,
    );
    single.advance_to_active_step(0, StepKind::Main);
    let ogre = single.force_permanent_on_battlefield(0, "Gray Ogre");
    clear_summoning_sickness(&mut single, ogre);
    single.advance_to_active_step(0, StepKind::DeclareAttackers);
    let single_group = single.game().group(PlayerId(0)).expect("1-creature group");
    assert!(matches!(
        single_group.spec,
        GroupSpec::DeclareAttackers { ref declarable, .. } if declarable.len() == 1
    ));
    let single_plan = GroupPlan::DeclareAttackers {
        attackers: vec![perm_render(single.game(), ogre)],
    };
    let applied = assert_apply_matches_manual(single.game_mut(), 0, &single_plan);
    assert_eq!(applied.steps.len(), 1);
}

#[test]
fn combat_must_attack_and_must_block_refusals_preserve_digest() {
    // Must-attack refusal (Juggernaut).
    let mut s = Scenario::new(
        deck(&[("Mountain", 30), ("Juggernaut", 4), ("Gray Ogre", 6)]),
        deck(&[("Forest", 40)]),
        7,
    );
    s.advance_to_active_step(0, StepKind::Main);
    let jugg = s.force_permanent_on_battlefield(0, "Juggernaut");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    clear_summoning_sickness(&mut s, jugg);
    clear_summoning_sickness(&mut s, ogre);
    s.game_mut().skip_trivial = false;
    s.advance_to_active_step(0, StepKind::DeclareAttackers);

    let g = s.game().group(PlayerId(0)).expect("attackers group");
    let before = witness(s.game());
    let omit_jugg = GroupPlan::DeclareAttackers {
        attackers: vec![perm_render(s.game(), ogre)],
    };
    let preview = s
        .game()
        .preview_group(PlayerId(0), g.revision, &g.fingerprint, &omit_jugg)
        .unwrap();
    assert!(!preview.complete);
    assert_eq!(
        preview.refused.as_ref().unwrap().reason,
        "Juggernaut must attack."
    );
    assert!(s
        .game_mut()
        .apply_group(PlayerId(0), g.revision, &g.fingerprint, &omit_jugg)
        .is_err());
    assert_eq!(witness(s.game()), before);

    let both = GroupPlan::DeclareAttackers {
        attackers: vec![perm_render(s.game(), jugg), perm_render(s.game(), ogre)],
    };
    assert_apply_matches_manual(s.game_mut(), 0, &both);

    // Must-block refusal (Taunting Challenge).
    let mut b = five_color_setup(&[
        "Taunting Challenge",
        "Wind Drake",
        "Gray Ogre",
        "Giant Spider",
        "Llanowar Elves",
    ]);
    let drake = b.force_permanent_on_battlefield(0, "Wind Drake");
    let ogre2 = b.force_permanent_on_battlefield(0, "Gray Ogre");
    clear_summoning_sickness(&mut b, drake);
    clear_summoning_sickness(&mut b, ogre2);
    let spider = b.force_permanent_on_battlefield(1, "Giant Spider");
    let elves = b.force_permanent_on_battlefield(1, "Llanowar Elves");
    cast(&mut b, "Taunting Challenge");
    assert!(b.choose_target(Target::Permanent(drake)));
    settle(&mut b);
    b.game_mut().skip_trivial = false;
    b.advance_to_active_step(0, StepKind::DeclareAttackers);
    let atk_plan = GroupPlan::DeclareAttackers {
        attackers: vec![perm_render(b.game(), drake), perm_render(b.game(), ogre2)],
    };
    assert_apply_matches_manual(b.game_mut(), 0, &atk_plan);
    b.advance_until(
        |s| s.action_space().kind == ActionSpaceKind::DeclareBlocker,
        "reach blockers".into(),
    );
    let bg = b.game().group(PlayerId(1)).expect("blockers group");
    let bad_blocks = GroupPlan::DeclareBlockers {
        blocks: vec![PlannedBlock {
            blocker: perm_render(b.game(), elves),
            attacker: perm_render(b.game(), ogre2),
        }],
    };
    let before_block = witness(b.game());
    let bad_preview = b
        .game()
        .preview_group(PlayerId(1), bg.revision, &bg.fingerprint, &bad_blocks)
        .unwrap();
    assert!(!bad_preview.complete);
    assert_eq!(
        bad_preview.refused.as_ref().unwrap().reason,
        "Giant Spider must block Wind Drake."
    );
    assert!(b
        .game_mut()
        .apply_group(PlayerId(1), bg.revision, &bg.fingerprint, &bad_blocks)
        .is_err());
    assert_eq!(witness(b.game()), before_block);

    let good_blocks = GroupPlan::DeclareBlockers {
        blocks: vec![
            PlannedBlock {
                blocker: perm_render(b.game(), spider),
                attacker: perm_render(b.game(), drake),
            },
            PlannedBlock {
                blocker: perm_render(b.game(), elves),
                attacker: perm_render(b.game(), ogre2),
            },
        ],
    };
    assert_apply_matches_manual(b.game_mut(), 1, &good_blocks);
}

/// Set up a creature with a `{0}: <effects>` activated ability, pin the top of
/// player 0's library, and activate the ability so resolution pauses at its
/// first decision.
fn setup_library_effect(
    top_cards: &[&str],
    effects: Vec<Effect>,
    skip_trivial: bool,
) -> (Scenario, Vec<ObjectId>) {
    let mut s = five_color_setup(&["Gray Ogre", "Forest", "Mountain", "Island", "Swamp"]);
    s.game_mut().skip_trivial = skip_trivial;
    let host = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let host_obj = s.game().permanent_object(host).unwrap();
    s.game_mut().state[host_obj]
        .activated_abilities
        .push(ActivatedAbilityDefinition {
            effects,
            ..Default::default()
        });
    let pinned = pin_top(&mut s, top_cards);
    activate(&mut s, host, 0);
    s.advance_until(
        |s| s.game().state.suspended_decision.is_some(),
        "resolve activated library ability".into(),
    );
    (s, pinned)
}

#[test]
fn scry_three_bottom_order_incomplete_refusal_and_mid_group_raw_step() {
    // Scry 3: pin top 3 as [Forest (top), Mountain (2nd), Island (3rd)].
    let (mut s, pinned) = setup_library_effect(
        &["Forest", "Mountain", "Island"],
        vec![Effect::Scry { count: 3 }],
        true,
    );
    let g = s.game().group(PlayerId(0)).expect("Scry 3 group");
    let (c0, c1, c2) = (
        card_render(s.game(), pinned[0]),
        card_render(s.game(), pinned[1]),
        card_render(s.game(), pinned[2]),
    );
    match &g.spec {
        GroupSpec::Scry {
            items,
            committed_top,
            committed_bottom,
        } => {
            assert_eq!(items.len(), 3);
            assert!(committed_top.is_empty());
            assert!(committed_bottom.is_empty());
        }
        other => panic!("expected Scry, got {other:?}"),
    }

    // Incomplete plan returns complete=false, refused=None, steps=[], outcome=None without mutating.
    let before = witness(s.game());
    let incomplete = GroupPlan::Scry {
        top: vec![c2],
        bottom: vec![c0],
    };
    let inc_preview = s
        .game()
        .preview_group(PlayerId(0), g.revision, &g.fingerprint, &incomplete)
        .unwrap();
    assert!(!inc_preview.complete);
    assert!(inc_preview.refused.is_none());
    assert!(inc_preview.steps.is_empty());
    assert!(inc_preview.outcome.is_none());
    assert!(s
        .game_mut()
        .apply_group(PlayerId(0), g.revision, &g.fingerprint, &incomplete)
        .is_err());
    assert_eq!(witness(s.game()), before);

    // Duplicate card is refused.
    let dup = GroupPlan::Scry {
        top: vec![c0, c0],
        bottom: vec![c1],
    };
    let dup_preview = s
        .game()
        .preview_group(PlayerId(0), g.revision, &g.fingerprint, &dup)
        .unwrap();
    assert!(!dup_preview.complete);
    assert!(dup_preview
        .refused
        .as_ref()
        .unwrap()
        .reason
        .contains("only be placed once"));

    // Plan: top=[c2 (Island)], bottom=[c0 (Forest), c1 (Mountain)].
    // Reading order for bottom: c0 sits ABOVE c1, and c1 sits on the VERY bottom (index 0).
    let mut mid_raw = s.game().clone();
    let plan = GroupPlan::Scry {
        top: vec![c2],
        bottom: vec![c0, c1],
    };
    let applied = assert_apply_matches_manual(s.game_mut(), 0, &plan);
    assert_eq!(applied.steps.len(), 3);
    assert_eq!(
        applied.outcome,
        Some(GroupOutcome::Scry {
            top: vec![c2],
            bottom: vec![c0, c1],
        })
    );
    let lib = s
        .game()
        .state
        .zones
        .zone_objects(ZoneType::Library, PlayerId(0));
    assert_eq!(
        *lib.last().unwrap(),
        pinned[2],
        "c2 (Island) must be on top"
    );
    assert_eq!(
        lib[0], pinned[1],
        "c1 (Mountain) must be on the very bottom"
    );
    assert_eq!(
        lib[1], pinned[0],
        "c0 (Forest) must sit directly above c1 on the bottom"
    );

    // Mid-group raw step: put c1 on bottom via raw step, verify committed_bottom, then finish with apply_group.
    let raw_idx = mid_raw
        .action_space()
        .unwrap()
        .actions
        .iter()
        .position(|a| {
            matches!(
                a,
                Action::ScryCard {
                    object,
                    to_bottom: true,
                    ..
                } if *object == pinned[1]
            )
        })
        .unwrap();
    mid_raw.step(raw_idx).unwrap();
    let after_one = mid_raw.group(PlayerId(0)).expect("2 cards still remain");
    assert_eq!(
        after_one.spec,
        GroupSpec::Scry {
            items: vec![
                managym::flow::group::GroupCard {
                    item: c0,
                    label: "Forest".into()
                },
                managym::flow::group::GroupCard {
                    item: c2,
                    label: "Island".into()
                },
            ],
            committed_top: vec![],
            committed_bottom: vec![c1],
        }
    );
    // Finish the remaining two cards via apply_group: c2 on top, c0 on bottom (above c1!).
    let rest_plan = GroupPlan::Scry {
        top: vec![c2],
        bottom: vec![c0],
    };
    let rest_applied = assert_apply_matches_manual(&mut mid_raw, 0, &rest_plan);
    assert_eq!(
        rest_applied.outcome,
        Some(GroupOutcome::Scry {
            top: vec![c2],
            bottom: vec![c0, c1],
        })
    );
    let lib2 = mid_raw
        .state
        .zones
        .zone_objects(ZoneType::Library, PlayerId(0));
    assert_eq!(*lib2.last().unwrap(), pinned[2]);
    assert_eq!(lib2[0], pinned[1]);
    assert_eq!(lib2[1], pinned[0]);
    assert_eq!(
        mid_raw.state.deterministic_hash(),
        s.game().state.deterministic_hash(),
        "mid-group raw step + apply_group must match full apply_group"
    );
}

#[test]
fn scry_two_then_draw_completes_cleanly_past_post_group_draw() {
    // The Legend of Kuruk Chapter I: "Scry 2, then draw a card." (Preordain pattern).
    let mut s = five_color_setup(&["The Legend of Kuruk", "Forest", "Mountain", "Island"]);
    let pinned = pin_top(&mut s, &["Forest", "Mountain"]);
    cast(&mut s, "The Legend of Kuruk");
    // Resolve the Saga entering and its Chapter I trigger onto the stack, then to Scry 2.
    s.advance_until(
        |s| s.action_space().kind == ActionSpaceKind::Scry,
        "reach Legend of Kuruk scry 2".into(),
    );
    let g = s.game().group(PlayerId(0)).expect("Scry 2 group");
    let (c0, c1) = (
        card_render(s.game(), pinned[0]),
        card_render(s.game(), pinned[1]),
    );
    assert!(matches!(g.spec, GroupSpec::Scry { ref items, .. } if items.len() == 2));

    // Take 1 raw step on a clone to verify domain threshold: 1 remaining card -> group() is None.
    let mut one_left = s.game().clone();
    one_left.step(0).unwrap();
    assert!(
        one_left.group(PlayerId(0)).is_none(),
        "1 remaining Scry card is below the >= 2 threshold"
    );

    // Put Mountain (c1) on top to draw it, Forest (c0) on bottom.
    // The post-group draw fires after Scry finishes and must NOT trip the purity boundary.
    let plan = GroupPlan::Scry {
        top: vec![c1],
        bottom: vec![c0],
    };
    let applied = assert_apply_matches_manual(s.game_mut(), 0, &plan);
    assert_eq!(applied.steps.len(), 2);
    assert!(
        s.game()
            .state
            .zone_cards(ZoneType::Hand, PlayerId(0))
            .any(|card| s.game().state.object_of(card) == pinned[1]
                || s.game().state[s.game().state.object_of(card)].name == "Mountain"),
        "top-scried Mountain must have been drawn"
    );
    let lib = s
        .game()
        .state
        .zones
        .zone_objects(ZoneType::Library, PlayerId(0));
    assert_eq!(lib[0], pinned[0], "bottom-scried Forest must sit on bottom");
}

#[test]
fn surveil_two_to_order_library_top_under_both_skip_trivial_settings() {
    for skip_trivial in [true, false] {
        let mut s = five_color_setup(&["Esper Origins", "Forest", "Mountain", "Island"]);
        s.game_mut().skip_trivial = skip_trivial;
        let pinned = pin_top(&mut s, &["Forest", "Mountain"]);
        cast(&mut s, "Esper Origins");
        s.advance_until(
            |s| {
                matches!(
                    s.game()
                        .state
                        .suspended_decision
                        .as_ref()
                        .map(|d| &d.decision),
                    Some(Decision::Surveil { .. })
                )
            },
            "reach Esper Origins surveil".into(),
        );

        let surveil_group = s.game().group(PlayerId(0)).expect("Surveil 2 group");
        let (c0, c1) = (
            card_render(s.game(), pinned[0]),
            card_render(s.game(), pinned[1]),
        );
        assert!(matches!(
            surveil_group.spec,
            GroupSpec::Surveil { ref items, ref committed_graveyard }
                if items.len() == 2 && committed_graveyard.is_empty()
        ));

        // Keep both on top (graveyard = []) -> emits 1 Decline step with item=None.
        let keep_all = GroupPlan::Surveil { graveyard: vec![] };
        let surveil_applied = assert_apply_matches_manual(s.game_mut(), 0, &keep_all);
        assert_eq!(surveil_applied.steps.len(), 1);
        assert_eq!(surveil_applied.steps[0].item, None);
        assert_eq!(
            surveil_applied.outcome,
            Some(GroupOutcome::Surveil {
                graveyard: vec![],
                kept: vec![c0, c1],
            })
        );

        // Immediately opens OrderLibrary { end: Top } with 2 cards even when skip_trivial=true.
        let order_group: Group = s
            .game()
            .group(PlayerId(0))
            .expect("OrderLibrary Top group after keeping 2 on Surveil");
        assert!(matches!(
            order_group.spec,
            GroupSpec::OrderLibrary {
                end: LibraryEnd::Top,
                ref items,
                ref committed,
            } if items.len() == 2 && committed.is_empty()
        ));

        // Incomplete OrderLibrary plan returns complete=false without cloning.
        let incomplete = GroupPlan::OrderLibrary { order: vec![c1] };
        let inc_preview = s
            .game()
            .preview_group(
                PlayerId(0),
                order_group.revision,
                &order_group.fingerprint,
                &incomplete,
            )
            .unwrap();
        assert!(!inc_preview.complete);
        assert!(inc_preview.refused.is_none());
        assert!(inc_preview.steps.is_empty());

        // Order Mountain (c1) on very top, Forest (c0) second.
        let order_plan = GroupPlan::OrderLibrary {
            order: vec![c1, c0],
        };
        let order_applied = assert_apply_matches_manual(s.game_mut(), 0, &order_plan);
        assert_eq!(order_applied.steps.len(), if skip_trivial { 1 } else { 2 });
        assert_eq!(
            order_applied.outcome,
            Some(GroupOutcome::OrderLibrary {
                end: LibraryEnd::Top,
                order: vec![c1, c0],
            })
        );
        let lib = s
            .game()
            .state
            .zones
            .zone_objects(ZoneType::Library, PlayerId(0));
        let len = lib.len();
        assert_eq!(lib[len - 1], pinned[1], "Mountain must be top of library");
        assert_eq!(lib[len - 2], pinned[0], "Forest must be second from top");
    }

    // Surveil putting ALL cards into graveyard finishes automatically without Decline.
    let mut s_all = five_color_setup(&["Esper Origins", "Forest", "Mountain", "Island"]);
    let pinned = pin_top(&mut s_all, &["Forest", "Mountain"]);
    cast(&mut s_all, "Esper Origins");
    s_all.advance_until(
        |s| {
            matches!(
                s.game()
                    .state
                    .suspended_decision
                    .as_ref()
                    .map(|d| &d.decision),
                Some(Decision::Surveil { .. })
            )
        },
        "reach Esper Origins surveil".into(),
    );
    let (c0, c1) = (
        card_render(s_all.game(), pinned[0]),
        card_render(s_all.game(), pinned[1]),
    );
    let bin_all = GroupPlan::Surveil {
        graveyard: vec![c0, c1],
    };
    let applied = assert_apply_matches_manual(s_all.game_mut(), 0, &bin_all);
    assert_eq!(
        applied.steps.len(),
        2,
        "no trailing Decline when binning all"
    );
    assert!(applied.steps.iter().all(|step| step.item.is_some()));
    assert_eq!(
        applied.outcome,
        Some(GroupOutcome::Surveil {
            graveyard: vec![c0, c1],
            kept: vec![],
        })
    );
    assert!(s_all.game().group(PlayerId(0)).is_none());
}

#[test]
fn surveil_three_mid_group_raw_step_exposes_committed_graveyard() {
    let (mut s, pinned) = setup_library_effect(
        &["Forest", "Mountain", "Island"],
        vec![Effect::Surveil { count: 3 }],
        true,
    );
    let (c0, c1, c2) = (
        card_render(s.game(), pinned[0]),
        card_render(s.game(), pinned[1]),
        card_render(s.game(), pinned[2]),
    );

    // Take 1 raw step marking Forest (c0) for the graveyard.
    let raw_idx = s
        .action_space()
        .actions
        .iter()
        .position(|a| matches!(a, Action::SelectCard { object, .. } if *object == pinned[0]))
        .unwrap();
    s.step_action(raw_idx);

    let g = s
        .game()
        .group(PlayerId(0))
        .expect("2 cards remain in Surveil");
    match &g.spec {
        GroupSpec::Surveil {
            items,
            committed_graveyard,
        } => {
            assert_eq!(items.len(), 2);
            assert_eq!(committed_graveyard, &vec![c0]);
        }
        other => panic!("expected Surveil, got {other:?}"),
    }

    // Put Island (c2) in graveyard and keep Mountain (c1) on top.
    let plan = GroupPlan::Surveil {
        graveyard: vec![c2],
    };
    let applied = assert_apply_matches_manual(s.game_mut(), 0, &plan);
    assert_eq!(applied.steps.len(), 2);
    assert_eq!(applied.steps[0].item, Some(c2));
    assert_eq!(applied.steps[1].item, None);
    assert_eq!(
        applied.outcome,
        Some(GroupOutcome::Surveil {
            graveyard: vec![c0, c2],
            kept: vec![c1],
        })
    );
}

#[test]
fn order_library_bottom_two_and_three_cards_including_mid_group_raw_step() {
    // 1. Accumulate Wisdom: look at top 3, put 1 in hand, order remaining 2 on bottom.
    for skip_trivial in [true, false] {
        let mut s = five_color_setup(&["Accumulate Wisdom", "Forest", "Mountain", "Island"]);
        s.game_mut().skip_trivial = skip_trivial;
        let pinned = pin_top(&mut s, &["Forest", "Mountain", "Island"]);
        cast(&mut s, "Accumulate Wisdom");
        s.advance_until(
            |s| {
                matches!(
                    s.game()
                        .state
                        .suspended_decision
                        .as_ref()
                        .map(|d| &d.decision),
                    Some(Decision::LookAndSelect { .. })
                )
            },
            "reach Accumulate Wisdom look".into(),
        );
        // Pick Forest (pinned[0]) for hand via ordinary LookAndSelect step.
        let pick_forest = s
            .action_space()
            .actions
            .iter()
            .position(|a| matches!(a, Action::SelectCard { object, .. } if *object == pinned[0]))
            .unwrap();
        s.step_action(pick_forest);

        let g = s
            .game()
            .group(PlayerId(0))
            .expect("OrderLibrary Bottom (2 cards)");
        let (c1, c2) = (
            card_render(s.game(), pinned[1]),
            card_render(s.game(), pinned[2]),
        );
        assert!(matches!(
            g.spec,
            GroupSpec::OrderLibrary {
                end: LibraryEnd::Bottom,
                ref items,
                ref committed,
            } if items.len() == 2 && committed.is_empty()
        ));

        // Order Island (c2) above Mountain (c1) on the bottom (c1 at index 0).
        let plan = GroupPlan::OrderLibrary {
            order: vec![c2, c1],
        };
        let applied = assert_apply_matches_manual(s.game_mut(), 0, &plan);
        assert_eq!(applied.steps.len(), if skip_trivial { 1 } else { 2 });
        assert_eq!(
            applied.outcome,
            Some(GroupOutcome::OrderLibrary {
                end: LibraryEnd::Bottom,
                order: vec![c2, c1],
            })
        );
        let lib = s
            .game()
            .state
            .zones
            .zone_objects(ZoneType::Library, PlayerId(0));
        assert_eq!(lib[0], pinned[1], "Mountain must sit on the very bottom");
        assert_eq!(lib[1], pinned[2], "Island must sit directly above Mountain");
    }

    // 2. 3-card OrderLibrary Bottom with mid-group raw step.
    let (mut s, pinned) = setup_library_effect(
        &["Forest", "Mountain", "Island", "Swamp"],
        vec![Effect::LookAndSelect {
            max_mana_value: None,
            to_battlefield: false,
            random_order: false,
            reveal: false,
            look: EffectValue::Fixed(4),
            min_select: 1,
            max_select: 1,
            predicate: CardPredicate::default(),
        }],
        true,
    );
    // Pick Forest (pinned[0]) for hand; leaves [Mountain, Island, Swamp] to order on bottom.
    let pick_forest = s
        .action_space()
        .actions
        .iter()
        .position(|a| matches!(a, Action::SelectCard { object, .. } if *object == pinned[0]))
        .unwrap();
    s.step_action(pick_forest);

    let (m, i, sw) = (
        card_render(s.game(), pinned[1]),
        card_render(s.game(), pinned[2]),
        card_render(s.game(), pinned[3]),
    );
    // Take 1 raw step placing Island (pinned[2]) first (highest of the bottom 3).
    let pick_island = s
        .action_space()
        .actions
        .iter()
        .position(|a| matches!(a, Action::SelectCard { object, .. } if *object == pinned[2]))
        .unwrap();
    s.step_action(pick_island);

    let g = s
        .game()
        .group(PlayerId(0))
        .expect("2 cards remain in OrderLibrary Bottom");
    match &g.spec {
        GroupSpec::OrderLibrary {
            end,
            items,
            committed,
        } => {
            assert_eq!(*end, LibraryEnd::Bottom);
            assert_eq!(items.len(), 2);
            assert_eq!(committed, &vec![i]);
        }
        other => panic!("expected OrderLibrary, got {other:?}"),
    }

    // Finish remaining two: Swamp above Mountain (so bottom is Mountain, Swamp, Island from index 0 up).
    let plan = GroupPlan::OrderLibrary { order: vec![sw, m] };
    let applied = assert_apply_matches_manual(s.game_mut(), 0, &plan);
    assert_eq!(
        applied.outcome,
        Some(GroupOutcome::OrderLibrary {
            end: LibraryEnd::Bottom,
            order: vec![i, sw, m],
        })
    );
    let lib = s
        .game()
        .state
        .zones
        .zone_objects(ZoneType::Library, PlayerId(0));
    assert_eq!(lib[0], pinned[1], "Mountain at very bottom (index 0)");
    assert_eq!(lib[1], pinned[3], "Swamp above Mountain (index 1)");
    assert_eq!(lib[2], pinned[2], "Island above Swamp (index 2)");
}
