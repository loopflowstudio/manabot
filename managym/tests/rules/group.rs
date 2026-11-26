//! Native unified decision API tests covering `Game::decision` / `preview` /
//! `submit` across single offers, combat declarations, and library arrangements
//! (Scry, Surveil, OrderLibrary Top/Bottom), `GroupCard` committed labels,
//! purity boundary behavior, mid-group raw steps, and digest parity between
//! `submit` and step-by-step execution (including mixed `[Agent, Heuristic]` `Env`).
use managym::{
    agent::{
        action::{Action, ActionSpaceKind},
        env::Env,
        player_policy::PlayerPolicy,
        structured_offer::OfferVerb,
    },
    experience::{ObjectRenderId, SubjectRef},
    flow::{
        decision::Decision,
        group::{
            ActiveDecision, DecisionAddress, DecisionChoice, DecisionOutcome, DecisionPreview,
            DecisionSpec, GroupCard, LibraryEnd, PlannedBlock,
        },
        turn::StepKind,
    },
    search_state::witness,
    state::{
        ability::{Effect, EffectValue},
        card::ActivatedAbilityDefinition,
        game_object::{ObjectId, PermanentId, PlayerId},
        player::PlayerConfig,
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

fn follow_steps(game: &mut Game, preview: &DecisionPreview) {
    for step in &preview.steps {
        let frame = game.semantic_decision_frame().unwrap();
        assert_eq!(
            (frame.revision, &frame.fingerprint),
            (step.revision, &step.fingerprint)
        );
        game.step(step.action_index).unwrap();
    }
}

/// Assert that `preview` is read-only and that `submit` reaches the exact same
/// canonical state digest as stepping `preview.steps` manually.
fn assert_submit_matches_manual(
    game: &mut Game,
    viewer: usize,
    choice: &DecisionChoice,
) -> DecisionPreview {
    let before_witness = witness(game);
    let before_digest = game.state.deterministic_hash();
    let active = game.decision(Some(PlayerId(viewer))).unwrap();
    assert_eq!(active.address.viewer, viewer);
    assert!(!active.offers.is_empty());

    // Default `at=None` and explicit `at=Some(&active.address)` produce identical previews.
    let preview = game.preview(Some(&active.address), choice).unwrap();
    assert_eq!(game.preview(None, choice).unwrap(), preview);
    assert_eq!(witness(game), before_witness, "preview must be read-only");
    assert_eq!(game.state.deterministic_hash(), before_digest);
    assert!(preview.complete, "expected complete preview: {preview:?}");
    assert!(preview.refused.is_none(), "{:?}", preview.refused);

    let mut manual = game.clone();
    follow_steps(&mut manual, &preview);

    let submitted = game.submit(Some(&active.address), choice).unwrap();
    assert_eq!(submitted, preview);
    assert_eq!(
        game.state.deterministic_hash(),
        manual.state.deterministic_hash(),
        "submit and manual step execution must produce identical digests"
    );
    submitted
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
fn single_decision_preview_and_submit_match_manual_step() {
    let mut s = Scenario::new(
        deck(&[("Mountain", 30), ("Gray Ogre", 10)]),
        deck(&[("Forest", 40)]),
        7,
    );
    s.advance_to_active_step(0, StepKind::Main);

    let d = s.game().decision(None).expect("priority decision active");
    assert_eq!(d.address.viewer, 0);
    assert_eq!(d.spec, DecisionSpec::Single);
    assert!(s.game().decision(Some(PlayerId(1))).is_none());
    assert!(!d.offers.is_empty());

    // Grouped choice against a Single decision errors without mutating.
    let before = witness(s.game());
    assert!(s
        .game()
        .preview(
            None,
            &DecisionChoice::DeclareAttackers { attackers: vec![] }
        )
        .is_err());
    // Out-of-bounds single offer id returns refused without mutating.
    let oob = s
        .game()
        .preview(None, &DecisionChoice::Single { offer_id: 999 })
        .unwrap();
    assert!(!oob.complete);
    assert!(oob.refused.is_some());
    assert!(s
        .game_mut()
        .submit(None, &DecisionChoice::Single { offer_id: 999 })
        .is_err());
    assert_eq!(witness(s.game()), before);

    // Legal single offer submits atomically and matches manual step.
    let pass_id = d
        .offers
        .iter()
        .find(|o| o.label.contains("Pass"))
        .map(|o| o.id)
        .unwrap_or(d.offers[0].id);
    let submitted = assert_submit_matches_manual(
        s.game_mut(),
        0,
        &DecisionChoice::Single { offer_id: pass_id },
    );
    assert_eq!(
        submitted.outcome,
        Some(DecisionOutcome::Single { offer_id: pass_id })
    );
}

#[test]
fn combat_three_attackers_one_attacker_multi_block_and_refusals() {
    // 1. Three-creature attack + multi-block + menace refusal.
    let (mut s, attackers, blockers) = combat_three_vs_three();
    let active = s
        .game()
        .decision(Some(PlayerId(0)))
        .expect("attack decision active");
    assert_eq!(s.game().decision(None), Some(active.clone()));
    assert!(
        s.game().decision(Some(PlayerId(1))).is_none(),
        "non-actor gets None"
    );
    match &active.spec {
        DecisionSpec::DeclareAttackers {
            declarable,
            committed,
        } => {
            assert_eq!(declarable.len(), 3);
            assert!(committed.is_empty());
        }
        other => panic!("expected DeclareAttackers, got {other:?}"),
    }

    // Wrong choice kind and stale revision error without mutating.
    let before = witness(s.game());
    assert!(s
        .game()
        .preview(
            Some(&active.address),
            &DecisionChoice::Scry {
                top: vec![],
                bottom: vec![],
            },
        )
        .is_err());
    let stale_addr = DecisionAddress {
        viewer: active.address.viewer,
        revision: active.address.revision + 1,
        fingerprint: active.address.fingerprint.clone(),
    };
    assert!(s
        .game_mut()
        .submit(
            Some(&stale_addr),
            &DecisionChoice::DeclareAttackers { attackers: vec![] },
        )
        .is_err());
    assert_eq!(witness(s.game()), before);

    // Mid-combat raw step: submit one attack offer id, verify decision() shows remaining + committed.
    let mut mid_combat = s.game().clone();
    let first_attack_offer = mid_combat
        .action_space()
        .unwrap()
        .actions
        .iter()
        .position(|a| matches!(a, Action::DeclareAttacker { attack: true, .. }))
        .unwrap();
    mid_combat
        .submit(
            None,
            &DecisionChoice::Single {
                offer_id: first_attack_offer,
            },
        )
        .unwrap();
    let mid_decision = mid_combat.decision(None).expect("2 attackers remain");
    match &mid_decision.spec {
        DecisionSpec::DeclareAttackers {
            declarable,
            committed,
        } => {
            assert_eq!(declarable.len(), 2);
            assert_eq!(committed.len(), 1);
        }
        other => panic!("expected DeclareAttackers mid-combat, got {other:?}"),
    }

    // Attack with all three so we can test blockers next.
    let all_attackers: Vec<_> = attackers
        .iter()
        .map(|&id| perm_render(s.game(), id))
        .collect();
    let attack_choice = DecisionChoice::DeclareAttackers {
        attackers: all_attackers.clone(),
    };
    let attack_result = assert_submit_matches_manual(s.game_mut(), 0, &attack_choice);
    assert_eq!(attack_result.steps.len(), 3);
    assert!(matches!(
        attack_result.outcome,
        Some(DecisionOutcome::DeclareAttackers { ref attackers }) if attackers.len() == 3
    ));

    // Advance to blockers.
    s.advance_until(
        |s| s.action_space().kind == ActionSpaceKind::DeclareBlocker,
        "failed to reach blockers".into(),
    );
    let blocker_decision = s
        .game()
        .decision(Some(PlayerId(1)))
        .expect("blocker decision active");
    assert!(matches!(
        blocker_decision.spec,
        DecisionSpec::DeclareBlockers { ref declarable, .. } if declarable.len() == 3
    ));

    // Menace refusal: lone Elf on Boggart Brute refuses without mutating.
    let lone_menace = DecisionChoice::DeclareBlockers {
        blocks: vec![PlannedBlock {
            blocker: perm_render(s.game(), blockers[0]),
            attacker: perm_render(s.game(), attackers[0]),
        }],
    };
    let before_menace = witness(s.game());
    let menace_preview = s
        .game()
        .preview(Some(&blocker_decision.address), &lone_menace)
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
        .submit(Some(&blocker_decision.address), &lone_menace)
        .is_err());
    assert_eq!(witness(s.game()), before_menace);

    // Legal multi-block: two Elves on Brute, third on Ogre.
    let multi_block = DecisionChoice::DeclareBlockers {
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
    assert_submit_matches_manual(s.game_mut(), 1, &multi_block);

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
    let single_decision = single
        .game()
        .decision(Some(PlayerId(0)))
        .expect("1-creature decision");
    assert!(matches!(
        single_decision.spec,
        DecisionSpec::DeclareAttackers { ref declarable, .. } if declarable.len() == 1
    ));
    let single_choice = DecisionChoice::DeclareAttackers {
        attackers: vec![perm_render(single.game(), ogre)],
    };
    let submitted = assert_submit_matches_manual(single.game_mut(), 0, &single_choice);
    assert_eq!(submitted.steps.len(), 1);
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

    let d = s.game().decision(Some(PlayerId(0))).expect("attackers");
    let before = witness(s.game());
    let omit_jugg = DecisionChoice::DeclareAttackers {
        attackers: vec![perm_render(s.game(), ogre)],
    };
    let preview = s.game().preview(Some(&d.address), &omit_jugg).unwrap();
    assert!(!preview.complete);
    assert_eq!(
        preview.refused.as_ref().unwrap().reason,
        "Juggernaut must attack."
    );
    assert!(s.game_mut().submit(Some(&d.address), &omit_jugg).is_err());
    assert_eq!(witness(s.game()), before);

    let both = DecisionChoice::DeclareAttackers {
        attackers: vec![perm_render(s.game(), jugg), perm_render(s.game(), ogre)],
    };
    assert_submit_matches_manual(s.game_mut(), 0, &both);

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
    let atk_choice = DecisionChoice::DeclareAttackers {
        attackers: vec![perm_render(b.game(), drake), perm_render(b.game(), ogre2)],
    };
    assert_submit_matches_manual(b.game_mut(), 0, &atk_choice);
    b.advance_until(
        |s| s.action_space().kind == ActionSpaceKind::DeclareBlocker,
        "reach blockers".into(),
    );
    let bd = b.game().decision(Some(PlayerId(1))).expect("blockers");
    let bad_blocks = DecisionChoice::DeclareBlockers {
        blocks: vec![PlannedBlock {
            blocker: perm_render(b.game(), elves),
            attacker: perm_render(b.game(), ogre2),
        }],
    };
    let before_block = witness(b.game());
    let bad_preview = b.game().preview(Some(&bd.address), &bad_blocks).unwrap();
    assert!(!bad_preview.complete);
    assert_eq!(
        bad_preview.refused.as_ref().unwrap().reason,
        "Giant Spider must block Wind Drake."
    );
    assert!(b.game_mut().submit(Some(&bd.address), &bad_blocks).is_err());
    assert_eq!(witness(b.game()), before_block);

    let good_blocks = DecisionChoice::DeclareBlockers {
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
    assert_submit_matches_manual(b.game_mut(), 1, &good_blocks);
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
    let d = s.game().decision(Some(PlayerId(0))).expect("Scry 3");
    let (c0, c1, c2) = (
        card_render(s.game(), pinned[0]),
        card_render(s.game(), pinned[1]),
        card_render(s.game(), pinned[2]),
    );
    match &d.spec {
        DecisionSpec::Scry {
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

    // Incomplete choice returns complete=false, refused=None, steps=[], outcome=None without mutating.
    let before = witness(s.game());
    let incomplete = DecisionChoice::Scry {
        top: vec![c2],
        bottom: vec![c0],
    };
    let inc_preview = s.game().preview(Some(&d.address), &incomplete).unwrap();
    assert!(!inc_preview.complete);
    assert!(inc_preview.refused.is_none());
    assert!(inc_preview.steps.is_empty());
    assert!(inc_preview.outcome.is_none());
    assert!(s.game_mut().submit(Some(&d.address), &incomplete).is_err());
    assert_eq!(witness(s.game()), before);

    // Duplicate card is refused.
    let dup = DecisionChoice::Scry {
        top: vec![c0, c0],
        bottom: vec![c1],
    };
    let dup_preview = s.game().preview(Some(&d.address), &dup).unwrap();
    assert!(!dup_preview.complete);
    assert!(dup_preview
        .refused
        .as_ref()
        .unwrap()
        .reason
        .contains("only be placed once"));

    // Choice: top=[c2 (Island)], bottom=[c0 (Forest), c1 (Mountain)].
    // Reading order for bottom: c0 sits ABOVE c1, and c1 sits on the VERY bottom (index 0).
    let mut mid_raw = s.game().clone();
    let choice = DecisionChoice::Scry {
        top: vec![c2],
        bottom: vec![c0, c1],
    };
    let submitted = assert_submit_matches_manual(s.game_mut(), 0, &choice);
    assert_eq!(submitted.steps.len(), 3);
    assert_eq!(
        submitted.outcome,
        Some(DecisionOutcome::Scry {
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

    // Mid-group raw step: put c1 on bottom via raw step, verify committed_bottom carries GroupCard label.
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
    let after_one = mid_raw
        .decision(Some(PlayerId(0)))
        .expect("2 cards still remain");
    assert_eq!(
        after_one.spec,
        DecisionSpec::Scry {
            items: vec![
                GroupCard {
                    item: c0,
                    label: "Forest".into()
                },
                GroupCard {
                    item: c2,
                    label: "Island".into()
                },
            ],
            committed_top: vec![],
            committed_bottom: vec![GroupCard {
                item: c1,
                label: "Mountain".into()
            }],
        }
    );
    // Finish the remaining two cards via submit: c2 on top, c0 on bottom (above c1!).
    let rest_choice = DecisionChoice::Scry {
        top: vec![c2],
        bottom: vec![c0],
    };
    let rest_submitted = assert_submit_matches_manual(&mut mid_raw, 0, &rest_choice);
    assert_eq!(
        rest_submitted.outcome,
        Some(DecisionOutcome::Scry {
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
        "mid-group raw step + submit must match full submit"
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
    let d = s.game().decision(Some(PlayerId(0))).expect("Scry 2");
    let (c0, c1) = (
        card_render(s.game(), pinned[0]),
        card_render(s.game(), pinned[1]),
    );
    assert!(matches!(d.spec, DecisionSpec::Scry { ref items, .. } if items.len() == 2));

    // Take 1 raw step on a clone to verify domain threshold: 1 remaining card drops to Single.
    let mut one_left = s.game().clone();
    one_left.step(0).unwrap();
    let dropped = one_left.decision(Some(PlayerId(0))).expect("1 card left");
    assert_eq!(
        dropped.spec,
        DecisionSpec::Single,
        "1 remaining Scry card is below the >= 2 threshold and drops to Single"
    );

    // Put Mountain (c1) on top to draw it, Forest (c0) on bottom.
    // The post-group draw fires after Scry finishes and must NOT trip the purity boundary.
    let choice = DecisionChoice::Scry {
        top: vec![c1],
        bottom: vec![c0],
    };
    let submitted = assert_submit_matches_manual(s.game_mut(), 0, &choice);
    assert_eq!(submitted.steps.len(), 2);
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

        let surveil_decision = s.game().decision(Some(PlayerId(0))).expect("Surveil 2");
        let (c0, c1) = (
            card_render(s.game(), pinned[0]),
            card_render(s.game(), pinned[1]),
        );
        assert!(matches!(
            surveil_decision.spec,
            DecisionSpec::Surveil { ref items, ref committed_graveyard }
                if items.len() == 2 && committed_graveyard.is_empty()
        ));

        // Keep both on top (graveyard = []) -> emits 1 Decline step with item=None.
        let keep_all = DecisionChoice::Surveil { graveyard: vec![] };
        let surveil_submitted = assert_submit_matches_manual(s.game_mut(), 0, &keep_all);
        assert_eq!(surveil_submitted.steps.len(), 1);
        assert_eq!(surveil_submitted.steps[0].item, None);
        assert_eq!(
            surveil_submitted.outcome,
            Some(DecisionOutcome::Surveil {
                graveyard: vec![],
                kept: vec![c0, c1],
            })
        );

        // Immediately opens OrderLibrary { end: Top } with 2 cards even when skip_trivial=true.
        let order_decision: ActiveDecision = s
            .game()
            .decision(Some(PlayerId(0)))
            .expect("OrderLibrary Top after keeping 2 on Surveil");
        assert!(matches!(
            order_decision.spec,
            DecisionSpec::OrderLibrary {
                end: LibraryEnd::Top,
                ref items,
                ref committed,
            } if items.len() == 2 && committed.is_empty()
        ));

        // Incomplete OrderLibrary choice returns complete=false without cloning.
        let incomplete = DecisionChoice::OrderLibrary { order: vec![c1] };
        let inc_preview = s
            .game()
            .preview(Some(&order_decision.address), &incomplete)
            .unwrap();
        assert!(!inc_preview.complete);
        assert!(inc_preview.refused.is_none());
        assert!(inc_preview.steps.is_empty());

        // Order Mountain (c1) on very top, Forest (c0) second.
        let order_choice = DecisionChoice::OrderLibrary {
            order: vec![c1, c0],
        };
        let order_submitted = assert_submit_matches_manual(s.game_mut(), 0, &order_choice);
        assert_eq!(
            order_submitted.steps.len(),
            if skip_trivial { 1 } else { 2 }
        );
        assert_eq!(
            order_submitted.outcome,
            Some(DecisionOutcome::OrderLibrary {
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
    let bin_all = DecisionChoice::Surveil {
        graveyard: vec![c0, c1],
    };
    let submitted = assert_submit_matches_manual(s_all.game_mut(), 0, &bin_all);
    assert_eq!(
        submitted.steps.len(),
        2,
        "no trailing Decline when binning all"
    );
    assert!(submitted.steps.iter().all(|step| step.item.is_some()));
    assert_eq!(
        submitted.outcome,
        Some(DecisionOutcome::Surveil {
            graveyard: vec![c0, c1],
            kept: vec![],
        })
    );
    assert!(matches!(
        s_all.game().decision(Some(PlayerId(0))).map(|d| d.spec),
        None | Some(DecisionSpec::Single)
    ));
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

    let d = s
        .game()
        .decision(Some(PlayerId(0)))
        .expect("2 cards remain in Surveil");
    match &d.spec {
        DecisionSpec::Surveil {
            items,
            committed_graveyard,
        } => {
            assert_eq!(items.len(), 2);
            assert_eq!(
                committed_graveyard,
                &vec![GroupCard {
                    item: c0,
                    label: "Forest".into(),
                }]
            );
        }
        other => panic!("expected Surveil, got {other:?}"),
    }

    // Put Island (c2) in graveyard and keep Mountain (c1) on top.
    let choice = DecisionChoice::Surveil {
        graveyard: vec![c2],
    };
    let submitted = assert_submit_matches_manual(s.game_mut(), 0, &choice);
    assert_eq!(submitted.steps.len(), 2);
    assert_eq!(submitted.steps[0].item, Some(c2));
    assert_eq!(submitted.steps[1].item, None);
    assert_eq!(
        submitted.outcome,
        Some(DecisionOutcome::Surveil {
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

        let d = s
            .game()
            .decision(Some(PlayerId(0)))
            .expect("OrderLibrary Bottom (2 cards)");
        let (c1, c2) = (
            card_render(s.game(), pinned[1]),
            card_render(s.game(), pinned[2]),
        );
        assert!(matches!(
            d.spec,
            DecisionSpec::OrderLibrary {
                end: LibraryEnd::Bottom,
                ref items,
                ref committed,
            } if items.len() == 2 && committed.is_empty()
        ));

        // Order Island (c2) above Mountain (c1) on the bottom (c1 at index 0).
        let choice = DecisionChoice::OrderLibrary {
            order: vec![c2, c1],
        };
        let submitted = assert_submit_matches_manual(s.game_mut(), 0, &choice);
        assert_eq!(submitted.steps.len(), if skip_trivial { 1 } else { 2 });
        assert_eq!(
            submitted.outcome,
            Some(DecisionOutcome::OrderLibrary {
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

    // 2. 3-card OrderLibrary Bottom with mid-group raw step -> committed carries GroupCard label.
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

    let d = s
        .game()
        .decision(Some(PlayerId(0)))
        .expect("2 cards remain in OrderLibrary Bottom");
    match &d.spec {
        DecisionSpec::OrderLibrary {
            end,
            items,
            committed,
        } => {
            assert_eq!(*end, LibraryEnd::Bottom);
            assert_eq!(items.len(), 2);
            assert_eq!(
                committed,
                &vec![GroupCard {
                    item: i,
                    label: "Island".into(),
                }]
            );
        }
        other => panic!("expected OrderLibrary, got {other:?}"),
    }

    // Finish remaining two: Swamp above Mountain (so bottom is Mountain, Swamp, Island from index 0 up).
    let choice = DecisionChoice::OrderLibrary { order: vec![sw, m] };
    let submitted = assert_submit_matches_manual(s.game_mut(), 0, &choice);
    assert_eq!(
        submitted.outcome,
        Some(DecisionOutcome::OrderLibrary {
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

#[test]
fn env_submit_settles_mixed_policy_identically_to_env_step() {
    let setup = vec![
        PlayerConfig::new("Hero", deck(&[("Mountain", 20), ("Gray Ogre", 20)])),
        PlayerConfig::new("Bot", deck(&[("Forest", 20), ("Llanowar Elves", 20)])),
    ];
    let mut env = Env::new(19, true, false, false)
        .with_player_policies([PlayerPolicy::Agent, PlayerPolicy::Heuristic]);
    env.reset(setup).unwrap();

    // Keep opening hand, then put ready attackers and heuristic blockers on the board.
    while let Some(d) = env.decision(None).unwrap() {
        let Some(keep) = d.offers.iter().find(|o| o.label.contains("Keep")) else {
            break;
        };
        env.submit(None, &DecisionChoice::Single { offer_id: keep.id })
            .unwrap();
    }
    env.scenario_force_battlefield(0, "Gray Ogre", true)
        .unwrap();
    env.scenario_force_battlefield(0, "Gray Ogre", true)
        .unwrap();
    env.scenario_force_battlefield(1, "Llanowar Elves", true)
        .unwrap();
    env.scenario_force_battlefield(1, "Llanowar Elves", true)
        .unwrap();
    env.scenario_refresh().unwrap();

    // Advance until player 0 reaches DeclareAttackers with both Gray Ogres.
    for _ in 0..80 {
        let d = env.decision(None).unwrap().expect("active decision");
        assert_eq!(d.address.viewer, 0, "mixed env must settle back to seat 0");
        if let DecisionSpec::DeclareAttackers { ref declarable, .. } = d.spec {
            assert_eq!(declarable.len(), 2);
            let choice = DecisionChoice::DeclareAttackers {
                attackers: declarable.iter().map(|c| c.item).collect(),
            };
            let preview = env.preview(None, &choice).unwrap();
            assert!(preview.complete);
            assert_eq!(preview.steps.len(), 2);

            let mut manual = env.fork().unwrap();
            let mut manual_out = None;
            for step in &preview.steps {
                manual_out = Some(manual.step(step.action_index as i64).unwrap());
            }
            let (m_obs, m_reward, m_done, m_trunc, _) = manual_out.unwrap();
            let (s_obs, s_reward, s_done, s_trunc, _) = env.submit(None, &choice).unwrap();

            assert_eq!(env.state_digest().unwrap(), manual.state_digest().unwrap());
            assert_eq!(s_obs.agent.player_index, m_obs.agent.player_index);
            assert_eq!(s_reward, m_reward);
            assert_eq!(s_done, m_done);
            assert_eq!(s_trunc, m_trunc);
            return;
        }
        let offer_id = d
            .offers
            .iter()
            .find(|o| o.verb == OfferVerb::PassPriority)
            .unwrap_or(&d.offers[0])
            .id;
        env.submit(None, &DecisionChoice::Single { offer_id })
            .unwrap();
    }
    panic!("did not reach DeclareAttackers in mixed-policy env");
}

#[test]
fn single_decision_offers_have_readable_labels_sources_and_unchanged_fingerprint() {
    let mut s = five_color_setup(&["Mountain", "Island", "Gray Ogre", "Hill Giant"]);
    s.force_card_in_hand(0, "Mountain");
    s.force_card_in_hand(0, "Island");
    s.force_card_in_hand(0, "Gray Ogre");
    s.force_card_in_hand(0, "Hill Giant");
    s.game_mut().scenario_refresh_priority().unwrap();

    let raw_frame = s.game().semantic_decision_frame().unwrap();
    let d = s.game().decision(None).expect("priority decision");
    assert_eq!(d.spec, DecisionSpec::Single);
    assert_eq!(
        d.address.fingerprint, raw_frame.fingerprint,
        "Offer enrichment must not alter semantic_decision_frame().fingerprint"
    );
    // Raw DecisionFrame still carries the unchanged structured_search_offers labels.
    assert!(
        raw_frame
            .offers
            .iter()
            .any(|o| o.label == "PriorityPlayLand" && o.source.is_none()),
        "raw DecisionFrame offers must remain untouched"
    );

    // Enriched Decision offers distinguish lands and spells by name and populate SubjectRef.
    let play_mountain = d
        .offers
        .iter()
        .find(|o| o.label == "Play Mountain")
        .expect("Play Mountain offer");
    let play_island = d
        .offers
        .iter()
        .find(|o| o.label == "Play Island")
        .expect("Play Island offer");
    let cast_ogre = d
        .offers
        .iter()
        .find(|o| o.label == "Cast Gray Ogre")
        .expect("Cast Gray Ogre offer");
    let cast_giant = d
        .offers
        .iter()
        .find(|o| o.label == "Cast Hill Giant")
        .expect("Cast Hill Giant offer");
    let pass = d
        .offers
        .iter()
        .find(|o| o.label == "Pass priority")
        .expect("Pass priority offer");

    for offer in [play_mountain, play_island, cast_ogre, cast_giant] {
        assert!(
            matches!(offer.source, Some(SubjectRef::Object { .. })),
            "expected Object SubjectRef on {offer:?}"
        );
    }
    assert_ne!(play_mountain.source, play_island.source);
    assert_ne!(cast_ogre.source, cast_giant.source);
    assert_eq!(pass.source, None);
}

#[test]
fn partition_piles_full_mid_group_threshold_refusal_and_choose_pile_transition() {
    let (mut s, pinned) = setup_library_effect(
        &["Gray Ogre", "Forest", "Mountain", "Island", "Swamp"],
        vec![Effect::SeparatePiles { look: 5 }],
        true,
    );
    let ids: Vec<_> = pinned.iter().map(|&obj| card_render(s.game(), obj)).collect();
    let d = s.game().decision(Some(PlayerId(0))).expect("PartitionPiles 5");
    assert!(s.game().decision(Some(PlayerId(1))).is_none());
    match &d.spec {
        DecisionSpec::PartitionPiles {
            items,
            committed_face_up,
            committed_face_down,
        } => {
            assert_eq!(items.len(), 5);
            assert_eq!(
                items.iter().map(|c| c.label.as_str()).collect::<Vec<_>>(),
                vec!["Gray Ogre", "Forest", "Mountain", "Island", "Swamp"]
            );
            assert!(committed_face_up.is_empty());
            assert!(committed_face_down.is_empty());
        }
        other => panic!("expected PartitionPiles, got {other:?}"),
    }

    // Duplicate and unknown cards are refused without mutating.
    let before = witness(s.game());
    let dup = s
        .game()
        .preview(
            Some(&d.address),
            &DecisionChoice::PartitionPiles {
                face_up: vec![ids[0], ids[0]],
            },
        )
        .unwrap();
    assert!(!dup.complete);
    assert!(dup
        .refused
        .as_ref()
        .unwrap()
        .reason
        .contains("only be chosen once"));
    let unknown = s
        .game()
        .preview(
            Some(&d.address),
            &DecisionChoice::PartitionPiles {
                face_up: vec![ObjectRenderId {
                    entity: 99999,
                    incarnation: 0,
                }],
            },
        )
        .unwrap();
    assert!(!unknown.complete);
    assert!(unknown
        .refused
        .as_ref()
        .unwrap()
        .reason
        .contains("not in the pending pile split"));
    assert_eq!(witness(s.game()), before);

    // Empty face-up pile and all-face-up pile both preview cleanly.
    let all_down = s
        .game()
        .preview(
            Some(&d.address),
            &DecisionChoice::PartitionPiles { face_up: vec![] },
        )
        .unwrap();
    assert!(all_down.complete);
    assert_eq!(all_down.steps.len(), 5);
    assert_eq!(
        all_down.outcome,
        Some(DecisionOutcome::PartitionPiles {
            face_up: vec![],
            face_down: ids.clone(),
        })
    );

    // Clone for mid-group raw stepping before committing the full split.
    let mut mid = s.game().clone();

    // Full split: Gray Ogre (0) and Mountain (2) face-up; Forest (1), Island (3), Swamp (4) face-down.
    // The final step emits CardRevealed and transitions to ChoosePile for opponent (seat 1)
    // without tripping the inter-step purity boundary.
    let full_choice = DecisionChoice::PartitionPiles {
        face_up: vec![ids[0], ids[2]],
    };
    let submitted = assert_submit_matches_manual(s.game_mut(), 0, &full_choice);
    assert_eq!(submitted.steps.len(), 5);
    assert_eq!(
        submitted.outcome,
        Some(DecisionOutcome::PartitionPiles {
            face_up: vec![ids[0], ids[2]],
            face_down: vec![ids[1], ids[3], ids[4]],
        })
    );
    let next_decision = s
        .game()
        .decision(Some(PlayerId(1)))
        .expect("opponent ChoosePile active");
    assert_eq!(next_decision.spec, DecisionSpec::Single);
    assert!(matches!(
        s.game().state.suspended_decision.as_ref().map(|sd| &sd.decision),
        Some(Decision::ChoosePile { player, .. }) if *player == PlayerId(1)
    ));

    // Mid-group raw stepping: put card 0 face-up (mode 0), card 1 face-down (mode 1).
    mid.step(0).unwrap();
    mid.step(1).unwrap();
    let mid_decision = mid.decision(Some(PlayerId(0))).expect("3 cards remain");
    assert_eq!(
        mid_decision.spec,
        DecisionSpec::PartitionPiles {
            items: vec![
                GroupCard {
                    item: ids[2],
                    label: "Mountain".into(),
                },
                GroupCard {
                    item: ids[3],
                    label: "Island".into(),
                },
                GroupCard {
                    item: ids[4],
                    label: "Swamp".into(),
                },
            ],
            committed_face_up: vec![GroupCard {
                item: ids[0],
                label: "Gray Ogre".into(),
            }],
            committed_face_down: vec![GroupCard {
                item: ids[1],
                label: "Forest".into(),
            }],
        }
    );

    // Already-committed card is refused.
    let committed_refusal = mid
        .preview(
            Some(&mid_decision.address),
            &DecisionChoice::PartitionPiles {
                face_up: vec![ids[0]],
            },
        )
        .unwrap();
    assert!(!committed_refusal.complete);
    assert!(committed_refusal
        .refused
        .as_ref()
        .unwrap()
        .reason
        .contains("already been placed"));

    // Threshold check: stepping to 1 remaining card drops to Single.
    let mut one_left = mid.clone();
    one_left.step(0).unwrap();
    one_left.step(1).unwrap();
    assert_eq!(
        one_left.decision(Some(PlayerId(0))).unwrap().spec,
        DecisionSpec::Single
    );

    // Submitting the remaining 3 cards (Mountain face-up; Island, Swamp face-down) matches full submit digest.
    let rest_submitted = assert_submit_matches_manual(
        &mut mid,
        0,
        &DecisionChoice::PartitionPiles {
            face_up: vec![ids[2]],
        },
    );
    assert_eq!(
        rest_submitted.outcome,
        Some(DecisionOutcome::PartitionPiles {
            face_up: vec![ids[0], ids[2]],
            face_down: vec![ids[1], ids[3], ids[4]],
        })
    );
    assert_eq!(
        mid.state.deterministic_hash(),
        s.game().state.deterministic_hash(),
        "mid-group raw steps + submit must match full submit"
    );
}
