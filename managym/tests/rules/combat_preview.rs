//! The whole-declaration preview replays the engine's own per-creature
//! answers on a clone: its steps, sent one by one, declare exactly the plan,
//! and asking it changes nothing.
use managym::{
    agent::action::ActionSpaceKind,
    experience::ObjectRenderId,
    flow::{
        combat_preview::{CombatPlan, CombatPreview, DeclarationKind, PlannedBlock},
        turn::StepKind,
    },
    search_state::witness,
    state::game_object::{PermanentId, PlayerId},
    Game,
};

use crate::support::game::*;

fn render(game: &Game, permanent: PermanentId) -> ObjectRenderId {
    let permanent = game.state.permanents[permanent]
        .as_ref()
        .expect("on the battlefield");
    ObjectRenderId {
        entity: permanent.id.0,
        incarnation: permanent.object.generation(),
    }
}

/// Player 0 attacks with Boggart Brute (menace) and two Gray Ogres into three
/// of player 1's Llanowar Elves.
fn combat() -> (Scenario, Vec<PermanentId>, Vec<PermanentId>) {
    let mut s = Scenario::new(
        deck(&[("Mountain", 30), ("Boggart Brute", 4), ("Gray Ogre", 6)]),
        deck(&[("Forest", 30), ("Llanowar Elves", 10)]),
        7,
    );
    s.advance_to_active_step(0, StepKind::Main);
    let attackers: Vec<_> = ["Boggart Brute", "Gray Ogre", "Gray Ogre"]
        .iter()
        .map(|name| s.force_permanent_on_battlefield(0, name))
        .collect();
    for attacker in &attackers {
        let permanent = s.game_mut().state.permanents[*attacker].as_mut().unwrap();
        permanent.summoning_sick = false;
        permanent.tapped = false;
    }
    let blockers: Vec<_> = (0..3)
        .map(|_| s.force_permanent_on_battlefield(1, "Llanowar Elves"))
        .collect();
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    assert_eq!(s.action_space().kind, ActionSpaceKind::DeclareAttacker);
    (s, attackers, blockers)
}

fn preview(game: &Game, viewer: usize, plan: &CombatPlan) -> CombatPreview {
    let frame = game.semantic_decision_frame().unwrap();
    game.combat_preview(PlayerId(viewer), frame.revision, &frame.fingerprint, plan)
        .unwrap()
}

/// Send each step at the revision and fingerprint it predicts.
fn follow(game: &mut Game, preview: &CombatPreview) {
    for step in &preview.steps {
        let frame = game.semantic_decision_frame().unwrap();
        assert_eq!(
            (frame.revision, &frame.fingerprint),
            (step.revision, &step.fingerprint)
        );
        game.step(step.action_index).unwrap();
    }
}

#[test]
fn an_attack_plan_is_read_only_and_its_steps_declare_exactly_it() {
    let (mut s, attackers, _) = combat();
    let game = s.game();
    let plan = CombatPlan {
        attackers: vec![
            render(game, attackers[0]).into(),
            render(game, attackers[2]).into(),
        ],
        blocks: Vec::new(),
    };
    let before = witness(game);
    let frame = game.semantic_decision_frame().unwrap();
    let result = preview(game, 0, &plan);
    assert_eq!(witness(game), before);
    let after = game.semantic_decision_frame().unwrap();
    assert_eq!(
        (after.revision, after.fingerprint),
        (frame.revision, frame.fingerprint)
    );

    assert_eq!(result.kind, DeclarationKind::Attackers);
    assert!(result.refused.is_none(), "{:?}", result.refused);
    assert_eq!(result.declarable.len(), 3);
    assert_eq!(result.steps.len(), 3, "one answer per creature asked");
    assert_eq!(result.steps.iter().filter(|step| step.attack).count(), 2);
    let mut declared = result.declared_attackers.clone();
    declared.sort_by_key(|a| a.creature.entity);
    let mut planned = plan.attackers.clone();
    planned.sort_by_key(|a| a.creature.entity);
    assert_eq!(declared, planned);

    follow(s.game_mut(), &result);
    let combat = s.game().state.combat.as_ref().unwrap();
    let mut live: Vec<_> = combat.attackers().map(|id| render(s.game(), id)).collect();
    live.sort_by_key(|id| id.entity);
    assert_eq!(live, planned.iter().map(|a| a.creature).collect::<Vec<_>>());
}

#[test]
fn a_stale_address_is_refused_and_a_creature_that_cannot_attack_is_named() {
    let (s, _, blockers) = combat();
    let game = s.game();
    let frame = game.semantic_decision_frame().unwrap();
    let plan = CombatPlan::default();
    assert!(game
        .combat_preview(PlayerId(0), frame.revision + 1, &frame.fingerprint, &plan)
        .is_err());
    assert!(game
        .combat_preview(PlayerId(1), frame.revision, &frame.fingerprint, &plan)
        .is_err());

    let elf = render(game, blockers[0]);
    let result = preview(
        game,
        0,
        &CombatPlan {
            attackers: vec![elf.into()],
            blocks: Vec::new(),
        },
    );
    let refused = result
        .refused
        .expect("an opponent's creature cannot attack");
    assert_eq!(refused.creature, elf);
    assert_eq!(refused.reason, "Llanowar Elves can't attack now.");
    assert!(result.steps.is_empty());
}

fn at_blocks() -> (Scenario, Vec<PermanentId>, Vec<PermanentId>) {
    let (mut s, attackers, blockers) = combat();
    let game = s.game();
    let all = CombatPlan {
        attackers: attackers
            .iter()
            .map(|id| render(game, *id).into())
            .collect(),
        blocks: Vec::new(),
    };
    let result = preview(game, 0, &all);
    follow(s.game_mut(), &result);
    s.advance_until(
        |s| s.action_space().kind == ActionSpaceKind::DeclareBlocker,
        "failed to reach blockers".into(),
    );
    (s, attackers, blockers)
}

#[test]
fn a_multi_block_is_declared_as_planned() {
    let (mut s, attackers, blockers) = at_blocks();
    let game = s.game();
    let block = |blocker: usize, attacker: usize| PlannedBlock {
        blocker: render(game, blockers[blocker]),
        attacker: render(game, attackers[attacker]),
    };
    // Two Elves on the Brute satisfy menace; the third blocks an Ogre.
    let plan = CombatPlan {
        attackers: Vec::new(),
        blocks: vec![block(0, 0), block(1, 0), block(2, 1)],
    };
    let before = witness(game);
    let result = preview(game, 1, &plan);
    assert_eq!(witness(game), before);
    assert_eq!(result.kind, DeclarationKind::Blockers);
    assert!(result.refused.is_none(), "{:?}", result.refused);
    assert_eq!(result.declarable.len(), 3);
    assert!(result
        .declarable
        .iter()
        .all(|creature| creature.can_block.len() == 3));
    assert_eq!(result.steps.len(), 3);
    let mut declared = result.declared_blocks.clone();
    declared.sort_by_key(|block| (block.attacker.entity, block.blocker.entity));
    let mut planned = plan.blocks.clone();
    planned.sort_by_key(|block| (block.attacker.entity, block.blocker.entity));
    assert_eq!(declared, planned);

    follow(s.game_mut(), &result);
    let combat = s.game().state.combat.as_ref().unwrap();
    assert_eq!(combat.attacker_to_blockers[&attackers[0]].len(), 2);
    assert_eq!(
        combat.attacker_to_blockers[&attackers[1]],
        vec![blockers[2]]
    );
}

#[test]
fn a_lone_block_of_a_menace_creature_is_refused_with_the_reason() {
    let (s, attackers, blockers) = at_blocks();
    let game = s.game();
    let plan = CombatPlan {
        attackers: Vec::new(),
        blocks: vec![PlannedBlock {
            blocker: render(game, blockers[0]),
            attacker: render(game, attackers[0]),
        }],
    };
    let result = preview(game, 1, &plan);
    let refused = result.refused.expect("menace removes a lone block");
    assert_eq!(refused.creature, render(game, blockers[0]));
    assert!(
        refused
            .reason
            .starts_with("Boggart Brute can't be blocked that way"),
        "{}",
        refused.reason
    );
    assert!(result.declared_blocks.is_empty());

    let twice = CombatPlan {
        attackers: Vec::new(),
        blocks: vec![
            PlannedBlock {
                blocker: render(game, blockers[0]),
                attacker: render(game, attackers[1]),
            },
            PlannedBlock {
                blocker: render(game, blockers[0]),
                attacker: render(game, attackers[2]),
            },
        ],
    };
    let refused = preview(game, 1, &twice).refused.unwrap();
    assert_eq!(
        refused.reason,
        "Llanowar Elves can block only one attacker."
    );
}

/// Juggernaut attacks each combat (CR 508.1d): the preview marks it, refuses
/// a plan that leaves it home, and a plan with it declares as planned.
#[test]
fn a_creature_that_must_attack_is_marked_and_a_plan_omitting_it_is_refused() {
    for skip_trivial in [false, true] {
        for order in [["Juggernaut", "Gray Ogre"], ["Gray Ogre", "Juggernaut"]] {
            let mut s = Scenario::new(
                deck(&[("Mountain", 30), ("Juggernaut", 4), ("Gray Ogre", 6)]),
                deck(&[("Forest", 30), ("Llanowar Elves", 10)]),
                7,
            );
            s.advance_to_active_step(0, StepKind::Main);
            let ids: Vec<_> = order
                .iter()
                .map(|name| {
                    let id = s.force_permanent_on_battlefield(0, name);
                    clear_summoning_sickness(&mut s, id);
                    id
                })
                .collect();
            s.game_mut().skip_trivial = skip_trivial;
            s.advance_to_active_step(0, StepKind::DeclareAttackers);
            let game = s.game();
            let juggernaut = render(
                game,
                ids[order.iter().position(|n| *n == "Juggernaut").unwrap()],
            );
            let ogre = render(
                game,
                ids[order.iter().position(|n| *n == "Gray Ogre").unwrap()],
            );
            let context = format!("skip_trivial {skip_trivial}, order {order:?}");

            let home = preview(
                game,
                0,
                &CombatPlan {
                    attackers: vec![ogre.into()],
                    blocks: Vec::new(),
                },
            );
            match home
                .declarable
                .iter()
                .find(|creature| creature.creature == juggernaut)
            {
                Some(creature) => {
                    assert!(creature.must_attack, "{context}");
                    let refused = home
                        .refused
                        .as_ref()
                        .expect("omitting Juggernaut is refused");
                    assert_eq!(refused.creature, juggernaut, "{context}");
                    assert_eq!(refused.reason, "Juggernaut must attack.", "{context}");
                }
                // Already declared without asking: the plan covers the rest.
                None => {
                    assert!(home.refused.is_none(), "{context}: {:?}", home.refused);
                    assert!(
                        home.declared_attackers.contains(&juggernaut.into()),
                        "{context}"
                    );
                }
            }
            assert!(home
                .declarable
                .iter()
                .all(|creature| creature.creature == juggernaut || !creature.must_attack));

            let both = CombatPlan {
                attackers: vec![juggernaut.into(), ogre.into()],
                blocks: Vec::new(),
            };
            let result = preview(game, 0, &both);
            assert!(result.refused.is_none(), "{context}: {:?}", result.refused);
            let mut declared = result.declared_attackers.clone();
            declared.sort_by_key(|a| a.creature.entity);
            let mut planned = both.attackers.clone();
            planned.sort_by_key(|a| a.creature.entity);
            assert_eq!(declared, planned, "{context}");
        }
    }
}

/// Taunting Challenge makes Giant Spider block Wind Drake (CR 509.1c). When
/// the engine has already declared that block without asking, the preview
/// accepts plans that include or omit it, refuses moving it, and still asks
/// for the Llanowar Elves' answer.
#[test]
fn a_block_the_engine_already_declared_is_kept_by_the_plan() {
    use crate::support::play::{cast, settle};
    use managym::state::target::Target;
    let mut reached_predeclared = false;
    for order in [
        ["Giant Spider", "Llanowar Elves"],
        ["Llanowar Elves", "Giant Spider"],
    ] {
        let mut s = five_color_setup(&[
            "Taunting Challenge",
            "Wind Drake",
            "Gray Ogre",
            "Giant Spider",
            "Llanowar Elves",
        ]);
        let drake = s.force_permanent_on_battlefield(0, "Wind Drake");
        let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
        for id in [drake, ogre] {
            clear_summoning_sickness(&mut s, id);
        }
        let blockers: Vec<_> = order
            .iter()
            .map(|name| s.force_permanent_on_battlefield(1, name))
            .collect();
        let spider = blockers[order.iter().position(|n| *n == "Giant Spider").unwrap()];
        let elves = blockers[order.iter().position(|n| *n == "Llanowar Elves").unwrap()];
        cast(&mut s, "Taunting Challenge");
        assert!(s.choose_target(Target::Permanent(drake)));
        settle(&mut s);
        s.game_mut().skip_trivial = true;
        s.advance_to_active_step(0, StepKind::DeclareAttackers);
        while s.action_space().kind == ActionSpaceKind::DeclareAttacker {
            let index = s
                .action_space()
                .actions
                .iter()
                .position(|action| {
                    matches!(
                        action,
                        managym::agent::action::Action::DeclareAttacker {
                            attack: Some(_),
                            ..
                        }
                    )
                })
                .unwrap();
            s.step_action(index);
        }
        s.advance_until(
            |s| s.action_space().kind == ActionSpaceKind::DeclareBlocker,
            "failed to reach blockers".into(),
        );
        let game = s.game();
        let predeclared =
            !game.state.combat.as_ref().unwrap().attacker_to_blockers[&drake].is_empty();
        reached_predeclared |= predeclared;
        let context = format!("order {order:?}, spider pre-declared {predeclared}");
        let forced = PlannedBlock {
            blocker: render(game, spider),
            attacker: render(game, drake),
        };
        let elves_on_ogre = PlannedBlock {
            blocker: render(game, elves),
            attacker: render(game, ogre),
        };

        for blocks in [vec![forced, elves_on_ogre], vec![elves_on_ogre]] {
            let plan = CombatPlan {
                attackers: Vec::new(),
                blocks: blocks.clone(),
            };
            let result = preview(game, 1, &plan);
            if predeclared || blocks.contains(&forced) {
                assert!(result.refused.is_none(), "{context}: {:?}", result.refused);
                let mut declared = result.declared_blocks.clone();
                declared.sort_by_key(|block| block.blocker.entity);
                let mut wanted = vec![forced, elves_on_ogre];
                wanted.sort_by_key(|block| block.blocker.entity);
                assert_eq!(declared, wanted, "{context}");
            } else {
                assert_eq!(
                    result.refused.expect("omitting the forced block").reason,
                    "Giant Spider must block Wind Drake.",
                    "{context}"
                );
            }
        }

        if predeclared {
            let moved = CombatPlan {
                attackers: Vec::new(),
                blocks: vec![PlannedBlock {
                    blocker: render(game, spider),
                    attacker: render(game, ogre),
                }],
            };
            assert_eq!(
                preview(game, 1, &moved).refused.unwrap().reason,
                "Giant Spider already blocks Wind Drake.",
                "{context}"
            );
        }
    }
    assert!(reached_predeclared, "one order declares the Spider first");
}
