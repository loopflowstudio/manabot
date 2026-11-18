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
        attackers: vec![render(game, attackers[0]), render(game, attackers[2])],
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
    declared.sort_by_key(|id| id.entity);
    let mut planned = plan.attackers.clone();
    planned.sort_by_key(|id| id.entity);
    assert_eq!(declared, planned);

    follow(s.game_mut(), &result);
    let combat = s.game().state.combat.as_ref().unwrap();
    let mut live: Vec<_> = combat
        .attackers
        .iter()
        .map(|id| render(s.game(), *id))
        .collect();
    live.sort_by_key(|id| id.entity);
    assert_eq!(live, planned);
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
            attackers: vec![elf],
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
        attackers: attackers.iter().map(|id| render(game, *id)).collect(),
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
