//! The pick shape of combat declaration (`ExperimentSwitches::combat_pick`,
//! `MANAGYM_EXPERIMENT_COMBAT_PICK=1`): one repeated decision over every
//! undeclared creature, finished by `Decline`, against the default shape's
//! one decision per creature. Both must reach the same game for the same
//! declarations.

use managym::{
    agent::action::{Action, ActionSpaceKind},
    flow::{event::GameEvent, turn::StepKind},
    state::{effect::Modification, game_object::PermanentId},
};

use crate::support::{
    game::{clear_summoning_sickness, deck, until_end_of_turn, Scenario},
    play::choose,
};

const ATTACKERS: [&str; 3] = ["Llanowar Elves", "Gray Ogre", "Hill Giant"];
const BLOCKERS: [&str; 2] = ["Gray Ogre", "Hill Giant"];

/// Player 0 at declare attackers with `attackers` ready; player 1 holds
/// `blockers` untapped.
fn combat(
    pick: bool,
    attackers: &[&str],
    blockers: &[&str],
) -> (Scenario, Vec<PermanentId>, Vec<PermanentId>) {
    let list = deck(&[
        ("Forest", 24),
        ("Llanowar Elves", 12),
        ("Gray Ogre", 12),
        ("Hill Giant", 8),
        ("Wind Drake", 4),
    ]);
    let mut s = Scenario::new(list.clone(), list, 265);
    s.game_mut().experiments.combat_pick = pick;
    let attackers: Vec<_> = attackers
        .iter()
        .map(|name| s.force_permanent_on_battlefield(0, name))
        .collect();
    let blockers: Vec<_> = blockers
        .iter()
        .map(|name| s.force_permanent_on_battlefield(1, name))
        .collect();
    for id in attackers.iter().chain(&blockers) {
        clear_summoning_sickness(&mut s, *id);
    }
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    assert_eq!(s.action_space().kind, ActionSpaceKind::DeclareAttacker);
    (s, attackers, blockers)
}

fn attack(permanent: PermanentId) -> impl Fn(&Action) -> bool {
    move |action| matches!(action, Action::DeclareAttacker { permanent: p, attack: true, .. } if *p == permanent)
}

fn block(blocker: PermanentId, attacker: PermanentId) -> impl Fn(&Action) -> bool {
    move |action| {
        matches!(action, Action::DeclareBlocker { blocker: b, attacker: Some(a), .. }
            if *b == blocker && *a == attacker)
    }
}

fn done(action: &Action) -> bool {
    matches!(action, Action::Decline { .. })
}

/// Answer the default shape's prompts: attack with exactly `chosen`.
fn declare_attackers_by_prompt(s: &mut Scenario, chosen: &[PermanentId]) -> Vec<PermanentId> {
    let mut asked = Vec::new();
    while s.action_space().kind == ActionSpaceKind::DeclareAttacker {
        let Action::DeclareAttacker { permanent, .. } = s.action_space().actions[0] else {
            panic!("default attacker prompt: {:?}", s.action_space());
        };
        asked.push(permanent);
        let attacks = chosen.contains(&permanent);
        choose(
            s,
            |action| matches!(action, Action::DeclareAttacker { attack, .. } if *attack == attacks),
        );
    }
    asked
}

/// Answer the default shape's blocker prompts from `blocks` (blocker, attacker).
fn declare_blockers_by_prompt(s: &mut Scenario, blocks: &[(PermanentId, PermanentId)]) {
    while s.action_space().kind == ActionSpaceKind::DeclareBlocker {
        let Action::DeclareBlocker { blocker, .. } = s.action_space().actions[0] else {
            panic!("default blocker prompt: {:?}", s.action_space());
        };
        let wanted = blocks.iter().find(|(b, _)| *b == blocker).map(|(_, a)| *a);
        choose(
            s,
            |action| matches!(action, Action::DeclareBlocker { attacker, .. } if *attacker == wanted),
        );
    }
}

fn offered_attackers(s: &Scenario) -> Vec<PermanentId> {
    s.action_space()
        .actions
        .iter()
        .filter_map(|action| match action {
            Action::DeclareAttacker {
                permanent,
                attack: true,
                ..
            } => Some(*permanent),
            _ => None,
        })
        .collect()
}

fn offered_blocks(s: &Scenario) -> Vec<(PermanentId, PermanentId)> {
    s.action_space()
        .actions
        .iter()
        .filter_map(|action| match action {
            Action::DeclareBlocker {
                blocker,
                attacker: Some(attacker),
                ..
            } => Some((*blocker, *attacker)),
            _ => None,
        })
        .collect()
}

/// Reach the blocker declaration (or whatever follows it) with `attacking`.
fn to_blockers(s: &mut Scenario, attacking: &[PermanentId]) {
    if s.game().experiments.combat_pick {
        // Pick in the offered order, which is the order the default shape
        // declares in, so both shapes record the same attacker order.
        for attacker in offered_attackers(s) {
            if attacking.contains(&attacker) {
                choose(s, attack(attacker));
            }
        }
        if s.action_space().kind == ActionSpaceKind::DeclareAttacker {
            choose(s, done);
        }
    } else {
        declare_attackers_by_prompt(s, attacking);
    }
    s.advance_until(
        |s| s.current_step() == StepKind::DeclareBlockers,
        "declare blockers".into(),
    );
}

#[test]
fn switch_off_keeps_one_creature_prompts() {
    let (mut s, attackers, blockers) = combat(false, &ATTACKERS, &BLOCKERS);
    let space = s.action_space().clone();
    assert!(
        matches!(
            space.actions.as_slice(),
            [
                Action::DeclareAttacker { permanent: a, attack: true, .. },
                Action::DeclareAttacker { permanent: b, attack: false, .. },
            ] if a == b
        ),
        "{space:?}"
    );
    to_blockers(&mut s, &attackers[..1]);
    let space = s.action_space().clone();
    assert_eq!(space.kind, ActionSpaceKind::DeclareBlocker);
    assert!(
        matches!(
            space.actions.as_slice(),
            [
                Action::DeclareBlocker { blocker: a, attacker: Some(_), .. },
                Action::DeclareBlocker { blocker: b, attacker: None, .. },
            ] if a == b && blockers.contains(a)
        ),
        "{space:?}"
    );
}

#[test]
fn picked_attackers_then_done_match_the_default_shape() {
    let (mut by_prompt, attackers, _) = combat(false, &ATTACKERS, &[]);
    let (mut by_pick, same, _) = combat(true, &ATTACKERS, &[]);
    assert_eq!(attackers, same);
    by_prompt.drain_events();
    by_pick.drain_events();

    // The default shape asks about each creature in engine order; attack
    // with all but the last it asks about (yes, yes, no).
    let chosen: Vec<_> = {
        let mut probe = combat(false, &ATTACKERS, &[]).0;
        let asked = declare_attackers_by_prompt(&mut probe, &[]);
        assert_eq!(asked.len(), 3);
        asked[..2].to_vec()
    };
    let asked = declare_attackers_by_prompt(&mut by_prompt, &chosen);

    // Pick: "done" first, then every queued creature in that same order.
    let space = by_pick.action_space().clone();
    assert!(done(&space.actions[0]), "{space:?}");
    assert_eq!(offered_attackers(&by_pick), asked);
    assert_eq!(space.actions.len(), 4);
    choose(&mut by_pick, attack(chosen[0]));
    assert_eq!(offered_attackers(&by_pick), asked[1..]);
    choose(&mut by_pick, attack(chosen[1]));
    assert_eq!(offered_attackers(&by_pick), asked[2..]);
    choose(&mut by_pick, done);

    assert_ne!(
        by_pick.action_space().kind,
        ActionSpaceKind::DeclareAttacker
    );
    let combat_state = by_pick.game().state.combat.as_ref().unwrap();
    assert_eq!(combat_state.attackers, chosen);
    assert!(combat_state.attackers_to_declare.is_empty());
    let events = by_pick.drain_events();
    assert!(events
        .iter()
        .any(|event| matches!(event, GameEvent::CombatAttackersDeclared { attackers, .. } if attackers.len() == 2)));
    assert_eq!(events, by_prompt.drain_events());
    assert_eq!(
        by_pick.game().state.deterministic_hash(),
        by_prompt.game().state.deterministic_hash()
    );
}

#[test]
fn picking_every_attacker_ends_declaration_without_done() {
    let (mut s, attackers, _) = combat(true, &ATTACKERS, &[]);
    for attacker in &attackers {
        assert_eq!(s.action_space().kind, ActionSpaceKind::DeclareAttacker);
        choose(&mut s, attack(*attacker));
    }
    assert_ne!(s.action_space().kind, ActionSpaceKind::DeclareAttacker);
    assert_eq!(s.game().state.combat.as_ref().unwrap().attackers, attackers);
}

#[test]
fn picked_blocks_then_done_match_the_default_shape() {
    let (mut by_prompt, attackers, blockers) = combat(false, &ATTACKERS, &BLOCKERS);
    let (mut by_pick, _, _) = combat(true, &ATTACKERS, &BLOCKERS);
    let attacking = &attackers[1..];
    to_blockers(&mut by_prompt, attacking);
    to_blockers(&mut by_pick, attacking);
    by_prompt.drain_events();
    by_pick.drain_events();

    // One block, the other creature does not block.
    let blocks = [(blockers[1], attacking[0])];
    declare_blockers_by_prompt(&mut by_prompt, &blocks);

    let space = by_pick.action_space().clone();
    assert_eq!(space.kind, ActionSpaceKind::DeclareBlocker);
    assert!(done(&space.actions[0]), "{space:?}");
    // Every pair across both queued blockers, grouped by blocker.
    let pairs = offered_blocks(&by_pick);
    assert_eq!(pairs.len(), 4, "{space:?}");
    assert_eq!(pairs[0].0, pairs[1].0);
    assert_eq!(pairs[2].0, pairs[3].0);
    choose(&mut by_pick, block(blocks[0].0, blocks[0].1));
    // The declared blocker left the queue; the other still has both pairs.
    let pairs = offered_blocks(&by_pick);
    assert_eq!(pairs.len(), 2);
    assert!(pairs.iter().all(|(blocker, _)| *blocker == blockers[0]));
    choose(&mut by_pick, done);

    assert_ne!(by_pick.action_space().kind, ActionSpaceKind::DeclareBlocker);
    let combat_state = by_pick.game().state.combat.as_ref().unwrap();
    assert_eq!(
        combat_state.attacker_to_blockers[&attacking[0]],
        [blockers[1]]
    );
    assert!(combat_state.attacker_to_blockers[&attacking[1]].is_empty());
    let events = by_pick.drain_events();
    assert!(events
        .iter()
        .any(|event| matches!(event, GameEvent::BlockersDeclared { .. })));
    assert_eq!(events, by_prompt.drain_events());
    assert_eq!(
        by_pick.game().state.deterministic_hash(),
        by_prompt.game().state.deterministic_hash()
    );
}

/// CR 509.1c — while a queued creature can block an attacker every able
/// creature must block, only those blocks are offered and there is no "done".
#[test]
fn required_blocks_are_offered_alone_until_none_remains() {
    for skip_trivial in [false, true] {
        let (mut s, attackers, blockers) = combat(true, &ATTACKERS, &BLOCKERS);
        let lure = attackers[1];
        until_end_of_turn(&mut s, lure, Modification::MustBeBlockedByAllAble);
        to_blockers(&mut s, &attackers[1..]);
        s.game_mut().skip_trivial = skip_trivial;

        let space = s.action_space().clone();
        assert_eq!(space.kind, ActionSpaceKind::DeclareBlocker);
        assert!(!space.actions.iter().any(done), "{space:?}");
        let pairs = offered_blocks(&s);
        assert_eq!(pairs.len(), 2);
        assert!(pairs.iter().all(|(_, attacker)| *attacker == lure));
        s.drain_events();
        choose(&mut s, block(blockers[0], lure));
        let events = s.drain_events();
        if !skip_trivial {
            // The last required block is the only action.
            assert_eq!(offered_blocks(&s), [(blockers[1], lure)]);
            assert_eq!(s.action_space().actions.len(), 1);
            choose(&mut s, block(blockers[1], lure));
        }
        // With forced decisions skipped the last required block is taken
        // without a prompt and play runs on, so read the declaration event.
        assert_ne!(s.action_space().kind, ActionSpaceKind::DeclareBlocker);
        assert!(events.iter().chain(&s.drain_events()).any(|event| {
            matches!(event, GameEvent::BlockersDeclared { assignments }
                if assignments.len() == 1 && assignments[0].1.len() == 2)
        }));
    }
}

/// A queue with no legal block ends the declaration without a prompt, even
/// when forced decisions are shown.
#[test]
fn no_legal_block_ends_declaration_without_a_prompt() {
    let (mut s, attackers, _) = combat(true, &["Wind Drake"], &BLOCKERS);
    let drake = attackers[0];
    to_blockers(&mut s, &[drake]);
    assert_ne!(s.action_space().kind, ActionSpaceKind::DeclareBlocker);
    let combat_state = s.game().state.combat.as_ref().unwrap();
    assert!(combat_state.blockers_to_declare.is_empty());
    assert!(combat_state.attacker_to_blockers[&drake].is_empty());
}
