use managym::flow::{
    event::{EventSubject, GameEvent},
    turn::StepKind,
};

use crate::support::game::*;

#[test]
fn combat_death_and_next_turn_are_one_ordered_committed_tape() {
    let mut scenario = Scenario::new(
        deck(&[("Forest", 24), ("Llanowar Elves", 36)]),
        deck(&[("Forest", 24), ("Llanowar Elves", 36)]),
        203,
    );
    scenario.play_land_and_cast_creature(0, "Forest", "Llanowar Elves");
    scenario.advance_to_active_step(1, StepKind::Main);
    scenario.play_land_and_cast_creature(1, "Forest", "Llanowar Elves");
    scenario.advance_to_active_step(0, StepKind::DeclareAttackers);

    let attacker = scenario.battlefield_permanents_named(0, "Llanowar Elves")[0];
    let blocker = scenario.battlefield_permanents_named(1, "Llanowar Elves")[0];
    let attacker_render_id = scenario.game().state.permanents[attacker]
        .as_ref()
        .expect("attacker")
        .id;
    let blocker_render_id = scenario.game().state.permanents[blocker]
        .as_ref()
        .expect("blocker")
        .id;

    // The tape starts at the curated combat decision, not at game setup.
    scenario.game_mut().take_observation_events();
    scenario.declare_attack();
    scenario.advance_to_active_step(0, StepKind::DeclareBlockers);
    scenario.declare_block();
    scenario.advance_to_active_step(0, StepKind::CombatDamage);
    scenario.advance_to_active_step(1, StepKind::Main);

    let events = scenario.game_mut().take_observation_events();
    let semantic = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                GameEvent::CombatAttackersDeclared { .. }
                    | GameEvent::BlockersDeclared { .. }
                    | GameEvent::CombatDamageDealt { .. }
                    | GameEvent::PermanentsDied { .. }
                    | GameEvent::TurnStarted { .. }
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(semantic.len(), 6, "unexpected semantic tape: {semantic:#?}");

    assert!(matches!(
        semantic[0],
        GameEvent::CombatAttackersDeclared {
            player,
            defender,
            attackers,
        }
            if player.0 == 0
                && defender.0 == 1
                && attackers.len() == 1
                && attackers[0].entity == attacker_render_id
    ));
    assert!(matches!(
        semantic[1],
        GameEvent::BlockersDeclared { assignments }
            if assignments.len() == 1
                && assignments[0].0.entity == attacker_render_id
                && assignments[0].1.len() == 1
                && assignments[0].1[0].entity == blocker_render_id
    ));
    assert!(matches!(
        semantic[2],
        GameEvent::CombatDamageDealt {
            source,
            target: EventSubject::Object(target),
            amount: 1,
        } if source.entity == blocker_render_id && target.entity == attacker_render_id
    ));
    assert!(matches!(
        semantic[3],
        GameEvent::CombatDamageDealt {
            source,
            target: EventSubject::Object(target),
            amount: 1,
        } if source.entity == attacker_render_id && target.entity == blocker_render_id
    ));
    assert!(matches!(
        semantic[4],
        GameEvent::PermanentsDied { objects }
            if objects.len() == 2
                && objects[0].entity == attacker_render_id
                && objects[1].entity == blocker_render_id
    ));
    assert!(matches!(
        semantic[5],
        GameEvent::TurnStarted {
            player,
            turn_number: 4,
        } if player.0 == 1
    ));
}

/// Academy's presentation sink (`flow/presentation.rs`): a creature killed in
/// combat dies to lethal damage, because of the damage dealt to it, and each
/// of those lines has the board as it was just after it.
#[test]
fn a_combat_death_follows_from_the_damage_and_each_line_keeps_its_board() {
    use managym::{
        experience::PresentationKind,
        flow::presentation::{PresentationFact, Rule},
        state::game_object::PlayerId,
    };

    let mut scenario = Scenario::new(
        deck(&[("Forest", 24), ("Llanowar Elves", 36)]),
        deck(&[("Forest", 24), ("Llanowar Elves", 36)]),
        203,
    );
    scenario.play_land_and_cast_creature(0, "Forest", "Llanowar Elves");
    scenario.advance_to_active_step(1, StepKind::Main);
    scenario.play_land_and_cast_creature(1, "Forest", "Llanowar Elves");
    scenario.advance_to_active_step(0, StepKind::DeclareAttackers);
    scenario.game_mut().enable_presentation();
    scenario.declare_attack();
    scenario.advance_to_active_step(0, StepKind::DeclareBlockers);
    scenario.declare_block();
    scenario.advance_to_active_step(0, StepKind::CombatDamage);
    scenario.advance_to_active_step(1, StepKind::Main);

    let facts = scenario.game().presentation_facts(PlayerId(0));
    let positions = scenario.game_mut().take_presentation_positions();
    let deaths: Vec<&PresentationFact> = facts
        .iter()
        .filter(|fact| {
            fact.lines
                .iter()
                .any(|line| matches!(line.kind, PresentationKind::Died { .. }))
        })
        .collect();
    assert!(!deaths.is_empty(), "no death told: {facts:#?}");
    assert_eq!(
        facts
            .iter()
            .flat_map(|fact| &fact.lines)
            .filter(|line| matches!(line.kind, PresentationKind::Damage { .. }))
            .count(),
        2
    );
    for death in deaths {
        assert_eq!(death.rule, Some(Rule::LethalDamage));
        let cause = facts
            .iter()
            .find(|fact| Some(fact.id) == death.cause)
            .expect("the death names a told cause");
        // Combat damage is told once, as combat damage.
        assert!(
            cause.lines.iter().any(|line| line.combat),
            "cause: {cause:#?}"
        );
        // The damage is seen before the death: the creature is still there.
        let (Some(damage_board), Some(death_board)) = (cause.position, death.position) else {
            panic!("combat lines keep their boards: {cause:#?} {death:#?}");
        };
        assert!(damage_board < death_board);
        let elves = |board: u32| {
            positions[board as usize]
                .state
                .permanents
                .iter()
                .flatten()
                .count()
        };
        assert!(elves(damage_board) > elves(death_board));
        assert!(positions[death_board as usize]
            .current_action_space
            .is_none());
    }
}
