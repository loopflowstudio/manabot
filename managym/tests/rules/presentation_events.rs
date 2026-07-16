use managym::flow::{
    event::{EventSubject, GameEvent},
    turn::StepKind,
};

use super::helpers::*;

#[test]
fn combat_death_and_next_turn_are_one_ordered_committed_tape() {
    let mut scenario = Scenario::new(forest_elves_deck(), forest_elves_deck(), 203);
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
