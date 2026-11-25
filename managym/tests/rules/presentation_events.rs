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

/// Slice 4 remaining kinds: `sacrificed`, `searched`, `looked` (scry and
/// surveil), and `lost` (`zero_life` and `empty_library`) from real games.
#[test]
fn remaining_kinds_searched_looked_sacrificed_and_lost_from_real_games() {
    use managym::{
        agent::action::{Action, ActionType},
        experience::{LookAction, PresentationKind},
        flow::presentation::Rule,
        state::{game_object::PlayerId, target::Target, zone::ZoneType},
    };

    let cast_named = |scenario: &mut Scenario, name: &str| {
        let index = scenario
            .action_space()
            .actions
            .iter()
            .position(|a| matches!(a, Action::CastSpell { object, .. } if scenario.game().state[*object].name == name))
            .unwrap_or_else(|| panic!("CastSpell {name} not offered"));
        scenario.step_action(index);
    };

    // 1. Evolving Wilds -> activated, sacrificed, ability_resolved, searched.
    //    Opt -> cast, resolved, looked(scry), drew.
    //    Wary Thespian -> cast, resolved, triggered, ability_resolved, looked(surveil).
    let mut scenario = five_color_setup(&["Evolving Wilds", "Opt", "Wary Thespian"]);
    scenario.force_permanent_on_battlefield(0, "Evolving Wilds");
    scenario.force_card_in_hand(0, "Opt");
    scenario.force_card_in_hand(0, "Wary Thespian");
    scenario.game_mut().scenario_refresh_priority().unwrap();
    scenario.game_mut().enable_presentation();

    assert!(scenario.take_action_by_type(ActionType::PriorityActivateAbility));
    for _ in 0..16 {
        let told = scenario.game().presentation_facts(PlayerId(0));
        if told
            .iter()
            .flat_map(|f| &f.lines)
            .any(|l| matches!(l.kind, PresentationKind::Searched { .. }))
        {
            break;
        }
        scenario.advance_default_action();
    }

    cast_named(&mut scenario, "Opt");
    for _ in 0..16 {
        let told = scenario.game().presentation_facts(PlayerId(0));
        if told.iter().flat_map(|f| &f.lines).any(|l| {
            matches!(
                l.kind,
                PresentationKind::Looked {
                    action: LookAction::Scry,
                    ..
                }
            )
        }) {
            break;
        }
        scenario.advance_default_action();
    }

    cast_named(&mut scenario, "Wary Thespian");
    for _ in 0..20 {
        let told = scenario.game().presentation_facts(PlayerId(0));
        if told.iter().flat_map(|f| &f.lines).any(|l| {
            matches!(
                l.kind,
                PresentationKind::Looked {
                    action: LookAction::Surveil,
                    ..
                }
            )
        }) {
            break;
        }
        scenario.advance_default_action();
    }

    let facts = scenario.game().presentation_facts(PlayerId(0));
    let opp_facts = scenario.game().presentation_facts(PlayerId(1));
    let lines: Vec<_> = facts.iter().flat_map(|f| &f.lines).collect();
    assert!(
        lines.iter().any(|l| matches!(
            &l.kind,
            PresentationKind::Sacrificed { player, objects }
                if player.0 == 0 && !objects.is_empty()
                    && l.labels.objects.values().any(|n| n == "Evolving Wilds")
        )),
        "expected sacrificed Evolving Wilds in {lines:#?}"
    );
    assert!(
        lines
            .iter()
            .any(|l| matches!(l.kind, PresentationKind::Searched { player } if player.0 == 0)),
        "expected searched in {lines:#?}"
    );
    assert!(
        lines.iter().any(|l| matches!(
            l.kind,
            PresentationKind::Looked {
                player,
                action: LookAction::Scry
            } if player.0 == 0
        )),
        "expected scry looked in {lines:#?}"
    );
    assert!(
        lines.iter().any(|l| matches!(
            l.kind,
            PresentationKind::Looked {
                player,
                action: LookAction::Surveil
            } if player.0 == 0
        )),
        "expected surveil looked in {lines:#?}"
    );
    // Neither search nor scry/surveil leaks hidden card names to the opponent.
    for fact in &opp_facts {
        for line in &fact.lines {
            if matches!(
                line.kind,
                PresentationKind::Searched { .. } | PresentationKind::Looked { .. }
            ) {
                assert!(line.labels.objects.is_empty());
                assert!(line.labels.stacks.is_empty());
            }
        }
    }

    // 2. Lethal burn -> Lost with Rule::ZeroLife caused by the spell's resolution.
    let mut burn = five_color_setup(&["Lightning Bolt"]);
    burn.force_card_in_hand(0, "Lightning Bolt");
    burn.game_mut().state.players[1].life = 3;
    burn.game_mut().scenario_refresh_priority().unwrap();
    burn.game_mut().enable_presentation();
    cast_named(&mut burn, "Lightning Bolt");
    let target_opp = burn
        .action_space()
        .actions
        .iter()
        .position(|a| matches!(a, Action::ChooseTarget { target: Target::Player(PlayerId(1)), .. }))
        .expect("target player 1");
    burn.step_action(target_opp);
    for _ in 0..8 {
        if burn.game().is_game_over() {
            break;
        }
        burn.advance_default_action();
    }
    assert!(burn.game().is_game_over());
    let burn_facts = burn.game().presentation_facts(PlayerId(0));
    let lost_fact = burn_facts
        .iter()
        .find(|f| {
            f.lines
                .iter()
                .any(|l| matches!(l.kind, PresentationKind::Lost { player } if player.0 == 1))
        })
        .expect("expected lost fact");
    assert_eq!(lost_fact.rule, Some(Rule::ZeroLife));
    assert!(lost_fact.cause.is_some());

    // 3. Empty library draw -> Lost with Rule::EmptyLibrary.
    let mut mill = five_color_setup(&["Opt"]);
    mill.force_card_in_hand(0, "Opt");
    let lib_cards: Vec<_> = mill
        .game()
        .state
        .latest_objects()
        .filter(|(id, obj)| {
            obj.owner == PlayerId(0)
                && mill.game().state.zones.zone_of(*id) == Some(ZoneType::Library)
        })
        .map(|(id, _)| id)
        .collect();
    for card in lib_cards {
        mill.game_mut()
            .state
            .place_object(card, PlayerId(0), ZoneType::Exile);
    }
    mill.game_mut().scenario_refresh_priority().unwrap();
    mill.game_mut().enable_presentation();
    cast_named(&mut mill, "Opt");
    for _ in 0..8 {
        if mill.game().is_game_over() {
            break;
        }
        mill.advance_default_action();
    }
    assert!(mill.game().is_game_over());
    let mill_facts = mill.game().presentation_facts(PlayerId(0));
    let empty_lost = mill_facts
        .iter()
        .find(|f| {
            f.lines
                .iter()
                .any(|l| matches!(l.kind, PresentationKind::Lost { player } if player.0 == 0))
        })
        .expect("expected empty-library lost fact");
    assert_eq!(empty_lost.rule, Some(Rule::EmptyLibrary));
}
