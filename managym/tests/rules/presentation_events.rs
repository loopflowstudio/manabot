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
        experience::{PresentationKind, PresentationRule},
        flow::presentation::PresentationFact,
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
        assert_eq!(death.rule, Some(PresentationRule::LethalDamage));
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
        experience::{LookAction, PresentationKind, PresentationRule},
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
        .position(|a| {
            matches!(
                a,
                Action::ChooseTarget {
                    target: Target::Player(PlayerId(1)),
                    ..
                }
            )
        })
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
    assert_eq!(lost_fact.rule, Some(PresentationRule::ZeroLife));
    // Damage to a player tells its own life loss, never a second line.
    assert!(!burn_facts
        .iter()
        .flat_map(|f| &f.lines)
        .any(|l| matches!(l.kind, PresentationKind::Life { .. })));
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
    assert_eq!(empty_lost.rule, Some(PresentationRule::EmptyLibrary));
}

/// A drain trigger names its target and tells both life changes.
#[test]
fn a_trigger_names_its_target_and_life_changes_are_told_once() {
    use managym::{
        agent::action::Action,
        experience::{PresentationKind, SubjectRef},
        state::{game_object::PlayerId, target::Target},
    };

    let mut scenario = five_color_setup(&["Vengeful Bloodwitch", "Lightning Bolt"]);
    let witch = scenario.force_permanent_on_battlefield(0, "Vengeful Bloodwitch");
    scenario.force_card_in_hand(0, "Lightning Bolt");
    scenario.game_mut().scenario_refresh_priority().unwrap();
    scenario.game_mut().enable_presentation();

    let cast_bolt_at = |scenario: &mut Scenario, target: Target| {
        let cast = scenario
            .action_space()
            .actions
            .iter()
            .position(|a| matches!(a, Action::CastSpell { object, .. } if scenario.game().state[*object].name == "Lightning Bolt"))
            .expect("Lightning Bolt offered");
        scenario.step_action(cast);
        let choose = scenario
            .action_space()
            .actions
            .iter()
            .position(|a| matches!(a, Action::ChooseTarget { target: t, .. } if *t == target))
            .expect("target offered");
        scenario.step_action(choose);
    };
    let life_lines = |scenario: &Scenario| -> Vec<(u8, i32, i32)> {
        scenario
            .game()
            .presentation_facts(PlayerId(0))
            .iter()
            .flat_map(|fact| &fact.lines)
            .filter_map(|line| match line.kind {
                PresentationKind::Life {
                    player,
                    amount,
                    total,
                } => Some((player.0, amount, total)),
                _ => None,
            })
            .collect()
    };

    // Bolt our own Bloodwitch: it dies, its trigger targets the opponent,
    // who loses 1 while we gain 1.
    cast_bolt_at(&mut scenario, Target::Permanent(witch));
    for _ in 0..16 {
        if life_lines(&scenario).len() == 2 {
            break;
        }
        scenario.advance_default_action();
    }
    assert_eq!(life_lines(&scenario), vec![(1, -1, 19), (0, 1, 21)]);

    let lines: Vec<_> = scenario
        .game()
        .presentation_facts(PlayerId(0))
        .into_iter()
        .flat_map(|fact| fact.lines)
        .collect();
    let triggered = lines
        .iter()
        .position(|line| matches!(line.kind, PresentationKind::Triggered { .. }))
        .expect("trigger told");
    let PresentationKind::Triggered { stack, .. } = lines[triggered].kind else {
        unreachable!()
    };
    assert!(
        matches!(
            &lines[triggered + 1].kind,
            PresentationKind::Targeted {
                source: SubjectRef::Stack { id },
                target: SubjectRef::Player { id: player },
            } if *id == stack && player.0 == 1
        ),
        "the trigger's target follows it: {lines:#?}"
    );
}

/// Slice 7: exile, destroy, token creation, P/T pump, control change + cleanup
/// return, and token death labels from real games.
#[test]
fn spell_and_ability_results_exile_destroy_tokens_pump_and_control_change() {
    use managym::{
        agent::action::Action,
        experience::{PresentationKind, PresentationRule},
        state::{game_object::PlayerId, target::Target},
    };

    let cast_named = |scenario: &mut Scenario, name: &str| {
        let index = scenario
            .action_space()
            .actions
            .iter()
            .position(|a| {
                matches!(a, Action::CastSpell { object, .. } if scenario.game().state[*object].name == name)
            })
            .unwrap_or_else(|| panic!("CastSpell {name} not offered"));
        scenario.step_action(index);
    };

    // 1. Airbending Lesson exiles Gray Ogre (and draws); Treetop Freedom Fighters
    //    creates an Ally token; Wandering Musicians pumps creatures on attack;
    //    Lightning Bolt kills the Ally token with its label intact.
    let mut scenario = five_color_setup(&[
        "Airbending Lesson",
        "Treetop Freedom Fighters",
        "Wandering Musicians",
        "Gray Ogre",
        "Lightning Bolt",
    ]);
    let ogre = scenario.force_permanent_on_battlefield(1, "Gray Ogre");
    let musicians = scenario.force_permanent_on_battlefield(0, "Wandering Musicians");
    scenario.game_mut().state.permanents[musicians]
        .as_mut()
        .unwrap()
        .summoning_sick = false;
    scenario.force_card_in_hand(0, "Airbending Lesson");
    scenario.force_card_in_hand(0, "Treetop Freedom Fighters");
    scenario.force_card_in_hand(0, "Lightning Bolt");
    scenario.game_mut().scenario_refresh_priority().unwrap();
    scenario.game_mut().enable_presentation();

    cast_named(&mut scenario, "Airbending Lesson");
    let target_ogre = scenario
        .action_space()
        .actions
        .iter()
        .position(|a| matches!(a, Action::ChooseTarget { target: Target::Permanent(p), .. } if *p == ogre))
        .expect("target Gray Ogre");
    scenario.step_action(target_ogre);
    for _ in 0..16 {
        let told = scenario.game().presentation_facts(PlayerId(0));
        if told
            .iter()
            .flat_map(|f| &f.lines)
            .any(|l| matches!(l.kind, PresentationKind::Exiled { .. }))
        {
            break;
        }
        scenario.advance_default_action();
    }

    cast_named(&mut scenario, "Treetop Freedom Fighters");
    for _ in 0..20 {
        let told = scenario.game().presentation_facts(PlayerId(0));
        if told
            .iter()
            .flat_map(|f| &f.lines)
            .any(|l| matches!(l.kind, PresentationKind::TokenCreated { .. }))
        {
            break;
        }
        scenario.advance_default_action();
    }

    // Bolt the Ally token so we verify token death carries its label.
    let ally = scenario.battlefield_permanents_named(0, "Ally")[0];
    cast_named(&mut scenario, "Lightning Bolt");
    let target_ally = scenario
        .action_space()
        .actions
        .iter()
        .position(|a| matches!(a, Action::ChooseTarget { target: Target::Permanent(p), .. } if *p == ally))
        .expect("target Ally");
    scenario.step_action(target_ally);
    for _ in 0..16 {
        let told = scenario.game().presentation_facts(PlayerId(0));
        if told
            .iter()
            .flat_map(|f| &f.lines)
            .any(|l| matches!(l.kind, PresentationKind::Died { .. }))
        {
            break;
        }
        scenario.advance_default_action();
    }

    // Attack with Wandering Musicians -> trigger pumps controlled creatures +1/+0.
    scenario.advance_to_active_step(0, StepKind::DeclareAttackers);
    for _ in 0..4 {
        let actions = scenario.action_space().actions.clone();
        if let Some(attack_musicians) = actions
            .iter()
            .position(|a| matches!(a, Action::DeclareAttacker { permanent, attack: true, .. } if *permanent == musicians))
        {
            scenario.step_action(attack_musicians);
            break;
        }
        if let Some(decline) = actions
            .iter()
            .position(|a| matches!(a, Action::DeclareAttacker { attack: false, .. }))
        {
            scenario.step_action(decline);
        } else {
            break;
        }
    }
    for _ in 0..24 {
        let told = scenario.game().presentation_facts(PlayerId(0));
        if told
            .iter()
            .flat_map(|f| &f.lines)
            .any(|l| matches!(l.kind, PresentationKind::PtChanged { .. }))
        {
            break;
        }
        scenario.advance_default_action();
    }

    let facts = scenario.game().presentation_facts(PlayerId(0));
    let lines: Vec<_> = facts.iter().flat_map(|f| &f.lines).collect();
    assert!(
        lines.iter().any(|l| matches!(
            &l.kind,
            PresentationKind::Exiled { objects }
                if objects.len() == 1 && l.labels.objects.values().any(|n| n == "Gray Ogre")
        )),
        "expected exiled Gray Ogre in {lines:#?}"
    );
    assert!(
        lines.iter().any(|l| matches!(
            &l.kind,
            PresentationKind::TokenCreated { player, objects }
                if player.0 == 0 && objects.len() == 1 && l.labels.objects.values().any(|n| n == "Ally")
        )),
        "expected Ally token_created in {lines:#?}"
    );
    assert!(
        lines.iter().any(|l| matches!(
            &l.kind,
            PresentationKind::Died { objects }
                if objects.len() == 1 && l.labels.objects.values().any(|n| n == "Ally")
        )),
        "expected Ally token died with label in {lines:#?}"
    );
    assert!(
        lines.iter().any(|l| matches!(
            &l.kind,
            PresentationKind::PtChanged {
                objects,
                power: 1,
                toughness: 0,
            } if !objects.is_empty() && l.labels.objects.values().any(|n| n == "Wandering Musicians")
        )),
        "expected +1/+0 pt_changed on Wandering Musicians in {lines:#?}"
    );

    // 2. Airbender's Reversal mode 0 destroys an attacking creature (`Destroyed`).
    let mut destroy_s = five_color_setup(&["Airbender's Reversal", "Gray Ogre"]);
    let attacker = destroy_s.force_permanent_on_battlefield(0, "Gray Ogre");
    destroy_s.game_mut().state.permanents[attacker]
        .as_mut()
        .unwrap()
        .summoning_sick = false;
    destroy_s.advance_to_active_step(0, StepKind::DeclareAttackers);
    destroy_s.force_card_in_hand(0, "Airbender's Reversal");
    destroy_s.game_mut().enable_presentation();
    let attack_ogre = destroy_s
        .action_space()
        .actions
        .iter()
        .position(|a| matches!(a, Action::DeclareAttacker { permanent, attack: true, .. } if *permanent == attacker))
        .expect("attack with Gray Ogre");
    destroy_s.step_action(attack_ogre);
    destroy_s.set_player_mana_pool(0, "20WWWW");
    destroy_s.game_mut().scenario_refresh_priority().unwrap();
    cast_named(&mut destroy_s, "Airbender's Reversal");
    // Choose mode 0 (Destroy target attacking creature), then target Gray Ogre if prompted.
    let mode0 = destroy_s
        .action_space()
        .actions
        .iter()
        .position(|a| matches!(a, Action::ChooseMode { mode: 0, .. }))
        .expect("mode 0");
    destroy_s.step_action(mode0);
    if let Some(target_atk) = destroy_s
        .action_space()
        .actions
        .iter()
        .position(|a| matches!(a, Action::ChooseTarget { target: Target::Permanent(p), .. } if *p == attacker))
    {
        destroy_s.step_action(target_atk);
    }
    for _ in 0..16 {
        let told = destroy_s.game().presentation_facts(PlayerId(0));
        if told
            .iter()
            .flat_map(|f| &f.lines)
            .any(|l| matches!(l.kind, PresentationKind::Destroyed { .. }))
        {
            break;
        }
        destroy_s.advance_default_action();
    }
    let destroy_facts = destroy_s.game().presentation_facts(PlayerId(0));
    let destroyed_fact = destroy_facts
        .iter()
        .find(|f| {
            f.lines
                .iter()
                .any(|l| matches!(l.kind, PresentationKind::Destroyed { .. }))
        })
        .expect("expected destroyed fact");
    assert!(destroyed_fact.cause.is_some());
    assert!(destroyed_fact
        .lines
        .iter()
        .any(|l| l.labels.objects.values().any(|n| n == "Gray Ogre")));

    // 3. Jet's Brainwashing (kicked) steals opponent's creature, creates a Clue,
    //    and returns control at cleanup with Rule::Cleanup.
    let mut control_s = five_color_setup(&["Jet's Brainwashing", "Gray Ogre"]);
    let stolen = control_s.force_permanent_on_battlefield(1, "Gray Ogre");
    control_s.force_card_in_hand(0, "Jet's Brainwashing");
    control_s.game_mut().scenario_refresh_priority().unwrap();
    control_s.game_mut().enable_presentation();

    cast_named(&mut control_s, "Jet's Brainwashing");
    // Choose kicker (Yes), then target opponent's Gray Ogre.
    for _ in 0..24 {
        let actions = control_s.action_space().actions.clone();
        if let Some(kick) = actions
            .iter()
            .position(|a| matches!(a, Action::PayCost { .. }))
        {
            control_s.step_action(kick);
            continue;
        }
        if let Some(target) = actions
            .iter()
            .position(|a| matches!(a, Action::ChooseTarget { target: Target::Permanent(p), .. } if *p == stolen))
        {
            control_s.step_action(target);
            continue;
        }
        let told = control_s.game().presentation_facts(PlayerId(0));
        if told.iter().flat_map(|f| &f.lines).any(|l| {
            matches!(
                l.kind,
                PresentationKind::ControlChanged {
                    returned: false,
                    ..
                }
            )
        }) && told.iter().flat_map(|f| &f.lines).any(|l| {
            matches!(
                l.kind,
                PresentationKind::TokenCreated { .. }
            )
        }) {
            break;
        }
        control_s.advance_default_action();
    }

    // Advance into opponent's turn so cleanup returns control of Gray Ogre.
    control_s.advance_to_active_step(1, StepKind::Main);
    let control_facts = control_s.game().presentation_facts(PlayerId(0));
    let gain = control_facts
        .iter()
        .find(|f| {
            f.lines.iter().any(|l| {
                matches!(
                    l.kind,
                    PresentationKind::ControlChanged {
                        controller,
                        returned: false,
                        ..
                    } if controller.0 == 0
                )
            })
        })
        .expect("expected control gain fact");
    assert!(gain.cause.is_some());
    assert!(gain.rule.is_none());
    assert!(gain
        .lines
        .iter()
        .any(|l| l.labels.objects.values().any(|n| n == "Gray Ogre")));

    let returned = control_facts
        .iter()
        .find(|f| {
            f.lines.iter().any(|l| {
                matches!(
                    l.kind,
                    PresentationKind::ControlChanged {
                        controller,
                        returned: true,
                        ..
                    } if controller.0 == 1
                )
            })
        })
        .expect("expected control return fact");
    assert_eq!(returned.rule, Some(PresentationRule::Cleanup));
    assert!(returned.position.is_some());
    assert!(returned
        .lines
        .iter()
        .any(|l| l.labels.objects.values().any(|n| n == "Gray Ogre")));
    assert!(
        control_facts
            .iter()
            .flat_map(|f| &f.lines)
            .any(|l| matches!(&l.kind, PresentationKind::TokenCreated { player, .. } if player.0 == 0 && l.labels.objects.values().any(|n| n == "Clue"))),
        "expected Clue token_created in {control_facts:#?}"
    );
}

/// Slice 8: CR 608.2b illegal-target fizzles (for both spells and targeted
/// abilities) carry `rule: Some(IllegalTargets)` and `cause: None`, while real
/// counterspells carry `rule: None` and `cause: Some(counterspell_resolution)`.
#[test]
fn illegal_target_fizzles_for_spells_and_abilities_are_separate_from_counterspells() {
    use managym::{
        agent::action::Action,
        experience::{PresentationKind, PresentationRule},
        state::{game_object::PlayerId, target::Target},
    };

    let cast_named = |scenario: &mut Scenario, name: &str| {
        let index = scenario
            .action_space()
            .actions
            .iter()
            .position(|a| {
                matches!(a, Action::CastSpell { object, .. } if scenario.game().state[*object].name == name)
            })
            .unwrap_or_else(|| panic!("CastSpell {name} not offered"));
        scenario.step_action(index);
    };
    let choose_target = |scenario: &mut Scenario, target: Target| {
        let index = scenario
            .action_space()
            .actions
            .iter()
            .position(|a| matches!(a, Action::ChooseTarget { target: t, .. } if *t == target))
            .unwrap_or_else(|| panic!("ChooseTarget {target:?} not offered"));
        scenario.step_action(index);
    };
    let pay_if_prompted = |scenario: &mut Scenario| {
        for _ in 0..4 {
            if let Some(pay) = scenario
                .action_space()
                .actions
                .iter()
                .position(|a| matches!(a, Action::PayCost { .. }))
            {
                scenario.step_action(pay);
            } else {
                break;
            }
        }
    };

    // 1. Double-Bolt on the same Gray Ogre: top Bolt kills it, bottom Bolt fizzles.
    let mut fizzle_s = five_color_setup(&["Lightning Bolt", "Gray Ogre"]);
    let ogre = fizzle_s.force_permanent_on_battlefield(1, "Gray Ogre");
    force_new_card_in_hand(&mut fizzle_s, 0, "Lightning Bolt");
    force_new_card_in_hand(&mut fizzle_s, 0, "Lightning Bolt");
    fizzle_s.game_mut().scenario_refresh_priority().unwrap();
    fizzle_s.game_mut().enable_presentation();

    cast_named(&mut fizzle_s, "Lightning Bolt");
    choose_target(&mut fizzle_s, Target::Permanent(ogre));
    pay_if_prompted(&mut fizzle_s);
    cast_named(&mut fizzle_s, "Lightning Bolt");
    choose_target(&mut fizzle_s, Target::Permanent(ogre));
    pay_if_prompted(&mut fizzle_s);
    for _ in 0..16 {
        let told = fizzle_s.game().presentation_facts(PlayerId(0));
        if told
            .iter()
            .flat_map(|f| &f.lines)
            .any(|l| matches!(l.kind, PresentationKind::Countered { .. }))
        {
            break;
        }
        fizzle_s.advance_default_action();
    }
    let fizzle_facts = fizzle_s.game().presentation_facts(PlayerId(0));
    let spell_fizzle = fizzle_facts
        .iter()
        .find(|f| {
            f.lines
                .iter()
                .any(|l| matches!(l.kind, PresentationKind::Countered { .. }))
        })
        .expect("expected spell fizzle fact");
    assert_eq!(spell_fizzle.rule, Some(PresentationRule::IllegalTargets));
    assert_eq!(spell_fizzle.cause, None);
    assert!(spell_fizzle.position.is_some());
    assert!(spell_fizzle
        .lines
        .iter()
        .any(|l| l.labels.stacks.values().any(|n| n == "Lightning Bolt")));

    // 2. Real Counterspell counters Lightning Bolt with `rule: None` and `cause: Some(resolved)`.
    let mut counter_s = five_color_setup(&["Lightning Bolt", "Counterspell", "Gray Ogre"]);
    let ogre2 = counter_s.force_permanent_on_battlefield(1, "Gray Ogre");
    counter_s.force_card_in_hand(0, "Lightning Bolt");
    counter_s.force_card_in_hand(1, "Counterspell");
    counter_s.set_player_mana_pool(1, "10UU");
    counter_s.game_mut().scenario_refresh_priority().unwrap();
    counter_s.game_mut().enable_presentation();

    cast_named(&mut counter_s, "Lightning Bolt");
    choose_target(&mut counter_s, Target::Permanent(ogre2));
    pay_if_prompted(&mut counter_s);
    // Player 0 passes priority so Player 1 can cast Counterspell.
    assert!(counter_s.take_action_by_type(managym::agent::action::ActionType::PriorityPassPriority));
    cast_named(&mut counter_s, "Counterspell");
    pay_if_prompted(&mut counter_s);
    for _ in 0..16 {
        let told = counter_s.game().presentation_facts(PlayerId(0));
        if told
            .iter()
            .flat_map(|f| &f.lines)
            .any(|l| matches!(l.kind, PresentationKind::Countered { .. }))
        {
            break;
        }
        counter_s.advance_default_action();
    }
    let counter_facts = counter_s.game().presentation_facts(PlayerId(0));
    let counter_resolved = counter_facts
        .iter()
        .find(|f| {
            f.lines.iter().any(|l| {
                matches!(l.kind, PresentationKind::Resolved { .. })
                    && l.labels.stacks.values().any(|n| n == "Counterspell")
            })
        })
        .expect("expected Counterspell resolved fact");
    let countered = counter_facts
        .iter()
        .find(|f| {
            f.lines
                .iter()
                .any(|l| matches!(l.kind, PresentationKind::Countered { .. }))
        })
        .expect("expected countered spell fact");
    assert_eq!(countered.rule, None);
    assert_eq!(countered.cause, Some(counter_resolved.id));
    assert!(countered
        .lines
        .iter()
        .any(|l| l.labels.stacks.values().any(|n| n == "Lightning Bolt")));

    // 3. Targeted triggered ability fizzles when its target leaves before resolution.
    let mut ability_s =
        five_color_setup(&["Earth Kingdom Jailer", "Lightning Bolt", "Gray Ogre"]);
    let ogre3 = ability_s.force_permanent_on_battlefield(1, "Gray Ogre");
    ability_s.force_card_in_hand(0, "Earth Kingdom Jailer");
    ability_s.force_card_in_hand(0, "Lightning Bolt");
    ability_s.game_mut().scenario_refresh_priority().unwrap();
    ability_s.game_mut().enable_presentation();

    cast_named(&mut ability_s, "Earth Kingdom Jailer");
    // Resolve Earth Kingdom Jailer and target Gray Ogre with its ETB trigger.
    for _ in 0..16 {
        let actions = ability_s.action_space().actions.clone();
        if let Some(target) = actions
            .iter()
            .position(|a| matches!(a, Action::ChooseTarget { target: Target::Permanent(p), .. } if *p == ogre3))
        {
            ability_s.step_action(target);
            break;
        }
        ability_s.advance_default_action();
    }
    // In response to the Jailer's ETB trigger, Bolt Gray Ogre so the trigger fizzles.
    cast_named(&mut ability_s, "Lightning Bolt");
    choose_target(&mut ability_s, Target::Permanent(ogre3));
    pay_if_prompted(&mut ability_s);
    for _ in 0..16 {
        let told = ability_s.game().presentation_facts(PlayerId(0));
        if told
            .iter()
            .flat_map(|f| &f.lines)
            .any(|l| matches!(l.kind, PresentationKind::Countered { .. }))
        {
            break;
        }
        ability_s.advance_default_action();
    }
    let ability_facts = ability_s.game().presentation_facts(PlayerId(0));
    let trigger_stack = ability_facts
        .iter()
        .flat_map(|f| &f.lines)
        .find_map(|l| match l.kind {
            PresentationKind::Triggered { stack, .. } => Some(stack),
            _ => None,
        })
        .expect("expected Earth Kingdom Jailer trigger");
    let ability_fizzle = ability_facts
        .iter()
        .find(|f| {
            f.lines.iter().any(|l| {
                matches!(l.kind, PresentationKind::Countered { stack } if stack == trigger_stack)
            })
        })
        .expect("expected fizzled ability fact");
    assert_eq!(ability_fizzle.rule, Some(PresentationRule::IllegalTargets));
    assert_eq!(ability_fizzle.cause, None);
    assert!(ability_fizzle.position.is_some());
    assert_eq!(
        ability_fizzle.lines[0]
            .labels
            .stacks
            .get(&trigger_stack.0.to_string())
            .map(String::as_str),
        Some("Earth Kingdom Jailer")
    );
}
