//! Manifest Dread's private library choice and viewer-specific battlefield projection.
use crate::support::game::{five_color_setup, pin_top};
use managym::{
    agent::{
        observation::Observation,
        observation_encoder::{encode, ObservationEncoderConfig},
    },
    state::{game_object::PlayerId, player::DecklistDisclosure, zone::ZoneType},
};

fn assert_anonymous_manifest(view: &Observation) {
    assert_eq!(view.opponent_permanents.len(), 1);
    let row = &view.opponent_permanents[0];
    assert_eq!(row.registry_key, -1);
    assert_eq!(row.object_id, Some(row.id));
    assert_eq!(
        (row.characteristics.power, row.characteristics.toughness),
        (2, 2)
    );
    assert!(row.characteristics.name.is_empty());
}

#[test]
fn manifest_dread_hides_the_printing_at_opponent_priority() {
    for disclosure in [DecklistDisclosure::Open, DecklistDisclosure::Closed] {
        manifest_projections_match(disclosure);
    }
}

fn manifest_projections_match(disclosure: DecklistDisclosure) {
    let projections = [("Gray Ogre", "Forest"), ("Forest", "Gray Ogre")].map(|(hidden, other)| {
        let mut s = five_color_setup(&["Gray Ogre", "Lightning Bolt"]);
        s.game_mut().state.disclosure = disclosure;
        s.force_card_in_hand(1, "Lightning Bolt");
        let cards = pin_top(&mut s, &[hidden, "Plains", other]);
        let printing = s.game().state[cards[0]].definition_id;
        let before = Observation::for_player(s.game(), PlayerId(0));
        let permanent = s.game_mut().manifest(cards[0], PlayerId(0));
        s.game_mut().move_object(cards[1], ZoneType::Graveyard);
        s.game_mut().scenario_refresh_priority().unwrap();
        s.pass_priority();
        assert_eq!(s.action_space().player, Some(PlayerId(1)));

        let mut fork = s.snapshot().restore(Default::default()).unwrap();
        for game in [s.game(), &*fork.benchmark_game_mut()] {
            assert_eq!(game.state[permanent].retained_base().1.name, hidden);
            assert_eq!(game.state.zones.zone_of(cards[2]), Some(ZoneType::Library));
            let controller = Observation::for_player(game, PlayerId(0));
            assert_eq!(
                controller.agent.unseen_counts.as_ref().unwrap()[&printing.0] + 1,
                before.agent.unseen_counts.as_ref().unwrap()[&printing.0]
            );
            assert_eq!(
                controller.agent_permanents[0].registry_key,
                printing.0 as i32
            );
        }
        let opponent = Observation::for_player(s.game(), PlayerId(1));
        let encoded = encode(&opponent, &ObservationEncoderConfig::default()).unwrap();
        assert_anonymous_manifest(&opponent);
        assert_eq!(
            opponent.opponent.unseen_counts.is_none(),
            disclosure == DecklistDisclosure::Closed
        );
        (
            opponent.to_json().to_string(),
            opponent.opponent_permanents,
            encoded.opponent_permanents,
            opponent.opponent.unseen_counts,
            encoded.decklists,
        )
    });
    assert_eq!(projections[0], projections[1]);
}

#[test]
fn manifest_dread_restores_both_private_commands() {
    use crate::support::{
        inspect::offered_cards,
        play::{cast, resolve_top, spend_colorless},
    };
    use managym::{agent::action::Action, decision::Command, flow::event::GameEvent};

    let mut s = five_color_setup(&["Manifest Dread", "Gray Ogre", "Lightning Bolt"]);
    s.force_card_in_hand(1, "Lightning Bolt");
    cast(&mut s, "Manifest Dread");
    spend_colorless(&mut s);
    let names = ["Gray Ogre", "Forest"];
    let cards = pin_top(&mut s, &names);
    resolve_top(&mut s);
    assert_eq!(s.action_space().player, Some(PlayerId(0)));
    assert_eq!(offered_cards(&s), cards);
    let private = Observation::for_player(s.game(), PlayerId(0));
    let opponent = Observation::for_player(s.game(), PlayerId(1));
    assert!(opponent.action_space.actions.is_empty());
    assert!(opponent.action_space.focus.is_empty());
    for (position, card) in cards.iter().enumerate() {
        assert_eq!(
            private.agent.known_library[&(position as u32)],
            s.game().state[*card].definition_id.0
        );
        assert!(!opponent
            .opponent
            .known_library
            .contains_key(&(position as u32)));
    }
    let frame = s.game().semantic_decision_frame().unwrap();
    assert_eq!(frame.offers.len(), 2);
    let snapshot = s.snapshot();
    for (i, offer) in frame.offers.iter().enumerate() {
        let mut env = snapshot.restore(Default::default()).unwrap();
        let game = env.benchmark_game_mut();
        assert_eq!(game.semantic_decision_frame().unwrap().offers, frame.offers);
        let printing = game.state[cards[i]].definition_id;
        let cmd = Command {
            command_id: format!("manifest-{i}"),
            expected_revision: frame.revision,
            offer_id: offer.id.0,
            answers: vec![],
            object_preconditions: vec![],
        };
        let selected = game.state.card_id(cards[i]).unwrap();
        game.execute_semantic_command(&cmd).unwrap();
        let manifested = game.state.zones.object_of(selected);
        assert_eq!(
            game.state.zones.zone_of(manifested),
            Some(ZoneType::Battlefield)
        );
        assert!(game.execute_semantic_command(&cmd).is_err());
        let pass = game
            .action_space()
            .unwrap()
            .actions
            .iter()
            .position(|a| matches!(a, Action::PassPriority { .. }))
            .unwrap();
        game.step(pass).unwrap();
        assert_eq!(game.action_space().unwrap().player, Some(PlayerId(1)));
        let own = Observation::for_player(game, PlayerId(0));
        assert_eq!(own.agent_permanents[0].registry_key, printing.0 as i32);
        let view = Observation::for_player(game, PlayerId(1));
        assert_anonymous_manifest(&view);
        assert!(view
            .opponent_cards
            .iter()
            .any(|c| c.zone == ZoneType::Graveyard && c.characteristics.name == names[1 - i]));
        let other = cards[1 - i];
        let graveyard_moves = game.state.events.iter().filter(|event| {
            matches!(event, GameEvent::CardMoved {
                departed: Some(id),
                to: ZoneType::Graveyard,
                ..
            } if *id == other)
        });
        assert_eq!(graveyard_moves.count(), 1);
    }
}

#[test]
fn manifest_dread_resolves_with_one_or_zero_library_cards() {
    use crate::support::{
        inspect::library,
        play::{cast, resolve_top, spend_colorless},
    };
    for remaining in [0, 1] {
        let mut s = five_color_setup(&["Manifest Dread"]);
        cast(&mut s, "Manifest Dread");
        spend_colorless(&mut s);
        for card in library(&s, 0).into_iter().skip(remaining) {
            s.game_mut().move_object(card, ZoneType::Exile);
        }
        resolve_top(&mut s);
        let view = Observation::for_player(s.game(), PlayerId(1));
        assert_eq!(view.opponent_permanents.len(), remaining);
        assert!(library(&s, 0).is_empty());
        assert!(s.game().state.suspended_decision.is_none());
        assert!(view
            .opponent_cards
            .iter()
            .any(|c| c.zone == ZoneType::Graveyard && c.characteristics.name == "Manifest Dread"));
    }
}

#[test]
fn two_generations_copy_only_the_manifested_face() {
    use crate::support::{
        inspect::card_of,
        play::{cast, resolve_top, spend_colorless},
    };
    use managym::state::game_object::Target;

    let mut s = five_color_setup(&["Clone Legion", "Aang, Swift Savior"]);
    let card = pin_top(&mut s, &["Aang, Swift Savior"])[0];
    let original = s.game_mut().manifest(card, PlayerId(0));
    s.game_mut().scenario_refresh_priority().unwrap();
    s.restore(&s.snapshot());
    for generation in 1..=2 {
        cast(&mut s, "Clone Legion");
        assert!(s.choose_target(Target::Player(PlayerId(0))));
        spend_colorless(&mut s);
        resolve_top(&mut s);
        let copies = s.battlefield_permanents_named(0, "");
        let target = *copies.last().unwrap();
        let object = &s.game().state[card_of(&s, target)];
        assert!(!object.is_card());
        assert!(
            !s.game().state.permanents[target]
                .as_ref()
                .unwrap()
                .manifested
        );
        assert!(managym::cards::content_pack::default_content_pack()
            .object_row(object)
            .back_face
            .is_none());
        for viewer in [PlayerId(0), PlayerId(1)] {
            let view = Observation::for_player(s.game(), viewer);
            encode(&view, &ObservationEncoderConfig::default()).unwrap();
            let rows = view
                .agent_permanents
                .iter()
                .chain(&view.opponent_permanents);
            let tokens: Vec<_> = rows.filter(|p| p.is_token).collect();
            assert_eq!(tokens.len(), generation);
            for token in tokens {
                assert!(token.characteristics.name.is_empty());
                assert_eq!(token.registry_key, -1);
                assert_eq!(
                    (token.characteristics.power, token.characteristics.toughness),
                    (2, 2)
                );
            }
        }
        if generation == 1 {
            s.game_mut().move_object(original, ZoneType::Graveyard);
            s.game_mut().scenario_refresh_priority().unwrap();
        }
    }
}

#[test]
fn both_samplers_deal_the_hidden_manifest_under_a_pending_bolt() {
    use crate::support::play::cast_card;
    use managym::{
        flow::game::Game,
        possible_worlds::{HiddenHandConstraints, PossibleWorldSpace},
        state::{game_object::Target, stack_object::StackObject},
    };

    let controller = PlayerId(0);
    let opponent = PlayerId(1);
    let outcomes = [("Forest", "Gray Ogre"), ("Gray Ogre", "Forest")].map(|(hidden, other)| {
        let mut s = five_color_setup(&["Gray Ogre", "Lightning Bolt"]);
        s.force_card_in_hand(0, "Lightning Bolt");
        s.force_card_in_hand(1, "Lightning Bolt");
        let cards = pin_top(&mut s, &[hidden, "Plains", other]);
        let library = s
            .game()
            .state
            .zones
            .zone_objects(ZoneType::Library, controller)
            .to_vec();
        for object in library {
            if !cards.contains(&object) {
                s.game_mut()
                    .state
                    .place_object(object, controller, ZoneType::Exile);
            }
        }
        let object = s.game_mut().manifest(cards[0], controller);
        let plains = s.game_mut().move_object(cards[1], ZoneType::Graveyard);
        let permanent = s.game().lookup_current_permanent(object).unwrap();
        let bolt = s
            .game()
            .state
            .zones
            .zone_objects(ZoneType::Hand, controller)[0];
        cast_card(&mut s, bolt);
        assert!(s.choose_target(Target::Permanent(permanent)));
        if s.action_space().player == Some(controller) {
            s.pass_priority();
        }
        assert_eq!(s.action_space().player, Some(opponent));
        let snapshot = s.snapshot();
        let assignment = |game: &Game| {
            assert_eq!(game.permanent_object(permanent), Some(object));
            assert_eq!(game.state.zones.zone_of(plains), Some(ZoneType::Graveyard));
            assert_eq!(game.state[plains].name, "Plains");
            let Some(StackObject::Spell(bolt)) = game.state.stack_objects.last() else {
                panic!("pending Bolt")
            };
            assert_eq!(bolt.targets, [Target::Permanent(permanent)]);
            let library = game.state.zones.zone_objects(ZoneType::Library, controller);
            assert_eq!(library.len(), 1);
            let (printing, row) = game.state[object].retained_base();
            let other = &game.state[library[0]];
            assert_eq!(
                game.state.cards[game.state.card_of(object).unwrap()].definition,
                printing
            );
            assert_eq!(
                game.state.cards[game.state.card_of(library[0]).unwrap()].definition,
                other.definition_id
            );
            assert_eq!(
                Observation::for_player(game, controller).agent_permanents[0].registry_key,
                printing.0 as i32
            );
            let mut pair = [row.name.as_str(), other.name.as_str()];
            pair.sort();
            assert_eq!(pair, ["Forest", "Gray Ogre"]);
            row.name.clone()
        };
        let constraints = HiddenHandConstraints::for_viewer(s.game(), opponent);
        let space = PossibleWorldSpace::for_viewer(s.game(), opponent);
        assert_eq!(space.worlds().len(), 1);
        let mut results = [Vec::new(), Vec::new()];
        for seed in 0..8 {
            let mut direct = s.game().clone();
            direct.determinize(opponent, seed).unwrap();
            results[0].push(assignment(&direct));
            let counted = constraints
                .materialize_hand(s.game(), &Default::default(), seed)
                .unwrap();
            let indexed = space
                .materialize(s.game(), &space.worlds()[0], seed)
                .unwrap();
            assert_eq!(assignment(&indexed), assignment(&counted));
            results[1].push(assignment(&counted));
        }
        for sequence in &results {
            assert!(sequence.iter().any(|name| name == hidden));
            assert!(sequence.iter().any(|name| name == other));
        }
        let mut known = s.game().clone();
        known.determinize(controller, 2).unwrap();
        assert_eq!(assignment(&known), hidden);
        assert_eq!(
            assignment(
                snapshot
                    .restore(Default::default())
                    .unwrap()
                    .benchmark_game_mut()
            ),
            hidden
        );
        assert_eq!(assignment(s.game()), hidden);
        results
    });
    assert_eq!(outcomes[0], outcomes[1]);
}

#[test]
fn manifested_cecil_reveals_in_place_between_two_ember_island_copies() {
    use crate::support::{
        game::{set_counters, Scenario},
        play::{cast, choose, resolve_top, spend_colorless},
    };
    use managym::{
        agent::action::Action,
        decision::Command,
        state::{game_object::Target, permanent::CounterKind},
    };
    let mut s = five_color_setup(&[
        "Cecil, Dark Knight",
        "Lightning Bolt",
        "Ember Island Production",
    ]);
    let player = PlayerId(0);
    s.force_card_in_hand(1, "Lightning Bolt");
    let cards = pin_top(&mut s, &["Cecil, Dark Knight", "Forest"]);
    let original = s.game_mut().manifest(cards[0], player);
    s.game_mut().manifest(cards[1], player);
    let target = s.game().lookup_current_permanent(original).unwrap();
    set_counters(&mut s, target, CounterKind::PlusOnePlusOne, 1);
    let copy = |s: &mut Scenario| {
        cast(s, "Ember Island Production");
        choose(s, |a| matches!(a, Action::ChooseMode { mode: 0, .. }));
        assert!(s.choose_target(Target::Permanent(target)));
        spend_colorless(s);
        resolve_top(s);
    };
    copy(&mut s);
    s.set_player_mana_pool(0, "R");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::TurnFaceUp { .. })));
    s.set_player_mana_pool(0, "BR");
    cast(&mut s, "Lightning Bolt");
    assert!(s.choose_target(Target::Permanent(target)));
    let turns: Vec<_> = s
        .action_space()
        .actions
        .iter()
        .enumerate()
        .filter_map(|(index, a)| match a {
            Action::TurnFaceUp { object, .. } => Some((index, *object)),
            _ => None,
        })
        .collect();
    assert_eq!(turns.len(), 1);
    assert_eq!(turns[0].1, original);
    let view = Observation::for_player(s.game(), PlayerId(1));
    let public = view
        .opponent_permanents
        .iter()
        .find(|p| p.plus1_counters == 1)
        .unwrap()
        .id;
    let snapshot = s.snapshot();
    let frame = s.game().semantic_decision_frame().unwrap();
    let cmd = Command {
        command_id: "turn-cecil-up".into(),
        expected_revision: frame.revision,
        offer_id: frame.offers[turns[0].0].id.0,
        answers: vec![],
        object_preconditions: vec![],
    };
    s.game_mut().execute_semantic_command(&cmd).unwrap();
    assert!(s.game_mut().execute_semantic_command(&cmd).is_err());
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    assert_eq!(s.game().state.players[0].mana_pool.mana().total(), 0);
    assert_eq!(s.game().permanent_object(target), Some(original));
    assert_eq!(
        s.game().state.stack_objects.last().unwrap().targets(),
        &[Target::Permanent(target)]
    );
    for viewer in [player, PlayerId(1)] {
        let view = Observation::for_player(s.game(), viewer);
        let row = view
            .agent_permanents
            .iter()
            .chain(&view.opponent_permanents)
            .find(|p| p.id == public)
            .unwrap();
        assert_eq!(row.object_id, Some(public));
        assert_eq!(row.plus1_counters, 1);
        assert_eq!(row.characteristics.name, "Cecil, Dark Knight");
        encode(&view, &ObservationEncoderConfig::default()).unwrap();
    }
    resolve_top(&mut s);
    assert_eq!(
        s.game().state.permanents[target].as_ref().unwrap().damage,
        3
    );
    s.set_player_mana_pool(0, "3UU");
    copy(&mut s);
    let content = managym::cards::content_pack::default_content_pack();
    let copies = Observation::for_player(s.game(), player);
    for name in ["", "Cecil, Dark Knight"] {
        let row = copies
            .agent_permanents
            .iter()
            .find(|p| p.is_token && p.characteristics.name == name)
            .unwrap();
        let id = s
            .battlefield_permanents_named(0, name)
            .into_iter()
            .find(|id| s.game().state.permanents[*id].as_ref().unwrap().id.0 as i32 == row.id)
            .unwrap();
        let object = s.game().permanent_object(id).unwrap();
        assert_eq!(
            content
                .object_row(&s.game().state[object])
                .back_face
                .is_some(),
            !name.is_empty()
        );
    }
    let mut fork = snapshot.restore(Default::default()).unwrap();
    let game = fork.benchmark_game_mut();
    let pass = game
        .action_space()
        .unwrap()
        .actions
        .iter()
        .position(|a| matches!(a, Action::PassPriority { .. }))
        .unwrap();
    game.step(pass).unwrap();
    assert_eq!(game.state.players[0].mana_pool.mana().total(), 1);
    assert!(game.state[original].name.is_empty());
    assert_eq!(
        game.state.stack_objects.last().unwrap().targets(),
        &[Target::Permanent(target)]
    );
}

#[test]
fn turn_up_shows_the_creature_while_its_controller_chooses_mana() {
    use crate::support::play::{choose, spend_named};
    use managym::{agent::action::Action, state::mana::Mana};
    let mut s = five_color_setup(&["Gray Ogre"]);
    let card = pin_top(&mut s, &["Gray Ogre"])[0];
    let printing = s.game().state[card].definition_id;
    let object = s.game_mut().manifest(card, PlayerId(0));
    s.set_player_mana_pool(0, "1WUR");
    s.game_mut().scenario_refresh_priority().unwrap();
    choose(
        &mut s,
        |a| matches!(a, Action::TurnFaceUp { object: chosen, .. } if *chosen == object),
    );
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    let opponent = Observation::for_player(s.game(), PlayerId(1));
    let shown = &opponent.opponent_permanents[0];
    assert_eq!(shown.registry_key, printing.0 as i32);
    assert!(shown.characteristics.name.is_empty());
    let snapshot = s.snapshot();
    for (spend, keep) in [("{W}", "U"), ("{U}", "W")] {
        s.restore(&snapshot);
        spend_named(&mut s, |description| description.contains(spend));
        if s.game().state[object].name.is_empty() {
            spend_named(&mut s, |description| description.contains("{C}"));
        }
        assert_eq!(s.game().state[object].name, "Gray Ogre");
        assert_eq!(s.game().state.players[0].mana_pool, Mana::parse(keep));
    }
}

#[test]
fn swapped_hidden_manifests_hash_apart_and_look_alike() {
    let mut s = five_color_setup(&["Gray Ogre"]);
    let cards = pin_top(&mut s, &["Gray Ogre", "Forest"]);
    let ogre = s.game_mut().manifest(cards[0], PlayerId(0));
    let forest = s.game_mut().manifest(cards[1], PlayerId(0));
    let mut swapped = s.game().clone();
    let printing = |object: managym::state::game_object::ObjectId| {
        let (id, row) = s.game().state[object].retained_definition();
        (id, row.clone())
    };
    let ((ogre_id, ogre_row), (forest_id, forest_row)) = (printing(ogre), printing(forest));
    swapped.state[ogre].assign_hidden_printing(forest_id, forest_row);
    swapped.state[forest].assign_hidden_printing(ogre_id, ogre_row);
    assert_eq!(
        swapped.state[ogre].retained_base().1.name,
        "Forest",
        "the hidden printings changed places"
    );
    // The authoritative states differ although no opponent can tell them apart.
    assert_ne!(
        swapped.state.deterministic_hash(),
        s.game().state.deterministic_hash()
    );
    assert_eq!(
        Observation::for_player(&swapped, PlayerId(1)).opponent_permanents,
        Observation::for_player(s.game(), PlayerId(1)).opponent_permanents
    );
}

#[test]
fn a_face_down_attacker_deals_damage_without_naming_its_card() {
    use crate::support::{game::clear_summoning_sickness, play::attack_with};
    use managym::{
        agent::observation::{EventEntityKind, EventType},
        flow::turn::StepKind,
    };
    let rows = [("Gray Ogre", "Forest"), ("Forest", "Gray Ogre")].map(|(hidden, other)| {
        let mut s = five_color_setup(&["Gray Ogre"]);
        let cards = pin_top(&mut s, &[hidden, "Plains", other]);
        let object = s.game_mut().manifest(cards[0], PlayerId(0));
        let permanent = s.game().lookup_current_permanent(object).unwrap();
        let public = s.game().state.permanents[permanent].as_ref().unwrap().id;
        clear_summoning_sickness(&mut s, permanent);
        s.advance_to_active_step(0, StepKind::DeclareAttackers);
        attack_with(&mut s, permanent);
        s.game_mut().take_observation_events();
        s.advance_to_active_step(0, StepKind::CombatDamage);
        assert_eq!(s.life(1), 18);
        let events = s.game_mut().take_observation_events();
        let rows = Observation::new(s.game(), &events).recent_events;
        let damage: Vec<_> = rows
            .iter()
            .filter(|row| row.event_type == EventType::DamageDealt as i32)
            .collect();
        assert!(!damage.is_empty());
        for row in damage {
            assert_eq!(
                (row.source_kind, row.source_id),
                (EventEntityKind::Object as i32, public.0 as i32)
            );
        }
        rows
    });
    assert_eq!(rows[0], rows[1]);
}

#[test]
fn a_face_down_creature_trading_in_combat_names_only_its_permanent() {
    use crate::support::{game::clear_summoning_sickness, play::attack_with};
    use managym::{
        agent::{action::ActionSpaceKind, observation::EventType},
        flow::turn::StepKind,
    };
    let rows = [("Gray Ogre", "Forest"), ("Forest", "Gray Ogre")].map(|(hidden, other)| {
        let mut s = five_color_setup(&["Gray Ogre"]);
        s.force_permanent_on_battlefield(1, "Gray Ogre");
        let cards = pin_top(&mut s, &[hidden, "Plains", other]);
        let object = s.game_mut().manifest(cards[0], PlayerId(0));
        let permanent = s.game().lookup_current_permanent(object).unwrap();
        clear_summoning_sickness(&mut s, permanent);
        s.advance_to_active_step(0, StepKind::DeclareAttackers);
        attack_with(&mut s, permanent);
        while s.action_space().kind != ActionSpaceKind::DeclareBlocker {
            s.advance_default_action();
        }
        s.declare_block();
        s.game_mut().take_observation_events();
        s.advance_to_active_step(0, StepKind::CombatDamage);
        assert!(s.game().state.permanents[permanent].is_none(), "traded");
        let events = s.game_mut().take_observation_events();
        let rows: Vec<_> = Observation::new(s.game(), &events)
            .recent_events
            .into_iter()
            // The card is revealed as it reaches the graveyard (CR 708.9).
            .filter(|row| row.event_type != EventType::CardMoved as i32)
            .collect();
        assert!(rows
            .iter()
            .any(|row| row.event_type == EventType::DamageDealt as i32));
        rows
    });
    assert_eq!(rows[0], rows[1]);
}

#[test]
fn a_face_down_permanent_bounced_to_hand_names_no_card() {
    use crate::support::game::move_card;
    use managym::agent::observation::{EventEntityKind, EventType};
    let rows = [("Gray Ogre", "Forest"), ("Forest", "Gray Ogre")].map(|(hidden, other)| {
        let mut s = five_color_setup(&["Gray Ogre"]);
        let cards = pin_top(&mut s, &[hidden, "Plains", other]);
        let object = s.game_mut().manifest(cards[0], PlayerId(0));
        s.game_mut().take_observation_events();
        move_card(&mut s, object, ZoneType::Hand);
        let events = s.game_mut().take_observation_events();
        let rows = Observation::new(s.game(), &events).recent_events;
        let moved = rows
            .iter()
            .find(|row| row.event_type == EventType::CardMoved as i32)
            .expect("bounce");
        assert_eq!(
            (moved.source_kind, moved.source_id),
            (EventEntityKind::None as i32, -1)
        );
        rows
    });
    assert_eq!(rows[0], rows[1]);
}

/// Undying Malice and Fake Your Own Death grant a face-down manifest two
/// dies triggers. When it dies, its controller orders triggers whose source
/// shows no definition (CR 708.2); capture, encoding and the stacked
/// abilities publish none, and the countered ability is named only as a
/// face-down permanent's.
#[test]
fn granted_triggers_of_a_dead_manifest_stack_and_counter_without_its_printing() {
    use crate::support::{
        inspect::offered_targets,
        play::{activate, cast, settle},
    };
    use managym::{
        agent::action::ActionSpaceKind,
        state::{
            ability::{Effect, ObjectSelector, TargetSpec},
            card::ActivatedAbilityDefinition,
            game_object::Target,
        },
    };
    let views = [("Gray Ogre", "Forest"), ("Forest", "Gray Ogre")].map(|(hidden, other)| {
        let mut s = five_color_setup(&["Gray Ogre", "Undying Malice", "Fake Your Own Death"]);
        let cards = pin_top(&mut s, &[hidden, "Plains", other]);
        let object = s.game_mut().manifest(cards[0], PlayerId(0));
        let permanent = s.game().lookup_current_permanent(object).unwrap();
        for grant in ["Undying Malice", "Fake Your Own Death"] {
            cast(&mut s, grant);
            assert!(s.choose_target(Target::Permanent(permanent)));
            settle(&mut s);
        }
        s.game_mut().move_object(object, ZoneType::Graveyard);
        s.game_mut().scenario_refresh_priority().unwrap();
        s.pass_priority();
        assert_eq!(s.action_space().kind, ActionSpaceKind::OrderTriggers);
        s.game()
            .capture_decision(PlayerId(0), 8)
            .expect("ordering captures");
        encode(
            &Observation::for_player(s.game(), PlayerId(0)),
            &ObservationEncoderConfig::default(),
        )
        .expect("ordering encodes");
        s.step_action(0);
        while s.action_space().kind == ActionSpaceKind::OrderTriggers {
            s.step_action(0);
        }
        assert_eq!(s.game().state.stack_objects.len(), 2);

        // Answer the top ability with a counter ability: no catalog card
        // counters an ability other than through ward.
        let seat = s.action_space().player.unwrap();
        let counter = s.force_permanent_on_battlefield(seat.0, "Gray Ogre");
        let counter_object = s.game().state.permanents[counter].as_ref().unwrap().object;
        s.game_mut().state[counter_object]
            .activated_abilities
            .push(ActivatedAbilityDefinition {
                effects: vec![Effect::CounterAbility {
                    what: ObjectSelector::Target(TargetSpec::Ability),
                }],
                ..Default::default()
            });
        activate(&mut s, counter, 0);
        let targets = offered_targets(&s);
        assert_eq!(targets.len(), 2);
        assert!(s.choose_target(targets[0]));
        while s.game().state.stack_objects.len() > 2 {
            s.pass_priority();
        }
        while s.game().state.stack_objects.len() == 2 {
            s.pass_priority();
        }
        let viewer = s.action_space().player.unwrap();
        let opponent = Observation::for_player(s.game(), PlayerId(1));
        encode(&opponent, &ObservationEncoderConfig::default()).expect("opponent encodes");
        let context = s.game().decision_context(viewer).unwrap();
        assert!(
            context
                .recent_facts
                .iter()
                .any(|fact| fact == "An ability of a face-down permanent was countered."),
            "{:?}",
            context.recent_facts
        );
        s.game()
            .capture_decision(viewer, 8)
            .expect("stack captures");
        assert_eq!(opponent.stack_objects.len(), 1);
        assert_eq!(opponent.stack_objects[0].source_card_registry_key, -1);
        let description = opponent.stack_objects[0].description.as_deref().unwrap();
        assert!(
            description.starts_with("a face-down permanent — "),
            "{description}"
        );
        opponent.stack_objects
    });
    assert_eq!(views[0], views[1]);
}

/// Human labels read what a viewer may see: a face-down manifest is "a
/// face-down permanent" (CR 708.2), and a creature under Witness Protection
/// is named by its effective name (CR 613.1c), in attacker offers and in
/// the target a spell announces.
#[test]
fn labels_name_a_face_down_manifest_and_a_witness_protected_creature() {
    use crate::support::{
        game::clear_summoning_sickness,
        play::{cast, settle},
    };
    use managym::{flow::turn::StepKind, state::game_object::Target};
    let mut s = five_color_setup(&["Gray Ogre", "Witness Protection", "Lightning Bolt"]);
    let protected = s.force_permanent_on_battlefield(0, "Gray Ogre");
    cast(&mut s, "Witness Protection");
    assert!(s.choose_target(Target::Permanent(protected)));
    settle(&mut s);
    let card = pin_top(&mut s, &["Forest"])[0];
    let object = s.game_mut().manifest(card, PlayerId(0));
    let face_down = s.game().lookup_current_permanent(object).unwrap();
    for permanent in [protected, face_down] {
        clear_summoning_sickness(&mut s, permanent);
    }
    let before_bolt = s.snapshot();
    cast(&mut s, "Lightning Bolt");
    let subjects = serde_json::to_string(&s.game().decision_offer_subjects(PlayerId(0))).unwrap();
    for label in ["a face-down permanent", "Legitimate Businessperson"] {
        assert!(
            subjects.contains(&format!("\"label\":\"{label}\"")),
            "{subjects}"
        );
    }
    s.restore(&before_bolt);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    let offers =
        serde_json::to_string(s.game().structured_attacker_offers().unwrap().projection()).unwrap();
    for label in ["a face-down permanent", "Legitimate Businessperson"] {
        assert!(
            offers.contains(&format!("\"label\":\"{label}\"")),
            "{offers}"
        );
    }
}
