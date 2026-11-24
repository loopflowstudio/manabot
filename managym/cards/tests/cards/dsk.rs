//! Manifest Dread's private library choice and viewer-specific battlefield projection.
use crate::support::game::{five_color_setup, pin_top};
use managym::{
    agent::{
        observation::Observation,
        observation_encoder::{encode, ObservationEncoderConfig},
    },
    state::{game_object::PlayerId, zone::ZoneType},
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
    let projections = [("Gray Ogre", "Forest"), ("Forest", "Gray Ogre")].map(|(hidden, other)| {
        let mut s = five_color_setup(&["Gray Ogre", "Lightning Bolt"]);
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
        (
            opponent.opponent_permanents,
            encoded.opponent_permanents,
            opponent.opponent.unseen_counts.unwrap(),
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
    *s.game_mut() = s
        .snapshot()
        .restore(Default::default())
        .unwrap()
        .benchmark_game_mut()
        .clone();
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
