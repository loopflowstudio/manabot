//! Dark Ascension card behavior.
use crate::support::{
    game::{clear_summoning_sickness, five_color_setup, move_card, until_end_of_turn},
    inspect::{card_of, hand},
    play::{attack_with, command, pass_until, resolve_top},
};
use managym::{
    agent::{
        action::{Action, ActionSpaceKind},
        observation::Observation,
        observation_encoder::{encode, ObservationEncoderConfig},
        observation_layout::{permanent, player},
    },
    cards::tokens::{name_value, TokenKind},
    flow::turn::StepKind,
    state::{
        card::{Keyword, Keywords},
        effect::Modification,
        game_object::PlayerId,
        player::DecklistDisclosure,
        zone::ZoneType,
    },
};

#[test]
fn drogskol_reaver_draws_once_in_each_damage_step() {
    let mut s = five_color_setup(&["Drogskol Reaver"]);
    s.game_mut().state.disclosure = DecklistDisclosure::Closed;
    let reaver = s.force_permanent_on_battlefield(0, "Drogskol Reaver");
    clear_summoning_sickness(&mut s, reaver);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    let opening_hand = hand(&s, 0).len();
    attack_with(&mut s, reaver);
    s.advance_to_active_step(0, StepKind::CombatDamage);
    for hit in 1..=2 {
        assert_eq!(s.current_step(), StepKind::CombatDamage);
        let id = s.game().state.permanents[reaver].as_ref().unwrap().id.0 as i32;
        for seat in 0..2 {
            let obs = Observation::for_player(s.game(), PlayerId(seat));
            assert_eq!(obs.turn.another_damage_step, hit == 1);
            assert!(obs
                .opponent_cards
                .iter()
                .all(|card| card.zone != ZoneType::Hand));
            let encoded = encode(&obs, &ObservationEncoderConfig::default()).unwrap();
            assert_eq!(
                encoded.agent_player[player::ANOTHER_DAMAGE_STEP],
                f32::from(hit == 1)
            );
            let (permanents, rows) = if seat == 0 {
                (&obs.agent_permanents, &encoded.agent_permanents)
            } else {
                (&obs.opponent_permanents, &encoded.opponent_permanents)
            };
            let row = permanents.iter().position(|p| p.id == id).unwrap();
            assert!(permanents[row].first_strike_participant);
            assert_eq!(
                rows[row * permanent::WIDTH + permanent::FIRST_STRIKE_PARTICIPANT],
                1.0
            );
        }
        assert_eq!(s.life(0), 20 + 3 * hit);
        assert_eq!(s.life(1), 20 - 3 * hit);
        assert_eq!(s.game().state.stack_objects.len(), 1);
        assert_eq!(hand(&s, 0).len(), opening_hand + hit as usize - 1);
        resolve_top(&mut s);
        assert!(s.game().state.stack_objects.is_empty());
        assert_eq!(hand(&s, 0).len(), opening_hand + hit as usize);
        if hit == 1 {
            let mut removed = command(&mut s, |a| matches!(a, Action::PassPriority { .. }));
            let projected =
                |game: &managym::flow::game::Game| Observation::for_player(game, PlayerId(0));
            assert_eq!(projected(s.game()), projected(removed.game()));
            let mut changed = removed.clone();
            until_end_of_turn(&mut changed, reaver, Modification::LoseAllAbilities);
            changed.pass_priority();
            let second = projected(changed.game());
            assert_eq!(second.turn.step, StepKind::CombatDamage);
            assert!(!second.turn.another_damage_step);
            let participant = second.agent_permanents.iter().find(|p| p.id == id).unwrap();
            assert!(participant.first_strike_participant);
            assert!(!participant.keywords.double_strike);
            assert_eq!(changed.life(1), 17);
            let object = card_of(&removed, reaver);
            move_card(&mut removed, object, ZoneType::Graveyard);
            removed.advance_to_active_step(0, StepKind::EndOfCombat);
            assert_eq!((removed.life(0), removed.life(1)), (23, 17));
            assert_eq!(hand(&removed, 0).len(), opening_hand + 1);
        } else {
            s.pass_priority();
        }
        s.pass_priority();
    }
    assert_eq!(s.current_step(), StepKind::EndOfCombat);
}

#[test]
fn first_strike_assignment_names_the_next_step_and_participant_in_learning_tokens() {
    let mut s = five_color_setup(&["Drogskol Reaver", "Healer's Hawk"]);
    let reaver = s.force_permanent_on_battlefield(0, "Drogskol Reaver");
    s.force_permanent_on_battlefield(1, "Healer's Hawk");
    clear_summoning_sickness(&mut s, reaver);
    until_end_of_turn(
        &mut s,
        reaver,
        Modification::AddKeywords(Keywords::of(&[Keyword::Trample])),
    );
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, reaver);
    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    s.declare_block();
    pass_until(&mut s, ActionSpaceKind::Modal);
    let obs = Observation::for_player(s.game(), PlayerId(0));
    let frame = obs.continuation.last().unwrap();
    let label = |name| (TokenKind::Text as u16, name_value(name));
    assert!(frame
        .effect_execution
        .windows(2)
        .any(|pair| pair == [label("another damage step"), (TokenKind::Boolean as u16, 1)]));
    let source = frame
        .effect_execution
        .windows(2)
        .find(|pair| pair[0] == label("combat damage source"))
        .unwrap()[1];
    assert!(frame
        .effect_execution
        .windows(2)
        .any(|pair| pair == [label("first-strike participant"), source]));
}
