//! Dark Ascension through ordinary combat and spell resolution.
use crate::support::game::{
    clear_summoning_sickness, five_color_setup, move_card, until_end_of_turn,
};
use crate::support::inspect::{card_of, hand};
use crate::support::play::{attack_with, cast, command, pass_until, resolve_top, settle};
use managym::agent::action::{Action, ActionSpaceKind};
use managym::agent::observation::Observation;
use managym::agent::observation_encoder::ObservationEncoderConfig;
use managym::agent::observation_layout::{permanent, player};
use managym::agent::observation_rows::InputRows;
use managym::cards::tokens::{name_value, TokenKind};
use managym::flow::turn::StepKind;
use managym::state::card::{Keyword, Keywords};
use managym::state::effect::Modification;
use managym::state::game_object::PlayerId;
use managym::state::player::DecklistDisclosure;
use managym::state::zone::ZoneType;

#[test]
fn predator_ooze_grows_on_attack_and_when_its_surviving_blocker_dies() {
    let mut s = five_color_setup(&["Predator Ooze", "Hill Giant", "Fumigate"]);
    let ooze = s.force_permanent_on_battlefield(0, "Predator Ooze");
    let giant = s.force_permanent_on_battlefield(1, "Hill Giant");
    clear_summoning_sickness(&mut s, ooze);
    pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
    attack_with(&mut s, ooze);
    pass_until(&mut s, ActionSpaceKind::DeclareBlocker);
    assert_eq!(s.game().characteristics(ooze).pt(), (2, 2));
    s.declare_block();
    s.advance_to_active_step(0, StepKind::PostcombatMain);
    assert_eq!(s.game().state.permanents[giant].as_ref().unwrap().damage, 2);
    s.set_player_mana_pool(0, "3WW");
    cast(&mut s, "Fumigate");
    settle(&mut s);
    assert!(s.game().state.permanents[giant].is_none());
    assert_eq!(s.game().characteristics(ooze).pt(), (3, 3));
    s.assert_life(0, 21);
}

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
        let expected_step = if hit == 1 {
            StepKind::FirstStrikeCombatDamage
        } else {
            StepKind::CombatDamage
        };
        for seat in 0..2 {
            let obs = Observation::for_player(s.game(), PlayerId(seat));
            assert_eq!(obs.turn.step, expected_step);
            assert!(obs
                .opponent_cards
                .iter()
                .all(|card| card.zone != ZoneType::Hand));
            let encoded = InputRows::read(&obs, &ObservationEncoderConfig::default()).unwrap();
            let player_row = encoded.row("agent_player", 0);
            assert_eq!(player_row[player::STEP_COMBAT_DAMAGE], 1.0);
            assert_eq!(
                player_row[player::STEP_FIRST_STRIKE_COMBAT_DAMAGE],
                f32::from(hit == 1)
            );
            let (permanents, table) = if seat == 0 {
                (&obs.agent_permanents, "agent_permanents")
            } else {
                (&obs.opponent_permanents, "opponent_permanents")
            };
            let row = permanents.iter().position(|p| p.id == id).unwrap();
            assert!(permanents[row].first_strike_participant);
            assert_eq!(
                encoded.row(table, row)[permanent::FIRST_STRIKE_PARTICIPANT],
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
            let (mut removed, _) = command(&mut s, |a| matches!(a, Action::PassPriority { .. }));
            let projected =
                |game: &managym::flow::game::Game| Observation::for_player(game, PlayerId(0));
            assert_eq!(projected(s.game()), projected(removed.game()));
            let mut changed = removed.clone();
            until_end_of_turn(&mut changed, reaver, Modification::LoseAllAbilities);
            changed.pass_priority();
            let second = projected(changed.game());
            assert_eq!(second.turn.step, StepKind::CombatDamage);
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
