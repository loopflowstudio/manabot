//! Tests for cards first printed in Scars of Mirrodin.
use crate::support::{
    game::{five_color_setup, force_new_card_in_hand},
    play::{cast, cast_card, choose_mode, settle},
};
use managym::flow::turn::StepKind;

#[test]
fn exsanguinate_gains_the_life_actually_lost() {
    for (protected, x, lost) in [(false, 3, 3), (true, 3, 0), (false, 0, 0)] {
        let mut s = five_color_setup(&["Exsanguinate", "Teferi's Protection"]);
        if protected {
            cast(&mut s, "Teferi's Protection");
            settle(&mut s);
        }
        s.advance_to_active_step(1, StepKind::Main);
        s.set_player_mana_pool(1, "3BB");
        let spell = force_new_card_in_hand(&mut s, 1, "Exsanguinate");
        cast_card(&mut s, spell);
        choose_mode(&mut s, x);
        settle(&mut s);
        s.assert_life(0, 20 - lost);
        s.assert_life(1, 20 + lost);
    }
}

#[test]
fn steel_hellkite_spends_its_own_activation_and_destroys_exact_mana_value() {
    use crate::support::{
        game::clear_summoning_sickness,
        play::{activate, attack_with, pass_until},
    };
    use managym::{
        agent::{action::ActionSpaceKind, observation::Observation},
        state::game_object::PlayerId,
    };
    let mut s = five_color_setup(&["Steel Hellkite", "Hill Giant"]);
    let hit = s.force_permanent_on_battlefield(0, "Steel Hellkite");
    let other = s.force_permanent_on_battlefield(0, "Steel Hellkite");
    assert_ne!(hit, other);
    clear_summoning_sickness(&mut s, hit);
    pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
    attack_with(&mut s, hit);
    s.advance_to_active_step(0, StepKind::PostcombatMain);
    s.assert_life(1, 15);
    let land = s.force_permanent_on_battlefield(1, "Forest");
    let giant = s.force_permanent_on_battlefield(1, "Hill Giant");
    let own = s
        .game_mut()
        .create_token("token.food", PlayerId(0), false)
        .unwrap();
    let target = s
        .game_mut()
        .create_token("token.food", PlayerId(1), false)
        .unwrap();
    s.set_player_mana_pool(0, "2");
    let saved = s.game().clone();
    for source in [other, hit] {
        *s.game_mut() = saved.clone();
        activate(&mut s, source, 1);
        choose_mode(&mut s, 0);
        let view = Observation::for_player(s.game(), PlayerId(0));
        for (id, used) in [
            (source, true),
            (if source == hit { other } else { hit }, false),
        ] {
            let object = s.game().state.permanents[id].as_ref().unwrap();
            let row = view
                .agent_permanents
                .iter()
                .find(|row| row.id == object.id.0 as i32)
                .unwrap();
            assert_eq!(row.used_abilities.contains(&1), used);
        }
        settle(&mut s);
        assert_eq!(s.game().state.permanents[target].is_none(), source == hit);
        for kept in [own, land, giant] {
            assert!(s.game().state.permanents[kept].is_some());
        }
    }
    *s.game_mut() = saved.clone();
    s.set_player_mana_pool(0, "4");
    activate(&mut s, hit, 1);
    choose_mode(&mut s, 4);
    settle(&mut s);
    assert!(s.game().state.permanents[giant].is_none());
    assert!(s.game().state.permanents[target].is_some());

    *s.game_mut() = saved;
    activate(&mut s, hit, 1);
    choose_mode(&mut s, 0);
    settle(&mut s);
    s.advance_to_active_step(1, StepKind::Main);
    let view = Observation::for_player(s.game(), PlayerId(0));
    assert!(view
        .agent_permanents
        .iter()
        .all(|p| p.used_abilities.is_empty() && p.combat_damage_to_players.is_empty()));
    s.pass_priority();
    s.set_player_mana_pool(0, "1");
    activate(&mut s, hit, 1);
    choose_mode(&mut s, 0);
    let fresh = s
        .game_mut()
        .create_token("token.food", PlayerId(1), false)
        .unwrap();
    settle(&mut s);
    assert!(s.game().state.permanents[fresh].is_some());
}
