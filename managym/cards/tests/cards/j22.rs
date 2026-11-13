//! Lita's vigilance, scoped untap, and Zeppelin activation.
use crate::support::{
    game::{clear_summoning_sickness, five_color_setup, five_color_unfunded_setup},
    inspect::{card_of, tapped},
    play::{activate, attack_with, choose, pass_until, settle},
};
use managym::{
    agent::action::{Action, ActionSpaceKind},
    flow::turn::StepKind,
    state::{card::Keyword, game_object::PlayerId, mana::Mana},
};

const LITA: &str = "Lita, Mechanical Engineer";

#[test]
fn lita_attacks_without_tapping() {
    let mut s = five_color_setup(&[LITA]);
    let lita = s.force_permanent_on_battlefield(0, LITA);
    clear_summoning_sickness(&mut s, lita);
    pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
    attack_with(&mut s, lita);
    assert!(!tapped(&s, lita));
}

#[test]
fn lita_untaps_only_other_artifact_creatures_on_her_controllers_end_step() {
    let mut s = five_color_setup(&[LITA, "Gray Ogre"]);
    let lita = s.force_permanent_on_battlefield(0, LITA);
    let own = s
        .game_mut()
        .create_token("token.ballistic_boulder", PlayerId(0), false)
        .unwrap();
    let enemy = s
        .game_mut()
        .create_token("token.ballistic_boulder", PlayerId(1), false)
        .unwrap();
    let vehicle = s
        .game_mut()
        .create_token("token.zeppelin", PlayerId(0), false)
        .unwrap();
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    for turn in [1, 0] {
        s.advance_to_active_step(turn, StepKind::Main);
        for id in [lita, own, enemy, vehicle, ogre] {
            s.game_mut().state.permanents[id].as_mut().unwrap().tapped = true;
        }
        s.advance_to_active_step(turn, StepKind::End);
        settle(&mut s);
        assert_eq!(tapped(&s, own), turn != 0);
        for id in [lita, enemy, vehicle, ogre] {
            assert!(tapped(&s, id));
        }
    }
}

#[test]
fn lita_creates_a_zeppelin_that_crews_until_end_of_turn() {
    let mut s = five_color_unfunded_setup(&[LITA, "Gray Ogre", "Llanowar Elves"]);
    let lita = s.force_permanent_on_battlefield(0, LITA);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let elf = s.force_permanent_on_battlefield(0, "Llanowar Elves");
    clear_summoning_sickness(&mut s, lita);
    s.set_player_mana_pool(0, "3W");
    activate(&mut s, lita, 0);
    assert!(tapped(&s, lita));
    settle(&mut s);
    assert_eq!(s.game().state.players[0].mana_pool, Mana::default());
    let zeppelin = s.battlefield_permanents_named(0, "Zeppelin")[0];
    let view = s.game().characteristics(zeppelin);
    assert!(!view.is_creature());
    assert_eq!(view.pt(), (5, 5));
    assert!(view.keywords.has(Keyword::Flying));
    activate(&mut s, zeppelin, 0);
    for crew in [ogre, elf] {
        let card = card_of(&s, crew);
        choose(
            &mut s,
            |a| matches!(a, Action::SelectCard { card: c, .. } if *c == card),
        );
    }
    settle(&mut s);
    assert!(tapped(&s, ogre) && tapped(&s, elf));
    assert!(s.game().characteristics(zeppelin).is_creature());
    s.advance_to_active_step(0, StepKind::End);
    settle(&mut s);
    s.advance_to_active_step(1, StepKind::Main);
    assert!(!s.game().characteristics(zeppelin).is_creature());
}
