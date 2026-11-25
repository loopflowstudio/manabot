//! Lita's vigilance, scoped untap, and Zeppelin activation.
use crate::support::{
    game::{clear_summoning_sickness, five_color_setup, five_color_unfunded_setup},
    inspect::{card_of, observed, tapped},
    play::{activate, attack_with, choose, offers_ability, pass_until, settle},
};
use managym::{
    agent::action::{Action, ActionSpaceKind},
    flow::turn::StepKind,
    state::{card::Keyword, game_object::PlayerId, mana::Mana, zone::ZoneType},
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
            |a| matches!(a, Action::SelectCard { object: c, .. } if *c == card),
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

#[test]
fn foundations_spell_deadly_plot() {
    use crate::support::game::{five_color_setup, move_card};
    use crate::support::inspect::{card_of, permanent, permanent_of, zone_of};
    use crate::support::play::{cast, choose_mode, settle};
    use managym::state::game_object::Target;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Deadly Plot", "Gray Ogre", "Hungry Ghoul"]);
    let subject = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let object = card_of(&s, subject);
    let zombie = s.force_permanent_on_battlefield(0, "Hungry Ghoul");
    let zombie_card = card_of(&s, zombie);
    let grave = move_card(&mut s, zombie_card, ZoneType::Graveyard);
    cast(&mut s, "Deadly Plot");
    assert!(crate::support::inspect::mode_labels(&s)[1].contains("Zombie creature card"));
    choose_mode(&mut s, 0);
    assert!(s.choose_target(Target::Permanent(subject)));
    settle(&mut s);
    assert_eq!(zone_of(&s, object), Some(ZoneType::Graveyard));
    cast(&mut s, "Deadly Plot");
    choose_mode(&mut s, 1);
    assert!(s.choose_target(Target::GraveyardCard(grave)));
    settle(&mut s);
    assert!(permanent(&s, permanent_of(&s, grave).unwrap()).tapped);
}

#[test]
fn mild_mannered_librarian_becomes_a_werewolf_only_once() {
    let mut s = five_color_unfunded_setup(&["Mild-Mannered Librarian"]);
    let librarian = s.force_permanent_on_battlefield(0, "Mild-Mannered Librarian");
    let hand = s.zone_size(0, ZoneType::Hand);
    s.set_player_mana_pool(0, "3G");
    activate(&mut s, librarian, 0);
    settle(&mut s);
    assert_eq!(s.zone_size(0, ZoneType::Hand), hand + 1);
    let view = s.game().characteristics(librarian);
    assert_eq!(view.pt(), (3, 3));
    assert_eq!(view.creature_types.listed(), ["Werewolf".to_string()]);
    let row = observed(&s, librarian);
    assert_eq!(row.used_abilities, [0]);
    assert_eq!(row.exhausted_abilities, 0, "not an exhaust ability");
    assert_eq!(row.creature_types, Some(vec!["Werewolf".to_string()]));

    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    s.set_player_mana_pool(0, "3G");
    assert!(!offers_ability(&mut s, librarian, 0), "activate only once");
    assert_eq!(s.game().characteristics(librarian).pt(), (3, 3));
}
