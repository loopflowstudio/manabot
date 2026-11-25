//! Tests for cards filed in `src/grn.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    inspect::{card_of, hand},
    play::{cast, settle},
};
use managym::state::{game_object::PlayerId, player::ManaPayment};

/// Midnight Reaper triggers for each nontoken creature its controller
/// controlled that died, itself and a land copying a creature included, and not
/// for a creature token dying beside them.
#[test]
fn midnight_reaper_pings_and_draws_for_each_nontoken_death() {
    let mut s = five_color_setup(&["Midnight Reaper", "Gray Ogre", "Fumigate"]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    s.force_permanent_on_battlefield(0, "Midnight Reaper");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    // A noncreature card copying a creature is a nontoken creature.
    let island = s.force_permanent_on_battlefield(0, "Island");
    let original = s.game().state[card_of(&s, ogre)].clone();
    let copy = card_of(&s, island);
    s.game_mut().state[copy].become_copy_of(&original, &[]);
    s.game_mut()
        .create_token("token.cat_1_1_w", PlayerId(0))
        .unwrap();
    let before = hand(&s, 0).len();
    cast(&mut s, "Fumigate");
    s.advance_until(
        |s| s.game().state.stack_objects.len() == 3,
        "three Reaper triggers".into(),
    );
    settle(&mut s);
    // Fumigate gains one for each of four creatures; each trigger deals one.
    s.assert_life(0, 21);
    assert_eq!(hand(&s, 0).len(), before + 3);
}

#[test]
fn printed_creature_swiftblade_vindicator() {
    use managym::state::card::Keyword;
    crate::printed_creature_combat(
        "Swiftblade Vindicator",
        1,
        1,
        &[Keyword::Vigilance, Keyword::Trample, Keyword::DoubleStrike],
    );
}

/// "When this creature enters, each opponent discards a card": the opponent
/// chooses among only their own cards, without a target, and the
/// controller's hand is untouched.
#[test]
fn foundations_5f_burglar_rat() {
    use crate::support::{
        game::{five_color_setup, move_card, put_onto_battlefield},
        inspect::{hand, library_card, offered_cards, zone_of},
        play::{fork_selecting, pass_until, settle},
    };
    use managym::{
        agent::action::ActionSpaceKind,
        state::{game_object::PlayerId, zone::ZoneType},
    };
    let mut s = five_color_setup(&["Burglar Rat"]);
    let theirs: Vec<_> = ["Plains", "Island"]
        .into_iter()
        .map(|name| {
            let id = library_card(&s, 1, name);
            move_card(&mut s, id, ZoneType::Hand)
        })
        .collect();
    let id = library_card(&s, 0, "Forest");
    let ours = move_card(&mut s, id, ZoneType::Hand);
    put_onto_battlefield(&mut s, 0, "Burglar Rat");
    pass_until(&mut s, ActionSpaceKind::Discard);
    assert_eq!(s.action_space().player, Some(PlayerId(1)));
    let mut offered = offered_cards(&s);
    offered.sort();
    let mut wanted = theirs.clone();
    wanted.sort();
    assert_eq!(offered, wanted);
    for discarded in theirs {
        let mut fork = fork_selecting(&s, discarded);
        settle(&mut fork);
        assert_eq!(zone_of(&fork, discarded), Some(ZoneType::Graveyard));
        assert_eq!(hand(&fork, 1).len(), 1);
        assert_eq!(hand(&fork, 0), vec![ours]);
    }
}

/// "Search your library for up to two basic land cards and/or Gate cards,
/// put them onto the battlefield tapped, then shuffle": basics and a Gate are
/// offered, a creature never; the chosen lands enter tapped.
#[test]
fn foundations_5f_circuitous_route() {
    use crate::support::{
        game::five_color_setup,
        inspect::{offered_cards, permanent_of, tapped},
        play::{cast, fork_selecting, pass_until, select_card, settle, spend_colorless},
    };
    use managym::agent::action::ActionSpaceKind;
    let mut s = five_color_setup(&["Circuitous Route", "Gray Ogre", "Azorius Guildgate"]);
    cast(&mut s, "Circuitous Route");
    spend_colorless(&mut s);
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    let offered = offered_cards(&s);
    let name = |s: &crate::support::game::Scenario, id: managym::state::game_object::ObjectId| {
        s.game().state[id].name.clone()
    };
    assert!(offered
        .iter()
        .any(|id| name(&s, *id) == "Azorius Guildgate"));
    assert!(offered.iter().any(|id| name(&s, *id) == "Forest"));
    assert!(offered.iter().all(|id| name(&s, *id) != "Gray Ogre"));
    let gate = *offered
        .iter()
        .find(|id| name(&s, **id) == "Azorius Guildgate")
        .unwrap();
    let forest = *offered
        .iter()
        .find(|id| name(&s, **id) == "Forest")
        .unwrap();
    let mut none = s.clone();
    settle(&mut none);
    assert_eq!(none.battlefield_permanents_named(0, "Forest").len(), 0);

    select_card(&mut s, gate);
    let _ = fork_selecting;
    select_card(&mut s, forest);
    settle(&mut s);
    for card in [gate, forest] {
        let permanent = permanent_of(&s, card).expect("on the battlefield");
        assert!(tapped(&s, permanent));
    }
}
