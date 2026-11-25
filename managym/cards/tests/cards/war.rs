//! Tests for cards filed in `src/war.rs`, in card order.

use crate::support::{
    game::{five_color_setup, until_end_of_turn},
    inspect::permanent_named,
    play::{
        activate_first, play_and_settle, select_card, settle_targeting_opponent, spend_colorless,
        target_opponent,
    },
};
use managym::state::{
    card::{Keyword, Keywords},
    effect::Modification,
};

#[test]
fn mayhem_keeps_granted_lifelink_after_it_is_sacrificed() {
    let mut s = five_color_setup(&["Mayhem Devil", "Weaponize the Monsters"]);
    let object = play_and_settle(&mut s, "Mayhem Devil");
    play_and_settle(&mut s, "Weaponize the Monsters");
    let devil = permanent_named(&s, "Mayhem Devil");
    // Model a resolved until-end-of-turn grant before the sacrifice.
    until_end_of_turn(
        &mut s,
        devil,
        Modification::AddKeywords(Keywords::of(&[Keyword::Lifelink])),
    );
    let weapon = permanent_named(&s, "Weaponize the Monsters");
    activate_first(&mut s, weapon);
    target_opponent(&mut s);
    select_card(&mut s, object);
    spend_colorless(&mut s);
    settle_targeting_opponent(&mut s);
    assert_eq!(s.game().state.players[0].life, 21);
    assert_eq!(s.game().state.players[1].life, 17);
}

#[test]
fn liliana_dreadhorde_general_loyalty_abilities() {
    use crate::support::{
        game::{enter_battlefield, five_color_setup, set_counters},
        inspect::{card_of, counters_of},
        play::{activate, choose, select_card, settle},
    };
    use managym::{
        agent::action::{Action, ActionSpaceKind},
        flow::turn::StepKind,
        state::{game_object::PlayerId, permanent::CounterKind, zone::ZoneType},
    };

    // +1 creates a 2/2 black Zombie; -4 makes each player sacrifice 2 creatures and triggers Liliana's draw.
    let mut s = five_color_setup(&["Liliana, Dreadhorde General", "Gray Ogre"]);
    let liliana = enter_battlefield(&mut s, "Liliana, Dreadhorde General");
    assert_eq!(counters_of(&s, liliana, CounterKind::Loyalty), 6);
    activate(&mut s, liliana, 0);
    settle(&mut s);
    assert_eq!(counters_of(&s, liliana, CounterKind::Loyalty), 7);
    let zombies = s.battlefield_permanents_named(0, "Zombie");
    assert_eq!(zombies.len(), 1);
    assert_eq!(s.game().characteristics(zombies[0]).pt(), (2, 2));

    let own_ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let _opp_a = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let _opp_b = s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    let hand_before = s.game().state.zones.size(ZoneType::Hand, PlayerId(0));
    activate(&mut s, liliana, 1);
    settle(&mut s);
    // Both players had exactly 2 creatures, so all 4 were sacrificed; Liliana triggers twice for own 2 creatures.
    assert!(s.game().state.permanents[own_ogre].is_none());
    assert!(s.battlefield_permanents_named(0, "Zombie").is_empty());
    assert!(s.battlefield_permanents_named(1, "Gray Ogre").is_empty());
    assert_eq!(
        s.game().state.zones.size(ZoneType::Hand, PlayerId(0)),
        hand_before + 2,
        "drew one card for each own creature that died"
    );

    // -9 keep one of each permanent type, with multi-type permanents (Artifact Creature) and 30-permanent stress board.
    let mut s = five_color_setup(&[
        "Liliana, Dreadhorde General",
        "Gray Ogre",
        "Hill Giant",
        "Prophetic Prism",
        "Sol Ring",
        "Way of the Mentor",
        "Way of the Warlord",
        "Ajani, Caller of the Pride",
        "Chandra, Flameshaper",
    ]);
    let liliana = enter_battlefield(&mut s, "Liliana, Dreadhorde General");
    set_counters(&mut s, liliana, CounterKind::Loyalty, 10);
    // Opponent controls 30 permanents: 10 Artifact Creature tokens, 6 Creatures (4 Ogres + 2 Giants),
    // 5 Artifacts (4 Prisms + 1 Sol Ring), 5 Lands, 2 Enchantments, 2 distinct Planeswalkers.
    let boulders: Vec<_> = (0..10)
        .map(|_| {
            s.game_mut()
                .create_token("token.ballistic_boulder", PlayerId(1))
                .unwrap()
        })
        .collect();
    let mut creatures = Vec::new();
    for _ in 0..4 {
        creatures.push(s.force_permanent_on_battlefield(1, "Gray Ogre"));
    }
    for _ in 0..2 {
        creatures.push(s.force_permanent_on_battlefield(1, "Hill Giant"));
    }
    for _ in 0..4 {
        s.force_permanent_on_battlefield(1, "Prophetic Prism");
    }
    s.force_permanent_on_battlefield(1, "Sol Ring");
    for _ in 0..5 {
        s.force_permanent_on_battlefield(1, "Plains");
    }
    s.force_permanent_on_battlefield(1, "Way of the Mentor");
    s.force_permanent_on_battlefield(1, "Way of the Warlord");
    let pw1 = s.force_permanent_on_battlefield(1, "Ajani, Caller of the Pride");
    set_counters(&mut s, pw1, CounterKind::Loyalty, 4);
    let pw2 = s.force_permanent_on_battlefield(1, "Chandra, Flameshaper");
    set_counters(&mut s, pw2, CounterKind::Loyalty, 6);
    assert_eq!(
        s.game()
            .state
            .zones
            .size(ZoneType::Battlefield, PlayerId(1)),
        30
    );
    activate(&mut s, liliana, 2);
    // Pass priority to resolve -9.
    while s.action_space().kind == ActionSpaceKind::Priority {
        choose(&mut s, |a| matches!(a, Action::PassPriority { .. }));
    }
    // 1. Artifact choice: 10 Artifact Creatures + 5 Artifacts = 15 choices (linear!).
    assert_eq!(s.action_space().kind, ActionSpaceKind::LookAndSelect);
    assert_eq!(s.action_space().actions.len(), 15);
    let kept_boulder = boulders[0];
    let kept_boulder_card = card_of(&s, kept_boulder);
    select_card(&mut s, kept_boulder_card);
    // 2. Battle: none (skipped automatically).
    // 3. Creature choice: 10 Artifact Creatures + 6 Creatures = 16 choices; choose an Ogre so both boulder & ogre are kept!
    assert_eq!(s.action_space().actions.len(), 16);
    let kept_ogre = creatures[0];
    let kept_ogre_card = card_of(&s, kept_ogre);
    select_card(&mut s, kept_ogre_card);
    // 4. Enchantment: 2 choices.
    assert_eq!(s.action_space().actions.len(), 2);
    choose(&mut s, |a| matches!(a, Action::SelectCard { .. }));
    // 5. Land: 5 choices.
    assert_eq!(s.action_space().actions.len(), 5);
    choose(&mut s, |a| matches!(a, Action::SelectCard { .. }));
    // 6. Planeswalker: 2 choices.
    assert_eq!(s.action_space().actions.len(), 2);
    choose(&mut s, |a| matches!(a, Action::SelectCard { .. }));
    settle(&mut s);
    assert!(s.game().state.permanents[kept_boulder].is_some());
    assert!(s.game().state.permanents[kept_ogre].is_some());
    assert_eq!(
        s.game()
            .state
            .zones
            .size(ZoneType::Battlefield, PlayerId(1)),
        5,
        "kept one of each of the 5 present permanent types and sacrificed the other 25"
    );
}
