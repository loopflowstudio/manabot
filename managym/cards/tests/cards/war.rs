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
    // 1. Artifact choice: 10 Artifact Creatures + 5 Artifacts = 15 choices, folded into 3 canonical offers (Boulder, Prism, Sol Ring).
    assert_eq!(s.action_space().kind, ActionSpaceKind::LookAndSelect);
    assert_eq!(
        crate::support::inspect::offered_cards_with_aliases(&s).len(),
        15
    );
    assert_eq!(s.action_space().actions.len(), 3);
    let kept_boulder = boulders[0];
    let kept_boulder_card = card_of(&s, kept_boulder);
    select_card(&mut s, kept_boulder_card);
    // 2. Battle: none (skipped automatically).
    // 3. Creature choice: 10 Artifact Creatures + 6 Creatures = 16 choices, folded into 4 canonical offers (remembered Boulder, other Boulders, Ogres, Giants); choose an Ogre so both boulder & ogre are kept!
    assert_eq!(
        crate::support::inspect::offered_cards_with_aliases(&s).len(),
        16
    );
    assert_eq!(s.action_space().actions.len(), 4);
    let kept_ogre = creatures[0];
    let kept_ogre_card = card_of(&s, kept_ogre);
    select_card(&mut s, kept_ogre_card);
    // 4. Enchantment: 2 choices.
    assert_eq!(s.action_space().actions.len(), 2);
    choose(&mut s, |a| matches!(a, Action::SelectCard { .. }));
    // 5. Land: 5 identical Plains fold into 1 canonical offer (5 with aliases).
    assert_eq!(
        crate::support::inspect::offered_cards_with_aliases(&s).len(),
        5
    );
    assert_eq!(s.action_space().actions.len(), 1);
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

#[test]
fn foundations_stack_copy_bolt_bend_changes_one_target_at_its_actual_cost() {
    use crate::support::{
        inspect::{now, offered_targets},
        play::{cast, resolve_top, settle},
    };
    use managym::state::{
        game_object::{PlayerId, Target},
        player::ManaPayment,
        stack_object::StackObject,
    };
    for ferocious in [false, true] {
        let mut s = five_color_setup(&[
            "Bolt Bend",
            "Bite Down",
            "Lightning Bolt",
            "Gray Ogre",
            "Shivan Dragon",
        ]);
        s.game_mut().mana_payment[0] = ManaPayment::Auto;
        if ferocious {
            s.force_permanent_on_battlefield(0, "Shivan Dragon");
        }
        let ours = s.force_permanent_on_battlefield(0, "Gray Ogre");
        let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
        let bite = cast(&mut s, "Bite Down");
        assert!(s.choose_target(Target::Permanent(ours)));
        assert!(s.choose_target(Target::Permanent(theirs)));
        let bolt = cast(&mut s, "Lightning Bolt");
        assert!(s.choose_target(Target::Player(PlayerId(0))));
        cast(&mut s, "Bolt Bend");
        assert!(!offered_targets(&s).contains(&Target::StackSpell(now(&s, bite))));
        assert!(s.choose_target(Target::StackSpell(now(&s, bolt))));
        let Some(StackObject::Spell(bend)) = s.game().state.stack_objects.last() else {
            panic!("expected Bolt Bend")
        };
        assert_eq!(bend.spent.total(), if ferocious { 1 } else { 4 });
        resolve_top(&mut s);
        assert!(!offered_targets(&s).contains(&Target::Player(PlayerId(0))));
        assert!(s.choose_target(Target::Player(PlayerId(1))));
        settle(&mut s);
        assert_eq!((s.life(0), s.life(1)), (20, 17));
    }
}

#[test]
fn finale_of_revelation_draws_x_and_at_ten_shuffles_untaps_up_to_five_and_exiles() {
    use crate::support::{
        game::{force_lands, plant_in_graveyard},
        inspect::{card_of, hand, zone_of},
        play::{cast, choose, choose_mode, pass_until, select_card, settle},
    };
    use managym::{
        agent::action::{Action, ActionSpaceKind},
        state::{player::ManaPayment, zone::ZoneType},
    };

    // X = 2: draw 2, exile Finale, graveyard untouched.
    let mut s = five_color_setup(&["Finale of Revelation", "Gray Ogre"]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let dead = plant_in_graveyard(&mut s, 0, "Gray Ogre");
    let before = hand(&s, 0).len();
    let finale = cast(&mut s, "Finale of Revelation");
    choose_mode(&mut s, 2);
    settle(&mut s);
    assert_eq!(hand(&s, 0).len(), before + 2);
    assert_eq!(zone_of(&s, dead), Some(ZoneType::Graveyard));
    assert_eq!(zone_of(&s, finale), Some(ZoneType::Exile));
    assert!(!s
        .game()
        .has_no_maximum_hand_size_effect(managym::state::game_object::PlayerId(0)));

    // X = 10: shuffle graveyard, draw 10, untap up to 5 lands, no max hand size, exile Finale.
    let mut s = five_color_setup(&["Finale of Revelation", "Gray Ogre"]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    s.set_player_mana_pool(0, "10UU");
    let dead = plant_in_graveyard(&mut s, 0, "Gray Ogre");
    force_lands(&mut s, 0, "Island", 6);
    let lands = s.battlefield_permanents_named(0, "Island");
    for &land in &lands {
        s.game_mut().state.permanents[land].as_mut().unwrap().tapped = true;
    }
    let before = hand(&s, 0).len();
    let finale = cast(&mut s, "Finale of Revelation");
    choose_mode(&mut s, 10);
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    for &land in &lands[..5] {
        let card = card_of(&s, land);
        select_card(&mut s, card);
    }
    if s.action_space().kind == ActionSpaceKind::LookAndSelect {
        choose(&mut s, |a| matches!(a, Action::Decline { .. }));
    }
    settle(&mut s);
    assert_eq!(hand(&s, 0).len(), before + 10);
    assert_ne!(zone_of(&s, dead), Some(ZoneType::Graveyard));
    assert_eq!(zone_of(&s, finale), Some(ZoneType::Exile));
    assert_eq!(s.untapped_permanents_named(0, "Island"), 5);
    assert!(s
        .game()
        .has_no_maximum_hand_size_effect(managym::state::game_object::PlayerId(0)));
}
