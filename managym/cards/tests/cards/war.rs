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
