//! Tests for cards filed in `src/kld.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    inspect::tapped,
    play::{cast_owned, settle},
};

#[test]
fn dramatic_reversal_untaps_your_nonland_permanents_only() {
    let mut s = five_color_setup(&["Dramatic Reversal", "Gray Ogre", "Sol Ring"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let ring = s.force_permanent_on_battlefield(0, "Sol Ring");
    let island = s.force_permanent_on_battlefield(0, "Island");
    let foe = s.force_permanent_on_battlefield(1, "Gray Ogre");
    for id in [ogre, ring, island, foe] {
        s.game_mut().state.permanents[id].as_mut().unwrap().tapped = true;
    }
    s.game_mut().scenario_refresh_priority().unwrap();
    cast_owned(&mut s, "Dramatic Reversal");
    settle(&mut s);
    assert!(!tapped(&s, ogre));
    assert!(!tapped(&s, ring));
    assert!(tapped(&s, island));
    assert!(tapped(&s, foe));
}

#[test]
fn noxious_gearhulk_gains_the_destroyed_creatures_toughness_only_if_destroyed() {
    use crate::support::{
        game::until_end_of_turn,
        play::{cast, choose},
    };
    use managym::{
        agent::action::Action,
        state::{
            card::{Keyword, Keywords},
            effect::Modification,
            game_object::Target,
        },
    };
    // (accept, indestructible, expected life)
    for (accept, indestructible, life) in [(true, false, 23), (false, false, 20), (true, true, 20)]
    {
        let mut s = five_color_setup(&["Noxious Gearhulk", "Agent of Treachery"]);
        let victim = s.force_permanent_on_battlefield(1, "Agent of Treachery");
        if indestructible {
            until_end_of_turn(
                &mut s,
                victim,
                Modification::AddKeywords(Keywords::of(&[Keyword::Indestructible])),
            );
        }
        cast(&mut s, "Noxious Gearhulk");
        for _ in 0..10 {
            let space = s.action_space().clone();
            if space
                .actions
                .iter()
                .any(|a| matches!(a, Action::ChooseTarget { .. }))
            {
                assert!(s.choose_target(Target::Permanent(victim)));
            } else if space
                .actions
                .iter()
                .any(|a| matches!(a, Action::PayCost { .. }))
            {
                if accept {
                    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
                } else {
                    choose(&mut s, |a| matches!(a, Action::Decline { .. }));
                }
            } else if s.game().state.stack_objects.is_empty()
                && s.game().state.pending_triggers.is_empty()
            {
                break;
            } else {
                s.pass_priority();
            }
        }
        settle(&mut s);
        let destroyed = accept && !indestructible;
        assert_eq!(s.game().state.permanents[victim].is_none(), destroyed);
        s.assert_life(0, life);
    }
}
