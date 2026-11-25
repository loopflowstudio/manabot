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
