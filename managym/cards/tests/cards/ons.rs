//! Tests for cards filed in `src/ons.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    inspect::{keywords, permanent, tapped},
    play::{cast_owned, settle},
};
use managym::{
    flow::turn::StepKind,
    state::{card::Keyword, game_object::PlayerId},
};

#[test]
fn insurrection_untaps_and_steals_every_creature_with_haste_this_turn() {
    let mut s = five_color_setup(&["Insurrection", "Gray Ogre"]);
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let mine = s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.game_mut().state.permanents[theirs]
        .as_mut()
        .unwrap()
        .tapped = true;
    s.game_mut().scenario_refresh_priority().unwrap();
    cast_owned(&mut s, "Insurrection");
    settle(&mut s);
    assert!(!tapped(&s, theirs));
    assert_eq!(permanent(&s, theirs).controller, PlayerId(0));
    assert_eq!(permanent(&s, mine).controller, PlayerId(0));
    assert!(keywords(&s, theirs).has(Keyword::Haste));
    s.advance_to_active_step(1, StepKind::Upkeep);
    assert_eq!(permanent(&s, theirs).controller, PlayerId(1));
}

#[test]
fn gratuitous_violence_checks_the_departed_firebrands_effective_type_and_controller() {
    use crate::support::{
        game::{move_card, until_end_of_turn},
        inspect::card_of,
        play::{activate, fork_choosing},
    };
    use managym::{
        agent::action::Action,
        state::{card::CardType, effect::Modification, game_object::Target, zone::ZoneType},
    };

    for (controller, creature, damage) in [(0, true, 2), (0, false, 1), (1, true, 1)] {
        let mut s = five_color_setup(&["Gratuitous Violence", "Fanatical Firebrand"]);
        s.force_permanent_on_battlefield(0, "Gratuitous Violence");
        s.advance_to_active_step(controller, StepKind::Main);
        let firebrand = s.force_permanent_on_battlefield(controller, "Fanatical Firebrand");
        let source = card_of(&s, firebrand);
        if !creature {
            until_end_of_turn(
                &mut s,
                firebrand,
                Modification::SetTypes {
                    types: vec![CardType::Artifact],
                    subtypes: vec![],
                },
            );
        }
        activate(&mut s, firebrand, 0);
        assert!(s.choose_target(Target::Player(PlayerId(0))));
        assert_eq!(s.game().object_lki(source).unwrap().was_creature, creature);
        let mut returned = fork_choosing(&s, |a| matches!(a, Action::PassPriority { .. }));
        let new_source = move_card(&mut returned, source, ZoneType::Battlefield);
        assert_ne!(source, new_source);
        let new_permanent = returned
            .game()
            .lookup_current_permanent(new_source)
            .unwrap();
        if creature {
            until_end_of_turn(
                &mut returned,
                new_permanent,
                Modification::SetTypes {
                    types: vec![CardType::Artifact],
                    subtypes: vec![],
                },
            );
        }
        assert_ne!(
            returned.game().characteristics(new_permanent).is_creature(),
            creature
        );
        for branch in [&mut s, &mut returned] {
            settle(branch);
            branch.assert_life(0, 20 - damage);
        }
    }
}

#[test]
fn gratuitous_violence_leaves_a_friendly_lightning_bolt_at_three_damage() {
    use managym::state::game_object::Target;
    let mut s = five_color_setup(&["Gratuitous Violence", "Lightning Bolt", "Wall of Stone"]);
    s.force_permanent_on_battlefield(0, "Gratuitous Violence");
    let wall = s.force_permanent_on_battlefield(0, "Wall of Stone");
    cast_owned(&mut s, "Lightning Bolt");
    assert!(s.choose_target(Target::Permanent(wall)));
    settle(&mut s);
    assert_eq!(permanent(&s, wall).damage, 3);
}

#[test]
fn gratuitous_violence_doubles_a_creatures_entry_damage_to_a_friendly_permanent() {
    use crate::support::play::{cast, pass_until, spend_colorless};
    use managym::{agent::action::ActionSpaceKind, state::game_object::Target};

    let mut s = five_color_setup(&["Gratuitous Violence", "Mongoose Lizard", "Wall of Stone"]);
    s.force_permanent_on_battlefield(0, "Gratuitous Violence");
    let wall = s.force_permanent_on_battlefield(0, "Wall of Stone");
    cast(&mut s, "Mongoose Lizard");
    spend_colorless(&mut s);
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(wall)));
    settle(&mut s);
    assert_eq!(permanent(&s, wall).damage, 2);
}
