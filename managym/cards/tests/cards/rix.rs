//! Tests for cards filed in `src/rix.rs`, in card order.

/// Fanatical Firebrand has haste, so it pays its own {T} the turn it arrives,
/// and is sacrificed to deal 1 damage.
#[test]
fn fanatical_firebrand_taps_and_sacrifices_itself_for_one_damage() {
    use crate::support::{
        game::five_color_unfunded_setup,
        inspect::{card_of, zone_of},
        play::{activate, settle},
    };
    use managym::state::{
        game_object::{PlayerId, Target},
        zone::ZoneType,
    };
    let mut s = five_color_unfunded_setup(&["Fanatical Firebrand"]);
    let firebrand = s.force_permanent_on_battlefield(0, "Fanatical Firebrand");
    s.game_mut().state.permanents[firebrand]
        .as_mut()
        .unwrap()
        .summoning_sick = true;
    let firebrand_card = card_of(&s, firebrand);
    activate(&mut s, firebrand, 0);
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    assert_eq!(zone_of(&s, firebrand_card), Some(ZoneType::Graveyard));
    settle(&mut s);
    s.assert_life(1, 19);
}

/// Thrashing Brontodon pays {1} and itself to destroy an artifact or
/// enchantment, never a creature.
#[test]
fn thrashing_brontodon_sacrifices_itself_to_destroy_an_artifact() {
    crate::support::play::sacrifices_itself_to_destroy_an_artifact("Thrashing Brontodon", "Forest");
}

#[test]
fn token_producer_gleaming_barrier() {
    use crate::support::{game::five_color_setup, play::enter_and_settle};
    let mut s = five_color_setup(&["Gleaming Barrier"]);
    let barrier = enter_and_settle(&mut s, "Gleaming Barrier");
    assert!(s.battlefield_permanents_named(0, "Treasure").is_empty());
    crate::support::play::move_and_settle(
        &mut s,
        barrier,
        managym::state::zone::ZoneType::Graveyard,
    );
    assert_eq!(s.battlefield_permanents_named(0, "Treasure").len(), 1);
}

#[test]
fn printed_creature_swab_goblin() {
    crate::printed_creature_combat("Swab Goblin", 2, 2, &[]);
}

#[test]
fn printed_creature_zetalpa_primal_dawn() {
    use managym::state::card::Keyword;
    crate::printed_creature_combat(
        "Zetalpa, Primal Dawn",
        4,
        8,
        &[
            Keyword::Flying,
            Keyword::Indestructible,
            Keyword::Vigilance,
            Keyword::Trample,
            Keyword::DoubleStrike,
        ],
    );
}

#[test]
fn foundations_5e_moment_of_craving() {
    crate::assert_targeted_pump("Moment of Craving", "Hill Giant", 1, (1, 1), &[], 2, 0);
}

#[test]
fn foundations_5e_moment_of_triumph() {
    crate::assert_targeted_pump("Moment of Triumph", "Gray Ogre", 0, (4, 4), &[], 2, 0);
}

/// "This creature has flying as long as it's attacking": flying from the
/// declaration until combat ends.
#[test]
fn foundations_5f_kitesail_corsair() {
    use crate::support::{
        game::{clear_summoning_sickness, five_color_setup},
        inspect::keywords,
        play::attack_with,
    };
    use managym::{flow::turn::StepKind, state::card::Keyword};
    let mut s = five_color_setup(&["Kitesail Corsair"]);
    let corsair = s.force_permanent_on_battlefield(0, "Kitesail Corsair");
    clear_summoning_sickness(&mut s, corsair);
    assert!(!keywords(&s, corsair).has(Keyword::Flying));
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, corsair);
    assert!(keywords(&s, corsair).has(Keyword::Flying));
    s.advance_to_active_step(0, StepKind::PostcombatMain);
    assert!(!keywords(&s, corsair).has(Keyword::Flying));
}

#[test]
fn etali_casts_exiled_spells_in_either_order_or_stops() {
    use crate::support::{
        game::{five_color_unfunded_setup, force_lands, move_from_library, pin_top},
        inspect::{library_card, now, top_spell, zone_of},
        play::{attack_with, choose},
    };
    use managym::{
        agent::action::{Action, ActionSpaceKind},
        flow::turn::StepKind,
        state::{
            card::SpellProfile,
            game_object::{PlayerId, Target},
            zone::ZoneType,
        },
    };
    for (bite_first, stop) in [(true, false), (false, false), (true, true)] {
        let mut s = five_color_unfunded_setup(&[
            "Etali, Primal Storm",
            "Twinmaw Stormbrood",
            "Gray Ogre",
            "Counterspell",
        ]);
        let counter = move_from_library(&mut s, 1, "Counterspell", ZoneType::Hand);
        force_lands(&mut s, 1, "Island", 2);
        let etali = s.force_permanent_on_battlefield(0, "Etali, Primal Storm");
        s.game_mut().state.permanents[etali]
            .as_mut()
            .unwrap()
            .summoning_sick = false;
        let dragon = pin_top(&mut s, &["Twinmaw Stormbrood"])[0];
        let ogre = library_card(&s, 1, "Gray Ogre");
        s.game_mut()
            .state
            .zones
            .reposition(ZoneType::Library, PlayerId(1), ogre, usize::MAX);
        s.advance_to_active_step(0, StepKind::DeclareAttackers);
        attack_with(&mut s, etali);
        s.pass_priority();
        s.pass_priority();
        let (dragon, ogre) = (now(&s, dragon), now(&s, ogre));
        assert_eq!(s.action_space().actions.len(), 4);
        for bite in [bite_first, !bite_first] {
            if stop && !bite {
                choose(&mut s, |a| matches!(a, Action::Decline { .. }));
                assert_eq!(zone_of(&s, ogre), Some(ZoneType::Exile));
                break;
            }
            assert_eq!(s.action_space().kind, ActionSpaceKind::LookAndSelect);
            if bite {
                choose(
                    &mut s,
                    |a| matches!(a, Action::CastSpell { object, profile: SpellProfile::Omen, .. } if *object == dragon),
                );
                assert!(s.choose_target(Target::Permanent(etali)));
            } else {
                choose(
                    &mut s,
                    |a| matches!(a, Action::SelectCard { object, .. } if *object == ogre),
                );
            }
            assert_eq!(top_spell(&s).controller, PlayerId(0));
            assert!(top_spell(&s).spent.is_empty());
            assert_eq!(
                top_spell(&s).profile,
                if bite {
                    SpellProfile::Omen
                } else {
                    SpellProfile::Front
                }
            );
        }
        assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
        assert_eq!(s.game().state.stack_objects.len(), if stop { 1 } else { 2 });
        if stop {
            s.pass_priority();
            choose(
                &mut s,
                |a| matches!(a, Action::CastSpell { object, .. } if *object == counter),
            );
            let bite = now(&s, dragon);
            assert!(s.choose_target(Target::StackSpell(bite)));
            while !s.game().state.stack_objects.is_empty() {
                assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
                s.pass_priority();
            }
            assert_eq!(zone_of(&s, dragon), Some(ZoneType::Graveyard));
            assert_eq!(zone_of(&s, ogre), Some(ZoneType::Exile));
            assert!(s.game().state.cast_caller.is_none());
        }
    }
}
