//! March of the Machine card behavior.

#[test]
fn foundations_5d_wary_thespian() {
    use crate::support::game::{five_color_setup, move_card, put_onto_battlefield};
    use crate::support::inspect::card_of;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Wary Thespian"]);
    let id = put_onto_battlefield(&mut s, 0, "Wary Thespian");
    crate::assert_surveil_one(&mut s);
    let card = card_of(&s, id);
    move_card(&mut s, card, ZoneType::Graveyard);
    s.game_mut().scenario_refresh_priority().unwrap();
    crate::assert_surveil_one(&mut s);
}

/// "Elspeth's Smite deals 3 damage to target attacking or blocking creature.
/// If that creature would die this turn, exile it instead." Only creatures
/// in combat are offered. A blocker it kills is exiled; an attacker that
/// survives it and then dies to combat damage is exiled; one that survives
/// the turn goes to the graveyard as usual.
#[test]
fn foundations_5g_elspeths_smite() {
    use crate::support::{
        game::{clear_summoning_sickness, five_color_setup, move_card, Scenario},
        inspect::{card_of, offered_targets, permanent_of, zone_of},
        play::{attack_with, cast, settle},
    };
    use managym::{
        agent::action::ActionSpaceKind,
        flow::turn::StepKind,
        state::{
            game_object::{PermanentId, Target},
            zone::ZoneType,
        },
    };
    // Player 0's `attacker` attacks; player 1's Gray Ogre blocks it when `block`.
    let combat =
        |attacker: &str, block: bool| -> (Scenario, PermanentId, PermanentId, PermanentId) {
            let mut s = five_color_setup(&["Elspeth's Smite", "Craw Wurm", "Gray Ogre"]);
            let attacking = s.force_permanent_on_battlefield(0, attacker);
            let idle = s.force_permanent_on_battlefield(0, "Gray Ogre");
            let blocker = s.force_permanent_on_battlefield(1, "Gray Ogre");
            clear_summoning_sickness(&mut s, attacking);
            s.advance_to_active_step(0, StepKind::DeclareAttackers);
            attack_with(&mut s, attacking);
            s.advance_to_active_step(0, StepKind::DeclareBlockers);
            if block {
                s.declare_block();
            } else {
                s.decline_block();
            }
            assert_ne!(s.action_space().kind, ActionSpaceKind::DeclareBlocker);
            s.set_player_mana_pool(0, "W");
            cast(&mut s, "Elspeth's Smite");
            (s, attacking, idle, blocker)
        };

    let (mut s, attacking, idle, blocker) = combat("Gray Ogre", true);
    let offers = offered_targets(&s);
    assert!(offers.contains(&Target::Permanent(attacking)));
    assert!(offers.contains(&Target::Permanent(blocker)));
    assert!(!offers.contains(&Target::Permanent(idle)));
    let card = card_of(&s, blocker);
    assert!(s.choose_target(Target::Permanent(blocker)));
    settle(&mut s);
    assert_eq!(zone_of(&s, card), Some(ZoneType::Exile));

    // The Wurm survives 3 damage, then dies to the Ogre's 2.
    let (mut s, wurm, _, _) = combat("Craw Wurm", true);
    let card = card_of(&s, wurm);
    assert!(s.choose_target(Target::Permanent(wurm)));
    settle(&mut s);
    assert_eq!(zone_of(&s, card), Some(ZoneType::Battlefield));
    s.advance_to_active_step(0, StepKind::EndOfCombat);
    assert_eq!(zone_of(&s, card), Some(ZoneType::Exile));

    // Unblocked, it survives the turn and the replacement ends at cleanup.
    let (mut s, wurm, _, _) = combat("Craw Wurm", false);
    let card = card_of(&s, wurm);
    assert!(s.choose_target(Target::Permanent(wurm)));
    settle(&mut s);
    assert!(s.game().exiled_if_it_would_die(wurm));
    s.advance_to_active_step(1, StepKind::Upkeep);
    assert!(!s
        .game()
        .exiled_if_it_would_die(permanent_of(&s, card).unwrap()));
    let card = move_card(&mut s, card, ZoneType::Graveyard);
    assert_eq!(zone_of(&s, card), Some(ZoneType::Graveyard));
}
