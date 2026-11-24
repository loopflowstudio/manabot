//! Tests for cards filed in `src/isd.rs`, in card order.

use crate::support::{
    game::{five_color_setup, five_color_unfunded_setup, move_card},
    inspect::{can_cast, owned_card, permanent_of, tapped},
    play::{cast_card, cast_owned, resolve_top, settle},
};
use managym::agent::action::ActionSpaceKind;
use managym::state::zone::ZoneType;

#[test]
fn blasphemous_act_costs_one_less_per_creature_on_the_battlefield() {
    let mut s = five_color_unfunded_setup(&["Blasphemous Act", "Gray Ogre"]);
    let act = owned_card(&s, 0, "Blasphemous Act");
    let act = move_card(&mut s, act, ZoneType::Hand);
    s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.force_permanent_on_battlefield(1, "Gray Ogre");
    // Four creatures: {4}{R} against five mana's worth less one.
    s.set_player_mana_pool(0, "3R");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_cast(&s, act));
    // Either player's creature reduces the cost.
    s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_cast(&s, act));
    cast_card(&mut s, act);
    settle(&mut s);
    assert!(s.battlefield_permanents_named(0, "Gray Ogre").is_empty());
    assert!(s.battlefield_permanents_named(1, "Gray Ogre").is_empty());
    s.assert_zone_size(1, ZoneType::Graveyard, 3);
}

#[test]
fn diregraf_ghoul_enters_tapped() {
    let mut s = five_color_setup(&["Diregraf Ghoul", "Typhoid Rats"]);
    for (name, enters_tapped) in [("Diregraf Ghoul", true), ("Typhoid Rats", false)] {
        let spell = cast_owned(&mut s, name);
        resolve_top(&mut s);
        assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
        let creature = permanent_of(&s, spell).expect("creature spell resolved");
        assert_eq!(tapped(&s, creature), enters_tapped, "{name}");
        assert!(s.game().state.stack_objects.is_empty());
    }
}

/// "This creature can't be blocked by Humans", and a combat-damage counter:
/// a Human blocker is offered no block against it while another creature
/// is; it grows only when its damage reaches the player.
#[test]
fn foundations_5f_stromkirk_noble() {
    use crate::support::game::clear_summoning_sickness;
    use crate::support::inspect::counters;
    use crate::support::play::attack_with;
    use managym::flow::turn::StepKind;
    let mut s = five_color_setup(&["Stromkirk Noble", "Gray Ogre", "Glory Seeker"]);
    let noble = s.force_permanent_on_battlefield(0, "Stromkirk Noble");
    clear_summoning_sickness(&mut s, noble);
    let human = s.force_permanent_on_battlefield(1, "Glory Seeker");
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, noble);
    let blocks = crate::walk_block_declaration(&mut s);
    assert!(blocks.contains(&(human, vec![])), "{blocks:?}");
    assert!(blocks.contains(&(ogre, vec![noble])), "{blocks:?}");
    s.advance_to_active_step(0, StepKind::PostcombatMain);
    s.assert_life(1, 19);
    assert_eq!(counters(&s, noble), 1);
    assert_eq!(s.game().characteristics(noble).pt(), (2, 2));
}

/// "This creature can't block": an untapped Interloper is never asked to
/// block, while its controller's other creature is.
#[test]
fn foundations_5f_vampire_interloper() {
    use crate::support::game::clear_summoning_sickness;
    use crate::support::play::attack_with;
    use managym::flow::turn::StepKind;
    use managym::state::card::Keyword;
    let mut s = five_color_setup(&["Vampire Interloper", "Gray Ogre"]);
    let interloper = s.force_permanent_on_battlefield(1, "Vampire Interloper");
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let attacker = s.force_permanent_on_battlefield(0, "Gray Ogre");
    clear_summoning_sickness(&mut s, attacker);
    assert!(s
        .game()
        .characteristics(interloper)
        .keywords
        .has(Keyword::Flying));
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, attacker);
    let blocks = crate::walk_block_declaration(&mut s);
    assert_eq!(blocks, [(ogre, vec![attacker])]);
}
