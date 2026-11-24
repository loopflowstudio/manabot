//! Tests for cards filed in `src/dtk.rs`, in card order.

use crate::support::inspect::now;
use crate::support::{
    game::five_color_setup,
    inspect::offered_targets,
    play::{cast, choose, settle, target_opponent},
};
use managym::{
    agent::action::Action,
    state::game_object::{PlayerId, Target},
};

/// One token per creature the target player controls, each a copy under the
/// caster's control; the originals stay with their controller.
#[test]
fn clone_legion_copies_each_creature_target_player_controls() {
    let mut s = five_color_setup(&["Clone Legion", "Gray Ogre", "Agent of Treachery"]);
    s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.force_permanent_on_battlefield(1, "Agent of Treachery");
    s.force_permanent_on_battlefield(0, "Gray Ogre");
    cast(&mut s, "Clone Legion");
    target_opponent(&mut s);
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Gray Ogre").len(), 2);
    let copies = s.battlefield_permanents_named(0, "Agent of Treachery");
    assert_eq!(copies.len(), 1);
    let copy = s.game().state.permanents[copies[0]]
        .as_ref()
        .unwrap()
        .object;
    assert!(!s.game().state[copy].is_card());
    assert_eq!(s.battlefield_permanents_named(1, "Gray Ogre").len(), 1);
    assert_eq!(
        s.battlefield_permanents_named(1, "Agent of Treachery")
            .len(),
        1
    );
}

#[test]
fn rending_volley_targets_white_or_blue_creatures_and_cannot_be_countered() {
    let mut s = five_color_setup(&[
        "Rending Volley",
        "Gray Ogre",
        "Agent of Treachery",
        "Counterspell",
    ]);
    let red = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let blue = s.force_permanent_on_battlefield(1, "Agent of Treachery");
    s.force_card_in_hand(1, "Counterspell");
    let volley = cast(&mut s, "Rending Volley");
    let targets = offered_targets(&s);
    assert!(targets.contains(&Target::Permanent(blue)));
    assert!(!targets.contains(&Target::Permanent(red)));
    assert!(s.choose_target(Target::Permanent(blue)));
    s.pass_priority();
    choose(&mut s, |action| {
        matches!(
            action,
            Action::CastSpell {
                player: PlayerId(1),
                ..
            }
        )
    });
    assert!(s.choose_target(Target::StackSpell(now(&s, volley))));
    settle(&mut s);
    assert!(s.game().state.permanents[blue].is_none());
    assert!(s.game().state.permanents[red].is_some());
}

/// Dragonlord's Servant makes its controller's Dragon spells cost {1} less,
/// and no other spell.
#[test]
fn dragonlords_servant_reduces_only_dragon_spells() {
    use crate::support::{
        game::{five_color_unfunded_setup, force_lands, force_new_card_in_hand},
        inspect::top_spell,
        play::{cast_card, castable, settle},
    };
    use managym::state::player::ManaPayment;
    let mut s = five_color_unfunded_setup(&["Dragonlord's Servant", "Shivan Dragon", "Hill Giant"]);
    force_lands(&mut s, 0, "Mountain", 5);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let dragon = force_new_card_in_hand(&mut s, 0, "Shivan Dragon");
    let giant = force_new_card_in_hand(&mut s, 0, "Hill Giant");
    assert!(
        !castable(&mut s, dragon),
        "{{4}}{{R}}{{R}} with five Mountains"
    );
    s.force_permanent_on_battlefield(0, "Dragonlord's Servant");
    assert!(castable(&mut s, dragon));
    cast_card(&mut s, dragon);
    assert_eq!(top_spell(&s).spent.total(), 5);
    settle(&mut s);
    // The {3}{R} Giant gets no reduction.
    for land in s.battlefield_permanents_named(0, "Mountain") {
        s.game_mut().state.permanents[land].as_mut().unwrap().tapped = false;
    }
    s.game_mut().scenario_refresh_priority().unwrap();
    cast_card(&mut s, giant);
    assert_eq!(top_spell(&s).spent.total(), 4);
}

#[test]
fn impact_tremors_triggers_for_each_goblin() {
    use crate::support::{
        inspect::card_of,
        play::{cast_owned, pass_until, resolve_top, spend_colorless},
    };
    use managym::agent::action::ActionSpaceKind;
    use managym::state::{
        mana::{Color, Colors},
        stack_object::StackObject,
    };
    use std::collections::HashSet;

    for (doubling_season, count) in [(false, 2), (true, 4)] {
        let mut s = five_color_setup(&["Impact Tremors", "Dragon Fodder", "Doubling Season"]);
        let tremors = s.force_permanent_on_battlefield(0, "Impact Tremors");
        s.set_player_mana_pool(0, "1R");
        if doubling_season {
            s.force_permanent_on_battlefield(0, "Doubling Season");
        }
        cast_owned(&mut s, "Dragon Fodder");
        spend_colorless(&mut s);
        resolve_top(&mut s);
        pass_until(&mut s, ActionSpaceKind::Priority);
        let goblins = s.battlefield_permanents_named(0, "Goblin");
        assert_eq!(goblins.len(), count);
        assert_eq!(goblins.iter().collect::<HashSet<_>>().len(), count);
        for goblin in goblins {
            let face = s.game().characteristics(goblin);
            assert_eq!(face.pt(), (1, 1));
            assert_eq!(face.colors, Colors::from_iter([Color::Red]));
            assert_eq!(face.creature_types, ["Goblin"]);
        }
        s.assert_life(1, 20);
        let stack = s.game().state.stack_objects.clone();
        assert_eq!(stack.len(), count);
        assert!(stack
            .iter()
            .all(|item| matches!(item, StackObject::TriggeredAbility(_))
                && item.source_object() == card_of(&s, tremors)));
        let mut fork = s.clone();
        resolve_top(&mut fork);
        fork.assert_life(1, 19);
        assert_eq!(fork.game().state.stack_objects, stack[..count - 1]);
        s.assert_life(1, 20);
        assert_eq!(s.game().state.stack_objects, stack);
        resolve_top(&mut s);
        s.assert_life(1, 19);
        assert_eq!(
            s.game().state.stack_objects,
            fork.game().state.stack_objects
        );
        for branch in [&mut s, &mut fork] {
            settle(branch);
            branch.assert_life(1, 20 - count as i32);
        }
    }
}

#[test]
fn foundations_5e_seismic_rupture() {
    use crate::support::game::{five_color_setup, until_end_of_turn};
    use crate::support::inspect::{card_of, permanent, zone_of};
    use crate::support::play::{cast, settle};
    use managym::state::card::{Keyword, Keywords};
    use managym::state::effect::Modification;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Seismic Rupture", "Gray Ogre", "Hill Giant", "Wind Drake"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let giant = s.force_permanent_on_battlefield(1, "Hill Giant");
    let drake = s.force_permanent_on_battlefield(1, "Wind Drake");
    // A creature that gained flying is a creature with flying.
    let lifted = s.force_permanent_on_battlefield(1, "Gray Ogre");
    until_end_of_turn(
        &mut s,
        lifted,
        Modification::AddKeywords(Keywords::of(&[Keyword::Flying])),
    );
    let ogre_card = card_of(&s, ogre);
    cast(&mut s, "Seismic Rupture");
    settle(&mut s);
    assert_eq!(zone_of(&s, ogre_card), Some(ZoneType::Graveyard));
    assert_eq!(permanent(&s, giant).damage, 2);
    assert_eq!(permanent(&s, drake).damage, 0);
    assert_eq!(permanent(&s, lifted).damage, 0);
}
