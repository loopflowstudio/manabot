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

/// Inspiring Call draws one card for each creature its caster controls with
/// a +1/+1 counter and makes exactly those creatures indestructible.
#[test]
fn inspiring_call_draws_for_and_protects_creatures_with_counters() {
    use crate::support::{
        game::{five_color_setup, set_counters},
        inspect::{hand, keywords},
        play::{cast, settle},
    };
    use managym::state::{card::Keyword, permanent::CounterKind};
    let mut s = five_color_setup(&["Inspiring Call", "Gray Ogre"]);
    let first = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let second = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let plain = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    for id in [first, second, theirs] {
        set_counters(&mut s, id, CounterKind::PlusOnePlusOne, 1);
    }
    let before = hand(&s, 0).len();
    cast(&mut s, "Inspiring Call");
    settle(&mut s);
    assert_eq!(hand(&s, 0).len(), before + 2);
    for (id, protected) in [
        (first, true),
        (second, true),
        (plain, false),
        (theirs, false),
    ] {
        assert_eq!(keywords(&s, id).has(Keyword::Indestructible), protected);
    }
}

#[test]
fn surrak_sums_signed_power_and_rechecks_after_targeting() {
    use crate::support::{
        game::{set_counters, until_end_of_turn},
        inspect::offered_targets,
    };
    use managym::flow::turn::StepKind;
    use managym::state::card::Keyword;
    use managym::state::{effect::Modification, game_object::Target, permanent::CounterKind};
    let name = "Surrak, the Hunt Caller";
    let mut s = five_color_setup(&[name, "Hill Giant", "Gray Ogre"]);
    let surrak = s.force_permanent_on_battlefield(0, name);
    let giant = s.force_permanent_on_battlefield(0, "Hill Giant");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    until_end_of_turn(
        &mut s,
        ogre,
        Modification::ModifyPt {
            power: -3,
            toughness: 0,
        },
    );
    let root = s.game().clone(); // 5 + 3 - 1
    s.advance_to_active_step(0, StepKind::BeginningOfCombat);
    assert!(s.game().state.stack_objects.is_empty());
    *s.game_mut() = root;
    set_counters(&mut s, giant, CounterKind::PlusOnePlusOne, 1);
    s.advance_to_active_step(0, StepKind::BeginningOfCombat);
    let offered = offered_targets(&s);
    assert_eq!(offered.len(), 3);
    assert!([surrak, giant, ogre]
        .iter()
        .all(|&id| offered.contains(&Target::Permanent(id))));
    s.choose_target_named("Gray Ogre");
    let pending = s.game().clone();
    for meets in [false, true] {
        *s.game_mut() = pending.clone();
        if !meets {
            set_counters(&mut s, giant, CounterKind::PlusOnePlusOne, 0);
        }
        settle(&mut s);
        assert_eq!(
            s.game().characteristics(ogre).keywords.has(Keyword::Haste),
            meets
        );
    }
}
