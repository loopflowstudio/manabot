//! Tests for cards filed in `src/thb.rs`, in card order.

#[test]
fn nessian_hornbeetle_excludes_itself_and_rechecks_other_power() {
    use crate::support::game::five_color_setup;
    use crate::support::game::move_card;
    use crate::support::inspect::card_of;
    use crate::support::play::settle;
    use crate::support::{game::set_counters, inspect::counters};
    use managym::flow::turn::StepKind;
    use managym::state::permanent::CounterKind;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Nessian Hornbeetle", "Gray Ogre"]);
    let beetle = s.force_permanent_on_battlefield(0, "Nessian Hornbeetle");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    set_counters(&mut s, beetle, CounterKind::PlusOnePlusOne, 2);
    let root = s.game().clone();
    s.advance_to_active_step(0, StepKind::BeginningOfCombat);
    assert!(s.game().state.stack_objects.is_empty());
    *s.game_mut() = root;
    set_counters(&mut s, ogre, CounterKind::PlusOnePlusOne, 2);
    s.advance_to_active_step(0, StepKind::BeginningOfCombat);
    assert_eq!(s.game().state.stack_objects.len(), 1);
    let pending = s.game().clone();
    for stays in [true, false] {
        *s.game_mut() = pending.clone();
        if !stays {
            let obj = card_of(&s, ogre);
            move_card(&mut s, obj, ZoneType::Hand);
        }
        settle(&mut s);
        assert_eq!(counters(&s, beetle), 2 + i32::from(stays));
    }
}

#[test]
fn foundations_permanent_soul_guide_lantern() {
    use crate::support::game::{five_color_setup, plant_in_graveyard};
    use crate::support::inspect::{card_of, permanent_named, zone_of};
    use crate::support::play::{activate, cast, pass_until, settle, spend_colorless};
    use managym::agent::action::ActionSpaceKind;
    use managym::state::game_object::Target;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Soul-Guide Lantern", "Giant Growth"]);
    let card = plant_in_graveyard(&mut s, 0, "Giant Growth");
    cast(&mut s, "Soul-Guide Lantern");
    spend_colorless(&mut s);
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::GraveyardCard(card)));
    settle(&mut s);
    assert_eq!(zone_of(&s, card), Some(ZoneType::Exile));
    let lantern = permanent_named(&s, "Soul-Guide Lantern");
    let source = card_of(&s, lantern);
    let own = plant_in_graveyard(&mut s, 0, "Forest");
    let other = plant_in_graveyard(&mut s, 1, "Island");
    let mut fork = s.clone();
    activate(&mut s, lantern, 0);
    assert_eq!(zone_of(&s, source), Some(ZoneType::Graveyard));
    settle(&mut s);
    assert_eq!(zone_of(&s, other), Some(ZoneType::Exile));
    assert_eq!(zone_of(&s, own), Some(ZoneType::Graveyard));
    let hand = fork.zone_size(0, ZoneType::Hand);
    fork.set_player_mana_pool(0, "1");
    activate(&mut fork, lantern, 1);
    assert_eq!(zone_of(&fork, source), Some(ZoneType::Graveyard));
    settle(&mut fork);
    assert_eq!(fork.zone_size(0, ZoneType::Hand), hand + 1);
    assert_eq!(zone_of(&fork, other), Some(ZoneType::Graveyard));
}

/// "Creatures you control gain double strike until end of turn": the
/// controller's creatures, the Dragon included, for this turn only.
#[test]
fn foundations_5f_terror_of_mount_velus() {
    use crate::support::game::five_color_setup;
    use crate::support::inspect::{keywords, permanent_of};
    use crate::support::play::enter_and_settle;
    use managym::flow::turn::StepKind;
    use managym::state::card::Keyword;
    let mut s = five_color_setup(&["Terror of Mount Velus", "Gray Ogre"]);
    let ours = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let card = enter_and_settle(&mut s, "Terror of Mount Velus");
    let terror = permanent_of(&s, card).unwrap();
    assert!(keywords(&s, ours).has(Keyword::DoubleStrike));
    assert!(!keywords(&s, theirs).has(Keyword::DoubleStrike));
    s.advance_to_active_step(1, StepKind::Main);
    assert!(!keywords(&s, ours).has(Keyword::DoubleStrike));
    assert!(keywords(&s, terror).has(Keyword::DoubleStrike));
    assert!(keywords(&s, terror).has(Keyword::Flying));
}
