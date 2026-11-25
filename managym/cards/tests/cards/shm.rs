//! Tests for cards filed in `src/shm.rs`, in card order.

use crate::support::{
    game::{clear_summoning_sickness, five_color_setup},
    play::{activate, settle},
};

/// The first ability makes a green-white Elf Warrior; the second copies each
/// creature token its controller has, and only tokens.
#[test]
fn rhys_the_redeemed_makes_elf_warriors_and_copies_each_creature_token() {
    let mut s = five_color_setup(&["Rhys the Redeemed", "Gray Ogre"]);
    let rhys = s.force_permanent_on_battlefield(0, "Rhys the Redeemed");
    s.force_permanent_on_battlefield(0, "Gray Ogre");
    clear_summoning_sickness(&mut s, rhys);
    activate(&mut s, rhys, 0);
    settle(&mut s);
    let elves = s.battlefield_permanents_named(0, "Elf Warrior");
    assert_eq!(elves.len(), 1);
    let elf = s.game().state.permanents[elves[0]].as_ref().unwrap().object;
    let colors = s.game().characteristics(elves[0]).colors.len();
    assert!(!s.game().state[elf].is_card());
    assert_eq!(colors, 2);

    s.game_mut().state.permanents[rhys].as_mut().unwrap().tapped = false;
    activate(&mut s, rhys, 1);
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Elf Warrior").len(), 2);
    assert_eq!(s.battlefield_permanents_named(0, "Gray Ogre").len(), 1);
    assert_eq!(
        s.battlefield_permanents_named(0, "Rhys the Redeemed").len(),
        1
    );
}

#[test]
fn wilt_leaf_liege_enters_on_opposing_discard_and_both_anthems_apply() {
    use crate::support::{
        game::move_from_library,
        inspect::zone_of,
        play::{cast, pass_until, select_card},
    };
    use managym::{agent::action::ActionSpaceKind, state::zone::ZoneType};
    let mut s = five_color_setup(&[
        "Wilt-Leaf Liege",
        "Tinybones, Bauble Burglar",
        "Dryad Militant",
        "Seize the Spoils",
    ]);
    s.game_mut().mana_payment[0] = managym::state::player::ManaPayment::Auto;
    let tiny = s.force_permanent_on_battlefield(0, "Tinybones, Bauble Burglar");
    clear_summoning_sickness(&mut s, tiny);
    let dryad = s.force_permanent_on_battlefield(1, "Dryad Militant");
    let liege = move_from_library(&mut s, 1, "Wilt-Leaf Liege", ZoneType::Hand);
    activate(&mut s, tiny, 0);
    pass_until(&mut s, ActionSpaceKind::Discard);
    select_card(&mut s, liege);
    assert_eq!(
        s.game().state.stack_objects.len(),
        1,
        "Tinybones triggered on the discard"
    );
    settle(&mut s);
    assert_eq!(zone_of(&s, liege), Some(ZoneType::Battlefield));
    assert_eq!(s.game().characteristics(dryad).pt(), (4, 3));

    let mine = crate::support::game::move_to_hand(&mut s, "Wilt-Leaf Liege");
    cast(&mut s, "Seize the Spoils");
    select_card(&mut s, mine);
    settle(&mut s);
    assert_eq!(zone_of(&s, mine), Some(ZoneType::Graveyard));
}

#[test]
fn wilt_leaf_stays_in_the_graveyard_after_looting_or_cleanup() {
    use crate::support::{
        game::move_to_hand,
        inspect::zone_of,
        play::{cast, pass_until, select_card},
    };
    use managym::{agent::action::ActionSpaceKind, state::zone::ZoneType};
    for cleanup in [false, true] {
        let mut s = five_color_setup(&["Wilt-Leaf Liege", "Dreadwing Scavenger"]);
        let liege = move_to_hand(&mut s, "Wilt-Leaf Liege");
        if cleanup {
            for _ in 0..7 {
                move_to_hand(&mut s, "Forest");
            }
            s.advance_to_active_step(0, managym::flow::turn::StepKind::Cleanup);
        } else {
            s.game_mut().mana_payment[0] = managym::state::player::ManaPayment::Auto;
            cast(&mut s, "Dreadwing Scavenger");
            pass_until(&mut s, ActionSpaceKind::Discard);
        }
        select_card(&mut s, liege);
        assert_eq!(zone_of(&s, liege), Some(ZoneType::Graveyard));
    }
}

#[test]
fn wilt_leaf_enters_when_paying_an_opponents_ward_discard() {
    use crate::support::{
        game::move_to_hand,
        inspect::{permanent, zone_of},
        play::{cast, select_card},
    };
    use managym::state::{game_object::Target, zone::ZoneType};
    let mut s = five_color_setup(&["Wilt-Leaf Liege", "Scavenger Regent", "Lightning Bolt"]);
    let dragon = s.force_permanent_on_battlefield(1, "Scavenger Regent");
    let liege = move_to_hand(&mut s, "Wilt-Leaf Liege");
    cast(&mut s, "Lightning Bolt");
    assert!(s.choose_target(Target::Permanent(dragon)));
    s.pass_priority();
    s.pass_priority();
    select_card(&mut s, liege);
    settle(&mut s);
    assert_eq!(zone_of(&s, liege), Some(ZoneType::Battlefield));
    assert_eq!(permanent(&s, dragon).damage, 3);
}
