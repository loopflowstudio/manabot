//! Tests for cards filed in `src/lci.rs`, in card order.

/// "Return up to one other target permanent you control to its owner's
/// hand": a land or creature of its controller is offered, never the Pup
/// or an opponent's permanent, and the target may be declined.
#[test]
fn foundations_5f_mischievous_pup() {
    use crate::support::game::five_color_setup;
    use crate::support::inspect::{card_of, library_card, offered_targets, permanent_of, zone_of};
    use crate::support::play::{choose, pass_until, settle};
    use managym::agent::action::{Action, ActionSpaceKind};
    use managym::state::game_object::Target;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Mischievous Pup", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let forest = s.force_permanent_on_battlefield(0, "Forest");
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let pup = library_card(&s, 0, "Mischievous Pup");
    crate::support::game::move_card(&mut s, pup, ZoneType::Battlefield);
    s.game_mut().scenario_refresh_priority().unwrap();
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    let source = permanent_of(&s, pup).unwrap();
    let offers = offered_targets(&s);
    assert!(offers.contains(&Target::Permanent(ogre)));
    assert!(offers.contains(&Target::Permanent(forest)));
    assert!(!offers.contains(&Target::Permanent(theirs)));
    assert!(!offers.contains(&Target::Permanent(source)));
    let mut declined = s.clone();
    choose(&mut declined, |a| matches!(a, Action::Decline { .. }));
    settle(&mut declined);
    assert!(declined.game().state.permanents[ogre].is_some());
    let card = card_of(&s, ogre);
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    assert_eq!(zone_of(&s, card), Some(ZoneType::Hand));
    assert!(s
        .game()
        .characteristics(source)
        .keywords
        .has(managym::state::card::Keyword::Flash));
}
