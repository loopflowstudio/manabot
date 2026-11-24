//! Magic 2015 Promos.

#[test]
fn foundations_5d_reclamation_sage() {
    use crate::support::game::{five_color_setup, put_onto_battlefield};
    use crate::support::inspect::{card_of, offered_targets, zone_of};
    use crate::support::play::{choose_mode, pass_until, settle};
    use managym::agent::action::ActionSpaceKind;
    use managym::state::game_object::{PlayerId, Target};
    use managym::state::zone::ZoneType;

    let mut s = five_color_setup(&[
        "Reclamation Sage",
        "Sol Ring",
        "Anthem of Champions",
        "Gray Ogre",
    ]);
    let ring = s.force_permanent_on_battlefield(0, "Sol Ring");
    let anthem = s.force_permanent_on_battlefield(1, "Anthem of Champions");
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    put_onto_battlefield(&mut s, 0, "Reclamation Sage");
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    let offers = offered_targets(&s);
    assert!(!offers.contains(&Target::Permanent(ogre)));
    for id in [ring, anthem] {
        let mut branch = s.clone();
        assert!(branch.choose_target(Target::Permanent(id)));
        let card = card_of(&branch, id);
        assert!(!branch.game().state.stack_objects.is_empty());
        pass_until(&mut branch, ActionSpaceKind::Modal);
        assert_eq!(branch.action_space().player, Some(PlayerId(0)));
        for (mode, dest) in [(0, ZoneType::Graveyard), (1, ZoneType::Battlefield)] {
            let mut fork = branch.clone();
            choose_mode(&mut fork, mode);
            settle(&mut fork);
            assert_eq!(zone_of(&fork, card), Some(dest));
        }
    }
}
