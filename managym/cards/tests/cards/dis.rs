//! Dissension card behavior.

#[test]
fn foundations_5d_macabre_waltz() {
    use crate::support::game::{five_color_setup, move_from_library};
    use crate::support::inspect::{offered_cards, offered_targets, zone_of};
    use crate::support::play::{cast, choose, pass_until, select_card, settle, spend_colorless};
    use managym::agent::action::{Action, ActionSpaceKind};
    use managym::state::game_object::{PlayerId, Target};
    use managym::state::zone::ZoneType;

    let mut s = five_color_setup(&["Macabre Waltz", "Gray Ogre", "Hill Giant"]);
    let one = move_from_library(&mut s, 0, "Gray Ogre", ZoneType::Graveyard);
    let two = move_from_library(&mut s, 0, "Hill Giant", ZoneType::Graveyard);
    let enemy = move_from_library(&mut s, 1, "Gray Ogre", ZoneType::Graveyard);
    let land = move_from_library(&mut s, 0, "Forest", ZoneType::Hand);
    cast(&mut s, "Macabre Waltz");
    assert!(!offered_targets(&s).contains(&Target::GraveyardCard(enemy)));
    for n in 0..=2 {
        let mut branch = s.clone();
        for id in [one, two].into_iter().take(n) {
            assert!(branch.choose_target(Target::GraveyardCard(id)));
        }
        if n < 2 {
            choose(&mut branch, |a| matches!(a, Action::Decline { .. }));
        }
        spend_colorless(&mut branch);
        pass_until(&mut branch, ActionSpaceKind::Discard);
        assert_eq!(branch.action_space().player, Some(PlayerId(0)));
        assert_eq!(offered_cards(&branch).len(), 1 + n);
        select_card(&mut branch, land);
        settle(&mut branch);
        for (i, id) in [one, two].into_iter().enumerate() {
            assert_eq!(
                zone_of(&branch, id),
                Some(if i < n {
                    ZoneType::Hand
                } else {
                    ZoneType::Graveyard
                })
            );
        }
        assert_eq!(zone_of(&branch, land), Some(ZoneType::Graveyard));
    }
}
