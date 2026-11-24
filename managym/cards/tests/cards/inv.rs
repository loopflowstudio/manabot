//! Tests for cards filed in `src/inv.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    inspect::{library_card, offered_cards, tapped, zone_of},
    play::{activate, select_card, settle},
};
use managym::state::zone::ZoneType;

#[test]
fn captain_sisay_finds_only_a_legendary_card() {
    let mut s = five_color_setup(&["Captain Sisay", "Gray Ogre"]);
    let sisay = s.force_permanent_on_battlefield(0, "Captain Sisay");
    crate::support::game::clear_summoning_sickness(&mut s, sisay);
    s.game_mut().scenario_refresh_priority().unwrap();
    let ogre = library_card(&s, 0, "Gray Ogre");
    let legend = library_card(&s, 0, "Captain Sisay");
    activate(&mut s, sisay, 0);
    for _ in 0..4 {
        if offered_cards(&s).contains(&legend) {
            break;
        }
        s.pass_priority();
    }
    assert!(!offered_cards(&s).contains(&ogre));
    select_card(&mut s, legend);
    settle(&mut s);
    assert_eq!(zone_of(&s, legend), Some(ZoneType::Hand));
    assert!(tapped(&s, sisay));
}

#[test]
fn foundations_5e_opt() {
    use crate::support::game::five_color_setup;
    use crate::support::inspect::{hand, library};
    use crate::support::play::{cast, choose, pass_until};
    use managym::agent::action::{Action, ActionSpaceKind};
    use managym::state::game_object::PlayerId;
    let mut s = five_color_setup(&["Opt"]);
    cast(&mut s, "Opt");
    pass_until(&mut s, ActionSpaceKind::Scry);
    assert_eq!(s.action_space().player, Some(PlayerId(0)));
    let order = library(&s, 0);
    let (top, second) = (order[order.len() - 1], order[order.len() - 2]);
    for bottom in [false, true] {
        let mut fork = s.clone();
        choose(
            &mut fork,
            |a| matches!(a, Action::ScryCard { to_bottom, .. } if *to_bottom == bottom),
        );
        // The draw follows the scry within the same resolution.
        assert!(fork.game().state.stack_objects.is_empty());
        let drawn = if bottom { second } else { top };
        assert!(hand(&fork, 0).contains(&crate::support::inspect::now(&fork, drawn)));
        if bottom {
            assert_eq!(library(&fork, 0)[0], top);
        }
    }
}
