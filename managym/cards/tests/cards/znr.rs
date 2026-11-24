//! Tests for cards filed in `src/znr.rs`, in card order.

use crate::support::{
    game::{five_color_setup, until_end_of_turn},
    inspect::offered_targets,
    play::{cast, settle},
};
use managym::state::{
    card::{Keyword, Keywords},
    effect::Modification,
    game_object::Target,
};

/// Feed the Swarm targets only an opponent's creature or enchantment, and
/// its caster loses life equal to that permanent's mana value as it last
/// existed, or as it still is when it survives.
#[test]
fn feed_the_swarm_loses_life_equal_to_the_targets_mana_value() {
    for indestructible in [false, true] {
        let mut s = five_color_setup(&["Feed the Swarm", "Gray Ogre"]);
        let own = s.force_permanent_on_battlefield(0, "Gray Ogre");
        let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
        let land = s.force_permanent_on_battlefield(1, "Plains");
        if indestructible {
            until_end_of_turn(
                &mut s,
                ogre,
                Modification::AddKeywords(Keywords::of(&[Keyword::Indestructible])),
            );
        }
        cast(&mut s, "Feed the Swarm");
        let offered = offered_targets(&s);
        assert!(offered.contains(&Target::Permanent(ogre)));
        assert!(!offered.contains(&Target::Permanent(own)));
        assert!(!offered.contains(&Target::Permanent(land)));
        assert!(s.choose_target(Target::Permanent(ogre)));
        settle(&mut s);
        assert_eq!(s.game().state.permanents[ogre].is_some(), indestructible);
        s.assert_life(0, 17);
    }
}
