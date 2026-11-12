//! Tests for cards filed in `src/sth.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    inspect::tapped,
    play::{enter_and_settle, settle},
};
use managym::flow::turn::StepKind;

#[test]
fn intruder_alarm_stops_untapping_until_a_creature_enters() {
    let mut s = five_color_setup(&["Intruder Alarm", "Gray Ogre"]);
    s.force_permanent_on_battlefield(0, "Intruder Alarm");
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let island = s.force_permanent_on_battlefield(1, "Island");
    for id in [ogre, island] {
        s.game_mut().state.permanents[id].as_mut().unwrap().tapped = true;
    }
    s.game_mut().scenario_refresh_priority().unwrap();
    s.advance_to_active_step(1, StepKind::Main);
    assert!(tapped(&s, ogre), "creatures don't untap");
    assert!(!tapped(&s, island), "lands still untap");
    s.advance_to_active_step(0, StepKind::Main);
    enter_and_settle(&mut s, "Gray Ogre");
    settle(&mut s);
    assert!(!tapped(&s, ogre));
}
