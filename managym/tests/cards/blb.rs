//! Tests for cards filed in `src/cardsets/blb.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    inspect::permanent_named,
    play::{cast, settle},
};
use managym::{agent::action::Action, state::game_object::Target};

#[test]
fn savor_makes_food_with_a_real_tap_cost_and_cindermaw_prevents_its_life_gain() {
    for blocked in [false, true] {
        let mut s = five_color_setup(&["Savor", "Gnarled Professor", "Giant Cindermaw"]);
        let target = s.force_permanent_on_battlefield(1, "Gnarled Professor");
        if blocked {
            s.force_permanent_on_battlefield(1, "Giant Cindermaw");
        }
        cast(&mut s, "Savor");
        assert!(s.choose_target(Target::Permanent(target)));
        settle(&mut s);
        assert_eq!(s.game().characteristics(target).toughness, 2);
        let food = permanent_named(&s, "Food");
        s.game_mut().state.permanents[food].as_mut().unwrap().tapped = true;
        s.game_mut().scenario_refresh_priority().unwrap();
        assert!(!s
            .action_space()
            .actions
            .iter()
            .any(|a| matches!(a,Action::ActivateAbility{permanent,..} if *permanent==food)));
        s.game_mut().state.permanents[food].as_mut().unwrap().tapped = false;
        s.game_mut().scenario_refresh_priority().unwrap();
        let index = s
            .action_space()
            .actions
            .iter()
            .position(|a| matches!(a,Action::ActivateAbility{permanent,..} if *permanent==food))
            .unwrap();
        let life = s.life(0);
        s.step_action(index);
        assert!(s.game().state.permanents[food].is_none());
        settle(&mut s);
        assert_eq!(s.life(0), life + if blocked { 0 } else { 3 });
    }
}
