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
