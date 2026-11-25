//! Apocalypse card behavior.

#[test]
fn foundations_permanent_phyrexian_arena() {
    use crate::support::game::five_color_setup;
    use managym::flow::turn::StepKind;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Phyrexian Arena"]);
    s.force_permanent_on_battlefield(0, "Phyrexian Arena");
    s.advance_to_active_step(1, StepKind::Main);
    s.assert_life(0, 20);
    let hand = s.zone_size(0, ZoneType::Hand);
    s.advance_to_active_step(0, StepKind::Main);
    assert_eq!(s.zone_size(0, ZoneType::Hand), hand + 2);
    s.assert_life(0, 19);
}
