//! Tests for cards filed in `src/drk.rs`, in card order.

/// "At the beginning of the end step, sacrifice this creature": any
/// player's end step, so it leaves on the turn it arrives either way.
#[test]
fn foundations_5f_ball_lightning() {
    use crate::support::game::five_color_setup;
    use crate::support::inspect::{card_of, zone_of};
    use crate::support::play::attack_with;
    use managym::flow::turn::StepKind;
    use managym::state::card::Keyword;
    use managym::state::zone::ZoneType;
    for active in [0, 1] {
        let mut s = five_color_setup(&["Ball Lightning"]);
        s.advance_to_active_step(active, StepKind::Main);
        let ball = s.force_permanent_on_battlefield(0, "Ball Lightning");
        let card = card_of(&s, ball);
        let face = s.game().characteristics(ball);
        assert_eq!(face.pt(), (6, 1));
        assert!(face.keywords.has(Keyword::Haste) && face.keywords.has(Keyword::Trample));
        if active == 0 {
            // Haste: it attacks the turn it arrives.
            s.advance_to_active_step(0, StepKind::DeclareAttackers);
            attack_with(&mut s, ball);
            s.advance_to_active_step(0, StepKind::PostcombatMain);
            s.assert_life(1, 14);
        }
        s.advance_to_active_step(1 - active, StepKind::Upkeep);
        assert_eq!(zone_of(&s, card), Some(ZoneType::Graveyard));
    }
}
