//! Zendikar Rising behavior.
#[test]
fn kicked_nullpriest_returns_only_creatures() {
    super::check_kicked_return("Nullpriest of Oblivion", "4BB", "Gray Ogre", "Sol Ring");
}

/// The counters stay; the animation is optional and lasts the turn, and a
/// Barrens that entered this turn can't attack (CR 302.6).
#[test]
fn crawling_barrens_keeps_its_counters_and_animates_only_when_chosen() {
    use crate::support::{
        game::five_color_setup,
        inspect::{counters, permanent_of},
        play::{activate, choose, enter_and_settle, settle, spend_colorless},
    };
    use managym::{
        agent::action::{Action, ActionSpaceKind},
        flow::turn::StepKind,
    };
    let resolve_to_choice = |s: &mut crate::support::game::Scenario| {
        while !s
            .action_space()
            .actions
            .iter()
            .any(|a| matches!(a, Action::Decline { .. }))
        {
            s.pass_priority();
        }
    };
    let mut s = five_color_setup(&["Crawling Barrens"]);
    let object = enter_and_settle(&mut s, "Crawling Barrens");
    let barrens = permanent_of(&s, object).unwrap();
    activate(&mut s, barrens, 0);
    spend_colorless(&mut s);
    resolve_to_choice(&mut s);
    choose(&mut s, |a| matches!(a, Action::Decline { .. }));
    settle(&mut s);
    assert_eq!(counters(&s, barrens), 2);
    assert!(!s.game().characteristics(barrens).is_creature());

    activate(&mut s, barrens, 0);
    spend_colorless(&mut s);
    resolve_to_choice(&mut s);
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    settle(&mut s);
    let view = s.game().characteristics(barrens);
    assert!(view.is_creature() && view.is_land() && view.has_subtype("Elemental"));
    assert_eq!(view.pt(), (4, 4));
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    assert_ne!(s.action_space().kind, ActionSpaceKind::DeclareAttacker);

    s.advance_to_active_step(1, StepKind::Upkeep);
    assert!(!s.game().characteristics(barrens).is_creature());
    assert_eq!(counters(&s, barrens), 4);
}
