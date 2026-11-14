//! Tests for cards filed in `src/gpt.rs`, in card order.

use managym::{
    agent::action::{Action, ActionSpaceKind},
    state::game_object::Target,
};

use crate::support::{
    game::five_color_setup,
    play::{cast, choose_mode, pass_until, settle},
};

/// The replicate count is announced like X and limited by red mana (four
/// red: the spell and up to three payments). Paying twice copies the spell
/// twice, each copy taking a new artifact target.
#[test]
fn shattering_spree_replicated_twice_destroys_three_artifacts() {
    let mut s = five_color_setup(&["Shattering Spree", "Sundial of the Infinite"]);
    let sundials: Vec<_> = (0..4)
        .map(|_| s.force_permanent_on_battlefield(1, "Sundial of the Infinite"))
        .collect();
    cast(&mut s, "Shattering Spree");
    let counts: Vec<usize> = s
        .action_space()
        .actions
        .iter()
        .filter_map(|action| match action {
            Action::ChooseMode { mode, .. } => Some(*mode),
            _ => None,
        })
        .collect();
    assert_eq!(counts, vec![0, 1, 2, 3]);
    choose_mode(&mut s, 2);
    assert!(s.choose_target(Target::Permanent(sundials[0])));
    for sundial in &sundials[1..3] {
        pass_until(&mut s, ActionSpaceKind::ChooseTarget);
        assert!(s.choose_target(Target::Permanent(*sundial)), "a new target");
    }
    settle(&mut s);
    assert_eq!(
        s.battlefield_permanents_named(1, "Sundial of the Infinite"),
        vec![sundials[3]]
    );
}

/// Without a replicate payment the spell is not copied.
#[test]
fn shattering_spree_unreplicated_destroys_one_artifact() {
    let mut s = five_color_setup(&["Shattering Spree", "Sundial of the Infinite"]);
    let first = s.force_permanent_on_battlefield(1, "Sundial of the Infinite");
    let second = s.force_permanent_on_battlefield(1, "Sundial of the Infinite");
    cast(&mut s, "Shattering Spree");
    choose_mode(&mut s, 0);
    assert!(s.choose_target(Target::Permanent(first)));
    settle(&mut s);
    assert_eq!(
        s.battlefield_permanents_named(1, "Sundial of the Infinite"),
        vec![second]
    );
}
