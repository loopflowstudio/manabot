//! Tests for cards filed in `src/lci.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    inspect::observed,
    play::{cast, settle},
};
use managym::state::{
    card::{CardType, Keyword},
    game_object::Target,
    mana::Color,
};

/// The Skeleton loses its other creature types, colors and abilities
/// (CR 205.1a): Llanowar Elves becomes a black 1/1 that taps for nothing.
#[test]
fn eaten_by_piranhas_makes_a_black_skeleton_without_its_other_types() {
    let mut s = five_color_setup(&["Eaten by Piranhas", "Llanowar Elves"]);
    let elves = s.force_permanent_on_battlefield(1, "Llanowar Elves");
    cast(&mut s, "Eaten by Piranhas");
    assert!(s.choose_target(Target::Permanent(elves)));
    settle(&mut s);
    let view = s.game().characteristics(elves);
    assert_eq!(view.subtypes, vec!["Skeleton".to_string()]);
    assert_eq!(
        view.types.types.iter().copied().collect::<Vec<_>>(),
        vec![CardType::Creature]
    );
    assert_eq!(
        view.colors.iter().copied().collect::<Vec<_>>(),
        vec![Color::Black]
    );
    assert_eq!(view.pt(), (1, 1));
    assert!(view.activated_abilities.is_empty(), "no mana ability");
    assert!(!view.keywords.has(Keyword::Flash));
    let row = observed(&s, elves);
    assert_eq!(row.creature_types, Some(vec!["Skeleton".to_string()]));
    assert!(row.temp_subtypes.is_empty(), "set, not added");
    assert_eq!(row.name, None);
}

/// A hasty creature that loses haste on the turn it entered is summoning
/// sick again (CR 302.6): the Skeleton can't attack until its controller's
/// next turn.
#[test]
fn eaten_by_piranhas_stops_a_hasty_creature_attacking_the_turn_it_entered() {
    use crate::support::{inspect::permanent_of, play::enter_and_settle};
    use managym::{
        agent::action::{Action, ActionSpaceKind},
        flow::turn::StepKind,
    };
    let mut s = five_color_setup(&["Eaten by Piranhas", "Raging Goblin"]);
    let object = enter_and_settle(&mut s, "Raging Goblin");
    let goblin = permanent_of(&s, object).unwrap();
    assert!(!observed(&s, goblin).is_summoning_sick, "haste");
    cast(&mut s, "Eaten by Piranhas");
    assert!(s.choose_target(Target::Permanent(goblin)));
    settle(&mut s);
    assert!(observed(&s, goblin).is_summoning_sick);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::DeclareAttacker { permanent, .. } if *permanent == goblin)));
    s.advance_to_active_step(1, StepKind::Upkeep);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    assert_eq!(s.action_space().kind, ActionSpaceKind::DeclareAttacker);
}
