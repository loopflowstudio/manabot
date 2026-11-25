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

/// "Return up to one other target permanent you control to its owner's
/// hand": a land or creature of its controller is offered, never the Pup
/// or an opponent's permanent, and the target may be declined.
#[test]
fn foundations_5f_mischievous_pup() {
    use crate::support::game::five_color_setup;
    use crate::support::inspect::{card_of, offered_targets, permanent_of, zone_of};
    use crate::support::play::{choose, enter_until_target, settle};
    use managym::agent::action::Action;
    use managym::state::game_object::Target;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Mischievous Pup", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let forest = s.force_permanent_on_battlefield(0, "Forest");
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let pup = enter_until_target(&mut s, "Mischievous Pup");
    let source = permanent_of(&s, pup).unwrap();
    let offers = offered_targets(&s);
    assert!(offers.contains(&Target::Permanent(ogre)));
    assert!(offers.contains(&Target::Permanent(forest)));
    assert!(!offers.contains(&Target::Permanent(theirs)));
    assert!(!offers.contains(&Target::Permanent(source)));
    let mut declined = s.clone();
    choose(&mut declined, |a| matches!(a, Action::Decline { .. }));
    settle(&mut declined);
    assert!(declined.game().state.permanents[ogre].is_some());
    let card = card_of(&s, ogre);
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    assert_eq!(zone_of(&s, card), Some(ZoneType::Hand));
    assert!(s
        .game()
        .characteristics(source)
        .keywords
        .has(managym::state::card::Keyword::Flash));
}

/// Eaten by Piranhas on Juggernaut: a black 1/1 Skeleton creature with no
/// abilities that keeps its name; removing the Aura restores it.
#[test]
fn foundations_5f_eaten_by_piranhas() {
    crate::assert_sets_juggernaut_characteristics(
        "Eaten by Piranhas",
        "Juggernaut",
        "Skeleton",
        &[managym::state::mana::Color::Black],
    );
}

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
