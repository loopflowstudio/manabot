//! Tests for cards filed in `src/zen.rs`, in card order.

use crate::support::{
    game::five_color_setup,
    inspect::{counters, library_card, permanent_named},
    play::{activate_first, choose, play_and_settle, settle},
};
use managym::{
    agent::action::{Action, ActionSpaceKind},
    state::zone::ZoneType,
};

#[test]
fn landfall_creates_resources_and_counters() {
    let mut s = five_color_setup(&[
        "Sazh's Chocobo",
        "Tireless Tracker",
        "Lotus Cobra",
        "Tireless Provisioner",
    ]);
    for name in [
        "Sazh's Chocobo",
        "Tireless Tracker",
        "Lotus Cobra",
        "Tireless Provisioner",
    ] {
        play_and_settle(&mut s, name);
    }
    // The Cobra's color and Provisioner's token are separate resolution choices.
    let land = library_card(&s, 0, "Forest");
    s.game_mut().move_card(land, ZoneType::Battlefield);
    s.game_mut().scenario_refresh_priority().unwrap();
    s.pass_priority();
    for _ in 0..20 {
        if s.action_space().kind == ActionSpaceKind::Modal {
            choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 0, .. }));
        } else if s.action_space().kind == ActionSpaceKind::OrderTriggers {
            s.step_action(0);
        } else if s.game().state.stack_objects.is_empty()
            && s.action_space().kind == ActionSpaceKind::Priority
        {
            break;
        } else {
            s.pass_priority();
        }
    }
    assert_eq!(counters(&s, permanent_named(&s, "Sazh's Chocobo")), 1);
    assert_eq!(s.battlefield_permanents_named(0, "Clue").len(), 1);
    assert_eq!(s.battlefield_permanents_named(0, "Food").len(), 1);
    let clue = permanent_named(&s, "Clue");
    activate_first(&mut s, clue);
    settle(&mut s);
    assert_eq!(counters(&s, permanent_named(&s, "Tireless Tracker")), 1);
}

#[test]
fn beastmaster_ascension_collects_quest_counters_and_pumps_at_seven() {
    use crate::support::{
        game::{clear_summoning_sickness, set_counters},
        inspect::counters_of,
        play::{attack_with, choose, pass_until, resolve_everything},
    };
    use managym::{
        agent::action::{Action, ActionSpaceKind},
        state::permanent::CounterKind,
    };
    let mut s = five_color_setup(&["Beastmaster Ascension", "Gray Ogre"]);
    let ascension = s.force_permanent_on_battlefield(0, "Beastmaster Ascension");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    clear_summoning_sickness(&mut s, ogre);
    s.game_mut().scenario_refresh_priority().unwrap();
    pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
    attack_with(&mut s, ogre);
    for _ in 0..8 {
        if s.action_space()
            .actions
            .iter()
            .any(|a| matches!(a, Action::PayCost { .. }))
        {
            break;
        }
        s.pass_priority();
    }
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    resolve_everything(&mut s);
    assert_eq!(counters_of(&s, ascension, CounterKind::Quest), 1);
    assert_eq!(s.game().characteristics(ogre).power, 2);
    set_counters(&mut s, ascension, CounterKind::Quest, 7);
    let view = s.game().characteristics(ogre);
    assert_eq!((view.power, view.toughness), (7, 7));
}

#[test]
fn valakut_deals_three_when_a_sixth_mountain_enters() {
    use crate::support::play::{choose, enter_and_settle};
    use managym::{
        agent::action::Action,
        state::game_object::{PlayerId, Target},
    };
    for others in [4, 5] {
        let mut s = five_color_setup(&["Valakut, the Molten Pinnacle"]);
        s.force_permanent_on_battlefield(0, "Valakut, the Molten Pinnacle");
        for _ in 0..others {
            s.force_permanent_on_battlefield(0, "Mountain");
        }
        s.game_mut().scenario_refresh_priority().unwrap();
        let id = crate::support::inspect::library_card(&s, 0, "Mountain");
        s.game_mut()
            .move_card(id, managym::state::zone::ZoneType::Battlefield);
        s.game_mut().scenario_refresh_priority().unwrap();
        for _ in 0..8 {
            let space = s.action_space();
            if space
                .actions
                .iter()
                .any(|a| matches!(a, Action::ChooseTarget { .. }))
            {
                assert!(s.choose_target(Target::Player(PlayerId(1))));
            } else if space
                .actions
                .iter()
                .any(|a| matches!(a, Action::PayCost { .. }))
            {
                choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
            } else if s.game().state.stack_objects.is_empty()
                && s.game().state.pending_triggers.is_empty()
            {
                break;
            } else {
                s.pass_priority();
            }
        }
        let _ = enter_and_settle;
        s.assert_life(1, if others == 5 { 17 } else { 20 });
    }
}

/// Drive the stack to rest, accepting each "you may" (an optional payment of
/// no mana) when `accept`, declining it otherwise.
fn settle_choosing(s: &mut crate::support::game::Scenario, accept: bool) {
    use managym::agent::action::{Action, ActionSpaceKind};
    for _ in 0..40 {
        let space = s.action_space();
        if space.kind == ActionSpaceKind::Priority
            && s.game().state.stack_objects.is_empty()
            && s.game().state.pending_triggers.is_empty()
        {
            return;
        }
        let pay = space
            .actions
            .iter()
            .position(|a| matches!(a, Action::PayCost { .. }));
        let decline = space
            .actions
            .iter()
            .position(|a| matches!(a, Action::Decline { .. }));
        match (pay, decline) {
            (Some(index), _) if accept => s.step_action(index),
            (_, Some(index)) => s.step_action(index),
            _ if space.kind == ActionSpaceKind::OrderTriggers => s.step_action(0),
            _ => s.pass_priority(),
        }
    }
    panic!("stack did not settle");
}

#[test]
fn bloodchief_ascension_counts_life_lost_this_turn_at_each_end_step() {
    use crate::support::{inspect::counters_of, play::cast};
    use managym::{
        flow::turn::StepKind,
        state::{
            game_object::{PlayerId, Target},
            permanent::CounterKind,
        },
    };
    for (strike, expected) in [(false, 0), (true, 1)] {
        let mut s = five_color_setup(&["Bloodchief Ascension", "Lightning Strike"]);
        let ascension = s.force_permanent_on_battlefield(0, "Bloodchief Ascension");
        s.game_mut().scenario_refresh_priority().unwrap();
        if strike {
            cast(&mut s, "Lightning Strike");
            assert!(s.choose_target(Target::Player(PlayerId(1))));
            settle_choosing(&mut s, true);
            assert_eq!(s.game().state.turn.history[1].life_lost, 3);
        }
        s.advance_to_step(StepKind::End);
        settle_choosing(&mut s, true);
        assert_eq!(
            counters_of(&s, ascension, CounterKind::Quest),
            expected,
            "opponent struck: {strike}"
        );
    }
}

#[test]
fn bloodchief_ascension_drains_for_each_card_put_into_an_opponents_graveyard() {
    use crate::support::{
        game::{move_from_library, set_counters},
        inspect::counters_of,
    };
    use managym::state::permanent::CounterKind;
    for (quests, accept, drained) in [(3, true, true), (3, false, false), (2, true, false)] {
        let mut s = five_color_setup(&["Bloodchief Ascension", "Gray Ogre"]);
        let ascension = s.force_permanent_on_battlefield(0, "Bloodchief Ascension");
        set_counters(&mut s, ascension, CounterKind::Quest, quests);
        s.game_mut().scenario_refresh_priority().unwrap();
        let (mine, theirs) = (s.life(0), s.life(1));
        // A card of the Ascension's controller triggers nothing.
        move_from_library(&mut s, 0, "Gray Ogre", ZoneType::Graveyard);
        move_from_library(&mut s, 1, "Gray Ogre", ZoneType::Graveyard);
        s.game_mut().scenario_refresh_priority().unwrap();
        settle_choosing(&mut s, accept);
        let delta = if drained { 2 } else { 0 };
        assert_eq!((s.life(0), s.life(1)), (mine + delta, theirs - delta));
        assert_eq!(counters_of(&s, ascension, CounterKind::Quest), quests);
    }
}
