//! Tests for cards filed in `src/m20.rs`, in card order.

use crate::support::game::move_card;
use crate::support::inspect::{now, permanent_of, zone_of};
use crate::support::{
    game::five_color_setup,
    play::{cast, settle, spend_colorless},
};
use managym::{
    agent::action::Action,
    state::{game_object::Target, zone::ZoneType},
};

#[test]
fn spectral_sailor_draw_ability_survives_the_sources_destruction() {
    let mut s = five_color_setup(&["Spectral Sailor", "Lightning Strike"]);
    let sailor = s.force_permanent_on_battlefield(0, "Spectral Sailor");
    s.game_mut().scenario_refresh_priority().unwrap();
    let index = s
        .action_space()
        .actions
        .iter()
        .position(|a| matches!(a,Action::ActivateAbility{permanent,..} if *permanent==sailor))
        .unwrap();
    s.step_action(index);
    spend_colorless(&mut s);
    cast(&mut s, "Lightning Strike");
    assert!(s.choose_target(Target::Permanent(sailor)));
    settle(&mut s);
    assert!(s.game().state.permanents[sailor].is_none());
    assert_eq!(s.zone_size(0, ZoneType::Hand), 1);
}

#[test]
fn agent_of_treachery_steals_and_draws_three_with_three_unowned_permanents() {
    use crate::support::{inspect::permanent, play::resolve_everything};
    use managym::{flow::turn::StepKind, state::game_object::PlayerId};
    let mut s = five_color_setup(&["Agent of Treachery", "Gray Ogre"]);
    s.set_player_mana_pool(0, "30UUUUUUUU");
    let ogres: Vec<_> = (0..3)
        .map(|_| s.force_permanent_on_battlefield(1, "Gray Ogre"))
        .collect();
    for (stolen, ogre) in ogres.iter().enumerate() {
        cast(&mut s, "Agent of Treachery");
        spend_colorless(&mut s);
        settle_until_target(&mut s, Target::Permanent(*ogre));
        assert_eq!(permanent(&s, *ogre).controller, PlayerId(0));
        assert_eq!(
            s.battlefield_permanents_named(0, "Gray Ogre").len(),
            stolen + 1
        );
    }
    let hand = s.zone_size(0, ZoneType::Hand);
    s.advance_to_step(StepKind::End);
    resolve_everything(&mut s);
    // Each of the three Agents triggers.
    s.assert_zone_size(0, ZoneType::Hand, hand + 9);
}

#[test]
fn agent_of_treachery_draws_nothing_with_two_unowned_permanents() {
    use crate::support::play::resolve_everything;
    use managym::flow::turn::StepKind;
    let mut s = five_color_setup(&["Agent of Treachery", "Gray Ogre"]);
    s.set_player_mana_pool(0, "30UUUUUUUU");
    for _ in 0..2 {
        let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
        cast(&mut s, "Agent of Treachery");
        spend_colorless(&mut s);
        settle_until_target(&mut s, Target::Permanent(ogre));
    }
    let hand = s.zone_size(0, ZoneType::Hand);
    s.advance_to_step(StepKind::End);
    resolve_everything(&mut s);
    s.assert_zone_size(0, ZoneType::Hand, hand);
}

/// Resolve the cast creature, aiming its entry trigger at `target`.
fn settle_until_target(s: &mut crate::support::game::Scenario, target: Target) {
    for _ in 0..20 {
        if s.action_space().kind == managym::agent::action::ActionSpaceKind::ChooseTarget {
            assert!(s.choose_target(target));
        } else if s.action_space().kind == managym::agent::action::ActionSpaceKind::Priority
            && s.game().state.stack_objects.is_empty()
            && s.game().state.pending_triggers.is_empty()
        {
            return;
        } else {
            s.pass_priority();
        }
    }
    panic!("entry trigger did not settle");
}

#[test]
fn brought_back_returns_only_cards_put_into_the_graveyard_from_the_battlefield_this_turn() {
    use crate::support::{
        game::{move_from_library, set_counters},
        inspect::{card_of, offered_targets, permanent},
    };
    use managym::state::permanent::CounterKind;
    let mut s = five_color_setup(&["Brought Back", "Gray Ogre", "Lightning Strike"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    set_counters(&mut s, ogre, CounterKind::PlusOnePlusOne, 1);
    let died = card_of(&s, ogre);
    let other = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let left = card_of(&s, other);
    let milled = move_from_library(&mut s, 0, "Gray Ogre", ZoneType::Graveyard);
    s.game_mut().scenario_refresh_priority().unwrap();
    cast(&mut s, "Lightning Strike");
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    // The second Ogre leaves the battlefield and comes back to its owner's
    // graveyard by another route: graveyard to exile to graveyard is a new
    // object, so it no longer qualifies.
    let left = move_card(&mut s, left, ZoneType::Graveyard);
    let left = move_card(&mut s, left, ZoneType::Exile);
    let left = move_card(&mut s, left, ZoneType::Graveyard);
    s.game_mut().scenario_refresh_priority().unwrap();
    cast(&mut s, "Brought Back");
    let offered = offered_targets(&s);
    let graveyard = |object| {
        s.game()
            .state
            .live(now(&s, object))
            .map(Target::GraveyardCard)
    };
    assert!(offered.contains(&graveyard(died).unwrap()));
    assert!(!offered.contains(&graveyard(milled).unwrap()));
    assert!(!offered.contains(&graveyard(left).unwrap()));
    assert!(s.choose_target(graveyard(died).unwrap()));
    settle(&mut s);
    let returned = permanent_of(&s, died).expect("returned");
    assert!(permanent(&s, returned).tapped);
    // A new object: no counters and no damage carried over.
    assert_eq!(permanent(&s, returned).counters.iter().count(), 0);
    for object in [milled, left] {
        assert_eq!(zone_of(&s, object), Some(ZoneType::Graveyard));
    }
}

#[test]
fn token_producer_rapacious_dragon() {
    use crate::support::{game::five_color_setup, play::enter_and_settle};
    let mut s = five_color_setup(&["Rapacious Dragon"]);
    enter_and_settle(&mut s, "Rapacious Dragon");
    assert_eq!(s.battlefield_permanents_named(0, "Treasure").len(), 2);
}

#[test]
fn foundations_grant_empyrean_eagle() {
    use crate::support::game::{five_color_setup, move_card};
    use crate::support::inspect::card_of;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Empyrean Eagle", "Wind Drake", "Gray Ogre"]);
    let ally = s.force_permanent_on_battlefield(0, "Wind Drake");
    let enemy = s.force_permanent_on_battlefield(1, "Wind Drake");
    let other = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let base = s.game().characteristics(ally).pt();
    let source = s.force_permanent_on_battlefield(0, "Empyrean Eagle");
    assert_eq!(
        s.game().characteristics(ally).pt(),
        (base.0 + 1, base.1 + 1)
    );
    assert_eq!(s.game().characteristics(source).power, 2);
    assert_eq!(s.game().characteristics(other).pt(), (2, 2));
    assert_eq!(s.game().characteristics(enemy).pt(), base);
    let card = card_of(&s, source);
    move_card(&mut s, card, ZoneType::Graveyard);
    assert_eq!(s.game().characteristics(ally).pt(), base);
}

#[test]
fn foundations_5e_devout_decree() {
    use crate::support::game::five_color_setup;
    use crate::support::inspect::{card_of, offered_targets, zone_of};
    use crate::support::play::{cast, pass_until, settle};
    use managym::agent::action::ActionSpaceKind;
    use managym::state::game_object::{PlayerId, Target};
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&[
        "Devout Decree",
        "Typhoid Rats",
        "Gray Ogre",
        "Savannah Lions",
    ]);
    let rats = s.force_permanent_on_battlefield(1, "Typhoid Rats");
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let lions = s.force_permanent_on_battlefield(1, "Savannah Lions");
    let card = card_of(&s, rats);
    cast(&mut s, "Devout Decree");
    let offers = offered_targets(&s);
    assert!(offers.contains(&Target::Permanent(rats)));
    assert!(offers.contains(&Target::Permanent(ogre)));
    assert!(!offers.contains(&Target::Permanent(lions)));
    assert!(s.choose_target(Target::Permanent(rats)));
    spend_colorless(&mut s);
    pass_until(&mut s, ActionSpaceKind::Scry);
    assert_eq!(s.action_space().player, Some(PlayerId(0)));
    assert_eq!(zone_of(&s, card), Some(ZoneType::Exile));
    settle(&mut s);
}

/// "{T}: Another target creature with power 2 or less can't be blocked this
/// turn": it offers every other creature with power 2 or less, whoever
/// controls it, and the chosen Ogre is not asked to be blocked.
#[test]
fn foundations_5f_goblin_smuggler() {
    use crate::support::{
        game::{clear_summoning_sickness, five_color_setup},
        inspect::offered_targets,
        play::{activate, attack_with, settle},
    };
    use managym::{flow::turn::StepKind, state::game_object::Target};
    let mut s = five_color_setup(&["Goblin Smuggler", "Gray Ogre", "Craw Wurm"]);
    let smuggler = s.force_permanent_on_battlefield(0, "Goblin Smuggler");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    clear_summoning_sickness(&mut s, ogre);
    s.force_permanent_on_battlefield(0, "Craw Wurm");
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let blocker = s.force_permanent_on_battlefield(1, "Craw Wurm");
    // Haste: the Smuggler taps the turn it arrives.
    activate(&mut s, smuggler, 0);
    let mut offered = offered_targets(&s);
    offered.sort_by_key(|target| format!("{target:?}"));
    let mut wanted = vec![Target::Permanent(ogre), Target::Permanent(theirs)];
    wanted.sort_by_key(|target| format!("{target:?}"));
    assert_eq!(offered, wanted);
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, ogre);
    let blocks = crate::walk_block_declaration(&mut s);
    assert!(
        blocks.iter().all(|(_, attackers)| attackers.is_empty()),
        "{blocks:?}"
    );
    assert!(blocks.iter().any(|(who, _)| *who == blocker), "{blocks:?}");
}

/// "Whenever you gain life, put a +1/+1 counter on this creature. As long as
/// you have 25 or more life, this creature has double strike": gaining to 25
/// grants double strike at once, and damage below 25 removes it, with the
/// cached characteristics read before each change.
#[test]
fn foundations_5f_twinblade_paladin() {
    use crate::support::{
        game::{clear_summoning_sickness, five_color_setup},
        inspect::{counters, keywords},
        play::{activate, cast, settle, spend_colorless},
    };
    use managym::state::{
        card::Keyword,
        game_object::{PlayerId, Target},
    };
    let mut s = five_color_setup(&["Twinblade Paladin", "Vampire Neonate", "Lightning Bolt"]);
    let paladin = s.force_permanent_on_battlefield(0, "Twinblade Paladin");
    let neonate = s.force_permanent_on_battlefield(0, "Vampire Neonate");
    clear_summoning_sickness(&mut s, neonate);
    s.game_mut().state.players[0].life = 24;
    assert!(!keywords(&s, paladin).has(Keyword::DoubleStrike));

    activate(&mut s, neonate, 0);
    spend_colorless(&mut s);
    // The drain resolves; the gain's trigger waits on the stack.
    s.pass_priority();
    s.pass_priority();
    assert_eq!(s.life(0), 25);
    assert!(keywords(&s, paladin).has(Keyword::DoubleStrike));
    settle(&mut s);
    assert_eq!(counters(&s, paladin), 1);
    assert_eq!(s.game().characteristics(paladin).pt(), (4, 4));

    cast(&mut s, "Lightning Bolt");
    assert!(s.choose_target(Target::Player(PlayerId(0))));
    settle(&mut s);
    assert_eq!(s.life(0), 22);
    assert!(!keywords(&s, paladin).has(Keyword::DoubleStrike));
}
