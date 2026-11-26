//! Core Set 2021 card behavior.
use crate::support::game::{five_color_setup, move_card};
use crate::support::inspect::{card_of, offered_targets, permanent, permanent_of, zone_of};
use crate::support::play::{activate, cast, settle};
use managym::state::game_object::{PlayerId, Target};
use managym::state::player::ManaPayment;
use managym::state::zone::ZoneType;

#[test]
fn heartfire_immolator_deals_its_power_at_sacrifice_after_returning() {
    for expected in [2, 3] {
        let mut s = five_color_setup(&["Heartfire Immolator", "Craw Wurm", "Lightning Bolt"]);
        let source = s.force_permanent_on_battlefield(0, "Heartfire Immolator");
        let card = card_of(&s, source);
        let wurm = s.force_permanent_on_battlefield(1, "Craw Wurm");
        if expected == 3 {
            cast(&mut s, "Lightning Bolt");
            assert!(s.choose_target(Target::Player(managym::state::game_object::PlayerId(1))));
            settle(&mut s);
        }
        assert_eq!(s.game().characteristics(source).power, expected);
        activate(&mut s, source, 0);
        assert!(crate::support::inspect::offered_targets(&s).contains(&Target::Permanent(source)));
        assert!(s.choose_target(Target::Permanent(wurm)));
        assert_eq!(zone_of(&s, card), Some(ZoneType::Graveyard));
        move_card(&mut s, card, ZoneType::Battlefield);
        let obs = managym::agent::observation::Observation::for_player(
            s.game(),
            managym::state::game_object::PlayerId(1),
        );
        let [ability] = obs.stack_objects.as_slice() else {
            panic!("expected pending Immolator activation")
        };
        assert_eq!(ability.source_power, Some(expected));
        assert!(ability
            .description
            .as_ref()
            .unwrap()
            .contains(&format!("source power {expected}")));
        settle(&mut s);
        assert_eq!(permanent(&s, wurm).damage, expected);
    }
}

/// The reduction reads the turn's total gain as the cost is determined
/// (CR 601.2f): several gains add up and a later loss does not undo them.
#[test]
fn sanguine_indulgence_costs_three_less_after_three_life_gained_this_turn() {
    use crate::support::{
        game::{force_new_card_in_hand, plant_in_graveyard},
        inspect::{now, top_spell},
        play::{cast_card, select_card},
    };
    use managym::state::{game_object::PlayerId, player::ManaPayment};
    // Each gain sacrifices a creature to Ayli for its toughness.
    for (gains, bolt_self, spent, returned) in [
        (&[][..], false, 4, 2),
        (&["Gray Ogre"][..], false, 4, 1),
        (&["Gray Ogre", "Savannah Lions"][..], true, 1, 2),
        (&["Hill Giant"][..], false, 1, 0),
    ] {
        let mut s = five_color_setup(&[
            "Sanguine Indulgence",
            "Ayli, Eternal Pilgrim",
            "Lightning Bolt",
            "Gray Ogre",
            "Hill Giant",
            "Savannah Lions",
        ]);
        s.game_mut().mana_payment[0] = ManaPayment::Auto;
        let ayli = s.force_permanent_on_battlefield(0, "Ayli, Eternal Pilgrim");
        for name in gains {
            let creature = s.force_permanent_on_battlefield(0, name);
            let card = card_of(&s, creature);
            activate(&mut s, ayli, 0);
            select_card(&mut s, card);
            settle(&mut s);
        }
        if bolt_self {
            let bolt = force_new_card_in_hand(&mut s, 0, "Lightning Bolt");
            cast_card(&mut s, bolt);
            s.choose_target(Target::Player(PlayerId(0)));
            settle(&mut s);
        }
        let creatures: Vec<_> = ["Gray Ogre", "Hill Giant"][..returned]
            .iter()
            .map(|name| plant_in_graveyard(&mut s, 0, name))
            .collect();
        let spell = force_new_card_in_hand(&mut s, 0, "Sanguine Indulgence");
        cast_card(&mut s, spell);
        for creature in &creatures {
            let card = now(&s, *creature);
            assert!(s.choose_target(Target::GraveyardCard(card)));
        }
        if s.action_space().kind == managym::agent::action::ActionSpaceKind::ChooseTarget {
            // Up to two: finish with fewer.
            crate::support::play::choose(&mut s, |a| {
                matches!(a, managym::agent::action::Action::Decline { .. })
            });
        }
        assert_eq!(top_spell(&s).spent.total(), spent, "gains {gains:?}");
        settle(&mut s);
        for creature in creatures {
            assert_eq!(zone_of(&s, now(&s, creature)), Some(ZoneType::Hand));
        }
    }
}

#[test]
fn mazemind_tome_exiles_itself_once_its_fourth_page_counter_is_paid() {
    use crate::support::{game::set_counters, inspect::counters_of, play::counter_stacked_ability};
    use managym::state::{permanent::CounterKind::Page, stack_object::StackObject};

    let mut s = five_color_setup(&["Mazemind Tome", "Gray Ogre"]);
    let tome = s.force_permanent_on_battlefield(0, "Mazemind Tome");
    let card = card_of(&s, tome);
    set_counters(&mut s, tome, Page, 2);
    activate(&mut s, tome, 0);
    settle(&mut s);
    assert_eq!(counters_of(&s, tome, Page), 3);
    assert_eq!(zone_of(&s, card), Some(ZoneType::Battlefield));
    permanent_untap(&mut s, tome);

    // The cost puts the fourth counter on before the Scry is on the stack, so
    // the state trigger goes above it (CR 603.8) and triggers only once.
    activate(&mut s, tome, 0);
    let stack = &s.game().state.stack_objects;
    assert!(matches!(
        stack.as_slice(),
        [
            StackObject::ActivatedAbility(_),
            StackObject::TriggeredAbility(_)
        ]
    ));
    assert_eq!(counters_of(&s, tome, Page), 4);
    let life = s.life(0);
    let pending = s.game().clone();
    crate::support::play::resolve_top(&mut s);
    assert_eq!(zone_of(&s, card), Some(ZoneType::Exile));
    assert_eq!(s.life(0), life + 4);
    assert_eq!(
        s.game().state.stack_objects.len(),
        1,
        "the Scry still waits"
    );

    // Countered while four counters remain, it triggers again.
    *s.game_mut() = pending.clone();
    counter_stacked_ability(&mut s);
    assert_eq!(zone_of(&s, card), Some(ZoneType::Exile));
    assert_eq!(s.life(0), life + 4);

    // Exiled in response, "if you do" gains nothing.
    *s.game_mut() = pending;
    move_card(&mut s, card, ZoneType::Exile);
    settle(&mut s);
    assert_eq!(s.life(0), life);
}

/// Only the printed state trigger waits for itself (CR 603.8): a granted or
/// delayed trigger from the same Tome at the same ability index is another
/// ability, so the Tome's four page counters still trigger beside it.
#[test]
fn mazemind_tome_state_trigger_is_not_held_by_another_trigger_of_the_tome() {
    use crate::support::game::set_counters;
    use managym::state::{permanent::CounterKind::Page, stack_object::StackObject};

    let mut s = five_color_setup(&["Mazemind Tome", "Gray Ogre"]);
    let tome = s.force_permanent_on_battlefield(0, "Mazemind Tome");
    set_counters(&mut s, tome, Page, 3);
    activate(&mut s, tome, 0);
    let triggered = |s: &crate::support::game::Scenario| {
        s.game()
            .state
            .stack_objects
            .iter()
            .filter(|item| matches!(item, StackObject::TriggeredAbility(_)))
            .count()
    };
    assert_eq!(triggered(&s), 1);

    // The waiting state trigger becomes one with its own effects, as a
    // granted trigger of the Tome would be.
    for item in &mut s.game_mut().state.stack_objects {
        if let StackObject::TriggeredAbility(trigger) = item {
            assert_eq!(trigger.ability_index, 0);
            trigger.inline_effects = Some(Vec::new());
        }
    }
    s.game_mut().skip_trivial = false;
    s.pass_priority();
    assert_eq!(triggered(&s), 2);
}

fn permanent_untap(
    s: &mut crate::support::game::Scenario,
    id: managym::state::game_object::PermanentId,
) {
    s.game_mut().state.permanents[id].as_mut().unwrap().tapped = false;
}

#[test]
fn foundations_permanent_shipwreck_dowser() {
    use crate::support::game::{five_color_setup, plant_in_graveyard};
    use crate::support::inspect::{permanent_named, zone_of};
    use crate::support::play::{cast, pass_until, play_and_settle, settle, spend_colorless};
    use managym::agent::action::ActionSpaceKind;
    use managym::state::game_object::Target;
    use managym::state::zone::ZoneType;
    let mut s = five_color_setup(&["Shipwreck Dowser", "Gray Ogre", "Giant Growth", "Sol Ring"]);
    let wrong_type = plant_in_graveyard(&mut s, 0, "Gray Ogre");
    let theirs = plant_in_graveyard(&mut s, 1, "Giant Growth");
    let card = plant_in_graveyard(&mut s, 0, "Giant Growth");
    cast(&mut s, "Shipwreck Dowser");
    spend_colorless(&mut s);
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(!s.choose_target(Target::GraveyardCard(wrong_type)));
    assert!(!s.choose_target(Target::GraveyardCard(theirs)));
    assert!(s.choose_target(Target::GraveyardCard(card)));
    settle(&mut s);
    assert_eq!(zone_of(&s, card), Some(ZoneType::Hand));
    play_and_settle(&mut s, "Sol Ring");
    assert_eq!(
        s.game()
            .characteristics(permanent_named(&s, "Shipwreck Dowser"))
            .pt(),
        (4, 4)
    );
}

/// Prowess pumps the Immolator to 3/3; sacrificing it deals 3 to a 3/3 Hill
/// Giant. Returning the card to the battlefield before the ability resolves
/// makes a new 2/2 appearance, and the damage still reads the sacrificed one.
#[test]
fn heartfire_immolator_deals_its_sacrificed_appearances_pumped_power() {
    let mut s = five_color_setup(&["Heartfire Immolator", "Lightning Strike", "Hill Giant"]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let immolator = s.force_permanent_on_battlefield(0, "Heartfire Immolator");
    let immolator_card = card_of(&s, immolator);
    let giant = s.force_permanent_on_battlefield(1, "Hill Giant");
    let giant_card = card_of(&s, giant);

    cast(&mut s, "Lightning Strike");
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    settle(&mut s);
    assert_eq!(s.life(1), 17);
    assert_eq!(s.game().characteristics(immolator).pt(), (3, 3), "prowess");

    activate(&mut s, immolator, 0);
    let targets = offered_targets(&s);
    assert!(targets.contains(&Target::Permanent(giant)));
    assert!(!targets.iter().any(|t| matches!(t, Target::Player(_))));
    assert!(s.choose_target(Target::Permanent(giant)));
    assert_eq!(zone_of(&s, immolator_card), Some(ZoneType::Graveyard));
    let stacked = s.snapshot();
    settle(&mut s);
    assert_eq!(zone_of(&s, giant_card), Some(ZoneType::Graveyard));

    s.restore(&stacked);
    let returned = move_card(&mut s, immolator_card, ZoneType::Battlefield);
    let returned = permanent_of(&s, returned).expect("a new appearance");
    assert_eq!(s.game().characteristics(returned).pt(), (2, 2));
    settle(&mut s);
    assert_eq!(
        zone_of(&s, giant_card),
        Some(ZoneType::Graveyard),
        "3 damage from the sacrificed 3/3, not the returned 2/2"
    );
    assert!(s.game().state.permanents[returned].is_some());
}

#[test]
fn primal_might_pumps_and_optionally_fights() {
    use crate::support::play::{choose, choose_mode};
    use managym::agent::action::Action;

    // With fight target: +X/+X applies before the fight.
    let mut s = five_color_setup(&["Primal Might", "Gray Ogre", "Hill Giant"]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let giant = s.force_permanent_on_battlefield(1, "Hill Giant");
    let giant_card = card_of(&s, giant);
    cast(&mut s, "Primal Might");
    choose_mode(&mut s, 3);
    assert!(s.choose_target(Target::Permanent(ogre)));
    assert!(s.choose_target(Target::Permanent(giant)));
    settle(&mut s);
    assert_eq!(s.game().characteristics(ogre).pt(), (5, 5));
    assert_eq!(permanent(&s, ogre).damage, 3);
    assert_eq!(zone_of(&s, giant_card), Some(ZoneType::Graveyard));

    // Declining the optional opposing creature still pumps +X/+X.
    let mut s = five_color_setup(&["Primal Might", "Gray Ogre", "Hill Giant"]);
    s.game_mut().mana_payment[0] = ManaPayment::Auto;
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let giant = s.force_permanent_on_battlefield(1, "Hill Giant");
    cast(&mut s, "Primal Might");
    choose_mode(&mut s, 2);
    assert!(s.choose_target(Target::Permanent(ogre)));
    choose(&mut s, |a| matches!(a, Action::Decline { .. }));
    settle(&mut s);
    assert_eq!(s.game().characteristics(ogre).pt(), (4, 4));
    assert_eq!(permanent(&s, ogre).damage, 0);
    assert_eq!(permanent(&s, giant).damage, 0);
}

#[test]
fn garruks_uprising_checks_power_on_entry_and_on_resolution() {
    use crate::support::play::enter_and_settle;
    use managym::state::card::Keyword;
    let mut s = five_color_setup(&["Garruk's Uprising", "Goblin Boarders", "Hill Giant"]);
    let giant = s.force_permanent_on_battlefield(0, "Hill Giant");
    enter_and_settle(&mut s, "Garruk's Uprising");
    assert_eq!(s.zone_size(0, ZoneType::Hand), 0);
    assert!(s
        .game()
        .characteristics(giant)
        .keywords
        .has(Keyword::Trample));
    s.game_mut().state.turn.history[0].attacked = true;
    let boarders = enter_and_settle(&mut s, "Goblin Boarders");
    assert_eq!(s.zone_size(0, ZoneType::Hand), 1);
    enter_and_settle(&mut s, "Garruk's Uprising");
    assert_eq!(s.zone_size(0, ZoneType::Hand), 2);
    let third = crate::support::inspect::library_card(&s, 0, "Garruk's Uprising");
    move_card(&mut s, third, ZoneType::Battlefield);
    s.game_mut().scenario_refresh_priority().unwrap();
    move_card(&mut s, boarders, ZoneType::Graveyard);
    settle(&mut s);
    assert_eq!(s.zone_size(0, ZoneType::Hand), 2);
}

#[test]
fn wildwood_scourge_enters_with_x_and_grows_only_for_other_non_hydras() {
    use crate::support::{
        inspect::{counters, permanent_named},
        play::{choose_mode, enter_and_settle},
    };
    let mut s = five_color_setup(&[
        "Wildwood Scourge",
        "Heroes' Bane",
        "Giada, Font of Hope",
        "Angel of Vitality",
        "Rise of the Dark Realms",
    ]);
    cast(&mut s, "Wildwood Scourge");
    choose_mode(&mut s, 2);
    settle(&mut s);
    let hydra = permanent_named(&s, "Wildwood Scourge");
    assert_eq!(counters(&s, hydra), 2);
    enter_and_settle(&mut s, "Heroes' Bane");
    assert_eq!(counters(&s, hydra), 2);
    s.force_permanent_on_battlefield(0, "Giada, Font of Hope");
    for player in 0..2 {
        crate::support::game::plant_in_graveyard(&mut s, player, "Angel of Vitality");
    }
    cast(&mut s, "Rise of the Dark Realms");
    settle(&mut s);
    let angels = s.battlefield_permanents_named(0, "Angel of Vitality");
    assert_eq!(angels.len(), 2);
    for angel in angels {
        assert_eq!(
            counters(&s, angel),
            1,
            "only Giada was already on the battlefield"
        );
    }
    assert_eq!(counters(&s, hydra), 4);
}
