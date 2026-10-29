//! Tests for cards filed in `src/cardsets/arb.rs`, grouped by the helper family they
//! drive play with (`tests/support/`).

use crate::support::temur::{activate, can_cast, choose, enter};
use crate::support::{
    elemental::{card, cast, permanent, settle, setup},
    helpers::Scenario,
};
use managym::{
    agent::action::Action,
    state::{
        game_object::{CardId, PlayerId},
        mana::{Mana, ManaUse},
        zone::ZoneType,
    },
};
use std::collections::BTreeMap;

#[test]
fn prism_consumes_generic_before_adding_chosen_color() {
    let mut s = setup(&["Prophetic Prism"]);
    enter(&mut s, "Prophetic Prism");
    s.set_player_mana_pool(0, "1");
    activate(&mut s, "Prophetic Prism", 0);
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
    choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 1, .. }));
    assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("U"));
    assert!(s.game().state.stack_objects.is_empty());
}

#[test]
fn prism_payment_preserves_the_spells_required_green_mana() {
    for lands in [
        ["Forest", "Island", "Island", "Mountain"],
        ["Island", "Island", "Mountain", "Forest"],
    ] {
        let mut s = setup(&["Prophetic Prism", "Gnarled Professor"]);
        for land in lands {
            enter(&mut s, land);
        }
        enter(&mut s, "Prophetic Prism");
        s.set_player_mana_pool(0, "0");
        cast(&mut s, "Gnarled Professor");
        activate(&mut s, "Prophetic Prism", 0);
        assert!(
            !s.game().state.permanents[permanent(&s, "Forest")]
                .as_ref()
                .unwrap()
                .tapped
        );
        assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
        // Green is the only output that leaves the announced GG spell payable.
        assert_eq!(s.action_space().actions.len(), 1);
        choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 4, .. }));
        choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
        assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
        assert!(s.game().state.stack_objects.iter().any(|object| matches!(
            object,
            managym::state::stack_object::StackObject::Spell(spell)
                if s.game().state.cards[spell.card].name == "Gnarled Professor"
        )));
    }
}

#[test]
fn prism_payment_preserves_required_blue_already_in_the_pool() {
    let mut s = setup(&["Prophetic Prism", "Gnarled Professor", "Counterspell"]);
    enter(&mut s, "Forest");
    enter(&mut s, "Prophetic Prism");
    let professor = cast(&mut s, "Gnarled Professor");
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    s.set_player_mana_pool(0, "U");
    cast(&mut s, "Counterspell");
    choose(&mut s, |a| {
        matches!(a, Action::ChooseTarget {
        target: managym::state::game_object::Target::StackSpell(id), ..
    } if *id == professor)
    });
    activate(&mut s, "Prophetic Prism", 0);
    // Spend the Forest's G on Prism, retaining the pooled U for Counterspell.
    assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("U"));
    choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 1, .. }));
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
    assert_eq!(s.game().state.stack_objects.len(), 2);
}

/// CR 605.3a/b, 118.3: one Treasure can pay Prism's activation cost;
/// sacrifice precedes the chosen-color output, and neither uses the stack.
#[test]
fn audit_prism_treasure_empty_pool_activation() {
    let deck = BTreeMap::from([
        ("Mountain".to_string(), 8),
        ("Prophetic Prism".to_string(), 1),
    ]);
    let mut s = Scenario::new(deck.clone(), deck, 177_605);
    s.advance_to_active_step(0, managym::flow::turn::StepKind::Main);
    // Only the activation is under test; do not introduce Prism's ETB draw.
    s.force_permanent_on_battlefield(0, "Prophetic Prism");
    s.game_mut().create_token("Treasure", PlayerId(0), false);
    s.set_player_mana_pool(0, "0");
    activate(&mut s, "Prophetic Prism", 0);
    // Paying the upfront cost exposes the nested Treasure/color decisions.
    activate(&mut s, "Treasure", 0);
    choose(&mut s, |action| {
        matches!(action, Action::ChooseMode { mode: 4, .. })
    });
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
    choose(&mut s, |action| {
        matches!(action, Action::ChooseMode { mode: 1, .. })
    });
    assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("U"));
    assert!(s.battlefield_permanents_named(0, "Treasure").is_empty());
    assert!(
        s.game().state.permanents[permanent(&s, "Prophetic Prism")]
            .as_ref()
            .unwrap()
            .tapped
    );
    assert!(s.game().state.stack_objects.is_empty());
}

#[test]
fn priority_prism_activation_can_use_a_treasure_for_its_upfront_cost() {
    let mut s = setup(&["Prophetic Prism", "Tireless Provisioner"]);
    enter(&mut s, "Prophetic Prism");
    s.game_mut().create_token("Treasure", PlayerId(0), false);
    s.set_player_mana_pool(0, "0");
    s.game_mut().scenario_refresh_priority().unwrap();
    let prism = permanent(&s, "Prophetic Prism");
    let index = s
        .action_space()
        .actions
        .iter()
        .position(|action| matches!(action, Action::ActivateAbility { permanent, .. } if *permanent == prism))
        .expect("Prism is offered when Treasure can fund its upfront cost");
    s.step_action(index);
}

#[test]
fn prism_upfront_payment_preserves_professors_second_green() {
    for sources in [
        ["Forest", "Island", "Prophetic Prism", "Island", "Mountain"],
        ["Mountain", "Island", "Prophetic Prism", "Island", "Forest"],
    ] {
        let mut s = setup(&["Prophetic Prism", "Gnarled Professor"]);
        for name in sources {
            enter(&mut s, name);
        }
        s.set_player_mana_pool(0, "0");
        let professor = cast(&mut s, "Gnarled Professor");
        assert!(s
            .action_space()
            .actions
            .iter()
            .any(|a| matches!(a, Action::ActivateAbility { .. })));
        activate(&mut s, "Prophetic Prism", 0);
        assert!(
            !s.game().state.permanents[permanent(&s, "Forest")]
                .as_ref()
                .unwrap()
                .tapped
        );
        assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
        assert_eq!(s.action_space().actions.len(), 1);
        choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 4, .. }));
        choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
        assert_eq!(
            s.game().state.zones.zone_of(professor),
            Some(ZoneType::Stack)
        );
        settle(&mut s);
        assert!(!s
            .battlefield_permanents_named(0, "Gnarled Professor")
            .is_empty());
    }
}

#[test]
fn prism_payment_offers_execute_and_undo_without_mutating_sibling_forks() {
    use managym::{
        search_state::{witness, BenchCommand, BranchDriver, ClonePlusUndoDriver},
        Game,
    };

    fn complete_every_offer(game: &mut Game, spell: CardId, depth: usize) {
        if game.state.zones.zone_of(spell) == Some(ZoneType::Stack) {
            return;
        }
        assert!(
            depth > 0,
            "payment must complete within its three decisions"
        );
        let driver = ClonePlusUndoDriver::default();
        let root = witness(game);
        let actions = game.action_space().unwrap().actions.clone();
        assert!(!actions.is_empty());
        for action_index in 0..actions.len() {
            let mark = driver.mark(game);
            driver
                .apply(
                    game,
                    BenchCommand {
                        action_index,
                        expected_action_hash: None,
                        expected_state_hash: None,
                    },
                )
                .unwrap();
            complete_every_offer(game, spell, depth - 1);
            driver.rollback(game, mark);
            assert_eq!(witness(game), root);
        }
    }

    for forests in 0..=2 {
        let mut s = setup(&["Prophetic Prism", "Gnarled Professor"]);
        for _ in 0..forests {
            enter(&mut s, "Forest");
        }
        for name in ["Island", "Prophetic Prism", "Island", "Mountain"] {
            enter(&mut s, name);
        }
        s.set_player_mana_pool(0, "0");
        let professor = card(&s, 0, "Gnarled Professor");
        s.game_mut().move_card(professor, ZoneType::Hand);
        s.game_mut().scenario_refresh_priority().unwrap();
        assert_eq!(can_cast(&s, professor), forests > 0);
        if forests == 0 {
            continue;
        }
        choose(
            &mut s,
            |a| matches!(a, Action::CastSpell { card, .. } if *card == professor),
        );
        let root = witness(s.game());
        let driver = ClonePlusUndoDriver::default();
        let mut branch = driver.fork_exact(s.game());
        complete_every_offer(&mut branch, professor, 3);
        assert_eq!(witness(s.game()), root);
        assert_eq!(witness(&branch), root);
    }
}

#[test]
fn prism_upfront_payment_can_preserve_mana_already_in_the_pool() {
    let mut s = setup(&["Prophetic Prism", "Gnarled Professor"]);
    for name in ["Prophetic Prism", "Island", "Island", "Mountain"] {
        enter(&mut s, name);
    }
    s.set_player_mana_pool(0, "G");
    let professor = cast(&mut s, "Gnarled Professor");
    activate(&mut s, "Prophetic Prism", 0);
    assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("G"));
    choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 4, .. }));
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    assert_eq!(
        s.game().state.zones.zone_of(professor),
        Some(ZoneType::Stack)
    );
}

#[test]
fn prism_upfront_payment_search_preserves_a_required_pooled_white() {
    let mut s = setup(&["Prophetic Prism", "Serra Angel"]);
    enter(&mut s, "Prophetic Prism");
    s.set_player_mana_pool(0, "WUUUU");
    let angel = cast(&mut s, "Serra Angel");
    activate(&mut s, "Prophetic Prism", 0);
    assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("WUUU"));
    assert_eq!(s.action_space().actions.len(), 1);
    choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 0, .. }));
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    assert_eq!(s.game().state.zones.zone_of(angel), Some(ZoneType::Stack));
}

// Exhaustive tiny-board proof: expected accounting below is printed-cost
// arithmetic, not a second call to the feasibility model.
#[test]
fn nested_mana_every_offer_completes_replays_and_rolls_back() {
    use managym::{
        search_state::{witness, BenchCommand, BranchDriver, ClonePlusUndoDriver},
        state::game_object::PermanentId,
        Game,
    };

    struct Expected {
        spell: Option<CardId>,
        prism: PermanentId,
        treasure: PermanentId,
    }

    fn explore(
        game: &mut Game,
        root: &Game,
        expected: &Expected,
        prefix: &mut Vec<usize>,
        nodes: &mut usize,
    ) {
        *nodes += 1;
        assert!(*nodes <= 256, "tiny board node bound");
        assert!(prefix.len() <= 8, "one Treasure, one filter, one spell");
        let driver = ClonePlusUndoDriver::default();
        let complete = if let Some(spell) = expected.spell {
            game.state.zones.zone_of(spell) == Some(ZoneType::Stack)
        } else {
            game.state.suspended_decision.is_none()
        };
        if complete {
            // Treasure contributes exactly one; Prism spends one then adds one.
            // Goblin spends the remaining R. Mana abilities add no stack object.
            assert!(game.state.permanents[expected.treasure].is_none());
            let prism_tapped = game.state.permanents[expected.prism]
                .as_ref()
                .unwrap()
                .tapped;
            assert_eq!(
                game.state.players[0].mana_pool.total(),
                u16::from(expected.spell.is_none())
            );
            assert_eq!(
                game.state.stack_objects.len(),
                usize::from(expected.spell.is_some())
            );
            if expected.spell.is_none() {
                assert!(prism_tapped);
            }
            let mut replay = driver.fork_exact(root);
            for action_index in prefix.iter().copied() {
                driver
                    .apply(
                        &mut replay,
                        BenchCommand {
                            action_index,
                            expected_action_hash: None,
                            expected_state_hash: None,
                        },
                    )
                    .unwrap();
            }
            assert_eq!(
                witness(&replay),
                witness(game),
                "complete Command prefix replay"
            );
            return;
        }
        let before = witness(game);
        let actions = game.action_space().unwrap().actions.clone();
        assert!(
            !actions.is_empty(),
            "an offered transition must have a continuation"
        );
        for action_index in 0..actions.len() {
            let mark = driver.mark(game);
            driver
                .apply(
                    game,
                    BenchCommand {
                        action_index,
                        expected_action_hash: None,
                        expected_state_hash: None,
                    },
                )
                .unwrap();
            prefix.push(action_index);
            explore(game, root, expected, prefix, nodes);
            prefix.pop();
            driver.rollback(game, mark);
            assert_eq!(
                witness(game),
                before,
                "exact pools, objects, queues, RNG and offers"
            );
        }
    }

    for spell in [false, true] {
        let mut s = setup(&["Prophetic Prism", "Tireless Provisioner", "Raging Goblin"]);
        enter(&mut s, "Prophetic Prism");
        s.game_mut().create_token("Treasure", PlayerId(0), false);
        let prism = permanent(&s, "Prophetic Prism");
        let treasure = permanent(&s, "Treasure");
        s.set_player_mana_pool(0, "0");
        s.game_mut().scenario_refresh_priority().unwrap();
        let spell = if spell {
            Some(cast(&mut s, "Raging Goblin"))
        } else {
            activate(&mut s, "Prophetic Prism", 0);
            // The upfront cost precedes both tap and output.
            assert!(!s.game().state.permanents[prism].as_ref().unwrap().tapped);
            assert!(s.game().state.permanents[treasure].is_some());
            assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
            None
        };
        let root = s.game().clone();
        let original = witness(&root);
        let mut branch = root.clone();
        let mut nodes = 0;
        explore(
            &mut branch,
            &root,
            &Expected {
                spell,
                prism,
                treasure,
            },
            &mut Vec::new(),
            &mut nodes,
        );
        assert!(nodes >= 10, "visible color alternatives were explored");
        assert_eq!(witness(&root), original, "sibling isolation");
    }
}

#[test]
fn filters_require_a_real_seed_and_do_not_reuse_a_reserved_source() {
    for seed in [false, true] {
        let mut s = setup(&["Prophetic Prism", "Tireless Provisioner"]);
        // Use two distinct printed copies; identical output does not make
        // their tap and sacrifice identities interchangeable.
        enter(&mut s, "Prophetic Prism");
        enter(&mut s, "Prophetic Prism");
        if seed {
            s.game_mut().create_token("Treasure", PlayerId(0), false);
        }
        s.set_player_mana_pool(0, "0");
        s.game_mut().scenario_refresh_priority().unwrap();
        let filters: Vec<_> = s
            .action_space()
            .actions
            .iter()
            .filter_map(|action| match action {
                Action::ActivateAbility { permanent, .. }
                    if s.game().state.cards
                        [s.game().state.permanents[*permanent].as_ref().unwrap().card]
                        .name
                        == "Prophetic Prism" =>
                {
                    Some(*permanent)
                }
                _ => None,
            })
            .collect();
        assert_eq!(filters.len(), if seed { 2 } else { 0 });
        if seed {
            choose(
                &mut s,
                |a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == filters[0]),
            );
            // The unpaid filter is reserved, and the other seedless filter
            // cannot spend Treasure and leave the original payment unpaid.
            let actions = &s.action_space().actions;
            assert!(!actions.iter().any(|a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == filters[0])));
            activate(&mut s, "Treasure", 0);
            choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 1, .. }));
            assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
            choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 4, .. }));
            assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("G"));
            assert!(s.game().state.stack_objects.is_empty());
        }
    }
}

#[test]
fn prism_pool_alternatives_keep_restrictions_and_expiring_mana_classes() {
    for blue in [2, 3] {
        let mut s = setup(&["Prophetic Prism", "Serra Angel"]);
        enter(&mut s, "Prophetic Prism");
        s.set_player_mana_pool(0, &format!("W{}", "U".repeat(blue)));
        s.game_mut().state.players[0].combat_mana_pool = Mana::parse("U");
        let angel = card(&s, 0, "Serra Angel");
        s.game_mut().move_card(angel, ZoneType::Hand);
        s.game_mut().scenario_refresh_priority().unwrap();
        // Five total mana is necessary: the filter conserves total mana.
        assert_eq!(can_cast(&s, angel), blue == 3);
        if blue == 3 {
            choose(
                &mut s,
                |a| matches!(a, Action::CastSpell { card, .. } if *card == angel),
            );
            activate(&mut s, "Prophetic Prism", 0);
            assert_eq!(s.game().state.players[0].combat_mana_pool.total(), 0);
            assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("WUUU"));
            choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 0, .. }));
            choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
            assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
        }
    }
}

#[test]
fn filtering_cannot_launder_restricted_mana_from_a_mixed_pool() {
    for unrestricted in [false, true] {
        let mut s = setup(&["Prophetic Prism"]);
        enter(&mut s, "Prophetic Prism");
        s.set_player_mana_pool(0, if unrestricted { "U" } else { "0" });
        let lesson = ManaUse::spell(["Lesson"]);
        let restricted = Mana::restricted_to(lesson, [0, 1, 0, 0, 0, 0]);
        s.game_mut().state.players[0].mana_pool.add(&restricted);
        s.game_mut().scenario_refresh_priority().unwrap();
        let prism = permanent(&s, "Prophetic Prism");
        assert_eq!(s.action_space().actions.iter().any(|action|
            matches!(action, Action::ActivateAbility { permanent, .. } if *permanent == prism)
        ), unrestricted);
        if unrestricted {
            activate(&mut s, "Prophetic Prism", 0);
            assert_eq!(s.game().state.players[0].mana_pool.mana[1], 0);
            assert_eq!(
                s.game().state.players[0].mana_pool.restricted(),
                restricted.restricted()
            );
            choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 4, .. }));
            assert_eq!(s.game().state.players[0].mana_pool.mana[4], 1);
            assert_eq!(
                s.game().state.players[0].mana_pool.restricted(),
                restricted.restricted()
            );
        }
    }
}

#[test]
fn nested_mana_nonmana_life_cost_is_reserved_and_paid_once() {
    for life in [1, 3] {
        let mut s = setup(&["Prophetic Prism", "Tireless Provisioner"]);
        enter(&mut s, "Prophetic Prism");
        s.game_mut().create_token("Treasure", PlayerId(0), false);
        let treasure = permanent(&s, "Treasure");
        let source_card = s.game().state.permanents[treasure].as_ref().unwrap().card;
        s.game_mut().state.cards[source_card].activated_abilities[0].life_cost = 2;
        s.game_mut().state.players[0].life = life;
        s.set_player_mana_pool(0, "0");
        s.game_mut().scenario_refresh_priority().unwrap();
        let prism = permanent(&s, "Prophetic Prism");
        assert_eq!(
            s.action_space().actions.iter().any(
                |a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == prism)
            ),
            life >= 2
        );
        if life >= 2 {
            activate(&mut s, "Prophetic Prism", 0);
            activate(&mut s, "Treasure", 0);
            assert_eq!(s.game().state.players[0].life, 1);
            choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 1, .. }));
            choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 4, .. }));
            assert_eq!(s.game().state.players[0].life, 1);
            assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("G"));
        }
    }
}
