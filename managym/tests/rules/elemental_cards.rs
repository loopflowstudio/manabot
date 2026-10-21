// elemental_cards.rs
// Directed play evidence for the Elemental native combat and Learn operations.

use std::collections::{BTreeMap, BTreeSet};

use super::helpers::Scenario;
use managym::{
    agent::{
        action::{Action, ActionSpaceKind},
        observation::Observation,
    },
    flow::turn::StepKind,
    state::{
        game_object::{CardId, PermanentId, PlayerId, Target},
        zone::ZoneType,
    },
};

pub(super) fn setup(names: &[&str]) -> Scenario {
    let mut deck: BTreeMap<String, usize> = ["Plains", "Island", "Swamp", "Mountain", "Forest"]
        .into_iter()
        .map(|name| (name.into(), 16))
        .collect();
    for name in names {
        deck.insert((*name).into(), 4);
    }
    let mut s = Scenario::new(deck.clone(), deck, 161);
    s.advance_to_active_step(0, StepKind::Main);
    s.game_mut().scenario_clear_hand(PlayerId(0));
    s.game_mut().scenario_clear_hand(PlayerId(1));
    s.set_player_mana_pool(0, "20WWWWUUUUBBBBRRRRGGGG");
    s.set_player_mana_pool(1, "20WWWWUUUUBBBBRRRRGGGG");
    s
}

pub(super) fn card(s: &Scenario, player: usize, name: &str) -> CardId {
    s.game()
        .state
        .cards
        .iter()
        .enumerate()
        .find_map(|(i, c)| {
            (c.name == name
                && c.owner == PlayerId(player)
                && s.game().state.zones.zone_of(CardId(i)) == Some(ZoneType::Library))
            .then_some(CardId(i))
        })
        .unwrap_or_else(|| panic!("no {name}"))
}

pub(super) fn cast(s: &mut Scenario, name: &str) -> CardId {
    let id = card(s, 0, name);
    s.game_mut().move_card(id, ZoneType::Hand);
    s.game_mut().scenario_refresh_priority().unwrap();
    let index = s
        .action_space()
        .actions
        .iter()
        .position(|a| matches!(a,Action::CastSpell { card, .. } if *card == id))
        .unwrap_or_else(|| panic!("cannot cast {name}: {:?}", s.action_space()));
    s.step_action(index);
    id
}

pub(super) fn settle(s: &mut Scenario) {
    for _ in 0..100 {
        if s.action_space().kind == ActionSpaceKind::Priority
            && s.game().state.stack_objects.is_empty()
            && s.game().state.suspended_decision.is_none()
        {
            return;
        }
        let index = s
            .action_space()
            .actions
            .iter()
            .position(|a| matches!(a, Action::PassPriority { .. } | Action::Decline { .. }))
            .unwrap_or(0);
        s.step_action(index);
    }
    panic!("resolution did not settle")
}

pub(super) fn pass_until(s: &mut Scenario, kind: ActionSpaceKind) {
    for _ in 0..10 {
        if s.action_space().kind == kind {
            return;
        }
        s.pass_priority();
    }
    assert_eq!(s.action_space().kind, kind);
}

pub(super) fn permanent(s: &Scenario, name: &str) -> PermanentId {
    s.battlefield_permanents_named(0, name)[0]
}

pub(super) fn counters(s: &Scenario, id: PermanentId) -> i32 {
    s.game().state.permanents[id]
        .as_ref()
        .unwrap()
        .plus1_counters
}

#[test]
fn professors_enter_and_offer_learn_before_priority_returns() {
    for name in ["Professor of Symbology", "Gnarled Professor"] {
        let mut s = setup(&[name]);
        cast(&mut s, name);
        pass_until(&mut s, ActionSpaceKind::Learn);
        assert_eq!(s.battlefield_permanents_named(0, name).len(), 1);
        settle(&mut s);
    }
}

#[test]
fn hybrid_discard_is_optional_at_resolution_and_draws_exactly_two() {
    for discard in [false, true] {
        let mut s = setup(&["Abandon Attachments"]);
        let fodder = card(&s, 0, "Plains");
        s.game_mut().move_card(fodder, ZoneType::Hand);
        s.set_player_mana_pool(0, "1R");
        cast(&mut s, "Abandon Attachments");
        s.pass_priority();
        s.pass_priority();
        assert_eq!(s.action_space().kind, ActionSpaceKind::Discard);
        let owner = Observation::for_player(s.game(), PlayerId(0));
        let other = Observation::for_player(s.game(), PlayerId(1));
        assert!(owner.agent_cards.iter().any(|c| c.name == "Plains"));
        assert!(!other
            .opponent_cards
            .iter()
            .any(|c| c.name == "Plains" && c.zone == ZoneType::Hand));
        let index = s
            .action_space()
            .actions
            .iter()
            .position(|a| {
                if discard {
                    matches!(a,Action::SelectCard {card,..} if *card==fodder)
                } else {
                    matches!(a, Action::Decline { .. })
                }
            })
            .unwrap();
        s.step_action(index);
        settle(&mut s);
        assert_eq!(s.zone_size(0, ZoneType::Hand), if discard { 2 } else { 1 });
        assert_eq!(
            s.game().state.zones.zone_of(fodder),
            Some(if discard {
                ZoneType::Graveyard
            } else {
                ZoneType::Hand
            })
        );
    }
}

#[test]
fn library_search_is_private_enters_tapped_and_continues_to_learn() {
    for (name, forest) in [("Field Trip", true), ("Shared Roots", false)] {
        let mut s = setup(&[name]);
        cast(&mut s, name);
        s.pass_priority();
        s.pass_priority();
        assert_eq!(s.action_space().kind, ActionSpaceKind::LookAndSelect);
        let owner = Observation::for_player(s.game(), PlayerId(0));
        let other = Observation::for_player(s.game(), PlayerId(1));
        assert!(owner
            .agent_cards
            .iter()
            .any(|c| c.zone == ZoneType::Library));
        assert!(other
            .opponent_cards
            .iter()
            .all(|c| c.zone != ZoneType::Library));
        let selected = s
            .action_space()
            .actions
            .iter()
            .find_map(|a| {
                if let Action::SelectCard { card, .. } = a {
                    Some(*card)
                } else {
                    None
                }
            })
            .unwrap();
        if forest {
            assert_eq!(s.game().state.cards[selected].name, "Forest");
        }
        let index = s
            .action_space()
            .actions
            .iter()
            .position(|a| matches!(a,Action::SelectCard {card,..} if *card==selected))
            .unwrap();
        s.step_action(index);
        let id = s.game().state.card_to_permanent[selected].unwrap();
        assert!(s.game().state.permanents[id].as_ref().unwrap().tapped);
        assert_eq!(
            s.action_space().kind,
            if forest {
                ActionSpaceKind::Learn
            } else {
                ActionSpaceKind::Priority
            }
        );
        settle(&mut s);
    }
}

#[test]
fn search_selection_or_decline_and_shuffle_restore_exactly_under_undo() {
    use managym::search_state::{BenchCommand, BranchDriver, ClonePlusUndoDriver};
    let mut s = setup(&["Field Trip"]);
    cast(&mut s, "Field Trip");
    s.pass_priority();
    s.pass_priority();
    for select in [false, true] {
        let driver = ClonePlusUndoDriver::default();
        let root = driver.witness(s.game());
        let mut branch = driver.fork_exact(s.game());
        let mark = driver.mark(&mut branch);
        let index = branch
            .action_space()
            .unwrap()
            .actions
            .iter()
            .position(|a| {
                if select {
                    matches!(a, Action::SelectCard { .. })
                } else {
                    matches!(a, Action::Decline { .. })
                }
            })
            .unwrap();
        driver
            .apply(
                &mut branch,
                BenchCommand {
                    action_index: index,
                    expected_action_hash: None,
                    expected_state_hash: None,
                },
            )
            .unwrap();
        assert_ne!(driver.witness(&branch), root);
        driver.rollback(&mut branch, mark);
        assert_eq!(driver.witness(&branch), root);
        assert_eq!(driver.witness(s.game()), root);
    }
}

#[test]
fn targeted_combat_tricks_apply_through_play() {
    for name in ["Enthusiastic Study", "Octopus Form", "Fists of Flame"] {
        let mut s = setup(&[name, "Gnarled Professor"]);
        let target = s.force_permanent_on_battlefield(0, "Gnarled Professor");
        cast(&mut s, name);
        assert!(s.choose_target(Target::Permanent(target)));
        settle(&mut s);
        let data = Observation::for_player(s.game(), PlayerId(0));
        let p = data
            .agent_permanents
            .iter()
            .find(|p| p.id == s.game().state.permanents[target].as_ref().unwrap().id.0 as i32)
            .unwrap();
        if name == "Octopus Form" {
            assert!(p.keywords.hexproof);
            assert_eq!(s.game().effective_power(target), 6);
        } else {
            assert!(p.keywords.trample);
            assert!(s.game().effective_power(target) > 5);
        }
        if name == "Enthusiastic Study" {
            assert_eq!(s.game().effective_power(target), 8);
        }
    }
}

#[test]
fn prowess_resolves_above_the_noncreature_spell() {
    let mut s = setup(&["Monastery Swiftspear", "Abandon Attachments"]);
    let swift = s.force_permanent_on_battlefield(0, "Monastery Swiftspear");
    cast(&mut s, "Abandon Attachments");
    s.pass_priority();
    s.pass_priority();
    assert_eq!(s.game().effective_power(swift), 2);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
    assert_eq!(s.game().state.stack_objects.len(), 1);
    settle(&mut s);
}

#[test]
fn hybrid_draw_payoffs_use_draw_count_and_trigger_on_the_second_draw_only() {
    let mut s = setup(&[
        "Messenger Hawk",
        "Knowledge Seeker",
        "Foggy Swamp Spirit Keeper",
        "Spectral Sailor",
    ]);
    let hawk = s.force_permanent_on_battlefield(0, "Messenger Hawk");
    let fox = s.force_permanent_on_battlefield(0, "Knowledge Seeker");
    s.force_permanent_on_battlefield(0, "Foggy Swamp Spirit Keeper");
    let sailor = s.force_permanent_on_battlefield(0, "Spectral Sailor");
    for draw in 1..=3 {
        s.game_mut().scenario_refresh_priority().unwrap();
        let index = s
            .action_space()
            .actions
            .iter()
            .position(|a| matches!(a,Action::ActivateAbility{permanent,..} if *permanent==sailor))
            .unwrap();
        s.step_action(index);
        settle(&mut s);
        assert_eq!(counters(&s, fox), if draw >= 2 { 1 } else { 0 });
        assert_eq!(
            s.battlefield_permanents_named(0, "Spirit").len(),
            if draw >= 2 { 1 } else { 0 }
        );
        assert_eq!(
            s.game().effective_power(hawk),
            if draw >= 2 { 3 } else { 1 }
        );
    }
}

#[test]
fn eyetwitch_and_poxbearer_deaths_keep_their_different_rewards() {
    for name in ["Eyetwitch", "Pretending Poxbearers", "Knowledge Seeker"] {
        let mut s = setup(&[name, "Lightning Strike"]);
        let target = s.force_permanent_on_battlefield(0, name);
        cast(&mut s, "Lightning Strike");
        assert!(s.choose_target(Target::Permanent(target)));
        settle(&mut s);
        assert!(s.game().state.permanents[target].is_none());
        match name {
            "Pretending Poxbearers" => {
                assert_eq!(s.battlefield_permanents_named(0, "Ally").len(), 1)
            }
            "Knowledge Seeker" => assert_eq!(s.battlefield_permanents_named(0, "Clue").len(), 1),
            _ => assert_eq!(s.battlefield_permanents_named(0, "Ally").len(), 0),
        }
    }
}

#[test]
fn savor_makes_food_with_a_real_tap_cost_and_cindermaw_prevents_its_life_gain() {
    for blocked in [false, true] {
        let mut s = setup(&["Savor", "Gnarled Professor", "Giant Cindermaw"]);
        let target = s.force_permanent_on_battlefield(1, "Gnarled Professor");
        if blocked {
            s.force_permanent_on_battlefield(1, "Giant Cindermaw");
        }
        cast(&mut s, "Savor");
        assert!(s.choose_target(Target::Permanent(target)));
        settle(&mut s);
        assert_eq!(s.game().effective_toughness(target), 2);
        let food = permanent(&s, "Food");
        s.game_mut().state.permanents[food].as_mut().unwrap().tapped = true;
        s.game_mut().scenario_refresh_priority().unwrap();
        assert!(!s
            .action_space()
            .actions
            .iter()
            .any(|a| matches!(a,Action::ActivateAbility{permanent,..} if *permanent==food)));
        s.game_mut().state.permanents[food].as_mut().unwrap().tapped = false;
        s.game_mut().scenario_refresh_priority().unwrap();
        let index = s
            .action_space()
            .actions
            .iter()
            .position(|a| matches!(a,Action::ActivateAbility{permanent,..} if *permanent==food))
            .unwrap();
        let life = s.life(0);
        s.step_action(index);
        assert!(s.game().state.permanents[food].is_none());
        settle(&mut s);
        assert_eq!(s.life(0), life + if blocked { 0 } else { 3 });
    }
}

#[test]
fn cat_gator_uses_swamps_and_lifelink_on_its_enter_trigger() {
    let mut s = setup(&["Cat-Gator"]);
    s.force_permanent_on_battlefield(0, "Swamp");
    s.force_permanent_on_battlefield(0, "Swamp");
    cast(&mut s, "Cat-Gator");
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    settle(&mut s);
    assert_eq!(s.life(1), 18);
    assert_eq!(s.life(0), 22);
}

#[test]
fn court_official_and_ozai_make_the_target_player_choose_discards() {
    for name in ["Corrupt Court Official", "Ozai's Cruelty"] {
        let mut s = setup(&[name]);
        for _ in 0..3 {
            let id = card(&s, 1, "Plains");
            s.game_mut().move_card(id, ZoneType::Hand);
        }
        cast(&mut s, name);
        if name == "Corrupt Court Official" {
            pass_until(&mut s, ActionSpaceKind::ChooseTarget);
        }
        assert!(s.choose_target(Target::Player(PlayerId(1))));
        pass_until(&mut s, ActionSpaceKind::Discard);
        assert_eq!(s.action_space().player, Some(PlayerId(1)));
        settle(&mut s);
        assert_eq!(
            s.zone_size(1, ZoneType::Hand),
            if name == "Corrupt Court Official" {
                2
            } else {
                1
            }
        );
        assert_eq!(
            s.life(1),
            if name == "Corrupt Court Official" {
                20
            } else {
                18
            }
        );
    }
}

#[test]
fn boomerang_draws_only_when_its_target_was_controlled_by_the_caster() {
    for player in [0, 1] {
        let mut s = setup(&["Boomerang Basics", "Gnarled Professor"]);
        let target = s.force_permanent_on_battlefield(player, "Gnarled Professor");
        let target_card = s.game().state.permanents[target].as_ref().unwrap().card;
        cast(&mut s, "Boomerang Basics");
        assert!(s.choose_target(Target::Permanent(target)));
        settle(&mut s);
        assert_eq!(
            s.game().state.zones.zone_of(target_card),
            Some(ZoneType::Hand)
        );
        assert_eq!(
            s.zone_size(0, ZoneType::Hand),
            if player == 0 { 2 } else { 0 }
        );
    }
}

#[test]
fn removal_respects_mana_value_and_excluded_creature_types() {
    for name in ["Power Word Kill", "Epic Downfall"] {
        let mut s = setup(&[name, "Eyetwitch", "Gnarled Professor"]);
        let small = s.force_permanent_on_battlefield(1, "Eyetwitch");
        let big = s.force_permanent_on_battlefield(1, "Gnarled Professor");
        if name == "Power Word Kill" {
            let card = s.game().state.permanents[small].as_ref().unwrap().card;
            s.game_mut().state.cards[card]
                .subtypes
                .push("Dragon".into());
        }
        cast(&mut s, name);
        assert!(!s.action_space().actions.iter().any(|action| matches!(
            action,
            Action::ChooseTarget { target: managym::state::target::Target::Permanent(id), .. } if *id == small
        )));
        let target_card = s.game().state.permanents[big].as_ref().unwrap().card;
        assert!(s.choose_target(Target::Permanent(big)));
        settle(&mut s);
        assert_eq!(
            s.game().state.zones.zone_of(target_card),
            Some(if name == "Epic Downfall" {
                ZoneType::Exile
            } else {
                ZoneType::Graveyard
            })
        );
    }
}

#[test]
fn toph_enters_earthbends_and_counts_counters_on_lands() {
    let mut s = setup(&["Toph, the Blind Bandit"]);
    let land = s.force_permanent_on_battlefield(0, "Forest");
    cast(&mut s, "Toph, the Blind Bandit");
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(land)));
    settle(&mut s);
    assert_eq!(counters(&s, land), 2);
    assert_eq!(
        s.game()
            .effective_power(permanent(&s, "Toph, the Blind Bandit")),
        2
    );
}

#[test]
fn ally_entry_context_buffs_the_entering_ally_and_haru_earthbends() {
    let mut s = setup(&[
        "Wartime Protestors",
        "Haru, Hidden Talent",
        "Pretending Poxbearers",
    ]);
    s.force_permanent_on_battlefield(0, "Wartime Protestors");
    s.force_permanent_on_battlefield(0, "Haru, Hidden Talent");
    let land = s.force_permanent_on_battlefield(0, "Forest");
    cast(&mut s, "Pretending Poxbearers");
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(land)));
    settle(&mut s);
    let ally = permanent(&s, "Pretending Poxbearers");
    assert_eq!(counters(&s, ally), 1);
    assert_eq!(counters(&s, land), 1);
    let view = Observation::for_player(s.game(), PlayerId(0));
    assert!(
        view.agent_permanents
            .iter()
            .find(|p| p.id == s.game().state.permanents[ally].as_ref().unwrap().id.0 as i32)
            .unwrap()
            .keywords
            .haste
    );
}

#[test]
fn uncle_iroh_reduces_lesson_spells_without_reducing_ordinary_spells() {
    let mut s = setup(&["Uncle Iroh", "Abandon Attachments", "Lightning Strike"]);
    s.force_permanent_on_battlefield(0, "Uncle Iroh");
    let lesson = card(&s, 0, "Abandon Attachments");
    let ordinary = card(&s, 0, "Lightning Strike");
    s.game_mut().move_card(lesson, ZoneType::Hand);
    s.game_mut().move_card(ordinary, ZoneType::Hand);
    s.set_player_mana_pool(0, "R");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a,Action::CastSpell{card,..} if *card==lesson)));
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a,Action::CastSpell{card,..} if *card==ordinary)));
}

#[test]
fn messenger_hawk_accepts_either_hybrid_color_and_investigates_on_entry() {
    for pool in ["2U", "2B"] {
        let mut s = setup(&["Messenger Hawk"]);
        s.set_player_mana_pool(0, pool);
        cast(&mut s, "Messenger Hawk");
        settle(&mut s);
        assert_eq!(s.battlefield_permanents_named(0, "Clue").len(), 1);
        assert_eq!(s.game().effective_power(permanent(&s, "Messenger Hawk")), 1);
    }
}

#[test]
fn spectral_sailor_draw_ability_survives_the_sources_destruction() {
    let mut s = setup(&["Spectral Sailor", "Lightning Strike"]);
    let sailor = s.force_permanent_on_battlefield(0, "Spectral Sailor");
    s.game_mut().scenario_refresh_priority().unwrap();
    let index = s
        .action_space()
        .actions
        .iter()
        .position(|a| matches!(a,Action::ActivateAbility{permanent,..} if *permanent==sailor))
        .unwrap();
    s.step_action(index);
    cast(&mut s, "Lightning Strike");
    assert!(s.choose_target(Target::Permanent(sailor)));
    settle(&mut s);
    assert!(s.game().state.permanents[sailor].is_none());
    assert_eq!(s.zone_size(0, ZoneType::Hand), 1);
}

#[test]
fn tiger_dillo_needs_another_four_power_creature_to_be_an_attacker() {
    for enabled in [false, true] {
        let mut s = setup(&["Tiger-Dillo", "Gnarled Professor"]);
        let tiger = s.force_permanent_on_battlefield(0, "Tiger-Dillo");
        s.game_mut().state.permanents[tiger]
            .as_mut()
            .unwrap()
            .summoning_sick = false;
        if enabled {
            let other = s.force_permanent_on_battlefield(0, "Gnarled Professor");
            s.game_mut().state.permanents[other]
                .as_mut()
                .unwrap()
                .summoning_sick = false;
        }
        s.advance_to_active_step(0, StepKind::BeginningOfCombat);
        s.pass_priority();
        s.pass_priority();
        let mut offered = false;
        for _ in 0..3 {
            if s.action_space().kind != ActionSpaceKind::DeclareAttacker {
                break;
            }
            offered |= s.action_space().actions.iter().any(|action| {
                matches!(
                    action,
                    Action::DeclareAttacker { permanent, attack: true, .. } if *permanent == tiger
                )
            });
            s.decline_attack();
        }
        assert_eq!(offered, enabled);
    }
}

#[test]
fn master_piandao_reveals_a_selected_ally_and_bottoms_the_rest() {
    let mut s = setup(&[
        "Master Piandao",
        "Pretending Poxbearers",
        "Abandon Attachments",
    ]);
    let master = s.force_permanent_on_battlefield(0, "Master Piandao");
    s.game_mut().state.permanents[master]
        .as_mut()
        .unwrap()
        .summoning_sick = false;
    let top = pin_top(
        &mut s,
        &[
            "Plains",
            "Pretending Poxbearers",
            "Abandon Attachments",
            "Island",
        ],
    );
    let ally = top[1];
    s.advance_to_active_step(0, StepKind::BeginningOfCombat);
    s.pass_priority();
    s.pass_priority();
    s.declare_attack();
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    assert_eq!(s.action_space().kind, ActionSpaceKind::LookAndSelect);
    assert_eq!(
        s.action_space()
            .actions
            .iter()
            .filter(|a| matches!(a, Action::SelectCard { .. }))
            .count(),
        2
    );
    let index = s
        .action_space()
        .actions
        .iter()
        .position(|a| matches!(a,Action::SelectCard{card,..} if *card==ally))
        .unwrap();
    s.step_action(index);
    settle(&mut s);
    assert_eq!(s.game().state.zones.zone_of(ally), Some(ZoneType::Hand));
    let definition = s.game().state.cards[ally].definition_id.0;
    let other = Observation::for_player(s.game(), PlayerId(1));
    assert_eq!(other.opponent.known_hand.get(&definition), Some(&1));
    assert!(other
        .opponent_cards
        .iter()
        .all(|c| c.zone != ZoneType::Hand));
}

#[test]
fn resampled_private_search_choices_still_name_library_cards() {
    let mut s = setup(&["Field Trip"]);
    cast(&mut s, "Field Trip");
    s.pass_priority();
    s.pass_priority();
    for viewer in [PlayerId(0), PlayerId(1)] {
        for seed in 0..5 {
            let mut branch = s.game().clone();
            branch.determinize(viewer, seed);
            let expected: BTreeSet<_> = branch
                .state
                .zones
                .zone_cards(ZoneType::Library, PlayerId(0))
                .iter()
                .filter(|card| branch.state.cards[**card].name == "Forest")
                .map(|card| branch.state.cards[*card].id.0 as i32)
                .collect();
            let offered: BTreeSet<_> = branch
                .action_space()
                .unwrap()
                .actions
                .iter()
                .filter_map(|action| match action {
                    Action::SelectCard { card, .. } => Some(branch.state.cards[*card].id.0 as i32),
                    _ => None,
                })
                .collect();
            assert_eq!(offered, expected);
            let owner = Observation::for_player(&branch, PlayerId(0));
            let visible = owner
                .agent_cards
                .iter()
                .filter(|card| card.zone == ZoneType::Library)
                .map(|card| card.id)
                .collect::<BTreeSet<_>>();
            assert_eq!(visible, expected);
        }
    }
}

#[test]
fn library_search_semantic_command_replays_and_rejects_reuse_without_mutation() {
    use managym::decision::Command;
    use managym::search_state::witness;
    let mut s = setup(&["Field Trip"]);
    cast(&mut s, "Field Trip");
    s.pass_priority();
    s.pass_priority();
    let mut replay = s.game().clone();
    let frame = s.game().semantic_decision_frame().unwrap();
    let command = Command {
        command_id: "elemental-search".into(),
        expected_revision: frame.revision,
        offer_id: frame.offers[0].id.0,
        answers: vec![],
        object_preconditions: vec![],
    };
    s.game_mut().execute_semantic_command(&command).unwrap();
    replay.execute_semantic_command(&command).unwrap();
    assert_eq!(witness(s.game()), witness(&replay));
    let after = witness(s.game());
    assert!(s.game_mut().execute_semantic_command(&command).is_err());
    assert_eq!(witness(s.game()), after);
}

#[test]
fn spirit_tokens_cannot_block_or_be_blocked_by_nonspirits() {
    for spirit_attacking in [false, true] {
        let mut s = setup(&["Gnarled Professor"]);
        s.game_mut().create_token("Spirit", PlayerId(0), false);
        s.game_mut().create_token("Spirit", PlayerId(1), false);
        let spirit = s.battlefield_permanents_named(0, "Spirit")[0];
        let defender = s.battlefield_permanents_named(1, "Spirit")[0];
        let ordinary = s.force_permanent_on_battlefield(0, "Gnarled Professor");
        let other = s.force_permanent_on_battlefield(1, "Gnarled Professor");
        for id in [spirit, ordinary] {
            s.game_mut().state.permanents[id]
                .as_mut()
                .unwrap()
                .summoning_sick = false;
        }
        s.advance_to_active_step(0, StepKind::BeginningOfCombat);
        s.pass_priority();
        s.pass_priority();
        let attacker = if spirit_attacking { spirit } else { ordinary };
        for _ in 0..2 {
            if s.action_space().kind != ActionSpaceKind::DeclareAttacker {
                break;
            }
            let index = s.action_space().actions.iter().position(|action| matches!(
                action,
                Action::DeclareAttacker { permanent, attack, .. } if (*permanent == attacker) == *attack
            )).unwrap();
            s.step_action(index);
        }
        s.advance_to_active_step(0, StepKind::DeclareBlockers);
        let mut checked = 0;
        while s.action_space().kind == ActionSpaceKind::DeclareBlocker {
            let blocker = s
                .action_space()
                .actions
                .iter()
                .find_map(|a| {
                    if let Action::DeclareBlocker { blocker, .. } = a {
                        Some(*blocker)
                    } else {
                        None
                    }
                })
                .unwrap();
            let allowed =
                s.action_space().actions.iter().any(
                    |a| matches!(a,Action::DeclareBlocker{attacker:Some(id),..} if *id==attacker),
                );
            assert_eq!(
                allowed,
                if spirit_attacking {
                    blocker == defender
                } else {
                    blocker == other
                }
            );
            checked += 1;
            s.decline_block();
        }
        assert_eq!(checked, 2);
    }
}

fn mode(s: &mut Scenario, wanted: usize) {
    assert_eq!(s.action_space().kind, ActionSpaceKind::Modal);
    let index = s
        .action_space()
        .actions
        .iter()
        .position(|a| matches!(a, Action::ChooseMode { mode, .. } if *mode == wanted))
        .unwrap();
    s.step_action(index);
}

fn pin_top(s: &mut Scenario, names: &[&str]) -> Vec<CardId> {
    let mut cards = Vec::new();
    for name in names {
        let next = s
            .game()
            .state
            .zones
            .zone_cards(ZoneType::Library, PlayerId(0))
            .iter()
            .copied()
            .find(|id| s.game().state.cards[*id].name == *name && !cards.contains(id))
            .unwrap();
        cards.push(next);
    }
    let library = s
        .game_mut()
        .state
        .zones
        .zone_cards_mut(ZoneType::Library, PlayerId(0));
    library.retain(|id| !cards.contains(id));
    library.extend(cards.iter().rev());
    cards
}

#[test]
fn azula_announces_both_modes_then_binds_independent_targets_including_same_object() {
    for same in [false, true] {
        let mut s = setup(&["Azula Always Lies", "Gnarled Professor"]);
        let first = s.force_permanent_on_battlefield(0, "Gnarled Professor");
        let second = if same {
            first
        } else {
            s.force_permanent_on_battlefield(1, "Gnarled Professor")
        };
        let spell = cast(&mut s, "Azula Always Lies");
        assert_eq!(s.game().state.zones.zone_of(spell), Some(ZoneType::Hand));
        assert_eq!(s.action_space().actions.len(), 3);
        mode(&mut s, 2);
        assert!(s.choose_target(Target::Permanent(first)));
        assert!(s.choose_target(Target::Permanent(second)));
        settle(&mut s);
        assert_eq!(counters(&s, second), 1);
        assert_eq!(s.game().effective_power(first), if same { 5 } else { 4 });
    }
}

#[test]
fn iroh_modes_gate_targets_and_damage_only_opponent_creatures() {
    let mut s = setup(&[
        "Iroh's Demonstration",
        "Gnarled Professor",
        "Monastery Swiftspear",
    ]);
    let own = s.force_permanent_on_battlefield(0, "Gnarled Professor");
    let other = s.force_permanent_on_battlefield(1, "Gnarled Professor");
    cast(&mut s, "Iroh's Demonstration");
    mode(&mut s, 0);
    settle(&mut s);
    assert_eq!(s.game().state.permanents[own].as_ref().unwrap().damage, 0);
    assert_eq!(s.game().state.permanents[other].as_ref().unwrap().damage, 1);
    cast(&mut s, "Iroh's Demonstration");
    mode(&mut s, 1);
    assert!(s.choose_target(Target::Permanent(other)));
    settle(&mut s);
    assert!(s.game().state.permanents[other].is_none());
    let mut empty = setup(&["Iroh's Demonstration"]);
    cast(&mut empty, "Iroh's Demonstration");
    assert_eq!(empty.action_space().actions.len(), 1);
    assert!(matches!(
        empty.action_space().actions[0],
        Action::ChooseMode { mode: 0, .. }
    ));
}

#[test]
fn origin_destroys_artifacts_or_protects_with_counter_and_indestructible() {
    let mut s = setup(&[
        "Origin of Metalbending",
        "Lightning Strike",
        "Power Word Kill",
        "Eyetwitch",
    ]);
    s.game_mut().create_token("Clue", PlayerId(1), false);
    let clue = s.battlefield_permanents_named(1, "Clue")[0];
    cast(&mut s, "Origin of Metalbending");
    mode(&mut s, 0);
    assert!(s.choose_target(Target::Permanent(clue)));
    settle(&mut s);
    assert!(s.game().state.permanents[clue].is_none());
    let creature = s.force_permanent_on_battlefield(0, "Eyetwitch");
    cast(&mut s, "Origin of Metalbending");
    mode(&mut s, 1);
    assert!(s.choose_target(Target::Permanent(creature)));
    settle(&mut s);
    assert_eq!(counters(&s, creature), 1);
    for removal in ["Power Word Kill", "Lightning Strike"] {
        cast(&mut s, removal);
        assert!(s.choose_target(Target::Permanent(creature)));
        settle(&mut s);
        assert!(s.game().state.permanents[creature].is_some());
    }
}

#[test]
fn gather_creates_per_plains_and_scry_can_reverse_top_cards_privately() {
    let mut s = setup(&["Gather the White Lotus"]);
    s.force_permanent_on_battlefield(0, "Plains");
    s.force_permanent_on_battlefield(0, "Plains");
    let cards = pin_top(&mut s, &["Island", "Forest"]);
    cast(&mut s, "Gather the White Lotus");
    pass_until(&mut s, ActionSpaceKind::Scry);
    assert_eq!(s.battlefield_permanents_named(0, "Ally").len(), 2);
    assert_eq!(s.action_space().actions.len(), 4);
    let other = Observation::for_player(s.game(), PlayerId(1));
    assert!(other
        .opponent_cards
        .iter()
        .all(|c| c.zone != ZoneType::Library));
    for card in cards.iter().rev() {
        let index = s
            .action_space()
            .actions
            .iter()
            .position(
                |a| matches!(a, Action::ScryCard { card: c, to_bottom: false, .. } if c == card),
            )
            .unwrap();
        s.step_action(index);
    }
    let library = s
        .game()
        .state
        .zones
        .zone_cards(ZoneType::Library, PlayerId(0));
    assert_eq!(library[library.len() - 1], cards[1]);
    assert_eq!(library[library.len() - 2], cards[0]);
}

#[test]
fn dai_li_targets_same_land_twice_then_drains_for_creatures_with_counters() {
    let mut s = setup(&["Dai Li Agents"]);
    let land = s.force_permanent_on_battlefield(0, "Forest");
    cast(&mut s, "Dai Li Agents");
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(land)));
    assert!(s.choose_target(Target::Permanent(land)));
    settle(&mut s);
    assert_eq!(counters(&s, land), 2);
    let agents = permanent(&s, "Dai Li Agents");
    s.game_mut().state.permanents[agents]
        .as_mut()
        .unwrap()
        .summoning_sick = false;
    s.advance_to_active_step(0, StepKind::BeginningOfCombat);
    s.pass_priority();
    s.pass_priority();
    while s.action_space().kind == ActionSpaceKind::DeclareAttacker {
        let index = s.action_space().actions.iter().position(|a| matches!(a, Action::DeclareAttacker { permanent, attack, .. } if (*permanent == agents) == *attack)).unwrap();
        s.step_action(index);
    }
    settle(&mut s);
    s.assert_life(0, 21);
    s.assert_life(1, 19);
}

#[test]
fn jet_entry_damage_and_death_choose_up_to_two_distinct_creatures() {
    let mut s = setup(&[
        "Jet, Freedom Fighter",
        "Gnarled Professor",
        "Lightning Strike",
    ]);
    let own = s.force_permanent_on_battlefield(0, "Gnarled Professor");
    let other = s.force_permanent_on_battlefield(1, "Gnarled Professor");
    cast(&mut s, "Jet, Freedom Fighter");
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(other)));
    settle(&mut s);
    assert_eq!(s.game().state.permanents[other].as_ref().unwrap().damage, 2);
    let jet = permanent(&s, "Jet, Freedom Fighter");
    cast(&mut s, "Lightning Strike");
    assert!(s.choose_target(Target::Permanent(jet)));
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(own)));
    assert!(!s.action_space().actions.iter().any(|a| matches!(a, Action::ChooseTarget { target: managym::state::target::Target::Permanent(id), .. } if *id == own)));
    assert!(s.choose_target(Target::Permanent(other)));
    settle(&mut s);
    assert_eq!(counters(&s, own), 1);
    assert_eq!(counters(&s, other), 1);
}

#[test]
fn ostrich_mills_then_selects_only_a_milled_land_or_grows_when_declined() {
    for take in [false, true] {
        let mut s = setup(&["Ostrich-Horse", "Lightning Strike"]);
        let top = pin_top(&mut s, &["Lightning Strike", "Forest", "Island"]);
        cast(&mut s, "Ostrich-Horse");
        pass_until(&mut s, ActionSpaceKind::LookAndSelect);
        assert_eq!(s.action_space().actions.len(), 3);
        assert!(top
            .iter()
            .all(|card| s.game().state.zones.zone_of(*card) == Some(ZoneType::Graveyard)));
        let index = s
            .action_space()
            .actions
            .iter()
            .position(|a| {
                if take {
                    matches!(a, Action::SelectCard { card, .. } if *card == top[1])
                } else {
                    matches!(a, Action::Decline { .. })
                }
            })
            .unwrap();
        s.step_action(index);
        settle(&mut s);
        assert_eq!(
            counters(&s, permanent(&s, "Ostrich-Horse")),
            if take { 0 } else { 1 }
        );
        assert_eq!(
            s.game().state.zones.zone_of(top[1]),
            Some(if take {
                ZoneType::Hand
            } else {
                ZoneType::Graveyard
            })
        );
    }
}

#[test]
fn sokka_binds_spell_and_land_separately_and_resolves_with_one_illegal_target() {
    let mut s = setup(&["Sokka's Haiku", "Lightning Strike"]);
    let land = s.force_permanent_on_battlefield(1, "Forest");
    s.game_mut().state.permanents[land].as_mut().unwrap().tapped = true;
    let strike = cast(&mut s, "Lightning Strike");
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    cast(&mut s, "Sokka's Haiku");
    assert!(s.choose_target(Target::StackSpell(strike)));
    assert!(s.choose_target(Target::Permanent(land)));
    // Remove one target in response: draw/mill and the other clause still resolve.
    s.game_mut().move_card(strike, ZoneType::Graveyard);
    s.game_mut().state.stack_objects.retain(|object| !matches!(object, managym::state::stack_object::StackObject::Spell(spell) if spell.card == strike));
    let library = s.zone_size(0, ZoneType::Library);
    settle(&mut s);
    assert_eq!(s.zone_size(0, ZoneType::Library), library - 4);
    assert!(!s.game().state.permanents[land].as_ref().unwrap().tapped);
}

#[test]
fn zuko_exiles_and_gives_clue_to_the_targets_current_controller() {
    let mut s = setup(&["Zuko's Exile", "Gnarled Professor"]);
    let victim = s.force_permanent_on_battlefield(1, "Gnarled Professor");
    let card = s.game().state.permanents[victim].as_ref().unwrap().card;
    cast(&mut s, "Zuko's Exile");
    assert!(s.choose_target(Target::Permanent(victim)));
    settle(&mut s);
    assert_eq!(s.game().state.zones.zone_of(card), Some(ZoneType::Exile));
    assert_eq!(s.battlefield_permanents_named(1, "Clue").len(), 1);
    assert!(s.battlefield_permanents_named(0, "Clue").is_empty());
}

#[test]
fn true_ancestry_returns_only_a_current_permanent_card_and_can_decline() {
    for take in [false, true] {
        let mut s = setup(&["True Ancestry", "Lightning Strike"]);
        let land = card(&s, 0, "Forest");
        let instant = card(&s, 0, "Lightning Strike");
        s.game_mut().move_card(land, ZoneType::Graveyard);
        s.game_mut().move_card(instant, ZoneType::Graveyard);
        let object = s.game().current_object_ref(land).unwrap();
        cast(&mut s, "True Ancestry");
        assert_eq!(s.action_space().actions.len(), 2);
        if take {
            assert!(s.choose_target(Target::GraveyardCard(object)));
        } else {
            let index = s
                .action_space()
                .actions
                .iter()
                .position(|a| matches!(a, Action::Decline { .. }))
                .unwrap();
            s.step_action(index);
        }
        settle(&mut s);
        assert_eq!(
            s.game().state.zones.zone_of(land),
            Some(if take {
                ZoneType::Hand
            } else {
                ZoneType::Graveyard
            })
        );
        assert_eq!(s.battlefield_permanents_named(0, "Clue").len(), 1);
    }
}

#[test]
fn disciple_targets_another_small_creature_pays_tap_and_survives_source_departure() {
    let mut s = setup(&[
        "Deserter's Disciple",
        "Monastery Swiftspear",
        "Gnarled Professor",
    ]);
    let disciple = s.force_permanent_on_battlefield(0, "Deserter's Disciple");
    let small = s.force_permanent_on_battlefield(0, "Monastery Swiftspear");
    s.force_permanent_on_battlefield(0, "Gnarled Professor");
    s.game_mut().state.permanents[disciple]
        .as_mut()
        .unwrap()
        .summoning_sick = false;
    s.game_mut().scenario_refresh_priority().unwrap();
    let index = s
        .action_space()
        .actions
        .iter()
        .position(
            |a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == disciple),
        )
        .unwrap();
    s.step_action(index);
    assert_eq!(s.action_space().actions.len(), 1);
    assert!(s.choose_target(Target::Permanent(small)));
    let source = s.game().state.permanents[disciple].as_ref().unwrap().card;
    assert!(s.game().state.permanents[disciple].as_ref().unwrap().tapped);
    s.game_mut().move_card(source, ZoneType::Graveyard);
    // Resolve through the branch driver as well: the independent target
    // mutation must disappear when the search branch rolls back.
    assert_choice_contract(&s);
    settle(&mut s);
    assert!(
        s.game().state.permanents[small]
            .as_ref()
            .unwrap()
            .cant_be_blocked_this_turn
    );
}

#[test]
fn momo_reduces_first_matching_spell_only_and_grows_when_another_flier_enters() {
    let mut s = setup(&["Momo, Friendly Flier", "Messenger Hawk"]);
    let momo = s.force_permanent_on_battlefield(0, "Momo, Friendly Flier");
    let hawk = card(&s, 0, "Messenger Hawk");
    s.game_mut().move_card(hawk, ZoneType::Hand);
    s.set_player_mana_pool(0, "1U");
    s.game_mut().scenario_refresh_priority().unwrap();
    let index = s
        .action_space()
        .actions
        .iter()
        .position(|a| matches!(a, Action::CastSpell { card, .. } if *card == hawk))
        .unwrap();
    s.step_action(index);
    settle(&mut s);
    assert_eq!(s.game().effective_power(momo), 2);
    let next = card(&s, 0, "Messenger Hawk");
    s.game_mut().move_card(next, ZoneType::Hand);
    s.set_player_mana_pool(0, "1U");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::CastSpell { card, .. } if *card == next)));
}

/// Exercise every offered atomic continuation through the real Command
/// authority, with an independent replay and a nested exact-fork rollback.
pub(super) fn assert_choice_contract(s: &Scenario) {
    use managym::{
        decision::Command,
        search_state::{witness, BenchCommand, BranchDriver, ClonePlusUndoDriver},
    };
    let root = witness(s.game());
    let frame = s.game().semantic_decision_frame().unwrap();
    assert_eq!(frame.offers.len(), s.action_space().actions.len());
    for offer in &frame.offers {
        assert!(offer.choices.is_empty());
        let command = Command {
            command_id: format!("elemental-choice-{}", offer.id.0),
            expected_revision: frame.revision,
            offer_id: offer.id.0,
            answers: vec![],
            object_preconditions: vec![],
        };
        let mut branch = s.game().clone();
        let mut replay = s.game().clone();
        branch.execute_semantic_command(&command).unwrap();
        replay.execute_semantic_command(&command).unwrap();
        assert_eq!(witness(&branch), witness(&replay));
        let after = witness(&branch);
        assert!(branch.execute_semantic_command(&command).is_err());
        assert_eq!(witness(&branch), after);
    }
    let driver = ClonePlusUndoDriver::default();
    for index in 0..s.action_space().actions.len() {
        let mut branch = driver.fork_exact(s.game());
        let mark = driver.mark(&mut branch);
        driver
            .apply(
                &mut branch,
                BenchCommand {
                    action_index: index,
                    expected_action_hash: None,
                    expected_state_hash: None,
                },
            )
            .unwrap();
        let inner = driver.mark(&mut branch);
        if !branch.action_space().unwrap().actions.is_empty() {
            driver
                .apply(
                    &mut branch,
                    BenchCommand {
                        action_index: 0,
                        expected_action_hash: None,
                        expected_state_hash: None,
                    },
                )
                .unwrap();
        }
        driver.rollback(&mut branch, inner);
        driver.rollback(&mut branch, mark);
        assert_eq!(driver.witness(&branch), root);
    }
    assert_eq!(witness(s.game()), root);
}

#[test]
fn elemental_new_decisions_replay_reject_stale_commands_and_restore_exact_forks() {
    let mut s = setup(&["Azula Always Lies", "Gnarled Professor"]);
    let target = s.force_permanent_on_battlefield(0, "Gnarled Professor");
    cast(&mut s, "Azula Always Lies");
    assert_choice_contract(&s);
    mode(&mut s, 2);
    assert_choice_contract(&s);
    assert!(s.choose_target(Target::Permanent(target)));
    assert_choice_contract(&s);

    let mut s = setup(&["Gather the White Lotus"]);
    cast(&mut s, "Gather the White Lotus");
    pass_until(&mut s, ActionSpaceKind::Scry);
    assert_choice_contract(&s);
    s.step_action(0);
    assert_choice_contract(&s);

    let mut s = setup(&["Ostrich-Horse"]);
    pin_top(&mut s, &["Forest", "Island", "Plains"]);
    cast(&mut s, "Ostrich-Horse");
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    assert_choice_contract(&s);

    let mut s = setup(&["True Ancestry"]);
    let land = card(&s, 0, "Forest");
    s.game_mut().move_card(land, ZoneType::Graveyard);
    cast(&mut s, "True Ancestry");
    assert_choice_contract(&s);
}

#[test]
fn ancestry_does_not_follow_a_graveyard_target_into_a_new_incarnation() {
    let mut s = setup(&["True Ancestry"]);
    let land = card(&s, 0, "Forest");
    s.game_mut().move_card(land, ZoneType::Graveyard);
    let old = s.game().current_object_ref(land).unwrap();
    cast(&mut s, "True Ancestry");
    assert!(s.choose_target(Target::GraveyardCard(old)));
    s.game_mut().move_card(land, ZoneType::Exile);
    s.game_mut().move_card(land, ZoneType::Graveyard);
    assert_ne!(s.game().current_object_ref(land), Some(old));
    settle(&mut s);
    assert_eq!(
        s.game().state.zones.zone_of(land),
        Some(ZoneType::Graveyard)
    );
    assert!(
        s.battlefield_permanents_named(0, "Clue").is_empty(),
        "all targets illegal: the entire spell fails to resolve"
    );
}

#[test]
fn scry_possible_world_materialization_preserves_inspected_cards_and_order_choices() {
    use managym::possible_worlds::{MaterializeMode, PossibleWorldSpace};
    let mut s = setup(&["Gather the White Lotus"]);
    cast(&mut s, "Gather the White Lotus");
    pass_until(&mut s, ActionSpaceKind::Scry);
    for partial in [false, true] {
        if partial {
            s.step_action(0);
        }
        let view = Observation::for_player(s.game(), PlayerId(0)).to_json();
        let space = PossibleWorldSpace::for_viewer(s.game(), PlayerId(0));
        for seed in 0..5 {
            let branch = space
                .materialize_index(s.game(), 0, seed, MaterializeMode::PreserveViewerRoot)
                .unwrap();
            assert_eq!(
                Observation::for_player(&branch, PlayerId(0)).to_json(),
                view
            );
            assert_eq!(branch.action_space(), s.game().action_space());
        }
    }
}

#[test]
fn search_possible_worlds_refresh_private_candidates_without_exposing_library_cards() {
    use managym::possible_worlds::{MaterializeMode, PossibleWorldSpace};
    let mut s = setup(&["Field Trip"]);
    let hand = card(&s, 0, "Forest");
    s.game_mut().move_card(hand, ZoneType::Hand);
    cast(&mut s, "Field Trip");
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    let space = PossibleWorldSpace::for_viewer(s.game(), PlayerId(1));
    for index in 0..space.worlds().len() {
        let branch = space
            .materialize_index(
                s.game(),
                index,
                161,
                MaterializeMode::RefreshOpponentCommitment,
            )
            .unwrap();
        let observed = Observation::for_player(&branch, PlayerId(1));
        assert!(observed
            .opponent_cards
            .iter()
            .all(|card| card.zone != ZoneType::Library && card.zone != ZoneType::Hand));
        for action in &branch.action_space().unwrap().actions {
            if let Action::SelectCard { card, .. } = action {
                assert_eq!(branch.state.zones.zone_of(*card), Some(ZoneType::Library));
                assert_eq!(branch.state.cards[*card].name, "Forest");
            }
        }
    }
}

#[test]
fn momo_counts_earlier_matching_casts_before_it_entered_and_resets_next_turn() {
    let mut s = setup(&["Momo, Friendly Flier", "Messenger Hawk"]);
    cast(&mut s, "Messenger Hawk");
    settle(&mut s);
    s.force_permanent_on_battlefield(0, "Momo, Friendly Flier");
    let next = card(&s, 0, "Messenger Hawk");
    s.game_mut().move_card(next, ZoneType::Hand);
    s.set_player_mana_pool(0, "1U");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|action| matches!(action, Action::CastSpell { card, .. } if *card == next)));
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    s.set_player_mana_pool(0, "1U");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(s
        .action_space()
        .actions
        .iter()
        .any(|action| matches!(action, Action::CastSpell { card, .. } if *card == next)));
}

#[test]
fn boomerang_and_zuko_use_resolution_time_control_after_target_announcement() {
    for spell in ["Boomerang Basics", "Zuko's Exile"] {
        let mut s = setup(&[spell, "Gnarled Professor"]);
        let target = s.force_permanent_on_battlefield(1, "Gnarled Professor");
        cast(&mut s, spell);
        assert!(s.choose_target(Target::Permanent(target)));
        // The scenario seam models a completed control-change effect in response.
        s.game_mut().state.permanents[target]
            .as_mut()
            .unwrap()
            .controller = PlayerId(0);
        let hand = s.zone_size(0, ZoneType::Hand);
        settle(&mut s);
        if spell == "Boomerang Basics" {
            assert_eq!(s.zone_size(0, ZoneType::Hand), hand + 1);
            assert_eq!(s.zone_size(1, ZoneType::Hand), 1); // owner's hand, controller draws
        } else {
            assert_eq!(s.battlefield_permanents_named(0, "Clue").len(), 1);
            assert!(s.battlefield_permanents_named(1, "Clue").is_empty());
        }
    }
}

#[test]
fn announced_enter_trigger_keeps_its_controller_when_its_source_changes_control() {
    let mut s = setup(&["Cat-Gator"]);
    s.force_permanent_on_battlefield(0, "Swamp");
    cast(&mut s, "Cat-Gator");
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    let source = permanent(&s, "Cat-Gator");
    s.game_mut().state.permanents[source]
        .as_mut()
        .unwrap()
        .controller = PlayerId(1);
    let life = s.life(1);
    settle(&mut s);
    // The trigger counts its controller's Swamps, while lifelink follows the live source.
    assert_eq!(s.life(1), life);
    assert_eq!(s.life(0), 20);
}
