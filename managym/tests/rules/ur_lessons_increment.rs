// ur_lessons_increment.rs
// ETU-88: full Oracle behavior through ordinary play and semantic Commands.

use super::helpers::*;
use managym::{
    agent::{
        action::{Action, ActionSpaceKind},
        observation::Observation,
    },
    decision::Command,
    flow::turn::StepKind,
    semantic::SemanticPack,
    state::{
        game_object::{CardId, PermanentId, PlayerId, Target},
        zone::ZoneType,
    },
    Game,
};
use std::collections::BTreeMap;

fn deck() -> BTreeMap<String, usize> {
    [
        ("Island", 16),
        ("Mountain", 8),
        ("Gran-Gran", 3),
        ("Proft's Eidetic Memory", 3),
        ("Combustion Technique", 4),
        ("Accumulate Wisdom", 4),
        ("Pop Quiz", 2),
        ("Otter-Penguin", 2),
        ("Water Tribe Rallier", 2),
        ("Lightning Bolt", 2),
    ]
    .into_iter()
    .map(|(name, n)| (name.into(), n))
    .collect()
}

fn setup() -> Scenario {
    let mut s = Scenario::new(deck(), deck(), 88);
    s.advance_to_active_step(0, StepKind::Main);
    s
}

fn available(s: &Scenario, player: usize, name: &str) -> CardId {
    s.game()
        .state
        .cards
        .iter()
        .enumerate()
        .find_map(|(i, card)| {
            (card.owner == PlayerId(player)
                && card.name == name
                && matches!(
                    s.game().state.zones.zone_of(CardId(i)),
                    Some(ZoneType::Hand | ZoneType::Library)
                ))
            .then_some(CardId(i))
        })
        .unwrap_or_else(|| panic!("no available {name}"))
}

fn put(s: &mut Scenario, player: usize, name: &str) -> PermanentId {
    let card = available(s, player, name);
    s.game_mut().move_card(card, ZoneType::Battlefield);
    s.game().state.card_to_permanent[card].unwrap()
}

fn cast(s: &mut Scenario, name: &str) -> CardId {
    let card = available(s, 0, name);
    s.game_mut().move_card(card, ZoneType::Hand);
    s.game_mut().scenario_refresh_priority().unwrap();
    let index = s
        .action_space()
        .actions
        .iter()
        .position(|a| matches!(a, Action::CastSpell { card: c, .. } if *c == card))
        .unwrap_or_else(|| panic!("cannot cast {name}: {:?}", s.action_space()));
    s.step_action(index);
    card
}

fn resolve(s: &mut Scenario) {
    s.pass_priority();
    s.pass_priority();
}
fn counters(s: &Scenario, id: PermanentId) -> i32 {
    s.game().state.permanents[id]
        .as_ref()
        .unwrap()
        .plus1_counters
}
fn card_of(s: &Scenario, id: PermanentId) -> CardId {
    s.game().state.permanents[id].as_ref().unwrap().card
}
fn plant_lessons(s: &mut Scenario, player: usize, n: usize) {
    for _ in 0..n {
        let card = available(s, player, "Accumulate Wisdom");
        s.game_mut().move_card(card, ZoneType::Graveyard);
    }
}

#[test]
fn gran_gran_taps_to_loot_mandatorily_and_can_discard_the_drawn_card() {
    let mut s = setup();
    let gran = put(&mut s, 0, "Gran-Gran");
    s.game_mut().state.permanents[gran]
        .as_mut()
        .unwrap()
        .summoning_sick = false;
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    let top = *s
        .game()
        .state
        .zones
        .zone_cards(ZoneType::Library, PlayerId(0))
        .last()
        .unwrap();
    let before = s.zone_size(0, ZoneType::Hand);
    let index = s.action_space().actions.iter().position(|a| matches!(a, Action::DeclareAttacker { permanent, attack: true, .. } if *permanent == gran)).unwrap();
    s.step_action(index);
    resolve(&mut s);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Discard);
    assert_eq!(s.zone_size(0, ZoneType::Hand), before + 1);
    assert!(s
        .action_space()
        .actions
        .iter()
        .all(|a| matches!(a, Action::SelectCard { .. })));
    let index = s
        .action_space()
        .actions
        .iter()
        .position(|a| matches!(a, Action::SelectCard { card, .. } if *card == top))
        .unwrap();
    s.step_action(index);
    assert_eq!(s.game().state.zones.zone_of(top), Some(ZoneType::Graveyard));
    assert_eq!(s.zone_size(0, ZoneType::Hand), before);
}

#[test]
fn gran_gran_reduction_threshold_noncreature_and_generic_floor() {
    let mut s = setup();
    let gran = put(&mut s, 0, "Gran-Gran");
    let proft = available(&s, 0, "Proft's Eidetic Memory");
    let tiger = available(&s, 0, "Otter-Penguin");
    for card in [proft, tiger] {
        s.game_mut().move_card(card, ZoneType::Hand);
    }
    put(&mut s, 0, "Island");
    plant_lessons(&mut s, 0, 2);
    s.game_mut().scenario_refresh_priority().unwrap();
    let offered = |s: &Scenario, card| {
        s.action_space()
            .actions
            .iter()
            .any(|a| matches!(a, Action::CastSpell { card: c, .. } if *c == card))
    };
    assert!(!offered(&s, proft));
    plant_lessons(&mut s, 0, 1);
    let wisdom = available(&s, 0, "Accumulate Wisdom");
    s.game_mut().move_card(wisdom, ZoneType::Hand);
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(offered(&s, proft));
    assert!(!offered(&s, tiger)); // {1}{U} creature is never reduced.
    assert!(offered(&s, wisdom)); // {U} is still {U}.
    let gran_card = card_of(&s, gran);
    s.game_mut().move_card(gran_card, ZoneType::Hand);
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!offered(&s, proft));
}

#[test]
fn proft_etb_draw_and_combat_counts_at_resolution() {
    let mut s = setup();
    put(&mut s, 0, "Island");
    put(&mut s, 0, "Island");
    let gran = put(&mut s, 0, "Gran-Gran");
    s.game_mut().state.turn.cards_drawn_this_turn[0] = 1;
    let card = cast(&mut s, "Proft's Eidetic Memory");
    let before = s.zone_size(0, ZoneType::Hand);
    resolve(&mut s); // permanent enters, ETB on stack
    assert_eq!(
        s.game().state.zones.zone_of(card),
        Some(ZoneType::Battlefield)
    );
    resolve(&mut s);
    assert_eq!(s.zone_size(0, ZoneType::Hand), before + 1);
    s.advance_to_active_step(0, StepKind::BeginningOfCombat);
    assert_eq!(s.action_space().kind, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(gran)));
    // Further draws in response count when the ability resolves (CR 608.2h).
    s.game_mut().draw_cards(PlayerId(0), 2);
    resolve(&mut s);
    assert_eq!(counters(&s, gran), 3);
}

#[test]
fn proft_combat_requires_two_draws_and_own_turn() {
    for draws in [0, 1] {
        let mut s = setup();
        put(&mut s, 0, "Proft's Eidetic Memory");
        put(&mut s, 0, "Gran-Gran");
        s.pass_priority();
        resolve(&mut s); // ETB
        s.game_mut().state.turn.cards_drawn_this_turn[0] = draws;
        s.advance_to_active_step(0, StepKind::BeginningOfCombat);
        assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
        assert!(s.game().state.stack_objects.is_empty());
    }
    let mut s = setup();
    put(&mut s, 1, "Proft's Eidetic Memory");
    put(&mut s, 1, "Gran-Gran");
    s.pass_priority();
    resolve(&mut s);
    s.game_mut().state.turn.cards_drawn_this_turn[1] = 3;
    s.advance_to_active_step(0, StepKind::BeginningOfCombat);
    assert!(s.game().state.stack_objects.is_empty());
}

#[test]
fn proft_hand_limit_only_while_controlled_and_legend_rule_for_both_card_types() {
    for name in ["Gran-Gran", "Proft's Eidetic Memory"] {
        let mut s = setup();
        let first = put(&mut s, 0, name);
        let second = put(&mut s, 0, name);
        let first_card = card_of(&s, first);
        let second_card = card_of(&s, second);
        s.pass_priority();
        assert_eq!(s.action_space().kind, ActionSpaceKind::LegendRule);
        let index = s
            .action_space()
            .actions
            .iter()
            .position(|a| matches!(a, Action::SelectCard { card, .. } if *card == second_card))
            .unwrap();
        s.step_action(index);
        assert_eq!(
            s.game().state.zones.zone_of(first_card),
            Some(ZoneType::Graveyard)
        );
        assert_eq!(
            s.game().state.zones.zone_of(second_card),
            Some(ZoneType::Battlefield)
        );
    }
    let mut s = setup();
    let proft = put(&mut s, 0, "Proft's Eidetic Memory");
    s.pass_priority();
    resolve(&mut s);
    s.game_mut().draw_cards(PlayerId(0), 5);
    let before = s.zone_size(0, ZoneType::Hand);
    assert!(before > 7);
    s.advance_to_active_step(0, StepKind::End);
    resolve(&mut s);
    assert_eq!(s.zone_size(0, ZoneType::Hand), before);
    let card = card_of(&s, proft);
    s.game_mut().move_card(card, ZoneType::Hand);
    s.advance_to_active_step(0, StepKind::End);
    resolve(&mut s);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Discard);
    while s.action_space().kind == ActionSpaceKind::Discard {
        s.step_action(0);
    }
    assert_eq!(s.zone_size(0, ZoneType::Hand), 7);
}

#[test]
fn legend_rule_recognizes_existing_lowercase_supertypes() {
    let name = "Suki, Kyoshi Warrior";
    let mut cards = deck();
    cards.insert(name.into(), 2);
    let mut s = Scenario::new(cards.clone(), cards, 88);
    s.advance_to_active_step(0, StepKind::Main);
    let first = put(&mut s, 0, name);
    let second = put(&mut s, 0, name);
    let first_card = card_of(&s, first);
    let second_card = card_of(&s, second);
    s.pass_priority();
    assert_eq!(s.action_space().kind, ActionSpaceKind::LegendRule);
    let index = s
        .action_space()
        .actions
        .iter()
        .position(
            |action| matches!(action, Action::SelectCard { card, .. } if *card == second_card),
        )
        .unwrap();
    s.step_action(index);
    assert_eq!(
        s.game().state.zones.zone_of(first_card),
        Some(ZoneType::Graveyard)
    );
    assert_eq!(
        s.game().state.zones.zone_of(second_card),
        Some(ZoneType::Battlefield)
    );
}

#[test]
fn cleanup_sba_grants_priority_and_then_repeats_mandatory_discard() {
    let mut s = setup();
    let land = put(&mut s, 0, "Island");
    let card = card_of(&s, land);
    // A 0/0 animated land survives only while its until-end-of-turn buff lasts.
    let permanent = s.game_mut().state.permanents[land].as_mut().unwrap();
    permanent.animated = true;
    permanent.temp_toughness = 1;
    s.game_mut().draw_cards(PlayerId(0), 4);
    s.advance_to_active_step(0, StepKind::End);
    resolve(&mut s);
    while s.action_space().kind == ActionSpaceKind::Discard {
        s.step_action(0);
    }
    assert_eq!(s.zone_size(0, ZoneType::Hand), 7);
    assert_eq!(
        s.game().state.zones.zone_of(card),
        Some(ZoneType::Graveyard)
    );
    assert_eq!(s.game().state.turn.current_step_kind(), StepKind::Cleanup);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);

    // CR 514.3a: cards drawn during exceptional cleanup priority must be
    // discarded in a new cleanup step before the next turn starts.
    let turn = s.game().state.turn.turn_number;
    s.game_mut().draw_cards(PlayerId(0), 2);
    resolve(&mut s);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Discard);
    assert_eq!(s.game().state.turn.turn_number, turn);
    while s.action_space().kind == ActionSpaceKind::Discard {
        s.step_action(0);
    }
    assert_eq!(s.zone_size(0, ZoneType::Hand), 7);
    assert!(s.game().state.turn.turn_number > turn);
}

fn combustion(lessons: usize) -> (Scenario, CardId, PermanentId) {
    let mut s = setup();
    plant_lessons(&mut s, 0, lessons);
    plant_lessons(&mut s, 1, 3); // opponent's Lessons are irrelevant
    put(&mut s, 0, "Mountain");
    put(&mut s, 0, "Mountain");
    let victim = put(&mut s, 1, "Water Tribe Rallier"); // 3/2
    s.game_mut().state.permanents[victim]
        .as_mut()
        .unwrap()
        .plus1_counters = 5;
    let spell = cast(&mut s, "Combustion Technique");
    assert!(s.choose_target(Target::Permanent(victim)));
    resolve(&mut s);
    (s, spell, victim)
}

#[test]
fn combustion_damage_counts_own_lessons_excludes_resolving_spell_and_exiles_later_death() {
    for lessons in [0, 3] {
        let (mut s, spell, victim) = combustion(lessons);
        let permanent = s.game().state.permanents[victim].as_ref().unwrap();
        let card = permanent.card;
        assert_eq!(permanent.damage, 2 + lessons as i32);
        assert!(permanent.exile_if_dies_this_turn);
        assert_eq!(
            s.game().state.zones.zone_of(spell),
            Some(ZoneType::Graveyard)
        );
        let obs = Observation::new(s.game(), &[]);
        assert!(obs
            .opponent_permanents
            .iter()
            .any(|p| p.exile_if_dies_this_turn));
        // Any death, including sacrifice, is replaced (CR 614.1a).
        s.game_mut().move_card(card, ZoneType::Graveyard);
        assert_eq!(s.game().state.zones.zone_of(card), Some(ZoneType::Exile));
    }
}

#[test]
fn combustion_lethal_damage_exiles_without_a_graveyard_move() {
    let mut s = setup();
    put(&mut s, 0, "Mountain");
    put(&mut s, 0, "Mountain");
    let victim = put(&mut s, 1, "Gran-Gran");
    let card = card_of(&s, victim);
    cast(&mut s, "Combustion Technique");
    assert!(s.choose_target(Target::Permanent(victim)));
    resolve(&mut s);
    assert_eq!(s.game().state.zones.zone_of(card), Some(ZoneType::Exile));
}

#[test]
fn combustion_replacement_expires_at_cleanup_and_does_not_follow_reentry() {
    for bounce in [false, true] {
        let (mut s, _, victim) = combustion(0);
        let card = card_of(&s, victim);
        if bounce {
            s.game_mut().move_card(card, ZoneType::Hand);
            s.game_mut().move_card(card, ZoneType::Battlefield);
        } else {
            s.advance_to_active_step(1, StepKind::Upkeep);
        }
        let current = s.game().state.card_to_permanent[card].unwrap();
        assert!(
            !s.game().state.permanents[current]
                .as_ref()
                .unwrap()
                .exile_if_dies_this_turn
        );
        s.game_mut().move_card(card, ZoneType::Graveyard);
        assert_eq!(
            s.game().state.zones.zone_of(card),
            Some(ZoneType::Graveyard)
        );
    }
}

#[test]
fn combustion_is_an_owned_learn_candidate() {
    let mut s = Scenario::with_sideboards(
        deck(),
        deck(),
        BTreeMap::from([("Combustion Technique".into(), 1)]),
        BTreeMap::new(),
        88,
    );
    s.advance_to_active_step(0, StepKind::Main);
    for _ in 0..3 {
        put(&mut s, 0, "Island");
    }
    cast(&mut s, "Pop Quiz");
    resolve(&mut s);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Learn);
    let (index, card) = s
        .action_space()
        .actions
        .iter()
        .enumerate()
        .find_map(|(i, a)| match a {
            Action::LearnTakeLesson { card, .. } => Some((i, *card)),
            _ => None,
        })
        .unwrap();
    assert_eq!(s.game().state.cards[card].name, "Combustion Technique");
    s.step_action(index);
    assert_eq!(s.game().state.zones.zone_of(card), Some(ZoneType::Hand));
}

#[test]
fn revised_forty_card_deck_completes_and_replays_every_command_exactly() {
    let semantic = SemanticPack::two_deck().unwrap();
    let mut ur = semantic
        .player_config("UR candidate", "ur_lessons")
        .unwrap();
    for (name, n) in [
        ("It'll Quench Ya!", 2),
        ("First-Time Flyer", 2),
        ("Pop Quiz", 1),
        ("Igneous Inspiration", 1),
    ] {
        *ur.decklist.get_mut(name).unwrap() -= n;
    }
    ur.decklist.retain(|_, n| *n > 0);
    for (name, n) in [
        ("Combustion Technique", 2),
        ("Proft's Eidetic Memory", 1),
        ("Gran-Gran", 1),
        ("Accumulate Wisdom", 1),
    ] {
        *ur.decklist.entry(name.into()).or_default() += n;
    }
    assert_eq!(ur.decklist.values().sum::<usize>(), 40);
    let gw = semantic.player_config("GW", "gw_allies").unwrap();
    for seed in 0..4 {
        for reverse in [false, true] {
            let configs = if reverse {
                vec![gw.clone(), ur.clone()]
            } else {
                vec![ur.clone(), gw.clone()]
            };
            let mut game = Game::new(configs.clone(), seed, true);
            let mut replay = Game::new(configs, seed, true);
            for step in 0..20_000_u64 {
                if game.is_game_over() {
                    break;
                }
                let frame = game.semantic_decision_frame().unwrap();
                let i = ((step
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(seed + 1)) as usize)
                    % frame.offers.len();
                let command = Command {
                    command_id: format!("etu88-{step}"),
                    expected_revision: frame.revision,
                    offer_id: frame.offers[i].id.0,
                    answers: vec![],
                    object_preconditions: vec![],
                };
                game.execute_semantic_command(&command).unwrap();
                replay.execute_semantic_command(&command).unwrap();
                assert_eq!(
                    game.state.deterministic_hash(),
                    replay.state.deterministic_hash()
                );
                assert_eq!(
                    serde_json::to_value(&game.state.events).unwrap(),
                    serde_json::to_value(&replay.state.events).unwrap()
                );
                assert_eq!(
                    Observation::new(&game, &[]).to_json(),
                    Observation::new(&replay, &[]).to_json()
                );
            }
            assert!(game.is_game_over(), "seed {seed}, reverse {reverse}");
        }
    }
}

#[test]
fn legend_choice_and_lethal_damage_share_one_sba_batch() {
    let mut s = setup();
    let first = put(&mut s, 0, "Gran-Gran");
    let doomed = put(&mut s, 0, "Gran-Gran");
    let first_card = card_of(&s, first);
    let doomed_card = card_of(&s, doomed);
    s.game_mut().state.permanents[doomed]
        .as_mut()
        .unwrap()
        .damage = 2;
    s.pass_priority();
    assert_eq!(s.action_space().kind, ActionSpaceKind::LegendRule);
    // Keeping the lethally damaged legend does not rescue the other one.
    let i = s
        .action_space()
        .actions
        .iter()
        .position(|a| matches!(a, Action::SelectCard { card, .. } if *card == doomed_card))
        .unwrap();
    s.step_action(i);
    for card in [first_card, doomed_card] {
        assert_eq!(
            s.game().state.zones.zone_of(card),
            Some(ZoneType::Graveyard)
        );
    }
}

#[test]
fn proft_has_no_effect_if_its_only_target_leaves() {
    let mut s = setup();
    put(&mut s, 0, "Proft's Eidetic Memory");
    let gran = put(&mut s, 0, "Gran-Gran");
    s.pass_priority();
    resolve(&mut s);
    s.game_mut().state.turn.cards_drawn_this_turn[0] = 3;
    s.advance_to_active_step(0, StepKind::BeginningOfCombat);
    assert!(s.choose_target(Target::Permanent(gran)));
    let card = card_of(&s, gran);
    s.game_mut().move_card(card, ZoneType::Hand);
    s.game_mut().move_card(card, ZoneType::Battlefield);
    let new = s.game().state.card_to_permanent[card].unwrap();
    resolve(&mut s);
    assert_eq!(counters(&s, new), 0);
}

#[test]
fn combustion_invalid_target_does_not_damage_or_mark_new_incarnation() {
    let mut s = setup();
    put(&mut s, 0, "Mountain");
    put(&mut s, 0, "Mountain");
    let gran = put(&mut s, 1, "Gran-Gran");
    let spell = cast(&mut s, "Combustion Technique");
    assert!(s.choose_target(Target::Permanent(gran)));
    let card = card_of(&s, gran);
    s.game_mut().move_card(card, ZoneType::Hand);
    s.game_mut().move_card(card, ZoneType::Battlefield);
    let new = s.game().state.card_to_permanent[card].unwrap();
    resolve(&mut s);
    assert_eq!(
        s.game().state.zones.zone_of(spell),
        Some(ZoneType::Graveyard)
    );
    let permanent = s.game().state.permanents[new].as_ref().unwrap();
    assert_eq!(permanent.damage, 0);
    assert!(!permanent.exile_if_dies_this_turn);
}

#[test]
fn combustion_replacement_is_hashed_visible_and_rolls_back_with_its_damage() {
    use managym::search_state::{BenchCommand, BranchDriver, ClonePlusUndoDriver};
    let mut s = setup();
    put(&mut s, 0, "Mountain");
    put(&mut s, 0, "Mountain");
    let gran = put(&mut s, 1, "Gran-Gran");
    s.game_mut().state.permanents[gran]
        .as_mut()
        .unwrap()
        .plus1_counters = 3;
    cast(&mut s, "Combustion Technique");
    assert!(s.choose_target(Target::Permanent(gran)));
    s.pass_priority();
    let driver = ClonePlusUndoDriver::default();
    let root = driver.witness(s.game());
    let mut branch = driver.fork_exact(s.game());
    let mark = driver.mark(&mut branch);
    let index = branch
        .action_space()
        .unwrap()
        .actions
        .iter()
        .position(|a| matches!(a, Action::PassPriority { .. }))
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
    assert!(
        branch.state.permanents[gran]
            .as_ref()
            .unwrap()
            .exile_if_dies_this_turn
    );
    assert_ne!(driver.witness(&branch), root);
    driver.rollback(&mut branch, mark);
    assert_eq!(driver.witness(&branch), root);
    assert_eq!(driver.witness(s.game()), root);
}
