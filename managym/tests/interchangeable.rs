//! Pass 1–2 proofs for MTG-487: indistinguishable-object predicate, permanent
//! and card/selection offer folding, alias storage, compound lowering through
//! aliases, and a debug-build behavioral soundness oracle.

use std::collections::BTreeMap;

use managym::{
    agent::{
        action::{Action, ActionSpaceKind, ActionType},
        env::{Env, RestoreOptions},
        structured_offer::{ChoiceAnswer, ChoiceStep, OfferSubmission, OfferVerb},
    },
    flow::{
        cast_permission::PlayGrant,
        trigger::{DelayedEffects, DelayedTrigger, DelayedTriggerEvent},
    },
    state::{
        ability::{DelayedExpiry, DelayedFrequency},
        binding::ObjectBinding,
        card::StaticAbility,
        effect::{Affected, ContinuousEffect, EffectDuration, Modification},
        game_object::{AbilitySource, ObjectId, PermanentId, PlayerId, Target},
        mana::Mana,
        permanent::{CounterKind, SpellEntry},
        player::{GameSetup, ManaFloats, PlayerConfig},
        zone::ZoneType,
    },
    Game,
};

fn deck(entries: &[(&str, usize)]) -> BTreeMap<String, usize> {
    entries
        .iter()
        .map(|&(name, n)| (name.to_string(), n))
        .collect()
}

fn position(lands: &[&str], entries: &[(&str, usize)], spell: &str) -> Game {
    let list = deck(entries);
    let mut game = Game::new(
        vec![
            PlayerConfig::new("p0", list.clone()),
            PlayerConfig::new("p1", list),
        ],
        487,
        true,
    );
    for _ in 0..2 {
        game.step(0).unwrap();
    }
    let player = PlayerId(0);
    game.scenario_clear_hand(player);
    game.scenario_force_card_in_hand(player, spell).unwrap();
    for land in lands {
        game.scenario_force_battlefield(player, land, true).unwrap();
    }
    game.scenario_refresh_priority().unwrap();
    game
}

fn cast_first(game: &mut Game) {
    let idx = game
        .action_space()
        .unwrap()
        .actions
        .iter()
        .position(|a| a.action_type() == ActionType::PriorityCastSpell)
        .expect("castable spell");
    game.step(idx).unwrap();
}

fn labels(game: &Game) -> Vec<String> {
    game.semantic_decision_frame()
        .unwrap()
        .offers
        .into_iter()
        .map(|o| o.label)
        .collect()
}

#[test]
fn manual_payment_kyoshi_village_plus_three_plains_folds_to_three_offers_with_aliases() {
    let entries = [
        ("Kyoshi Village", 4),
        ("Plains", 24),
        ("Youthful Knight", 12),
    ];
    let mut game = position(
        &["Kyoshi Village", "Plains", "Plains", "Plains"],
        &entries,
        "Youthful Knight",
    );
    cast_first(&mut game);

    assert_eq!(
        labels(&game),
        [
            "Tap Kyoshi Village for {G}/{W}",
            "Tap Plains for {W}",
            "PayCost",
        ]
    );
    let plains_objects: Vec<ObjectId> = game
        .state
        .permanents
        .iter()
        .flatten()
        .filter(|p| game.state[p.object].name == "Plains")
        .map(|p| p.object)
        .collect();
    assert_eq!(plains_objects.len(), 3);
    assert_eq!(
        game.action_aliases(),
        &[(1, plains_objects[1]), (1, plains_objects[2])]
    );
    assert!(game.interchangeable(plains_objects[0], plains_objects[1]));
    assert!(game.interchangeable(plains_objects[0], plains_objects[2]));

    // Out-of-bounds step rolls back both action space and aliases.
    let saved_aliases = game.action_aliases().to_vec();
    assert!(game.step(99).is_err());
    assert_eq!(game.action_aliases(), saved_aliases.as_slice());
}

#[test]
fn manual_payment_eight_forests_and_one_mountain_folds_to_two_taps() {
    let entries = [("Forest", 24), ("Mountain", 8), ("Badgermole Cub", 8)];
    let mut game = position(
        &[
            "Forest", "Forest", "Forest", "Forest", "Forest", "Forest", "Forest", "Forest",
            "Mountain",
        ],
        &entries,
        "Badgermole Cub",
    );
    cast_first(&mut game);
    let actions = &game.action_space().unwrap().actions;
    let tap_names: Vec<String> = actions
        .iter()
        .filter_map(|a| match a {
            Action::TapForMana { permanent, .. } => {
                let obj = game.permanent_object(*permanent).unwrap();
                Some(game.state[obj].name.clone())
            }
            _ => None,
        })
        .collect();
    assert_eq!(tap_names, ["Forest", "Mountain"]);
    assert_eq!(game.action_aliases().len(), 7);
    assert!(game.action_aliases().iter().all(|&(idx, _)| idx == 0));
}

#[test]
fn priority_all_floats_eight_forests_folds_to_one_tap_with_seven_aliases() {
    let entries = [("Forest", 32), ("Badgermole Cub", 8)];
    let mut game = position(&["Forest"; 8], &entries, "Badgermole Cub");
    game.mana_floats = [ManaFloats::All, ManaFloats::Off];
    game.scenario_refresh_priority().unwrap();

    let tap_indices: Vec<usize> = game
        .action_space()
        .unwrap()
        .actions
        .iter()
        .enumerate()
        .filter_map(|(i, a)| matches!(a, Action::TapForMana { .. }).then_some(i))
        .collect();
    assert_eq!(tap_indices.len(), 1);
    let tap_idx = tap_indices[0];
    assert_eq!(game.action_aliases().len(), 7);
    assert!(game.action_aliases().iter().all(|&(idx, _)| idx == tap_idx));
}

#[test]
fn convoke_with_four_identical_tokens_folds_to_one_waterbend_tap() {
    let entries = [
        ("Plains", 24),
        ("Forest", 8),
        ("Guardian of Vitu-Ghazi", 8),
    ];
    let mut game = position(
        &["Plains", "Plains", "Plains", "Forest"],
        &entries,
        "Guardian of Vitu-Ghazi",
    );
    let player = PlayerId(0);
    let mut tokens = Vec::new();
    for _ in 0..4 {
        let id = game.create_token("token.ally", player, false).unwrap();
        game.state.permanents[id]
            .as_mut()
            .unwrap()
            .summoning_sick = false;
        tokens.push(game.permanent_object(id).unwrap());
    }
    game.scenario_refresh_priority().unwrap();
    cast_first(&mut game);

    let waterbend_taps: Vec<(usize, PermanentId)> = game
        .action_space()
        .unwrap()
        .actions
        .iter()
        .enumerate()
        .filter_map(|(i, a)| match *a {
            Action::WaterbendTap { permanent, .. } => Some((i, permanent)),
            _ => None,
        })
        .collect();
    assert_eq!(waterbend_taps.len(), 1, "expected 1 folded convoke tap");
    let tap_idx = waterbend_taps[0].0;
    let tap_aliases: Vec<ObjectId> = game
        .action_aliases()
        .iter()
        .filter(|&&(idx, _)| idx == tap_idx)
        .map(|&(_, obj)| obj)
        .collect();
    assert_eq!(tap_aliases, tokens[1..]);
}

#[test]
fn distinguishing_permanent_state_blocks_interchangeability() {
    let entries = [
        ("Plains", 24),
        ("Forest", 8),
        ("Gray Ogre", 8),
    ];
    let game = position(&["Plains"; 3], &entries, "Gray Ogre");
    let player = PlayerId(0);
    let p0 = PermanentId(0);
    let p1 = PermanentId(1);
    let o0 = game.permanent_object(p0).unwrap();
    let o1 = game.permanent_object(p1).unwrap();
    assert!(game.interchangeable(o0, o1));

    // Equal ordinary counters (+1/+1) still fold; unequal counters stay separate.
    let mut counter_game = game.clone();
    counter_game.state.permanents[p0]
        .as_mut()
        .unwrap()
        .change_counters(CounterKind::PlusOnePlusOne, 1);
    assert!(!counter_game.interchangeable(o0, o1));
    counter_game.state.permanents[p1]
        .as_mut()
        .unwrap()
        .change_counters(CounterKind::PlusOnePlusOne, 1);
    assert!(counter_game.interchangeable(o0, o1));

    // Keyword counters carry timestamps (CR 613.7c) and never fold.
    let mut kw_game = game.clone();
    for p in [p0, p1] {
        let perm = kw_game.state.permanents[p].as_mut().unwrap();
        perm.change_counters(CounterKind::Trample, 1);
        let ts = perm.timestamp;
        perm.keyword_counter_timestamps
            .insert(CounterKind::Trample, ts);
    }
    assert!(!kw_game.interchangeable(o0, o1));

    // Layered (including granted) static abilities carry timestamps and never fold.
    let mut static_game = game.clone();
    for &o in &[o0, o1] {
        let ts = static_game.state.permanents[static_game.state.permanent_of(o).unwrap()]
            .as_ref()
            .unwrap()
            .timestamp;
        static_game.state.effects.push(ContinuousEffect {
            timestamp: ts,
            source: None,
            controller: player,
            affected: Affected::Object(o),
            modification: Modification::GrantStatic(Box::new(StaticAbility::NoMaximumHandSize)),
            duration: EffectDuration::EndOfTurn,
            source_object: None,
        });
    }
    assert!(!static_game.interchangeable(o0, o1));

    // Face-down manifests never fold (even when hiding identical cards).
    let manifest_entries = [("Mountain", 24), ("Gray Ogre", 16)];
    let mut manifest_game = position(&["Mountain"; 4], &manifest_entries, "Gray Ogre");
    let mut manifested = Vec::new();
    for _ in 0..2 {
        manifest_game
            .scenario_force_card_in_hand(player, "Gray Ogre")
            .unwrap();
        let card = manifest_game
            .state
            .zones
            .zone_objects(ZoneType::Hand, player)
            .iter()
            .copied()
            .find(|&o| manifest_game.state[o].name == "Gray Ogre")
            .unwrap();
        manifested.push(manifest_game.manifest(card, player));
    }
    manifest_game.scenario_refresh_priority().unwrap();
    assert!(!manifest_game.interchangeable(manifested[0], manifested[1]));
    let turn_ups = manifest_game
        .action_space()
        .unwrap()
        .actions
        .iter()
        .filter(|a| matches!(a, Action::TurnFaceUp { .. }))
        .count();
    assert_eq!(turn_ups, 2);

    // Settled spell_entry compares only (payment.is_some(), kicked, x).
    let mut entry_game = game.clone();
    let paid_a = std::sync::Arc::new(SpellEntry {
        kicked: false,
        x: 0,
        payment: Some(Default::default()),
    });
    let mut receipt_b = managym::state::payment::PaymentReceipt::default();
    receipt_b.pay_life(2);
    let paid_b = std::sync::Arc::new(SpellEntry {
        kicked: false,
        x: 0,
        payment: Some(receipt_b),
    });
    entry_game.state.permanents[p0]
        .as_mut()
        .unwrap()
        .spell_entry = Some(paid_a);
    entry_game.state.permanents[p1]
        .as_mut()
        .unwrap()
        .spell_entry = Some(paid_b);
    assert!(entry_game.interchangeable(o0, o1));

    // Kicked vs unkicked stays distinct.
    entry_game.state.permanents[p1]
        .as_mut()
        .unwrap()
        .spell_entry = Some(std::sync::Arc::new(SpellEntry {
        kicked: true,
        x: 0,
        payment: Some(Default::default()),
    }));
    assert!(!entry_game.interchangeable(o0, o1));
}

#[test]
fn snapshot_and_game_over_preserve_and_clear_aliases() {
    let entries = [("Forest", 32), ("Badgermole Cub", 8)];
    let setup = GameSetup {
        players: vec![
            PlayerConfig::new("p0", deck(&entries)),
            PlayerConfig::new("p1", deck(&entries)),
        ],
        mana_floats: [ManaFloats::All, ManaFloats::Off],
        ..GameSetup::from(vec![])
    };
    let mut env = Env::new(487, true, false, false);
    env.reset(setup).unwrap();
    // Keep both opening hands.
    env.step(0).unwrap();
    env.step(0).unwrap();
    // Play a Forest from hand, pass until two Forests are on the battlefield, or inject.
    env.scenario_force_battlefield(0, "Forest", true).unwrap();
    env.scenario_force_battlefield(0, "Forest", true).unwrap();
    env.scenario_refresh().unwrap();

    let snapshot = env.snapshot().unwrap();
    let restored = snapshot.restore(RestoreOptions::default()).unwrap();
    // Both original and restored expose the same folded space and digest.
    assert_eq!(env.action_count().unwrap(), restored.action_count().unwrap());
    assert_eq!(
        env.selected_witness().unwrap(),
        restored.selected_witness().unwrap()
    );

    // Game over clears aliases cleanly.
    let mut game = position(&["Forest"; 3], &entries, "Badgermole Cub");
    game.mana_floats = [ManaFloats::All, ManaFloats::Off];
    game.scenario_refresh_priority().unwrap();
    assert!(!game.action_aliases().is_empty());
    game.lose_game(PlayerId(1));
    game.step(0).unwrap_or(false);
    // Tick into terminal space via a fresh game-over transition.
    let mut over = position(&["Forest"; 3], &entries, "Badgermole Cub");
    over.mana_floats = [ManaFloats::All, ManaFloats::Off];
    over.scenario_refresh_priority().unwrap();
    over.state.players[1].life = 0;
    let pass = over
        .action_space()
        .unwrap()
        .actions
        .iter()
        .position(|a| matches!(a, Action::PassPriority { .. }))
        .unwrap();
    assert!(over.step(pass).unwrap());
    assert_eq!(
        over.action_space().unwrap().kind,
        ActionSpaceKind::GameOver
    );
    assert!(over.action_aliases().is_empty());
}

// ---------------------------------------------------------------------------
// Pass 2: Cards, selections, gates, distinguishing cases, and compound lowering.

#[test]
fn hand_play_land_and_cast_spell_fold_with_aliases() {
    let entries = [
        ("Island", 16),
        ("Forest", 16),
        ("First-Time Flyer", 8),
    ];
    let mut game = position(&["Island", "Island"], &entries, "First-Time Flyer");
    let player = PlayerId(0);
    game.scenario_force_card_in_hand(player, "First-Time Flyer")
        .unwrap();
    game.scenario_force_card_in_hand(player, "Forest").unwrap();
    game.scenario_force_card_in_hand(player, "Forest").unwrap();
    game.scenario_refresh_priority().unwrap();

    let actions = &game.action_space().unwrap().actions;
    let cast_indices: Vec<usize> = actions
        .iter()
        .enumerate()
        .filter_map(|(i, a)| matches!(a, Action::CastSpell { .. }).then_some(i))
        .collect();
    let land_indices: Vec<usize> = actions
        .iter()
        .enumerate()
        .filter_map(|(i, a)| matches!(a, Action::PlayLand { .. }).then_some(i))
        .collect();
    assert_eq!(cast_indices.len(), 1, "two First-Time Flyers fold to one CastSpell");
    assert_eq!(land_indices.len(), 1, "two Forests fold to one PlayLand");

    let hand_flyers: Vec<ObjectId> = game
        .state
        .zones
        .zone_objects(ZoneType::Hand, player)
        .iter()
        .copied()
        .filter(|&o| game.state[o].name == "First-Time Flyer")
        .collect();
    let hand_forests: Vec<ObjectId> = game
        .state
        .zones
        .zone_objects(ZoneType::Hand, player)
        .iter()
        .copied()
        .filter(|&o| game.state[o].name == "Forest")
        .collect();
    assert_eq!(hand_flyers.len(), 2);
    assert_eq!(hand_forests.len(), 2);
    assert!(game.action_aliases().contains(&(cast_indices[0], hand_flyers[1])));
    assert!(game.action_aliases().contains(&(land_indices[0], hand_forests[1])));
}

fn pass_until(game: &mut Game, pred: impl Fn(&Game) -> bool) {
    for _ in 0..16 {
        if pred(game) {
            return;
        }
        let pass = game
            .action_space()
            .unwrap()
            .actions
            .iter()
            .position(|a| matches!(a, Action::PassPriority { .. }))
            .expect("expected PassPriority");
        game.step(pass).unwrap();
    }
    assert!(
        pred(game),
        "condition not reached; current space: {:?}",
        game.action_space().map(|s| s.kind)
    );
}

#[test]
fn discard_one_from_two_identical_cards_folds_and_skips_when_trivial() {
    // 38 Islands + 2 Chart a Course: after clearing hand and drawing 2, hand is 2 identical Islands.
    let entries = [("Island", 38), ("Chart a Course", 2)];
    let mut base = position(&["Island", "Island"], &entries, "Chart a Course");
    let player = PlayerId(0);
    // Ensure top 2 cards of library are Islands.
    let lib = base.state.zones.zone_objects(ZoneType::Library, player).to_vec();
    let islands: Vec<ObjectId> = lib
        .iter()
        .copied()
        .filter(|&o| base.state[o].name == "Island")
        .take(2)
        .collect();
    assert_eq!(islands.len(), 2);
    for obj in islands {
        base.state
            .zones
            .reposition(ZoneType::Library, player, obj, usize::MAX);
    }

    // With skip_trivial = false: Discard prompt folds the 2 identical Islands to 1 + 1 alias.
    let mut no_skip = base.clone();
    no_skip.skip_trivial = false;
    no_skip.scenario_refresh_priority().unwrap();
    cast_first(&mut no_skip);
    // Manual payment or auto-pay: if PayCost is offered, step it; then pass priority twice.
    if let Some(pay) = no_skip
        .action_space()
        .unwrap()
        .actions
        .iter()
        .position(|a| matches!(a, Action::PayCost { .. }))
    {
        no_skip.step(pay).unwrap();
    }
    pass_until(&mut no_skip, |g| {
        g.action_space()
            .is_some_and(|s| s.kind == ActionSpaceKind::Discard)
    });
    let hand: Vec<ObjectId> = no_skip
        .state
        .zones
        .zone_objects(ZoneType::Hand, player)
        .to_vec();
    assert_eq!(hand.len(), 2);
    assert_eq!(no_skip.action_space().unwrap().actions.len(), 1);
    assert_eq!(no_skip.action_aliases(), &[(0, hand[1])]);

    // With skip_trivial = true: Chart a Course resolves and skips the 2-identical-card Discard.
    let mut with_skip = base;
    with_skip.skip_trivial = true;
    with_skip.scenario_refresh_priority().unwrap();
    cast_first(&mut with_skip);
    pass_until(&mut with_skip, |g| g.state.stack_objects.is_empty());
    assert_eq!(with_skip.state.zones.size(ZoneType::Hand, player), 1);
    assert_eq!(with_skip.state.zones.size(ZoneType::Graveyard, player), 2);
    assert_eq!(
        with_skip.action_space().unwrap().kind,
        ActionSpaceKind::Priority
    );
}

#[test]
fn target_three_identical_opponent_tokens_folds_to_one_and_compound_lowers_alias() {
    let entries = [("Mountain", 24), ("Lightning Bolt", 16)];
    let mut game = position(&["Mountain", "Mountain"], &entries, "Lightning Bolt");
    let opponent = PlayerId(1);
    let mut tokens = Vec::new();
    for _ in 0..3 {
        let id = game.create_token("token.ally", opponent, false).unwrap();
        tokens.push(game.permanent_object(id).unwrap());
    }
    game.scenario_refresh_priority().unwrap();

    // Compound offer keeps all 3 token candidates; lowering a folded alias (token #2) succeeds.
    let offers = game.compound_offers().unwrap();
    let bolt_offer = offers
        .projection()
        .offers
        .iter()
        .find(|o| o.label.contains("Lightning Bolt") && !o.choices.is_empty())
        .expect("single-target Bolt compound offer");
    let ChoiceStep::Select {
        role, candidates, ..
    } = &bolt_offer.choices[0];
    let token_candidates: Vec<_> = candidates
        .initial
        .as_ref()
        .unwrap()
        .iter()
        .filter(|c| c.label == "Ally")
        .collect();
    assert_eq!(token_candidates.len(), 3);
    // Pick the 3rd Ally token (which is folded into the 1st Ally token at ChooseTarget).
    let submission = OfferSubmission {
        offer_id: bolt_offer.id,
        answers: vec![ChoiceAnswer::Candidates {
            role: *role,
            candidates: vec![token_candidates[2].id],
        }],
    };
    let mut compound_game = game.clone();
    compound_game
        .apply_offer_submission(&offers, &submission)
        .unwrap();
    let remaining_tokens = compound_game
        .state
        .permanents
        .iter()
        .flatten()
        .filter(|p| p.controller == opponent)
        .count();
    assert_eq!(
        remaining_tokens, 2,
        "compound Lightning Bolt on folded alias resolves and removes one opponent token"
    );
    let cmds = game.compound_commands(&offers, &submission).unwrap();
    assert_eq!(cmds.len(), 2);

    // Direct sequential cast: ChooseTarget folds the 3 identical opponent tokens to 1.
    cast_first(&mut game);
    assert_eq!(
        game.action_space().unwrap().kind,
        ActionSpaceKind::ChooseTarget
    );
    let token_targets: Vec<usize> = game
        .action_space()
        .unwrap()
        .actions
        .iter()
        .enumerate()
        .filter_map(|(i, a)| match *a {
            Action::ChooseTarget {
                target: Target::Permanent(perm),
                ..
            } if game.permanent_object(perm).is_some_and(|o| tokens.contains(&o)) => Some(i),
            _ => None,
        })
        .collect();
    assert_eq!(token_targets.len(), 1);
    let idx = token_targets[0];
    assert_eq!(
        game.action_aliases(),
        &[(idx, tokens[1]), (idx, tokens[2])]
    );
}

#[test]
fn foretell_and_graveyard_card_ability_fold_with_aliases() {
    // Two Sozin's Comet in hand -> one Foretell offer with one alias.
    let entries = [("Mountain", 24), ("Sozin's Comet", 16)];
    let mut game = position(&["Mountain", "Mountain"], &entries, "Sozin's Comet");
    let player = PlayerId(0);
    game.scenario_force_card_in_hand(player, "Sozin's Comet")
        .unwrap();
    game.scenario_refresh_priority().unwrap();
    let foretell_indices: Vec<usize> = game
        .action_space()
        .unwrap()
        .actions
        .iter()
        .enumerate()
        .filter_map(|(i, a)| matches!(a, Action::Foretell { .. }).then_some(i))
        .collect();
    assert_eq!(foretell_indices.len(), 1);
    let comets: Vec<ObjectId> = game
        .state
        .zones
        .zone_objects(ZoneType::Hand, player)
        .iter()
        .copied()
        .filter(|&o| game.state[o].name == "Sozin's Comet")
        .collect();
    assert_eq!(comets.len(), 2);
    assert!(game
        .action_aliases()
        .contains(&(foretell_indices[0], comets[1])));

    // Two Reassembling Skeleton in graveyard -> one ActivateCardAbility with one alias.
    let gy_entries = [("Swamp", 24), ("Reassembling Skeleton", 16)];
    let mut gy_game = position(&["Swamp"; 4], &gy_entries, "Swamp");
    gy_game.scenario_clear_hand(player);
    let mut skeletons = Vec::new();
    for _ in 0..2 {
        gy_game
            .scenario_force_card_in_hand(player, "Reassembling Skeleton")
            .unwrap();
        let card = *gy_game
            .state
            .zones
            .zone_objects(ZoneType::Hand, player)
            .last()
            .unwrap();
        skeletons.push(gy_game.move_object(card, ZoneType::Graveyard));
    }
    gy_game.scenario_refresh_priority().unwrap();
    let card_abil_indices: Vec<usize> = gy_game
        .action_space()
        .unwrap()
        .actions
        .iter()
        .enumerate()
        .filter_map(|(i, a)| matches!(a, Action::ActivateCardAbility { .. }).then_some(i))
        .collect();
    assert_eq!(card_abil_indices.len(), 1);
    assert!(gy_game
        .action_aliases()
        .contains(&(card_abil_indices[0], skeletons[1])));

    // If one copy entered the graveyard from the battlefield this turn, they stay separate.
    gy_game
        .state
        .turn
        .graveyard_from_battlefield
        .push(skeletons[0]);
    gy_game.scenario_refresh_priority().unwrap();
    let separate_indices: Vec<usize> = gy_game
        .action_space()
        .unwrap()
        .actions
        .iter()
        .enumerate()
        .filter_map(|(i, a)| matches!(a, Action::ActivateCardAbility { .. }).then_some(i))
        .collect();
    assert_eq!(separate_indices.len(), 2);
}

#[test]
fn learn_folds_hand_discards_and_sideboard_lessons_via_entry_object() {
    let main = deck(&[("Island", 24), ("Pop Quiz", 16)]);
    let sb = deck(&[("Firebending Lesson", 4)]);
    let mut p0 = PlayerConfig::new("p0", main.clone());
    p0.deck.sideboard = sb;
    let p1 = PlayerConfig::new("p1", main);
    let mut game = Game::new(vec![p0, p1], 487, true);
    for _ in 0..2 {
        game.step(0).unwrap();
    }
    let player = PlayerId(0);
    game.scenario_clear_hand(player);
    game.scenario_force_card_in_hand(player, "Pop Quiz")
        .unwrap();
    game.scenario_force_card_in_hand(player, "Island").unwrap();
    game.scenario_force_card_in_hand(player, "Island").unwrap();
    for _ in 0..3 {
        game.scenario_force_battlefield(player, "Island", true)
            .unwrap();
    }
    game.scenario_refresh_priority().unwrap();
    cast_first(&mut game);
    // Resolve Pop Quiz: pass priority twice, then land on Decision::Learn.
    for _ in 0..8 {
        if game
            .action_space()
            .is_some_and(|s| s.kind == ActionSpaceKind::Learn)
        {
            break;
        }
        let pass = game
            .action_space()
            .unwrap()
            .actions
            .iter()
            .position(|a| matches!(a, Action::PassPriority { .. }))
            .unwrap();
        game.step(pass).unwrap();
    }
    assert_eq!(game.action_space().unwrap().kind, ActionSpaceKind::Learn);
    let actions = &game.action_space().unwrap().actions;
    let take_indices: Vec<usize> = actions
        .iter()
        .enumerate()
        .filter_map(|(i, a)| matches!(a, Action::LearnTakeLesson { .. }).then_some(i))
        .collect();
    let discard_islands: Vec<usize> = actions
        .iter()
        .enumerate()
        .filter_map(|(i, a)| match *a {
            Action::LearnDiscard { object, .. } if game.state[object].name == "Island" => Some(i),
            _ => None,
        })
        .collect();
    assert_eq!(take_indices.len(), 1, "4 identical sideboard Lessons fold to 1");
    assert_eq!(discard_islands.len(), 1, "identical hand Islands fold to 1 LearnDiscard");
    let sb_aliases: Vec<_> = game
        .action_aliases()
        .iter()
        .filter(|&&(idx, _)| idx == take_indices[0])
        .collect();
    assert_eq!(sb_aliases.len(), 3);
}

#[test]
fn cost_component_sacrifice_and_delve_fold_with_aliases() {
    // Sacrifice additional cost (Deadly Precision) with 3 identical tokens -> 1 SelectCard.
    let sac_entries = [("Swamp", 24), ("Deadly Precision", 16)];
    let mut game = position(&["Swamp"], &sac_entries, "Deadly Precision");
    let player = PlayerId(0);
    let opponent = PlayerId(1);
    let opp_token = game.create_token("token.ally", opponent, false).unwrap();
    let mut my_tokens = Vec::new();
    for _ in 0..3 {
        let id = game.create_token("token.ally", player, false).unwrap();
        my_tokens.push(game.permanent_object(id).unwrap());
    }
    game.skip_trivial = false;
    game.scenario_refresh_priority().unwrap();
    cast_first(&mut game);
    // Choose the opponent's token as target so none of my 3 tokens is referenced by the spell.
    let target_idx = game
        .action_space()
        .unwrap()
        .actions
        .iter()
        .position(|a| {
            matches!(
                a,
                Action::ChooseTarget {
                    target: Target::Permanent(id),
                    ..
                } if *id == opp_token
            )
        })
        .unwrap();
    game.step(target_idx).unwrap();
    // Now at sacrifice cost component: 3 identical tokens fold to 1 SelectCard + 2 aliases.
    let sac_indices: Vec<usize> = game
        .action_space()
        .unwrap()
        .actions
        .iter()
        .enumerate()
        .filter_map(|(i, a)| matches!(a, Action::SelectCard { .. }).then_some(i))
        .collect();
    assert_eq!(sac_indices.len(), 1);
    assert_eq!(
        game.action_aliases(),
        &[(sac_indices[0], my_tokens[1]), (sac_indices[0], my_tokens[2])]
    );

    // Delve (Teval, Arbiter of Virtue + 3 identical cards in graveyard) -> 1 Delve SelectCard.
    let delve_entries = [
        ("Swamp", 16),
        ("Forest", 16),
        ("Island", 8),
        ("Teval, Arbiter of Virtue", 4),
        ("Gray Ogre", 8),
    ];
    let mut delve_game = position(&["Swamp", "Forest"], &delve_entries, "Gray Ogre");
    delve_game
        .scenario_force_battlefield(player, "Teval, Arbiter of Virtue", true)
        .unwrap();
    let mut gy_cards = Vec::new();
    for _ in 0..3 {
        delve_game
            .scenario_force_card_in_hand(player, "Island")
            .unwrap();
        let card = *delve_game
            .state
            .zones
            .zone_objects(ZoneType::Hand, player)
            .last()
            .unwrap();
        gy_cards.push(delve_game.move_object(card, ZoneType::Graveyard));
    }
    delve_game.state.players[0]
        .mana_pool
        .add(&Mana::parse("R"), None);
    delve_game.scenario_refresh_priority().unwrap();
    cast_first(&mut delve_game);
    let delve_selects: Vec<usize> = delve_game
        .action_space()
        .unwrap()
        .actions
        .iter()
        .enumerate()
        .filter_map(|(i, a)| matches!(a, Action::SelectCard { .. }).then_some(i))
        .collect();
    assert_eq!(delve_selects.len(), 1, "3 identical graveyard Islands fold to 1 delve exile");
    let delve_aliases: Vec<ObjectId> = delve_game
        .action_aliases()
        .iter()
        .filter(|&&(idx, _)| idx == delve_selects[0])
        .map(|&(_, obj)| obj)
        .collect();
    assert_eq!(delve_aliases, gy_cards[1..]);
}

#[test]
fn distinguishing_card_and_selection_cases_stay_separate() {
    let entries = [
        ("Island", 16),
        ("Plains", 16),
        ("Swamp", 8),
        ("First-Time Flyer", 8),
        ("Fancy Footwork", 8),
        ("Duress", 8),
        ("Yue, the Moon Spirit", 4),
    ];

    // 1. Pinned reveal (ObjectBinding.known_to) keeps identical hand cards separate.
    let mut pin_game = position(&["Island", "Island"], &entries, "First-Time Flyer");
    let player = PlayerId(0);
    pin_game
        .scenario_force_card_in_hand(player, "First-Time Flyer")
        .unwrap();
    let flyers: Vec<ObjectId> = pin_game
        .state
        .zones
        .zone_objects(ZoneType::Hand, player)
        .to_vec();
    assert!(pin_game.interchangeable(flyers[0], flyers[1]));
    pin_game.state.delayed_triggers.push(DelayedTrigger {
        remembered: vec![ObjectBinding {
            object: flyers[0],
            selection: None,
            known_to: [true, true],
        }],
        event: DelayedTriggerEvent::EndStep { your_turn: false },
        controller: player,
        source_identity: AbilitySource::exact(flyers[0]),
        effects: DelayedEffects::new(Vec::new()),
        frequency: DelayedFrequency::Once,
        expiry: DelayedExpiry::ThisTurn,
    });
    pin_game.scenario_refresh_priority().unwrap();
    assert!(!pin_game.interchangeable(flyers[0], flyers[1]));
    let cast_count = pin_game
        .action_space()
        .unwrap()
        .actions
        .iter()
        .filter(|a| matches!(a, Action::CastSpell { .. }))
        .count();
    assert_eq!(cast_count, 2, "pinned hand copy must stay separate");

    // 2. Cast permission (PlayGrant on one copy) keeps copies separate.
    let mut grant_game = position(&["Island", "Island"], &entries, "First-Time Flyer");
    grant_game
        .scenario_force_card_in_hand(player, "First-Time Flyer")
        .unwrap();
    let g_flyers: Vec<ObjectId> = grant_game
        .state
        .zones
        .zone_objects(ZoneType::Hand, player)
        .to_vec();
    grant_game.state.cast_permissions.grants.push(PlayGrant {
        key: 1,
        origin: None,
        object: g_flyers[0],
        player,
        play_lands: false,
        cost: None,
        this_turn: true,
        through_next_turn: None,
        source: None,
        own_turn_only: false,
        waterbend: false,
        flash: true,
        any_mana: false,
        after_turn: None,
        foretold: false,
        on_adventure: false,
        follows_source_controller: false,
        flashback: false,
        prepared: false,
        face_down: false,
        condition: None,
    });
    assert!(!grant_game.interchangeable(g_flyers[0], g_flyers[1]));

    // 3. Multi-target spell (Fancy Footwork, up to 2 targets): chosen-so-far is named,
    // remaining unchosen identical tokens still fold.
    let mut multi_game = position(&["Plains", "Plains", "Plains"], &entries, "Fancy Footwork");
    multi_game.skip_trivial = false;
    let mut tokens = Vec::new();
    for _ in 0..3 {
        let id = multi_game.create_token("token.ally", player, false).unwrap();
        tokens.push(multi_game.permanent_object(id).unwrap());
    }
    multi_game.scenario_refresh_priority().unwrap();
    cast_first(&mut multi_game);
    // First target prompt: 3 identical tokens fold to 1 with 2 aliases.
    let first_targets: Vec<usize> = multi_game
        .action_space()
        .unwrap()
        .actions
        .iter()
        .enumerate()
        .filter_map(|(i, a)| matches!(a, Action::ChooseTarget { .. }).then_some(i))
        .collect();
    assert_eq!(first_targets.len(), 1);
    assert_eq!(multi_game.action_aliases().len(), 2);
    // Pick the first token.
    multi_game.step(first_targets[0]).unwrap();
    // Second target prompt: token[0] is in `chosen`, so tokens[1] and tokens[2] fold to 1 target + Decline.
    assert!(!multi_game.interchangeable(tokens[0], tokens[1]));
    assert!(multi_game.interchangeable(tokens[1], tokens[2]));
    let second_targets: Vec<usize> = multi_game
        .action_space()
        .unwrap()
        .actions
        .iter()
        .enumerate()
        .filter_map(|(i, a)| matches!(a, Action::ChooseTarget { .. }).then_some(i))
        .collect();
    assert_eq!(second_targets.len(), 1);
    assert_eq!(multi_game.action_aliases(), &[(second_targets[0], tokens[2])]);

    // 4. DiscardFromRevealedHand (deleted definition-only dedup): folds unreferenced copies
    // with alias, keeps pinned copy separate.
    let mut rev_base = position(&["Swamp"], &entries, "Duress");
    let opp = PlayerId(1);
    rev_base.scenario_clear_hand(opp);
    rev_base
        .scenario_force_card_in_hand(opp, "Fancy Footwork")
        .unwrap();
    rev_base
        .scenario_force_card_in_hand(opp, "Fancy Footwork")
        .unwrap();
    let mut opp_hand: Vec<ObjectId> = rev_base
        .state
        .zones
        .zone_objects(ZoneType::Hand, opp)
        .to_vec();
    opp_hand.sort_unstable();

    // Unpinned: resolves Duress -> DiscardFromRevealedHand folds to 1 SelectCard + 1 alias.
    let mut rev_unpinned = rev_base.clone();
    rev_unpinned.skip_trivial = false;
    rev_unpinned.scenario_refresh_priority().unwrap();
    cast_first(&mut rev_unpinned);
    if let Some(t) = rev_unpinned
        .action_space()
        .unwrap()
        .actions
        .iter()
        .position(|a| matches!(a, Action::ChooseTarget { .. }))
    {
        rev_unpinned.step(t).unwrap();
    }
    if let Some(p) = rev_unpinned
        .action_space()
        .unwrap()
        .actions
        .iter()
        .position(|a| matches!(a, Action::PayCost { .. }))
    {
        rev_unpinned.step(p).unwrap();
    }
    pass_until(&mut rev_unpinned, |g| {
        g.action_space()
            .is_some_and(|s| s.kind == ActionSpaceKind::Discard)
    });
    assert_eq!(rev_unpinned.action_space().unwrap().actions.len(), 1);
    assert_eq!(rev_unpinned.action_aliases(), &[(0, opp_hand[1])]);

    // Pinned copy: pinning opp_hand[0] keeps both copies separate (previously broken by dedup).
    let mut rev_pinned = rev_base;
    rev_pinned.skip_trivial = false;
    rev_pinned.state.delayed_triggers.push(DelayedTrigger {
        remembered: vec![ObjectBinding {
            object: opp_hand[0],
            selection: None,
            known_to: [true, true],
        }],
        event: DelayedTriggerEvent::EndStep { your_turn: false },
        controller: player,
        source_identity: AbilitySource::exact(opp_hand[0]),
        effects: DelayedEffects::new(Vec::new()),
        frequency: DelayedFrequency::Once,
        expiry: DelayedExpiry::ThisTurn,
    });
    rev_pinned.scenario_refresh_priority().unwrap();
    cast_first(&mut rev_pinned);
    if let Some(t) = rev_pinned
        .action_space()
        .unwrap()
        .actions
        .iter()
        .position(|a| matches!(a, Action::ChooseTarget { .. }))
    {
        rev_pinned.step(t).unwrap();
    }
    if let Some(p) = rev_pinned
        .action_space()
        .unwrap()
        .actions
        .iter()
        .position(|a| matches!(a, Action::PayCost { .. }))
    {
        rev_pinned.step(p).unwrap();
    }
    pass_until(&mut rev_pinned, |g| {
        g.action_space()
            .is_some_and(|s| s.kind == ActionSpaceKind::Discard)
    });
    assert_eq!(rev_pinned.action_space().unwrap().actions.len(), 2);
    assert!(rev_pinned.action_aliases().is_empty());

    // 5. CastDuringResolution from hand (Yue, the Moon Spirit; deleted definition-only dedup):
    // folds unreferenced noncreature hand copies with alias, keeps pinned copy separate.
    let mut free_base = position(&["Island"; 5], &entries, "Fancy Footwork");
    free_base
        .scenario_force_card_in_hand(player, "Fancy Footwork")
        .unwrap();
    let yue = free_base
        .scenario_force_battlefield(player, "Yue, the Moon Spirit", true)
        .unwrap();
    free_base.state.permanents[yue]
        .as_mut()
        .unwrap()
        .summoning_sick = false;
    let mut free_hand: Vec<ObjectId> = free_base
        .state
        .zones
        .zone_objects(ZoneType::Hand, player)
        .to_vec();
    free_hand.sort_unstable();

    let mut free_unpinned = free_base.clone();
    free_unpinned.scenario_refresh_priority().unwrap();
    let activate_yue = free_unpinned
        .action_space()
        .unwrap()
        .actions
        .iter()
        .position(|a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == yue))
        .unwrap();
    free_unpinned.step(activate_yue).unwrap();
    if let Some(pay) = free_unpinned
        .action_space()
        .unwrap()
        .actions
        .iter()
        .position(|a| matches!(a, Action::PayCost { .. }))
    {
        free_unpinned.step(pay).unwrap();
    }
    pass_until(&mut free_unpinned, |g| {
        g.action_space()
            .is_some_and(|s| s.kind == ActionSpaceKind::LookAndSelect)
    });
    assert_eq!(
        free_unpinned.action_space().unwrap().actions.len(),
        2,
        "1 folded free-cast + Decline"
    );
    assert_eq!(free_unpinned.action_aliases(), &[(0, free_hand[1])]);

    let mut free_pinned = free_base;
    free_pinned.state.delayed_triggers.push(DelayedTrigger {
        remembered: vec![ObjectBinding {
            object: free_hand[0],
            selection: None,
            known_to: [true, true],
        }],
        event: DelayedTriggerEvent::EndStep { your_turn: false },
        controller: player,
        source_identity: AbilitySource::exact(free_hand[0]),
        effects: DelayedEffects::new(Vec::new()),
        frequency: DelayedFrequency::Once,
        expiry: DelayedExpiry::ThisTurn,
    });
    free_pinned.scenario_refresh_priority().unwrap();
    let activate_yue_p = free_pinned
        .action_space()
        .unwrap()
        .actions
        .iter()
        .position(|a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == yue))
        .unwrap();
    free_pinned.step(activate_yue_p).unwrap();
    if let Some(pay) = free_pinned
        .action_space()
        .unwrap()
        .actions
        .iter()
        .position(|a| matches!(a, Action::PayCost { .. }))
    {
        free_pinned.step(pay).unwrap();
    }
    pass_until(&mut free_pinned, |g| {
        g.action_space()
            .is_some_and(|s| s.kind == ActionSpaceKind::LookAndSelect)
    });
    assert_eq!(
        free_pinned.action_space().unwrap().actions.len(),
        3,
        "pinned copy stays separate + Decline"
    );
    assert!(free_pinned.action_aliases().is_empty());
}

#[test]
fn select_card_decision_and_zone_gates_block_disallowed_folding() {
    // Decision gate: Crashing Wave (X = 0) -> DistributeCounters never folds identical
    // tapped opponent creatures.
    let wave_entries = [("Island", 24), ("Crashing Wave", 16)];
    let mut wave_game = position(&["Island", "Island"], &wave_entries, "Crashing Wave");
    let opp = PlayerId(1);
    for _ in 0..2 {
        let id = wave_game.create_token("token.ally", opp, false).unwrap();
        wave_game.state.permanents[id].as_mut().unwrap().tapped = true;
    }
    wave_game.scenario_refresh_priority().unwrap();
    cast_first(&mut wave_game);
    // Announce X = 0 if not already auto-chosen by skip_trivial.
    if let Some(x0) = wave_game
        .action_space()
        .unwrap()
        .actions
        .iter()
        .position(|a| matches!(a, Action::ChooseMode { mode: 0, .. }))
    {
        wave_game.step(x0).unwrap();
    }
    pass_until(&mut wave_game, |g| {
        g.action_space()
            .is_some_and(|s| s.kind == ActionSpaceKind::LookAndSelect)
    });
    assert_eq!(
        wave_game.action_space().unwrap().actions.len(),
        2,
        "DistributeCounters must not fold identical tapped opponent creatures"
    );
    assert!(wave_game.action_aliases().is_empty());

    // Zone & decision gates: Accumulate Wisdom over 3 identical top-of-library Islands
    // keeps LookAndSelect (Library zone gate) and OrderLibrary (decision + zone gate) unfolded.
    let wisdom_entries = [("Island", 38), ("Accumulate Wisdom", 2)];
    let mut lib_game = position(&["Island", "Island"], &wisdom_entries, "Accumulate Wisdom");
    let player = PlayerId(0);
    let lib = lib_game
        .state
        .zones
        .zone_objects(ZoneType::Library, player)
        .to_vec();
    let top_islands: Vec<ObjectId> = lib
        .iter()
        .copied()
        .filter(|&o| lib_game.state[o].name == "Island")
        .take(3)
        .collect();
    assert_eq!(top_islands.len(), 3);
    for obj in top_islands {
        lib_game
            .state
            .zones
            .reposition(ZoneType::Library, player, obj, usize::MAX);
    }
    lib_game.scenario_refresh_priority().unwrap();
    cast_first(&mut lib_game);
    pass_until(&mut lib_game, |g| {
        g.action_space()
            .is_some_and(|s| s.kind == ActionSpaceKind::LookAndSelect)
    });
    let select_count = lib_game
        .action_space()
        .unwrap()
        .actions
        .iter()
        .filter(|a| matches!(a, Action::SelectCard { .. }))
        .count();
    assert_eq!(
        select_count, 3,
        "Library SelectCard must not fold identical library cards"
    );
    assert!(lib_game.action_aliases().is_empty());

    // Pick one of the 3 Islands for hand -> OrderLibrary over the remaining 2 identical Islands.
    lib_game.step(0).unwrap();
    let order_count = lib_game
        .action_space()
        .unwrap()
        .actions
        .iter()
        .filter(|a| matches!(a, Action::SelectCard { .. }))
        .count();
    assert_eq!(
        order_count, 2,
        "OrderLibrary must not fold identical library cards"
    );
    assert!(lib_game.action_aliases().is_empty());
}

#[test]
fn compound_waterbend_lowers_folded_aliases_under_both_skip_trivial_modes() {
    for skip_trivial in [false, true] {
        let entries = [
            ("Forest", 16),
            ("Water Tribe Rallier", 16),
            ("Gray Ogre", 8),
        ];
        let mut game = position(
            &["Water Tribe Rallier", "Gray Ogre", "Gray Ogre", "Gray Ogre"],
            &entries,
            "Forest",
        );
        game.skip_trivial = skip_trivial;
        // Give player {2} mana in pool so waterbend {5} needs min=3, max=4 taps.
        game.state.players[0]
            .mana_pool
            .add(&Mana::parse("GG"), None);
        game.scenario_refresh_priority().unwrap();
        let activate = game
            .action_space()
            .unwrap()
            .actions
            .iter()
            .position(|a| matches!(a, Action::ActivateAbility { .. }))
            .unwrap();
        game.step(activate).unwrap();
        assert_eq!(game.action_space().unwrap().kind, ActionSpaceKind::Waterbend);

        let offers = game.compound_offers().unwrap();
        let offer = &offers.projection().offers[0];
        assert_eq!(offer.verb, OfferVerb::PayWaterbend);
        let ChoiceStep::Select {
            role,
            min,
            max,
            candidates,
            ..
        } = &offer.choices[0];
        assert_eq!((*min, *max), (3, 4));
        let all = candidates.initial.as_ref().unwrap();
        assert_eq!(all.len(), 4);

        // Select Rallier (0) + the 2nd and 3rd Gray Ogres (2, 3), skipping the canonical 1st Ogre (1),
        // plus paying remaining {2} mana via PayCost.
        let submission = OfferSubmission {
            offer_id: offer.id,
            answers: vec![ChoiceAnswer::Candidates {
                role: *role,
                candidates: vec![all[0].id, all[2].id, all[3].id],
            }],
        };
        let cmds = game.compound_commands(&offers, &submission).unwrap();
        assert!(!cmds.is_empty());
        let mut applied = game.clone();
        applied.apply_offer_submission(&offers, &submission).unwrap();
        assert!(!matches!(
            applied.action_space().map(|s| s.kind),
            Some(ActionSpaceKind::Waterbend)
        ));
        // Exactly 3 creatures were tapped (Rallier + 2 Ogres), leaving 1 Ogre untapped.
        let tapped_ogres = applied
            .state
            .permanents
            .iter()
            .flatten()
            .filter(|p| applied.state[p.object].name == "Gray Ogre" && p.tapped)
            .count();
        assert_eq!(
            tapped_ogres, 2,
            "skip_trivial={skip_trivial}: expected 2 tapped ogres + 2 mana spent"
        );
    }
}

// ---------------------------------------------------------------------------
// Debug-build behavioral soundness oracle over seeded Academy-matchup games.

fn curated_players() -> Vec<PlayerConfig> {
    let lessons = deck(&[
        ("Island", 9),
        ("Mountain", 8),
        ("Tiger-Seal", 2),
        ("Otter-Penguin", 2),
        ("Fire Nation Cadets", 2),
        ("First-Time Flyer", 2),
        ("Forecasting Fortune Teller", 1),
        ("Dragonfly Swarm", 1),
        ("Firebending Lesson", 4),
        ("Igneous Inspiration", 2),
        ("Pop Quiz", 2),
        ("Divide by Zero", 2),
        ("It'll Quench Ya!", 2),
        ("Accumulate Wisdom", 2),
    ]);
    let allies = deck(&[
        ("Plains", 9),
        ("Forest", 8),
        ("Water Tribe Rallier", 2),
        ("Invasion Reinforcements", 2),
        ("Compassionate Healer", 2),
        ("Earth Kingdom Jailer", 2),
        ("White Lotus Reinforcements", 2),
        ("Earth King's Lieutenant", 2),
        ("Kyoshi Warriors", 2),
        ("Badgermole Cub", 2),
        ("Suki, Kyoshi Warrior", 1),
        ("South Pole Voyager", 1),
        ("Allies at Last", 2),
        ("Yip Yip!", 1),
        ("Fancy Footwork", 2),
    ]);
    let mut lessons_player = PlayerConfig::new("lessons", lessons);
    lessons_player.deck.sideboard = deck(&[("Firebending Lesson", 2), ("Igneous Inspiration", 2)]);
    vec![lessons_player, PlayerConfig::new("allies", allies)]
}

fn substitute_alias(game: &Game, action: &Action, alias_object: ObjectId) -> Action {
    let perm = game.state.permanent_of(alias_object);
    match action.clone() {
        Action::TapForMana { player, .. } => Action::TapForMana {
            player,
            permanent: perm.expect("permanent alias for TapForMana"),
        },
        Action::ActivateAbility {
            player,
            ability_index,
            ..
        } => Action::ActivateAbility {
            player,
            permanent: perm.expect("permanent alias for ActivateAbility"),
            ability_index,
        },
        Action::WaterbendTap { player, .. } => Action::WaterbendTap {
            player,
            permanent: perm.expect("permanent alias for WaterbendTap"),
        },
        Action::TurnFaceUp { player, .. } => Action::TurnFaceUp {
            player,
            object: alias_object,
        },
        Action::PlayLand {
            profile,
            player,
            route,
            ..
        } => Action::PlayLand {
            profile,
            player,
            object: alias_object,
            route,
        },
        Action::CastSpell {
            profile,
            player,
            route,
            ..
        } => Action::CastSpell {
            profile,
            player,
            object: alias_object,
            route,
        },
        Action::ActivateCardAbility {
            player,
            ability_index,
            ..
        } => Action::ActivateCardAbility {
            player,
            object: alias_object,
            ability_index,
        },
        Action::Foretell { player, .. } => Action::Foretell {
            player,
            object: alias_object,
        },
        Action::ChooseTarget {
            player,
            target: Target::Permanent(_),
        } => Action::ChooseTarget {
            player,
            target: Target::Permanent(perm.expect("permanent alias for ChooseTarget")),
        },
        Action::ChooseTarget {
            player,
            target: Target::GraveyardCard(_),
        } => Action::ChooseTarget {
            player,
            target: Target::GraveyardCard(alias_object),
        },
        Action::SelectCard { player, .. } => Action::SelectCard {
            player,
            object: alias_object,
        },
        Action::LearnDiscard { player, .. } => Action::LearnDiscard {
            player,
            object: alias_object,
        },
        Action::LearnTakeLesson { player, .. } => Action::LearnTakeLesson {
            player,
            object: alias_object,
        },
        other => panic!("unexpected folded action variant in Pass 2: {other:?}"),
    }
}

/// Id-free observable description of one object, used both to tie-break the
/// oracle's deterministic continuation policy and to compare post-step traces.
fn object_signature(game: &Game, object: ObjectId) -> String {
    let Some(zone) = game.state.zones.zone_of(object) else {
        let obj = &game.state.objects[object.slot()];
        return format!("outside:{}:o{}", obj.name, obj.owner.0);
    };
    let obj = &game.state[object];
    if let Some(perm_id) = game.state.permanent_of(object) {
        let p = game.state.permanents[perm_id].as_ref().unwrap();
        let view = game.characteristics(perm_id);
        let counters: Vec<String> = p
            .counters
            .iter()
            .map(|(k, n)| format!("{}:{n}", k.name()))
            .collect();
        format!(
            "bf:{}:o{}:c{}:tap{}:dmg{}:atk{}:pt{}/{}:fd{}:cnt[{}]",
            view.name,
            obj.owner.0,
            p.controller.0,
            u8::from(p.tapped),
            p.damage,
            u8::from(p.attacking),
            view.power,
            view.toughness,
            u8::from(obj.is_face_down()),
            counters.join(",")
        )
    } else {
        format!("{zone:?}:{}:o{}", obj.name, obj.owner.0)
    }
}

/// Id-free action key combining the human label and the id-free state of any
/// object the action names, so ties between e.g. tapped and untapped copies
/// resolve identically across clones regardless of storage slot ordering.
fn action_key(game: &Game, label: &str, action: &Action) -> String {
    let subject = match *action {
        Action::PlayLand { object, .. }
        | Action::CastSpell { object, .. }
        | Action::ActivateCardAbility { object, .. }
        | Action::Foretell { object, .. }
        | Action::TurnFaceUp { object, .. }
        | Action::ScryCard { object, .. }
        | Action::LearnTakeLesson { object, .. }
        | Action::LearnDiscard { object, .. }
        | Action::SelectCard { object, .. } => object_signature(game, object),
        Action::ActivateAbility { permanent, .. }
        | Action::DeclareAttacker { permanent, .. }
        | Action::WaterbendTap { permanent, .. }
        | Action::TapForMana { permanent, .. } => game
            .permanent_object(permanent)
            .map(|o| object_signature(game, o))
            .unwrap_or_default(),
        Action::DeclareBlocker {
            blocker, attacker, ..
        } => {
            let b = game
                .permanent_object(blocker)
                .map(|o| object_signature(game, o))
                .unwrap_or_default();
            let a = attacker
                .and_then(|id| game.permanent_object(id))
                .map(|o| object_signature(game, o))
                .unwrap_or_else(|| "none".into());
            format!("{b}->{a}")
        }
        Action::ChooseTarget { target, .. } => match target {
            Target::Player(p) => format!("player:{}", p.0),
            Target::Permanent(id) => game
                .permanent_object(id)
                .map(|o| object_signature(game, o))
                .unwrap_or_default(),
            Target::StackSpell(o) | Target::GraveyardCard(o) => object_signature(game, o),
            Target::StackAbility(_) => "stack_ability".into(),
        },
        Action::ChooseTrigger { .. }
        | Action::PassPriority { .. }
        | Action::Decline { .. }
        | Action::PayCost { .. }
        | Action::ChooseMode { .. } => String::new(),
    };
    format!("{label}|{subject}")
}

#[derive(Debug, PartialEq, Eq)]
struct TraceStep {
    winner: Option<usize>,
    turn: u32,
    life: [i32; 2],
    mana: [[u8; 6]; 2],
    zones: Vec<Vec<String>>,
    offered_keys: Vec<String>,
}

fn snapshot_trace(game: &Game) -> TraceStep {
    let mut zones = Vec::new();
    for seat in [PlayerId(0), PlayerId(1)] {
        for zone in [
            ZoneType::Battlefield,
            ZoneType::Hand,
            ZoneType::Graveyard,
            ZoneType::Exile,
        ] {
            let mut entries: Vec<String> = game
                .state
                .zones
                .zone_objects(zone, seat)
                .iter()
                .map(|&o| object_signature(game, o))
                .collect();
            entries.sort();
            zones.push(entries);
        }
        let lib: Vec<String> = game
            .state
            .zones
            .zone_objects(ZoneType::Library, seat)
            .iter()
            .map(|&o| game.state[o].name.clone())
            .collect();
        zones.push(lib);
    }
    let stack: Vec<String> = game
        .state
        .stack_objects
        .iter()
        .map(|s| format!("{:?}:{}", s.controller(), s.source_definition_id().0))
        .collect();
    zones.push(stack);

    let mut offered_keys = Vec::new();
    if !game.is_game_over() {
        let space = game.action_space().expect("live action space");
        let frame = game.semantic_decision_frame().expect("decision frame");
        offered_keys = space
            .actions
            .iter()
            .zip(&frame.offers)
            .map(|(a, o)| action_key(game, &o.label, a))
            .collect();
        offered_keys.sort();
    }

    TraceStep {
        winner: game.winner_index(),
        turn: game.state.turn.turn_number,
        life: [game.state.players[0].life, game.state.players[1].life],
        mana: [
            game.state.players[0].mana_pool.mana().mana,
            game.state.players[1].mana_pool.mana().mana,
        ],
        zones,
        offered_keys,
    }
}

fn pick_deterministic_index(game: &Game, step_num: usize) -> usize {
    let space = game.action_space().unwrap();
    let frame = game.semantic_decision_frame().unwrap();
    // `PayCost` is an automatic-payment shortcut whose tie-breaker among
    // equal-score colors intentionally reads battlefield `PermanentId` order
    // (see `flow/mana/autotapper.rs`). When explicit source taps are offered
    // beside `PayCost`, choose among the explicit actions so the oracle tests
    // rules-level equivalence rather than autotapper's slot-order tiebreak.
    let has_manual_taps = space.actions.iter().any(|a| {
        matches!(
            a,
            Action::TapForMana { .. }
                | Action::ActivateAbility { .. }
                | Action::WaterbendTap { .. }
        )
    });
    let mut indexed: Vec<(String, usize)> = space
        .actions
        .iter()
        .zip(&frame.offers)
        .enumerate()
        .filter(|(_, (a, _))| !(has_manual_taps && matches!(a, Action::PayCost { .. })))
        .map(|(i, (a, o))| (action_key(game, &o.label, a), i))
        .collect();
    indexed.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    indexed[step_num % indexed.len()].1
}

fn verify_folded_pair(base: &Game, surviving_idx: usize, alias_object: ObjectId, horizon: usize) {
    let kept_action = base.action_space().unwrap().actions[surviving_idx].clone();
    let removed_action = substitute_alias(base, &kept_action, alias_object);

    let mut kept_branch = base.clone();
    let mut removed_branch = base.clone();
    removed_branch
        .current_action_space
        .as_mut()
        .unwrap()
        .actions[surviving_idx] = removed_action.clone();

    kept_branch.step(surviving_idx).unwrap();
    removed_branch.step(surviving_idx).unwrap();

    let mut history = Vec::new();
    for step_num in 0..horizon {
        let left = snapshot_trace(&kept_branch);
        let right = snapshot_trace(&removed_branch);
        assert_eq!(
            left, right,
            "oracle divergence at continuation step {step_num} after folding {kept_action:?} vs {removed_action:?} (alias {alias_object:?}); history={history:?}"
        );
        if kept_branch.is_game_over() {
            break;
        }
        let was_waterbend = kept_branch
            .action_space()
            .is_some_and(|s| s.kind == ActionSpaceKind::Waterbend);
        let left_idx = pick_deterministic_index(&kept_branch, step_num);
        let right_idx = pick_deterministic_index(&removed_branch, step_num);
        let la = kept_branch.action_space().unwrap().actions[left_idx].clone();
        let ra = removed_branch.action_space().unwrap().actions[right_idx].clone();
        // `autotapper` intentionally breaks ties between equal-reluctance
        // colors (e.g. Island vs Mountain) by `PermanentId` order; stop this
        // continuation before an automatic payment shortcut fires.
        // Likewise, a London mulligan redraw shuffles the unordered hand into
        // the library with Fisher-Yates (uniformly distributed, but dependent
        // on the pre-shuffle hand slot index under a fixed PRNG seed).
        if matches!(la, Action::PayCost { .. })
            || (kept_branch.state.turn.turn_number == 0
                && matches!(la, Action::ChooseMode { mode: 1, .. }))
        {
            break;
        }
        history.push((step_num, la, ra));
        kept_branch.step(left_idx).unwrap();
        removed_branch.step(right_idx).unwrap();
        // Waterbend does not offer manual land taps: if the last WaterbendTap
        // auto-completed the remaining generic mana through `autotapper`, stop
        // here rather than comparing `autotapper`'s `PermanentId`-order tiebreak.
        if was_waterbend
            && kept_branch
                .action_space()
                .is_none_or(|s| s.kind != ActionSpaceKind::Waterbend)
        {
            break;
        }
    }
}

#[test]
fn soundness_oracle_on_curated_matchups() {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    let mut checked_folds = 0usize;
    let mut decisions = 0usize;
    let mut folded_decisions = 0usize;
    let mut actions_before = 0usize;
    let mut actions_after = 0usize;
    for (floats, seeds, max_per_game) in [
        ([ManaFloats::Off; 2], 0u64..8, 12usize),
        ([ManaFloats::Usable; 2], 12u64..15, 8usize),
        ([ManaFloats::All; 2], 18u64..20, 8usize),
    ] {
        for seed in seeds {
            let mut setup = GameSetup::from(curated_players());
            setup.mana_floats = floats;
            let mut game = Game::new(setup, seed, true);
            let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_mul(1_000_003) ^ 487);
            let mut game_checks = 0usize;

            for _ in 0..140 {
                if game.is_game_over() {
                    break;
                }
                let aliases = game.action_aliases().to_vec();
                let after = game.action_space().unwrap().actions.len();
                if floats == [ManaFloats::Off; 2] {
                    decisions += 1;
                    actions_after += after;
                    actions_before += after + aliases.len();
                    if !aliases.is_empty() {
                        folded_decisions += 1;
                    }
                }
                if !aliases.is_empty() && game_checks < max_per_game {
                    // Check the first alias of each surviving class in this space.
                    let mut checked_indices = Vec::new();
                    for (surviving_idx, alias_obj) in aliases {
                        if !checked_indices.contains(&surviving_idx) && game_checks < max_per_game {
                            checked_indices.push(surviving_idx);
                            verify_folded_pair(&game, surviving_idx, alias_obj, 12);
                            checked_folds += 1;
                            game_checks += 1;
                        }
                    }
                }
                let pick = rng.gen_range(0..after);
                game.step(pick).unwrap();
            }
        }
    }
    eprintln!(
        "soundness_oracle_on_curated_matchups: verified {checked_folds} folded pairs; \
         default-mode fold share {folded_decisions}/{decisions} ({:.2}%), \
         mean actions {:.3} -> {:.3}",
        100.0 * folded_decisions as f64 / decisions.max(1) as f64,
        actions_before as f64 / decisions.max(1) as f64,
        actions_after as f64 / decisions.max(1) as f64,
    );
    assert!(
        checked_folds >= 40,
        "oracle must exercise folded spaces; only checked {checked_folds}"
    );
}
