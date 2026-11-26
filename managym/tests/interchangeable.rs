//! Pass 1 proofs for MTG-487: indistinguishable-object predicate, permanent
//! offer folding, alias storage, and a debug-build behavioral soundness oracle.

use std::collections::BTreeMap;

use managym::{
    agent::{
        action::{Action, ActionSpaceKind, ActionType},
        env::{Env, RestoreOptions},
    },
    state::{
        card::StaticAbility,
        effect::{Affected, ContinuousEffect, EffectDuration, Modification},
        game_object::{ObjectId, PermanentId, PlayerId, Target},
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
    vec![
        PlayerConfig::new("lessons", lessons),
        PlayerConfig::new("allies", allies),
    ]
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
        other => panic!("unexpected folded action variant in Pass 1: {other:?}"),
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
        if matches!(la, Action::PayCost { .. }) {
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
