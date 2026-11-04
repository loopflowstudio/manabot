//! Bounded same-layer dependency fixture plus ordinary action/search execution.
//! `--allocations` is a separate attribution run, never a timing comparator.
mod curated;
#[path = "curated/search.rs"]
mod search;

use search::{best_action, digest_decision};

use managym::{
    benchmark_allocations, benchmark_layers,
    state::{
        ability::Condition,
        card::{
            Card, CardDefinition, CardType, CardTypes, StaticAbility, StaticLandType, StaticScope,
            StaticTypeGrant,
        },
        game_object::PermanentId,
        predicate::CardPredicate,
    },
    Env,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::{sync::Arc, time::Instant};

fn root() -> (Env, PermanentId) {
    let mut env = Env::new(244, true, false, false);
    env.reset(curated::players()).unwrap();
    curated::add_layer_board(&mut env);
    // Urborg precedes Moon: timestamp order would incorrectly grant Swamp.
    // The fixture changes two existing physical cards, never the catalog.
    let urborg = PermanentId(env.scenario_force_battlefield(0, "Island", true).unwrap());
    let moon = PermanentId(env.scenario_force_battlefield(0, "Mountain", true).unwrap());
    for (id, definition) in [
        (
            urborg,
            CardDefinition {
                name: "Urborg witness".into(),
                types: CardTypes::new([CardType::Land]),
                statics: vec![StaticAbility::TypeGrant(StaticTypeGrant {
                    scope: StaticScope::All(CardPredicate {
                        card_type: Some(CardType::Land),
                        ..Default::default()
                    }),
                    types: vec![],
                    subtypes: vec!["Swamp".into()],
                    condition: None,
                })],
                ..Default::default()
            },
        ),
        (
            moon,
            CardDefinition {
                name: "Moon witness".into(),
                types: CardTypes::new([CardType::Enchantment]),
                statics: vec![StaticAbility::SetsLandType(StaticLandType {
                    predicate: CardPredicate {
                        card_type: Some(CardType::Land),
                        nonbasic: true,
                        ..Default::default()
                    },
                    subtype: "Mountain".into(),
                    condition: Condition::Always,
                })],
                ..Default::default()
            },
        ),
    ] {
        let game = env.benchmark_game_mut();
        let card_id = game.state.permanents[id].as_ref().unwrap().card;
        let old = &game.state.cards[card_id];
        let card =
            Card::from_definition(old.id, old.owner, old.definition_id, Arc::new(definition));
        game.state.cards[card_id] = card;
    }
    env.scenario_refresh().unwrap();
    let view = env.benchmark_game_mut().characteristics(urborg);
    assert!(view.has_subtype("Mountain") && !view.has_subtype("Swamp"));
    (env, urborg)
}

fn main() {
    let allocations = std::env::args().any(|arg| arg == "--allocations");
    let (mut env, id) = root();
    benchmark_layers::probe(env.benchmark_game_mut(), id);
    if std::env::args().any(|arg| arg == "--probe-only") {
        return;
    }
    benchmark_layers::reset();
    if allocations {
        benchmark_allocations::start();
    }
    let start = Instant::now();
    let mut digest = blake3::Hasher::new();
    let mut steps = 0;
    let mut capped = 0;
    for seed in 0..16 {
        let (mut env, _) = root();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut game_steps = 0;
        while !env.is_game_over() && game_steps < 2000 {
            let count = env.action_count().unwrap();
            let action = rng.gen_range(0..count);
            digest.update(&(count as u64).to_le_bytes());
            digest.update(&(action as u64).to_le_bytes());
            env.step(action as i64).unwrap();
            steps += 1;
            game_steps += 1;
        }
        digest.update(&(env.winner_index().unwrap_or(2) as u64).to_le_bytes());
        capped += usize::from(!env.is_game_over());
    }
    let elapsed = start.elapsed().as_secs_f64();
    let traffic = if allocations {
        benchmark_allocations::stop()
    } else {
        (0, 0)
    };
    println!(
        "{}",
        serde_json::json!({"workload":"dependency_steps", "steps":steps, "games":16, "capped":capped,
        "elapsed_s":elapsed, "digest":digest.finalize().to_hex().as_str(),
        "metrics":benchmark_layers::snapshot(), "allocation_count":traffic.0, "allocated_bytes":traffic.1,
        "counting_allocations":allocations})
    );

    let (mut env, _) = root();
    for decision in 0..8 {
        benchmark_layers::reset();
        if allocations {
            benchmark_allocations::start();
        }
        let start = Instant::now();
        let result = env
            .flat_mc_scores(16, 4, 244 * 1_000_003 + decision, 2000)
            .unwrap();
        let elapsed = start.elapsed().as_secs_f64();
        let traffic = if allocations {
            benchmark_allocations::stop()
        } else {
            (0, 0)
        };
        let best = best_action(&result);
        let mut digest = blake3::Hasher::new();
        digest_decision(&mut digest, &result, best);
        println!(
            "{}",
            serde_json::json!({"workload":"dependency_search", "decision":decision,
            "elapsed_s":elapsed, "actions":result.scores.len(), "simulations":result.simulations,
            "cap_hits":result.cap_hits, "digest":digest.finalize().to_hex().as_str(),
            "metrics":benchmark_layers::snapshot(), "allocation_count":traffic.0, "allocated_bytes":traffic.1,
            "counting_allocations":allocations})
        );
        if env.step(best as i64).unwrap().2 {
            break;
        }
    }
}
