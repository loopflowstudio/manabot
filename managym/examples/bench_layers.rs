//! Bounded same-layer dependency fixture plus ordinary action/search execution.
//! `--allocations` is a separate attribution run, never a timing comparator.
mod curated;
#[path = "curated/search.rs"]
mod search;

use search::{best_action, digest_decision};

use managym::{
    agent::action::Action,
    benchmark_allocations, benchmark_layers,
    state::{
        ability::Condition,
        card::{
            Card, CardDefinition, CardType, CardTypes, StaticAbility, StaticLandType, StaticScope,
            StaticTypeGrant,
        },
        game_object::PermanentId,
        mana::Mana,
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
    if std::env::args().any(|arg| arg == "--payment-only") {
        for pool in [true, false] {
            let (env, source) = payment_root(pool);
            for sample in 0..8 {
                println!("{}", payment(&env, source, pool, sample, allocations));
            }
        }
        return;
    }
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
    let mut step_samples = Vec::new();
    for seed in 0..16 {
        let (mut env, _) = root();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut game_steps = 0;
        while !env.is_game_over() && game_steps < 2000 {
            let count = env.action_count().unwrap();
            let action = rng.gen_range(0..count);
            digest.update(&(count as u64).to_le_bytes());
            digest.update(&(action as u64).to_le_bytes());
            // A bounded prefix, in execution order; no population percentile.
            let step_start = (step_samples.len() < 128).then(Instant::now);
            env.step(action as i64).unwrap();
            if let Some(start) = step_start {
                step_samples.push(start.elapsed().as_secs_f64());
            }
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
        "step_prefix_samples_s":step_samples,
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

/// A real {2} Cadets activation on the dependency board. Compare spending an
/// existing pool with tapping two layered/fixed sources; setup is unmeasured.
fn payment_root(pool: bool) -> (Env, PermanentId) {
    let (mut env, _) = root();
    // Leave a second activation payable, so trivial priority skipping does
    // not also resolve the measured activation before its receipt is captured.
    for _ in 0..3 {
        env.scenario_force_battlefield(0, "Mountain", true).unwrap();
    }
    let source = {
        let game = env.benchmark_game_mut();
        game.state.players[0].mana_pool = if pool {
            Mana::parse("RR")
        } else {
            Mana::default()
        };
        game.state
            .permanents
            .iter()
            .enumerate()
            .find_map(|(i, p)| {
                let p = p.as_ref()?;
                (game.state.cards[p.card].name == "Fire Nation Cadets").then_some(PermanentId(i))
            })
            .unwrap()
    };
    env.scenario_refresh().unwrap();
    assert_eq!(env.current_agent_index(), Some(0));
    (env, source)
}

fn payment_surface(env: &mut Env) -> serde_json::Value {
    let digest = env.state_digest().unwrap();
    let game = env.benchmark_game_mut();
    serde_json::json!({
        "digest": digest,
        "legal_actions": game.current_action_space,
        "players": game.state.players.iter().map(|p| serde_json::json!({
            "life":p.life, "mana":p.mana_pool, "combat_mana":p.combat_mana_pool,
        })).collect::<Vec<_>>(),
        "permanents":game.state.permanents.iter().enumerate().filter_map(|(id,p)| {
            p.as_ref().map(|p| serde_json::json!({"id":id,"card":p.card,"tapped":p.tapped}))
        }).collect::<Vec<_>>(),
        "stack":game.state.stack_objects,
    })
}

fn payment(
    root: &Env,
    source: PermanentId,
    pool: bool,
    sample: usize,
    allocations: bool,
) -> serde_json::Value {
    let mut env = root.fork().unwrap();
    let before = payment_surface(&mut env);
    let mut trace = Vec::new();
    let mut elapsed = 0.0;
    let mut traffic = (0, 0);
    benchmark_layers::reset();
    for decision in 0..8 {
        let actions = env
            .benchmark_game_mut()
            .current_action_space
            .as_ref()
            .unwrap()
            .actions
            .clone();
        let index = actions.iter().position(|action| {
            if decision == 0 {
                matches!(action, Action::ActivateAbility { permanent, ability_index: 0, .. } if *permanent == source)
            } else {
                matches!(action, Action::PayCost { .. })
            }
        }).unwrap_or_else(|| panic!("payment decision {decision}, pool={pool}: {actions:?}"));
        if allocations {
            benchmark_allocations::start();
        }
        let start = Instant::now();
        env.step(index as i64).unwrap();
        elapsed += start.elapsed().as_secs_f64();
        if allocations {
            let counts = benchmark_allocations::stop();
            traffic.0 += counts.0;
            traffic.1 += counts.1;
        }
        trace.push(serde_json::json!({"index":index, "action":actions[index], "after":payment_surface(&mut env)}));
        if !env.benchmark_game_mut().state.stack_objects.is_empty() {
            break;
        }
    }
    let metrics = benchmark_layers::snapshot();
    let game = env.benchmark_game_mut();
    assert_eq!(
        game.state.stack_objects.len(),
        1,
        "payment must finish stacking the ability"
    );
    assert!(game.pending_choice.is_none());
    assert_eq!(game.state.players[0].mana_pool.total(), 0);
    assert_eq!(
        game.state
            .permanents
            .iter()
            .flatten()
            .filter(|p| p.tapped)
            .count(),
        if pool { 0 } else { 2 }
    );
    assert_eq!(
        root.state_digest().unwrap(),
        before["digest"].as_str().unwrap(),
        "fork cannot spend root resources"
    );
    serde_json::json!({"workload":"complete_payment", "funding":if pool {"pool"} else {"lands"},
        "sample":sample, "before":before, "trace":trace, "metrics":metrics, "elapsed_s":elapsed,
        "allocation_count":traffic.0,"allocated_bytes":traffic.1,"counting_allocations":allocations})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_payment_spends_resources_and_preserves_the_root() {
        for pool in [true, false] {
            let (env, source) = payment_root(pool);
            let first = payment(&env, source, pool, 0, false);
            let second = payment(&env, source, pool, 1, false);
            assert_eq!(first["before"], second["before"]);
            assert_eq!(first["trace"], second["trace"]);
        }
    }
}
