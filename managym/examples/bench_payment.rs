//! Mana payment search work on fixed workloads, counted by
//! `managym::benchmark_payments` (plans, witness calls, expanded states) with
//! wall time beside it. `replay <script.py>` replays a reporter script's SEED,
//! DECKS and ACTIONS literals on a scalar `Env` and reports the heaviest
//! steps; `boards` builds priority positions with many mana sources and one
//! spell whose cost passes the optimistic bounds, and reports the work of
//! computing that position's legal actions. A digest folds every action count
//! so a behavior change shows beside the counts. Run in release:
//! `cargo run --release --features benchmark --example bench_payment --manifest-path managym/Cargo.toml -- boards`.
use managym::agent::env::Env;
use managym::benchmark_payments::{reporter_game, reset, snapshot, Metrics};
use managym::state::player::PlayerConfig;
use std::collections::BTreeMap;
use std::time::Instant;

fn replay(path: &str) {
    let (mut env, actions) = reporter_game(&std::fs::read_to_string(path).unwrap());
    let mut digest = env.action_count().unwrap() as u64;
    let mut total = Metrics::default();
    let mut steps: Vec<(u64, usize, Metrics, f64)> = Vec::new();
    for (number, action) in actions.iter().enumerate() {
        reset();
        let start = Instant::now();
        env.step(*action).unwrap();
        let seconds = start.elapsed().as_secs_f64();
        let m = snapshot();
        let count = if env.is_game_over() {
            0
        } else {
            env.action_count().unwrap()
        };
        digest = digest
            .wrapping_mul(31)
            .wrapping_add(count as u64 * 7 + *action as u64);
        total.witness_digest = total
            .witness_digest
            .wrapping_mul(31)
            .wrapping_add(m.witness_digest);
        total.plans += m.plans;
        total.calls += m.calls;
        total.states += m.states;
        total.max_plan_calls = total.max_plan_calls.max(m.max_plan_calls);
        total.plans_over_1k += m.plans_over_1k;
        total.max_sources = total.max_sources.max(m.max_sources);
        steps.push((m.calls, number + 1, m, seconds));
    }
    let max_step_s = steps.iter().map(|s| s.3).fold(0.0, f64::max);
    steps.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    println!(
        "{}",
        serde_json::json!({"workload":"replay","steps":actions.len(),"digest":digest,
            "total":total,"max_step_s":max_step_s,
            "heaviest_steps":steps.iter().take(5).map(|(_, number, m, seconds)|
                serde_json::json!({"step":number,"seconds":seconds,"payment":m})).collect::<Vec<_>>()})
    );
}

fn deck(cards: &[(&str, usize)]) -> BTreeMap<String, usize> {
    cards
        .iter()
        .map(|&(name, n)| (name.to_string(), n))
        .collect()
}

/// One priority position for seat 0: `battlefield` forced ready, `hand` the
/// only cards in hand. The measured region is the legal-action computation.
fn board(label: &str, battlefield: &[(&str, usize)], hand: &[&str]) {
    let mut cards: BTreeMap<String, usize> = BTreeMap::new();
    for (name, n) in battlefield {
        *cards.entry(name.to_string()).or_default() += n;
    }
    for name in hand {
        *cards.entry(name.to_string()).or_default() += 1;
    }
    // Padding keeps the library non-empty; it never reaches the battlefield.
    *cards.entry("Plains".into()).or_default() += 20;
    let mut env = Env::new(7, true, false, false);
    env.reset(vec![
        PlayerConfig::new("subject", cards),
        PlayerConfig::new("idle", deck(&[("Plains", 40)])),
    ])
    .unwrap();
    env.scenario_clear_hand(0).unwrap();
    let mut permanents = 0;
    for (name, n) in battlefield {
        for _ in 0..*n {
            env.scenario_force_battlefield(0, name, true).unwrap();
            permanents += 1;
        }
    }
    for name in hand {
        env.scenario_force_card_in_hand(0, name).unwrap();
    }
    reset();
    let start = Instant::now();
    env.scenario_refresh().unwrap();
    let seconds = start.elapsed().as_secs_f64();
    println!(
        "{}",
        serde_json::json!({"workload":label,"permanents":permanents,"hand":hand,
            "actions":env.action_count().unwrap(),"seconds":seconds,"payment":snapshot()})
    );
}

fn boards(max_lands: usize) {
    for lands in (6..=max_lands).step_by(2) {
        // A two-color land's optimistic output is both colors at once.
        board(
            &format!("dual-unpayable-{lands}"),
            &[("Forest", lands), ("Airship Engine Room", 1)],
            &["Sokka, Bold Boomeranger"],
        );
        // Herbalist's any-color ability is optimistically one of each color.
        board(
            &format!("herbalist-unpayable-{lands}"),
            &[("Forest", lands), ("Hermitic Herbalist", 1)],
            &["Sokka, Bold Boomeranger"],
        );
        board(
            &format!("herbalist-payable-{lands}"),
            &[("Forest", lands), ("Hermitic Herbalist", 2)],
            &["Dragonfly Swarm"],
        );
        // A paid filter nests a payment inside the unpayable one.
        board(
            &format!("filter-unpayable-{lands}"),
            &[("Forest", lands), ("Prophetic Prism", 1)],
            &["Sokka, Bold Boomeranger"],
        );
        // A counted tap cost chooses which creatures to tap for one color.
        board(
            &format!("counted-tap-unpayable-{lands}"),
            &[
                ("Forest", lands),
                ("Supportive Parents", 1),
                ("Compassionate Healer", 4),
            ],
            &["Sokka, Bold Boomeranger"],
        );
        // Lesson-only mana beside a non-Lesson and a Lesson spell.
        board(
            &format!("herbalist-lesson-{lands}"),
            &[("Mountain", lands), ("Hermitic Herbalist", 1)],
            &["Crashing Wave", "Waterbending Lesson"],
        );
        // Waterbend X: one payment question per X over lands and helpers.
        board(
            &format!("waterbend-x-{lands}"),
            &[
                ("Forest", lands),
                ("Katara, Water Tribe's Hope", 1),
                ("Bender's Waterskin", 1),
                ("Hermitic Herbalist", 1),
            ],
            &["Sokka, Bold Boomeranger"],
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("replay") => replay(&args[2]),
        Some("boards") => boards(args.get(2).map_or(12, |n| n.parse().unwrap())),
        _ => panic!("usage: bench_payment replay <script.py> | boards [max_lands]"),
    }
}
