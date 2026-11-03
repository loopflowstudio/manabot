use std::hint::black_box;

#[cfg(not(feature = "benchmark"))]
#[path = "../src/benchmark_allocations.rs"]
mod benchmark_allocations;
#[cfg(feature = "benchmark")]
use managym::benchmark_allocations;

use managym::{match_fixtures::build_fixture, Game};
use serde::Serialize;

const FIXTURE_ID: &str = "interactive-heavy-80-w4-v1";
const WARMUP_CLONE_COUNT: usize = 64;
const MEASURED_CLONE_COUNT: usize = 1_024;
const SYNTHETIC_TEXT_BYTES: usize = 64 * 1_024;
const MINIMUM_PAYLOAD_DELTA_BYTES: usize = 4 * 1_024 * 1_024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
struct AllocationCounts {
    allocations: u64,
    allocated_bytes: u64,
}

#[derive(Debug, Serialize)]
struct CloneReceipt {
    baseline: AllocationCounts,
    expanded: AllocationCounts,
    delta_allocations: i64,
    delta_allocated_bytes: i64,
}

#[derive(Debug, Serialize)]
struct AllocationReceipt {
    schema: &'static str,
    build_profile: &'static str,
    fixture: &'static str,
    fixture_cards: usize,
    fixture_permanent_slots: usize,
    fixture_committed_events: usize,
    warmup_clone_count: usize,
    measured_clone_count: usize,
    synthetic_text_bytes_each: usize,
    minimum_payload_delta_bytes: usize,
    actual_payload_delta_bytes: usize,
    game_state_clone: CloneReceipt,
    exact_game_clone: CloneReceipt,
    threshold: &'static str,
    passed: bool,
}

fn measure(operation: impl FnOnce()) -> AllocationCounts {
    benchmark_allocations::start();
    operation();
    let (allocations, allocated_bytes) = benchmark_allocations::stop();
    AllocationCounts {
        allocations,
        allocated_bytes,
    }
}

/// Give every card of a copy its own large definition, through the same
/// copy-on-write seam a scenario mutation uses.
fn with_large_definitions(game: &Game) -> Game {
    let mut expanded = game.clone();
    for card in expanded.state.cards.iter_mut() {
        card.text_box = "x".repeat(SYNTHETIC_TEXT_BYTES);
    }
    expanded
}

fn warm_clone_fixture(game: &Game) {
    for _ in 0..WARMUP_CLONE_COUNT {
        let cloned = black_box(game).clone();
        black_box(cloned.state.cards.len());
        drop(cloned);
    }
}

fn measure_game_state_clones(game: &Game) -> AllocationCounts {
    measure(|| {
        for _ in 0..MEASURED_CLONE_COUNT {
            let cloned = black_box(&game.state).clone();
            black_box(cloned.cards.len());
            drop(cloned);
        }
    })
}

fn measure_game_clones(game: &Game) -> AllocationCounts {
    measure(|| {
        for _ in 0..MEASURED_CLONE_COUNT {
            let cloned = black_box(game).clone();
            black_box(cloned.state.cards.len());
            drop(cloned);
        }
    })
}

fn clone_receipt(baseline: AllocationCounts, expanded: AllocationCounts) -> CloneReceipt {
    CloneReceipt {
        baseline,
        expanded,
        delta_allocations: expanded.allocations as i64 - baseline.allocations as i64,
        delta_allocated_bytes: expanded.allocated_bytes as i64 - baseline.allocated_bytes as i64,
    }
}

/// Cards hold definitions by shared pointer, so a clone's cost is the mutable
/// state's alone, however large the definitions behind it.
#[test]
fn content_pack_clone_allocations_do_not_scale_with_definition_bytes() {
    let root = build_fixture(FIXTURE_ID).expect("deterministic fixture");
    let baseline_game = root.clone();
    let expanded_game = with_large_definitions(&baseline_game);
    let payload_delta = root.state.cards.len() * SYNTHETIC_TEXT_BYTES;
    assert!(
        payload_delta >= MINIMUM_PAYLOAD_DELTA_BYTES,
        "expanded immutable payload is only {payload_delta} bytes larger"
    );

    assert_eq!(
        baseline_game.state.cards.len(),
        expanded_game.state.cards.len()
    );
    assert_eq!(
        baseline_game.state.permanents.len(),
        expanded_game.state.permanents.len()
    );
    assert_eq!(
        baseline_game.state.events.len(),
        expanded_game.state.events.len()
    );

    warm_clone_fixture(&baseline_game);
    warm_clone_fixture(&expanded_game);

    let game_state_clone = clone_receipt(
        measure_game_state_clones(&baseline_game),
        measure_game_state_clones(&expanded_game),
    );
    let exact_game_clone = clone_receipt(
        measure_game_clones(&baseline_game),
        measure_game_clones(&expanded_game),
    );

    for receipt in [&game_state_clone, &exact_game_clone] {
        assert!(receipt.baseline.allocations > 0);
        assert!(receipt.baseline.allocated_bytes > 0);
        assert_eq!(receipt.delta_allocations, 0);
        assert_eq!(receipt.delta_allocated_bytes, 0);
    }

    let receipt = AllocationReceipt {
        schema: "manabot.content-pack-clone-allocations.v2",
        build_profile: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        fixture: FIXTURE_ID,
        fixture_cards: root.state.cards.len(),
        fixture_permanent_slots: root.state.permanents.len(),
        fixture_committed_events: root.state.events.len(),
        warmup_clone_count: WARMUP_CLONE_COUNT,
        measured_clone_count: MEASURED_CLONE_COUNT,
        synthetic_text_bytes_each: SYNTHETIC_TEXT_BYTES,
        minimum_payload_delta_bytes: MINIMUM_PAYLOAD_DELTA_BYTES,
        actual_payload_delta_bytes: payload_delta,
        game_state_clone,
        exact_game_clone,
        threshold: "mutable clone totals must be positive; expanded-minus-baseline clone allocations and bytes must equal 0 exactly",
        passed: true,
    };
    println!(
        "{}",
        serde_json::to_string(&receipt).expect("allocation receipt serializes")
    );
}
