use std::hint::black_box;

use managym::{match_fixtures::build_fixture, Game};
use serde::Serialize;

// In the default build the library leaves allocator selection to the binary,
// so this test installs and reads its own counting wrapper.
#[cfg(not(any(feature = "benchmark", feature = "mimalloc")))]
#[global_allocator]
static ALLOCATOR: managym::allocator::CountingAllocator =
    managym::allocator::CountingAllocator::new();

#[cfg(not(any(feature = "benchmark", feature = "mimalloc")))]
fn start_counting() {
    ALLOCATOR.start();
}

#[cfg(not(any(feature = "benchmark", feature = "mimalloc")))]
fn stop_counting() -> (u64, u64) {
    ALLOCATOR.stop()
}

#[cfg(feature = "benchmark")]
fn start_counting() {
    managym::allocator::start();
}

#[cfg(feature = "benchmark")]
fn stop_counting() -> (u64, u64) {
    managym::allocator::stop()
}

// The library installed mimalloc without a counter; refuse rather than report zero.
#[cfg(all(feature = "mimalloc", not(feature = "benchmark")))]
fn start_counting() {
    panic!("run with --features benchmark,mimalloc to count allocations under mimalloc");
}

#[cfg(all(feature = "mimalloc", not(feature = "benchmark")))]
fn stop_counting() -> (u64, u64) {
    unreachable!("start_counting refuses this build")
}

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

fn clone_repeatedly<T: Clone>(source: &T, count: usize) {
    for _ in 0..count {
        black_box(source.clone());
    }
}

fn measure_clones<T: Clone>(source: &T) -> AllocationCounts {
    start_counting();
    clone_repeatedly(source, MEASURED_CLONE_COUNT);
    let (allocations, allocated_bytes) = stop_counting();
    AllocationCounts {
        allocations,
        allocated_bytes,
    }
}

fn clone_receipt<T: Clone>(baseline: &T, expanded: &T) -> CloneReceipt {
    let baseline = measure_clones(baseline);
    let expanded = measure_clones(expanded);
    CloneReceipt {
        baseline,
        expanded,
        delta_allocations: expanded.allocations as i64 - baseline.allocations as i64,
        delta_allocated_bytes: expanded.allocated_bytes as i64 - baseline.allocated_bytes as i64,
    }
}

fn definition_text_bytes(game: &Game) -> usize {
    game.state
        .objects
        .iter()
        .map(|object| object.text_box.len())
        .sum()
}

/// Cards hold definitions by shared pointer, so a clone's cost is the mutable
/// state's alone, however large the definitions behind it.
#[test]
#[cfg_attr(
    all(feature = "mimalloc", not(feature = "benchmark")),
    ignore = "run with --features benchmark,mimalloc to count allocations under mimalloc"
)]
fn content_pack_clone_allocations_do_not_scale_with_definition_bytes() {
    let baseline_game = build_fixture(FIXTURE_ID).expect("deterministic fixture");
    let mut expanded_game = baseline_game.clone();
    // Mutating through the object gives each copy its own definition.
    let extra_text = "x".repeat(SYNTHETIC_TEXT_BYTES);
    for object in expanded_game.state.objects.iter_mut() {
        object.text_box.push_str(&extra_text);
    }
    let payload_delta =
        definition_text_bytes(&expanded_game).saturating_sub(definition_text_bytes(&baseline_game));
    assert!(
        payload_delta >= MINIMUM_PAYLOAD_DELTA_BYTES,
        "expanded immutable payload is only {payload_delta} bytes larger"
    );

    assert_eq!(
        baseline_game.state.objects.len(),
        expanded_game.state.objects.len()
    );
    assert_eq!(
        baseline_game.state.permanents.len(),
        expanded_game.state.permanents.len()
    );
    assert_eq!(
        baseline_game.state.events.len(),
        expanded_game.state.events.len()
    );

    clone_repeatedly(&baseline_game, WARMUP_CLONE_COUNT);
    clone_repeatedly(&expanded_game, WARMUP_CLONE_COUNT);

    let game_state_clone = clone_receipt(&baseline_game.state, &expanded_game.state);
    let exact_game_clone = clone_receipt(&baseline_game, &expanded_game);

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
        fixture_cards: baseline_game.state.objects.len(),
        fixture_permanent_slots: baseline_game.state.permanents.len(),
        fixture_committed_events: baseline_game.state.events.len(),
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
