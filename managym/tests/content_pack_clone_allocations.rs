use std::{
    alloc::{GlobalAlloc, Layout, System},
    hint::black_box,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
};

use managym::{
    benchmark::build_fixture,
    cardsets::alpha::{default_content_pack, ContentPack},
    state::card::CardDefinition,
    Game,
};
use serde::Serialize;

const FIXTURE_ID: &str = "interactive-heavy-80-v1";
const ARC_REFERENCE_COUNT: usize = 4_096;
const WARMUP_CLONE_COUNT: usize = 64;
const MEASURED_CLONE_COUNT: usize = 1_024;
const EXPANDED_DEFINITION_COUNT: usize = 256;
const SYNTHETIC_TEXT_BYTES: usize = 16 * 1_024;
const MINIMUM_PAYLOAD_DELTA_BYTES: usize = 4 * 1_024 * 1_024;

struct CountingAllocator;

static MEASURING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if MEASURING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        System.alloc(layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if MEASURING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        System.alloc_zeroed(layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if MEASURING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        }
        System.realloc(ptr, layout, new_size)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
struct AllocationCounts {
    allocations: u64,
    allocated_bytes: u64,
}

#[derive(Debug, Serialize)]
struct PackReceipt {
    definitions: usize,
    content_digest: String,
    serialized_definition_bytes: usize,
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
    arc_reference_count: usize,
    warmup_clone_count: usize,
    measured_clone_count: usize,
    synthetic_definition_count: usize,
    synthetic_text_bytes_each: usize,
    minimum_payload_delta_bytes: usize,
    actual_payload_delta_bytes: usize,
    baseline_pack: PackReceipt,
    expanded_pack: PackReceipt,
    arc_reference_clone: CloneReceipt,
    game_state_clone: CloneReceipt,
    exact_game_clone: CloneReceipt,
    threshold: &'static str,
    passed: bool,
}

fn measure(operation: impl FnOnce()) -> AllocationCounts {
    assert!(!MEASURING.swap(true, Ordering::SeqCst));
    ALLOCATIONS.store(0, Ordering::SeqCst);
    ALLOCATED_BYTES.store(0, Ordering::SeqCst);
    operation();
    MEASURING.store(false, Ordering::SeqCst);
    AllocationCounts {
        allocations: ALLOCATIONS.load(Ordering::SeqCst),
        allocated_bytes: ALLOCATED_BYTES.load(Ordering::SeqCst),
    }
}

fn serialized_definition_bytes(pack: &ContentPack) -> usize {
    let definitions: Vec<_> = pack.definitions().collect();
    serde_json::to_vec(&definitions)
        .expect("definitions serialize")
        .len()
}

fn pack_receipt(pack: &ContentPack) -> PackReceipt {
    PackReceipt {
        definitions: pack.len(),
        content_digest: pack.content_digest(),
        serialized_definition_bytes: serialized_definition_bytes(pack),
    }
}

fn expanded_pack(baseline: &ContentPack) -> ContentPack {
    let mut expanded = baseline.clone();
    for index in 0..EXPANDED_DEFINITION_COUNT {
        expanded.register_card(CardDefinition {
            name: format!("W2-208 allocation control {index:03}"),
            text_box: "x".repeat(SYNTHETIC_TEXT_BYTES),
            ..CardDefinition::default()
        });
    }
    expanded
}

fn measure_arc_references(content: &Arc<ContentPack>) -> AllocationCounts {
    let before = Arc::strong_count(content);
    let mut retained = Vec::with_capacity(ARC_REFERENCE_COUNT);
    let counts = measure(|| {
        for _ in 0..ARC_REFERENCE_COUNT {
            retained.push(Arc::clone(content));
        }
    });
    assert_eq!(
        Arc::strong_count(content),
        before + ARC_REFERENCE_COUNT,
        "the reference-bookkeeping workload must execute"
    );
    drop(retained);
    assert_eq!(Arc::strong_count(content), before);
    counts
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

#[test]
fn content_pack_clone_allocations_do_not_scale_with_definition_bytes() {
    let (root, fixture) = build_fixture(FIXTURE_ID).expect("deterministic fixture");
    let admitted = default_content_pack();
    assert!(Arc::ptr_eq(&root.state.content, &admitted));

    let baseline_content = Arc::new((*admitted).clone());
    let expanded_content = Arc::new(expanded_pack(&baseline_content));
    let baseline_pack_receipt = pack_receipt(&baseline_content);
    let expanded_pack_receipt = pack_receipt(&expanded_content);
    let payload_delta = expanded_pack_receipt.serialized_definition_bytes
        - baseline_pack_receipt.serialized_definition_bytes;
    assert_ne!(
        baseline_pack_receipt.content_digest, expanded_pack_receipt.content_digest,
        "the size control must contain different immutable content"
    );
    assert!(
        payload_delta >= MINIMUM_PAYLOAD_DELTA_BYTES,
        "expanded immutable payload is only {payload_delta} bytes larger"
    );

    let mut baseline_game = root.clone();
    baseline_game.state.content = Arc::clone(&baseline_content);
    let mut expanded_game = baseline_game.clone();
    expanded_game.state.content = Arc::clone(&expanded_content);
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

    let arc_reference_clone = clone_receipt(
        measure_arc_references(&baseline_content),
        measure_arc_references(&expanded_content),
    );
    let game_state_clone = clone_receipt(
        measure_game_state_clones(&baseline_game),
        measure_game_state_clones(&expanded_game),
    );
    let exact_game_clone = clone_receipt(
        measure_game_clones(&baseline_game),
        measure_game_clones(&expanded_game),
    );

    for counts in [arc_reference_clone.baseline, arc_reference_clone.expanded] {
        assert_eq!(counts.allocations, 0, "Arc cloning allocated storage");
        assert_eq!(counts.allocated_bytes, 0, "Arc cloning allocated bytes");
    }
    for receipt in [&game_state_clone, &exact_game_clone] {
        assert!(receipt.baseline.allocations > 0);
        assert!(receipt.baseline.allocated_bytes > 0);
        assert_eq!(receipt.delta_allocations, 0);
        assert_eq!(receipt.delta_allocated_bytes, 0);
    }

    let receipt = AllocationReceipt {
        schema: "manabot.content-pack-clone-allocations.v1",
        build_profile: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        fixture: FIXTURE_ID,
        fixture_cards: fixture.card_count,
        fixture_permanent_slots: fixture.allocated_permanent_slots,
        fixture_committed_events: fixture.committed_event_count,
        arc_reference_count: ARC_REFERENCE_COUNT,
        warmup_clone_count: WARMUP_CLONE_COUNT,
        measured_clone_count: MEASURED_CLONE_COUNT,
        synthetic_definition_count: EXPANDED_DEFINITION_COUNT,
        synthetic_text_bytes_each: SYNTHETIC_TEXT_BYTES,
        minimum_payload_delta_bytes: MINIMUM_PAYLOAD_DELTA_BYTES,
        actual_payload_delta_bytes: payload_delta,
        baseline_pack: baseline_pack_receipt,
        expanded_pack: expanded_pack_receipt,
        arc_reference_clone,
        game_state_clone,
        exact_game_clone,
        threshold: "Arc allocations and bytes must be 0; mutable clone totals must be positive; expanded-minus-baseline clone allocations and bytes must equal 0 exactly",
        passed: true,
    };
    println!(
        "{}",
        serde_json::to_string(&receipt).expect("allocation receipt serializes")
    );
}
