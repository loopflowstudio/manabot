//! Inline sizes and single-clone allocation traffic for the versioned game fixtures.

use std::{hint::black_box, mem::size_of};

use serde_json::json;

use crate::{
    benchmark_allocations,
    flow::{
        game::{GameState, TrackedState},
        statics::DerivedCache,
    },
    match_fixtures::build_fixture,
    search_state::witness,
    state::{game_object::Object, permanent::Permanent},
    Game,
};

pub fn inline_sizes() -> serde_json::Value {
    json!({
        "Game": size_of::<Game>(),
        "GameState": size_of::<GameState>(),
        "TrackedState": size_of::<TrackedState>(),
        "DerivedCache": size_of::<DerivedCache>(),
        "Object": size_of::<Object>(),
        "Permanent": size_of::<Permanent>(),
    })
}

fn measure_clone<T: Clone>(fixture: &str, field: &str, value: &T) -> u64 {
    benchmark_allocations::start();
    let copy = black_box(value.clone());
    let (calls, bytes) = benchmark_allocations::stop();
    drop(copy);
    println!(
        "{}",
        json!({
            "fixture": fixture, "field": field, "inline": size_of::<T>(), "calls": calls, "bytes": bytes,
        })
    );
    bytes
}

/// Print one root's witness and field measurements. Return the measured source
/// and its requested clone heap bytes so the bank can retain identical forks.
pub fn measure_fixture(id: &str, cold_caches: bool) -> Result<(Game, u64), String> {
    let mut root = build_fixture(id)?;
    // Viewer projection can warm a cache. Always witness a disposable fork.
    let original = witness(&root.clone());
    if cold_caches {
        *root.derived.get_mut() = Default::default();
        *root.payment_cache.get_mut() = Default::default();
        assert_eq!(original, witness(&root.clone()), "cold fork of {id}");
    }
    let state = &root.state;
    println!(
        "{}",
        json!({
            "fixture": id, "authority": original.authority.hash,
            "legal": original.legal_surface.hash, "actions": original.legal_surface.action_count,
            "objects": state.objects.len(), "slots": state.permanents.len(), "events": state.events.len(),
        })
    );
    let bytes = measure_clone(id, "Game", &root);
    measure_clone(id, "TrackedState", state);
    measure_clone(id, "CardVec", &state.cards);
    measure_clone(id, "objects", &state.objects);
    measure_clone(id, "permanents", &state.permanents);
    measure_clone(id, "zones", &state.zones);
    measure_clone(id, "players", &state.players);
    measure_clone(id, "suspended_decision", &state.suspended_decision);
    measure_clone(id, "trigger_placement", &state.trigger_placement);
    measure_clone(id, "cast_permissions", &state.cast_permissions);
    measure_clone(id, "effects", &state.effects);
    measure_clone(id, "observation_events", &state.observation_events);
    measure_clone(id, "object_lki", &state.object_lki);
    measure_clone(id, "derived", &*root.derived.borrow());
    measure_clone(id, "payment_cache", &*root.payment_cache.borrow());
    measure_clone(id, "pending_choice", &root.pending_choice);
    Ok((root, bytes))
}
