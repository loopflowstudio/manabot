//! Inline sizes and single-clone allocation traffic for the versioned game fixtures.

use std::{collections::BTreeMap, hint::black_box, mem::size_of};

use serde::Serialize;

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

pub fn inline_sizes() -> BTreeMap<&'static str, usize> {
    BTreeMap::from([
        ("Game", size_of::<Game>()),
        ("GameState", size_of::<GameState>()),
        ("TrackedState", size_of::<TrackedState>()),
        ("DerivedCache", size_of::<DerivedCache>()),
        ("Object", size_of::<Object>()),
        ("Permanent", size_of::<Permanent>()),
    ])
}

#[derive(Serialize)]
pub struct Position<'a> {
    fixture: &'a str,
    authority: String,
    legal: String,
    actions: usize,
    objects: usize,
    slots: usize,
    events: usize,
}

#[derive(Serialize)]
pub struct CloneTraffic<'a> {
    fixture: &'a str,
    field: &'static str,
    inline: usize,
    calls: u64,
    pub bytes: u64,
}

pub struct FixtureFootprint<'a> {
    pub root: Game,
    pub position: Position<'a>,
    pub game: CloneTraffic<'a>,
    pub fields: Vec<CloneTraffic<'a>>,
}

fn measure_clone<'a, T: Clone>(
    fixture: &'a str,
    field: &'static str,
    value: &T,
) -> CloneTraffic<'a> {
    benchmark_allocations::start();
    let copy = black_box(value.clone());
    let (calls, bytes) = benchmark_allocations::stop();
    drop(copy);
    CloneTraffic {
        fixture,
        field,
        inline: size_of::<T>(),
        calls,
        bytes,
    }
}

/// Retain the measured root so the bank can clone that exact position.
pub fn measure_fixture(id: &str, cold_caches: bool) -> Result<FixtureFootprint<'_>, String> {
    let mut root = build_fixture(id)?;
    // Viewer projection can warm a cache. Always witness a disposable fork.
    let original = witness(&root.clone());
    if cold_caches {
        *root.derived.get_mut() = Default::default();
        *root.payment_cache.get_mut() = Default::default();
        assert_eq!(original, witness(&root.clone()), "cold fork of {id}");
    }
    let state = &root.state;
    let position = Position {
        fixture: id,
        authority: original.authority.hash,
        legal: original.legal_surface.hash,
        actions: original.legal_surface.action_count,
        objects: state.objects.len(),
        slots: state.permanents.len(),
        events: state.events.len(),
    };
    let game = measure_clone(id, "Game", &root);
    let fields = vec![
        measure_clone(id, "TrackedState", state),
        measure_clone(id, "CardVec", &state.cards),
        measure_clone(id, "objects", &state.objects),
        measure_clone(id, "permanents", &state.permanents),
        measure_clone(id, "zones", &state.zones),
        measure_clone(id, "players", &state.players),
        measure_clone(id, "suspended_decision", &state.suspended_decision),
        measure_clone(id, "trigger_placement", &state.trigger_placement),
        measure_clone(id, "cast_permissions", &state.cast_permissions),
        measure_clone(id, "effects", &state.effects),
        measure_clone(id, "observation_events", &state.observation_events),
        measure_clone(id, "object_lki", &state.object_lki),
        measure_clone(id, "derived", &*root.derived.borrow()),
        measure_clone(id, "payment_cache", &*root.payment_cache.borrow()),
        measure_clone(id, "pending_choice", &root.pending_choice),
    ];
    Ok(FixtureFootprint {
        root,
        position,
        game,
        fields,
    })
}
