//! Logical search-state witness for one exact rules state.
//!
//! Search forks by full clone. A witness contains logical rules facts only and
//! deliberately excludes pointers, allocator layout, timings and RSS.
//! `ContentPack` definitions are named once by digest; every mutable match fact
//! a fork must preserve is serialized.

use rand::RngCore;
use serde::Serialize;

use crate::{
    agent::{action::ActionSpace, observation::Observation},
    flow::game::PendingChoice,
    state::game_object::PlayerId,
    Game,
};

/// Schema 2 closes the identity and fixed-viewer omissions in the benchmark's
/// original shallow schema 1 witness.
pub const SEARCH_WITNESS_SCHEMA_VERSION: u32 = 2;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthorityFingerprint {
    /// Canonical bytes for the complete mutable rules authority.
    pub bytes: Vec<u8>,
    pub hash: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LegalSurfaceFingerprint {
    pub bytes: Vec<u8>,
    pub hash: String,
    pub action_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ViewerProjectionWitness {
    pub bytes: Vec<u8>,
    pub hash: String,
}

impl ViewerProjectionWitness {
    fn new(observation: Observation) -> Self {
        let bytes = observation.to_json().into_bytes();
        Self {
            hash: blake3::hash(&bytes).to_hex().to_string(),
            bytes,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContractDiagnostics {
    pub event_boundary: [usize; 3],
    pub rng_probe: [u64; 8],
    pub terminal: bool,
}

/// Evidence used to compare two search states without implying that the
/// evidence can itself restore either state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SearchStateWitness {
    pub authority: AuthorityFingerprint,
    pub legal_surface: LegalSurfaceFingerprint,
    /// Compatibility projection for the current acting-player observation.
    pub acting_projection: ViewerProjectionWitness,
    pub viewers: [ViewerProjectionWitness; 2],
    pub diagnostics: ContractDiagnostics,
}

#[derive(Serialize)]
struct CanonicalAuthorityState<'a> {
    schema_version: u32,
    match_state: serde_json::Value,
    current_action_space: &'a Option<ActionSpace>,
    decision_epoch: u64,
    pending_choice: &'a Option<PendingChoice>,
    skip_trivial: bool,
    skip_trivial_count: usize,
    rng_probe: [u64; 8],
}

/// The authority and legal surface checked before committing a search Command.
/// Viewer projections belong to post-apply evidence, not Command validation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SearchStateIdentity {
    pub authority: AuthorityFingerprint,
    pub legal_surface: LegalSurfaceFingerprint,
}

fn rng_probe(game: &Game) -> [u64; 8] {
    let mut rng = game.state.rng.clone();
    std::array::from_fn(|_| rng.next_u64())
}

#[cfg(feature = "python")]
pub(crate) fn identity(game: &Game) -> SearchStateIdentity {
    identity_with_rng(game, rng_probe(game))
}

fn identity_with_rng(game: &Game, rng_probe: [u64; 8]) -> SearchStateIdentity {
    #[cfg(feature = "benchmark")]
    let _authority = crate::benchmark_search::Timer::new("witness/authority");
    let authority_state = CanonicalAuthorityState {
        schema_version: SEARCH_WITNESS_SCHEMA_VERSION,
        match_state: game.state.deterministic_hash_value(),
        current_action_space: &game.current_action_space,
        decision_epoch: game.decision_epoch,
        pending_choice: &game.pending_choice,
        skip_trivial: game.skip_trivial,
        skip_trivial_count: game.skip_trivial_count,
        rng_probe,
    };
    let authority_bytes = serde_json::to_vec(&authority_state).expect("authority state serializes");
    let action_bytes = serde_json::to_vec(&game.current_action_space)
        .expect("action space serializes deterministically");
    SearchStateIdentity {
        authority: AuthorityFingerprint {
            hash: blake3::hash(&authority_bytes).to_hex().to_string(),
            bytes: authority_bytes,
        },
        legal_surface: LegalSurfaceFingerprint {
            hash: blake3::hash(&action_bytes).to_hex().to_string(),
            action_count: game
                .current_action_space
                .as_ref()
                .map_or(0, |space| space.actions.len()),
            bytes: action_bytes,
        },
    }
}

/// Derive independent authority, legal-surface, viewer, and diagnostic evidence
/// for one exact logical rules state using fixed serialization order.
pub fn witness(game: &Game) -> SearchStateWitness {
    #[cfg(feature = "benchmark")]
    let _witness = crate::benchmark_search::Timer::new("witness_construction");
    let rng_probe = rng_probe(game);
    let SearchStateIdentity {
        authority,
        legal_surface,
    } = identity_with_rng(game, rng_probe);
    #[cfg(feature = "benchmark")]
    let _views = crate::benchmark_search::Timer::new("witness/projections_and_hashes");
    let viewers = [PlayerId(0), PlayerId(1)]
        .map(|viewer| ViewerProjectionWitness::new(Observation::for_player(game, viewer)));
    // The acting projection includes presentation events and exposes actions.
    // With no events and a published chooser, its inputs equal that fixed
    // viewer's projection. Reuse bytes and hash only in that exact case.
    let acting_projection = if game.state.observation_events.is_empty()
        && game
            .current_action_space
            .as_ref()
            .and_then(|space| space.player)
            == Some(game.agent_player())
    {
        viewers[game.agent_player().0].clone()
    } else {
        ViewerProjectionWitness::new(Observation::new(game, &game.state.observation_events))
    };
    SearchStateWitness {
        authority,
        legal_surface,
        acting_projection,
        viewers,
        diagnostics: ContractDiagnostics {
            event_boundary: [
                game.state.events.len(),
                game.state.pending_events.len(),
                game.state.observation_events.len(),
            ],
            rng_probe,
            terminal: game.is_game_over(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        flow::event::GameEvent,
        state::{
            game_object::{CardId, PermanentId},
            zone::ZoneType,
        },
    };

    fn assert_authority_change(baseline: &SearchStateWitness, changed: &Game, field: &str) {
        let changed = witness(changed);
        assert_ne!(
            baseline.authority.bytes, changed.authority.bytes,
            "{field} bytes"
        );
        assert_ne!(
            baseline.authority.hash, changed.authority.hash,
            "{field} hash"
        );
    }

    #[test]
    fn authority_fingerprint_covers_identity_rng_events_allocation_zones_and_decisions() {
        let game = crate::match_fixtures::build_fixture("interactive-midgame-48-w4-v1")
            .expect("contract fixture");
        let baseline = witness(&game);

        let mut incarnation = game.clone();
        incarnation.state.object_incarnations[CardId(0)] = incarnation.state.object_incarnations
            [CardId(0)]
        .checked_next()
        .expect("next incarnation");
        assert_authority_change(&baseline, &incarnation, "incarnation");

        let permanent = PermanentId(
            game.state
                .permanents
                .iter()
                .position(Option::is_some)
                .expect("fixture has a live permanent"),
        );
        let card = game.state.permanents[permanent]
            .as_ref()
            .expect("live permanent")
            .card;
        let lki = game
            .snapshot_current_permanent(card)
            .expect("live permanent has snapshot facts");
        let mut object_lki = game.clone();
        object_lki.state.object_lki.insert(lki);
        assert_authority_change(&baseline, &object_lki, "object LKI");

        let mut rng = game.clone();
        rng.reseed(0x197_5eed);
        assert_authority_change(&baseline, &rng, "RNG continuation");

        let mut events = game.clone();
        let turn_number = events.state.turn.turn_number;
        events.state.events.push(GameEvent::TurnStarted {
            player: PlayerId(0),
            turn_number,
        });
        assert_authority_change(&baseline, &events, "event ledger");

        let mut allocation = game.clone();
        allocation.state.id_gen.next_id();
        assert_authority_change(&baseline, &allocation, "allocation watermark");

        let mut zone_order = game.clone();
        let library = zone_order
            .state
            .zones
            .zone_cards_mut(ZoneType::Library, PlayerId(0));
        let last = library.len() - 1;
        library.swap(0, last);
        assert_authority_change(&baseline, &zone_order, "zone order");

        let mut decision = game.clone();
        decision.decision_epoch += 1;
        assert_authority_change(&baseline, &decision, "decision epoch");
    }
}
