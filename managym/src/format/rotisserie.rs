//! Public shared-supply drafting before Limited construction.
//!
//! `Rotisserie` commits episode/revision/chooser-bound picks atomically. Its
//! owned observation exposes every pool and remaining copy on the catalog
//! axis, with one legal choice per definition. A completed pool feeds the
//! existing `LimitedBuild`; there are no in-game Actions or hidden game state.

use serde::Serialize;

use crate::{
    cards::{
        printings::{printing_digest, set_printings},
        ContentError,
    },
    format::limited::{card_count, CardPool},
    state::card::CardDefId,
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RotisserieObservation {
    pub episode_id: String,
    pub set: String,
    pub printing_digest: String,
    /// Number of committed picks. Round is revision / pools.len().
    pub revision: usize,
    pub picks_per_seat: usize,
    /// None after every seat has made its picks.
    pub chooser: Option<usize>,
    pub initial: Vec<u32>,
    pub remaining: Vec<u32>,
    pub pools: Vec<Vec<u32>>,
    pub legal: Vec<CardDefId>,
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "python", pyo3::pyclass(module = "managym.format"))]
pub struct Rotisserie {
    state: RotisserieObservation,
}

impl Rotisserie {
    pub fn new(
        set: &str,
        episode_id: &str,
        seats: usize,
        picks_per_seat: usize,
    ) -> Result<Self, ContentError> {
        if episode_id.is_empty() {
            return Err(ContentError("draft episode_id must not be empty".into()));
        }
        let rows = set_printings(set)?;
        let mut initial = vec![0; card_count()];
        for row in rows {
            initial[row.card_def_id.0 as usize] = row.printing.rarity.rotisserie_copies();
        }
        let total = initial.iter().map(|&n| n as usize).sum::<usize>();
        let picks = seats.checked_mul(picks_per_seat);
        if seats < 2 || picks_per_seat == 0 || picks.is_none_or(|n| n > total) {
            return Err(ContentError(format!(
                "{set}: seats and picks must fit the shared supply of {total}"
            )));
        }
        let mut draft = Self {
            state: RotisserieObservation {
                episode_id: episode_id.into(),
                set: set.into(),
                printing_digest: printing_digest(set)?,
                revision: 0,
                picks_per_seat,
                chooser: Some(0),
                remaining: initial.clone(),
                initial,
                pools: vec![vec![0; card_count()]; seats],
                legal: Vec::new(),
            },
        };
        draft.refresh();
        Ok(draft)
    }

    pub fn observe(&self) -> RotisserieObservation {
        self.state.clone()
    }

    fn refresh(&mut self) {
        let seats = self.state.pools.len();
        let round = self.state.revision / seats;
        let offset = self.state.revision % seats;
        self.state.chooser = (round < self.state.picks_per_seat).then(|| {
            if round.is_multiple_of(2) {
                offset
            } else {
                seats - offset - 1
            }
        });
        self.state.legal = if self.state.chooser.is_some() {
            self.state
                .remaining
                .iter()
                .enumerate()
                .filter(|(_, n)| **n > 0)
                .map(|(i, _)| CardDefId(i as u32))
                .collect()
        } else {
            Vec::new()
        };
    }

    /// Rejection leaves all state intact; callers journal this bound choice
    /// before committing it. No policy can select a physical copy separately.
    pub fn pick(
        &mut self,
        episode_id: &str,
        revision: usize,
        chooser: usize,
        card: CardDefId,
    ) -> Result<(), ContentError> {
        if episode_id != self.state.episode_id
            || revision != self.state.revision
            || Some(chooser) != self.state.chooser
        {
            return Err(ContentError(
                "draft pick has wrong episode, stale revision or wrong chooser".into(),
            ));
        }
        let index = card.0 as usize;
        if self.state.remaining.get(index).copied().unwrap_or(0) == 0 {
            return Err(ContentError(format!(
                "definition {}: not available in draft",
                card.0
            )));
        }
        self.state.remaining[index] -= 1;
        self.state.pools[chooser][index] += 1;
        self.state.revision += 1;
        self.refresh();
        Ok(())
    }

    /// Construction cannot begin until all public drafting has finished.
    pub fn pool(&self, seat: usize) -> Result<CardPool, ContentError> {
        if self.state.chooser.is_some() {
            return Err(ContentError("draft is not complete".into()));
        }
        CardPool::from_counts(
            self.state
                .pools
                .get(seat)
                .ok_or_else(|| ContentError(format!("unknown draft seat {seat}")))?,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotisserie_conserves_every_copy_and_snakes_through_complete_pools() {
        let mut draft = Rotisserie::new("tla", "episode", 8, 45).unwrap();
        assert!(draft.pool(0).is_err());
        for revision in 0..360 {
            let before = draft.observe();
            let chooser = if (revision / 8usize).is_multiple_of(2) {
                revision % 8
            } else {
                7 - revision % 8
            };
            assert_eq!(before.chooser, Some(chooser));
            for (ep, rev, seat, card) in [
                ("other", revision, chooser, before.legal[0]),
                ("episode", revision + 1, chooser, before.legal[0]),
                ("episode", revision, (chooser + 1) % 8, before.legal[0]),
                ("episode", revision, chooser, CardDefId(u32::MAX)),
            ] {
                assert!(draft.pick(ep, rev, seat, card).is_err());
                assert_eq!(draft.observe(), before);
            }
            draft
                .pick("episode", revision, chooser, before.legal[0])
                .unwrap();
            let after = draft.observe();
            for i in 0..card_count() {
                assert_eq!(
                    after.remaining[i] + after.pools.iter().map(|p| p[i]).sum::<u32>(),
                    after.initial[i]
                );
            }
        }
        let done = draft.observe();
        assert_eq!(done.chooser, None);
        assert!(done.legal.is_empty());
        assert_eq!(done.remaining.iter().sum::<u32>(), 304);
        for seat in 0..8 {
            assert_eq!(draft.pool(seat).unwrap().counts().iter().sum::<u32>(), 45);
        }
        assert!(draft.pick("episode", 360, 0, CardDefId(0)).is_err());
        assert!(draft.pool(8).is_err());
    }

    #[test]
    fn rotisserie_rejects_invalid_setup_and_exhausted_card_without_mutation() {
        for (seats, picks) in [(0, 45), (1, 45), (8, 0), (8, 84), (usize::MAX, 2)] {
            assert!(Rotisserie::new("tla", "episode", seats, picks).is_err());
        }
        assert!(Rotisserie::new("tla", "", 8, 45).is_err());
        let mut draft = Rotisserie::new("tla", "episode", 8, 45).unwrap();
        let card = draft.observe().legal[0];
        while draft.observe().remaining[card.0 as usize] > 0 {
            let state = draft.observe();
            draft
                .pick("episode", state.revision, state.chooser.unwrap(), card)
                .unwrap();
        }
        let state = draft.observe();
        assert!(!state.legal.contains(&card));
        assert!(draft
            .pick("episode", state.revision, state.chooser.unwrap(), card)
            .is_err());
        assert_eq!(draft.observe(), state);
    }
}
