//! Deprecated policy inputs v9 and v10: the object rows before input v11.
//!
//! Input v11 gives each game thing one row: cards outside the battlefield and
//! stack, permanents with their characteristics, and one stack segment. The
//! checkpoints trained before it read v9/v10 rows, in which a battlefield
//! object or a stack spell was also a card row, permanent rows had no
//! characteristics and stack abilities had no row. This module keeps those
//! rows for those checkpoints, selected by `InputLayout::Legacy`, and nothing
//! else depends on it.
//!
//! Its whole surface is frozen; deleting it (and the `InputLayout::Legacy`
//! branches that call it) removes inputs v9 and v10. Removal release: 0.4.0.
//! MTG-353 owns the general mechanism for checkpoints across worlds.

use std::collections::HashMap;

use super::{
    observation::{CardData, Observation},
    policy_history::{encode_rows, HISTORY_DIM},
};

/// A v9/v10 permanent row ends at `dealt_damage_this_turn`, then validity.
pub const PERMANENT_VALID: usize = 37;
pub const PERMANENT_DIM: usize = PERMANENT_VALID + 1;

/// `obs` with v9/v10 card lists: after each side's v11 cards, that side's
/// stack spells (by owner, top first), then its battlefield objects (by
/// controller, in permanent order). A historical capture, whose card lists
/// already hold them, is returned unchanged.
pub fn view(obs: &Observation) -> Observation {
    let mut legacy = obs.clone();
    let historical = obs
        .agent_permanents
        .iter()
        .chain(&obs.opponent_permanents)
        .any(|permanent| permanent.card.is_none());
    if historical {
        return legacy;
    }
    let spells = obs
        .stack_objects
        .iter()
        .filter_map(|object| object.card.as_ref());
    for spell in spells {
        if spell.owner_id == obs.agent.id {
            legacy.agent_cards.push(spell.clone());
        } else {
            legacy.opponent_cards.push(spell.clone());
        }
    }
    let battlefield = |permanents: &[super::observation::PermanentData]| {
        permanents
            .iter()
            .filter_map(|permanent| permanent.card.clone())
            .collect::<Vec<CardData>>()
    };
    legacy
        .agent_cards
        .extend(battlefield(&obs.agent_permanents));
    legacy
        .opponent_cards
        .extend(battlefield(&obs.opponent_permanents));
    legacy
}

/// History v1 rows of a v9/v10 view, whose contexts resolve only to card rows.
pub fn history(
    obs: &Observation,
    capacity: usize,
    cards_per_player: usize,
) -> Vec<[f32; HISTORY_DIM]> {
    let agent: HashMap<i32, usize> = obs
        .agent_cards
        .iter()
        .enumerate()
        .rev()
        .map(|(slot, card)| (card.id, 2 + slot))
        .collect();
    let opponent: HashMap<i32, usize> = obs
        .opponent_cards
        .iter()
        .enumerate()
        .rev()
        .map(|(slot, card)| (card.id, 2 + cards_per_player + slot))
        .collect();
    encode_rows(obs, capacity, |id| {
        agent.get(&id).or_else(|| opponent.get(&id)).copied()
    })
}
