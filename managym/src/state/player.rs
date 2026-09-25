use std::collections::BTreeMap;

use super::{
    card::CardDefId,
    game_object::{CardId, ObjectId},
    mana::Mana,
};

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct PlayerConfig {
    pub name: String,
    pub decklist: BTreeMap<String, usize>,
    pub sideboard: BTreeMap<String, usize>,
}

impl PlayerConfig {
    pub fn new(name: impl Into<String>, decklist: BTreeMap<String, usize>) -> Self {
        Self {
            name: name.into(),
            decklist,
            sideboard: BTreeMap::new(),
        }
    }

    pub fn with_sideboard(mut self, sideboard: BTreeMap<String, usize>) -> Self {
        self.sideboard = sideboard;
        self
    }

    pub fn deck_list(&self) -> String {
        self.decklist
            .iter()
            .map(|(name, qty)| format!("{} x{}", name, qty))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Player {
    pub id: ObjectId,
    pub index: usize,
    pub deck: Vec<CardId>,
    /// Immutable owned outside-game roster; availability is derived from zones.
    pub sideboard: Vec<CardId>,
    /// Public minimum counts in hand, without physical-copy or slot identity.
    pub known_hand: BTreeMap<CardDefId, u32>,
    pub name: String,
    pub life: i32,
    pub drew_when_empty: bool,
    pub alive: bool,
    pub mana_pool: Mana,
    /// Mana that lasts until end of combat (firebending) — persists across
    /// combat steps instead of emptying, empties as combat ends.
    pub combat_mana_pool: Mana,
}

impl Player {
    pub fn new(id: ObjectId, index: usize, name: impl Into<String>) -> Self {
        Self {
            id,
            index,
            deck: Vec::new(),
            sideboard: Vec::new(),
            known_hand: BTreeMap::new(),
            name: name.into(),
            life: 20,
            drew_when_empty: false,
            alive: true,
            mana_pool: Mana::default(),
            combat_mana_pool: Mana::default(),
        }
    }
}
