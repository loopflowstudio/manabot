use std::collections::BTreeMap;

use super::{
    game_object::{CardId, ObjectId},
    mana::Mana,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlayerConfig {
    pub name: String,
    pub decklist: BTreeMap<String, usize>,
}

impl PlayerConfig {
    pub fn new(name: impl Into<String>, decklist: BTreeMap<String, usize>) -> Self {
        Self {
            name: name.into(),
            decklist,
        }
    }

    pub fn deck_list(&self) -> String {
        self.decklist
            .iter()
            .map(|(name, qty)| format!("{} x{}", name, qty))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Player {
    pub id: ObjectId,
    pub index: usize,
    pub deck: Vec<CardId>,
    pub name: String,
    pub life: i32,
    pub drew_when_empty: bool,
    pub alive: bool,
    pub mana_pool: Mana,
}

impl Player {
    pub fn new(id: ObjectId, index: usize, name: impl Into<String>) -> Self {
        Self {
            id,
            index,
            deck: Vec::new(),
            name: name.into(),
            life: 20,
            drew_when_empty: false,
            alive: true,
            mana_pool: Mana::default(),
        }
    }

    pub fn take_damage(&mut self, damage: i32) {
        self.life -= damage;
    }
}
