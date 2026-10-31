use std::collections::BTreeMap;

use super::{
    card::CardDefId,
    deck::Deck,
    game_object::{CardId, ObjectId},
    mana::Mana,
};

/// Whether a player's opponent knows that player's main deck and sideboard.
/// An open decklist discloses both lists and a closed one neither; there is
/// no partial disclosure. Open is the default, and records that predate the
/// option read as open.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum DecklistDisclosure {
    #[default]
    Open,
    Closed,
}

impl DecklistDisclosure {
    pub fn is_open(&self) -> bool {
        *self == Self::Open
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct PlayerConfig {
    pub name: String,
    #[serde(flatten)]
    pub deck: Deck,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub world: Option<String>,
    /// Serialized only when closed, so default-mode digests are unchanged.
    #[serde(skip_serializing_if = "DecklistDisclosure::is_open")]
    pub disclosure: DecklistDisclosure,
}

impl PlayerConfig {
    pub fn new(name: impl Into<String>, decklist: BTreeMap<String, usize>) -> Self {
        Self {
            name: name.into(),
            deck: Deck::new(decklist, BTreeMap::new()),
            world: None,
            disclosure: DecklistDisclosure::Open,
        }
    }

    pub fn with_sideboard(mut self, sideboard: BTreeMap<String, usize>) -> Self {
        self.deck.sideboard = sideboard;
        self
    }

    pub fn with_disclosure(mut self, disclosure: DecklistDisclosure) -> Self {
        self.disclosure = disclosure;
        self
    }

    pub fn deck_list(&self) -> String {
        self.deck
            .main
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
    /// What the opponent knows of `deck` and `sideboard`; fixed for the game.
    #[serde(skip_serializing_if = "DecklistDisclosure::is_open")]
    pub disclosure: DecklistDisclosure,
    /// Public minimum counts in hand, without physical-copy or slot identity.
    pub known_hand: BTreeMap<CardDefId, u32>,
    pub name: String,
    pub life: i32,
    pub drew_when_empty: bool,
    /// "You have no maximum hand size for the rest of the game" (CR 402.2).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub no_maximum_hand_size: bool,
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
            disclosure: DecklistDisclosure::Open,
            known_hand: BTreeMap::new(),
            name: name.into(),
            life: 20,
            drew_when_empty: false,
            no_maximum_hand_size: false,
            alive: true,
            mana_pool: Mana::default(),
            combat_mana_pool: Mana::default(),
        }
    }
}
