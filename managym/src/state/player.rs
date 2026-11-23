use std::collections::BTreeMap;

use super::{
    card::CardDefId,
    deck::Deck,
    game_object::{CardId, PublicId},
    mana::ManaPool,
};

/// Whether both players know each other's main deck and sideboard.
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

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlayerConfig {
    pub name: String,
    #[serde(flatten)]
    pub deck: Deck,
    /// The deck was submitted to the world ([`crate::world::WORLD_POOL`]):
    /// game creation admits it by development construction
    /// ([`crate::cards::deck_admission::admit_submitted`]). Fixed setups and
    /// tests leave it unset and run as given. Serialized as
    /// `"world": "world"`, the form traces have always recorded.
    #[serde(
        rename = "world",
        skip_serializing_if = "std::ops::Not::not",
        default,
        serialize_with = "serialize_submission",
        deserialize_with = "deserialize_submission"
    )]
    pub submitted: bool,
}

fn serialize_submission<S: serde::Serializer>(_: &bool, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(crate::world::WORLD_POOL)
}

fn deserialize_submission<'de, D: serde::Deserializer<'de>>(d: D) -> Result<bool, D::Error> {
    use serde::{de::Error, Deserialize};
    match Option::<String>::deserialize(d)?.as_deref() {
        None => Ok(false),
        Some(crate::world::WORLD_POOL) => Ok(true),
        Some(_) => Err(D::Error::custom("unknown submitted deck world")),
    }
}

impl PlayerConfig {
    pub fn new(name: impl Into<String>, decklist: BTreeMap<String, usize>) -> Self {
        Self {
            name: name.into(),
            deck: Deck::new(decklist, BTreeMap::new()),
            submitted: false,
        }
    }

    /// Submit the deck to the world, so game creation admits it.
    pub fn submit_to_world(mut self) -> Self {
        self.submitted = true;
        self
    }

    pub fn with_sideboard(mut self, sideboard: BTreeMap<String, usize>) -> Self {
        self.deck.sideboard = sideboard;
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
    pub id: PublicId,
    pub index: usize,
    pub deck: Vec<CardId>,
    /// Immutable owned outside-game roster; availability is derived from zones.
    pub sideboard: Vec<CardId>,
    /// Public minimum counts in hand, without physical-copy or slot identity.
    pub known_hand: BTreeMap<CardDefId, u32>,
    #[serde(skip_serializing_if = "super::knowledge::LibraryKnowledge::is_empty")]
    pub known_library: super::knowledge::LibraryKnowledge,
    pub name: String,
    pub life: i32,
    pub drew_when_empty: bool,
    pub alive: bool,
    pub mana_pool: ManaPool,
    /// Mana that lasts until end of combat (firebending) — persists across
    /// combat steps instead of emptying, empties as combat ends.
    pub combat_mana_pool: ManaPool,
}

impl Player {
    pub const STARTING_LIFE: i32 = 20;

    pub fn new(id: PublicId, index: usize, name: impl Into<String>) -> Self {
        Self {
            id,
            index,
            deck: Vec::new(),
            sideboard: Vec::new(),
            known_hand: BTreeMap::new(),
            known_library: Default::default(),
            name: name.into(),
            life: Self::STARTING_LIFE,
            drew_when_empty: false,
            alive: true,
            mana_pool: ManaPool::default(),
            combat_mana_pool: ManaPool::default(),
        }
    }
}

/// Game-level disclosure selection. Mixed draws one fair, independent coin per
/// reset seed, without consuming shuffle or roster RNG.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DecklistMode {
    #[default]
    Open,
    Closed,
    Mixed,
}

impl From<DecklistDisclosure> for DecklistMode {
    fn from(disclosure: DecklistDisclosure) -> Self {
        match disclosure {
            DecklistDisclosure::Open => Self::Open,
            DecklistDisclosure::Closed => Self::Closed,
        }
    }
}

impl DecklistMode {
    pub fn resolve(self, seed: u64) -> DecklistDisclosure {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        match self {
            Self::Open => DecklistDisclosure::Open,
            Self::Closed => DecklistDisclosure::Closed,
            Self::Mixed => {
                let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x646973636c6f7365);
                if rng.gen_bool(0.5) {
                    DecklistDisclosure::Open
                } else {
                    DecklistDisclosure::Closed
                }
            }
        }
    }
}

/// Who chooses sources and colors after a seat commits to a mana payment.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ManaPayment {
    Auto,
    #[default]
    Manual,
}

impl ManaPayment {
    pub fn is_manual(&self) -> bool {
        *self == Self::Manual
    }

    /// The serialized and Python name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Manual => "manual",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        [Self::Auto, Self::Manual]
            .into_iter()
            .find(|mode| mode.name() == name)
    }
}

/// Priority float offers for the chooser.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ManaFloats {
    #[default]
    Off,
    Usable,
    All,
}

impl ManaFloats {
    pub fn is_off(&self) -> bool {
        *self == Self::Off
    }

    /// The serialized and Python name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Usable => "usable",
            Self::All => "all",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        [Self::Off, Self::Usable, Self::All]
            .into_iter()
            .find(|mode| mode.name() == name)
    }
}

/// Complete root setup; disclosure belongs to the game, never a seat.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GameSetup {
    pub players: Vec<PlayerConfig>,
    pub decklists: DecklistMode,
    pub mana_floats: [ManaFloats; 2],
    pub mana_payment: [ManaPayment; 2],
}

impl From<Vec<PlayerConfig>> for GameSetup {
    fn from(players: Vec<PlayerConfig>) -> Self {
        Self {
            players,
            decklists: DecklistMode::Open,
            mana_floats: [ManaFloats::Off; 2],
            mana_payment: [ManaPayment::Manual; 2],
        }
    }
}

impl std::ops::Deref for GameSetup {
    type Target = [PlayerConfig];
    fn deref(&self) -> &Self::Target {
        &self.players
    }
}
