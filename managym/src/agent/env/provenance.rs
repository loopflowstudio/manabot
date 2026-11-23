//! Private inputs for replaying the external calls to one scalar game.
use super::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub(super) enum RecordedAction {
    Index(i64),
    Command(String),
}

fn yes() -> bool {
    true
}
fn is_true(value: &bool) -> bool {
    *value
}
fn agents() -> [PlayerPolicy; 2] {
    [PlayerPolicy::Agent; 2]
}
fn both_agents(policies: &[PlayerPolicy; 2]) -> bool {
    *policies == agents()
}

/// A fork shares its origin and committed prefix until its first write.
/// An unjournaled mutation drops the prefix while retaining the reset setup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct GameProvenance {
    #[serde(rename = "player_configs", with = "seats")]
    pub setup: Arc<GameSetup>,
    #[serde(default = "yes", skip_serializing_if = "is_true")]
    pub skip_trivial: bool,
    #[serde(default = "agents", skip_serializing_if = "both_agents")]
    pub player_policies: [PlayerPolicy; 2],
    #[serde(default = "yes", skip_serializing_if = "is_true")]
    pub decision_memory: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub decision_memory_include_own: bool,
    pub actions: Arc<Vec<RecordedAction>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub interventions: Arc<Vec<BoundaryIntervention>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct BoundaryIntervention {
    pub at_action: usize,
    #[serde(flatten)]
    pub op: Intervention,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Intervention {
    Reseed { seed: u64 },
}

impl GameProvenance {
    pub fn record_reseed(&mut self, seed: u64) {
        Arc::make_mut(&mut self.interventions).push(BoundaryIntervention {
            at_action: self.actions.len(),
            op: Intervention::Reseed { seed },
        });
    }

    pub fn push(&mut self, action: RecordedAction) {
        Arc::make_mut(&mut self.actions).push(action);
    }
}

// Retained evidence spells game-level settings on each seat.
mod seats {
    use super::*;
    use crate::state::deck::Deck;
    use serde::{de::Error, Deserializer, Serializer};

    #[derive(Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Seat {
        name: String,
        decklist: BTreeMap<String, usize>,
        #[serde(default)]
        sideboard: BTreeMap<String, usize>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        world: Option<String>,
        #[serde(default, skip_serializing_if = "ManaFloats::is_off")]
        mana_floats: ManaFloats,
        #[serde(default, skip_serializing_if = "ManaPayment::is_manual")]
        mana_payment: ManaPayment,
        #[serde(default, skip_serializing_if = "DecklistDisclosure::is_open")]
        disclosure: DecklistDisclosure,
    }

    pub(super) fn serialize<S: Serializer>(setup: &GameSetup, s: S) -> Result<S::Ok, S::Error> {
        let seats: Vec<_> = setup
            .players
            .iter()
            .enumerate()
            .map(|(i, p)| Seat {
                name: p.name.clone(),
                decklist: p.deck.main.clone(),
                sideboard: p.deck.sideboard.clone(),
                world: p.submitted.then(|| crate::world::WORLD_POOL.to_string()),
                mana_floats: setup.mana_floats[i],
                mana_payment: setup.mana_payment[i],
                disclosure: setup.decklists.resolve(0), // fixed to open or closed at reset
            })
            .collect();
        seats.serialize(s)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Arc<GameSetup>, D::Error> {
        let seats = <[Seat; 2]>::deserialize(d)?;
        if seats[0].disclosure != seats[1].disclosure {
            return Err(D::Error::custom("inconsistent seat disclosure"));
        }
        let decklists = seats[0].disclosure.into();
        let mana_floats = seats.each_ref().map(|s| s.mana_floats);
        let mana_payment = seats.each_ref().map(|s| s.mana_payment);
        let mut players = Vec::with_capacity(2);
        for seat in seats {
            if seat
                .world
                .as_deref()
                .is_some_and(|w| w != crate::world::WORLD_POOL)
            {
                return Err(D::Error::custom("unknown submitted deck world"));
            }
            players.push(PlayerConfig {
                name: seat.name,
                deck: Deck::new(seat.decklist, seat.sideboard),
                submitted: seat.world.is_some(),
            });
        }
        Ok(Arc::new(GameSetup {
            players,
            decklists,
            mana_floats,
            mana_payment,
        }))
    }
}
