// payment.rs
// What one cast or activation actually paid (CR 601.2h). `PaymentReceipt` is
// the record a spell on the stack, the permanent it becomes and the frames
// that resolve it share; `ManaOrigin` names the object that produced a unit
// of mana, from the pool (`mana::ManaPool`) through to the receipt.

use std::sync::Arc;

use crate::{
    card::CardDefId,
    game_object::{exact, ObjectId},
    mana::{Color, SpendEffect},
};

/// The object whose ability produced a unit of mana: the appearance that was
/// on the battlefield (or was the resolving source) as the mana was made, and
/// what it was defined as then. The appearance may since have left; read what
/// it was from last known information, never from a later appearance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub struct ManaOrigin {
    #[serde(serialize_with = "exact")]
    pub object: ObjectId,
    pub definition: CardDefId,
}

/// One unit of mana as it was spent.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub struct SpentMana {
    /// W/U/B/R/G/C, as `Mana.mana` indexes it.
    pub color: u8,
    /// `None` only for mana no object produced.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<ManaOrigin>,
    /// The unit's spend rider, when it applied to this payment (CR 106.6).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effect: Option<SpendEffect>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, serde::Serialize)]
struct Paid {
    mana: Vec<SpentMana>,
}

static NOTHING: Paid = Paid { mana: Vec::new() };

/// What one payment spent, one entry per unit of mana in spend order. A cast
/// spell keeps it for "mana spent to cast", converge and spend riders; it
/// never reads the printed cost.
///
/// Its holders share one record: a clone is a reference, and an empty receipt
/// allocates nothing. Only the payment in progress appends to it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct PaymentReceipt(Option<Arc<Paid>>);

impl PaymentReceipt {
    fn paid(&self) -> &Paid {
        self.0.as_deref().unwrap_or(&NOTHING)
    }

    pub fn is_empty(&self) -> bool {
        self.paid().mana.is_empty()
    }

    /// The units spent, in spend order.
    pub fn mana(&self) -> &[SpentMana] {
        &self.paid().mana
    }

    /// Append the units one spending choice committed.
    pub fn record_mana(&mut self, units: impl IntoIterator<Item = SpentMana>) {
        let mut units = units.into_iter().peekable();
        if units.peek().is_some() {
            Arc::make_mut(self.0.get_or_insert_with(Default::default))
                .mana
                .extend(units);
        }
    }

    /// "The amount of mana spent to cast" it.
    pub fn total(&self) -> u16 {
        self.mana().len() as u16
    }

    /// Amounts by W/U/B/R/G/C.
    pub fn amounts(&self) -> [u8; 6] {
        let mut amounts = [0u8; 6];
        for unit in self.mana() {
            amounts[unit.color as usize] += 1;
        }
        amounts
    }

    /// Converge (an ability word, CR 207.2c): the number of colors of mana
    /// spent; colorless is not a color (CR 105.1).
    pub fn colors(&self) -> u8 {
        self.amounts()[..Color::Colorless as usize]
            .iter()
            .filter(|amount| **amount > 0)
            .count() as u8
    }

    /// +1/+1 counters the spell's permanent enters with: one for each unit
    /// whose `CreatureEntersWithPlusOneCounter` rider applied.
    pub fn entry_counters(&self) -> u8 {
        self.mana()
            .iter()
            .filter(|unit| {
                matches!(
                    unit.effect,
                    Some(SpendEffect::CreatureEntersWithPlusOneCounter { .. })
                )
            })
            .count() as u8
    }

    /// The objects whose mana paid, in first-spent order, each once.
    pub fn mana_sources(&self) -> Vec<ManaOrigin> {
        let mut sources = Vec::new();
        for origin in self.mana().iter().filter_map(|unit| unit.source) {
            if !sources.contains(&origin) {
                sources.push(origin);
            }
        }
        sources
    }
}

impl serde::Serialize for PaymentReceipt {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.paid().serialize(serializer)
    }
}
