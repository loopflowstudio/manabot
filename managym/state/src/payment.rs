// payment.rs
// What one cast or activation actually paid (CR 601.2h). `PaymentReceipt` is
// the record a spell on the stack, the permanent it becomes and the frames
// that resolve it share; `ManaOrigin` names the object that produced a unit
// of mana, from the pool (`mana::ManaPool`) through to the receipt;
// `Contribution` names an object that paid a cost with something other than
// mana.

use std::sync::Arc;

use crate::{
    card::{CardDefId, Keyword, Keywords},
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

/// What was done to an object to pay with it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum How {
    Tapped,
    Sacrificed,
    Exiled,
    Discarded,
}

/// The part of the cost an object paid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub enum Toward {
    /// The spell's or ability's own stated cost.
    Printed,
    /// {1} of a waterbend cost.
    Waterbend,
    /// A crew cost (CR 702.122a).
    Crew,
    /// A saddle cost (CR 702.171a).
    Saddle,
    /// A cost paid rather than the mana cost (CR 118.9).
    Alternative,
    /// {1} or one mana of the tapped creature's color (CR 702.51a).
    Convoke,
    /// {1} for the tapped artifact (CR 702.126a).
    Improvise,
    /// {1} for the card exiled from a graveyard (CR 702.66a).
    Delve,
    /// The permanent sacrificed to bargain (CR 702.166a).
    Bargain,
    /// The Food sacrificed or the cards exiled to forage (CR 701.61a).
    Forage,
    /// The creature tapped to station (CR 702.184a).
    Station,
}

/// An object that paid part of a cost instead of, or beside, mana. `object`
/// is the exact appearance that paid: the permanent tapped or sacrificed, the
/// card in hand that was discarded or exiled. It may since have left; read
/// what it was from last known information, never from a later appearance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub struct Contribution {
    #[serde(serialize_with = "exact")]
    pub object: ObjectId,
    pub how: How,
    pub toward: Toward,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, serde::Serialize)]
struct Paid {
    mana: Vec<SpentMana>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    contributions: Vec<Contribution>,
}

static NOTHING: Paid = Paid {
    mana: Vec::new(),
    contributions: Vec::new(),
};

/// What one payment spent: one entry per unit of mana in spend order, and the
/// objects that paid otherwise in the order they were chosen. A cast spell
/// keeps it for "mana spent to cast", converge and spend riders; it never
/// reads the printed cost. Contributions are a separate list, so mana spent
/// never counts a tapped or sacrificed object.
///
/// Its holders share one record: a clone is a reference, and an empty receipt
/// allocates nothing. Only the payment in progress appends to it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct PaymentReceipt(Option<Arc<Paid>>);

impl PaymentReceipt {
    fn paid(&self) -> &Paid {
        self.0.as_deref().unwrap_or(&NOTHING)
    }

    /// Nothing was paid: no mana and no contributing object.
    pub fn is_empty(&self) -> bool {
        self.0.is_none()
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

    /// The objects that paid other than with mana, in the order chosen.
    pub fn contributions(&self) -> &[Contribution] {
        &self.paid().contributions
    }

    /// The objects that paid `toward` by being `how`, in the order chosen.
    pub fn contributed(&self, how: How, toward: Toward) -> impl Iterator<Item = ObjectId> + '_ {
        self.contributions()
            .iter()
            .filter(move |paid| paid.how == how && paid.toward == toward)
            .map(|paid| paid.object)
    }

    /// The permanents this payment sacrifices as it completes.
    pub fn sacrificed(&self) -> impl Iterator<Item = ObjectId> + '_ {
        self.contributions()
            .iter()
            .filter(|paid| paid.how == How::Sacrificed)
            .map(|paid| paid.object)
    }

    /// Record the object one payment choice committed.
    pub fn contribute(&mut self, object: ObjectId, how: How, toward: Toward) {
        Arc::make_mut(self.0.get_or_insert_with(Default::default))
            .contributions
            .push(Contribution {
                object,
                how,
                toward,
            });
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

    /// Keywords the spell gained from the mana spent on it, which the
    /// permanent it becomes keeps until end of turn (`SpellGainsKeyword`).
    pub fn keywords_gained(&self) -> Keywords {
        let gained: Vec<Keyword> = self
            .mana()
            .iter()
            .filter_map(|unit| match unit.effect {
                Some(SpendEffect::SpellGainsKeyword { keyword, .. }) => Some(keyword),
                _ => None,
            })
            .collect();
        Keywords::of(&gained)
    }

    /// The origin of each unit whose `CopySpell` rider applied: one copy
    /// trigger apiece, controlled by the caster, from that object.
    pub fn copy_triggers(&self) -> impl Iterator<Item = Option<ManaOrigin>> + '_ {
        self.mana()
            .iter()
            .filter(|unit| matches!(unit.effect, Some(SpendEffect::CopySpell { .. })))
            .map(|unit| unit.source)
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
