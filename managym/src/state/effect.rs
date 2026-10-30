// effect.rs
// Continuous effects from resolving spells and abilities (CR 611), as
// records on `GameState::effects`.
//
// A card applies one by composing a closed vocabulary: what it modifies
// (`Modification`, grouped by CR 613 layer), whom (`Affected`) and how long
// (`EffectDuration`). Adding cards adds records, never engine fields; a new
// mechanic adds a variant here, and the exhaustive observability guard then
// requires its tensor, token and LLM-text encoding.
//
// Static abilities are not recorded: they stay derived from the permanents
// that have them and join the same layer application with the permanent's
// own timestamp (`Permanent::timestamp`, CR 613.7a/d/e). `flow/statics.rs`
// applies both by layer, then by timestamp. Expiry is one function per
// boundary in `flow/effects.rs`.

use super::{
    card::{CardDefId, Keywords},
    game_object::{ObjectId, ObjectRef, PlayerId},
    mana::Color,
};

/// What a continuous effect changes. `N` is the type of its P/T amounts:
/// `EffectValue` in a card definition, `i32` once resolved into a record.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Modification<N = i32> {
    /// Layer 4: adds creature subtypes ("becomes an Ally in addition to its
    /// other types").
    AddSubtypes(Vec<String>),
    /// Layer 4: becomes an artifact creature; its printed P/T apply
    /// (CR 301.7b; crew, CR 702.122a).
    BecomeArtifactCreature,
    /// Layer 6: gains keywords.
    AddKeywords(Keywords),
    /// Layer 6: loses all abilities (CR 613.1f).
    LoseAllAbilities,
    /// Layer 6: gains firebending N; grants add up.
    Firebending(u8),
    /// Layer 7b: sets base power and/or toughness; an absent value leaves
    /// that base alone.
    SetBasePt {
        power: Option<N>,
        toughness: Option<N>,
    },
    /// Layer 7c: gets +power/+toughness.
    ModifyPt { power: N, toughness: N },
    /// Rules effect: can't be blocked.
    CantBeBlocked,
    /// Replacement: if it would die, exile it instead.
    ExileIfWouldDie,
    /// Rules effect on a player: unspent mana of this color stays as steps
    /// and phases end (CR 106.4).
    KeepUnspentMana(Color),
    /// Rules effect on a player: no maximum hand size (CR 402.2).
    NoMaximumHandSize,
}

impl<N> Modification<N> {
    /// The record's form, with every amount resolved.
    pub fn resolve(&self, mut amount: impl FnMut(&N) -> i32) -> Modification<i32> {
        match self {
            Self::AddSubtypes(subtypes) => Modification::AddSubtypes(subtypes.clone()),
            Self::BecomeArtifactCreature => Modification::BecomeArtifactCreature,
            Self::AddKeywords(keywords) => Modification::AddKeywords(*keywords),
            Self::LoseAllAbilities => Modification::LoseAllAbilities,
            Self::Firebending(amount) => Modification::Firebending(*amount),
            Self::SetBasePt { power, toughness } => Modification::SetBasePt {
                power: power.as_ref().map(&mut amount),
                toughness: toughness.as_ref().map(&mut amount),
            },
            Self::ModifyPt { power, toughness } => Modification::ModifyPt {
                power: amount(power),
                toughness: amount(toughness),
            },
            Self::CantBeBlocked => Modification::CantBeBlocked,
            Self::ExileIfWouldDie => Modification::ExileIfWouldDie,
            Self::KeepUnspentMana(color) => Modification::KeepUnspentMana(*color),
            Self::NoMaximumHandSize => Modification::NoMaximumHandSize,
        }
    }
}

/// How long a continuous effect lasts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum EffectDuration {
    /// Ends in the cleanup step (CR 514.2).
    EndOfTurn,
    /// No stated duration: for the rest of the game, or for as long as an
    /// affected object remains the same object (CR 400.7).
    Indefinite,
}

/// Whom a continuous effect record affects.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Affected {
    Player(PlayerId),
    /// One battlefield incarnation; the record ends when it changes zones
    /// (CR 400.7). An effect affecting a set fixed at resolution (CR 611.2c)
    /// is one record per object, sharing a timestamp.
    Object(ObjectRef),
}

/// One continuous effect from a resolved spell or ability.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ContinuousEffect {
    /// From the game's object-id clock, shared with permanents' timestamps
    /// (CR 613.7).
    pub timestamp: ObjectId,
    /// The definition of the spell or ability's source, kept after the
    /// source leaves (last-known information).
    pub source: Option<CardDefId>,
    pub controller: PlayerId,
    pub affected: Affected,
    pub modification: Modification,
    pub duration: EffectDuration,
}

impl ContinuousEffect {
    pub fn affects_object(&self, object: ObjectRef) -> bool {
        self.affected == Affected::Object(object)
    }

    pub fn affects_player(&self, player: PlayerId) -> bool {
        self.affected == Affected::Player(player)
    }
}
