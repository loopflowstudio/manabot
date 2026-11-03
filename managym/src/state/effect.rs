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
//
// A copy effect (CR 707, layer 1) is not a record: it is `CopyEffect` on
// the copy's `Card`, and its exceptions are the same `Modification`
// vocabulary, applied once into the copy's definition (its copiable values).

use std::collections::HashMap;

use super::{
    card::{CardDefId, CardDefinition, CardType, Keywords},
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
    /// Layer 4: loses the legendary supertype ("except it isn't
    /// legendary", CR 707.9b).
    Nonlegendary,
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
    /// Rules effect: can't block (CR 509.1b).
    CantBlock,
    /// Rules effect: doesn't untap during its controller's untap step
    /// (CR 502.3).
    DoesntUntap,
    /// Layer 6: loses these keywords, whatever grants them.
    LoseKeywords(Keywords),
    /// Replacement: if it would die, exile it instead.
    ExileIfWouldDie,
    /// Rules effect on a player: unspent mana of this color stays as steps
    /// and phases end (CR 106.4).
    KeepUnspentMana(Color),
    /// Rules effect on a player: no maximum hand size (CR 402.2).
    NoMaximumHandSize,
    /// Layer 2 (CR 613.1b): the record's controller controls the object;
    /// when the record ends with the object still theirs, control returns
    /// to `previous`. Only resolutions create it (`Effect::GainControl`).
    ControlChanged { previous: PlayerId },
    /// The source's triggered ability chose mode `mode` ("choose one that
    /// hasn't been chosen"); lasts as long as the object.
    ModeChosen { mode: u8 },
    /// "has all activated and triggered abilities of" the record's source
    /// definition (Koh, the Face Stealer's last chosen card), layer 6.
    HasAbilitiesOfSource,
}

impl<N> Modification<N> {
    /// The record's form, with every amount resolved.
    pub fn resolve(&self, mut amount: impl FnMut(&N) -> i32) -> Modification<i32> {
        match self {
            Self::AddSubtypes(subtypes) => Modification::AddSubtypes(subtypes.clone()),
            Self::BecomeArtifactCreature => Modification::BecomeArtifactCreature,
            Self::Nonlegendary => Modification::Nonlegendary,
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
            Self::CantBlock => Modification::CantBlock,
            Self::DoesntUntap => Modification::DoesntUntap,
            Self::LoseKeywords(keywords) => Modification::LoseKeywords(*keywords),
            Self::ExileIfWouldDie => Modification::ExileIfWouldDie,
            Self::KeepUnspentMana(color) => Modification::KeepUnspentMana(*color),
            Self::NoMaximumHandSize => Modification::NoMaximumHandSize,
            Self::ControlChanged { previous } => Modification::ControlChanged {
                previous: *previous,
            },
            Self::ModeChosen { mode } => Modification::ModeChosen { mode: *mode },
            Self::HasAbilitiesOfSource => Modification::HasAbilitiesOfSource,
        }
    }
}

impl Modification {
    /// Apply this modification to copiable values, as a copy exception
    /// (CR 707.9b). Only modifications of printed characteristics are
    /// copiable; a rules effect returns false and leaves `definition` alone
    /// (no definition authors one as an exception).
    pub fn apply_copiable(&self, definition: &mut CardDefinition) -> bool {
        match self {
            Self::AddSubtypes(subtypes) => {
                for subtype in subtypes {
                    if !definition.subtypes.contains(subtype) {
                        definition.subtypes.push(subtype.clone());
                    }
                }
            }
            Self::BecomeArtifactCreature => {
                definition.types.types.insert(CardType::Artifact);
                definition.types.types.insert(CardType::Creature);
            }
            Self::Nonlegendary => definition
                .supertypes
                .retain(|supertype| !supertype.eq_ignore_ascii_case("legendary")),
            Self::AddKeywords(keywords) => {
                definition.keywords = definition.keywords.union(keywords);
            }
            Self::SetBasePt { power, toughness } => {
                definition.power = power.or(definition.power);
                definition.toughness = toughness.or(definition.toughness);
            }
            Self::LoseAllAbilities
            | Self::Firebending(_)
            | Self::ModifyPt { .. }
            | Self::CantBeBlocked
            | Self::CantBlock
            | Self::DoesntUntap
            | Self::LoseKeywords(_)
            | Self::ExileIfWouldDie
            | Self::KeepUnspentMana(_)
            | Self::NoMaximumHandSize
            | Self::ControlChanged { .. }
            | Self::ModeChosen { .. }
            | Self::HasAbilitiesOfSource => return false,
        }
        true
    }
}

/// A copy effect (CR 707.2) on the object it makes a copy: what it copies
/// and the exceptions that are part of its copiable values. `Card`'s
/// definition is the result; this keeps the inputs for observation.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct CopyEffect {
    /// The printed definition at the root of the copy chain.
    pub of: CardDefId,
    /// Exceptions in the order they applied, a copied copy's first: a copy
    /// of a copy inherits them (CR 707.9b).
    pub exceptions: Vec<Modification>,
}

/// How long a continuous effect lasts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum EffectDuration {
    /// Ends in the cleanup step (CR 514.2).
    EndOfTurn,
    /// No stated duration: for the rest of the game, or for as long as an
    /// affected object remains the same object (CR 400.7).
    Indefinite,
    /// "For as long as you control [this]": ends once the effect's
    /// controller no longer controls its exact source object (CR 611.2b).
    WhileYouControlSource,
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
    /// The exact source object a `WhileYouControlSource` duration follows.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_object: Option<ObjectRef>,
}

impl ContinuousEffect {
    pub fn affects_object(&self, object: ObjectRef) -> bool {
        self.affected == Affected::Object(object)
    }

    pub fn affects_player(&self, player: PlayerId) -> bool {
        self.affected == Affected::Player(player)
    }

    /// Whether the record's timestamp is later than `after` (no bound:
    /// always), so an earlier ability loss or characteristic-setting does
    /// not remove it (CR 613.7).
    pub fn follows(&self, after: Option<ObjectId>) -> bool {
        after.is_none_or(|after| self.timestamp > after)
    }
}

/// Every continuous-effect record in timestamp order, indexed by whom it
/// affects so a permanent's layers read only its own records. The canonical
/// state hash serializes the ordered records alone (`as_slice`); the index
/// is derived and rebuilt when records end.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EffectRecords {
    records: Vec<ContinuousEffect>,
    /// Indices into `records` by affected object, ascending.
    by_object: HashMap<ObjectRef, Vec<usize>>,
    /// Indices into `records` by affected player, ascending.
    by_player: [Vec<usize>; 2],
}

impl EffectRecords {
    pub fn push(&mut self, record: ContinuousEffect) {
        let index = self.records.len();
        match record.affected {
            Affected::Object(object) => self.by_object.entry(object).or_default().push(index),
            Affected::Player(player) => self.by_player[player.0].push(index),
        }
        self.records.push(record);
    }

    pub fn extend(&mut self, records: impl IntoIterator<Item = ContinuousEffect>) {
        for record in records {
            self.push(record);
        }
    }

    /// Records affecting `object`, in timestamp order.
    pub fn on(&self, object: ObjectRef) -> impl Iterator<Item = &ContinuousEffect> {
        self.by_object
            .get(&object)
            .into_iter()
            .flat_map(|indices| indices.iter().map(|&index| &self.records[index]))
    }

    /// Records affecting `player`, in timestamp order.
    pub fn on_player(&self, player: PlayerId) -> impl Iterator<Item = &ContinuousEffect> {
        self.by_player[player.0]
            .iter()
            .map(|&index| &self.records[index])
    }

    pub fn iter(&self) -> std::slice::Iter<'_, ContinuousEffect> {
        self.records.iter()
    }

    pub fn as_slice(&self) -> &[ContinuousEffect] {
        &self.records
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Keep the records `keep` accepts, in order, and rebuild the index.
    pub fn retain(&mut self, mut keep: impl FnMut(&ContinuousEffect) -> bool) {
        let records = std::mem::take(&mut self.records);
        self.by_object.clear();
        self.by_player = Default::default();
        self.extend(records.into_iter().filter(|record| keep(record)));
    }
}
