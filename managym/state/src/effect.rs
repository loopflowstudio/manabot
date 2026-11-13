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
// the copy's `Object`, and its exceptions are the same `Modification`
// vocabulary, applied once into the copy's definition (its copiable values).

use std::collections::BTreeMap;

use super::{
    ability::Ability,
    card::{
        ActivatedAbilityDefinition, CardDefId, CardType, Keywords, PrintedCharacteristics,
        StaticAbility,
    },
    game_object::{ObjectRef, PlayerId, PublicId},
    mana::{Color, Colors},
};

/// What a continuous effect changes. `N` is the type of its P/T amounts:
/// `EffectValue` in a card definition, `i32` once resolved into a record.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Modification<N = i32> {
    /// Replace creature subtypes, retaining land/artifact/enchantment types.
    SetCreatureTypes(Vec<String>),
    /// Add card types without replacing existing ones.
    AddTypes(Vec<CardType>),
    /// Replace card types and subtypes ("is a land ...").
    SetTypes {
        types: Vec<CardType>,
        subtypes: Vec<String>,
    },
    /// Layer 5 color replacement or addition.
    SetColors(Colors),
    AddColors(Colors),
    /// Layer 6 abilities granted by a continuous effect.
    GrantActivated(Box<ActivatedAbilityDefinition>),
    GrantTriggered(Box<Ability>),
    GrantStatic(Box<StaticAbility>),
    /// Layer 7d, after setters, modifiers and counters.
    SwitchPt,
    /// Layer 4: adds creature subtypes ("becomes an Ally in addition to its
    /// other types").
    AddSubtypes(Vec<String>),
    /// Layer 4: becomes a creature in addition to its other types. A crewed
    /// Vehicle "becomes an artifact creature" (CR 702.122a) with its printed
    /// P/T, already an artifact; an earthbent land "becomes a 0/0 creature
    /// ... that's still a land", with its base P/T and haste as records of
    /// the same timestamp.
    BecomeCreature,
    /// Layer 4: loses the legendary supertype ("except it isn't
    /// legendary", CR 707.9b).
    Nonlegendary,
    /// Layer 6: gains keywords.
    AddKeywords(Keywords),
    /// Layer 6: loses all abilities (CR 613.1f).
    LoseAllAbilities,
    /// Layer 6: retain mana abilities (Blood Sun).
    LoseNonManaAbilities,
    /// Layer 6: gains firebending N; grants add up.
    Firebending(u8),
    /// Layer 7b: sets base power and/or toughness; an absent value leaves
    /// that base alone.
    SetBasePt {
        power: Option<N>,
        toughness: Option<N>,
    },
    /// Layer 7c: gets +power/+toughness.
    ModifyPt {
        power: N,
        toughness: N,
    },
    /// Rules effect: can't be blocked.
    CantBeBlocked,
    /// Rules effect: can't block (CR 509.1b).
    CantBlock,
    /// Rules effect: doesn't untap during its controller's untap step
    /// (CR 502.3).
    DoesntUntap,
    /// Rules effect: its activated abilities, mana abilities included, can't
    /// be activated (CR 602.5).
    CantActivateAbilities,
    /// Rules effect: every creature able to block it does so (CR 509.1c).
    MustBeBlockedByAllAble,
    /// Layer 6: loses these keywords, whatever grants them.
    LoseKeywords(Keywords),
    /// Replacement: if it would die, exile it instead.
    ExileIfWouldDie,
    /// Replacement: if it would leave the battlefield, exile it instead of
    /// putting it anywhere else (unearth, CR 702.84a).
    ExileIfWouldLeave,
    /// Rules effect on a player: unspent mana of this color stays as steps
    /// and phases end (CR 106.4).
    KeepUnspentMana(Color),
    /// Rules effect on a player: no maximum hand size (CR 402.2).
    NoMaximumHandSize,
    /// Rules effect on a player: the next creature card they cast may be
    /// cast as though it had flash (Scout's Warning). Casting any creature
    /// card ends every such record, whether or not it needed the timing.
    NextCreatureCastFlash,
    /// Rules effect on a player: they skip every combat phase of their next
    /// turn, additional ones included (CR 500.11). It ends in the cleanup step
    /// of a turn that player takes.
    SkipsCombatPhasesNextTurn,
    /// Rules effect on a player: their life total can't change (CR 119.7):
    /// life gain, life loss and damage leave it alone, and they can pay no
    /// life other than 0 (CR 119.8).
    LifeTotalCantChange,
    /// Rules effect on a player: protection from everything (CR 702.16j).
    /// Nothing can target them, and all damage to them is prevented.
    ProtectionFromEverything,
    /// Rules effect: the permanent is phased out (CR 702.26). It stays on
    /// the battlefield with its counters and attachments, but the game
    /// treats it as though it doesn't exist; `Game::is_phased_out` is the
    /// one reader. It phases in, ending the record, during its controller's
    /// untap step before they untap (CR 702.26a); a permanent phased out
    /// indirectly with the one it is attached to phases in with it.
    PhasedOut,
    /// Layer 2 (CR 613.1b): the record's controller controls the object;
    /// when the record ends with the object still theirs, control returns
    /// to `previous`. Only resolutions create it (`Effect::GainControl`).
    ControlChanged {
        previous: PlayerId,
    },
    /// The source's triggered ability chose mode `mode` ("choose one that
    /// hasn't been chosen"); lasts as long as the object.
    ModeChosen {
        mode: u8,
    },
    /// "has all activated and triggered abilities of" the record's source
    /// definition (Koh, the Face Stealer's last chosen card), layer 6.
    HasAbilitiesOfSource,
}

impl<N> Modification<N> {
    /// The record's form, with every amount resolved.
    pub fn resolve(&self, amount: impl FnMut(&N) -> i32) -> Modification<i32> {
        self.map_amount(amount)
    }

    pub fn map_amount<M>(&self, mut amount: impl FnMut(&N) -> M) -> Modification<M> {
        match self {
            Self::SetCreatureTypes(types) => Modification::SetCreatureTypes(types.clone()),
            Self::AddTypes(types) => Modification::AddTypes(types.clone()),
            Self::SetTypes { types, subtypes } => Modification::SetTypes {
                types: types.clone(),
                subtypes: subtypes.clone(),
            },
            Self::SetColors(colors) => Modification::SetColors(colors.clone()),
            Self::AddColors(colors) => Modification::AddColors(colors.clone()),
            Self::GrantActivated(ability) => Modification::GrantActivated(ability.clone()),
            Self::GrantTriggered(ability) => Modification::GrantTriggered(ability.clone()),
            Self::GrantStatic(ability) => Modification::GrantStatic(ability.clone()),
            Self::SwitchPt => Modification::SwitchPt,
            Self::AddSubtypes(subtypes) => Modification::AddSubtypes(subtypes.clone()),
            Self::BecomeCreature => Modification::BecomeCreature,
            Self::Nonlegendary => Modification::Nonlegendary,
            Self::AddKeywords(keywords) => Modification::AddKeywords(*keywords),
            Self::LoseAllAbilities => Modification::LoseAllAbilities,
            Self::LoseNonManaAbilities => Modification::LoseNonManaAbilities,
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
            Self::CantActivateAbilities => Modification::CantActivateAbilities,
            Self::MustBeBlockedByAllAble => Modification::MustBeBlockedByAllAble,
            Self::LoseKeywords(keywords) => Modification::LoseKeywords(*keywords),
            Self::ExileIfWouldDie => Modification::ExileIfWouldDie,
            Self::ExileIfWouldLeave => Modification::ExileIfWouldLeave,
            Self::KeepUnspentMana(color) => Modification::KeepUnspentMana(*color),
            Self::NoMaximumHandSize => Modification::NoMaximumHandSize,
            Self::NextCreatureCastFlash => Modification::NextCreatureCastFlash,
            Self::SkipsCombatPhasesNextTurn => Modification::SkipsCombatPhasesNextTurn,
            Self::LifeTotalCantChange => Modification::LifeTotalCantChange,
            Self::ProtectionFromEverything => Modification::ProtectionFromEverything,
            Self::PhasedOut => Modification::PhasedOut,
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
    pub fn apply_copiable(&self, definition: &mut PrintedCharacteristics) -> bool {
        match self {
            Self::LoseNonManaAbilities
            | Self::SetCreatureTypes(_)
            | Self::AddTypes(_)
            | Self::SetTypes { .. }
            | Self::SetColors(_)
            | Self::AddColors(_)
            | Self::GrantActivated(_)
            | Self::GrantTriggered(_)
            | Self::SwitchPt => return false,
            Self::AddSubtypes(subtypes) => {
                for subtype in subtypes {
                    if !definition.subtypes.contains(subtype) {
                        definition.subtypes.push(subtype.clone());
                    }
                }
            }
            Self::BecomeCreature => {
                definition.types.types.insert(CardType::Creature);
            }
            // "Except it has [this card's] other abilities" (CR 707.9b).
            Self::GrantStatic(ability) => definition.statics.push((**ability).clone()),
            Self::Nonlegendary => definition
                .supertypes
                .retain(|supertype| !supertype.eq_ignore_ascii_case("legendary")),
            Self::AddKeywords(keywords) => {
                definition.keywords = definition.keywords.union(*keywords);
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
            | Self::CantActivateAbilities
            | Self::MustBeBlockedByAllAble
            | Self::LoseKeywords(_)
            | Self::ExileIfWouldDie
            | Self::ExileIfWouldLeave
            | Self::KeepUnspentMana(_)
            | Self::NoMaximumHandSize
            | Self::NextCreatureCastFlash
            | Self::SkipsCombatPhasesNextTurn
            | Self::LifeTotalCantChange
            | Self::ProtectionFromEverything
            | Self::PhasedOut
            | Self::ControlChanged { .. }
            | Self::ModeChosen { .. }
            | Self::HasAbilitiesOfSource => return false,
        }
        true
    }
}

/// A copy effect (CR 707.2) on the object it makes a copy: what it copies
/// and the exceptions that are part of its copiable values. `Object`'s
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
    /// "Until your next turn": ends as the record controller's next turn
    /// begins, before that turn's untap step phases anything in (CR 502.1).
    UntilYourNextTurn,
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
    pub timestamp: PublicId,
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
}

/// Every continuous-effect record in timestamp order, indexed by whom it
/// affects so a permanent's layers read only its own records. The canonical
/// state hash serializes the ordered records alone (`as_slice`); the index
/// is derived and rebuilt when records end.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EffectRecords {
    records: Vec<ContinuousEffect>,
    /// Indices into `records` by affected object, ascending.
    by_object: BTreeMap<ObjectRef, Vec<usize>>,
    /// Indices into `records` by affected player, ascending.
    by_player: [Vec<usize>; 2],
    /// How many records are `PhasedOut`, so battlefield readers skip the
    /// per-object lookup while nothing is phased out.
    phased_out: usize,
}

impl EffectRecords {
    pub fn push(&mut self, record: ContinuousEffect) {
        let index = self.records.len();
        match record.affected {
            Affected::Object(object) => self.by_object.entry(object).or_default().push(index),
            Affected::Player(player) => self.by_player[player.0].push(index),
        }
        self.phased_out += usize::from(record.modification == Modification::PhasedOut);
        self.records.push(record);
    }

    pub fn extend(&mut self, records: impl IntoIterator<Item = ContinuousEffect>) {
        for record in records {
            self.push(record);
        }
    }

    /// Records affecting `object`, in timestamp order.
    #[inline]
    pub fn on(&self, object: ObjectRef) -> impl Iterator<Item = &ContinuousEffect> {
        self.by_object
            .get(&object)
            .into_iter()
            .flat_map(|indices| indices.iter().map(|&index| &self.records[index]))
    }

    /// Records affecting `player`, in timestamp order.
    #[inline]
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

    /// Whether any record phases a permanent out.
    #[inline]
    pub fn any_phased_out(&self) -> bool {
        self.phased_out > 0
    }

    /// Keep the records `keep` accepts, in order, and rebuild the index.
    pub fn retain(&mut self, mut keep: impl FnMut(&ContinuousEffect) -> bool) {
        let records = std::mem::take(&mut self.records);
        self.by_object.clear();
        self.by_player = Default::default();
        self.phased_out = 0;
        self.extend(records.into_iter().filter(|record| keep(record)));
    }
}
