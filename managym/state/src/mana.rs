// mana.rs
// Mana costs, pooled classes and exact unit spending. SpendingContext checks
// restrictions; ManaPool keeps, unit for unit, which object produced a
// player's mana, so the receipt of a later payment (`payment`) can name it.

use std::{collections::BTreeSet, sync::Arc};

use crate::{
    card::{CardType, Keyword, PrintedCharacteristics},
    game_object::ObjectKind,
    payment::{ManaOrigin, SpentMana},
    predicate::CardPredicate,
    zone::ZoneType,
};

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Blue = 1,
    Black = 2,
    Red = 3,
    Green = 4,
    Colorless = 5,
    Generic = 6,
}

impl Color {
    pub fn symbol(self) -> &'static str {
        match self {
            Color::White => "W",
            Color::Blue => "U",
            Color::Black => "B",
            Color::Red => "R",
            Color::Green => "G",
            Color::Colorless => "C",
            Color::Generic => "*",
        }
    }
}

pub type Colors = BTreeSet<Color>;

/// One purpose mana may be spent on (CR 106.6), as a restriction names it.
/// Subtypes are printed names from any vocabulary; none is privileged.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub enum SpendPredicate {
    /// Casting a spell with this subtype.
    CastSubtype(String),
    /// Activating an ability of a source with this subtype.
    ActivateSubtype(String),
}

impl SpendPredicate {
    pub fn matches(&self, context: &SpendingContext) -> bool {
        match (self, context) {
            (Self::CastSubtype(name), SpendingContext::Cast(spell)) => {
                spell.0.subtypes.iter().any(|subtype| subtype == name)
            }
            (Self::ActivateSubtype(name), SpendingContext::Activate { source, .. }) => {
                source.has_subtype(name)
            }
            _ => false,
        }
    }
}

/// The spending restriction that mana keeps until it is spent or lost
/// (CR 106.6). A [`ManaClass`] can carry this restriction independently of its
/// spend effect and expiry; ordinary mana lives in [`Mana::mana`].
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub enum ManaRestriction {
    /// Spend only on a purpose matching one of these ("spend this mana only
    /// to cast a Lesson spell"). Resolving costs match none of them.
    Only(BTreeSet<SpendPredicate>),
    /// Spend on anything but casting a nonartifact spell (Powerstone),
    /// resolving costs included.
    ExceptNonartifactSpells,
}

impl ManaRestriction {
    /// Spend only to cast spells with `spell_subtypes` or to activate
    /// abilities of sources with `ability_source_subtypes`.
    pub fn only<'a>(
        spell_subtypes: impl IntoIterator<Item = &'a str>,
        ability_source_subtypes: impl IntoIterator<Item = &'a str>,
    ) -> Self {
        let predicates = spell_subtypes
            .into_iter()
            .map(|name| SpendPredicate::CastSubtype(name.into()))
            .chain(
                ability_source_subtypes
                    .into_iter()
                    .map(|name| SpendPredicate::ActivateSubtype(name.into())),
            )
            .collect::<BTreeSet<_>>();
        assert!(!predicates.is_empty(), "a restriction names a purpose");
        Self::Only(predicates)
    }

    /// Spend only to cast spells with one of these subtypes.
    pub fn spell<'a>(subtypes: impl IntoIterator<Item = &'a str>) -> Self {
        Self::only(subtypes, [])
    }

    /// Whether mana with this restriction may pay a cost spent in `context`.
    pub fn allows(&self, context: &SpendingContext) -> bool {
        match self {
            Self::Only(predicates) => predicates.iter().any(|p| p.matches(context)),
            Self::ExceptNonartifactSpells => !matches!(
                context,
                SpendingContext::Cast(spell) if !spell.0.types.is_artifact()
            ),
        }
    }

    /// The subtypes a spell may have to be cast with this mana, sorted.
    pub fn spell_subtypes(&self) -> impl Iterator<Item = &str> {
        self.predicates().filter_map(|predicate| match predicate {
            SpendPredicate::CastSubtype(name) => Some(name.as_str()),
            SpendPredicate::ActivateSubtype(_) => None,
        })
    }

    /// The subtypes a source may have for this mana to activate its
    /// abilities, sorted.
    pub fn source_subtypes(&self) -> impl Iterator<Item = &str> {
        self.predicates().filter_map(|predicate| match predicate {
            SpendPredicate::ActivateSubtype(name) => Some(name.as_str()),
            SpendPredicate::CastSubtype(_) => None,
        })
    }

    fn predicates(&self) -> impl Iterator<Item = &SpendPredicate> {
        match self {
            Self::Only(predicates) => Some(predicates),
            Self::ExceptNonartifactSpells => None,
        }
        .into_iter()
        .flatten()
    }

    /// "Lesson or Shrine spells", "Ally spells or abilities of Ally sources",
    /// "anything but nonartifact spells".
    pub fn describe(&self) -> String {
        if *self == Self::ExceptNonartifactSpells {
            return "anything but nonartifact spells".into();
        }
        let mut parts = Vec::new();
        let spells = self.spell_subtypes().collect::<Vec<_>>();
        if !spells.is_empty() {
            parts.push(format!("{} spells", spells.join(" or ")));
        }
        let sources = self.source_subtypes().collect::<Vec<_>>();
        if !sources.is_empty() {
            parts.push(format!("abilities of {} sources", sources.join(" or ")));
        }
        parts.join(" or ")
    }
}

/// What spending one mana does beyond paying (CR 106.6): the rider a mana
/// ability attaches to the mana it produces.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub enum SpendEffect {
    /// "If that mana is spent to cast a non-[subtype] creature spell, that
    /// creature enters with an additional +1/+1 counter on it" (Animal
    /// Attendant). One counter for each such mana spent.
    CreatureEntersWithPlusOneCounter { without_subtype: String },
    /// "If that mana is spent on a [spell], it gains [keyword] until end of
    /// turn" (Carnelian Orb of Dragonkind). The permanent the spell becomes
    /// keeps the keyword for the turn.
    SpellGainsKeyword {
        spell: Arc<CardPredicate>,
        keyword: Keyword,
    },
    /// "When that mana is spent to cast a [spell], copy that spell and you
    /// may choose new targets for the copy" (Pyromancer's Goggles): a
    /// triggered ability of the object that produced the mana, one for each
    /// such unit spent.
    CopySpell { spell: Arc<CardPredicate> },
}

impl SpendEffect {
    /// Whether spending this mana in `context` has its effect.
    pub fn applies(&self, context: &SpendingContext) -> bool {
        let SpendingContext::Cast(spell) = context else {
            return false;
        };
        let spell = &*spell.0;
        match self {
            Self::CreatureEntersWithPlusOneCounter { without_subtype } => {
                spell.types.is_creature()
                    && !spell.subtypes.iter().any(|name| name == without_subtype)
            }
            Self::SpellGainsKeyword {
                spell: predicate, ..
            }
            | Self::CopySpell { spell: predicate } => {
                predicate.matches_printed_spell(spell, ObjectKind::Card)
            }
        }
    }
}

/// Every rule-visible property that tells pooled mana apart from ordinary
/// mana of its type: a spending restriction, a spend effect, and combat expiry
/// in the combined pool. Mana merges only within one class; its units are
/// indistinguishable.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub struct ManaClass {
    /// Until-end-of-combat mana remains distinct during payment search.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub combat: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub restriction: Option<ManaRestriction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effect: Option<SpendEffect>,
}

/// Where the automatic payer spends a pooled unit among those that can pay
/// the same pip, lowest first: mana that only some costs can use (narrowest
/// first) or whose rider applies to this cost, then ordinary mana, and last
/// mana whose rider this cost would waste.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SpendOrder {
    /// The value counts the purposes a restriction allows; mana restricted
    /// some other way, or unrestricted with a rider that applies, is widest.
    Specific(usize),
    Ordinary,
    WastedRider,
}

impl From<ManaRestriction> for ManaClass {
    fn from(restriction: ManaRestriction) -> Self {
        Self {
            combat: false,
            restriction: Some(restriction),
            effect: None,
        }
    }
}

impl From<SpendEffect> for ManaClass {
    fn from(effect: SpendEffect) -> Self {
        Self {
            combat: false,
            restriction: None,
            effect: Some(effect),
        }
    }
}

impl ManaClass {
    /// Whether mana of this class may pay a cost spent in `context`.
    pub fn allows(&self, context: &SpendingContext) -> bool {
        self.restriction
            .as_ref()
            .is_none_or(|restriction| restriction.allows(context))
    }

    /// This class's place in the automatic payer's order for a cost spent
    /// in `context`. Until-end-of-combat mana has no place of its own.
    pub fn spend_order(&self, context: &SpendingContext) -> SpendOrder {
        if self
            .effect
            .as_ref()
            .is_some_and(|effect| !effect.applies(context))
        {
            return SpendOrder::WastedRider;
        }
        match (&self.restriction, &self.effect) {
            (Some(ManaRestriction::Only(purposes)), _) => SpendOrder::Specific(purposes.len()),
            (Some(ManaRestriction::ExceptNonartifactSpells), _) | (None, Some(_)) => {
                SpendOrder::Specific(usize::MAX)
            }
            (None, None) => SpendOrder::Ordinary,
        }
    }
}

/// An activation's source as its cost is priced: the subtypes the
/// `ActivateSubtype` predicate reads. A snapshot shared by the cost's clones,
/// witnesses and forks; it never follows the live object.
#[derive(Clone, Debug, serde::Serialize)]
pub struct SpendingObject {
    /// Sorted and deduplicated.
    pub subtypes: Vec<String>,
    /// A hash of `subtypes`, computed once: payment search hashes costs in
    /// every visited state.
    #[serde(skip)]
    digest: u64,
}

impl PartialEq for SpendingObject {
    fn eq(&self, other: &Self) -> bool {
        self.digest == other.digest && self.subtypes == other.subtypes
    }
}

impl Eq for SpendingObject {}

impl std::hash::Hash for SpendingObject {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u64(self.digest);
    }
}

impl SpendingObject {
    pub fn new<'a>(subtypes: impl IntoIterator<Item = &'a str>) -> Arc<Self> {
        use std::hash::{Hash, Hasher};
        let mut subtypes = subtypes.into_iter().map(String::from).collect::<Vec<_>>();
        subtypes.sort_unstable();
        subtypes.dedup();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        subtypes.hash(&mut hasher);
        Arc::new(Self {
            subtypes,
            digest: hasher.finish(),
        })
    }

    fn has_subtype(&self, name: &str) -> bool {
        self.subtypes
            .binary_search_by(|subtype| subtype.as_str().cmp(name))
            .is_ok()
    }

    /// Whether this snapshot names exactly these subtypes (in any order, with
    /// repeats), so a rebuilt view can keep sharing it without allocating.
    pub fn describes(&self, subtypes: &[String]) -> bool {
        let distinct = subtypes
            .iter()
            .enumerate()
            .filter(|(i, name)| !subtypes[..*i].contains(name))
            .count();
        distinct == self.subtypes.len() && subtypes.iter().all(|name| self.has_subtype(name))
    }
}

/// A spell as announced (CR 601.2): its own definition, shared rather than
/// copied. A card's definition is copy-on-write, so this never follows later
/// changes. Equal only to itself, which keeps pricing allocation-free;
/// distinct snapshots of equal characteristics are different purposes, never
/// a different restriction outcome.
#[derive(Clone, Debug)]
pub struct SpellSnapshot(pub Arc<PrintedCharacteristics>);

impl PartialEq for SpellSnapshot {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for SpellSnapshot {}

impl std::hash::Hash for SpellSnapshot {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(Arc::as_ptr(&self.0), state);
    }
}

impl serde::Serialize for SpellSnapshot {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut out = serializer.serialize_struct("SpellSnapshot", 2)?;
        out.serialize_field("subtypes", &self.0.subtypes)?;
        out.serialize_field("types", &self.0.types)?;
        out.end()
    }
}

/// Why a cost is paid (CR 106.6): casting an announced spell, activating an
/// ability of a source in a zone, or a cost an effect asks for while it
/// resolves, which no casting or activation restriction admits.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, serde::Serialize)]
pub enum SpendingContext {
    #[default]
    Resolving,
    Cast(SpellSnapshot),
    Activate {
        zone: ZoneType,
        source: Arc<SpendingObject>,
    },
}

impl SpendingContext {
    /// Casting a spell with these subtypes and card types.
    pub fn cast<'a>(
        subtypes: impl IntoIterator<Item = &'a str>,
        types: impl IntoIterator<Item = CardType>,
    ) -> Self {
        Self::Cast(SpellSnapshot(Arc::new(PrintedCharacteristics {
            subtypes: subtypes.into_iter().map(String::from).collect(),
            types: crate::card::CardTypes::new(types),
            ..Default::default()
        })))
    }

    /// Activating an ability of a battlefield source with these subtypes.
    pub fn ability<'a>(subtypes: impl IntoIterator<Item = &'a str>) -> Self {
        Self::Activate {
            zone: ZoneType::Battlefield,
            source: SpendingObject::new(subtypes),
        }
    }

    pub fn is_resolving(&self) -> bool {
        matches!(self, Self::Resolving)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, serde::Serialize)]
pub struct ManaCost {
    /// The payment's purpose, which decides the restricted mana it may use:
    /// set for a cast spell or an activation, never for resolving effects.
    #[serde(skip_serializing_if = "SpendingContext::is_resolving")]
    pub purpose: SpendingContext,
    pub cost: [u8; 7],
    pub mana_value: u8,
    /// Each two-color hybrid pip is paid with exactly one of its colors.
    /// Omission keeps frozen ordinary-cost manifests byte-for-byte stable.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub hybrid: Vec<[Color; 2]>,
    /// Number of {X} symbols. X is 0 outside the stack (CR 202.3e), so it
    /// adds nothing to `mana_value` until [`ManaCost::with_x`] fixes it.
    #[serde(skip_serializing_if = "crate::is_zero")]
    pub x: u8,
}

impl ManaCost {
    /// Mana value on the stack, counting each printed {X} at its announced value.
    pub fn mana_value_at(&self, x: u8) -> u8 {
        self.mana_value.saturating_add(self.x.saturating_mul(x))
    }

    pub fn parse(text: &str) -> Self {
        let mut out = ManaCost::default();
        let mut chars = text.chars().peekable();

        while let Some(ch) = chars.peek().copied() {
            if ch.is_ascii_digit() {
                let mut value: u16 = 0;
                while let Some(digit) = chars.peek().copied() {
                    if !digit.is_ascii_digit() {
                        break;
                    }
                    value = value * 10 + (digit as u16 - b'0' as u16);
                    chars.next();
                }
                out.cost[Color::Generic as usize] = value.min(u8::MAX as u16) as u8;
                continue;
            }

            if ch == '{' {
                chars.next();
                let mut symbol = String::new();
                for next in chars.by_ref() {
                    if next == '}' {
                        break;
                    }
                    symbol.push(next);
                }
                if let Some((left, right)) = symbol.split_once('/') {
                    let color = |s: &str| match s {
                        "W" => Color::White,
                        "U" => Color::Blue,
                        "B" => Color::Black,
                        "R" => Color::Red,
                        "G" => Color::Green,
                        _ => panic!("unsupported hybrid mana symbol: {symbol}"),
                    };
                    out.hybrid.push([color(left), color(right)]);
                } else {
                    let part = Self::parse(&symbol);
                    for (slot, amount) in out.cost.iter_mut().zip(part.cost) {
                        *slot = slot.saturating_add(amount);
                    }
                    out.x = out.x.saturating_add(part.x);
                }
                continue;
            }

            if ch == 'X' {
                out.x += 1;
                chars.next();
                continue;
            }
            let slot = match ch {
                'W' => Color::White,
                'U' => Color::Blue,
                'B' => Color::Black,
                'R' => Color::Red,
                'G' => Color::Green,
                'C' => Color::Colorless,
                _ => panic!("invalid mana symbol: {ch}"),
            };
            out.cost[slot as usize] += 1;
            chars.next();
        }

        out.update_mana_value();
        out
    }

    /// Component-wise sum of two costs (base + kicker).
    pub fn plus(&self, other: &ManaCost) -> ManaCost {
        let mut out = self.clone();
        for i in 0..out.cost.len() {
            out.cost[i] = out.cost[i].saturating_add(other.cost[i]);
        }
        out.hybrid.extend_from_slice(&other.hybrid);
        out.x = out.x.saturating_add(other.x);
        out.update_mana_value();
        out
    }

    /// This cost less `reduction` (CR 118.7b, 118.7c): each colored symbol removes
    /// one of that color, or one generic when none of that color is left;
    /// generic in `reduction` removes generic.
    pub fn reduced_by(&self, reduction: &ManaCost) -> ManaCost {
        let mut out = self.clone();
        let generic = Color::Generic as usize;
        for (slot, &count) in reduction.cost.iter().enumerate() {
            let colored = if slot == generic {
                0
            } else {
                out.cost[slot].min(count)
            };
            out.cost[slot] -= colored;
            out.cost[generic] = out.cost[generic].saturating_sub(count - colored);
        }
        out.update_mana_value();
        out
    }

    /// Reduce the generic component by `amount`, floored at zero
    /// (affinity-style cost reduction, waterbend payments).
    pub fn reduced_generic(&self, amount: u8) -> ManaCost {
        self.with_generic(self.generic().saturating_sub(amount))
    }

    /// This cost with one pip of `color` paid some other way; `Color::Generic`
    /// is {1} (convoke, improvise, delve).
    pub fn less_pip(&self, color: Color) -> ManaCost {
        let mut out = self.clone();
        debug_assert!(out.cost[color as usize] > 0, "the pip is owed");
        out.cost[color as usize] = out.cost[color as usize].saturating_sub(1);
        out.update_mana_value();
        out
    }

    /// What is left of this cost once one unit of `color` (W/U/B/R/G/C, as
    /// `Mana.mana` indexes it) pays a pip of it, one entry per pip it can
    /// pay, most flexible remainder first: its own colored pip, then a
    /// hybrid pip with its color, then {1}. Every unit that could pay the
    /// pip a later entry leaves could pay the one an earlier entry leaves,
    /// except between two hybrid pips, which the caller decides.
    pub fn paid_with(&self, color: u8) -> Vec<ManaCost> {
        let mut options = Vec::new();
        if self.cost[color as usize] > 0 {
            let mut out = self.clone();
            out.cost[color as usize] -= 1;
            out.update_mana_value();
            options.push(out);
        }
        for (index, pip) in self.hybrid.iter().enumerate() {
            if pip.iter().any(|c| *c as u8 == color) && !self.hybrid[..index].contains(pip) {
                let mut out = self.clone();
                out.hybrid.remove(index);
                out.update_mana_value();
                options.push(out);
            }
        }
        if self.generic() > 0 {
            options.push(self.less_pip(Color::Generic));
        }
        options
    }

    /// The pips of this cost that `rest`, a cost it was reduced to, no
    /// longer holds: what was paid between the two.
    pub fn less(&self, rest: &ManaCost) -> ManaCost {
        let mut out = ManaCost {
            purpose: self.purpose.clone(),
            ..ManaCost::default()
        };
        for (slot, (all, left)) in out.cost.iter_mut().zip(self.cost.iter().zip(rest.cost)) {
            *slot = all.saturating_sub(left);
        }
        let mut left = rest.hybrid.clone();
        for pip in &self.hybrid {
            match left.iter().position(|kept| kept == pip) {
                Some(index) => {
                    left.remove(index);
                }
                None => out.hybrid.push(*pip),
            }
        }
        out.update_mana_value();
        out
    }

    /// This cost with X announced as `value` (CR 601.2b): each {X} becomes
    /// `value` generic mana.
    pub fn with_x(&self, value: u8) -> ManaCost {
        let mut out = self.clone();
        out.cost[Color::Generic as usize] =
            out.generic().saturating_add(self.x.saturating_mul(value));
        out.x = 0;
        out.update_mana_value();
        out
    }

    /// Preserve the spending restriction and X while allowing any mana type for every pip.
    pub fn pay_with_any_mana(self) -> Self {
        Self {
            purpose: self.purpose,
            x: self.x,
            ..Self::default().with_generic(self.mana_value)
        }
    }

    /// The generic component of the cost.
    pub fn generic(&self) -> u8 {
        self.cost[Color::Generic as usize]
    }

    /// This cost with the generic component replaced by `generic`.
    pub fn with_generic(&self, generic: u8) -> ManaCost {
        let mut out = self.clone();
        out.cost[Color::Generic as usize] = generic;
        out.update_mana_value();
        out
    }

    fn update_mana_value(&mut self) {
        self.mana_value = self
            .cost
            .iter()
            .copied()
            .sum::<u8>()
            .saturating_add(self.hybrid.len() as u8);
    }

    pub fn colors(&self) -> Colors {
        let mut out = self.hybrid.iter().flatten().copied().collect::<Colors>();
        for color in [
            Color::White,
            Color::Blue,
            Color::Black,
            Color::Red,
            Color::Green,
        ] {
            if self.cost[color as usize] > 0 {
                out.insert(color);
            }
        }
        out
    }
}

/// One class and its pooled amounts by W/U/B/R/G/C.
pub type PooledClass = (ManaClass, [u8; 6]);

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, serde::Serialize)]
pub struct Mana {
    pub mana: [u8; 6],
    /// Mana that keeps a spending restriction or a spend effect until spent
    /// or lost (CR 106.6), by class and color. Canonical: sorted by class, no
    /// all-zero entry, `None` when empty. Boxed so ordinary pools stay small
    /// and allocation-free when cloned.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[allow(clippy::box_collection)] // a thin pointer; `Vec` would widen every pool
    pub classes: Option<Box<Vec<PooledClass>>>,
}

/// A native spending alternative, including the cost still owed afterwards.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ManaSpendChoice {
    pub unit: Mana,
    pub remaining: ManaCost,
}

impl Mana {
    pub fn parse(text: &str) -> Self {
        let mut out = Mana::default();
        let mut chars = text.chars().peekable();

        while let Some(ch) = chars.peek().copied() {
            if ch.is_ascii_digit() {
                let mut value: u16 = 0;
                while let Some(digit) = chars.peek().copied() {
                    if !digit.is_ascii_digit() {
                        break;
                    }
                    value = value * 10 + (digit as u16 - b'0' as u16);
                    chars.next();
                }
                out.mana[Color::Colorless as usize] += value.min(u8::MAX as u16) as u8;
                continue;
            }

            let slot = match ch {
                'W' => Color::White,
                'U' => Color::Blue,
                'B' => Color::Black,
                'R' => Color::Red,
                'G' => Color::Green,
                'C' => Color::Colorless,
                _ => panic!("invalid mana symbol: {ch}"),
            };
            out.mana[slot as usize] += 1;
            chars.next();
        }

        out
    }

    pub fn single(color: Color) -> Self {
        let mut out = Mana::default();
        if color == Color::Generic {
            return out;
        }
        out.mana[color as usize] = 1;
        out
    }

    /// Mana of `amounts` restricted to `restriction`.
    pub fn restricted_to(restriction: ManaRestriction, amounts: [u8; 6]) -> Self {
        Self::of_class(restriction.into(), amounts)
    }

    /// Mana of `amounts` in `class`.
    pub fn of_class(class: ManaClass, amounts: [u8; 6]) -> Self {
        let mut out = Mana::default();
        out.add_class(&class, amounts);
        out
    }

    /// Non-ordinary classes with their amounts by color, sorted by class.
    pub fn classes(&self) -> &[PooledClass] {
        self.classes.as_deref().map_or(&[], Vec::as_slice)
    }

    /// This mana without `unit`'s own kind (its class and color): the mana
    /// that could pay in its place with different consequences, whether a
    /// rider, a restriction or the color spent and the color left over.
    pub fn other_than(&self, unit: &Mana) -> Mana {
        fn clear(pooled: &mut [u8; 6], kind: &[u8; 6]) {
            for (pooled, kind) in pooled.iter_mut().zip(kind) {
                if *kind > 0 {
                    *pooled = 0;
                }
            }
        }
        let mut out = self.clone();
        clear(&mut out.mana, &unit.mana);
        for (class, kind) in unit.classes() {
            clear(out.class_mut(class), kind);
        }
        out.prune();
        out
    }

    /// One class's amounts, by color.
    pub fn class_of(&self, class: &ManaClass) -> [u8; 6] {
        self.classes()
            .iter()
            .find(|(pooled, _)| pooled == class)
            .map_or([0; 6], |(_, amounts)| *amounts)
    }

    /// The amounts restricted to exactly `restriction` with no spend effect,
    /// by color.
    pub fn restricted_of(&self, restriction: &ManaRestriction) -> [u8; 6] {
        self.class_of(&restriction.clone().into())
    }

    /// The class's amounts, inserting a zero entry; callers prune.
    fn class_mut(&mut self, class: &ManaClass) -> &mut [u8; 6] {
        let classes = self.classes.get_or_insert_with(Default::default);
        let index = match classes.binary_search_by(|(pooled, _)| pooled.cmp(class)) {
            Ok(index) => index,
            Err(index) => {
                classes.insert(index, (class.clone(), [0; 6]));
                index
            }
        };
        &mut classes[index].1
    }

    /// Restore the canonical form after spending.
    fn prune(&mut self) {
        if let Some(classes) = &mut self.classes {
            classes.retain(|(_, amounts)| *amounts != [0; 6]);
            if classes.is_empty() {
                self.classes = None;
            }
        }
    }

    fn add_class(&mut self, class: &ManaClass, amounts: [u8; 6]) {
        if amounts == [0; 6] {
            return;
        }
        let pooled = self.class_mut(class);
        for (slot, amount) in pooled.iter_mut().zip(amounts) {
            *slot = slot.saturating_add(amount);
        }
    }

    /// All spend-restricted mana by color, whatever it may be spent on.
    pub fn restricted_colors(&self) -> [u8; 6] {
        self.class_colors(|class| class.restriction.is_some())
    }

    /// Unrestricted mana by color: ordinary mana and mana that only carries
    /// a spend effect.
    pub fn unrestricted_colors(&self) -> [u8; 6] {
        let mut out = self.class_colors(|class| class.restriction.is_none());
        for (slot, amount) in out.iter_mut().zip(self.mana) {
            *slot = slot.saturating_add(amount);
        }
        out
    }

    fn class_colors(&self, include: impl Fn(&ManaClass) -> bool) -> [u8; 6] {
        let mut out = [0_u8; 6];
        for (_, amounts) in self.classes().iter().filter(|(class, _)| include(class)) {
            for (slot, amount) in out.iter_mut().zip(amounts) {
                *slot = slot.saturating_add(*amount);
            }
        }
        out
    }

    pub fn add(&mut self, other: &Mana) {
        for i in 0..self.mana.len() {
            self.mana[i] = self.mana[i].saturating_add(other.mana[i]);
        }
        for (class, amounts) in other.classes() {
            self.add_class(class, *amounts);
        }
    }

    /// Per class and color, the larger of the two (an optimistic bound).
    pub fn max_with(&mut self, other: &Mana) {
        for i in 0..self.mana.len() {
            self.mana[i] = self.mana[i].max(other.mana[i]);
        }
        for (class, amounts) in other.classes() {
            let pooled = self.class_mut(class);
            for (slot, amount) in pooled.iter_mut().zip(amounts) {
                *slot = (*slot).max(*amount);
            }
        }
    }

    pub fn total(&self) -> u16 {
        self.mana
            .iter()
            .chain(self.classes().iter().flat_map(|(_, amounts)| amounts))
            .map(|x| *x as u16)
            .sum()
    }

    /// Total usable for one spend purpose, without promising a color assignment.
    pub fn usable_total(&self, cost: &ManaCost) -> u16 {
        self.mana
            .iter()
            .chain(
                self.classes()
                    .iter()
                    .filter(|(class, _)| class.allows(&cost.purpose))
                    .flat_map(|(_, amounts)| amounts),
            )
            .map(|amount| u16::from(*amount))
            .sum()
    }

    /// The most units of this pool one payment of `cost` can spend: a maximum
    /// assignment of units to pips, over the classes the purpose allows. A
    /// colored pip takes its own color first, which never costs a unit a
    /// hybrid or generic pip could have used instead.
    pub fn spendable_units(&self, cost: &ManaCost) -> u16 {
        fn hybrids(mut available: [u8; 6], pips: &[[Color; 2]], generic: u8) -> u16 {
            let Some((pip, rest)) = pips.split_first() else {
                return available
                    .iter()
                    .map(|n| u16::from(*n))
                    .sum::<u16>()
                    .min(u16::from(generic));
            };
            let mut best = hybrids(available, rest, generic);
            for color in pip {
                let slot = *color as usize;
                if available[slot] > 0 {
                    available[slot] -= 1;
                    best = best.max(1 + hybrids(available, rest, generic));
                    available[slot] += 1;
                }
            }
            best
        }
        let mut available = self.mana;
        for (_, amounts) in self
            .classes()
            .iter()
            .filter(|(class, _)| class.allows(&cost.purpose))
        {
            for (total, amount) in available.iter_mut().zip(amounts) {
                *total = total.saturating_add(*amount);
            }
        }
        let mut spent = 0;
        for (amount, required) in available.iter_mut().zip(cost.cost) {
            let used = (*amount).min(required);
            *amount -= used;
            spent += u16::from(used);
        }
        spent + hybrids(available, &cost.hybrid, cost.generic())
    }

    /// Color feasibility only; this never chooses which class is spent.
    pub fn can_pay(&self, cost: &ManaCost) -> bool {
        fn hybrids(mut available: [u8; 6], pips: &[[Color; 2]], generic: u8) -> bool {
            let Some((pip, rest)) = pips.split_first() else {
                return available.iter().map(|n| u16::from(*n)).sum::<u16>() >= u16::from(generic);
            };
            for color in pip {
                let slot = *color as usize;
                if available[slot] > 0 {
                    available[slot] -= 1;
                    if hybrids(available, rest, generic) {
                        return true;
                    }
                    available[slot] += 1;
                }
            }
            false
        }
        let mut available = self.mana;
        for (_, amounts) in self
            .classes()
            .iter()
            .filter(|(class, _)| class.allows(&cost.purpose))
        {
            for (total, amount) in available.iter_mut().zip(amounts) {
                *total = total.saturating_add(*amount);
            }
        }
        for (amount, required) in available.iter_mut().zip(cost.cost) {
            let Some(left) = amount.checked_sub(required) else {
                return false;
            };
            *amount = left;
        }
        hybrids(available, &cost.hybrid, cost.generic())
    }

    /// One unit of one class pays the next colored/hybrid/generic pip. The
    /// caller checks the remaining obligations before offering any selection.
    pub fn spending_choices(&self, cost: &ManaCost) -> Vec<ManaSpendChoice> {
        // Every eligible unit pays the same next pip, so the remaining cost
        // is independent of its color and class.
        let mut remaining = cost.clone();
        let allowed: [bool; 6] = if let Some(pip) = remaining.cost[..6].iter().position(|n| *n > 0)
        {
            remaining.cost[pip] -= 1;
            std::array::from_fn(|color| color == pip)
        } else if !remaining.hybrid.is_empty() {
            let pip = remaining.hybrid.remove(0);
            std::array::from_fn(|color| pip.iter().any(|c| *c as usize == color))
        } else if remaining.generic() > 0 {
            remaining.cost[6] -= 1;
            [true; 6]
        } else {
            return Vec::new();
        };
        remaining.update_mana_value();
        let mut result = Vec::new();
        for (class, amounts) in std::iter::once((None, &self.mana)).chain(
            self.classes()
                .iter()
                .filter(|(class, _)| class.allows(&cost.purpose))
                .map(|(class, amounts)| (Some(class), amounts)),
        ) {
            for (color, amount) in amounts.iter().enumerate() {
                if *amount == 0 || !allowed[color] {
                    continue;
                }
                let mut amounts = [0; 6];
                amounts[color] = 1;
                let unit = match class {
                    Some(class) => Self::of_class(class.clone(), amounts),
                    None => Self {
                        mana: amounts,
                        ..Self::default()
                    },
                };
                result.push(ManaSpendChoice {
                    unit,
                    remaining: remaining.clone(),
                });
            }
        }
        result
    }

    /// Remove exactly the selected mana. No ordering preference is applied
    /// here.
    pub fn spend(&mut self, unit: &Mana) {
        for (amount, spent) in self.mana.iter_mut().zip(unit.mana) {
            *amount = amount.checked_sub(spent).expect("selected pooled mana");
        }
        for (class, amounts) in unit.classes() {
            let available = self.class_mut(class);
            for (color, spent) in amounts.iter().enumerate() {
                available[color] = available[color]
                    .checked_sub(*spent)
                    .expect("selected class");
            }
        }
        self.prune();
    }

    /// Combine combat mana without losing which units survive a step end.
    pub fn add_combat(&mut self, combat: &Mana) {
        self.add_class(
            &ManaClass {
                combat: true,
                restriction: None,
                effect: None,
            },
            combat.mana,
        );
        for (class, amounts) in combat.classes() {
            let mut class = class.clone();
            class.combat = true;
            self.add_class(&class, *amounts);
        }
    }

    /// All unspent mana becomes red, keeping its spending restrictions
    /// (Ozai, the Phoenix King).
    pub fn become_red(&mut self) {
        let red = Color::Red as usize;
        let all_red = |pool: &mut [u8; 6]| {
            let total = pool
                .iter()
                .fold(0u8, |sum, amount| sum.saturating_add(*amount));
            *pool = [0; 6];
            pool[red] = total;
        };
        all_red(&mut self.mana);
        if let Some(classes) = &mut self.classes {
            for (_, amounts) in classes.iter_mut() {
                all_red(amounts);
            }
        }
    }

    /// Only red mana, with its spending restrictions.
    pub fn red_only(&self) -> Self {
        let red = Color::Red as usize;
        let mut kept = Self::default();
        kept.mana[red] = self.mana[red];
        for (class, amounts) in self.classes() {
            let mut only_red = [0; 6];
            only_red[red] = amounts[red];
            kept.add_class(class, only_red);
        }
        kept
    }

    pub fn clear(&mut self) {
        // CR 106.4 — Unspent mana empties from pools as steps/phases end.
        self.mana = [0; 6];
        self.classes = None;
    }
}

/// One pooled unit: its class (`None` for ordinary mana), its color as
/// `Mana.mana` indexes it, and the object that produced it.
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize)]
struct PooledUnit {
    #[serde(skip_serializing_if = "Option::is_none")]
    class: Option<ManaClass>,
    color: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    origin: Option<ManaOrigin>,
}

/// A player's mana pool: the mana and, unit for unit, where it came from.
///
/// Invariant: `units` holds exactly one entry per pooled unit, agreeing with
/// `mana` per class and color, oldest first. Every writer is a method here,
/// and debug builds check the agreement after each one.
///
/// A property a rule reads belongs in `ManaClass`; the origin only names
/// which object produced a unit. Units of one class and color are
/// interchangeable by rule (CR 106.4), so spending takes the oldest: no
/// choice is hidden and payment search compares `Mana` alone.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, serde::Serialize)]
pub struct ManaPool {
    #[serde(flatten)]
    mana: Mana,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    units: Vec<PooledUnit>,
}

impl ManaPool {
    pub fn mana(&self) -> &Mana {
        &self.mana
    }

    /// Add `mana` produced by `origin` (`None` for mana no object produced).
    pub fn add(&mut self, mana: &Mana, origin: Option<ManaOrigin>) {
        let before = self.mana.clone();
        self.mana.add(mana);
        // Amounts saturate, so count what the pool actually gained.
        let ordinary = std::iter::once((None, before.mana, self.mana.mana));
        let classed = mana.classes().iter().map(|(class, _)| {
            let now = self.mana.class_of(class);
            (Some(class), before.class_of(class), now)
        });
        for (class, had, now) in ordinary.chain(classed) {
            for color in 0..had.len() {
                for _ in had[color]..now[color] {
                    self.units.push(PooledUnit {
                        class: class.cloned(),
                        color: color as u8,
                        origin,
                    });
                }
            }
        }
        self.check();
    }

    /// Join `other`'s units, keeping their origins, after this pool's.
    pub fn absorb(&mut self, other: ManaPool) {
        self.mana.add(&other.mana);
        self.units.extend(other.units);
        self.drop_uncounted_units();
        self.check();
    }

    /// Remove exactly the selected mana and return each unit as it was
    /// spent: its origin, and its class's rider when that applies to
    /// `purpose`. The oldest unit of a class and color is spent first.
    pub fn spend(&mut self, unit: &Mana, purpose: &SpendingContext) -> Vec<SpentMana> {
        self.mana.spend(unit);
        let mut spent = Vec::new();
        let ordinary = std::iter::once((None, &unit.mana));
        let classed = unit
            .classes()
            .iter()
            .map(|(class, amounts)| (Some(class), amounts));
        for (class, amounts) in ordinary.chain(classed) {
            let effect = class
                .and_then(|class| class.effect.as_ref())
                .filter(|effect| effect.applies(purpose));
            for (color, amount) in amounts.iter().enumerate() {
                for _ in 0..*amount {
                    let index = self
                        .units
                        .iter()
                        .position(|pooled| {
                            pooled.color as usize == color && pooled.class.as_ref() == class
                        })
                        .expect("a pooled unit has an origin entry");
                    spent.push(SpentMana {
                        color: color as u8,
                        source: self.units.remove(index).origin,
                        effect: effect.cloned(),
                    });
                }
            }
        }
        self.check();
        spent
    }

    /// How many units the pool holds. Units added later come after them, so
    /// a count taken before a production marks where its mana begins.
    pub fn unit_count(&self) -> usize {
        self.units.len()
    }

    /// The color of the pooled unit at `index` (W/U/B/R/G/C, as `Mana.mana`
    /// indexes it).
    pub fn unit_color(&self, index: usize) -> u8 {
        self.units[index].color
    }

    /// The pooled unit at `index`, as one unit of `Mana`.
    pub fn unit(&self, index: usize) -> Mana {
        let pooled = &self.units[index];
        let mut amounts = [0; 6];
        amounts[pooled.color as usize] = 1;
        match &pooled.class {
            Some(class) => Mana::of_class(class.clone(), amounts),
            None => Mana {
                mana: amounts,
                ..Mana::default()
            },
        }
    }

    /// Spend exactly the unit at `index` toward `purpose`, keeping its
    /// origin: the unit a production just added, not the oldest of its kind.
    pub fn spend_at(&mut self, index: usize, purpose: &SpendingContext) -> SpentMana {
        self.mana.spend(&self.unit(index));
        let pooled = self.units.remove(index);
        let effect = pooled
            .class
            .and_then(|class| class.effect)
            .filter(|effect| effect.applies(purpose));
        self.check();
        SpentMana {
            color: pooled.color,
            source: pooled.origin,
            effect,
        }
    }

    /// All unspent mana becomes red, keeping its spending restrictions and
    /// origins (Ozai, the Phoenix King).
    pub fn become_red(&mut self) {
        self.mana.become_red();
        for unit in &mut self.units {
            unit.color = Color::Red as u8;
        }
        self.drop_uncounted_units();
        self.check();
    }

    /// Only red mana, with its spending restrictions and origins.
    pub fn red_only(mut self) -> Self {
        self.mana = self.mana.red_only();
        self.units.retain(|unit| unit.color == Color::Red as u8);
        self.check();
        self
    }

    /// The origin of each pooled unit of `class` (`None` for ordinary mana)
    /// and `color`, oldest first.
    pub fn origins<'a>(
        &'a self,
        class: Option<&'a ManaClass>,
        color: Color,
    ) -> impl Iterator<Item = Option<ManaOrigin>> + 'a {
        self.units
            .iter()
            .filter(move |unit| unit.color == color as u8 && unit.class.as_ref() == class)
            .map(|unit| unit.origin)
    }

    /// Amounts saturate at 255 per class and color, so joining or recoloring
    /// can leave more units than the mana counts. The newest go: the mana
    /// they stood for no longer exists.
    fn drop_uncounted_units(&mut self) {
        let mut left = self.mana.clone();
        self.units.retain(|unit| {
            let slot = match &unit.class {
                Some(class) => &mut left.class_mut(class)[unit.color as usize],
                None => &mut left.mana[unit.color as usize],
            };
            let counted = *slot > 0;
            *slot = slot.saturating_sub(1);
            counted
        });
    }

    fn check(&self) {
        #[cfg(debug_assertions)]
        {
            let mut counted = Mana::default();
            for unit in &self.units {
                let mut amounts = [0u8; 6];
                amounts[unit.color as usize] = 1;
                match &unit.class {
                    Some(class) => counted.add_class(class, amounts),
                    None => counted.mana[unit.color as usize] += 1,
                }
            }
            debug_assert_eq!(counted, self.mana, "pool origins disagree with its mana");
        }
    }
}

impl std::ops::Deref for ManaPool {
    type Target = Mana;

    fn deref(&self) -> &Mana {
        &self.mana
    }
}

impl PartialEq<Mana> for ManaPool {
    fn eq(&self, other: &Mana) -> bool {
        self.mana == *other
    }
}

/// Mana no object produced: a constructed position's pool.
impl From<Mana> for ManaPool {
    fn from(mana: Mana) -> Self {
        let mut pool = Self::default();
        pool.add(&mana, None);
        pool
    }
}

#[cfg(test)]
mod tests {
    use super::{Mana, ManaCost, ManaRestriction, SpendingContext};
    use crate::card::CardType;

    /// A color holds at most 255 mana. Joining two pools or turning a pool
    /// red past that keeps one origin entry per unit of mana that remains.
    #[test]
    fn a_saturated_pool_keeps_one_origin_per_remaining_unit() {
        use super::{Color, ManaPool};
        let of = |color: Color, amount: u8| {
            let mut mana = Mana::default();
            mana.mana[color as usize] = amount;
            ManaPool::from(mana)
        };
        let mut joined = of(Color::Red, 200);
        joined.absorb(of(Color::Red, 100));
        assert_eq!(joined.total(), 255);
        assert_eq!(joined.origins(None, Color::Red).count(), 255);

        let mut red = of(Color::Green, 200);
        red.absorb(of(Color::Blue, 100));
        red.become_red();
        assert_eq!(red.total(), 255);
        assert_eq!(red.origins(None, Color::Red).count(), 255);
    }

    #[test]
    fn spendable_units_count_a_maximum_assignment_to_pips() {
        let cost = ManaCost::parse("1G");
        assert_eq!(Mana::parse("G").spendable_units(&cost), 1);
        assert_eq!(Mana::parse("R").spendable_units(&cost), 1);
        // Only one red unit fits: the other pip needs green.
        assert_eq!(Mana::parse("RR").spendable_units(&cost), 1);
        assert_eq!(Mana::parse("RG").spendable_units(&cost), 2);
        assert_eq!(Mana::parse("GGG").spendable_units(&cost), 2);
        // A hybrid pip takes either colour and may be left for a source.
        let hybrid = ManaCost::parse("{U/R}{R}");
        assert_eq!(Mana::parse("U").spendable_units(&hybrid), 1);
        assert_eq!(Mana::parse("RR").spendable_units(&hybrid), 2);
        assert_eq!(Mana::parse("G").spendable_units(&hybrid), 0);
    }

    #[test]
    fn hybrid_feasibility_backtracks_and_preserves_required_colors() {
        let cost = ManaCost::parse("{1}{U/R}{U/B}{R}");
        assert_eq!(cost.mana_value, 4);
        assert!(Mana::parse("1URR").can_pay(&cost));
        assert!(!Mana::parse("1UUU").can_pay(&cost));
        let cost = ManaCost::parse("{U/R}{U/B}");
        assert!(Mana::parse("UR").can_pay(&cost));
    }

    #[test]
    fn hybrid_cost_arithmetic_keeps_pips_and_color_identity() {
        use super::{Color, Colors};
        let cost = ManaCost::parse("{2}{R/W}{R/W}{R/W}");
        assert_eq!(cost.mana_value, 5);
        assert_eq!(cost.colors(), Colors::from([Color::Red, Color::White]));
        let reduced = cost.reduced_generic(1);
        assert_eq!(reduced.mana_value, 4);
        assert!(Mana::parse("1RRW").can_pay(&reduced));
        let kicked = reduced.plus(&ManaCost::parse("1G"));
        assert_eq!(kicked.mana_value, 6);
        assert!(Mana::parse("2RRWG").can_pay(&kicked));
        assert_eq!(kicked.with_generic(0).mana_value, 4);
    }

    #[test]
    fn colored_reductions_spill_into_generic_but_never_other_colors() {
        for (cost, reduction, expected) in [
            ("{5}{W}{W}{U}{R}", "{W}{W}{W}{U}{B}{R}{G}", "{2}"),
            ("{1}{U}{U}", "{W}{W}{W}", "{U}{U}"),
            ("{3}{G}", "{2}{G}{G}", "{0}"),
            ("{X}{2}{R/W}{R}", "{R}{R}{R}", "{X}{R/W}"),
        ] {
            assert_eq!(
                ManaCost::parse(cost).reduced_by(&ManaCost::parse(reduction)),
                ManaCost::parse(expected),
                "{cost} less {reduction}"
            );
        }
    }

    #[test]
    fn x_counts_zero_until_its_value_is_chosen() {
        let cost = ManaCost::parse("{X}{R}{R}{R}");
        assert_eq!((cost.x, cost.mana_value), (1, 3));
        let paid = cost.with_x(2);
        assert_eq!((paid.x, paid.mana_value, paid.generic()), (0, 5, 2));
        assert_eq!(ManaCost::parse("{X}{X}{B}").with_x(3).generic(), 6);
    }

    #[test]
    fn spending_choices_keep_color_restriction_rider_and_expiry() {
        let rider: super::ManaClass = super::SpendEffect::CreatureEntersWithPlusOneCounter {
            without_subtype: "Human".into(),
        }
        .into();
        let mut pool = Mana::parse("WU");
        pool.add(&Mana::of_class(rider.clone(), [1, 0, 0, 0, 0, 0]));
        pool.add_combat(&Mana::of_class(rider, [1, 0, 0, 0, 0, 0]));
        pool.add(&Mana::restricted_to(
            ManaRestriction::spell(["Lesson"]),
            [0, 0, 0, 0, 2, 0],
        ));
        let cost = ManaCost::parse("1");
        assert_eq!(pool.spending_choices(&cost).len(), 4);
        assert!(pool.spending_choices(&ManaCost::parse("G")).is_empty());
        let spell = ManaCost {
            purpose: SpendingContext::cast(["Lesson"], [CardType::Creature]),
            ..cost.clone()
        };
        let options = pool.spending_choices(&spell);
        assert_eq!(options.len(), 5);
        for choice in options {
            let mut remaining = super::ManaPool::from(pool.clone());
            let receipt = remaining.spend(&choice.unit, &spell.purpose);
            assert_eq!(remaining.total(), pool.total() - 1);
            assert_eq!(receipt.len(), 1);
            assert_eq!(
                u8::from(receipt[0].effect.is_some()),
                u8::from(
                    choice
                        .unit
                        .classes()
                        .iter()
                        .any(|(class, _)| class.effect.is_some())
                )
            );
        }
        assert_eq!(
            Mana::parse("WU").spending_choices(&ManaCost::parse("{W/U}")),
            Mana::parse("WU").spending_choices(&cost)
        );
    }

    #[test]
    fn parse_and_pay_mana() {
        let mut mana = Mana::parse("2RG");
        let cost = ManaCost::parse("1RG");
        assert!(mana.can_pay(&cost));
        mana.spend(&Mana::parse("1RG"));
        assert_eq!(mana.total(), 1);
    }

    #[test]
    fn colored_and_generic_cost_consumes_required_units() {
        use super::Color;
        let mut mana = Mana::default();
        mana.mana[Color::Colorless as usize] = 2;
        mana.mana[Color::Green as usize] = 1;

        let cost = ManaCost::parse("2G");
        assert!(mana.can_pay(&cost));
        mana.spend(&Mana::parse("2G"));

        assert_eq!(mana.mana[Color::Colorless as usize], 0);
        assert_eq!(mana.mana[Color::Green as usize], 0);
    }

    #[test]
    fn generic_cost_can_preserve_colored_mana() {
        use super::Color;
        let mut mana = Mana::default();
        mana.mana[Color::Colorless as usize] = 1;
        mana.mana[Color::Red as usize] = 1;

        let cost = ManaCost::parse("1");
        assert_eq!(mana.spending_choices(&cost).len(), 2);
        mana.spend(&Mana::parse("1"));

        // Colorless spent, red preserved
        assert_eq!(mana.mana[Color::Colorless as usize], 0);
        assert_eq!(mana.mana[Color::Red as usize], 1);
    }

    #[test]
    fn restricted_mana_pays_only_its_purposes_with_both_classes_offered() {
        let lesson = ManaRestriction::spell(["Lesson"]);
        let lesson_or_shrine = ManaRestriction::spell(["Lesson", "Shrine"]);
        let ally = ManaRestriction::only(["Ally"], ["Ally"]);
        let mut pool = Mana::restricted_to(lesson_or_shrine.clone(), [0, 1, 0, 0, 0, 0]);
        pool.add(&Mana::restricted_to(lesson.clone(), [0, 1, 0, 0, 0, 0]));
        pool.add(&Mana::restricted_to(ally.clone(), [1, 0, 0, 0, 0, 0]));
        let cost = |text: &str, purpose: SpendingContext| ManaCost {
            purpose,
            ..ManaCost::parse(text)
        };
        assert!(!pool.can_pay(&ManaCost::parse("1")));
        assert!(!pool.can_pay(&cost("U", SpendingContext::cast(["Ally"], []))));
        assert!(pool.can_pay(&cost("W", SpendingContext::ability(["Human", "Ally"]))));
        assert!(pool.can_pay(&cost("U", SpendingContext::cast(["Shrine"], []))));
        assert!(!pool.can_pay(&cost("UU", SpendingContext::cast(["Shrine"], []))));
        // A Lesson spends the Lesson-only mana and keeps the broader class.
        let mut paid = pool.clone();
        let payment = cost("U", SpendingContext::cast(["Lesson"], []));
        assert_eq!(pool.spending_choices(&payment).len(), 2);
        paid.spend(&Mana::restricted_to(lesson.clone(), [0, 1, 0, 0, 0, 0]));
        assert_eq!(paid.restricted_of(&lesson), [0; 6]);
        assert_eq!(paid.restricted_of(&lesson_or_shrine), [0, 1, 0, 0, 0, 0]);
        assert_eq!(paid.classes().len(), 2);
        assert_eq!(pool.restricted_colors(), [1, 2, 0, 0, 0, 0]);
        assert_eq!(ally.describe(), "Ally spells or abilities of Ally sources");
        assert_eq!(lesson_or_shrine.describe(), "Lesson or Shrine spells");
    }

    /// No subtype vocabulary is privileged: a restriction naming any subtype
    /// keeps it, admits exactly that purpose and never pays a resolving cost.
    #[test]
    fn restrictions_name_any_subtype_and_resolving_costs_use_none() {
        let dragon = ManaRestriction::spell(["Dragon"]);
        let elemental_sources = ManaRestriction::only([], ["Elemental"]);
        let mut pool = Mana::restricted_to(dragon.clone(), [0, 0, 0, 1, 0, 0]);
        pool.add(&Mana::restricted_to(
            elemental_sources.clone(),
            [0, 0, 0, 0, 1, 0],
        ));
        assert_eq!(pool.classes().len(), 2);
        assert_eq!(dragon.describe(), "Dragon spells");
        let cost = |text: &str, purpose: SpendingContext| ManaCost {
            purpose,
            ..ManaCost::parse(text)
        };
        let dragon_spell = SpendingContext::cast(["Dragon", "Elder"], [CardType::Creature]);
        assert!(pool.can_pay(&cost("R", dragon_spell.clone())));
        assert!(!pool.can_pay(&cost("G", dragon_spell)));
        // An Elemental spell is not an ability of an Elemental source.
        assert!(!pool.can_pay(&cost("G", SpendingContext::cast(["Elemental"], []))));
        let mut paid = pool.clone();
        paid.spend(&Mana::restricted_to(
            elemental_sources.clone(),
            [0, 0, 0, 0, 1, 0],
        ));
        assert_eq!(paid.restricted_of(&elemental_sources), [0; 6]);
        assert_eq!(paid.restricted_of(&dragon), [0, 0, 0, 1, 0, 0]);
        assert!(!pool.can_pay(&ManaCost::parse("1")));
    }

    #[test]
    fn exclusions_admit_every_other_purpose_including_resolving_costs() {
        let powerstone = ManaRestriction::ExceptNonartifactSpells;
        let pool = Mana::restricted_to(powerstone.clone(), [0, 0, 0, 0, 0, 2]);
        let cost = |purpose: SpendingContext| ManaCost {
            purpose,
            ..ManaCost::parse("2")
        };
        assert!(pool.can_pay(&ManaCost::parse("2")));
        assert!(pool.can_pay(&cost(SpendingContext::ability([]))));
        assert!(pool.can_pay(&cost(SpendingContext::cast([], [CardType::Artifact]))));
        assert!(!pool.can_pay(&cost(SpendingContext::cast([], [CardType::Creature]))));
        assert_eq!(powerstone.describe(), "anything but nonartifact spells");
        // Both eligible classes remain deliberate choices.
        let mut mixed = pool.clone();
        mixed.add(&Mana::restricted_to(
            ManaRestriction::only([], ["Golem"]),
            [0, 0, 0, 0, 0, 1],
        ));
        assert_eq!(
            mixed
                .spending_choices(&cost(SpendingContext::ability(["Golem"])).with_generic(1))
                .len(),
            2
        );
    }
}
