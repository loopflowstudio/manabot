use std::{collections::BTreeSet, fmt, sync::Arc};

use super::{
    ability::{
        effect_target_requirements, Ability, Condition, Effect, EffectValue, PaymentCost,
        TargetRequirement, TargetSpec,
    },
    game_object::{ObjectId, ObjectRef, PlayerId},
    mana::{Color, Colors, Mana, ManaCost},
    permanent::CounterKind,
    predicate::CardPredicate,
    zone::ZoneType,
};

/// Stable index of an immutable card definition in a `ContentPack`.
///
/// The numeric value deliberately matches the legacy `registry_key` exposed
/// through observations so the Python and fixed-action ABIs do not change.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub struct CardDefId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
pub enum CardType {
    Creature,
    Instant,
    Sorcery,
    Planeswalker,
    Land,
    Enchantment,
    Artifact,
    Kindred,
    Battle,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct CardTypes {
    pub types: BTreeSet<CardType>,
}

impl CardTypes {
    pub fn new(types: impl IntoIterator<Item = CardType>) -> Self {
        Self {
            types: types.into_iter().collect(),
        }
    }

    #[inline]
    pub fn is_castable(&self) -> bool {
        !self.is_land() && !self.types.is_empty()
    }

    #[inline]
    pub fn is_permanent(&self) -> bool {
        self.is_creature()
            || self.is_land()
            || self.is_artifact()
            || self.is_enchantment()
            || self.is_planeswalker()
            || self.is_battle()
    }

    #[inline]
    pub fn is_non_land_permanent(&self) -> bool {
        self.is_permanent() && !self.is_land()
    }

    #[inline]
    pub fn is_non_creature_permanent(&self) -> bool {
        self.is_permanent() && !self.is_creature()
    }

    #[inline]
    pub fn is_spell(&self) -> bool {
        self.types.contains(&CardType::Instant) || self.types.contains(&CardType::Sorcery)
    }

    #[inline]
    pub fn is_instant(&self) -> bool {
        self.types.contains(&CardType::Instant)
    }

    #[inline]
    pub fn is_creature(&self) -> bool {
        self.types.contains(&CardType::Creature)
    }

    #[inline]
    pub fn is_land(&self) -> bool {
        self.types.contains(&CardType::Land)
    }

    #[inline]
    pub fn is_planeswalker(&self) -> bool {
        self.types.contains(&CardType::Planeswalker)
    }

    #[inline]
    pub fn is_enchantment(&self) -> bool {
        self.types.contains(&CardType::Enchantment)
    }

    #[inline]
    pub fn is_artifact(&self) -> bool {
        self.types.contains(&CardType::Artifact)
    }

    #[inline]
    pub fn is_kindred(&self) -> bool {
        self.types.contains(&CardType::Kindred)
    }

    #[inline]
    pub fn is_battle(&self) -> bool {
        self.types.contains(&CardType::Battle)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ManaAbility {
    pub mana: Mana,
}

/// A triggered mana ability (CR 605.1b): "Whenever you tap a [predicate]
/// for mana, add [mana]." Resolves immediately — no stack, no priority.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct TriggeredManaAbility {
    pub predicate: CardPredicate,
    pub mana: Mana,
}

// One declaration owns the enum, iteration order and storage capacity. Append
// variants to preserve serde variant indices and present-keyword ordering.
macro_rules! keywords {
    ($( $(#[$meta:meta])* $variant:ident => $name:literal ),+ $(,)?) => {
        /// A parameterless keyword ability (CR 702), held in a [`Keywords`] set.
        /// Parameterized keywords (ward, firebending) have their own rules types.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
        pub enum Keyword {
            $( $(#[$meta])* #[serde(rename = $name)] $variant, )+
        }

        impl Keyword {
            /// Every keyword, in serialization order.
            pub const ALL: [Keyword; [$($name),+].len()] = [$(Self::$variant),+];

            /// The serialized name, as learning tokens and effect facts name it.
            pub fn name(self) -> &'static str {
                match self { $(Self::$variant => $name),+ }
            }
        }
    };
}

keywords! {
    Flying => "flying",
    Reach => "reach",
    Haste => "haste",
    /// CR 702.8 — May be cast any time the controller could cast an instant.
    Flash => "flash",
    Vigilance => "vigilance",
    Trample => "trample",
    FirstStrike => "first_strike",
    DoubleStrike => "double_strike",
    Deathtouch => "deathtouch",
    Lifelink => "lifelink",
    Defender => "defender",
    Menace => "menace",
    /// CR 702.11 — Can't be the target of spells or abilities opponents
    /// control.
    Hexproof => "hexproof",
    Indestructible => "indestructible",
}

impl Keyword {
    /// The keyword as printed on a card (`first strike`).
    pub fn printed(self) -> &'static str {
        match self {
            Keyword::FirstStrike => "first strike",
            Keyword::DoubleStrike => "double strike",
            other => other.name(),
        }
    }
}

/// Internal ordinal storage; its array grows at compile time with the vocabulary.
/// Keeping it inline preserves `Keywords: Copy` without allocating per permanent.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct KeywordBits<const WORDS: usize>([u64; WORDS]);

impl<const WORDS: usize> Default for KeywordBits<WORDS> {
    fn default() -> Self {
        Self([0; WORDS])
    }
}

impl<const WORDS: usize> KeywordBits<WORDS> {
    fn has(self, index: usize) -> bool {
        self.0[index / 64] & (1_u64 << (index % 64)) != 0
    }

    fn insert(&mut self, index: usize) {
        self.0[index / 64] |= 1_u64 << (index % 64);
    }

    fn without(self, removed: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i] & !removed.0[i]))
    }

    fn union(self, other: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i] | other.0[i]))
    }
}

/// A set of keywords. Serialized as the sequence of the keywords present, in
/// [`Keyword::ALL`] order (as `Counters` lists its present kinds), so a
/// definition's content digest and learning tokens carry only what the set
/// holds and a new keyword changes no other card's encoding.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Keywords(KeywordBits<{ Keyword::ALL.len().div_ceil(64) }>);

impl Keywords {
    /// Exactly these keywords.
    pub fn of(keywords: &[Keyword]) -> Self {
        let mut result = Self::default();
        for &keyword in keywords {
            result.insert(keyword);
        }
        result
    }

    pub fn has(self, keyword: Keyword) -> bool {
        self.0.has(keyword as usize)
    }

    pub fn insert(&mut self, keyword: Keyword) {
        self.0.insert(keyword as usize);
    }

    /// The keywords present, in [`Keyword::ALL`] order.
    pub fn iter(self) -> impl Iterator<Item = Keyword> {
        Keyword::ALL
            .into_iter()
            .filter(move |keyword| self.has(*keyword))
    }

    /// The names of the keywords present (as the learning tokens name them),
    /// in [`Keyword::ALL`] order.
    pub fn names(self) -> Vec<&'static str> {
        self.iter().map(Keyword::name).collect()
    }

    /// These keywords less those in `removed`.
    pub fn without(self, removed: Keywords) -> Keywords {
        Keywords(self.0.without(removed.0))
    }

    /// Keyword-set union (printed keywords plus until-EOT grants).
    pub fn union(self, other: Keywords) -> Keywords {
        Keywords(self.0.union(other.0))
    }
}

impl fmt::Debug for Keywords {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}

impl serde::Serialize for Keywords {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.iter())
    }
}

/// Characteristic-defining power (CR 604.3): recomputed on every read.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum PowerCda {
    CountersOnLandsYouControl,
    /// "This creature's power is equal to the number of creatures you
    /// control." (Suki, Kyoshi Warrior.)
    CreaturesYouControl,
    /// "This creature's power is equal to the number of [predicate] cards
    /// in your graveyard." (Dragonfly Swarm: noncreature, nonland.)
    GraveyardMatching(CardPredicate),
}

/// Which battlefield permanents a static P/T buff applies to.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum StaticScope {
    /// Every matching permanent, regardless of controller.
    All(CardPredicate),
    /// Every matching permanent except the source, regardless of controller.
    Other(CardPredicate),
    /// The source permanent itself ("This creature gets +1/+1 as long as
    /// ..." — First-Time Flyer).
    This,
    /// "Other [predicate] you control" (anthem — White Lotus
    /// Reinforcements). Predicates read the current layer context.
    OtherYouControl(CardPredicate),
    /// "[predicate] you control", including the source (Air Nomad Legacy's
    /// "creatures you control with flying", Badgermole's "with +1/+1
    /// counters on them").
    YouControl(CardPredicate),
    /// "Equipped creature" / "enchanted creature": the exact creature the
    /// source Equipment or Aura is attached to.
    Attached,
    /// "[predicate] spells you control" (Lo and Li's Lesson spells): stack
    /// spells only, read by `Game::spell_keywords`; no permanent matches.
    SpellsYouControl(CardPredicate),
}

/// A static continuous P/T effect (CR 613.3c layer 7c): "[scope] get(s)
/// +P/+T [as long as condition]". Applies only while the source is on the
/// battlefield.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct StaticPtBuff {
    pub scope: StaticScope,
    pub condition: Option<Condition>,
    pub power: i32,
    pub toughness: i32,
    /// "+P/+T for each of its creature types, to a maximum of N" (Diligent
    /// Zookeeper): the bonus is multiplied by the affected creature's
    /// creature-type count, capped at this value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub per_creature_type_max: Option<i32>,
    /// "+P/+T for each color among [predicate] permanents you control"
    /// (Earthen Ally): the bonus is multiplied by that color count.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub per_color_among: Option<CardPredicate>,
    /// "+P/+T for each [predicate] card in your graveyard" (Avatar
    /// Destiny): the bonus is multiplied by that count.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub per_graveyard_matching: Option<CardPredicate>,
}

/// A static type-adding effect (CR 613.1d layer 4): "[scope] are [types] in
/// addition to their other types [as long as condition]" (Toph, the First
/// Metalbender; War Balloon's "as long as it has three or more fire counters
/// on it, it's an artifact creature"). It grants no abilities. Scopes read
/// the current layer context, and the condition is asked of the source
/// permanent. Dependency ordering accounts for changes to those reads.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct StaticTypeGrant {
    pub scope: StaticScope,
    pub types: Vec<CardType>,
    /// Subtypes added with them ("is an Avatar in addition to its other
    /// types").
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub subtypes: Vec<String>,
    /// "as long as ...", asked of the source's controller with the source as
    /// its permanent; absent for an unconditional grant.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condition: Option<Condition>,
}

/// "Enchanted creature loses all abilities and is a [creature_types] with base
/// power and toughness P/T named [named]" (Honest Work): one effect across
/// layers 1, 4, 6 and 7b (CR 613.1) on the exact attached creature. It
/// replaces the creature's creature types and names, keeps its card types,
/// supertypes and other subtypes, and leaves the source's own grants
/// (`attached_grants`) as its only activated abilities.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct SetCharacteristics {
    /// The name it sets: a rules value, unlike a definition's own `name`, so
    /// it reaches the learning tokens.
    pub named: String,
    pub creature_types: Vec<String>,
    pub power: i32,
    pub toughness: i32,
}

/// A static keyword grant (CR 613.1f layer 6): "[scope] have [keywords]
/// [as long as condition]". Applies only while the source is on the
/// battlefield. Scopes read the current layer context; dependency ordering
/// accounts for another effect changing their matches.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct StaticKeywordGrant {
    pub scope: StaticScope,
    pub condition: Option<Condition>,
    pub keywords: Keywords,
}

/// [`StaticAbility::SetsLandType`]: matching lands become `subtype` while
/// the condition holds for the source (Zhao, the Moon Slayer: "as long as
/// Zhao has a conqueror counter on it").
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct StaticLandType {
    pub predicate: CardPredicate,
    pub subtype: String,
    /// A source question (counters): asked with the source as its permanent.
    pub condition: Condition,
}

/// A static triggered-ability grant (CR 613.1f layer 6): "[scope] have
/// [ability]". Each covered permanent is the source of its own occurrences.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct StaticTriggeredGrant {
    pub scope: StaticScope,
    pub ability: Box<Ability>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct TargetCostReduction {
    pub predicate: CardPredicate,
    pub generic: u8,
}

/// A static rule-changing restriction: "[scope] can't attack or block",
/// "[scope] doesn't untap during its controller's untap step". Applies only
/// while the source is on the battlefield.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct StaticRestriction {
    pub scope: StaticScope,
    pub kind: Restriction,
    /// "... unless [condition]" for the source's controller (The Lion-Turtle).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unless: Option<Condition>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Restriction {
    /// CR 508.1c, 509.1b.
    CantAttackOrBlock,
    /// CR 502.3.
    DoesntUntap,
    /// "Can't be blocked by more than one creature" (CR 509.1b).
    CantBeBlockedByMoreThanOne,
}

/// Declarative replacement/prevention effects supported by the curated
/// proposed-event slice (CR 614-616).
///
/// The event pipeline rechecks applicability after each replacement and retains
/// each exact source/definition identity so it applies once per event.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum ReplacementEffect {
    /// Prevent `amount` damage that would be dealt to this permanent's
    /// controller or a permanent they control.
    PreventDamageToController { amount: i32 },
    /// Double damage that would be dealt to this permanent's controller or a
    /// permanent they control. This driver makes ordering observable.
    DoubleDamageToController,
    /// This object enters the battlefield tapped (CR 614.1c).
    EntersTapped,
    /// This object enters tapped unless its entering controller already
    /// controls a permanent matching `predicate` (CR 614.1c, 614.12).
    EntersTappedUnlessControls { predicate: CardPredicate },
    /// This object enters with +1/+1 counters already present (CR 614.1c),
    /// if `condition` holds for its entering controller as it enters
    /// ("enters with a +1/+1 counter on it if you attacked this turn").
    EntersWithPlusOneCounters {
        count: i32,
        #[serde(skip_serializing_if = "Option::is_none")]
        condition: Option<Condition>,
    },
    /// This object enters with X counters of `kind`, X being the value
    /// announced as it was cast, otherwise 0 (CR 107.3m, 614.1c).
    EntersWithXCounters { kind: CounterKind },
    /// Damage a source this permanent's controller controls would deal to
    /// an opponent or a permanent an opponent controls is increased by the
    /// number of `kind` counters on this permanent (CR 614.1a).
    OpponentDamagePlusCounters { kind: CounterKind },
    /// While this permanent is on the battlefield, unspent mana its
    /// controller would lose as a step or phase ends becomes red instead
    /// (CR 106.4, 614.1a). Read by `Game::clear_mana_pools`.
    UnspentManaBecomesRed,
    /// Replace a positive placement on a matching permanent you control with
    /// `count * multiplier + additional` counters. Entry counters are placements
    /// too (CR 122.6). `None` matches every counter kind.
    ModifyCounters {
        /// Only placements caused by effects, including replacement effects
        /// applied to an original cost or turn action (CR 614.16).
        effects_only: bool,
        kind: Option<CounterKind>,
        predicate: CardPredicate,
        multiplier: i32,
        additional: i32,
    },
    /// Replace a logical creation under your control, before any token enters
    /// (CR 616.1g). Entry and copy terms belong to every resulting token.
    MultiplyTokens { multiplier: i32 },
}

impl ActivatedAbilityDefinition {
    pub fn mana_choices(&self) -> Option<Vec<Mana>> {
        let [effect] = self.effects.as_slice() else {
            return None;
        };
        match effect {
            Effect::AddMana {
                mana,
                amount: EffectValue::Fixed(1),
                until_end_of_combat: false,
            } => Some(vec![mana.clone()]),
            Effect::Modal { modes } => modes
                .iter()
                .map(|effects| match effects.as_slice() {
                    [Effect::AddMana {
                        mana,
                        amount: EffectValue::Fixed(1),
                        until_end_of_combat: false,
                    }] => Some(mana.clone()),
                    _ => None,
                })
                .collect(),
            _ => None,
        }
    }

    /// The activation's targeting clause (CR 602.2b, 601.2c); activated
    /// abilities have at most one.
    pub fn target_requirement(&self) -> Option<TargetRequirement> {
        self.target_spec()?;
        effect_target_requirements(&self.effects).into_iter().next()
    }

    /// The activation cannot proceed without choosing a target.
    pub fn requires_target(&self) -> bool {
        self.target_requirement()
            .is_some_and(|requirement| requirement.min > 0)
    }

    pub fn target_spec(&self) -> Option<&TargetSpec> {
        self.effects.iter().find_map(Effect::target_spec)
    }
}

fn is_battlefield(zone: &ActivationZone) -> bool {
    *zone == ActivationZone::Battlefield
}

/// Where an activated ability can be activated from (CR 113.6, 602.1).
/// Abilities of cards outside the battlefield function only from the zone
/// their text names, or where their cost (discarding this card) requires.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub enum ActivationZone {
    #[default]
    Battlefield,
    Hand,
    Graveyard,
}

impl ActivationZone {
    pub fn zone(self) -> ZoneType {
        match self {
            Self::Battlefield => ZoneType::Battlefield,
            Self::Hand => ZoneType::Hand,
            Self::Graveyard => ZoneType::Graveyard,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct SacrificeCost {
    pub predicate: CardPredicate,
    pub another: bool,
    pub or_mana: Option<ManaCost>,
}

/// The exact object whose `attached_grants().nth(index)` an ability granted to
/// another permanent came from; the stack keeps it after the grantor leaves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct AbilityGrant {
    pub grantor: ObjectRef,
    pub index: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct ActivatedAbilityDefinition {
    /// Omitted for battlefield abilities so existing definition digests hold.
    #[serde(skip_serializing_if = "is_battlefield")]
    pub zone: ActivationZone,
    /// Discard this card as an activation cost (cycling, CR 702.29); only
    /// hand abilities carry it.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub discard_source: bool,
    /// Discard a chosen card from hand as an activation cost.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub discard_card: bool,
    /// "Tap an untapped [predicate] you control" as an activation cost; the
    /// source itself qualifies, and summoning sickness doesn't matter (not a
    /// {T} cost, CR 302.6).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tap_cost: Option<CardPredicate>,
    /// Exhaust: each object may activate this ability only once (CR 702.177a).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub exhaust: bool,
    /// "Activate only once each turn" (CR 602.5b); counted per object.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub once_each_turn: bool,
    /// "Costs {1} less to activate for each [predicate] you control."
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost_reduction_per: Option<CardPredicate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sacrifice: Option<SacrificeCost>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub own_turn_only: bool,
    /// "Activate only as a sorcery" (CR 602.5d).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub sorcery_timing: bool,
    #[serde(skip_serializing_if = "crate::is_zero")]
    pub life_cost: u32,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub tap_source: bool,
    pub mana_cost: ManaCost,
    /// Sacrifice the source permanent as an additional activation cost
    /// (e.g. Clue tokens' "{2}, Sacrifice this token: Draw a card.").
    pub sacrifice_source: bool,
    /// Waterbend cost (CR-style alternative payment): the generic part of
    /// `mana_cost` may be paid by tapping untapped artifacts/creatures the
    /// activating player controls, {1} per permanent tapped. Taps count as
    /// cost payment rather than a mana ability (CR 701.67).
    pub waterbend: bool,
    /// "Crew N": tap any number of other untapped creatures you control
    /// with total power N or greater (CR 702.122a).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub crew: Option<i32>,
    /// The least X the activation may announce ("X can't be 0", CR 107.3).
    #[serde(skip_serializing_if = "crate::is_zero")]
    pub x_min: u8,
    /// Set only on abilities `Characteristics::activated_abilities` lists
    /// from a grantor; printed abilities have none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub granted_by: Option<AbilityGrant>,
    /// Resolution effects in printed order ("draw a card, then discard a card").
    pub effects: Vec<Effect>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct SpellCostReduction {
    pub first_on_own_turn: bool,
    pub predicate: CardPredicate,
    pub condition: Condition,
    pub generic: u8,
}

/// A P/T amount read from the current layer context for an affected object.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum LayerValue {
    Fixed(i32),
    /// The affected object's mana value (Opalescence, March of the Machines).
    ManaValue,
    /// Distinct card types among cards in all graveyards (Tarmogoyf).
    GraveyardCardTypes {
        offset: i32,
    },
}

/// One static effect: once its first part applies, later layers retain its
/// affected objects even if its generating ability disappears (CR 613.6).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct StaticContinuous {
    pub scope: StaticScope,
    pub condition: Option<Condition>,
    pub characteristic_defining: bool,
    pub modifications: Vec<crate::effect::Modification<LayerValue>>,
}

/// A static ability (CR 604), interpreted by the rules engine.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum StaticAbility {
    /// One continuous effect whose parts may apply in different layers.
    Continuous(StaticContinuous),
    /// "During your turn, spells in your graveyard have flashback": a Lesson's
    /// flashback cost is {1}, any other spell's is its mana cost (Iroh, Grand
    /// Lotus).
    GraveyardSpellsHaveFlashback,
    /// Each matching permanent you control has "{T}: Add one mana of any
    /// color."
    GrantsAnyColorManaAbility(CardPredicate),
    /// "This creature can't be blocked as long as [condition]."
    CantBeBlockedIf(Condition),
    /// "Players can't gain life."
    PlayersCantGainLife,
    /// "This creature can't attack or block unless you control another
    /// [predicate]."
    CantAttackOrBlockUnlessYouControlAnother(CardPredicate),
    /// A blocking restriction on this creature (CR 509.1b).
    BlockRestriction(BlockRestriction),
    /// "[predicate] spells you cast cost {generic} less" while its condition
    /// holds.
    SpellsCostLess(SpellCostReduction),
    /// "This spell costs {1} less to cast for each [predicate] you control"
    /// (CR 601.2f); functions while casting, generic floor 0.
    CostsLessPerPermanent(CardPredicate),
    /// "This spell costs {1} less to cast for each [predicate] card in your
    /// graveyard"; functions while casting, generic floor 0.
    CostsLessPerGraveyardCard(CardPredicate),
    /// "If [condition], you may cast this spell as though it had flash";
    /// functions from hand (see `Game::has_instant_timing`).
    FlashIf(Condition),
    /// Printed flashback (CR 702.34): castable from its owner's graveyard for
    /// this cost, then exiled instead of going anywhere else.
    Flashback(ManaCost),
    /// Printed foretell (CR 702.143): during its owner's turn it may be exiled
    /// face down from hand for {2}, then cast on a later turn for this cost.
    Foretell(ManaCost),
    /// "You have no maximum hand size."
    NoMaximumHandSize,
    /// A continuous P/T effect (layer 7c).
    PtBuff(StaticPtBuff),
    /// Characteristic-defining power (CR 604.3, layer 7a); printed `power` is
    /// ignored.
    CharacteristicPower(PowerCda),
    /// Ward (CR 702.21): registration expands it into its "becomes the
    /// target" trigger, "counter it unless that player pays [cost]". The
    /// cost is mana or a waterbend ("Ward—Waterbend {4}").
    Ward(PaymentCost),
    /// "You may look at the top card of your library any time" (CR 401.2).
    LookAtLibraryTop,
    /// "You may cast [predicate] spells from the top of your library."
    CastFromLibraryTop(CardPredicate),
    /// "Equipped creature has \"...\"": an activated ability the creature
    /// this is attached to has, listed after its own in printed order.
    AttachedHas(ActivatedAbilityDefinition),
    /// Characteristics this Aura sets on the creature it enchants.
    AttachedSetsCharacteristics(SetCharacteristics),
    /// "This spell costs {N} less to cast if it targets a [predicate]"
    /// (Swampsnare Trap), read once targets are chosen (CR 601.2f).
    CostsLessIfTargets(TargetCostReduction),
    /// A keyword grant this card projects while on the battlefield.
    KeywordGrant(StaticKeywordGrant),
    /// An attack, block or untap restriction this card projects while on the
    /// battlefield.
    Restriction(StaticRestriction),
    /// A type grant this card projects while on the battlefield, on itself
    /// ("as long as this has three or more fire counters on it, it's an
    /// artifact creature") or on what its scope covers.
    TypeGrant(StaticTypeGrant),
    /// "This creature can attack as though it didn't have defender" while
    /// the condition holds for its controller (Platypus-Bear).
    AttacksDespiteDefender(Condition),
    /// "[predicate] permanents enter tapped" (Zhao, the Moon Slayer's
    /// nonbasic lands), for every player (CR 614.1c).
    EntersTapped(CardPredicate),
    /// "As long as this has [threshold], [predicate] lands are [subtype]"
    /// (CR 305.7): they lose all other land types and abilities and have
    /// that basic land type's mana ability.
    SetsLandType(StaticLandType),
    /// "If a triggered ability of a [predicate] you control triggers, that
    /// ability triggers an additional time" (CR 603.2d). Each such permanent
    /// adds one occurrence.
    TriggersAdditionalTime(CardPredicate),
    /// A triggered ability this card grants while on the battlefield
    /// ("Other Allies you control have prowess").
    TriggeredGrant(StaticTriggeredGrant),
    /// "Spells you cast cost [mana] less to cast" (CR 118.7c): each colored
    /// symbol reduces that color, or generic mana when the cost has none of
    /// it ("This can reduce generic costs").
    SpellsCostLessMana(ManaCost),
    /// "Untap this during each other player's untap step" (CR 502.3): it
    /// untaps in its controller's opponents' untap steps as well as theirs.
    UntapsDuringOtherUntapSteps,
}

/// Which blocks a creature's blocking restriction forbids (CR 509.1b).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum BlockRestriction {
    /// "This creature can't block [predicate] creatures."
    CantBlock(CardPredicate),
    /// "This creature can't be blocked by [predicate] creatures."
    CantBeBlockedBy(CardPredicate),
}

/// An additional cost of casting a spell (CR 601.2b, 601.2f).
// Definitions are built once per content pack, so the sacrifice predicate's
// size costs nothing; boxing it would only add noise to card authoring.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum AdditionalCost {
    /// "As an additional cost, sacrifice [predicate]", or pay `or_mana`
    /// (boxed: clippy's enum-size bound).
    Sacrifice(Box<SacrificeCost>),
    /// Kicker (CR 702.33): an optional cost chosen while casting.
    Kicker(ManaCost),
    /// "As an additional cost, you may waterbend [cost]": an optional cost
    /// chosen like kicker whose generic mana may be paid by tapping artifacts
    /// and creatures.
    Waterbend(ManaCost),
    /// "As an additional cost to cast this spell, pay [life] life or pay
    /// [mana]": chosen like kicker, the mana marks the spell kicked;
    /// otherwise the life is paid as it is cast (CR 119.4: only from a life
    /// total of at least that much).
    PayLifeOrMana { life: u32, mana: ManaCost },
    /// "As an additional cost to cast this spell, waterbend [cost]": required,
    /// and its {X} is announced with the spell's. It adds to the total cost,
    /// not the mana value (CR 202.3).
    RequiredWaterbend(ManaCost),
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct CardDefinition {
    /// `<set>.<card>`: the set file of the card's first printing and its slug.
    /// Identity metadata, not rules: excluded from content digests and tokens.
    #[serde(skip)]
    pub key: &'static str,
    /// Scryfall card-object id of the printing in `key`'s set; Oracle text,
    /// rulings and images are reachable from it. Play never reads it.
    #[serde(skip)]
    pub scryfall_id: &'static str,
    pub name: String,
    pub mana_cost: Option<ManaCost>,
    pub types: CardTypes,
    pub supertypes: Vec<String>,
    pub subtypes: Vec<String>,
    pub abilities: Vec<Ability>,
    /// Static abilities (CR 604), in printed order.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub statics: Vec<StaticAbility>,
    pub mana_abilities: Vec<ManaAbility>,
    pub triggered_mana_abilities: Vec<TriggeredManaAbility>,
    pub activated_abilities: Vec<ActivatedAbilityDefinition>,
    pub spell_effects: Vec<Effect>,
    /// Explicit targeting clauses for multi-target or effect-decoupled
    /// spells. When empty, a single requirement is derived from the first
    /// targeted spell effect.
    pub targeting: Vec<TargetRequirement>,
    /// Additional costs of casting this spell (CR 601.2b, 601.2f).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub additional_costs: Vec<AdditionalCost>,
    pub keywords: Keywords,
    /// Explicit color identity for cards without a mana cost (tokens).
    pub color_override: Option<Colors>,
    /// Tokens cease to exist outside the battlefield (CR 111.7, 704.5d).
    pub is_token: bool,
    /// Replacement/prevention definitions consumed by the proposed-event
    /// pipeline. Empty vectors are omitted so existing content-pack digests
    /// remain stable until production content actually adopts the feature.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub replacement_effects: Vec<ReplacementEffect>,
    pub text_box: String,
    pub power: Option<i32>,
    pub toughness: Option<i32>,
    /// The back face of a transforming double-faced card (CR 712.1): its own
    /// characteristics, registered by the content pack as its own row after
    /// this one. Serialized as its name, a reference to that row, so the
    /// front's digest and tokens name it without repeating it.
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_back_face"
    )]
    pub back_face: Option<Box<CardDefinition>>,
}

fn serialize_back_face<S: serde::Serializer>(
    face: &Option<Box<CardDefinition>>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    match face {
        Some(face) => serializer.serialize_some(&face.name),
        None => serializer.serialize_none(),
    }
}

impl CardDefinition {
    /// A Saga (CR 714.1): an enchantment with the Saga subtype.
    pub fn is_saga(&self) -> bool {
        self.types.is_enchantment() && self.subtypes.iter().any(|subtype| subtype == "Saga")
    }

    /// The optional additional cost a cast may choose to pay: kicker or an
    /// optional waterbend, which both mark the spell kicked.
    pub fn optional_cost(&self) -> Option<&ManaCost> {
        self.additional_costs.iter().find_map(|cost| match cost {
            AdditionalCost::Kicker(kicker)
            | AdditionalCost::Waterbend(kicker)
            | AdditionalCost::PayLifeOrMana { mana: kicker, .. } => Some(kicker),
            AdditionalCost::Sacrifice(_) | AdditionalCost::RequiredWaterbend(_) => None,
        })
    }

    /// The life an unkicked cast pays ("pay 5 life or pay {2}").
    pub fn life_alternative(&self) -> Option<u32> {
        self.additional_costs.iter().find_map(|cost| match cost {
            AdditionalCost::PayLifeOrMana { life, .. } => Some(*life),
            _ => None,
        })
    }

    pub fn sacrifice_cost(&self) -> Option<&SacrificeCost> {
        self.additional_costs.iter().find_map(|cost| match cost {
            AdditionalCost::Sacrifice(sacrifice) => Some(sacrifice.as_ref()),
            _ => None,
        })
    }

    /// The required "waterbend [cost]" additional cost, if printed.
    pub fn required_waterbend(&self) -> Option<&ManaCost> {
        self.additional_costs.iter().find_map(|cost| match cost {
            AdditionalCost::RequiredWaterbend(cost) => Some(cost),
            _ => None,
        })
    }

    pub fn flashback(&self) -> Option<&ManaCost> {
        self.statics.iter().find_map(|ability| match ability {
            StaticAbility::Flashback(cost) => Some(cost),
            _ => None,
        })
    }

    pub fn foretell(&self) -> Option<&ManaCost> {
        self.statics.iter().find_map(|ability| match ability {
            StaticAbility::Foretell(cost) => Some(cost),
            _ => None,
        })
    }

    pub fn flash_if(&self) -> Option<&Condition> {
        self.statics.iter().find_map(|ability| match ability {
            StaticAbility::FlashIf(condition) => Some(condition),
            _ => None,
        })
    }

    pub fn ward(&self) -> Option<&PaymentCost> {
        self.statics.iter().find_map(|ability| match ability {
            StaticAbility::Ward(cost) => Some(cost),
            _ => None,
        })
    }

    pub fn power_cda(&self) -> Option<&PowerCda> {
        self.statics.iter().find_map(|ability| match ability {
            StaticAbility::CharacteristicPower(cda) => Some(cda),
            _ => None,
        })
    }

    /// The activated abilities this grants the creature it is attached to,
    /// in printed order; `AbilityGrant::index` indexes this sequence.
    pub fn attached_grants(&self) -> impl Iterator<Item = &ActivatedAbilityDefinition> {
        self.statics.iter().filter_map(|ability| match ability {
            StaticAbility::AttachedHas(granted) => Some(granted),
            _ => None,
        })
    }

    pub fn attached_characteristics(&self) -> Option<&SetCharacteristics> {
        self.statics.iter().find_map(|ability| match ability {
            StaticAbility::AttachedSetsCharacteristics(set) => Some(set),
            _ => None,
        })
    }

    pub fn cost_reduction_if_targets(&self) -> Option<&TargetCostReduction> {
        self.statics.iter().find_map(|ability| match ability {
            StaticAbility::CostsLessIfTargets(reduction) => Some(reduction),
            _ => None,
        })
    }

    pub fn colors(&self) -> Colors {
        if let Some(colors) = &self.color_override {
            return colors.clone();
        }
        self.mana_cost
            .as_ref()
            .map(|m| m.colors())
            .unwrap_or_default()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Card {
    pub id: ObjectId,
    pub definition_id: CardDefId,
    /// Compatibility name for the existing observation ABI. This is always
    /// the numeric representation of `definition_id`.
    pub registry_key: ObjectId,
    pub owner: PlayerId,
    /// The copy effect that made this object a copy (CR 707.2); its
    /// definition is then the copiable values that effect produced.
    pub copy: Option<crate::effect::CopyEffect>,
    definition: Arc<CardDefinition>,
    /// The front face's row while the back face is up (CR 712.8): set by
    /// [`Card::turn_back_face_up`], cleared by [`Card::turn_front_face_up`].
    front: Option<(CardDefId, Arc<CardDefinition>)>,
}

impl Card {
    pub fn from_definition(
        id: ObjectId,
        owner: PlayerId,
        definition_id: CardDefId,
        definition: Arc<CardDefinition>,
    ) -> Self {
        Self {
            id,
            definition_id,
            registry_key: ObjectId(definition_id.0),
            owner,
            copy: None,
            definition,
            front: None,
        }
    }

    /// Its mana cost for mana value: a transformed permanent's is its front
    /// face's (CR 712.8e).
    pub fn mana_value_cost(&self) -> Option<&ManaCost> {
        match &self.front {
            Some((_, front)) => front.mana_cost.as_ref(),
            None => self.mana_cost.as_ref(),
        }
    }

    /// Whether its back face is up (CR 712.8).
    pub fn is_transformed(&self) -> bool {
        self.front.is_some()
    }

    /// Show the back face, a row of the engine catalog’s `ContentPack::back_face`.
    pub fn turn_back_face_up(&mut self, (id, definition): (CardDefId, Arc<CardDefinition>)) {
        let front = (self.definition_id, Arc::clone(&self.definition));
        self.show(id, definition);
        self.front = Some(front);
    }

    /// Show the front face, if the back face is up.
    pub fn turn_front_face_up(&mut self) {
        if let Some((id, definition)) = self.front.take() {
            self.show(id, definition);
        }
    }

    /// Show one face. A token is a double-faced token on either face, and a
    /// copy's exceptions are in the copiable values of each (CR 707.8a, 707.9b).
    fn show(&mut self, id: CardDefId, definition: Arc<CardDefinition>) {
        let token = self.is_token;
        self.definition_id = id;
        self.registry_key = ObjectId(id.0);
        self.definition = definition;
        if token && !self.is_token {
            self.is_token = true;
        }
        if let Some(mut copy) = self.copy.take() {
            copy.of = id;
            for exception in &copy.exceptions {
                exception.apply_copiable(self);
            }
            self.copy = Some(copy);
        }
    }

    /// Snapshot the current copiable face without copying its ability programs.
    pub fn copiable_definition(&self) -> Arc<CardDefinition> {
        Arc::clone(&self.definition)
    }

    pub fn shares_definition_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.definition, &other.definition)
    }

    pub fn has_subtype(&self, subtype: &str) -> bool {
        self.subtypes.iter().any(|s| s == subtype)
    }

    pub fn is_legendary(&self) -> bool {
        self.supertypes
            .iter()
            .any(|kind| kind.eq_ignore_ascii_case("legendary"))
    }

    /// CR 117.1a, 702.8 — Instants and cards with flash use instant timing.
    /// Printed instant timing; a conditional `StaticAbility::FlashIf` is checked by
    /// `Game::has_instant_timing`.
    pub fn is_instant_speed(&self) -> bool {
        self.types.is_instant() || self.keywords.has(Keyword::Flash)
    }

    /// Modes are announced before targets (CR 601.2b-c). Only spell-level
    /// modal effects participate; choices inside resolving abilities do not.
    pub fn spell_modes(&self) -> Option<&[Vec<Effect>]> {
        self.spell_effects.iter().find_map(|effect| match effect {
            Effect::Modal { modes } => Some(modes.as_slice()),
            _ => None,
        })
    }

    /// An Aura's enchant restriction (CR 303.4a, 702.5c): what its spell
    /// targets and what its attached object must keep matching (CR 704.5m).
    pub fn enchant(&self) -> Option<&TargetSpec> {
        if !self.has_subtype("Aura") {
            return None;
        }
        self.spell_effects.iter().find_map(|effect| match effect {
            Effect::AttachSourceTo { what } => what.target_spec(),
            _ => None,
        })
    }

    pub fn spell_effects_for_mode(&self, mode: Option<usize>) -> Vec<Effect> {
        self.spell_effects
            .iter()
            .flat_map(|effect| match (effect, mode) {
                (Effect::Modal { modes }, Some(index)) => {
                    modes.get(index).map(Vec::as_slice).unwrap_or_default()
                }
                _ => std::slice::from_ref(effect),
            })
            .cloned()
            .collect()
    }

    pub fn target_requirements_for_mode(&self, mode: Option<usize>) -> Vec<TargetRequirement> {
        if !self.targeting.is_empty() {
            return self.targeting.clone();
        }
        crate::ability::effect_target_requirements(&self.spell_effects_for_mode(mode))
    }

    /// Explicit clauses bind distinct targets; repeated effects on "that"
    /// target share the inferred single clause.
    pub fn target_requirements(&self) -> Vec<TargetRequirement> {
        self.target_requirements_for_mode(None)
    }
}

impl std::ops::Deref for Card {
    type Target = CardDefinition;

    fn deref(&self) -> &Self::Target {
        &self.definition
    }
}

/// Preserve the existing Rust scenario-test seam without making ordinary
/// match clones copy definitions. Mutating printed characteristics detaches
/// only that physical card from the shared pack definition.
impl std::ops::DerefMut for Card {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::make_mut(&mut self.definition)
    }
}

impl std::fmt::Display for Card {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{name: {}}}", self.name)
    }
}

pub fn basic_land(name: &str, color: Color) -> CardDefinition {
    CardDefinition {
        name: name.to_string(),
        types: CardTypes::new([CardType::Land]),
        supertypes: vec!["basic".to_string()],
        subtypes: vec![name.to_string()],
        abilities: vec![],
        mana_abilities: vec![ManaAbility {
            mana: Mana::single(color),
        }],
        text_box: format!("{{T}}: Add {{{}}}.", color.symbol()),
        ..Default::default()
    }
}

#[cfg(test)]
mod keyword_tests {
    use super::{Keyword, KeywordBits, Keywords};

    #[test]
    fn keyword_storage_keeps_word_boundaries_independent() {
        // Ordinals exercise the production storage without declaring fictional
        // supported mechanics. Cover both sides of two word boundaries.
        let mut original = KeywordBits::<3>::default();
        let present = [0, 63, 64, 65, 127, 128, 191];
        for &index in present.iter().rev() {
            original.insert(index);
            original.insert(index);
        }
        for index in 0..192 {
            assert_eq!(original.has(index), present.contains(&index), "{index}");
        }

        // Keywords is Copy: a fork owns its words, including all later words.
        let mut fork = original;
        fork.insert(129);
        let mut removed = KeywordBits::default();
        for index in [0, 64, 128, 190] {
            removed.insert(index);
        }
        fork = fork.without(removed);
        let remaining = [63, 65, 127, 129, 191];
        let reunited = fork.union(removed);
        for index in 0..192 {
            assert_eq!(fork.has(index), remaining.contains(&index), "{index}");
            assert_eq!(original.has(index), present.contains(&index), "{index}");
            assert_eq!(
                reunited.has(index),
                present.contains(&index) || [129, 190].contains(&index),
                "{index}"
            );
        }
        assert!(original.without(original) == KeywordBits::default());
        assert!(original.union(original) == original);
    }

    #[test]
    fn keyword_sets_preserve_membership_order_and_copy_independence() {
        let original = Keywords::of(&[Keyword::Indestructible, Keyword::Flying, Keyword::Flying]);
        let mut fork = original;
        fork.insert(Keyword::Haste);
        fork = fork.without(Keywords::of(&[Keyword::Flying, Keyword::Reach]));
        assert_eq!(original.names(), ["flying", "indestructible"]);
        assert_eq!(fork.names(), ["haste", "indestructible"]);
        assert_eq!(
            fork.union(original).names(),
            ["flying", "haste", "indestructible"]
        );
        let all = Keywords::of(&Keyword::ALL);
        for keyword in Keyword::ALL {
            assert!(all.has(keyword));
            assert!(!Keywords::default().has(keyword));
        }
    }
}
