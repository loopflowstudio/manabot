use std::{collections::BTreeSet, sync::Arc};

use super::{
    ability::{
        effect_target_requirements, Ability, Effect, StaticCondition, TargetRequirement, TargetSpec,
    },
    game_object::{ObjectId, ObjectRef, PlayerId},
    mana::{Color, Colors, Mana, ManaCost},
    permanent::CounterKind,
    predicate::CardPredicate,
    zone::ZoneType,
};

/// Stable index of an immutable card definition in a [`ContentPack`](crate::cards::content_pack::ContentPack).
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

    pub fn is_castable(&self) -> bool {
        !self.is_land() && !self.types.is_empty()
    }

    pub fn is_permanent(&self) -> bool {
        self.is_creature()
            || self.is_land()
            || self.is_artifact()
            || self.is_enchantment()
            || self.is_planeswalker()
            || self.is_battle()
    }

    pub fn is_non_land_permanent(&self) -> bool {
        self.is_permanent() && !self.is_land()
    }

    pub fn is_non_creature_permanent(&self) -> bool {
        self.is_permanent() && !self.is_creature()
    }

    pub fn is_spell(&self) -> bool {
        self.types.contains(&CardType::Instant) || self.types.contains(&CardType::Sorcery)
    }

    pub fn is_instant(&self) -> bool {
        self.types.contains(&CardType::Instant)
    }

    pub fn is_creature(&self) -> bool {
        self.types.contains(&CardType::Creature)
    }

    pub fn is_land(&self) -> bool {
        self.types.contains(&CardType::Land)
    }

    pub fn is_planeswalker(&self) -> bool {
        self.types.contains(&CardType::Planeswalker)
    }

    pub fn is_enchantment(&self) -> bool {
        self.types.contains(&CardType::Enchantment)
    }

    pub fn is_artifact(&self) -> bool {
        self.types.contains(&CardType::Artifact)
    }

    pub fn is_kindred(&self) -> bool {
        self.types.contains(&CardType::Kindred)
    }

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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct Keywords {
    pub flying: bool,
    pub reach: bool,
    pub haste: bool,
    /// CR 702.8 — May be cast any time the controller could cast an instant.
    pub flash: bool,
    pub vigilance: bool,
    pub trample: bool,
    pub first_strike: bool,
    pub double_strike: bool,
    pub deathtouch: bool,
    pub lifelink: bool,
    pub defender: bool,
    pub menace: bool,
    /// CR 702.11 — Can't be the target of spells or abilities opponents
    /// control.
    pub hexproof: bool,
    pub indestructible: bool,
}

impl Keywords {
    /// The names of the keywords present (field names, as the learning
    /// tokens name them), in declaration order.
    pub fn names(&self) -> Vec<&'static str> {
        [
            ("flying", self.flying),
            ("reach", self.reach),
            ("haste", self.haste),
            ("flash", self.flash),
            ("vigilance", self.vigilance),
            ("trample", self.trample),
            ("first_strike", self.first_strike),
            ("double_strike", self.double_strike),
            ("deathtouch", self.deathtouch),
            ("lifelink", self.lifelink),
            ("defender", self.defender),
            ("menace", self.menace),
            ("hexproof", self.hexproof),
            ("indestructible", self.indestructible),
        ]
        .into_iter()
        .filter_map(|(name, present)| present.then_some(name))
        .collect()
    }

    /// Keyword-set union (printed keywords plus until-EOT grants).
    pub fn union(&self, other: &Keywords) -> Keywords {
        Keywords {
            flying: self.flying || other.flying,
            reach: self.reach || other.reach,
            haste: self.haste || other.haste,
            flash: self.flash || other.flash,
            vigilance: self.vigilance || other.vigilance,
            trample: self.trample || other.trample,
            first_strike: self.first_strike || other.first_strike,
            double_strike: self.double_strike || other.double_strike,
            deathtouch: self.deathtouch || other.deathtouch,
            lifelink: self.lifelink || other.lifelink,
            defender: self.defender || other.defender,
            menace: self.menace || other.menace,
            hexproof: self.hexproof || other.hexproof,
            indestructible: self.indestructible || other.indestructible,
        }
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
    /// The source permanent itself ("This creature gets +1/+1 as long as
    /// ..." — First-Time Flyer).
    This,
    /// "Other [predicate] you control" (anthem — White Lotus
    /// Reinforcements). Matched against printed characteristics (no
    /// power predicates, to keep P/T computation non-recursive).
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
    pub condition: Option<StaticCondition>,
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
}

/// A static type-adding effect (CR 613.1d layer 4): "[scope] are [types] in
/// addition to their other types" (Toph, the First Metalbender). It grants no
/// abilities. Scopes read printed types only, so type computation never
/// recurses.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct StaticTypeGrant {
    pub scope: StaticScope,
    pub types: Vec<CardType>,
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
/// battlefield. Scopes read printed characteristics, counters and
/// non-static keywords, so keyword computation never recurses.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct StaticKeywordGrant {
    pub scope: StaticScope,
    pub condition: Option<StaticCondition>,
    pub keywords: Keywords,
}

/// "As long as this has N or more [kind] counters on it" (War Balloon).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct CounterThreshold {
    pub kind: CounterKind,
    pub count: i32,
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
    pub unless: Option<StaticCondition>,
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
/// This is intentionally not a general replacement language. The event
/// pipeline applies matching definitions once in deterministic source order;
/// unsupported affected-player ordering choices are documented alongside the
/// pipeline.
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
    /// This object enters with +1/+1 counters already present (CR 614.1c).
    EntersWithPlusOneCounters { count: i32 },
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
}

impl ActivatedAbilityDefinition {
    pub fn mana_choices(&self) -> Option<Vec<Mana>> {
        let [effect] = self.effects.as_slice() else {
            return None;
        };
        match effect {
            Effect::AddMana {
                mana,
                until_end_of_combat: false,
            } => Some(vec![mana.clone()]),
            Effect::Modal { modes } => modes
                .iter()
                .map(|effects| match effects.as_slice() {
                    [Effect::AddMana {
                        mana,
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
    #[serde(skip_serializing_if = "crate::state::is_zero")]
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
    #[serde(skip_serializing_if = "crate::state::is_zero")]
    pub x_min: u8,
    /// Set only on abilities `Game::activated_abilities_for` lists from a
    /// grantor; printed abilities have none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub granted_by: Option<AbilityGrant>,
    /// Resolution effects in printed order ("draw a card, then discard a card").
    pub effects: Vec<Effect>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct SpellCostReduction {
    pub first_on_own_turn: bool,
    pub predicate: CardPredicate,
    pub condition: StaticCondition,
    pub generic: u8,
}

/// A static ability (CR 604). The engine applies each by matching its
/// variant; unless noted, it functions while its card is on the battlefield.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum StaticAbility {
    /// "During your turn, spells in your graveyard have flashback": a Lesson's
    /// flashback cost is {1}, any other spell's is its mana cost (Iroh, Grand
    /// Lotus).
    GraveyardSpellsHaveFlashback,
    /// Each matching permanent you control has "{T}: Add one mana of any
    /// color."
    GrantsAnyColorManaAbility(CardPredicate),
    /// "This creature can't be blocked as long as [condition]."
    CantBeBlockedIf(StaticCondition),
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
    FlashIf(StaticCondition),
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
    /// target" trigger, "counter it unless that player pays [cost]".
    Ward(ManaCost),
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
    /// "As long as this has N or more [kind] counters on it, it's an artifact
    /// creature" (CR 613.1d), read by `Permanent::animated_creature`.
    CreatureWithCounters(CounterThreshold),
    /// A type grant this card projects while on the battlefield.
    TypeGrant(StaticTypeGrant),
    /// "This creature can attack as though it didn't have defender" while
    /// the condition holds for its controller (Platypus-Bear).
    AttacksDespiteDefender(StaticCondition),
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
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum AdditionalCost {
    /// "As an additional cost, sacrifice [predicate]", or pay `or_mana`.
    Sacrifice(SacrificeCost),
    /// Kicker (CR 702.33): an optional cost chosen while casting.
    Kicker(ManaCost),
    /// "As an additional cost, you may waterbend [cost]": an optional cost
    /// chosen like kicker whose generic mana may be paid by tapping artifacts
    /// and creatures.
    Waterbend(ManaCost),
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
}

impl CardDefinition {
    /// The optional additional cost a cast may choose to pay: kicker or an
    /// optional waterbend, which both mark the spell kicked.
    pub fn optional_cost(&self) -> Option<&ManaCost> {
        self.additional_costs.iter().find_map(|cost| match cost {
            AdditionalCost::Kicker(kicker) | AdditionalCost::Waterbend(kicker) => Some(kicker),
            AdditionalCost::Sacrifice(_) | AdditionalCost::RequiredWaterbend(_) => None,
        })
    }

    pub fn sacrifice_cost(&self) -> Option<&SacrificeCost> {
        self.additional_costs.iter().find_map(|cost| match cost {
            AdditionalCost::Sacrifice(sacrifice) => Some(sacrifice),
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

    pub fn flash_if(&self) -> Option<&StaticCondition> {
        self.statics.iter().find_map(|ability| match ability {
            StaticAbility::FlashIf(condition) => Some(condition),
            _ => None,
        })
    }

    pub fn ward(&self) -> Option<&ManaCost> {
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

    pub fn library_top_casting(&self) -> Option<&CardPredicate> {
        self.statics.iter().find_map(|ability| match ability {
            StaticAbility::CastFromLibraryTop(predicate) => Some(predicate),
            _ => None,
        })
    }

    pub fn cost_reduction_if_targets(&self) -> Option<&TargetCostReduction> {
        self.statics.iter().find_map(|ability| match ability {
            StaticAbility::CostsLessIfTargets(reduction) => Some(reduction),
            _ => None,
        })
    }

    pub fn creature_with_counters(&self) -> Option<&CounterThreshold> {
        self.statics.iter().find_map(|ability| match ability {
            StaticAbility::CreatureWithCounters(threshold) => Some(threshold),
            _ => None,
        })
    }

    pub fn attacks_despite_defender(&self) -> Option<&StaticCondition> {
        self.statics.iter().find_map(|ability| match ability {
            StaticAbility::AttacksDespiteDefender(condition) => Some(condition),
            _ => None,
        })
    }

    pub fn pt_buffs(&self) -> impl Iterator<Item = &StaticPtBuff> {
        self.statics.iter().filter_map(|ability| match ability {
            StaticAbility::PtBuff(buff) => Some(buff),
            _ => None,
        })
    }

    pub fn keyword_grants(&self) -> impl Iterator<Item = &StaticKeywordGrant> {
        self.statics.iter().filter_map(|ability| match ability {
            StaticAbility::KeywordGrant(grant) => Some(grant),
            _ => None,
        })
    }

    pub fn restrictions(&self) -> impl Iterator<Item = &StaticRestriction> {
        self.statics.iter().filter_map(|ability| match ability {
            StaticAbility::Restriction(restriction) => Some(restriction),
            _ => None,
        })
    }

    pub fn type_grants(&self) -> impl Iterator<Item = &StaticTypeGrant> {
        self.statics.iter().filter_map(|ability| match ability {
            StaticAbility::TypeGrant(grant) => Some(grant),
            _ => None,
        })
    }

    pub fn has_static(&self, ability: &StaticAbility) -> bool {
        self.statics.contains(ability)
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
    definition: Arc<CardDefinition>,
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
            definition,
        }
    }

    pub fn shares_definition_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.definition, &other.definition)
    }

    pub fn has_subtype(&self, subtype: &str) -> bool {
        self.subtypes.iter().any(|s| s == subtype)
    }

    /// CR 117.1a, 702.8 — Instants and cards with flash use instant timing.
    /// Printed instant timing; a conditional `StaticAbility::FlashIf` is checked by
    /// [`crate::flow::game::Game::has_instant_timing`].
    pub fn is_instant_speed(&self) -> bool {
        self.types.is_instant() || self.keywords.flash
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
            Effect::AttachSourceTo { target } => Some(target),
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
        crate::state::ability::effect_target_requirements(&self.spell_effects_for_mode(mode))
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
