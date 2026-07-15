use std::{collections::BTreeSet, sync::Arc};

use super::{
    ability::{Ability, Effect, StaticCondition, TargetRequirement},
    game_object::{ObjectId, PlayerId},
    mana::{Color, Colors, Mana, ManaCost},
    predicate::CardPredicate,
};

/// Stable index of an immutable card definition in a [`ContentPack`](crate::cardsets::alpha::ContentPack).
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

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
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
}

impl Keywords {
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
        }
    }
}

/// Characteristic-defining power (CR 604.3): recomputed on every read.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum PowerCda {
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
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ActivatedAbilityDefinition {
    pub mana_cost: ManaCost,
    /// Sacrifice the source permanent as an additional activation cost
    /// (e.g. Clue tokens' "{2}, Sacrifice this token: Draw a card.").
    pub sacrifice_source: bool,
    /// Waterbend cost (CR-style alternative payment): the generic part of
    /// `mana_cost` may be paid by tapping untapped artifacts/creatures the
    /// activating player controls, {1} per permanent tapped. Taps count as
    /// "tapped for mana" (triggered mana abilities compose).
    pub waterbend: bool,
    pub effect: Effect,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct CardDefinition {
    pub name: String,
    pub mana_cost: Option<ManaCost>,
    pub types: CardTypes,
    pub supertypes: Vec<String>,
    pub subtypes: Vec<String>,
    pub abilities: Vec<Ability>,
    pub mana_abilities: Vec<ManaAbility>,
    pub triggered_mana_abilities: Vec<TriggeredManaAbility>,
    pub activated_abilities: Vec<ActivatedAbilityDefinition>,
    pub spell_effects: Vec<Effect>,
    /// Explicit targeting clauses for multi-target or effect-decoupled
    /// spells. When empty, a single requirement is derived from the first
    /// targeted spell effect.
    pub targeting: Vec<TargetRequirement>,
    /// Kicker (CR 702.33): optional additional cost chosen while casting.
    pub kicker: Option<ManaCost>,
    /// Ward (CR 702.21): "counter [the targeting spell] unless its
    /// controller pays [cost]". Synthesized into a BecomesTargeted
    /// triggered ability at registration.
    pub ward: Option<ManaCost>,
    /// Affinity-style cost reduction: "This spell costs {1} less to cast
    /// for each [predicate] you control", computed while casting (generic
    /// floor 0).
    pub cost_reduction_per: Option<CardPredicate>,
    pub keywords: Keywords,
    /// "This creature can't be blocked by creatures matching [predicate]."
    pub block_restriction: Option<CardPredicate>,
    /// Explicit color identity for cards without a mana cost (tokens).
    pub color_override: Option<Colors>,
    /// Tokens cease to exist outside the battlefield (CR 111.7, 704.5d).
    pub is_token: bool,
    /// Characteristic-defining power (`power` is ignored when set).
    pub power_cda: Option<PowerCda>,
    /// Static continuous P/T effects this card projects while on the
    /// battlefield (anthems, conditional self-buffs).
    pub static_pt_buffs: Vec<StaticPtBuff>,
    pub text_box: String,
    pub power: Option<i32>,
    pub toughness: Option<i32>,
}

impl CardDefinition {
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
    pub fn is_instant_speed(&self) -> bool {
        self.types.is_instant() || self.keywords.flash
    }

    /// The spell's targeting clauses, in choice order. Explicit `targeting`
    /// wins; otherwise a single 1-of-1 requirement is derived from the
    /// first targeted spell effect (the Stage-1 convention).
    pub fn target_requirements(&self) -> Vec<TargetRequirement> {
        if !self.targeting.is_empty() {
            return self.targeting.clone();
        }
        self.spell_effects
            .iter()
            .find_map(|effect| effect.target_spec())
            .map(|spec| vec![TargetRequirement::one(spec.clone())])
            .unwrap_or_default()
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
