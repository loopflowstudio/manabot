use std::{collections::BTreeSet, sync::Arc};

use super::{
    ability::{
        effect_target_requirements, Ability, Effect, StaticCondition, TargetRequirement, TargetSpec,
    },
    game_object::{ObjectId, PlayerId},
    mana::{Color, Colors, Mana, ManaCost},
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
    /// This object enters with +1/+1 counters already present (CR 614.1c).
    EntersWithPlusOneCounters { count: i32 },
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

    /// Whether the first target chosen on activation (CR 602.2b, 601.2c) is
    /// required (`Some(true)`) or "up to one" (`Some(false)`); `None` when the
    /// ability has no targets.
    fn first_target_required(&self) -> Option<bool> {
        effect_target_requirements(&self.effects)
            .first()
            .map(|requirement| requirement.min > 0)
    }

    /// The activation cannot proceed without choosing a target.
    pub fn requires_target(&self) -> bool {
        self.first_target_required() == Some(true)
    }

    /// The activation may decline its target ("up to one").
    pub fn target_is_optional(&self) -> bool {
        self.first_target_required() == Some(false)
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
    /// cost payment rather than a mana ability (CR 702.177).
    pub waterbend: bool,
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
    /// During your turn, grant graveyard flashback (Lessons cost {1}).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub grants_flashback: bool,
    /// "If [condition], you may cast this spell as though it had flash."
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flash_if: Option<StaticCondition>,
    /// `cost_reduction_per` counts cards in your graveyard, not permanents
    /// you control.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub cost_reduction_in_graveyard: bool,
    /// Each matching permanent you control gains a chosen-color tap ability.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub mana_grant: Vec<CardPredicate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unblockable_if: Option<StaticCondition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sacrifice_cost: Option<SacrificeCost>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub prevents_life_gain: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub combat_requires_other: Option<CardPredicate>,
    /// This creature cannot block attackers matching this predicate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocking_restriction: Option<CardPredicate>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub spell_cost_reductions: Vec<SpellCostReduction>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub no_maximum_hand_size: bool,
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
    /// The optional additional cost is "waterbend [kicker]": its generic
    /// mana may be paid by tapping artifacts and creatures.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub kicker_waterbend: bool,
    /// Printed flashback (CR 702.34): castable from its owner's graveyard
    /// for this cost, then exiled instead of going anywhere else.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flashback: Option<ManaCost>,
    /// Printed foretell (CR 702.143): during its owner's turn it may be
    /// exiled face down from hand for {2}, then cast on a later turn for
    /// this cost.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub foretell: Option<ManaCost>,
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
    /// Printed instant timing; a conditional `flash_if` is checked by
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
