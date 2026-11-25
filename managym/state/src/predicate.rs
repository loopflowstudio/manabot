// predicate.rs
// Reusable structural predicates over cards, shared by trigger subjects,
// blocking restrictions, and zone counting (no scattered string compares).

use super::card::{CardType, Keyword, PrintedCharacteristics};

use super::game_object::ObjectKind;
use super::mana::Color;

/// A structural predicate over cards. Every populated field must match.
/// Deserialization reads a definition's serialized predicate back, so
/// observation vocabularies can enumerate what the encoders describe.
#[derive(
    Clone,
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(default)]
pub struct CardPredicate {
    /// Shares the asking source permanent's layered name. Requires a source;
    /// card-only matching cannot answer it.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub namesake: bool,
    /// "named [name]": the object's name as it is (or last was) equals this
    /// one (CR 201.2). An empty name names nothing (CR 201.2a).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub named: Option<String>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub flying: bool,
    /// "without flying", as the object's characteristics are now.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub not_flying: bool,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub deathtouch: bool,
    /// "with no counters on it". Only battlefield permanents carry counters,
    /// so card-only matching ignores it.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub no_counters: bool,
    /// An attacking or blocking creature (CR 506.4); only battlefield
    /// permanents are in combat, so card-only matching ignores it.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub in_combat: bool,
    /// A token (`Some(true)`) or a nontoken (`Some(false)`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token: Option<bool>,
    /// "with a +1/+1 counter on it": only permanents carry counters.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub with_counter: bool,
    /// "attacking": all currently attacking permanents, including those put in attacking.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub attacking: bool,
    /// "tapped": only permanents are tapped.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub tapped: bool,
    /// "card in a graveyard that was put there from the battlefield this
    /// turn" (Brought Back): game history, so card-only matching refuses it
    /// and `Game::card_matches_predicate` answers it.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub from_battlefield_this_turn: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub subtypes_any: Vec<String>,
    /// Has at least one of these colors ("white or blue"), as its
    /// characteristics are now (layer 5).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub colors_any: Vec<Color>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub not_subtypes: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supertype: Option<String>,
    /// "nonbasic": the card must not have the basic supertype.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub nonbasic: bool,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub nonlegendary: bool,
    /// At least one alternative must match ("artifact or Lesson").
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub any_of: Vec<CardPredicate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_power: Option<i32>,
    /// "with toughness N or greater", checked against effective toughness
    /// on the battlefield.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_toughness: Option<u8>,
    /// Card must have this card type.
    pub card_type: Option<CardType>,
    /// Card must have at least one of these card types ("artifact,
    /// creature, or enchantment").
    pub card_types_any: Vec<CardType>,
    /// Card must have none of these card types ("noncreature, nonland").
    pub not_card_types: Vec<CardType>,
    /// Card must have this subtype (e.g. "Ally", "Lesson").
    pub subtype: Option<String>,
    /// Card's power must be `max_power` or less ("power N or less").
    /// Checked against effective power when a permanent is in play.
    pub max_power: Option<i32>,
    /// Card's mana value must be `min_mana_value` or greater ("with mana
    /// value 3 or greater"). Cards without a mana cost have mana value 0.
    pub min_mana_value: Option<u8>,
    /// "with mana value N or less".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_mana_value: Option<u8>,
}

impl CardPredicate {
    pub fn creature() -> Self {
        Self {
            card_type: Some(CardType::Creature),
            ..Self::default()
        }
    }

    pub fn permanent() -> Self {
        Self {
            card_types_any: vec![
                CardType::Artifact,
                CardType::Battle,
                CardType::Creature,
                CardType::Enchantment,
                CardType::Land,
                CardType::Planeswalker,
            ],
            ..Self::default()
        }
    }

    pub fn creature_or_planeswalker() -> Self {
        Self {
            card_types_any: vec![CardType::Creature, CardType::Planeswalker],
            ..Default::default()
        }
    }

    pub fn instant_or_sorcery() -> Self {
        Self {
            card_types_any: vec![CardType::Instant, CardType::Sorcery],
            ..Self::default()
        }
    }

    pub fn noncreature() -> Self {
        Self {
            not_card_types: vec![CardType::Creature],
            ..Self::default()
        }
    }

    pub fn basic_land() -> Self {
        Self {
            card_type: Some(CardType::Land),
            supertype: Some("basic".into()),
            ..Self::default()
        }
    }

    pub fn subtype(subtype: &str) -> Self {
        Self {
            subtype: Some(subtype.to_string()),
            ..Self::default()
        }
    }

    /// Whether matching needs the asking ability's source permanent.
    pub fn reads_source(&self) -> bool {
        self.namesake || self.any_of.iter().any(CardPredicate::reads_source)
    }

    /// Whether matching compares power, here or in an alternative.
    pub fn reads_power(&self) -> bool {
        self.min_power.is_some()
            || self.max_power.is_some()
            || self.any_of.iter().any(CardPredicate::reads_power)
    }

    /// Whether an object named `name` satisfies `named` (CR 201.2).
    pub fn accepts_name(&self, name: &str) -> bool {
        self.named
            .as_deref()
            .is_none_or(|named| !named.is_empty() && named == name)
    }

    /// Match the selected spell's printed characteristics before it goes on the stack.
    pub fn matches_printed_spell(&self, spell: &PrintedCharacteristics, kind: ObjectKind) -> bool {
        self.matches_printing(
            spell,
            kind,
            spell.mana_cost.as_ref().map_or(0, |cost| cost.mana_value),
            spell.power.unwrap_or(0),
            &[],
        )
    }

    /// Structural matching over retained printing provenance, without looking
    /// up a departed object or borrowing a later appearance's copiable face.
    /// `power` substitutes the printed power and `added_types` are card types
    /// the object has beyond its printed ones (an animated land's creature
    /// type, a granted land type).
    pub fn matches_printing(
        &self,
        object: &PrintedCharacteristics,
        kind: ObjectKind,
        mana_value: u8,
        power: i32,
        added_types: &[CardType],
    ) -> bool {
        let has_type = |ty| object.types.types.contains(&ty) || added_types.contains(&ty);
        if self.namesake
            || self.with_counter
            || self.attacking
            || self.tapped
            || self.from_battlefield_this_turn
            || (self.deathtouch && !object.keywords.has(Keyword::Deathtouch))
            || (self.flying && !object.keywords.has(Keyword::Flying))
            || (self.not_flying && object.keywords.has(Keyword::Flying))
            || self
                .min_toughness
                .is_some_and(|min| object.toughness.unwrap_or(0) < i32::from(min))
        {
            return false;
        }
        if !self.accepts_name(&object.name) {
            return false;
        }
        if self
            .token
            .is_some_and(|token| token != (kind == ObjectKind::Token))
        {
            return false;
        }
        if !self.subtypes_any.is_empty() && !self.subtypes_any.iter().any(|s| object.has_subtype(s))
        {
            return false;
        }
        if !self.colors_any.is_empty() {
            let colors = object.colors();
            if !self.colors_any.iter().any(|color| colors.contains(color)) {
                return false;
            }
        }
        if !self.any_of.is_empty()
            && !self.any_of.iter().any(|alternative| {
                alternative.matches_printing(object, kind, mana_value, power, added_types)
            })
        {
            return false;
        }
        if self.not_subtypes.iter().any(|s| object.has_subtype(s)) {
            return false;
        }
        if self
            .supertype
            .as_ref()
            .is_some_and(|s| !object.supertypes.iter().any(|t| t.eq_ignore_ascii_case(s)))
        {
            return false;
        }
        if self.nonlegendary
            && object
                .supertypes
                .iter()
                .any(|s| s.eq_ignore_ascii_case("legendary"))
        {
            return false;
        }
        if self.nonbasic
            && object
                .supertypes
                .iter()
                .any(|t| t.eq_ignore_ascii_case("basic"))
        {
            return false;
        }
        if self.min_power.is_some_and(|min| power < min) {
            return false;
        }
        if let Some(card_type) = self.card_type {
            if !has_type(card_type) {
                return false;
            }
        }
        if !self.card_types_any.is_empty() && !self.card_types_any.iter().any(|t| has_type(*t)) {
            return false;
        }
        if self.not_card_types.iter().any(|t| has_type(*t)) {
            return false;
        }
        if let Some(subtype) = &self.subtype {
            if !object.has_subtype(subtype) {
                return false;
            }
        }
        if let Some(max_power) = self.max_power {
            if power > max_power {
                return false;
            }
        }
        if self.min_mana_value.is_some_and(|min| mana_value < min)
            || self.max_mana_value.is_some_and(|max| mana_value > max)
        {
            return false;
        }
        true
    }
}
