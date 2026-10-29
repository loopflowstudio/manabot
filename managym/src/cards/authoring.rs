//! Typed, symbolic card authoring.
//!
//! Set files own rules and presentation. These types retain symbolic references
//! and binding roles; `semantic::compile` validates them and assigns numeric
//! identities in process before the shared runtime lowers IR. No JSON is parsed here.
//! Optional fields preserve absence versus explicit null in historical source hashes.
use serde::Serialize;

#[derive(Serialize)]
pub struct Card {
    pub characteristics: Characteristics,
    pub key: &'static str,
    pub programs: Vec<Program>,
    pub registry_name: &'static str,
    pub text_box: &'static str,
}

#[derive(Serialize, Default)]
pub struct Characteristics {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mana_cost: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subtypes: Option<Vec<&'static str>>,
    pub types: Vec<CardType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keywords: Option<Vec<Keyword>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub power: Option<PrintedPower>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub toughness: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supertypes: Option<Vec<&'static str>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub colors: Option<Vec<Color>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enters_tapped: Option<bool>,
    /// "This land enters tapped unless you control a [predicate] permanent."
    /// The entering player's battlefield is read as the land moves (CR 614.1c).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enters_tapped_unless: Option<Predicate>,
}

/// A printed `*` has no fixed numeric power; its program supplies the CDA.
#[derive(Serialize)]
#[serde(untagged)]
pub enum PrintedPower {
    Fixed(i32),
    CharacteristicDefined,
}

#[derive(Serialize)]
pub struct Program {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<Cost>,
    pub key: &'static str,
    pub kind: ProgramKind,
    pub ops: Vec<Op>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_binding: Option<TargetBinding>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub targets: Option<Vec<Target>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trigger: Option<Trigger>,
}

impl Program {
    /// Prowess always listens to the controller's noncreature casts and grants
    /// +1/+1 only until end of turn, including when authored on a token.
    pub fn prowess(key: &'static str) -> Self {
        Self::new(
            key,
            ProgramKind::Triggered,
            vec![Op::ModifyPt {
                duration: Duration::EndOfTurn,
                power: 1,
                target: "source",
                toughness: 1,
                per_creature_type_max: None,
            }],
        )
        .trigger(Trigger {
            predicate: Some(Predicate::NotCardTypes {
                values: vec![CardType::Creature],
            }),
            ..Trigger::new(Event::CastSpell, Subject::This)
        })
    }

    /// Firebending's attack trigger produces red mana retained through combat.
    /// `mana` uses the ordinary mana notation (for example, "RR" for two red).
    pub fn firebending(key: &'static str, mana: &'static str) -> Self {
        Self::new(
            key,
            ProgramKind::Triggered,
            vec![Op::AddMana {
                mana,
                until: Some(Until::EndOfCombat),
            }],
        )
        .trigger(Trigger::new(Event::Attacks, Subject::This))
    }

    /// "Equip [mana]": attach to target creature you control, only as a
    /// sorcery (CR 702.6a).
    pub fn equip(key: &'static str, mana: &'static str) -> Self {
        Self::new(
            key,
            ProgramKind::Activated,
            vec![Op::Attach { target: "creature" }],
        )
        .cost(Cost {
            sorcery_timing: Some(true),
            ..Cost::mana(mana)
        })
        .targets(vec![Target::one("creature", Selector::CreatureYouControl)])
    }

    /// The spell of an Aura with "Enchant [selector]": it targets what it
    /// can enchant and enters attached to it (CR 303.4a, 303.4f).
    pub fn enchant(mana: &'static str, selector: Selector) -> Self {
        Self::new(
            "spell",
            ProgramKind::Spell,
            vec![Op::Attach { target: "enchant" }],
        )
        .cost(Cost::mana(mana))
        .targets(vec![Target::one("enchant", selector)])
    }

    /// "Equipped creature gets +P/+T."
    pub fn equipped_pt(key: &'static str, power: i32, toughness: i32) -> Self {
        Self::new(
            key,
            ProgramKind::Static,
            vec![Op::ModifyPt {
                duration: Duration::Static,
                power,
                target: "each:equipped_creature",
                toughness,
                per_creature_type_max: None,
            }],
        )
    }

    pub fn new(key: &'static str, kind: ProgramKind, ops: Vec<Op>) -> Self {
        Self {
            key,
            kind,
            ops,
            cost: None,
            targets: None,
            trigger: None,
            target_binding: None,
        }
    }
    pub fn cost(mut self, cost: Cost) -> Self {
        self.cost = Some(cost);
        self
    }
    pub fn targets(mut self, targets: Vec<Target>) -> Self {
        self.targets = Some(targets);
        self
    }
    pub fn trigger(mut self, trigger: Trigger) -> Self {
        self.trigger = Some(trigger);
        self
    }
    pub fn inferred(mut self) -> Self {
        self.target_binding = Some(TargetBinding::Inferred);
        self
    }
}

#[derive(Serialize, Default)]
pub struct Cost {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mana: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sacrifice_source: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tap_source: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub waterbend: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sacrifice: Option<Sacrifice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub own_turn_only: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub life: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub affinity: Option<Predicate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kicker: Option<&'static str>,
    /// "Activate only as a sorcery" (CR 602.5d).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sorcery_timing: Option<bool>,
    /// "Exhaust — [Cost]": activate only once (CR 702.177a).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exhaust: Option<bool>,
    /// "This spell costs {N} less to cast if it targets a [predicate]."
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reduced_if_targets: Option<TargetReduction>,
}

#[derive(Serialize)]
pub struct TargetReduction {
    pub generic: i32,
    pub predicate: Predicate,
}

impl Cost {
    pub fn mana(mana: &'static str) -> Self {
        Self {
            mana: Some(mana),
            ..Self::default()
        }
    }
}

/// Resolving payments have exactly one payment method.
#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Payment {
    Mana(&'static str),
    Waterbend(i32),
}

#[derive(Serialize)]
pub struct Sacrifice {
    pub another: bool,
    pub or_mana: Option<&'static str>,
    pub predicate: Predicate,
}

#[derive(Serialize)]
pub struct Target {
    pub max: i32,
    pub min: i32,
    pub role: &'static str,
    pub selector: Selector,
}

impl Target {
    pub fn one(role: &'static str, selector: Selector) -> Self {
        Self {
            min: 1,
            max: 1,
            role,
            selector,
        }
    }
}

#[derive(Serialize)]
pub struct Trigger {
    pub event: Event,
    pub subject: Subject,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub predicate: Option<Predicate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub controller_only: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub once_each_turn: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub own_turn_only: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "if")]
    pub condition: Option<Condition>,
}

impl Trigger {
    pub fn new(event: Event, subject: Subject) -> Self {
        Self {
            event,
            subject,
            n: None,
            predicate: None,
            controller_only: None,
            once_each_turn: None,
            own_turn_only: None,
            condition: None,
        }
    }
}

#[derive(Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Op {
    OptionalDiscardDraw {
        count: i32,
    },
    Modal {
        modes: Vec<Vec<Op>>,
    },
    TargetClause {
        body: Vec<Op>,
        role: &'static str,
    },
    ModifyPt {
        duration: Duration,
        power: i32,
        target: &'static str,
        toughness: i32,
        /// "for each of its creature types, to a maximum of N" (static only).
        #[serde(skip_serializing_if = "Option::is_none")]
        per_creature_type_max: Option<i32>,
    },
    /// "[target] has base power and toughness P/T"; an absent value leaves
    /// that base unchanged.
    SetBasePt {
        duration: Duration,
        #[serde(skip_serializing_if = "Option::is_none")]
        power: Option<i32>,
        target: &'static str,
        #[serde(skip_serializing_if = "Option::is_none")]
        toughness: Option<i32>,
    },
    /// Remove up to `count` counters of any kinds, chosen one at a time.
    RemoveCounters {
        count: i32,
        target: &'static str,
    },
    PutCounters {
        count: i32,
        /// Absent for +1/+1 counters.
        #[serde(skip_serializing_if = "Option::is_none")]
        counter: Option<CounterKind>,
        target: &'static str,
    },
    ReturnToHandDrawIfControlled {
        target: &'static str,
    },
    DealDamageValue {
        amount: Quantity,
        target: &'static str,
    },
    ExileIfDies {
        target: &'static str,
    },
    DiscardTargetPlayer {
        count: i32,
        target: &'static str,
    },
    Earthbend {
        count: i32,
        target: &'static str,
    },
    DrainLifeValue {
        amount: Quantity,
    },
    CantBeBlockedTarget {
        target: &'static str,
    },
    AddSubtype {
        subtype: &'static str,
        target: &'static str,
    },
    GrantKeywords {
        duration: Duration,
        keywords: Vec<Keyword>,
        target: &'static str,
    },
    Learn,
    Exile {
        target: &'static str,
    },
    SearchBasicLand {
        subtype: Option<&'static str>,
    },
    DrawCards {
        count: i32,
    },
    BuffTargetValue {
        power: Quantity,
        target: &'static str,
        toughness: i32,
    },
    CreateToken {
        count: i32,
        definition_ref: &'static str,
        #[serde(skip_serializing_if = "Option::is_none")]
        tapped_and_attacking: Option<bool>,
    },
    /// "Create a [token], then attach this Equipment to it." The IR is a
    /// one-token `create_token` with `attach_source`.
    #[serde(rename = "create_token")]
    CreateTokenAttached {
        count: i32,
        definition_ref: &'static str,
        attach_source: bool,
    },
    CreateTokensValue {
        count: Quantity,
        definition_ref: &'static str,
    },
    /// Attach the source Equipment to the target creature; an Aura spell's
    /// one instruction.
    Attach {
        target: &'static str,
    },
    /// "[target] can't attack or block" (static).
    CantAttackOrBlock {
        target: &'static str,
    },
    /// "[target] doesn't untap during its controller's untap step" (static).
    DoesntUntap {
        target: &'static str,
    },
    /// "[target]'s owner shuffles it into their library."
    ShuffleIntoLibrary {
        target: &'static str,
    },
    Scry {
        count: i32,
    },
    PreventLifeGain,
    Discard,
    SpellCostReduction {
        condition: Condition,
        first_on_own_turn: bool,
        generic: i32,
        predicate: Predicate,
    },
    DamageOpponentCreatures {
        amount: i32,
    },
    DealDamage {
        amount: i32,
        target: &'static str,
    },
    ForEachTarget {
        body: Vec<Op>,
        role: &'static str,
    },
    LookAndSelect {
        destination: Destination,
        look: i32,
        max_select: i32,
        min_select: i32,
        predicate: Predicate,
        #[serde(skip_serializing_if = "Option::is_none")]
        reveal: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        remainder: Option<Remainder>,
        #[serde(skip_serializing_if = "Option::is_none")]
        remainder_order: Option<RemainderOrder>,
    },
    StaticPt {
        condition: Condition,
        power: i32,
        toughness: i32,
    },
    Untap {
        target: &'static str,
    },
    Destroy {
        target: &'static str,
    },
    MillAndSelect {
        count: i32,
        if_declined: Vec<Op>,
        predicate: Predicate,
    },
    CounterSpell {
        target: &'static str,
    },
    MillCards {
        count: i32,
    },
    CombatRequiresOther {
        predicate: Predicate,
    },
    PowerFromLandCounters,
    ReturnGraveyardTarget {
        target: &'static str,
    },
    AddMana {
        mana: &'static str,
        #[serde(skip_serializing_if = "Option::is_none")]
        until: Option<Until>,
    },
    ModifyTriggeringPermanent {
        counters: i32,
        keywords: Vec<Keyword>,
    },
    ExileWithControllerToken {
        definition_ref: &'static str,
        target: &'static str,
    },
    DrainLife {
        amount: i32,
    },
    DealDamageToController {
        amount: i32,
    },
    GainLife {
        amount: i32,
    },
    Sacrifice {
        predicate: Predicate,
    },
    SearchBasicLands {
        bottom_source: bool,
        count: i32,
        tapped: bool,
        untap_if_lands: Option<i32>,
    },
    UnblockableIf {
        condition: Condition,
    },
    ReturnToHand {
        target: &'static str,
    },
    GrantMana {
        predicates: Vec<Predicate>,
    },
    AddLessonMana {
        mana: &'static str,
    },
    GrantFlashback,
    Branch {
        condition: Condition,
        otherwise: Vec<Op>,
        then: Vec<Op>,
    },
    CreateNonlegendaryCopy,
    ModifyControlled {
        power: i32,
        predicate: Predicate,
        toughness: i32,
    },
    OptionalPayment {
        cost: Payment,
        if_declined: Vec<Op>,
        if_paid: Vec<Op>,
    },
    PutTopCardsInHand {
        count: i32,
    },
    DealPowerDamage {
        sources: &'static str,
        target: &'static str,
    },
    SetPowerFromCount {
        controller: Controller,
        predicate: Predicate,
        zone: Zone,
    },
    CounterUnlessPays {
        cost: &'static str,
        target: &'static str,
    },
    ExileUntilSourceLeaves {
        target: &'static str,
    },
    RestrictBlocking {
        duration: Duration,
        predicate: Predicate,
        target: &'static str,
    },
    Tap {
        target: &'static str,
    },
    BlockingRestriction {
        direction: Direction,
        predicate: Predicate,
    },
}

impl Op {
    /// Each mode produces one listed mana alternative. Separate fixed mana
    /// programs are additive in the runtime and cannot represent this choice.
    pub fn choose_mana(choices: &[&'static str]) -> Self {
        Self::Modal {
            modes: choices
                .iter()
                .map(|&mana| vec![Self::AddMana { mana, until: None }])
                .collect(),
        }
    }
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Predicate {
    NotCardTypes { values: Vec<CardType> },
    Subtype { value: &'static str },
    Supertype { value: &'static str },
    All { predicates: Vec<Predicate> },
    CardType { value: CardType },
    PowerAtMost { value: i32 },
    ManaValueAtLeast { value: i32 },
    SubtypesAny { values: Vec<&'static str> },
    Flying,
    NoCounters,
    NotSubtypes { values: Vec<&'static str> },
    TypesAny { values: Vec<CardType> },
    PowerAtLeast { value: i32 },
    Any,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Condition {
    GraveyardAtLeast {
        count: i32,
        predicate: Predicate,
    },
    CardsDrawnAtLeast {
        count: i32,
    },
    Always,
    Kicked,
    NthResolution {
        n: i32,
    },
    TargetMatches {
        predicate: Predicate,
        role: &'static str,
    },
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Quantity {
    BattlefieldMatching { predicate: Predicate },
    GraveyardMatching { base: i32, predicate: Predicate },
    CreaturesWithCounters,
    CardsDrawnMinus { subtract: i32 },
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Selector {
    Creature,
    Permanent {
        predicate: Predicate,
    },
    CreatureOrPlayer,
    Opponent,
    LandYouControl,
    PermanentYouControl {
        predicate: Predicate,
    },
    CreatureYouControl,
    CreatureOpponentControls,
    Player,
    Spell,
    GraveyardCardYouOwn {
        predicate: Predicate,
    },
    SpellOrPermanent {
        min_mana_value: i32,
    },
    PermanentOpponentControls {
        min_mana_value: i32,
        types_any: Vec<CardType>,
    },
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Subject {
    This,
    AnotherYouControl { predicate: Predicate },
    AnyYouControl { predicate: Predicate },
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CardType {
    Artifact,
    Battle,
    Creature,
    Enchantment,
    Instant,
    Land,
    Planeswalker,
    Sorcery,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Keyword {
    Deathtouch,
    Defender,
    DoubleStrike,
    FirstStrike,
    Flash,
    Flying,
    Haste,
    Hexproof,
    Indestructible,
    Lifelink,
    Menace,
    Reach,
    Trample,
    Vigilance,
}

/// Counter kinds other than +1/+1 (CR 122).
#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CounterKind {
    DoubleStrike,
    Fire,
    Stun,
}

#[derive(Serialize)]
pub enum Color {
    G,
    R,
    W,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProgramKind {
    Activated,
    Mana,
    Spell,
    Static,
    Triggered,
    TriggeredMana,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetBinding {
    Inferred,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Duration {
    EndOfTurn,
    Static,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Event {
    Attacks,
    BecomesTapped,
    BecomesTargeted,
    CastEnters,
    CastSpell,
    Dies,
    DrawNthCard,
    EntersBattlefield,
    Sacrificed,
    TappedForMana,
    UpkeepBegins,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Destination {
    Hand,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Remainder {
    LibraryBottom,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RemainderOrder {
    Random,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Until {
    EndOfCombat,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Controller {
    You,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Zone {
    Battlefield,
    YourGraveyard,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    BlockedBy,
    Blocking,
}

/// Every current definition, plus the TLA fixed tokens that every pool carries.
pub struct Catalog {
    pub definitions: Vec<Card>,
    pub token_templates: Vec<&'static str>,
    /// Definitions marked incomplete in their set file.
    pub incomplete: Vec<Incomplete>,
}

/// A defined card whose behavior is known to be incomplete: `(key, missing
/// behavior)`. It still runs in the fixed setups that name it, but the world
/// does not support it for deck entry.
pub type Incomplete = (&'static str, &'static str);
