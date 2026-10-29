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

    /// Landcycling (CR 702.29e): "[mana], Discard this card: Search your
    /// library for a [subtype] card, reveal it, put it into your hand, then
    /// shuffle." It functions only from its owner's hand.
    pub fn landcycling(key: &'static str, mana: &'static str, subtype: &'static str) -> Self {
        Self::new(
            key,
            ProgramKind::Activated,
            vec![Op::SearchBasicLand {
                destination: Some(Destination::Hand),
                subtype: Some(subtype),
            }],
        )
        .cost(Cost {
            mana: Some(mana),
            zone: Some(Zone::YourHand),
            discard_source: Some(true),
            ..Cost::default()
        })
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
    /// Printed flashback (CR 702.34) on a spell program: its graveyard cost.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flashback: Option<&'static str>,
    /// An activated ability of a card outside the battlefield functions only
    /// from this zone (CR 113.6); absent means the battlefield.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub zone: Option<Zone>,
    /// "Discard this card" as an activation cost; requires `zone: YourHand`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub discard_source: Option<bool>,
    /// Exhaust (CR 702.177a): "Activate only once."
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exhaust: Option<bool>,
    /// "Activate only once each turn" (CR 602.5b).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub once_each_turn: Option<bool>,
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
    /// "other target": the ability's source object is not a legal choice.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub another: bool,
}

impl Target {
    pub fn one(role: &'static str, selector: Selector) -> Self {
        Self {
            min: 1,
            max: 1,
            role,
            selector,
            another: false,
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
    /// "You may discard a card. If you do, draw `count` cards." With
    /// `up_to`, discard up to that many, then draw `count` for each.
    OptionalDiscardDraw {
        count: i32,
        #[serde(skip_serializing_if = "Option::is_none")]
        up_to: Option<i32>,
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
    },
    PutCounters {
        count: i32,
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
        count: Count,
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
    /// Airbend (CR 701.65): exile the target; while that exact card stays
    /// exiled, its owner may cast it for {2} rather than its mana cost.
    Airbend {
        target: &'static str,
    },
    /// Airbend every permanent matching `predicate` except the object chosen
    /// for the `except` role, if any (Avatar's Wrath).
    AirbendAll {
        except: &'static str,
        predicate: Predicate,
    },
    /// "Until your next turn, your opponents can't cast spells from anywhere
    /// other than their hands."
    OpponentsCastOnlyFromHand,
    /// "Exile the top card of your library. You may play that card this turn."
    ExileTopCardPlayThisTurn,
    /// "Sacrifice this [permanent]. If you do, [then]." Nothing happens when
    /// the exact source already left the battlefield.
    SacrificeSource {
        then: Vec<Op>,
    },
    /// Without a destination, a basic land onto the battlefield tapped. With
    /// `Hand`, any card of `subtype` is revealed and put into hand
    /// (landcycling: "search your library for a Plains card").
    SearchBasicLand {
        #[serde(skip_serializing_if = "Option::is_none")]
        destination: Option<Destination>,
        subtype: Option<&'static str>,
    },
    DrawCards {
        count: i32,
    },
    /// `target` may be `triggering_permanent`: the creature whose event
    /// fired the trigger ("it gets +X/+X").
    BuffTargetValue {
        power: Quantity,
        target: &'static str,
        toughness: Count,
    },
    CreateToken {
        count: i32,
        definition_ref: &'static str,
        #[serde(skip_serializing_if = "Option::is_none")]
        tapped_and_attacking: Option<bool>,
    },
    CreateTokensValue {
        count: Quantity,
        definition_ref: &'static str,
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
    /// Return the target graveyard card to its owner's hand, or with
    /// `BattlefieldTapped` put it onto the battlefield tapped.
    ReturnGraveyardTarget {
        target: &'static str,
        #[serde(skip_serializing_if = "Option::is_none")]
        destination: Option<Destination>,
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
    /// Mana with a spending restriction (CR 106.6); omitted `spend_only`
    /// means "Spend this mana only to cast Lesson spells."
    AddLessonMana {
        mana: &'static str,
        #[serde(skip_serializing_if = "Option::is_none")]
        spend_only: Option<SpendOnly>,
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

    /// "Spend this mana only to cast Lesson spells."
    pub fn lesson_mana(mana: &'static str) -> Self {
        Self::AddLessonMana {
            mana,
            spend_only: None,
        }
    }

    /// "Add one mana of any color. Spend this mana only to ..."
    pub fn choose_restricted_mana(spend_only: SpendOnly) -> Self {
        Self::Modal {
            modes: ["W", "U", "B", "R", "G"]
                .into_iter()
                .map(|mana| {
                    vec![Self::AddLessonMana {
                        mana,
                        spend_only: Some(spend_only.clone()),
                    }]
                })
                .collect(),
        }
    }
}

/// What restricted mana may be spent on: spells with any of `spell_subtypes`,
/// or abilities of sources with any of `ability_source_subtypes`.
#[derive(Serialize, Default, Clone)]
pub struct SpendOnly {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub spell_subtypes: Vec<&'static str>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub ability_source_subtypes: Vec<&'static str>,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Predicate {
    NotCardTypes {
        values: Vec<CardType>,
    },
    Subtype {
        value: &'static str,
    },
    All {
        predicates: Vec<Predicate>,
    },
    /// Matches when any listed predicate matches ("artifact or Lesson").
    AnyOf {
        predicates: Vec<Predicate>,
    },
    CardType {
        value: CardType,
    },
    PowerAtMost {
        value: i32,
    },
    ManaValueAtLeast {
        value: i32,
    },
    SubtypesAny {
        values: Vec<&'static str>,
    },
    Flying,
    WithCounter,
    /// An attacking creature (CR 508.1k); only permanents in combat match.
    Attacking,
    NotSubtypes {
        values: Vec<&'static str>,
    },
    TypesAny {
        values: Vec<CardType>,
    },
    PowerAtLeast {
        value: i32,
    },
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

/// A printed number or a resolution-time quantity ("earthbend X, where X is…").
/// Fixed counts serialize as plain integers.
#[derive(Serialize)]
#[serde(untagged)]
pub enum Count {
    Fixed(i32),
    Of(Quantity),
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
    /// Exactly one creature was declared as an attacker (CR 506.5).
    AttacksAlone,
    BecomesTapped,
    BecomesTargeted,
    CastEnters,
    CastSpell,
    /// "At the beginning of combat on your turn."
    CombatBegins,
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
    BattlefieldTapped,
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
    YourHand,
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
