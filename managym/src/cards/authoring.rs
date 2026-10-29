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
    /// An activated ability this source grants to the object a special role
    /// names ("Equipped creature has \"...\""), not one of its own.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub granted_to: Option<&'static str>,
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
                per_color_among: None,
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
                per_color_among: None,
            }],
        )
    }

    /// "Crew N": tap other creatures with total power N or more; this
    /// Vehicle becomes an artifact creature until end of turn (CR 702.122a).
    pub fn crew(key: &'static str, power: i32) -> Self {
        Self::new(
            key,
            ProgramKind::Activated,
            vec![Op::BecomeArtifactCreature {
                duration: Some(Duration::EndOfTurn),
                target: "source",
            }],
        )
        .cost(Cost {
            crew: Some(power),
            ..Cost::mana("0")
        })
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
            granted_to: None,
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
    /// Grant this activated ability to `role` instead of having it.
    pub fn granted_to(mut self, role: &'static str) -> Self {
        self.granted_to = Some(role);
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
    /// Where `affinity` counts; absent means permanents you control.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub affinity_zone: Option<Zone>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kicker: Option<&'static str>,
    /// "As an additional cost to cast this spell, [pay] [cost]" (CR 601.2b),
    /// required rather than optional like `kicker`. With `waterbend`, it is a
    /// waterbend cost; its {X} is announced with the spell's.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub additional: Option<&'static str>,
    /// Printed flashback (CR 702.34) on a spell program: its graveyard cost.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flashback: Option<&'static str>,
    /// Printed foretell (CR 702.143) on a spell program: the cost to cast it
    /// from exile on a later turn after foretelling it from hand for {2}.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub foretell: Option<&'static str>,
    /// An activated ability of a card outside the battlefield functions only
    /// from this zone (CR 113.6); absent means the battlefield.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub zone: Option<Zone>,
    /// "Discard this card" as an activation cost; requires `zone: YourHand`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub discard_source: Option<bool>,
    /// "Discard a card" as an activation cost (CR 601.2h); chosen on
    /// activation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub discard_card: Option<bool>,
    /// Exhaust (CR 702.177a): "Activate only once."
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exhaust: Option<bool>,
    /// "Activate only once each turn" (CR 602.5b).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub once_each_turn: Option<bool>,
    /// "Activate only as a sorcery" (CR 602.5d).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sorcery_timing: Option<bool>,
    /// "This spell costs {N} less to cast if it targets a [predicate]."
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reduced_if_targets: Option<TargetReduction>,
    /// "Crew N" (CR 702.122a).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub crew: Option<i32>,
    /// The least X the activation may announce ("X can't be 0").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x_min: Option<i32>,
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
    /// "other target": the ability's source object is not a legal choice.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub another: bool,
    /// "That share a creature type": the chosen targets must share one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shares_creature_type: Option<bool>,
    /// "Up to X targets": the announced X is the maximum (`max` is 0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_x: Option<bool>,
}

impl Target {
    pub fn one(role: &'static str, selector: Selector) -> Self {
        Self {
            min: 1,
            max: 1,
            role,
            selector,
            another: false,
            shares_creature_type: None,
            max_x: None,
        }
    }

    /// "Up to `max` targets".
    pub fn up_to(max: i32, role: &'static str, selector: Selector) -> Self {
        Self {
            min: 0,
            max,
            ..Self::one(role, selector)
        }
    }

    /// "Up to X targets", X announced on casting (CR 601.2b-c).
    pub fn up_to_x(role: &'static str, selector: Selector) -> Self {
        Self {
            max_x: Some(true),
            ..Self::up_to(0, role, selector)
        }
    }

    /// "Another target": the source can't be chosen.
    pub fn another(self) -> Self {
        Self {
            another: true,
            ..self
        }
    }

    /// "That share a creature type": the chosen targets must share one.
    pub fn sharing_creature_type(self) -> Self {
        Self {
            shares_creature_type: Some(true),
            ..self
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
        /// "for each of its creature types, to a maximum of N" (static only).
        #[serde(skip_serializing_if = "Option::is_none")]
        per_creature_type_max: Option<i32>,
        /// "for each color among [predicate] permanents you control" (static only).
        #[serde(skip_serializing_if = "Option::is_none")]
        per_color_among: Option<Predicate>,
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
    /// "Distribute `count` [counter] counters among any number of [target]",
    /// chosen one counter at a time as the effect resolves.
    DistributeCounters {
        count: i32,
        counter: CounterKind,
        target: &'static str,
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
    /// "Each creature you control gains firebending until end of turn", or
    /// only the creature chosen for `target`.
    GrantFirebendingUntilEot {
        amount: i32,
        #[serde(skip_serializing_if = "Option::is_none")]
        target: Option<&'static str>,
    },
    /// "A player exiles a matching card; its controller may cast it as linked."
    ExileLinked {
        /// `you`, or `target` (the target player chooses).
        chooser: &'static str,
        /// `waterbend_mana_value` (Hama) or `flash_any_mana` (Azula).
        grant: &'static str,
        predicate: Predicate,
        target: &'static str,
        up_to: bool,
        /// The target player's `battlefield` (nontoken permanents) or
        /// `graveyard`.
        zone: &'static str,
    },
    /// "Exile cards from the top of your library until you exile a nonland
    /// card. You may cast it without paying its mana cost if its mana value
    /// is less than `below`; otherwise put it into your hand."
    ExileUntilNonlandCastFree {
        below: Quantity,
    },
    /// "You may cast a matching spell from your hand without paying its mana cost."
    CastFromHandFree {
        predicate: Predicate,
    },
    /// "You have no maximum hand size for the rest of the game."
    NoMaximumHandSize,
    /// "Shuffle your graveyard into your library."
    ShuffleGraveyardIntoLibrary,
    /// "Sacrifice this [permanent]. If you do, [then]." Nothing happens when
    /// the exact source already left the battlefield.
    SacrificeSource {
        then: Vec<Op>,
    },
    /// "The owner of [target] puts it into their library second from the top
    /// or on the bottom." The owner chooses.
    PutIntoLibrarySecondOrBottom {
        target: &'static str,
    },
    /// "Search your library for up to `count` land cards with different
    /// names and reveal them. An opponent chooses `opponent_picks` of those
    /// cards. Put the chosen cards into your graveyard and the rest onto the
    /// battlefield tapped, then shuffle."
    SearchLandsOpponentSplits {
        count: i32,
        opponent_picks: i32,
    },
    /// "Target [player] reveals their hand. You choose a [predicate] card
    /// from it. That player discards that card."
    RevealHandDiscardChosen {
        predicate: Predicate,
        target: &'static str,
    },
    /// Without a destination, a basic land onto the battlefield tapped. With
    /// `Hand`, any card of `subtype` is revealed and put into hand
    /// (landcycling: "search your library for a Plains card"), or a basic
    /// land card when `subtype` is absent.
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
    /// "[target] becomes an artifact creature" until end of turn, with no
    /// duration when absent, or as a static effect under a branch condition.
    BecomeArtifactCreature {
        #[serde(skip_serializing_if = "Option::is_none")]
        duration: Option<Duration>,
        target: &'static str,
    },
    /// "Each creature you control [predicate] can't be blocked by more than
    /// `count` creatures" (static).
    CantBeBlockedByMoreThan {
        count: i32,
        predicate: Predicate,
    },
    /// "[target] can attack as though it didn't have defender" (static,
    /// usually under a branch condition).
    AttacksDespiteDefender {
        target: &'static str,
    },
    /// "[predicate] permanents you control are [types] in addition to their
    /// other types" (static, CR 613.1d). Abilities are not granted.
    AddCardTypes {
        predicate: Predicate,
        types: Vec<CardType>,
    },
    /// "[predicate] permanents you control gain [subtypes] until end of
    /// turn"; the affected set is fixed as the effect resolves (CR 611.2c).
    AddSubtypesControlled {
        predicate: Predicate,
        subtypes: Vec<&'static str>,
    },
    /// "If you would lose unspent mana, that mana becomes red instead"
    /// (static, CR 106.4, 614).
    UnspentManaBecomesRed,
    /// "Each creature with mana value [max_mana_value] or less loses all
    /// abilities until end of turn. Destroy those creatures."
    DestroyCreaturesLosingAbilities {
        max_mana_value: Quantity,
    },
    /// "[target] have base power and toughness [power]/[toughness]" with
    /// resolution-time values (CR 613.4b).
    SetBasePtValue {
        duration: Duration,
        power: Quantity,
        target: &'static str,
        toughness: Quantity,
    },
    /// "This permanent enters with X [counter] counters on it" (static).
    EntersWithXCounters {
        counter: CounterKind,
    },
    /// "If a source you control would deal damage to an opponent or a
    /// permanent an opponent controls, it deals that much damage plus the
    /// number of [counter] counters on this permanent instead" (static).
    OpponentDamagePlusCounters {
        counter: CounterKind,
    },
    /// "Add X mana of any one color", X read when the mana ability resolves.
    AddManaOfOneColor {
        amount: Quantity,
    },
    /// "[sources] fights [target]" (CR 701.14).
    Fight {
        sources: &'static str,
        target: &'static str,
    },
    /// "You gain life equal to [amount]."
    GainLifeValue {
        amount: Quantity,
    },
    /// "Add [mana] for each of [amount]" ("add that much {R}").
    AddManaValue {
        amount: Quantity,
        mana: &'static str,
    },
    /// "Until end of turn, you don't lose unspent red mana as steps and
    /// phases end."
    KeepUnspentRedMana,
    /// "Search your library for a [predicate] card, reveal it, put it into
    /// your hand, then shuffle."
    SearchLibraryToHand {
        predicate: Predicate,
    },
    /// "You may look at the top card of your library any time" (static).
    LookAtLibraryTop,
    /// "You may cast [predicate] spells from the top of your library"
    /// (static).
    CastFromLibraryTop {
        predicate: Predicate,
    },
    /// "[target] loses all abilities and is a [subtypes] with base power and
    /// toughness P/T named [name]" (static; the enchanted creature).
    SetCharacteristics {
        name: &'static str,
        power: i32,
        subtypes: Vec<&'static str>,
        target: &'static str,
        toughness: i32,
    },
    /// "Remove all counters from [target]."
    RemoveAllCounters {
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
        look: Count,
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
    /// Mill, then you may put a milled card matching `predicate` into hand.
    /// `then` runs if any milled card matches `milled_any` ("You gain 2 life
    /// if a Lesson card is milled this way").
    MillAndSelect {
        count: i32,
        if_declined: Vec<Op>,
        predicate: Predicate,
        #[serde(skip_serializing_if = "Option::is_none")]
        milled_any: Option<Predicate>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        then: Vec<Op>,
    },
    CounterSpell {
        target: &'static str,
    },
    /// Mill `count`; with `target`, that player mills instead of you.
    MillCards {
        count: i32,
        #[serde(skip_serializing_if = "Option::is_none")]
        target: Option<&'static str>,
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
    Supertype {
        value: &'static str,
    },
    /// "with no counters on it" (any kind, CR 122).
    NoCounters,
    /// A token (`true`) or a nontoken (`false`) object.
    Token {
        value: bool,
    },
    /// An attacking or blocking creature (CR 506.4).
    AttackingOrBlocking,
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
    /// "As long as this has `count` or more [counter] counters on it."
    SourceCountersAtLeast {
        count: i32,
        counter: CounterKind,
    },
    /// "If you control a [predicate] permanent."
    ControlsMatching {
        predicate: Predicate,
    },
    /// "As long as you have `count` or more unspent mana."
    UnspentManaAtLeast {
        count: i32,
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
    /// The resolving spell's announced X.
    X,
    BattlefieldMatching {
        predicate: Predicate,
    },
    GraveyardMatching {
        base: i32,
        predicate: Predicate,
    },
    CreaturesWithCounters,
    CardsDrawnMinus {
        subtract: i32,
    },
    /// The greatest number of creatures you control that share a creature type.
    GreatestSharedCreatureType,
    /// "The excess damage dealt this way" by the previous instruction
    /// (CR 120.4a).
    ExcessDamage,
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
    EndStepBegins,
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
