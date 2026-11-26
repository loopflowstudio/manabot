use super::card::{Keyword, Keywords};
use super::effect::{CopyException, EffectDuration, Modification};
use super::game_object::{PermanentId, PlayerId, PublicId};
use super::mana::{Mana, ManaCost};
use super::payment::How;
use super::permanent::CounterKind;
use super::predicate::CardPredicate;
use super::zone::ZoneType;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Ability {
    Triggered {
        condition: TriggerCondition,
        /// Intervening "if" clause (CR 603.4): the ability triggers only if
        /// this holds when its trigger event occurs, and does nothing unless
        /// it still holds as it resolves.
        #[serde(skip_serializing_if = "Option::is_none")]
        intervening_if: Option<Condition>,
        effects: Vec<Effect>,
    },
}

impl Ability {
    pub fn enters(effects: Vec<Effect>) -> Self {
        Self::when_enters(TriggerSubject::This, effects)
    }

    pub fn when_enters(subject: TriggerSubject, effects: Vec<Effect>) -> Self {
        Self::Triggered {
            condition: TriggerCondition::EntersTheBattlefield { subject },
            intervening_if: None,
            effects,
        }
    }

    /// A Saga's chapter ability (CR 714.2).
    pub fn chapter(chapter: u8, effects: Vec<Effect>) -> Self {
        Self::Triggered {
            intervening_if: None,
            condition: TriggerCondition::Chapter { chapter },
            effects,
        }
    }

    pub fn when_you_gain_life(effect: Effect) -> Self {
        Self::Triggered {
            condition: TriggerCondition::GainsLife {
                who: Actor::You,
                nth: None,
            },
            intervening_if: None,
            effects: vec![effect],
        }
    }

    /// The triggered rule represented by printed or granted ward (CR 702.21).
    pub fn ward(cost: PaymentCost) -> Self {
        Self::Triggered {
            intervening_if: None,
            condition: TriggerCondition::BecomesTargeted {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::CounterUnlessPays { cost }],
        }
    }

    /// Prowess (CR 702.108) is the same triggered ability on a card, token or grant.
    pub fn prowess() -> Self {
        Self::Triggered {
            intervening_if: None,
            condition: TriggerCondition::casts(Actor::You, CardPredicate::noncreature()),
            effects: vec![Effect::until_end_of_turn(
                ObjectSelector::This,
                Modification::pump(1, 1),
            )],
        }
    }

    /// The triggered rule represented by firebending N (CR 702.189a):
    /// "Whenever this creature attacks, add N {R}. This mana lasts until end
    /// of combat." Each instance triggers separately (CR 702.189b).
    pub fn firebending(amount: u8) -> Self {
        let mut mana = Mana::default();
        mana.mana[super::mana::Color::Red as usize] = amount;
        Self::Triggered {
            intervening_if: None,
            condition: TriggerCondition::attacks(TriggerSubject::This, Per::Object),
            effects: vec![Effect::AddMana {
                mana,
                amount: EffectValue::Fixed(1),
                until_end_of_combat: true,
            }],
        }
    }

    pub fn effects(&self) -> &[Effect] {
        match self {
            Ability::Triggered { effects, .. } => effects,
        }
    }

    pub fn reads_source_power(&self) -> bool {
        self.reads(ContextRead::SourcePower)
    }

    pub fn reads_source_lki(&self) -> bool {
        self.reads(ContextRead::SourceLki)
    }

    pub fn reads_triggering_object(&self) -> bool {
        self.reads(ContextRead::TriggeringObject)
    }

    fn reads(&self, read: ContextRead) -> bool {
        let Self::Triggered {
            effects,
            intervening_if,
            ..
        } = self;
        effects.iter().any(|effect| effect.reads(read))
            || intervening_if.as_ref().is_some_and(|c| c.reads(read))
    }
}

/// The event that causes a triggered ability to trigger (CR 603.1).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum TriggerCondition {
    /// The entering source must have been cast, rather than created or returned.
    WasCast {
        condition: Box<TriggerCondition>,
    },
    /// "Whenever [subject] is sacrificed" (CR 701.21): once per sacrificed
    /// permanent, matched as it last existed on the battlefield.
    Sacrificed {
        subject: TriggerSubject,
    },
    /// Trigger at most once per exact source incarnation (or grant clock) each
    /// turn, regardless of wrapper order or trigger doublers.
    OnceEachTurn {
        condition: Box<TriggerCondition>,
    },
    /// "When [this] has no [kind] counters on it": a state trigger (CR 603.8)
    /// of a battlefield source. It triggers whenever the state holds and
    /// not again while its last occurrence waits or is on the stack.
    NoCounters {
        kind: CounterKind,
    },
    /// "When [condition]": a state trigger (CR 603.8) of a battlefield
    /// source, asked of its source rather than an event ("When there are
    /// four or more page counters on Mazemind Tome"). It triggers whenever
    /// the condition holds and not again while its last occurrence waits or
    /// is on the stack.
    State(Condition),
    /// "When you cast this spell" (CR 603.2): functions while the source is
    /// a spell on the stack, and its caster controls the trigger.
    YouCastThis,
    /// Spend the threshold mana on spells during one turn.
    Expend(u32),
    /// The inner condition only while the source permanent is attacking
    /// ("whenever you cast a spell while Fire Lord Azula is attacking").
    WhileSourceAttacking {
        condition: Box<TriggerCondition>,
    },
    /// "Whenever a player casts a [predicate] spell that targets only this
    /// [permanent]": every target of the spell is this permanent. The spell
    /// is the context and its caster "that player".
    SpellTargetsOnlyThis {
        predicate: CardPredicate,
    },
    /// "When [subject] enters the battlefield."
    EntersTheBattlefield {
        subject: TriggerSubject,
    },
    /// "When [subject] dies." — battlefield to graveyard (CR 700.4).
    Dies {
        subject: TriggerSubject,
    },
    /// "When [subject] leaves the battlefield": to any other zone, looking
    /// back in time (CR 603.10a).
    LeavesTheBattlefield {
        subject: TriggerSubject,
    },
    /// "Whenever [subject] attacks" (CR 508.1, after the declare-attackers
    /// turn-based action). `per` says whether it triggers once for each
    /// matching attacker or once per combat ("whenever you attack", "one or
    /// more ... attack"); `attackers` limits how many creatures were
    /// declared in all ("attacks alone", "and at least one other creature
    /// attack"). Per-object occurrences each carry their matching attacker as context.
    Attacks {
        subject: TriggerSubject,
        per: Per,
        attackers: AttackSize,
        #[serde(skip_serializing_if = "Option::is_none")]
        nth: Option<u32>,
    },
    /// The inner cast or entry trigger only for an object that came from
    /// `zone` ("whenever you cast a spell from exile", "whenever a permanent
    /// you control enters from exile").
    FromZone {
        zone: ZoneType,
        condition: Box<TriggerCondition>,
    },
    /// The inner cast trigger only for a prepared spell: one cast from a
    /// prepared copy ("whenever you cast a prepared spell"). Copies of that
    /// spell are not prepared spells.
    Prepared {
        condition: Box<TriggerCondition>,
    },
    /// "Whenever [subject] becomes tapped."
    BecomesTapped {
        subject: TriggerSubject,
    },
    /// "Whenever [subject] becomes untapped": a tapped permanent untaps,
    /// in its untap step or otherwise; a stun counter removed instead
    /// (CR 122.1d) is no untapping.
    BecomesUntapped {
        subject: TriggerSubject,
    },
    /// "Whenever [subject] is tapped for mana."
    TappedForMana {
        subject: TriggerSubject,
    },
    /// "Whenever [subject] becomes the target of a spell or ability an
    /// opponent controls." (Ward, CR 702.21.) The triggering spell or ability
    /// is threaded to the ability's effects as the trigger context target.
    BecomesTargeted {
        subject: TriggerSubject,
    },
    /// "Whenever a creature you control attacking causes a triggered ability
    /// of that creature to trigger": that ability, once on the stack, is the
    /// trigger's context.
    YourAttackingCreatureTriggers,
    /// A Saga's chapter ability (CR 714.2b): triggers as lore counters put
    /// on this Saga bring their number from below `chapter` to at least it.
    /// Abilities printed "I, II" are one ability per chapter.
    Chapter {
        chapter: u8,
    },
    /// Triggers once when a placement crosses this count on the source.
    NthCounter {
        kind: CounterKind,
        nth: u8,
    },
    /// "Whenever [who] puts one or more [kind] counters on a [on]
    /// permanent": once per permanent an instruction that player controls
    /// puts them on. The trigger's X is how many were put ("that much").
    PutsCounters {
        who: Actor,
        kind: CounterKind,
        on: CardPredicate,
    },
    /// "Whenever [who] puts one or more [kind] counters on [subject]": once
    /// per permanent per placement, whichever objects the subject watches
    /// (`This` for "on this creature"). Entry counters (CR 122.6) count. The
    /// trigger's X is how many were put.
    PutsCountersOn {
        who: Actor,
        kind: CounterKind,
        subject: TriggerSubject,
    },
    /// "Whenever [who] casts a [predicate] spell" (CR 601.2i); with `nth`,
    /// only the spell that is the Nth its caster cast this turn ("whenever
    /// you cast your second spell each turn"). The caster is "that player".
    Casts {
        who: Actor,
        predicate: CardPredicate,
        nth: Option<u32>,
        /// What the spell must target as it is cast ("a spell that targets
        /// this creature").
        #[serde(skip_serializing_if = "SpellTargetFilter::is_any")]
        targets: SpellTargetFilter,
    },
    /// "Whenever [who] draws a card"; with `nth`, only that player's Nth
    /// draw this turn (see `PlayerTurnHistory::cards_drawn`). The drawer is
    /// "that player".
    Draws {
        who: Actor,
        nth: Option<u32>,
    },
    GainsLife {
        who: Actor,
        nth: Option<u32>,
    },
    /// "Whenever [who] loses life" (CR 119.3): once per committed decrease
    /// of a life total, damage included; prevented or replaced loss that
    /// leaves the total unchanged is no event. The life actually lost is the
    /// trigger's amount (`EffectValue::TriggeringAmount`, "that much") and
    /// the loser is "that player".
    LosesLife {
        who: Actor,
    },
    /// "At the beginning of [whose] [step]": `whose` is asked of the active
    /// player, who is "that player".
    BeginningOf {
        step: TurnStep,
        whose: Actor,
    },
    /// "Whenever [who] [does one of these keyword actions]": once per
    /// search (even one that finds nothing, CR 701.23), scry with a count
    /// above zero (CR 701.22), completed surveil (CR 701.25), bending
    /// instruction or waterbend payment. The performer is "that player".
    Performs {
        who: Actor,
        any_of: Vec<KeywordAction>,
    },
    /// "Whenever a [predicate] card is put into [whose] graveyard from
    /// anywhere" (Bloodchief Ascension): once per card, never a token
    /// (tokens are not cards); the graveyard's owner is "that player".
    PutIntoGraveyard {
        whose: Actor,
        predicate: CardPredicate,
    },
    /// One or more physical cards leave the controller’s graveyard in one event.
    CardsLeaveYourGraveyard,
    /// Whenever this source deals damage to any recipient.
    SourceDealsDamage,
    /// "When this creature deals combat damage" (CR 510.2) to any recipient,
    /// creature or player: once per combat damage step it deals any.
    SourceDealsCombatDamage,
    /// "Whenever [subject] deals combat damage to [to]" (CR 510.2), with
    /// prevented damage excluded. `to` is a player matching that actor, or
    /// `None` for any recipient ("deals combat damage"); then each creature
    /// that does triggers once with its total that step as the triggering amount ("that many")
    /// and itself as "it". `per` says whether it triggers once for each
    /// creature that does or once per combat damage step ("one or more").
    DealsCombatDamage {
        subject: TriggerSubject,
        to: Option<Actor>,
        per: Per,
    },
    /// "Whenever [source] deals [kind] damage to [to]" (CR 120.2): once per
    /// committed damage event, prevented damage excluded. The source is any
    /// object that dealt it, a spell included, matched as it is or last was
    /// with its controller (CR 120.2b); an empty predicate is "a source".
    /// The damage dealt is the trigger's amount (`EffectValue::TriggeringAmount`,
    /// "that many"), a damaged player is "that player", and the source is
    /// the triggering object. `DealsCombatDamage` keeps its once-per-batch
    /// combat-hit form for pinned pilots.
    DealsDamage {
        kind: DamageKind,
        source: TriggerSubject,
        to: DamageRecipientFilter,
    },
    /// The inner condition, for an ability that functions only while its
    /// card is in its owner's graveyard (CR 113.6k): "you" is the owner, and
    /// the ability's source is that graveyard object.
    InGraveyard {
        condition: Box<TriggerCondition>,
    },
    /// The inner condition only fires while `active_if` holds for the
    /// ability's controller: a conditionally granted triggered ability
    /// ("has firebending 2 as long as there's a Lesson card in your
    /// graveyard") or a trigger event limited to a game state ("during your
    /// turn"). Intervening-if clauses are `Ability::Triggered::intervening_if`.
    ActiveIf {
        active_if: Condition,
        condition: Box<TriggerCondition>,
    },
}

/// The one vocabulary for "if ..." and "as long as ...": a question asked
/// of a specific controller ("you"). Statics, `ActiveIf` and intervening-if
/// clauses ask it of the game state alone; `Effect::If` asks it as an
/// instruction resolves, when the resolution-only questions
/// ([`Condition::is_resolution_only`]) can also read the resolving spell or
/// ability. Definitions never use those outside `Effect::If`.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Condition {
    Compare {
        left: Box<EffectValue>,
        op: CompareOp,
        right: Box<EffectValue>,
    },
    AtMostHalfStartingLife,
    OwnsMeldPair {
        partner: String,
    },
    YourTurn,
    OpponentControlsMoreCreatures,
    OpponentHasMoreCardsInHand,
    Always,
    CardsDrawnAtLeast {
        count: u32,
    },
    /// "as long as there are `count` or more [predicate] cards in your
    /// graveyard".
    GraveyardAtLeast {
        count: usize,
        predicate: CardPredicate,
    },
    /// "if a graveyard has `count` or more cards in it": any player's
    /// graveyard (Visions of Beyond).
    AnyGraveyardAtLeast {
        count: usize,
    },
    /// "if you control `count` or more [predicate] permanents" (`count` 1:
    /// "if you control a [predicate]").
    ControlsAtLeast {
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        distinct_names: bool,
        count: usize,
        predicate: CardPredicate,
    },
    /// "if there are `count` or more [predicate] permanents on the
    /// battlefield", whoever controls them.
    BattlefieldAtLeast {
        count: usize,
        predicate: CardPredicate,
    },
    /// "if you control `count` or more permanents you don't own" (Agent of
    /// Treachery).
    ControlsUnownedAtLeast {
        count: usize,
    },
    /// "if you have `count` or more cards in hand".
    CardsInHandAtLeast {
        count: usize,
    },
    /// "as long as you have `amount` or more life".
    LifeAtLeast {
        amount: i32,
    },
    /// "as long as you have `count` or more unspent mana" (both pools).
    UnspentManaAtLeast {
        count: u32,
    },
    /// "if you attacked this turn" (raid): you declared one or more
    /// attackers this turn.
    AttackedThisTurn,
    /// "if a land entered the battlefield under your control this turn".
    LandEnteredThisTurn,
    /// "if a [nonland] permanent [who] controlled left the battlefield this
    /// turn" (disappear, void): read from each matching player's turn
    /// history, counted under the controller it last had.
    PermanentLeftThisTurn {
        who: Actor,
        nonland: bool,
    },
    /// "if [who] cast a spell for its warp cost this turn" (void's second
    /// half, CR 702.185).
    WarpedThisTurn {
        who: Actor,
    },
    /// "if you've cast a [predicate] spell this turn", read from the
    /// turn's recorded casts as each spell was cast.
    CastThisTurn(CardPredicate),
    /// "if you've done all four this turn": waterbent, earthbent, firebent
    /// and airbent.
    BentAllFourThisTurn,
    /// "if you've sacrificed `count` or more permanents this turn".
    SacrificedThisTurnAtLeast {
        count: u32,
    },
    /// "if an opponent lost `amount` or more life this turn" (Bloodchief
    /// Ascension): any one opponent's total loss this turn.
    OpponentLostLifeThisTurnAtLeast {
        amount: u32,
    },
    /// At least one alternative holds ("as long as you control a land
    /// creature or a land entered the battlefield under your control this
    /// turn").
    AnyOf(Vec<Condition>),
    AllOf(Vec<Condition>),
    /// The condition does not hold ("during an opponent's turn" is not
    /// `YourTurn`).
    Not(Box<Condition>),
    CreatureDiedThisTurn,
    /// The spell or the spell that became this permanent was kicked (CR 702.33).
    Kicked,
    /// The entering or resolving spell was cast (not a copy), optionally from
    /// a particular zone and/or a zone owned by `owner`.
    CastFrom {
        zone: Option<ZoneType>,
        owner: Option<Actor>,
    },
    /// Resolution only: the frame's primary target is still a permanent and
    /// matches [predicate] ("if that creature is an Ally").
    TargetMatches(CardPredicate),
    /// Resolution only: the frame's primary target was dealt damage this
    /// turn, read from its last known information once it has left ("Exile
    /// target creature. If it was dealt damage this turn, ...").
    TargetDealtDamageThisTurn,
    /// Source question: the ability's source permanent has `count` or more
    /// `kind` counters ("Then if it has four or more quest counters on it").
    SourceCountersAtLeast {
        kind: CounterKind,
        count: i32,
    },
    /// Resolution only: the permanent the resolving Aura or Equipment ability's
    /// source is (or, once the source has left, was) attached to has `count`
    /// or more `kind` counters ("Then if that creature has three or more
    /// +1/+1 counters on it"). False once that permanent has left.
    AttachedCountersAtLeast {
        kind: CounterKind,
        count: i32,
    },
    /// Source question: the ability's source permanent is tapped ("if Katara
    /// is tapped").
    SourceTapped,
    SourceUnprepared,
    /// "while saddled": the source Mount is saddled (CR 702.171b).
    SourceSaddled,
    /// The exact ability source has these current characteristics.
    SourceMatches(CardPredicate),
    /// Triggering question: the permanent whose event fired the trigger has
    /// (or, once it has left, had) counters on it ("if it had counters on
    /// it").
    TriggeringHadCounters,
    /// Triggering question: the permanent whose event fired the trigger
    /// matches [predicate] now, or, once it has left, matched it as it last
    /// existed ("if it was attacking", CR 603.10a).
    TriggeringMatches(CardPredicate),
    /// Resolution only: a card the scope's latest mill, discard or reveal found matches
    /// [predicate] ("When you discard a nonland card this way").
    CardsThisWayMatch(CardPredicate),
    /// Resolution only: the spell whose casting triggered the ability
    /// matches [predicate] ("If that spell is a Lesson").
    TriggeringSpellMatches(CardPredicate),
    /// Resolution only: the frame's primary target was a card in a
    /// graveyard that this resolution exiled, and it matches [predicate]
    /// ("If a creature card is exiled this way").
    TargetCardExiled(CardPredicate),
    /// Resolution only: the resolution's latest `Move`, `Sacrifice` or
    /// `Destroy` moved at least one object ("If you do", "if a creature is
    /// destroyed this way").
    Moved,
    /// Resolution only: the frame's primary target is a permanent the
    /// resolving player controls, or last controlled once it has left ("If
    /// you controlled it").
    TargetControlledByYou,
}

impl TriggerCondition {
    /// "Whenever [who] casts a [predicate] spell", any spell of the turn.
    pub fn casts(who: Actor, predicate: CardPredicate) -> Self {
        TriggerCondition::Casts {
            who,
            predicate,
            nth: None,
            targets: SpellTargetFilter::Any,
        }
    }

    /// "At the beginning of [whose] [step]".
    pub fn beginning_of(step: TurnStep, whose: Actor) -> Self {
        TriggerCondition::BeginningOf { step, whose }
    }

    /// "Whenever [subject] attacks", whatever else attacks with it.
    pub fn attacks(subject: TriggerSubject, per: Per) -> Self {
        TriggerCondition::Attacks {
            subject,
            per,
            attackers: AttackSize::Any,
            nth: None,
        }
    }
}

/// What a cast spell targets, for a cast trigger.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum SpellTargetFilter {
    Any,
    /// At least one of its targets is the trigger's source permanent.
    TargetsThis,
}

impl SpellTargetFilter {
    pub fn is_any(&self) -> bool {
        *self == Self::Any
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CompareOp {
    Eq,
    Ge,
    Le,
    Lt,
    Gt,
}
impl CompareOp {
    pub fn test(self, left: i32, right: i32) -> bool {
        match self {
            Self::Eq => left == right,
            Self::Ge => left >= right,
            Self::Le => left <= right,
            Self::Lt => left < right,
            Self::Gt => left > right,
        }
    }
    pub fn symbol(self) -> &'static str {
        match self {
            Self::Eq => "=",
            Self::Ge => "≥",
            Self::Le => "≤",
            Self::Lt => "<",
            Self::Gt => ">",
        }
    }
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum ContextRead {
    SourcePower,
    /// The permanent or spell whose event triggered the ability, which may
    /// have left its zone by the time the ability resolves.
    TriggeringObject,
    PaidCost,
    /// A characteristic of the source other than power, read as it last
    /// existed once it has left (CR 608.2h).
    SourceLki,
}

impl Condition {
    pub fn not(condition: Condition) -> Self {
        Self::Not(Box::new(condition))
    }

    pub fn compare(
        left: impl Into<EffectValue>,
        op: CompareOp,
        right: impl Into<EffectValue>,
    ) -> Self {
        Self::Compare {
            left: Box::new(left.into()),
            op,
            right: Box::new(right.into()),
        }
    }

    fn reads(&self, read: ContextRead) -> bool {
        match self {
            Self::Compare { left, right, .. } => left.reads(read) || right.reads(read),
            Self::AllOf(items) | Self::AnyOf(items) => items.iter().any(|item| item.reads(read)),
            Self::TriggeringHadCounters
            | Self::TriggeringMatches(_)
            | Self::TriggeringSpellMatches(_) => read == ContextRead::TriggeringObject,
            Self::Not(inner) => inner.reads(read),
            _ => false,
        }
    }
    /// Whether answering reads the asking ability's source permanent or the
    /// permanent that triggered it, so the condition is meaningful only in
    /// `Effect::If` and intervening-if clauses.
    pub fn reads_source(&self) -> bool {
        match self {
            Condition::Compare { left, right, .. } => left.reads_source() || right.reads_source(),
            Condition::Kicked
            | Condition::OwnsMeldPair { .. }
            | Condition::SourceCountersAtLeast { .. }
            | Condition::SourceUnprepared
            | Condition::SourceTapped
            | Condition::SourceSaddled
            | Condition::SourceMatches(_)
            | Condition::TriggeringHadCounters
            | Condition::TriggeringMatches(_) => true,
            Condition::ControlsAtLeast { predicate, .. }
            | Condition::BattlefieldAtLeast { predicate, .. } => predicate.reads_source(),
            Condition::AllOf(conditions) | Condition::AnyOf(conditions) => {
                conditions.iter().any(Condition::reads_source)
            }
            Condition::Not(condition) => condition.reads_source(),
            _ => false,
        }
    }

    /// Whether answering needs a resolving spell or ability, so the
    /// condition is meaningful only inside `Effect::If`.
    pub fn is_resolution_only(&self) -> bool {
        match self {
            Condition::Compare { left, right, .. } => {
                left.is_resolution_only() || right.is_resolution_only()
            }
            Condition::CastFrom { .. }
            | Condition::TargetMatches(_)
            | Condition::AtMostHalfStartingLife
            | Condition::TargetDealtDamageThisTurn
            | Condition::TargetCardExiled(_)
            | Condition::CardsThisWayMatch(_)
            | Condition::TriggeringSpellMatches(_)
            | Condition::Moved
            | Condition::TargetControlledByYou
            | Condition::AttachedCountersAtLeast { .. } => true,
            Condition::AllOf(conditions) | Condition::AnyOf(conditions) => {
                conditions.iter().any(Condition::is_resolution_only)
            }
            Condition::Not(condition) => condition.is_resolution_only(),
            Condition::Kicked
            | Condition::OwnsMeldPair { .. }
            | Condition::SourceCountersAtLeast { .. }
            | Condition::SourceUnprepared
            | Condition::SourceTapped
            | Condition::SourceSaddled
            | Condition::SourceMatches(_)
            | Condition::TriggeringHadCounters
            | Condition::TriggeringMatches(_)
            | Condition::OpponentControlsMoreCreatures
            | Condition::OpponentHasMoreCardsInHand
            | Condition::YourTurn
            | Condition::Always
            | Condition::CardsDrawnAtLeast { .. }
            | Condition::GraveyardAtLeast { .. }
            | Condition::AnyGraveyardAtLeast { .. }
            | Condition::ControlsAtLeast { .. }
            | Condition::BattlefieldAtLeast { .. }
            | Condition::ControlsUnownedAtLeast { .. }
            | Condition::CardsInHandAtLeast { .. }
            | Condition::LifeAtLeast { .. }
            | Condition::UnspentManaAtLeast { .. }
            | Condition::CreatureDiedThisTurn
            | Condition::AttackedThisTurn
            | Condition::LandEnteredThisTurn
            | Condition::PermanentLeftThisTurn { .. }
            | Condition::WarpedThisTurn { .. }
            | Condition::CastThisTurn(_)
            | Condition::BentAllFourThisTurn
            | Condition::SacrificedThisTurnAtLeast { .. }
            | Condition::OpponentLostLifeThisTurnAtLeast { .. } => false,
        }
    }
}

/// The player who performs a trigger's event, relative to the ability's
/// controller ("you"). Resolution-time players are `PlayerSelector`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Actor {
    You,
    Opponent,
    AnyPlayer,
}

impl Actor {
    /// Whether `player` is this actor for an ability controlled by `you`.
    pub fn includes(self, you: PlayerId, player: PlayerId) -> bool {
        match self {
            Actor::You => player == you,
            Actor::Opponent => player != you,
            Actor::AnyPlayer => true,
        }
    }
}

/// Which damage a `TriggerCondition::DealsDamage` watches (CR 120.2a, b).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum DamageKind {
    Any,
    Combat,
    Noncombat,
}

impl DamageKind {
    pub fn includes(self, combat: bool) -> bool {
        match self {
            Self::Any => true,
            Self::Combat => combat,
            Self::Noncombat => !combat,
        }
    }
}

/// What a `TriggerCondition::DealsDamage` event must damage: a player the
/// actor names, or anything (a player, creature, planeswalker or battle).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum DamageRecipientFilter {
    Player(Actor),
    Any,
}

/// A step whose beginning triggers abilities (CR 503.1a, 504.2, 505.4,
/// 507.1, 513.1). `FirstMain` is the precombat main phase; `Combat` is the
/// beginning of combat step.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum TurnStep {
    Upkeep,
    Draw,
    FirstMain,
    Combat,
    End,
}

/// A keyword action a `TriggerCondition::Performs` event watches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum KeywordAction {
    /// Searching one's library (CR 701.23).
    Search,
    Scry,
    Surveil,
    Waterbend,
    Earthbend,
    Firebend,
    Airbend,
    /// Discarding a card (CR 701.9): once for each card put into its
    /// owner's graveyard from their hand.
    Discard,
}

impl KeywordAction {
    /// "waterbend, earthbend, firebend, or airbend".
    pub const BENDING: [KeywordAction; 4] = [
        KeywordAction::Waterbend,
        KeywordAction::Earthbend,
        KeywordAction::Firebend,
        KeywordAction::Airbend,
    ];
}

/// How often an event over several objects at once triggers: once for each
/// matching object, or once for the batch ("one or more").
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Per {
    Object,
    Batch,
}

/// How many creatures in all an `Attacks` trigger requires to have been
/// declared as attackers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum AttackSize {
    Any,
    /// Exactly one (CR 506.5).
    Alone,
    /// At least this many declared creatures, including the matching attacker.
    AtLeast(u8),
}

impl AttackSize {
    pub fn matches(self, declared: usize) -> bool {
        match self {
            Self::Any => true,
            Self::Alone => declared == 1,
            Self::AtLeast(n) => declared >= usize::from(n),
        }
    }
}

/// Which game objects an event-based trigger condition watches.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum TriggerSubject {
    /// The source itself (even a token copy), or another matching subject.
    ThisOr(Box<TriggerSubject>),
    /// The permanent this ability is on.
    This,
    /// "enchanted creature" or "equipped creature": the object the source
    /// Aura or Equipment is attached to.
    Attached,
    /// A creature dealt damage by this exact source this turn.
    CreatureDamagedByThis,
    /// "a/another [predicate] [controlled_by controls]": an object whose
    /// controller is `controlled_by`, relative to the ability's controller.
    /// `another` excludes the source itself (CR 109.4).
    Matching {
        controlled_by: Actor,
        predicate: Box<CardPredicate>,
        another: bool,
    },
}

impl TriggerSubject {
    /// "a/another [predicate] [controlled_by controls]".
    pub fn matching(controlled_by: Actor, predicate: CardPredicate, another: bool) -> Self {
        Self::Matching {
            controlled_by,
            predicate: Box::new(predicate),
            another,
        }
    }

    /// "a [predicate] you control", this included.
    pub fn you_control(predicate: CardPredicate) -> Self {
        Self::matching(Actor::You, predicate, false)
    }

    /// "another [predicate] you control".
    pub fn another_you_control(predicate: CardPredicate) -> Self {
        Self::matching(Actor::You, predicate, true)
    }

    /// "a [predicate]", whoever controls it, this included.
    pub fn any(predicate: CardPredicate) -> Self {
        Self::matching(Actor::AnyPlayer, predicate, false)
    }

    /// "another [predicate]", whoever controls it.
    pub fn another(predicate: CardPredicate) -> Self {
        Self::matching(Actor::AnyPlayer, predicate, true)
    }
}

/// The players an effect affects, relative to its controller.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PlayerSelector {
    /// The effect's controller ("you").
    You,
    /// Each opponent of the controller.
    EachOpponent,
    /// Each player, in APNAP order (CR 101.4).
    EachPlayer,
    /// The trigger's player ("they": the caster of the triggering spell).
    TriggeringPlayer,
    /// "each of that player's opponents" (Standstill): the opponents of the
    /// trigger's player, whoever controls the effect.
    TriggeringPlayersOpponents,
    /// A chosen target player (boxed: a spec is far larger than the rest).
    Target(Box<TargetSpec>),
    /// The legal player target of a separately declared (zero-based) clause.
    TargetClause(usize),
    /// The controller of the frame's primary target permanent or spell, as
    /// it is or last existed ("its controller").
    TargetController,
    /// The owner of the object an instruction moves ("under its owner's
    /// control"); only [`Effect::Move`] reads it.
    Owner,
}

impl PlayerSelector {
    pub fn target_spec(&self) -> Option<&TargetSpec> {
        match self {
            Self::Target(spec) => Some(spec),
            Self::You
            | Self::EachOpponent
            | Self::EachPlayer
            | Self::TriggeringPlayer
            | Self::TriggeringPlayersOpponents
            | Self::TargetClause(_)
            | Self::TargetController
            | Self::Owner => None,
        }
    }
}

/// A permanent characteristic, read live or at its last battlefield appearance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ObjectProperty {
    Power,
    Toughness,
    ManaValue,
    /// How many of the five colors it is (CR 105.2), `0..=5`.
    Colors,
    /// How many counters of this kind are on it ("the number of soul
    /// counters on this artifact").
    Counters(CounterKind),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Aggregate {
    Sum,
    Max,
}

impl Aggregate {
    pub fn evaluate(self, values: impl Iterator<Item = i32>) -> i32 {
        match self {
            Self::Sum => values.sum(),
            Self::Max => values.max().unwrap_or(0),
        }
    }
}

/// A resolution-time quantity relative to the effect controller.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum EffectValue {
    Property {
        of: Box<ObjectSelector>,
        property: ObjectProperty,
        aggregate: Aggregate,
    },
    LifeTotal(PlayerSelector),
    LifeGainedThisTurn(PlayerSelector),
    LifeLostThisTurn(PlayerSelector),
    /// How much more life the players have than their starting life totals
    /// (CR 103.4): "at least 10 life more than your starting life total";
    /// negative when below it.
    LifeAboveStarting(PlayerSelector),
    SpellManaSpent(Box<ObjectSelector>),
    SpellManaValue(Box<ObjectSelector>),
    Fixed(i32),
    /// The resolving spell's announced X (CR 107.3).
    X,
    NegativeX,
    /// Creatures `controller` controls with +1/+1 counters on them (Dai Li Agents).
    CreaturesWithCounters,
    BattlefieldMatching {
        predicate: CardPredicate,
    },
    GraveyardMatching {
        base: i32,
        predicate: CardPredicate,
    },
    CardsDrawnMinus {
        subtract: u32,
    },
    /// The greatest number of creatures you control that share a creature
    /// type (CR 205.3m).
    GreatestSharedCreatureType,
    /// Number of different mana values among the selected battlefield permanents.
    DistinctManaValues {
        of: Box<ObjectSelector>,
    },
    /// Excess damage (CR 120.4a) the resolution's latest damage instruction
    /// dealt to its creature; 0 outside a resolution.
    ExcessDamage,
    /// Actual amount of the damage event that triggered this ability.
    TriggerDamage,
    /// Amount from the event that triggered this ability (damage dealt or life lost).
    TriggeringAmount,
    /// Total life actually lost to this scope's latest LoseLife instruction.
    LifeLostThisWay,
    /// The source permanent's current power ("where X is this creature's
    /// power"); last-known power after departure, 0 outside a resolution.
    SourcePower,
    /// The counters on the source as it died, less a stated amount.
    DeathCountersMinus {
        kind: CounterKind,
        subtract: i32,
    },
    /// The power of the creature the source is attached to ("where X is
    /// equipped creature's power"), at least 0; 0 when it is attached to none.
    AttachedCreaturePower,
    /// Attacking creatures `controller` controls matching `predicate`, as
    /// the effect resolves ("search for up to that many").
    AttackingMatching {
        predicate: CardPredicate,
    },
    /// Permanents `controller` has sacrificed this turn.
    PermanentsSacrificedThisTurn,
    /// "The number of spells you've cast this turn."
    SpellsCastThisTurn,
    /// The total power of the creatures tapped to pay the resolving
    /// ability's cost ("charge counters equal to its power", station, CR
    /// 702.184a), each read as the ability resolves or from its last known
    /// information once it has left; 0 outside a resolution.
    PowerTappedToPay,
    /// The power of the permanent whose event fired the trigger, from its
    /// last known information once it has left ("where X is that
    /// creature's power").
    TriggeringPower,
    /// The mana value of the spell whose casting fired the trigger ("you lose
    /// life equal to its mana value"), its announced X included while it is
    /// on the stack.
    TriggeringManaValue,
    /// The toughness of the frame's primary target, from its last known
    /// information once it has left ("you gain life equal to its
    /// toughness").
    TargetToughness,
    TargetPower,
    /// Permanents `controller` owns that an opponent controls.
    OwnedPermanentsOpponentsControl,
    /// Half the inner value, rounded down ("half X cards, rounded down").
    Half(Box<EffectValue>),
    /// `factor` times the inner value ("2 life for each Gate you control"):
    /// one quantity, so one event (CR 119.9) rather than `factor` of them.
    Scaled {
        factor: i32,
        value: Box<EffectValue>,
    },
    /// The greatest power among creatures `controller` controls that match
    /// [predicate] ("non-Human creatures you control"); 0 with none.
    GreatestPowerYouControl {
        predicate: CardPredicate,
    },
    /// Cards in `controller`'s hand.
    CardsInHand,
    /// Number of objects named by a resolution result or selector.
    Objects {
        of: Box<ObjectSelector>,
        /// Count only matching card objects when present, using their current
        /// characteristics. Unfiltered `Moved` counts the committed moves
        /// even if a successor has since changed zones again.
        #[serde(skip_serializing_if = "Option::is_none")]
        predicate: Option<CardPredicate>,
        /// Count only objects now in this zone: "put into a graveyard this
        /// way" excludes a milled card a replacement exiled instead (CR
        /// 614.6), "return it to your hand" one that went elsewhere.
        #[serde(skip_serializing_if = "Option::is_none")]
        in_zone: Option<ZoneType>,
    },
    /// "The amount of mana spent to cast" the resolving spell, or the
    /// permanent whose entering triggered the resolving ability (CR 601.2h):
    /// mana actually paid, additional costs included; 0 for anything else.
    ManaSpentToCast,
    /// Converge: the number of colors of that mana.
    ColorsSpentToCast,
}

/// A resolving obligation, distinct from the effects chosen by its outcome.
/// Waterbend pays only generic mana; taps never count as mana production.
/// Life is paid in one action by a player whose life total is at least the
/// amount (CR 119.4); no mana is owed for it.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum PaymentCost {
    Mana(ManaCost),
    Waterbend { generic: u8 },
    Life(u32),
    Discard,
}

impl PaymentCost {
    /// The mana this cost states, or `None` for a nonmana payment. For readers of a definition (a ward's price); a
    /// payment being made owes a `CostComponent`.
    pub fn mana(&self) -> Option<ManaCost> {
        match self {
            Self::Mana(cost) => Some(cost.clone()),
            Self::Waterbend { generic } => Some(ManaCost::default().with_generic(*generic)),
            Self::Life(_) | Self::Discard => None,
        }
    }
}

/// Where the revealed cards of a distinct-name search go.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum DistinctSearchEnd {
    /// An opponent chooses `picks` of them for the graveyard; the rest enter
    /// the battlefield tapped, then the searcher shuffles.
    OpponentSplits { picks: usize },
    /// Into the searcher's hand, then the searcher shuffles.
    Hand,
}

/// Counters a zone move puts on the entering permanent. A single counter
/// retains its original bare-kind representation in card learning programs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntryCounters {
    pub kind: CounterKind,
    pub count: EffectValue,
}

impl From<CounterKind> for EntryCounters {
    fn from(kind: CounterKind) -> Self {
        Self {
            kind,
            count: EffectValue::Fixed(1),
        }
    }
}

impl serde::Serialize for EntryCounters {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if self.count == EffectValue::Fixed(1) {
            self.kind.serialize(serializer)
        } else {
            (&self.kind, &self.count).serialize(serializer)
        }
    }
}

/// Which end of a library receives a moved or ordered card.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum LibraryEnd {
    Top,
    Bottom,
}

/// Which objects a verb acts on, read as the instruction resolves
/// (`Game::select`). A verb acts only on the objects it can: counters, taps
/// and sacrifices need battlefield permanents, a move any card object, a
/// continuous effect a permanent or the player `You` names.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ObjectSelector {
    /// The source: the exact permanent, graveyard card or spell the ability
    /// or spell belongs to. Once the source has left the battlefield, the
    /// card it became in the graveyard or exile it went to (CR 400.7,
    /// 603.10a: "return this card", "exile it"); never a later object.
    This,
    /// The bound source object only while that incarnation is live.
    ExactSource,
    /// The clause's chosen target. The spec declares the implicit targeting
    /// clause of an effect list without an explicit `TargetClause`.
    Target(TargetSpec),
    /// All eligible targets of the active clause, acted on by one instruction
    /// ("put a counter on each target"). Unlike `ForEachTarget`, this does not
    /// run a separate instruction sequence or commit a separate event per slot.
    Targets(TargetSpec),
    /// The clause's chosen target permanent and every other permanent with
    /// the same name as it (CR 201.2a: an empty name matches nothing), fixed
    /// together so one instruction acts on all of them at once ("destroy
    /// target nonland permanent and all other permanents with the same name
    /// as that permanent"). Nothing once the target is illegal (CR 608.2b).
    TargetAndSameName(TargetSpec),
    /// The permanent whose event fired the trigger ("that creature", "that
    /// token"), which the trigger keeps as its context.
    Triggering,
    /// Objects captured when a delayed trigger was created, still in the same
    /// incarnation; separate from the object supplied by its triggering event.
    Remembered,
    /// Cards in exile linked to the exact ability source (CR 607.2a).
    ExiledWithSource,
    /// The object the source Aura or Equipment is attached to.
    Attached,
    /// The token this resolution created last ("that token").
    CreatedThisWay,
    /// The exact permanent that granted the resolving ability ("return
    /// Trusty Boomerang"); nothing once it has left the battlefield.
    Grantor,
    /// The objects the resolution's latest `Move` or `Sacrifice` moved, as
    /// they now exist ("then return it to the battlefield").
    Moved,
    /// Cards moved by the latest mill or discard in this scope. Later Move
    /// instructions do not replace this distinct instruction result.
    CardsThisWay,
    /// Exact objects chosen by the enclosing SelectCards operation.
    Selected,
    /// The objects that paid the resolving spell's or ability's cost by
    /// being `How` ("the sacrificed creature", CR 602.2b). Property reads
    /// use each object's last known information once it has left; the
    /// selection names only those still in their paying appearance.
    Paid(How),
    /// The objects `of` names that match [predicate], without a choice
    /// ("put all instant and sorcery cards from among them into your hand").
    Matching {
        of: Box<ObjectSelector>,
        predicate: CardPredicate,
    },
    /// Cards the resolving player owns in a zone. Hidden selection retains
    /// the zone provenance across sampled worlds.
    Zone(ZoneType),
    /// Cards the clause's chosen target player owns in a zone ("search target
    /// opponent's library"); `spec` declares that player target. The
    /// resolving player selects, and only that zone is revealed to them.
    TargetPlayersZone {
        spec: Box<TargetSpec>,
        zone: ZoneType,
    },
    /// The resolving player ("you"), as whom a continuous effect affects
    /// ("you have no maximum hand size"), damage, or discard. Object-only
    /// verbs find nothing because a player is not a card object.
    You,
    /// A permanent the resolving player controls matching [predicate],
    /// chosen by them as the instruction resolves ("sacrifice a creature",
    /// "tap a creature you control"); the instruction suspends on the choice
    /// (`Decision::ChooseObject`) and runs over it.
    Chosen {
        predicate: CardPredicate,
        #[serde(skip_serializing_if = "Option::is_none")]
        except: Option<Except>,
    },
    /// Every battlefield permanent in `of` matching [predicate], fixed as
    /// the instruction resolves ("each creature you control", "all other
    /// creatures"); `max_mana_value` bounds the mana value as it resolves
    /// ("with mana value X or less").
    Each {
        of: Scope,
        predicate: CardPredicate,
        #[serde(skip_serializing_if = "Option::is_none")]
        except: Option<Except>,
        #[serde(skip_serializing_if = "Option::is_none")]
        max_mana_value: Option<Box<EffectValue>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        exact_mana_value: Option<Box<EffectValue>>,
    },
    /// Every card matching [predicate] in a public card zone owned by
    /// `owner`, fixed as the instruction resolves, in seat order ("all
    /// creature cards from all graveyards"). Ranges over physical cards by
    /// owner, unlike `Each`'s permanents by controller; a hidden zone is
    /// `Zone`, whose selection keeps sampling provenance.
    ZoneMatching {
        zone: ZoneType,
        owner: Actor,
        predicate: CardPredicate,
    },
}

/// Whose battlefield permanents an [`ObjectSelector::Each`] ranges over.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Scope {
    /// Permanents the resolving player controls.
    YouControl,
    /// Permanents the resolving player's opponents control.
    OpponentsControl,
    /// Every permanent on the battlefield.
    Battlefield,
    /// Permanents the frame's primary target player controls ("creatures
    /// target player controls").
    TargetPlayerControls,
    /// Permanents controlled by a player the exact source hit in combat this turn.
    DamagedBySourceInCombat,
}

/// The one permanent a `Chosen` or `Each` selector leaves out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Except {
    /// "another": the source permanent.
    This,
    /// "all other": the clause's chosen target (Avatar's Wrath).
    Target,
}

impl ObjectSelector {
    pub fn property(self, property: ObjectProperty, aggregate: Aggregate) -> EffectValue {
        EffectValue::Property {
            of: Box::new(self),
            property,
            aggregate,
        }
    }

    /// Count every object named by this selector.
    pub fn count(self) -> EffectValue {
        EffectValue::Objects {
            of: Box::new(self),
            predicate: None,
            in_zone: None,
        }
    }

    /// "Each [predicate] you control".
    pub fn each_you_control(predicate: CardPredicate) -> Self {
        Self::each(Scope::YouControl, predicate)
    }

    /// "Each [predicate]" on the battlefield.
    pub fn each(of: Scope, predicate: CardPredicate) -> Self {
        Self::Each {
            of,
            predicate,
            except: None,
            max_mana_value: None,
            exact_mana_value: None,
        }
    }

    /// "Each other [predicate] you control".
    pub fn each_other_you_control(predicate: CardPredicate) -> Self {
        Self::Each {
            of: Scope::YouControl,
            predicate,
            except: Some(Except::This),
            max_mana_value: None,
            exact_mana_value: None,
        }
    }

    /// "A [predicate] you control", chosen as the instruction resolves.
    pub fn chosen(predicate: CardPredicate) -> Self {
        Self::Chosen {
            predicate,
            except: None,
        }
    }

    /// "Another [predicate] you control", chosen as the instruction resolves.
    pub fn another_chosen(predicate: CardPredicate) -> Self {
        Self::Chosen {
            predicate,
            except: Some(Except::This),
        }
    }

    pub fn target_spec(&self) -> Option<&TargetSpec> {
        match self {
            Self::Target(spec) | Self::Targets(spec) | Self::TargetAndSameName(spec) => Some(spec),
            Self::TargetPlayersZone { spec, .. } => Some(spec),
            Self::Matching { of, .. } => of.target_spec(),
            Self::Each {
                of: Scope::TargetPlayerControls,
                ..
            } => Some(&TARGET_PLAYER),
            Self::This
            | Self::ExactSource
            | Self::Triggering
            | Self::Attached
            | Self::CreatedThisWay
            | Self::Remembered
            | Self::ExiledWithSource
            | Self::Grantor
            | Self::Moved
            | Self::CardsThisWay
            | Self::Selected
            | Self::Paid(_)
            | Self::Zone(_)
            | Self::You
            | Self::Chosen { .. }
            | Self::Each { .. }
            | Self::ZoneMatching { .. } => None,
        }
    }

    /// Whether the selector asks the resolving player to choose.
    pub fn is_chosen(&self) -> bool {
        matches!(self, Self::Chosen { .. })
    }
}

/// How a card a [`Effect::Move`] exiles stays linked to the exact source
/// (CR 607.2a); either link lasts only while the source is on the
/// battlefield as it exiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum LinkedExile {
    /// "Exiled with this creature": the link outlives the source, so a
    /// later `CastExiledWithSource` still finds the card.
    WithSource,
    /// "Until this creature leaves the battlefield": the card returns at
    /// once (no stack) when the source leaves. The implicit targeting clause
    /// is "up to one".
    UntilSourceLeaves,
}

/// Who chooses the card an `ExileLinked` effect exiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum LinkedChooser {
    Controller,
    TargetPlayer,
}

/// How the effect's controller may cast a card it exiled, during their turn
/// and while they control its source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum LinkedGrant {
    /// "by waterbending {X} rather than paying its mana cost, where X is its
    /// mana value" (Hama, the Bloodbender).
    WaterbendManaValue,
    /// "as though they had flash. Mana of any type can be spent" (Azula,
    /// Cunning Usurper).
    FlashAnyMana,
}

/// The next interval actually taken by a controlled player (CR 720).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum PlayerControlDuration {
    NextCombat,
    NextTurn,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Effect {
    /// A continuous effect from this resolution (CR 611): `modification`
    /// for each permanent `affected` names, or for the player `You` names,
    /// the set fixed as it resolves (CR 611.2c); amounts are read then.
    ApplyContinuous {
        affected: ObjectSelector,
        modification: Modification<EffectValue>,
        duration: EffectDuration,
    },
    /// Put `amount` counters of `kind` on each permanent `on` names,
    /// `amount` read as the instruction resolves.
    PutCounters {
        on: ObjectSelector,
        kind: CounterKind,
        amount: EffectValue,
    },
    /// "Put its counters on [on]": every counter `from` has, or had as it
    /// last existed on the battlefield (CR 603.10a), kind by kind.
    PutCountersFrom {
        from: ObjectSelector,
        on: ObjectSelector,
    },
    /// The resolving player sacrifices each permanent `what` names that
    /// they control (CR 701.17a); a `Chosen` selector asks them to pick one,
    /// and only then may `optional` let them decline ("you may sacrifice
    /// a creature"). What was sacrificed is read by a following "if you do"
    /// ([`Condition::Moved`]).
    Sacrifice {
        what: ObjectSelector,
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        optional: bool,
    },
    /// Move each object `what` names to `to`: a permanent, a graveyard or
    /// exiled card, or a spell on the stack (which ceases to be cast). A
    /// card entering the battlefield enters under the resolving player's
    /// control (CR 110.2b) unless `under` names another player, `tapped`
    /// and `with_counter` as stated; one moved to a library is shuffled
    /// into its owner's. `linked` keeps an exiled card linked to the exact
    /// source. A card leaving a graveyard for a hand is revealed. The moved
    /// objects become [`ObjectSelector::Moved`] and answer
    /// [`Condition::Moved`].
    Move {
        /// Move a hidden card into hand without revealing its identity.
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        unrevealed: bool,
        /// A stated end replaces the usual shuffle-into-library movement.
        #[serde(skip_serializing_if = "Option::is_none")]
        library_end: Option<LibraryEnd>,
        what: ObjectSelector,
        to: ZoneType,
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        tapped: bool,
        /// Enter with the back face up; a card without one stays where it is.
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        transformed: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        under: Option<PlayerSelector>,
        #[serde(skip_serializing_if = "Option::is_none")]
        with_counter: Option<EntryCounters>,
        #[serde(skip_serializing_if = "Option::is_none")]
        linked: Option<LinkedExile>,
    },
    /// Tap each permanent `what` names.
    Tap {
        what: ObjectSelector,
    },
    Prepare {
        what: ObjectSelector,
    },
    Unprepare {
        what: ObjectSelector,
    },
    /// Untap each permanent `what` names.
    Untap {
        what: ObjectSelector,
    },
    /// Remove every counter from each permanent `from` names.
    RemoveAllCounters {
        from: ObjectSelector,
    },
    /// Create a token that's a copy of each object `of` names (CR 707.2,
    /// 111.1): the source (`This`), the target, or every permanent of an
    /// `Each` ("for each creature target player controls, create a token
    /// that's a copy of that creature"), all copied as the instruction
    /// resolves and created simultaneously. The token copies the copiable values: the
    /// definition of the face that's up, including any copy effect already
    /// applied, and never counters, damage, tapped state or non-copy
    /// effects. A copy of a double-faced permanent has both faces and enters
    /// transformed if that permanent is (CR 707.8a). An object that has left
    /// its zone is copied as it last existed there ("a copy of it" after
    /// exiling it). `exceptions` ("except it's not legendary") become part of
    /// the token's own copiable values (CR 707.9b), so a copy of it inherits
    /// them.
    CreateTokenCopy {
        #[serde(skip_serializing_if = "is_fixed_one")]
        count: EffectValue,
        of: ObjectSelector,
        exceptions: Vec<CopyException<EffectValue>>,
    },
    OptionalPayment {
        cost: PaymentCost,
        if_paid: Vec<Effect>,
        if_declined: Vec<Effect>,
    },
    /// Cumulative upkeep's payment (CR 702.24a): sacrifice this unless you
    /// pay `each` for each age counter on it. The age counter is put on by a
    /// preceding instruction.
    CumulativeUpkeep {
        each: ManaCost,
    },
    /// Each opponent loses `amount` life and you gain that much.
    DrainLife {
        amount: EffectValue,
    },
    /// Mill `count` cards (CR 701.17a), all milled cards one simultaneous
    /// move bound as `CardsThisWay`.
    MillCards {
        count: EffectValue,
        /// The target player mills instead of the controller.
        target: Option<TargetSpec>,
        /// The players who mill ("each player mills X cards"), in APNAP
        /// order; `None` means `target`, else the controller.
        #[serde(skip_serializing_if = "Option::is_none")]
        player: Option<PlayerSelector>,
    },
    /// For each selected player: reveal cards from the top of their library
    /// until a card matching `predicate` is revealed or the library runs
    /// out, then put every revealed card into their graveyard (Consuming
    /// Aberration). All players' cards move as one batch bound as
    /// `CardsThisWay`.
    MillUntil {
        player: PlayerSelector,
        predicate: CardPredicate,
    },
    /// Choose a subset of exact objects from `from`, then run `effects` in
    /// a child scope where `Selected` names them. An empty selection runs
    /// `if_declined`; the caller retains its earlier results and suffix.
    SelectCards {
        from: ObjectSelector,
        predicate: CardPredicate,
        min: usize,
        max: EffectValue,
        effects: Vec<Effect>,
        if_declined: Vec<Effect>,
    },
    /// A distinct targeting clause. Effects within it refer to the same
    /// chosen objects; different clauses may choose the same object.
    TargetClause {
        requirement: TargetRequirement,
        effects: Vec<Effect>,
    },
    /// Internal cursor emitted when clauses are lowered into a frame queue.
    SelectTargetRequirement {
        index: usize,
    },
    /// The player `what` names discards `count` cards, chosen by them.
    Discard {
        what: ObjectSelector,
        #[serde(skip_serializing_if = "is_one")]
        count: usize,
    },
    /// Discard the selected hands as one simultaneous event.
    DiscardHand {
        who: PlayerSelector,
    },
    /// Destroy each permanent `what` names (CR 701.7), as one simultaneous
    /// event: leave-the-battlefield abilities look back at the permanents
    /// as they all were before (CR 603.10a).
    Destroy {
        what: ObjectSelector,
    },
    /// Airbend (CR 701.65) each permanent `what` names: exile it; its owner
    /// may cast that exact exiled card for {2} rather than its mana cost.
    Airbend {
        what: ObjectSelector,
    },
    /// "After this phase, there is an additional combat phase [in which
    /// only `attackers` can attack]" (CR 500.8).
    AdditionalCombatPhase {
        #[serde(skip_serializing_if = "Option::is_none")]
        attackers: Option<CardPredicate>,
    },
    /// "Starting with you, each player chooses up to one [predicate]
    /// permanent from among permanents your opponents control. Exile those
    /// permanents."
    EachPlayerChoosesToExile {
        predicate: CardPredicate,
    },
    /// "Choose a card name. Search [the target player]'s graveyard, hand and
    /// library for up to `count` cards with that name and exile them. Then
    /// that player shuffles." The name is a resolution choice among the
    /// world's card names (one mode each, `ExileNamed`).
    NameAndExileFromTargetPlayer {
        count: usize,
    },
    /// Privately search the target player's graveyard, hand and library,
    /// choosing zero through `count` cards named `name` to exile; then shuffle.
    ExileNamed {
        name: String,
        count: usize,
    },
    /// Choose X and optionally pay that much generic mana during resolution.
    /// The paid branch reads X from the frame; `WhenYouDo` retains it.
    OptionalManaPaymentX {
        if_paid: Vec<Effect>,
    },
    /// "You may exile it": the card the triggering creature became in its
    /// graveyard, exiled with this ability's source (CR 607.2a).
    MayExileTriggeringWithSource,
    /// "Choose a [predicate] card exiled with [this]": one mode per such
    /// card, each `HaveAbilitiesOf` it.
    ChooseExiledWithSource {
        predicate: CardPredicate,
    },
    /// The source permanent has all activated and triggered abilities of
    /// `definition` from now on, until a later choice (CR 613.1f).
    HaveAbilitiesOf {
        definition: crate::card::CardDefId,
    },
    /// "Take an extra turn after this one" (CR 500.7): the resolving
    /// player's, before any extra turn created earlier.
    TakeExtraTurn,
    /// "End the turn" (CR 723.1): every object on the stack is exiled,
    /// waiting triggered abilities cease to exist, and the turn skips to its
    /// cleanup step, which proceeds as usual.
    EndTheTurn,
    LoseGame,
    WinGame,
    /// Control the selected player during their next actual turn or combat.
    /// This changes decision authority, never ownership of their resources.
    ControlPlayer {
        what: ObjectSelector,
        duration: PlayerControlDuration,
    },
    /// "Transform [this]" (CR 701.27): the source permanent turns over and
    /// stays the same object (CR 701.27a). Nothing happens to a permanent that
    /// isn't a double-faced card (CR 701.27c).
    TransformSource,
    ExileMeldPair {
        partner: String,
    },
    MeldExiled {
        fronts: [String; 2],
        result: String,
    },
    /// "[player] gains control of [what]" for `duration` (CR 613.1b).
    GainControl {
        player: PlayerSelector,
        what: ObjectSelector,
        duration: EffectDuration,
    },
    /// Until the controller's next turn, opponents cast spells only from
    /// their hands.
    OpponentsCastOnlyFromHand,
    /// The controller may cast the targeted graveyard card this turn at its normal cost and timing.
    GrantGraveyardCastThisTurn {
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        flashback: bool,
        target: TargetSpec,
    },
    /// "Exile the top `count` cards of your library [face down]. You may
    /// play them [`until`] [if `condition`]." A face-down card is hidden
    /// from everyone, its exiler included, while the condition fails
    /// (Flameshape).
    ExileTopCardsToPlay {
        count: usize,
        until: PlayWindow,
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        face_down: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        condition: Option<Condition>,
    },
    /// A player exiles a matching card; its controller may cast it as linked.
    ExileLinked {
        chooser: LinkedChooser,
        zone: ZoneType,
        predicate: CardPredicate,
        up_to: bool,
        grant: LinkedGrant,
        target: TargetSpec,
    },
    /// Exile cards from the top of your library until you exile a nonland
    /// card; you may cast it without paying its mana cost if its mana value
    /// is less than `below`, otherwise put it into your hand.
    ExileUntilNonlandCastFree {
        below: EffectValue,
    },
    /// Cascade (CR 702.85a): exile cards from the top of your library until
    /// you exile a nonland card with lesser mana value than the source spell;
    /// you may cast it without paying its mana cost. Every other card exiled
    /// this way, and that one if not cast, goes on the bottom in a random
    /// order.
    Cascade,
    ExileEachLibraryTopCastFree,
    /// "Look at the top card of your library. You may cast that card without
    /// paying its mana cost." Inside `MayOnceEachTurn`, casting spends the
    /// option.
    CastLibraryTopFree,
    /// You may cast a matching spell from your hand without paying its mana cost.
    CastFromHandFree {
        predicate: CardPredicate,
    },
    CastFromGraveyard {
        predicate: CardPredicate,
        exile_instants_and_sorceries: bool,
    },
    /// Shuffle your graveyard into your library.
    ShuffleGraveyardIntoLibrary,
    /// The target's owner chooses where in their library it goes: second
    /// from the top or the bottom (modes of [`Effect::PlaceTargetInLibrary`]),
    /// or with `top`, the top or the bottom.
    OwnerPutsIntoLibrary {
        target: TargetSpec,
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        top: bool,
    },
    /// Reveal each object `what` names (CR 701.20): its definition becomes
    /// public, and a later library placement in this resolution keeps that
    /// knowledge ("reveal it, then shuffle and put that card on top").
    Reveal {
        what: ObjectSelector,
    },
    /// "Reveal cards from the top of your library until you reveal a
    /// [predicate] card. Put that card into your hand and the rest on the
    /// bottom of your library in a random order." The cards stay in the
    /// library while revealed; with no match, every card is revealed and the
    /// library is put back in a random order.
    RevealUntilToHand {
        predicate: CardPredicate,
    },
    /// The target player reveals their hand (it becomes public knowledge);
    /// the effect's controller chooses a matching card and that player
    /// discards it.
    RevealHandDiscardChosen {
        target: TargetSpec,
        predicate: CardPredicate,
    },
    /// Search your library for up to `count` [predicate] cards with
    /// different names, revealing each, then `end` places them (Elemental
    /// Teachings, Three Dreams).
    SearchDistinctNames {
        predicate: CardPredicate,
        count: usize,
        end: DistinctSearchEnd,
    },
    /// Put the primary target into its owner's library, second from the top
    /// or on the bottom.
    PlaceTargetInLibrary {
        second_from_top: bool,
    },
    /// "You may [effects]. Do this only once each turn." Offered only while
    /// the exact source has not taken this ability's option this turn; only
    /// accepting spends it (the ability itself triggers every time).
    MayOnceEachTurn {
        effects: Vec<Effect>,
    },
    /// A reflexive trigger (CR 603.12): "When you do, [effects]." It
    /// triggers where it appears, so it only runs if the instruction before
    /// it happened; its targets are chosen as it is put on the stack.
    WhenYouDo {
        effects: Vec<Effect>,
    },
    /// "If [condition], [then]; otherwise, [otherwise]", asked as this
    /// instruction resolves (CR 608.2c), of the resolving player and frame.
    If {
        condition: Condition,
        then: Vec<Effect>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        otherwise: Vec<Effect>,
    },
    /// The source deals `amount` damage to each creature, permanent or
    /// player `what` names (a `Target` selector may name a player),
    /// `amount` read as the instruction resolves; the excess dealt to the
    /// last creature is kept for a following "the excess damage". `by`
    /// names another object as the source of the damage (CR 120.2: "it deals
    /// damage equal to its power"), as it is or last existed.
    DealDamage {
        amount: EffectValue,
        what: ObjectSelector,
        #[serde(skip_serializing_if = "Option::is_none")]
        by: Option<ObjectSelector>,
    },
    /// Counter each spell `what` names (CR 701.6a); abilities are not
    /// affected.
    CounterSpell {
        what: ObjectSelector,
        /// "If that spell is countered this way, exile it instead of
        /// putting it into its owner's graveyard."
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        exile: bool,
    },
    /// "Counter target activated or triggered ability" (CR 701.6b): the
    /// ability leaves the stack and does nothing; spells are not affected.
    /// A stack ability is no card object, so the clause's chosen target is
    /// read directly.
    CounterAbility {
        what: ObjectSelector,
    },
    /// Resolving player draws `count` cards, read at resolution. No target.
    DrawCards {
        count: EffectValue,
    },
    /// The selected players lose `amount` life (not damage, CR 119.3).
    LoseLife {
        player: PlayerSelector,
        amount: EffectValue,
    },
    GivePlayerCounters {
        player: PlayerSelector,
        kind: CounterKind,
        count: i32,
    },
    /// The source deals `amount` damage to the selected players.
    DealDamageToPlayer {
        player: PlayerSelector,
        amount: EffectValue,
    },
    /// Create `count` tokens from the token definition keyed `token`
    /// (`token.<name>`), under the resolving player's control, `count` read
    /// as the instruction resolves. Serialized as `token_name`, the learning
    /// schema's field for the created token's definition reference.
    CreateToken {
        #[serde(rename = "token_name")]
        token: String,
        count: EffectValue,
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        tapped_and_attacking: bool,
        /// "create a tapped [token]": enters tapped, not attacking.
        #[serde(skip_serializing_if = "std::ops::Not::not", default)]
        tapped: bool,
    },
    /// Attach the source Equipment to the target creature (CR 701.3). A
    /// resolving Aura spell instead enters attached to it (CR 303.4f).
    AttachSourceTo {
        what: ObjectSelector,
    },
    /// "Distribute `count` [kind] counters among any number of tapped
    /// creatures your opponents control": the controller places them one
    /// at a time, choosing among the eligible creatures at each placement.
    DistributeCountersToTappedOpponentCreatures {
        kind: CounterKind,
        count: usize,
    },
    /// "[player] [effects]": the effects resolve with that player as "you"
    /// ("Its controller may search their library"), then the controller
    /// returns to the resolving one. Several players ("each player draws a
    /// card") each run the effects in turn, in the selector's order.
    ForPlayer {
        player: PlayerSelector,
        effects: Vec<Effect>,
    },
    /// Internal cursor emitted by `ForPlayer`: the resolving player again.
    RestoreController(PlayerId),
    /// "Each player chooses any number of creatures they control with total
    /// power `max_total_power` or less, then sacrifices all other creatures
    /// they control": the active player chooses first (CR 101.4), and the
    /// sacrifices are one event.
    EachPlayerKeepsCreatures {
        max_total_power: i32,
    },
    /// "Each player sacrifices `count` creatures of their choice": the active
    /// player chooses first (CR 101.4), a player with no more than `count`
    /// sacrifices them all, and the sacrifices are one event.
    EachPlayerSacrificesCreatures {
        count: usize,
    },
    /// Create a delayed triggered ability (CR 603.7): when `event` occurs,
    /// `effects` trigger from this spell or ability's source. Their targets
    /// are chosen as the delayed ability is put on the stack.
    CreateDelayedTrigger {
        event: DelayedEvent,
        frequency: DelayedFrequency,
        expiry: DelayedExpiry,
        /// Resolve this selector now, not when the delayed ability resolves.
        /// Its exact objects become `ObjectSelector::Remembered` there.
        #[serde(skip_serializing_if = "Option::is_none")]
        remember: Option<ObjectSelector>,
        effects: Vec<Effect>,
    },
    /// Resolving player gains `amount` life, read as the instruction
    /// resolves.
    GainLife {
        amount: EffectValue,
    },
    /// Execute the inner effect only if this is the `n`th time this
    /// ability has resolved this turn ("if this is the second time this
    /// ability has resolved this turn, ...").
    OnNthResolutionThisTurn {
        n: u32,
        effect: Box<Effect>,
    },
    /// Scry `count` (CR 701.26), with arbitrary top and bottom ordering.
    Scry {
        count: usize,
    },
    ManifestDread,
    /// Surveil `count` (CR 701.25): look at the top cards and put any
    /// number of them into the graveyard, then order the kept cards on top.
    Surveil {
        count: usize,
    },
    /// "Look at the top `look` cards of your library. Put up to
    /// `max_select` cards matching [predicate] from among them into your
    /// hand and the rest on the bottom of your library."
    /// `min_select` > 0 makes the selection mandatory when possible.
    LookAndSelect {
        /// Otherwise the owner chooses the order (CR 401.4).
        random_order: bool,
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        reveal: bool,
        /// Selected cards go onto the battlefield instead of into the hand.
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        to_battlefield: bool,
        look: EffectValue,
        min_select: usize,
        max_select: usize,
        predicate: CardPredicate,
        /// Read as the instruction resolves: selectable cards have at most
        /// this mana value ("with mana value less than or equal to the
        /// number of lands you control").
        #[serde(skip_serializing_if = "Option::is_none")]
        max_mana_value: Option<Box<EffectValue>>,
    },
    /// Reveal the top card and bind it as a card revealed this way.
    RevealLibraryTop,
    /// Put the top `count` cards of the controller's library into their
    /// hand (not a draw — no draw triggers).
    PutTopCardsInHand {
        count: usize,
    },
    /// Copy the spell or ability the trigger names (its context): "copy it",
    /// "copy that ability" (CR 707.10). The copy may get new targets.
    CopyTriggeringObject,
    /// "Copy that spell for each [each] the spell could target. Each copy
    /// targets a different one of those" (Mirrorwing Dragon): the triggering
    /// spell is copied under the resolving player once per selected
    /// permanent that is a legal new target for it, and each copy's targets
    /// become that permanent; nobody chooses.
    CopyTriggeringSpellForEach {
        each: ObjectSelector,
    },
    /// "Change the target of target spell or ability with a single target"
    /// (CR 115.7): to another legal target, if there is one.
    ChangeTarget {
        target: TargetSpec,
    },
    /// Learn, without a sideboard (1v1 constructed): "You may discard a
    /// card. If you do, draw a card."
    Learn,
    /// "Choose one —" of a triggered ability (CR 700.2b): the controller
    /// chooses a mode with a legal choice of targets as the ability is put
    /// on the stack, then that mode's targets. It must be the ability's only
    /// effect.
    TriggerModes {
        modes: Vec<Vec<Effect>>,
        /// "Choose up to [value] —": that many distinct modes or fewer,
        /// read as the ability is put on the stack; their effects run in
        /// printed order (CR 700.2d).
        #[serde(skip_serializing_if = "Option::is_none")]
        up_to: Option<EffectValue>,
        /// "Choose one that hasn't been chosen": a mode the source object has
        /// chosen before is not offered (recorded as `ModeChosen`).
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        unchosen: bool,
    },
    /// Spell-level modes are selected during casting. Within a resolving
    /// ability this remains a resolution-time choice.
    Modal {
        modes: Vec<Vec<Effect>>,
    },
    /// "Counter [the spell or ability] unless its controller pays [cost]."
    /// It is the frame's primary target: either a chosen `TargetSpec::Spell`
    /// target (It'll Quench Ya!) or the trigger-context spell or ability
    /// (ward).
    /// Reports no target spec of its own — targeting is declared on the
    /// card (`PrintedCharacteristics::targeting`) when a choice is required.
    CounterUnlessPays {
        cost: PaymentCost,
    },
    /// Allies at Last: all chosen targets except the last each deal damage
    /// equal to their power to the last target. Targeting is declared on
    /// the card (`PrintedCharacteristics::targeting`).
    TargetCreaturesDealPowerDamageToLastTarget,
    /// Earthbend N: each land `what` names becomes a 0/0
    /// creature with haste that's still a land; put `count` +1/+1 counters
    /// on it. A delayed trigger returns it to the battlefield tapped when
    /// it dies or is exiled.
    Earthbend {
        count: EffectValue,
        what: ObjectSelector,
    },
    /// "You may cast a [predicate] spell from among cards you own exiled
    /// with this creature", paying its costs, as the effect resolves.
    CastExiledWithSource {
        predicate: CardPredicate,
    },
    /// Add `mana` to the resolving player's pool `amount` times, `amount`
    /// read at resolution (firebending X). `until_end_of_combat` routes it
    /// to the combat pool (firebending — persists across combat steps,
    /// empties at end of combat).
    AddMana {
        mana: Mana,
        #[serde(skip_serializing_if = "is_fixed_one")]
        amount: EffectValue,
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        until_end_of_combat: bool,
    },
    /// "Add X mana of any one color", X read as the mana ability resolves.
    AddOneColorManaValue {
        amount: EffectValue,
    },

    /// The first chosen creature fights the last (CR 701.14); records the
    /// excess damage dealt to the last.
    FightLastTarget,
    /// Shuffle the resolving player's library (CR 701.20).
    ShuffleLibrary,
    /// "Remove up to `up_to` counters from [from]." With `chosen` unset the
    /// controller picks a present kind or stops; with it set, one counter of
    /// that kind is removed and the choice repeats for `up_to - 1`.
    RemoveCounters {
        from: ObjectSelector,
        up_to: i32,
        chosen: Option<CounterKind>,
    },
    /// Run the inner effects once per chosen target, with that target as
    /// the primary target (Fancy Footwork: "untap one or two target
    /// creatures; they each get +2/+2"). Children may suspend.
    ForEachTarget {
        effects: Vec<Effect>,
    },
}

/// How long a player may play cards an effect exiled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum PlayWindow {
    /// "this turn".
    ThisTurn,
    /// "until the end of your next turn".
    ThroughYourNextTurn,
    /// "for as long as they remain exiled".
    WhileExiled,
}

/// The event a delayed triggered ability created by
/// [`Effect::CreateDelayedTrigger`] waits for (CR 603.7).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum DelayedEvent {
    /// Watch each captured object's death during that exact battlefield stint.
    Dies,
    CreatureDies,
    EndStep {
        your_turn: bool,
    },
    /// The matching spell supplies event context, separate from captured subjects.
    YouCastSpell(Box<CardPredicate>),
    Upkeep,
}

/// Whether a matching event consumes the watcher, independently of resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum DelayedFrequency {
    Once,
    EachMatch,
}

/// When to stop waiting. Expiry never removes an already-created occurrence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum DelayedExpiry {
    ThisTurn,
    /// No stated expiry; exact-object watchers still retire when that object leaves.
    Never,
}

impl Modification<EffectValue> {
    /// Fixed base power and/or toughness (layer 7b).
    pub fn base_pt(power: Option<i32>, toughness: Option<i32>) -> Self {
        Modification::SetBasePt {
            power: power.map(EffectValue::Fixed),
            toughness: toughness.map(EffectValue::Fixed),
        }
    }
}

/// The target spec of [`Scope::TargetPlayerControls`].
static TARGET_PLAYER: TargetSpec = TargetSpec::Player;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TargetSpec {
    PermanentYouControl {
        predicate: CardPredicate,
    },
    GraveyardCardYouOwn {
        predicate: CardPredicate,
    },
    /// "target [predicate] card from your graveyard with mana value less
    /// than or equal to this creature's power": the power of the ability's
    /// source as it is or last existed bounds the choice.
    GraveyardCardYouOwnWithinSourcePower {
        predicate: CardPredicate,
    },
    /// "target [predicate] card from a graveyard": any player's.
    GraveyardCard {
        predicate: CardPredicate,
    },
    /// "[predicate] card from that player's graveyard": an opponent's.
    GraveyardCardOpponentOwns {
        predicate: CardPredicate,
    },
    Player,
    Opponent,
    Permanent {
        predicate: CardPredicate,
    },
    Creature,
    /// Creature, player, planeswalker or battle (CR 115.4).
    AnyTarget,
    /// "target player or planeswalker".
    PlayerOrPlaneswalker,
    Spell,
    /// "target spell or ability with a single target".
    SpellOrAbilityWithSingleTarget,
    /// "target activated or triggered ability" on the stack.
    Ability,
    /// "target spell or permanent with mana value `min_mana_value` or
    /// greater" (Divide by Zero).
    SpellOrPermanent {
        min_mana_value: u8,
    },
    /// "target creature or spell".
    CreatureOrSpell,
    /// "target [predicate] spell" ("noncreature spell").
    SpellMatching {
        predicate: CardPredicate,
    },
    /// "target creature you control" — relative to the caster/controller.
    CreatureYouControl,
    /// "target creature an opponent controls".
    CreatureOpponentControls,
    /// "target land you control" (earthbend).
    LandYouControl,
    /// "target [predicate] permanent an opponent controls" (Earth Kingdom
    /// Jailer: artifact/creature/enchantment with mana value 3+).
    PermanentOpponentControls {
        predicate: CardPredicate,
    },
}

/// One targeting clause of a spell: "[up to] N target [spec]".
/// `min == 0` encodes "up to"; `max` bounds how many may be chosen.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct TargetRequirement {
    /// "[N] target creatures that share a creature type" (CR 115.1, 608.2b):
    /// every chosen target must share one creature type with the others.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub shares_creature_type: bool,
    pub spec: TargetSpec,
    pub min: usize,
    pub max: usize,
    /// "other target": the ability's own source object is not a candidate.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub another: bool,
    /// The announced X replaces `max` (CR 601.2c): "up to X targets" with
    /// `min` 0, or "X targets", exactly X, with any other `min`.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub max_x: bool,
    /// The targets in this clause have total mana value at most X.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub total_mana_value_x: bool,
    /// All graveyard cards in this clause come from a single graveyard.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub same_graveyard: bool,
    /// "Target creature of an opponent's choice": the controller's opponent
    /// chooses this clause's targets as the ability is activated; legality
    /// is still from the controller's side (CR 115.4).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub opponent_chooses: bool,
}

impl ObjectSelector {
    fn reads(&self, read: ContextRead) -> bool {
        match self {
            Self::Triggering => read == ContextRead::TriggeringObject,
            Self::Paid(_) => read == ContextRead::PaidCost,
            Self::Each {
                exact_mana_value,
                max_mana_value,
                ..
            } => [exact_mana_value, max_mana_value]
                .into_iter()
                .flatten()
                .any(|value| value.reads(read)),
            _ => {
                read == ContextRead::SourcePower
                    && self
                        .target_spec()
                        .is_some_and(TargetSpec::reads_source_power)
            }
        }
    }
}

impl TargetSpec {
    fn reads_source_power(&self) -> bool {
        matches!(self, Self::GraveyardCardYouOwnWithinSourcePower { .. })
    }
}

impl TargetRequirement {
    pub fn reads_source_power(&self) -> bool {
        self.spec.reads_source_power()
    }

    /// `max` for "any number of targets" (CR 601.2c): the legal candidates
    /// bound the choice; the value stays within the learning tokens' i32.
    pub const ANY_NUMBER: usize = i32::MAX as usize;

    pub fn one(spec: TargetSpec) -> Self {
        Self {
            shares_creature_type: false,
            spec,
            min: 1,
            max: 1,
            another: false,
            max_x: false,
            total_mana_value_x: false,
            same_graveyard: false,
            opponent_chooses: false,
        }
    }

    /// "Another target" excludes the source or its immediate graveyard successor.
    pub fn another(spec: TargetSpec) -> Self {
        Self {
            another: true,
            ..Self::one(spec)
        }
    }

    /// Whether X must equal the number of targets ("X targets").
    pub fn exact_x(&self) -> bool {
        self.max_x && self.min > 0
    }

    /// The fewest targets this requirement allows with X announced as `x`.
    pub fn min_for(&self, x: u8) -> usize {
        if self.exact_x() {
            usize::from(x)
        } else {
            self.min
        }
    }

    /// The most targets this requirement allows with X announced as `x`.
    pub fn max_for(&self, x: u8) -> usize {
        if self.max_x {
            usize::from(x)
        } else {
            self.max
        }
    }

    pub fn up_to(max: usize, spec: TargetSpec) -> Self {
        Self {
            min: 0,
            max,
            ..Self::one(spec)
        }
    }
}

fn is_one(value: &usize) -> bool {
    *value == 1
}
fn is_fixed_one(value: &EffectValue) -> bool {
    *value == EffectValue::Fixed(1)
}

impl Default for Effect {
    fn default() -> Self {
        Self::DrawCards {
            count: EffectValue::Fixed(0),
        }
    }
}

impl Effect {
    pub fn reads_source_power(&self) -> bool {
        self.reads(ContextRead::SourcePower)
    }
    /// Reads a source characteristic other than power, which a stacked
    /// ability keeps as last known information once its source has left.
    pub fn reads_source_lki(&self) -> bool {
        self.reads(ContextRead::SourceLki)
    }
    pub fn reads_triggering_object(&self) -> bool {
        self.reads(ContextRead::TriggeringObject)
    }

    fn reads(&self, read: ContextRead) -> bool {
        let target_reads_power = match self {
            Self::TargetClause { requirement, .. } => requirement.reads_source_power(),
            _ => self
                .target_spec()
                .is_some_and(TargetSpec::reads_source_power),
        };
        if (read == ContextRead::SourcePower && target_reads_power)
            || self
                .find_selector(|s| s.reads(read).then_some(()))
                .is_some()
        {
            return true;
        }
        let modification_reads = |modification: &Modification<EffectValue>| match modification {
            Modification::SetBasePt { power, toughness } => [power, toughness]
                .into_iter()
                .flatten()
                .any(|v| v.reads(read)),
            Modification::ModifyPt { power, toughness } => {
                power.reads(read) || toughness.reads(read)
            }
            _ => false,
        };
        if let Some(amount) = self.amount() {
            return amount.reads(read);
        }
        match self {
            Self::MayExileTriggeringWithSource
            | Self::CopyTriggeringObject
            | Self::CopyTriggeringSpellForEach { .. } => read == ContextRead::TriggeringObject,
            Self::ApplyContinuous { modification, .. } => modification_reads(modification),
            Self::CreateTokenCopy {
                count, exceptions, ..
            } => {
                count.reads(read)
                    || exceptions
                        .iter()
                        .any(|exception| modification_reads(&exception.modification()))
            }
            Self::If { condition, .. } if condition.reads(read) => true,
            Self::SelectCards { max, .. } if max.reads(read) => true,
            Self::TriggerModes {
                up_to: Some(limit), ..
            } if limit.reads(read) => true,
            _ => self.any_child_effect(|effect| effect.reads(read)),
        }
    }

    /// The quantity this instruction acts by ("deals N damage", "gain N
    /// life"), when it has one.
    fn amount(&self) -> Option<&EffectValue> {
        match self {
            Self::DealDamage { amount, .. }
            | Self::DealDamageToPlayer { amount, .. }
            | Self::LoseLife { amount, .. }
            | Self::GainLife { amount }
            | Self::PutCounters { amount, .. }
            | Self::DrawCards { count: amount }
            | Self::MillCards { count: amount, .. }
            | Self::Earthbend { count: amount, .. }
            | Self::DrainLife { amount }
            | Self::AddMana { amount, .. }
            | Self::CreateToken { count: amount, .. }
            | Self::AddOneColorManaValue { amount }
            | Self::ExileUntilNonlandCastFree { below: amount }
            | Self::LookAndSelect { look: amount, .. } => Some(amount),
            _ => None,
        }
    }

    /// The first amount in this program read from the objects that paid its
    /// cost. A pending ability publishes the current value, using last known
    /// information for a payer that has left the battlefield (CR 608.2h).
    pub fn paid_cost_amount(&self) -> Option<EffectValue> {
        if let Some(amount) = self.amount().filter(|a| a.reads(ContextRead::PaidCost)) {
            return Some(amount.clone());
        }
        let mut found = None;
        self.any_child_effect(|effect| {
            found = effect.paid_cost_amount();
            found.is_some()
        });
        found
    }

    /// Whether this program reads its enclosing ability's resolution ordinal.
    pub fn reads_resolution_count(&self) -> bool {
        matches!(self, Self::OnNthResolutionThisTurn { .. })
            || self.any_child_effect(Self::reads_resolution_count)
    }

    /// Whether this program can make a permanent a creature.
    pub fn may_become_creature(&self) -> bool {
        match self {
            Self::ApplyContinuous { modification, .. } => modification.grants_creature_type(),
            _ => self.any_child_effect(Self::may_become_creature),
        }
    }

    /// Instructions within this resolution. Delayed and reflexive triggers
    /// inspect their programs when those abilities resolve.
    fn any_child_effect(&self, mut test: impl FnMut(&Effect) -> bool) -> bool {
        match self {
            Self::OnNthResolutionThisTurn { effect, .. } => test(effect),
            Self::If {
                then, otherwise, ..
            }
            | Self::OptionalPayment {
                if_paid: then,
                if_declined: otherwise,
                ..
            }
            | Self::SelectCards {
                effects: then,
                if_declined: otherwise,
                ..
            } => then.iter().chain(otherwise).any(test),
            Self::OptionalManaPaymentX { if_paid: effects }
            | Self::ForEachTarget { effects }
            | Self::TargetClause { effects, .. }
            | Self::ForPlayer { effects, .. }
            | Self::MayOnceEachTurn { effects } => effects.iter().any(test),
            Self::Modal { modes } | Self::TriggerModes { modes, .. } => {
                modes.iter().flatten().any(test)
            }
            _ => false,
        }
    }

    /// Walk the selectors a verb acts through, in field order, without allocating.
    fn find_selector<'a, T>(
        &'a self,
        mut f: impl FnMut(&'a ObjectSelector) -> Option<T>,
    ) -> Option<T> {
        match self {
            Effect::CreateDelayedTrigger { remember, .. } => remember.iter().find_map(f),
            Effect::PutCountersFrom { from, on } => f(from).or_else(|| f(on)),
            Effect::DealDamage { what, by, .. } => f(what).or_else(|| by.as_ref().and_then(f)),
            Effect::SelectCards { from: selector, .. }
            | Effect::PutCounters { on: selector, .. }
            | Effect::CreateTokenCopy { of: selector, .. }
            | Effect::CopyTriggeringSpellForEach { each: selector }
            | Effect::ApplyContinuous {
                affected: selector, ..
            }
            | Effect::Sacrifice { what: selector, .. }
            | Effect::Move { what: selector, .. }
            | Effect::Prepare { what: selector }
            | Effect::Unprepare { what: selector }
            | Effect::Tap { what: selector }
            | Effect::Untap { what: selector }
            | Effect::Destroy { what: selector }
            | Effect::Airbend { what: selector }
            | Effect::GainControl { what: selector, .. }
            | Effect::ControlPlayer { what: selector, .. }
            | Effect::CounterSpell { what: selector, .. }
            | Effect::CounterAbility { what: selector }
            | Effect::AttachSourceTo { what: selector }
            | Effect::Earthbend { what: selector, .. }
            | Effect::Discard { what: selector, .. }
            | Effect::Reveal { what: selector }
            | Effect::RemoveCounters { from: selector, .. }
            | Effect::RemoveAllCounters { from: selector } => f(selector),
            _ => None,
        }
    }

    /// The choice a `Chosen` selector asks of the resolving player, if the
    /// verb has one; only `Sacrifice` may let them decline.
    pub fn chosen(&self) -> Option<(&CardPredicate, Option<Except>, bool)> {
        let (predicate, except) = self.find_selector(|selector| match selector {
            ObjectSelector::Chosen { predicate, except } => Some((predicate, *except)),
            _ => None,
        })?;
        let optional = matches!(self, Effect::Sacrifice { optional: true, .. });
        Some((predicate, except, optional))
    }

    /// One immediate mana output per mode, preserving the authored color order.
    pub fn mana_choice(outputs: impl IntoIterator<Item = Mana>) -> Self {
        Self::Modal {
            modes: outputs
                .into_iter()
                .map(|mana| vec![Self::add_mana(mana)])
                .collect(),
        }
    }

    pub fn any_color_mana() -> Self {
        Self::mana_choice(["W", "U", "B", "R", "G"].map(Mana::parse))
    }

    /// "another target [spec]": one target other than the source, which
    /// `effects` read through the spec they are given.
    pub fn another_target(
        spec: TargetSpec,
        effects: impl FnOnce(TargetSpec) -> Vec<Effect>,
    ) -> Self {
        Effect::TargetClause {
            requirement: TargetRequirement {
                another: true,
                ..TargetRequirement::one(spec.clone())
            },
            effects: effects(spec),
        }
    }

    /// "[affected] [modification] until end of turn."
    pub fn until_end_of_turn(
        affected: ObjectSelector,
        modification: Modification<EffectValue>,
    ) -> Self {
        Effect::ApplyContinuous {
            affected,
            modification,
            duration: EffectDuration::EndOfTurn,
        }
    }

    /// "[Affected] gets +P/+T and gains [keyword] until end of turn."
    pub fn pump_and_grant_until_end_of_turn(
        affected: ObjectSelector,
        power: i32,
        toughness: i32,
        keyword: Keyword,
    ) -> Vec<Self> {
        vec![
            Self::until_end_of_turn(affected.clone(), Modification::pump(power, toughness)),
            Self::until_end_of_turn(
                affected,
                Modification::AddKeywords(Keywords::of(&[keyword])),
            ),
        ]
    }

    /// A continuous effect with no stated duration.
    pub fn indefinitely(affected: ObjectSelector, modification: Modification<EffectValue>) -> Self {
        Effect::ApplyContinuous {
            affected,
            modification,
            duration: EffectDuration::Indefinite,
        }
    }

    pub fn target_spec(&self) -> Option<&TargetSpec> {
        if let Some(spec) = self.find_selector(ObjectSelector::target_spec) {
            return Some(spec);
        }
        match self {
            Effect::DiscardHand { who: player }
            | Effect::GivePlayerCounters { player, .. }
            | Effect::LoseLife { player, .. }
            | Effect::DealDamageToPlayer { player, .. } => player.target_spec(),
            Effect::TargetClause { requirement, .. } => Some(&requirement.spec),
            Effect::MillCards {
                target,
                player: None,
                ..
            } => target.as_ref(),
            Effect::MillCards {
                player: Some(player),
                ..
            }
            | Effect::MillUntil { player, .. } => player.target_spec(),
            Effect::GrantGraveyardCastThisTurn { target, .. }
            | Effect::ExileLinked { target, .. } => Some(target),
            Effect::ChangeTarget { target }
            | Effect::OwnerPutsIntoLibrary { target, .. }
            | Effect::RevealHandDiscardChosen { target, .. } => Some(target),
            Effect::OnNthResolutionThisTurn { effect, .. } => effect.target_spec(),
            Effect::OptionalPayment {
                if_paid: then,
                if_declined: otherwise,
                ..
            }
            | Effect::If {
                then, otherwise, ..
            }
            | Effect::SelectCards {
                effects: then,
                if_declined: otherwise,
                ..
            } => then
                .iter()
                .chain(otherwise.iter())
                .find_map(Effect::target_spec),
            Effect::ForPlayer { player, effects } => player
                .target_spec()
                .or_else(|| effects.iter().find_map(Effect::target_spec)),
            Effect::MayOnceEachTurn { effects: then } => {
                then.iter().find_map(|effect| effect.target_spec())
            }
            Effect::CreateTokenCopy { .. }
            | Effect::PutCounters { .. }
            | Effect::PutCountersFrom { .. }
            | Effect::Sacrifice { .. }
            | Effect::Move { .. }
            | Effect::Tap { .. }
            | Effect::Prepare { .. }
            | Effect::Unprepare { .. }
            | Effect::Untap { .. }
            | Effect::RemoveCounters { .. }
            | Effect::RemoveAllCounters { .. }
            | Effect::Destroy { .. }
            | Effect::Airbend { .. }
            | Effect::GainControl { .. }
            | Effect::DealDamage { .. }
            | Effect::CounterSpell { .. }
            | Effect::CounterAbility { .. }
            | Effect::Earthbend { .. }
            | Effect::Discard { .. }
            | Effect::Reveal { .. }
            | Effect::RevealUntilToHand { .. }
            | Effect::DrainLife { .. }
            | Effect::SelectTargetRequirement { .. }
            | Effect::ApplyContinuous { .. }
            | Effect::ShuffleLibrary
            | Effect::OpponentsCastOnlyFromHand
            | Effect::ExileTopCardsToPlay { .. }
            | Effect::ExileUntilNonlandCastFree { .. }
            | Effect::ExileEachLibraryTopCastFree
            | Effect::Cascade
            | Effect::CumulativeUpkeep { .. }
            | Effect::CastFromHandFree { .. }
            | Effect::CastFromGraveyard { .. }
            | Effect::CastLibraryTopFree
            | Effect::WhenYouDo { .. }
            | Effect::TriggerModes { .. }
            | Effect::ShuffleGraveyardIntoLibrary
            | Effect::PlaceTargetInLibrary { .. }
            | Effect::SearchDistinctNames { .. }
            | Effect::DrawCards { .. }
            | Effect::CreateToken { .. }
            | Effect::ExileMeldPair { .. }
            | Effect::MeldExiled { .. }
            | Effect::TransformSource
            | Effect::TakeExtraTurn
            | Effect::WinGame
            | Effect::LoseGame
            | Effect::EndTheTurn
            | Effect::ControlPlayer { .. }
            | Effect::AdditionalCombatPhase { .. }
            | Effect::EachPlayerChoosesToExile { .. }
            | Effect::MayExileTriggeringWithSource
            | Effect::ChooseExiledWithSource { .. }
            | Effect::HaveAbilitiesOf { .. }
            | Effect::NameAndExileFromTargetPlayer { .. }
            | Effect::ExileNamed { .. }
            | Effect::OptionalManaPaymentX { .. }
            | Effect::CreateDelayedTrigger { .. }
            | Effect::CopyTriggeringObject
            | Effect::CopyTriggeringSpellForEach { .. }
            | Effect::CastExiledWithSource { .. }
            | Effect::RestoreController(_)
            | Effect::EachPlayerKeepsCreatures { .. }
            | Effect::EachPlayerSacrificesCreatures { .. }
            | Effect::DistributeCountersToTappedOpponentCreatures { .. }
            | Effect::GainLife { .. }
            | Effect::FightLastTarget
            | Effect::Scry { .. }
            | Effect::ManifestDread
            | Effect::Surveil { .. }
            | Effect::LookAndSelect { .. }
            | Effect::RevealLibraryTop
            | Effect::PutTopCardsInHand { .. }
            | Effect::Learn
            | Effect::Modal { .. }
            | Effect::CounterUnlessPays { .. }
            | Effect::AddMana { .. }
            | Effect::AddOneColorManaValue { .. }
            | Effect::ForEachTarget { .. }
            | Effect::TargetCreaturesDealPowerDamageToLastTarget
            | Effect::AttachSourceTo { .. } => None,
        }
    }

    // -----------------------------------------------------------------------
    // Constructors that keep card definitions readable.

    /// "Put `amount` [kind] counters on [on]."
    pub fn put_counters(
        on: ObjectSelector,
        kind: CounterKind,
        amount: impl Into<EffectValue>,
    ) -> Self {
        Effect::PutCounters {
            on,
            kind,
            amount: amount.into(),
        }
    }

    /// "Sacrifice this."
    pub fn sacrifice_this() -> Self {
        Effect::Sacrifice {
            what: ObjectSelector::This,
            optional: false,
        }
    }

    /// "Sacrifice a [predicate]."
    pub fn sacrifice(predicate: CardPredicate) -> Self {
        Effect::Sacrifice {
            what: ObjectSelector::chosen(predicate),
            optional: false,
        }
    }

    /// "You may sacrifice [another] [predicate]."
    pub fn may_sacrifice(predicate: CardPredicate, another: bool) -> Self {
        Effect::Sacrifice {
            what: ObjectSelector::Chosen {
                predicate,
                except: another.then_some(Except::This),
            },
            optional: true,
        }
    }

    /// "Destroy [what]."
    pub fn destroy(what: ObjectSelector) -> Self {
        Effect::Destroy { what }
    }

    /// "Deal [amount] damage to [what]."
    pub fn deal_damage(amount: impl Into<EffectValue>, what: ObjectSelector) -> Self {
        Effect::DealDamage {
            amount: amount.into(),
            what,
            by: None,
        }
    }

    /// "Exile the top `count` cards of your library. You may play them
    /// [`until`]."
    pub fn exile_top_to_play(count: usize, until: PlayWindow) -> Self {
        Effect::ExileTopCardsToPlay {
            count,
            until,
            face_down: false,
            condition: None,
        }
    }

    /// "You gain [amount] life."
    pub fn gain_life(amount: impl Into<EffectValue>) -> Self {
        Effect::GainLife {
            amount: amount.into(),
        }
    }

    /// Search a library subset, run its selected branch, then shuffle even
    /// on failure to find. The runtime announces completion after that scope.
    pub fn search_library(
        predicate: CardPredicate,
        max: EffectValue,
        effects: Vec<Effect>,
    ) -> Self {
        Self::library_search(predicate, 0, max, effects)
    }

    /// A search for an unrestricted card must find one in a nonempty library.
    pub fn search_for_card(destination: ZoneType) -> Self {
        Self::library_search(
            CardPredicate::default(),
            1,
            1.into(),
            vec![Self::zone_move(
                ObjectSelector::Selected,
                destination,
                destination == ZoneType::Hand,
            )],
        )
    }

    fn library_search(
        predicate: CardPredicate,
        min: usize,
        max: EffectValue,
        mut effects: Vec<Effect>,
    ) -> Self {
        effects.push(Effect::ShuffleLibrary);
        Effect::SelectCards {
            from: ObjectSelector::Zone(ZoneType::Library),
            predicate,
            min,
            max,
            effects,
            if_declined: vec![Effect::ShuffleLibrary],
        }
    }

    /// Select private hand cards as one subset, then discard them together.
    /// The selected child retains its bindings through any subsequent question.
    pub fn may_discard(up_to: usize, then: Vec<Effect>) -> Self {
        let mut effects = vec![Self::move_to(ObjectSelector::Selected, ZoneType::Graveyard)];
        effects.push(Self::if_you_do(then));
        Effect::SelectCards {
            from: ObjectSelector::Zone(ZoneType::Hand),
            predicate: CardPredicate::default(),
            min: 0,
            max: EffectValue::Fixed(up_to as i32),
            effects,
            if_declined: Vec::new(),
        }
    }

    /// "Discard up to N cards, then draw for each card discarded this way."
    pub fn discard_draw(up_to: usize, per_card: usize) -> Self {
        Self::may_discard(
            up_to,
            vec![
                Effect::DrawCards {
                    count: ObjectSelector::Moved.count(),
                };
                per_card
            ],
        )
    }

    /// "Discard a card."
    pub fn discard() -> Self {
        Effect::Discard {
            what: ObjectSelector::You,
            count: 1,
        }
    }

    /// "Create [count] [token] tokens", naming the token by its key.
    pub fn create_tokens(token: &str, count: impl Into<EffectValue>) -> Self {
        Effect::CreateToken {
            token: token.into(),
            count: count.into(),
            tapped_and_attacking: false,
            tapped: false,
        }
    }

    /// Create token copies of each selected object with its copiable values.
    pub fn create_token_copies(of: ObjectSelector, count: impl Into<EffectValue>) -> Self {
        Effect::CreateTokenCopy {
            of,
            count: count.into(),
            exceptions: Vec::new(),
        }
    }

    /// "Add [mana]."
    pub fn add_mana(mana: Mana) -> Self {
        Effect::AddMana {
            mana,
            amount: EffectValue::Fixed(1),
            until_end_of_combat: false,
        }
    }

    /// "Put [what] into [zone]" with no further terms.
    pub fn move_to(what: ObjectSelector, to: ZoneType) -> Self {
        Self::zone_move(what, to, false)
    }

    fn zone_move(what: ObjectSelector, to: ZoneType, unrevealed: bool) -> Self {
        Effect::Move {
            unrevealed,
            library_end: None,
            what,
            to,
            tapped: false,
            transformed: false,
            under: None,
            with_counter: None,
            linked: None,
        }
    }

    /// Put an object on the stated end without shuffling. Placement commits
    /// with the move, before linked returns can ask a subsequent question.
    pub fn put_in_library(what: ObjectSelector, end: LibraryEnd) -> Self {
        Effect::Move {
            unrevealed: false,
            library_end: Some(end),
            what,
            to: ZoneType::Library,
            tapped: false,
            transformed: false,
            under: None,
            with_counter: None,
            linked: None,
        }
    }

    /// "Exile [what]."
    pub fn exile(what: ObjectSelector) -> Self {
        Self::move_to(what, ZoneType::Exile)
    }

    /// "Return [what] to its owner's hand."
    pub fn return_to_hand(what: ObjectSelector) -> Self {
        Self::move_to(what, ZoneType::Hand)
    }

    /// The second move follows the appearance produced by exile.
    pub fn exile_and_return_transformed(
        under: Option<PlayerSelector>,
        with_counter: Option<CounterKind>,
    ) -> Vec<Effect> {
        vec![
            Effect::exile(ObjectSelector::This),
            Effect::Move {
                unrevealed: false,
                library_end: None,
                what: ObjectSelector::Moved,
                to: ZoneType::Battlefield,
                tapped: false,
                transformed: true,
                under,
                with_counter: with_counter.map(Into::into),
                linked: None,
            },
        ]
    }

    /// Return the named cards under their owners' control, with the stated entry terms.
    pub fn return_to_owner(
        what: ObjectSelector,
        tapped: bool,
        with_counter: Option<EntryCounters>,
    ) -> Self {
        Self::Move {
            unrevealed: false,
            what,
            to: ZoneType::Battlefield,
            tapped,
            with_counter,
            under: Some(PlayerSelector::Owner),
            transformed: false,
            linked: None,
            library_end: None,
        }
    }

    /// Capture exact objects for one occurrence at the next end step.
    pub fn at_next_end_step(remember: ObjectSelector, effects: Vec<Effect>) -> Self {
        Self::CreateDelayedTrigger {
            event: DelayedEvent::EndStep { your_turn: false },
            remember: Some(remember),
            effects,
            frequency: DelayedFrequency::Once,
            expiry: DelayedExpiry::Never,
        }
    }

    /// "Return [what] to the battlefield [tapped]" under the resolving
    /// player's control.
    pub fn return_to_battlefield(what: ObjectSelector, tapped: bool) -> Self {
        Effect::Move {
            unrevealed: false,
            library_end: None,
            what,
            to: ZoneType::Battlefield,
            tapped,
            transformed: false,
            under: None,
            with_counter: None,
            linked: None,
        }
    }

    /// Exile a stated number of opposing permanents until the source leaves.
    pub fn exile_until_source_leaves(spec: TargetSpec, min: usize) -> Self {
        Self::TargetClause {
            requirement: TargetRequirement {
                min,
                ..TargetRequirement::one(spec.clone())
            },
            effects: vec![Self::exile_linked(
                ObjectSelector::Target(spec),
                LinkedExile::UntilSourceLeaves,
            )],
        }
    }

    /// "Exile [what] with this" or "until this leaves the battlefield".
    pub fn exile_linked(what: ObjectSelector, linked: LinkedExile) -> Self {
        Effect::Move {
            unrevealed: false,
            library_end: None,
            what,
            to: ZoneType::Exile,
            tapped: false,
            transformed: false,
            under: None,
            with_counter: None,
            linked: Some(linked),
        }
    }

    pub fn tap(what: ObjectSelector) -> Self {
        Effect::Tap { what }
    }

    /// An Aura spell's "Enchant [spec]" (CR 702.5): it resolves attached to
    /// one target matching `spec`. `Object::enchant` reads the spec back, so
    /// the cast target and the CR 704.5m recheck come from this one clause.
    pub fn enchant(spec: TargetSpec) -> Self {
        Effect::AttachSourceTo {
            what: ObjectSelector::Target(spec),
        }
    }

    pub fn untap(what: ObjectSelector) -> Self {
        Effect::Untap { what }
    }

    /// "If you do, [then]": only after the latest move or sacrifice
    /// happened.
    pub fn if_you_do(then: Vec<Effect>) -> Self {
        Effect::If {
            condition: Condition::Moved,
            then,
            otherwise: Vec::new(),
        }
    }
}

impl EffectValue {
    fn reads(&self, read: ContextRead) -> bool {
        match self {
            Self::Property {
                of,
                property: ObjectProperty::Power,
                ..
            } if **of == ObjectSelector::This => read == ContextRead::SourcePower,
            Self::Property { of, .. } if **of == ObjectSelector::This => {
                read == ContextRead::SourceLki
            }
            Self::SourcePower => read == ContextRead::SourcePower,
            Self::PowerTappedToPay => read == ContextRead::PaidCost,
            Self::TriggeringPower | Self::TriggeringManaValue => {
                read == ContextRead::TriggeringObject
            }
            Self::Half(inner) | Self::Scaled { value: inner, .. } => inner.reads(read),
            Self::Property { of, .. }
            | Self::Objects { of, .. }
            | Self::DistinctManaValues { of }
            | Self::SpellManaValue(of)
            | Self::SpellManaSpent(of) => of.reads(read),
            _ => false,
        }
    }

    pub fn reads_source(&self) -> bool {
        match self {
            Self::SourcePower => true,
            Self::Half(inner) | Self::Scaled { value: inner, .. } => inner.reads_source(),
            Self::Property { of, .. }
            | Self::Objects { of, .. }
            | Self::DistinctManaValues { of } => match of.as_ref() {
                ObjectSelector::This | ObjectSelector::Triggering => true,
                ObjectSelector::Each {
                    except,
                    predicate,
                    exact_mana_value,
                    max_mana_value,
                    ..
                } => {
                    *except == Some(Except::This)
                        || predicate.reads_source()
                        || [exact_mana_value, max_mana_value]
                            .into_iter()
                            .flatten()
                            .any(|v| v.reads_source())
                }
                _ => false,
            },
            _ => false,
        }
    }
    pub fn is_resolution_only(&self) -> bool {
        match self {
            Self::Half(inner) | Self::Scaled { value: inner, .. } => inner.is_resolution_only(),
            Self::LifeTotal(who)
            | Self::LifeLostThisTurn(who)
            | Self::LifeGainedThisTurn(who)
            | Self::LifeAboveStarting(who) => {
                !matches!(who, PlayerSelector::You | PlayerSelector::EachOpponent)
            }
            Self::Property { of, .. }
            | Self::Objects { of, .. }
            | Self::DistinctManaValues { of } => match of.as_ref() {
                ObjectSelector::This | ObjectSelector::Triggering => false,
                ObjectSelector::Each {
                    of: Scope::YouControl | Scope::OpponentsControl | Scope::Battlefield,
                    except,
                    exact_mana_value,
                    max_mana_value,
                    ..
                } => {
                    *except == Some(Except::Target)
                        || [exact_mana_value, max_mana_value]
                            .into_iter()
                            .flatten()
                            .any(|v| v.is_resolution_only())
                }
                _ => true,
            },
            Self::SpellManaSpent(_)
            | Self::SpellManaValue(_)
            | Self::X
            | Self::NegativeX
            | Self::TriggerDamage
            | Self::TriggeringAmount
            | Self::ExcessDamage
            | Self::LifeLostThisWay
            | Self::ManaSpentToCast
            | Self::ColorsSpentToCast
            | Self::PowerTappedToPay
            | Self::TriggeringPower
            | Self::TriggeringManaValue
            | Self::TargetPower
            | Self::TargetToughness => true,
            _ => false,
        }
    }
}

impl From<i32> for EffectValue {
    fn from(value: i32) -> Self {
        EffectValue::Fixed(value)
    }
}

/// The effects a printed triggered ability runs once its modes are chosen
/// (CR 700.2b): a modal rule (`[Effect::TriggerModes]`) runs the chosen
/// modes in printed order, possibly none under "up to"; any other program
/// runs as printed.
pub fn chosen_trigger_effects<'a>(
    printed: &'a [Effect],
    chosen_modes: &[usize],
) -> std::borrow::Cow<'a, [Effect]> {
    match printed {
        [Effect::TriggerModes { modes, .. }] => modes
            .iter()
            .enumerate()
            .filter(|(index, _)| chosen_modes.contains(index))
            .flat_map(|(_, effects)| effects.iter().cloned())
            .collect::<Vec<_>>()
            .into(),
        _ => printed.into(),
    }
}

/// Explicit clauses preserve independent targets, including repeated specs.
/// Unwrapped effects retain the shared "that target" convention.
pub fn effect_target_requirements(effects: &[Effect]) -> Vec<TargetRequirement> {
    let clauses: Vec<_> = effects
        .iter()
        .filter_map(|effect| match effect {
            Effect::TargetClause { requirement, .. } => Some(requirement.clone()),
            _ => None,
        })
        .collect();
    if clauses.is_empty() {
        effects
            .iter()
            .find_map(Effect::target_spec)
            .map(|spec| vec![TargetRequirement::one(spec.clone())])
            .unwrap_or_default()
    } else {
        clauses
    }
}

pub fn bind_effect_clauses(effects: Vec<Effect>) -> Vec<Effect> {
    let mut bound = Vec::new();
    let mut index = 0;
    for effect in effects {
        match effect {
            Effect::TargetClause { effects, .. } => {
                bound.push(Effect::SelectTargetRequirement { index });
                index += 1;
                bound.extend(effects);
            }
            effect => bound.push(effect),
        }
    }
    bound
}

/// Origin of a layer-six ability on its current recipient.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbilityOrigin {
    Own,
    Effect(PublicId),
    Counter,
    Permanent(PermanentId),
}

/// Effective triggered ability with the provenance needed after its source leaves.
/// Catalog grants have retained input joins; a continuous-effect program does not.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct EffectiveTrigger {
    /// Battlefield inspection uses the live origin before this trigger leaves play.
    #[serde(skip)]
    pub origin: AbilityOrigin,
    /// Grant timestamp, static ability slot and inherited ability slot for a once-per-turn clock.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clock: Option<GrantClock>,
    pub ability: Ability,
    pub program: bool,
}

/// Identifies one limited ability supplied by a continuous effect. Several
/// permanents can receive the same static grant at one timestamp.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize)]
pub struct GrantClock {
    pub grantor: crate::game_object::PermanentId,
    pub timestamp: crate::game_object::PublicId,
    pub static_slot: Option<usize>,
    pub ability_index: usize,
}

impl TriggerCondition {
    /// The condition whose event this wrapper restricts.
    pub fn inner(&self) -> Option<&Self> {
        match self {
            Self::WasCast { condition }
            | Self::OnceEachTurn { condition }
            | Self::WhileSourceAttacking { condition }
            | Self::FromZone { condition, .. }
            | Self::Prepared { condition }
            | Self::InGraveyard { condition }
            | Self::ActiveIf { condition, .. } => Some(condition),
            Self::Sacrificed { .. }
            | Self::NoCounters { .. }
            | Self::State(_)
            | Self::YouCastThis
            | Self::Expend(_)
            | Self::SpellTargetsOnlyThis { .. }
            | Self::EntersTheBattlefield { .. }
            | Self::Dies { .. }
            | Self::LeavesTheBattlefield { .. }
            | Self::Attacks { .. }
            | Self::BecomesTapped { .. }
            | Self::BecomesUntapped { .. }
            | Self::TappedForMana { .. }
            | Self::BecomesTargeted { .. }
            | Self::YourAttackingCreatureTriggers
            | Self::Chapter { .. }
            | Self::NthCounter { .. }
            | Self::PutsCounters { .. }
            | Self::PutsCountersOn { .. }
            | Self::Casts { .. }
            | Self::Draws { .. }
            | Self::GainsLife { .. }
            | Self::LosesLife { .. }
            | Self::BeginningOf { .. }
            | Self::Performs { .. }
            | Self::PutIntoGraveyard { .. }
            | Self::DealsCombatDamage { .. }
            | Self::CardsLeaveYourGraveyard
            | Self::SourceDealsDamage
            | Self::SourceDealsCombatDamage
            | Self::DealsDamage { .. } => None,
        }
    }

    /// The underlying event condition after peeling every wrapper.
    pub fn leaf(&self) -> &Self {
        let mut condition = self;
        while let Some(inner) = condition.inner() {
            condition = inner;
        }
        condition
    }

    pub fn once_each_turn(&self) -> bool {
        matches!(self, Self::OnceEachTurn { .. }) || self.inner().is_some_and(Self::once_each_turn)
    }
}

#[cfg(test)]
mod tests {
    use super::TriggerCondition;

    #[test]
    fn prepared_preserves_a_nested_once_each_turn_limit() {
        let prepared = |condition| TriggerCondition::Prepared {
            condition: Box::new(condition),
        };
        assert!(!prepared(TriggerCondition::YouCastThis).once_each_turn());
        assert!(prepared(TriggerCondition::OnceEachTurn {
            condition: Box::new(TriggerCondition::YouCastThis),
        })
        .once_each_turn());
    }
}
