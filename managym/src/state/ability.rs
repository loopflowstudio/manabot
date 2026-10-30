use super::effect::{EffectDuration, Modification};
use super::game_object::PlayerId;
use super::mana::{Mana, ManaCost};
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
    pub fn effects(&self) -> &[Effect] {
        match self {
            Ability::Triggered { effects, .. } => effects,
        }
    }
}

/// The event that causes a triggered ability to trigger (CR 603.1).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum TriggerCondition {
    /// The entering source must have been cast, rather than created or returned.
    WasCast {
        condition: Box<TriggerCondition>,
    },
    Sacrificed {
        predicate: CardPredicate,
        controller_only: bool,
        /// "another permanent": the exact source object's own sacrifice
        /// does not trigger it.
        #[serde(skip_serializing_if = "std::ops::Not::not", default)]
        another: bool,
    },
    /// Wrap the whole condition, including ActiveIf. The limit is recorded
    /// when the event fires, once per exact source incarnation and turn.
    OnceEachTurn {
        condition: Box<TriggerCondition>,
    },
    YouCastSpell {
        predicate: CardPredicate,
    },
    /// "Whenever a player casts a [predicate] spell": any player, including
    /// you. The caster is the trigger context ("they").
    PlayerCastsSpell {
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
    /// "Whenever [subject] attacks." Fires after the declare-attackers
    /// turn-based action (CR 508.1); a subject of `AnotherYouControl` /
    /// `AnyYouControl` fires once per combat ("whenever one or more ...
    /// attack"), not once per attacker.
    Attacks {
        subject: TriggerSubject,
    },
    /// "Whenever [this] and at least one other creature attack": the source
    /// is among two or more declared attackers.
    AttacksWithAnother,
    /// "Whenever [subject] attacks alone": exactly one creature was
    /// declared as an attacker (CR 506.5). The attacker is the context.
    AttacksAlone {
        subject: TriggerSubject,
    },
    /// The inner cast or entry trigger only for an object that came from
    /// `zone` ("whenever you cast a spell from exile", "whenever a permanent
    /// you control enters from exile").
    FromZone {
        zone: ZoneType,
        condition: Box<TriggerCondition>,
    },
    /// "Whenever an opponent searches their library" (CR 701.23), even when
    /// the search finds nothing.
    OpponentSearchesLibrary,
    /// "Whenever [subject] deals combat damage to a player": once for each
    /// creature that does (`AnyYouControl`: "a creature you control").
    CreatureDealsCombatDamageToPlayer {
        subject: TriggerSubject,
    },
    /// "Whenever one or more [predicate] you control deal combat damage to a
    /// player": once per combat damage step.
    OneOrMoreDealCombatDamageToPlayer {
        predicate: CardPredicate,
    },
    /// "Whenever [subject] becomes tapped."
    BecomesTapped {
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
    /// "At the beginning of your upkeep."
    BeginningOfYourUpkeep,
    BeginningOfYourCombat,
    BeginningOfYourEndStep,
    /// "Whenever you draw your Nth card each turn." The per-player draw
    /// count resets every turn (see `PlayerTurnHistory::cards_drawn`).
    YouDrawNthCardThisTurn {
        n: u32,
    },
    /// "Whenever you put one or more [kind] counters on a creature": once
    /// per creature an instruction you control puts them on. The trigger's
    /// X is how many were put ("that much").
    YouPutCountersOnCreature {
        kind: CounterKind,
    },
    /// "Whenever you scry" (CR 701.22): once per scry instruction with a
    /// count above zero. (No current effect surveils.)
    YouScry,
    /// "Whenever an opponent draws their Nth card each turn."
    OpponentDrawsNthCardThisTurn {
        n: u32,
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
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Condition {
    YourTurn,
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
    /// "if you control a [predicate] permanent".
    ControlsMatching {
        predicate: CardPredicate,
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
    /// "if you've sacrificed `count` or more permanents this turn".
    SacrificedThisTurnAtLeast {
        count: u32,
    },
    /// At least one alternative holds ("as long as you control a land
    /// creature or a land entered the battlefield under your control this
    /// turn").
    AnyOf(Vec<Condition>),
    /// Resolution only: the resolving spell was kicked (CR 702.33).
    Kicked,
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
    /// Source question: the ability's source permanent is tapped ("if Katara
    /// is tapped").
    SourceTapped,
    /// Triggering question: the permanent whose event fired the trigger has
    /// (or, once it has left, had) counters on it ("if it had counters on
    /// it").
    TriggeringHadCounters,
    /// Resolution only: the frame's primary target was a card in a
    /// graveyard that this resolution exiled, and it matches [predicate]
    /// ("If a creature card is exiled this way").
    TargetCardExiled(CardPredicate),
}

impl Condition {
    /// Whether answering reads the asking ability's source permanent or the
    /// permanent that triggered it, so the condition is meaningful only in
    /// `Effect::If` and intervening-if clauses.
    pub fn reads_source(&self) -> bool {
        match self {
            Condition::SourceCountersAtLeast { .. }
            | Condition::SourceTapped
            | Condition::TriggeringHadCounters => true,
            Condition::AnyOf(conditions) => conditions.iter().any(Condition::reads_source),
            _ => false,
        }
    }

    /// Whether answering needs a resolving spell or ability, so the
    /// condition is meaningful only inside `Effect::If`.
    pub fn is_resolution_only(&self) -> bool {
        match self {
            Condition::Kicked
            | Condition::TargetMatches(_)
            | Condition::TargetDealtDamageThisTurn
            | Condition::TargetCardExiled(_) => true,
            Condition::AnyOf(conditions) => conditions.iter().any(Condition::is_resolution_only),
            Condition::SourceCountersAtLeast { .. }
            | Condition::SourceTapped
            | Condition::TriggeringHadCounters
            | Condition::YourTurn
            | Condition::Always
            | Condition::CardsDrawnAtLeast { .. }
            | Condition::GraveyardAtLeast { .. }
            | Condition::ControlsMatching { .. }
            | Condition::UnspentManaAtLeast { .. }
            | Condition::AttackedThisTurn
            | Condition::LandEnteredThisTurn
            | Condition::SacrificedThisTurnAtLeast { .. } => false,
        }
    }
}

/// Which game objects an event-based trigger condition watches.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum TriggerSubject {
    /// The permanent this ability is on.
    This,
    /// "another [predicate] you control"
    AnotherYouControl(CardPredicate),
    /// "this or another [predicate] you control"
    AnyYouControl(CardPredicate),
}

/// The players an effect affects, relative to its controller.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum PlayerSelector {
    /// The effect's controller ("you").
    You,
    /// Each opponent of the controller.
    EachOpponent,
    /// The trigger's player ("they": the caster of the triggering spell).
    TriggeringPlayer,
    /// A chosen target player (boxed: a spec is far larger than the rest).
    Target(Box<TargetSpec>),
    /// The controller of the frame's primary target permanent, as it is or
    /// last existed ("its controller").
    TargetController,
}

impl PlayerSelector {
    pub fn target_spec(&self) -> Option<&TargetSpec> {
        match self {
            Self::Target(spec) => Some(spec),
            Self::You | Self::EachOpponent | Self::TriggeringPlayer | Self::TargetController => {
                None
            }
        }
    }
}

/// A resolution-time quantity relative to the effect controller.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum EffectValue {
    Fixed(i32),
    /// The resolving spell's announced X (CR 107.3).
    X,
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
    /// Excess damage (CR 120.4a) the resolution's latest damage instruction
    /// dealt to its creature; 0 outside a resolution.
    ExcessDamage,
    /// The source permanent's current power ("where X is this creature's
    /// power"); 0 once it has left the battlefield or outside a resolution.
    SourcePower,
    /// Attacking creatures `controller` controls matching `predicate`, as
    /// the effect resolves ("search for up to that many").
    AttackingMatching {
        predicate: CardPredicate,
    },
    /// Permanents `controller` has sacrificed this turn.
    PermanentsSacrificedThisTurn,
    /// The power of the permanent whose event fired the trigger, from its
    /// last known information once it has left ("where X is that
    /// creature's power").
    TriggeringPower,
    /// Half the inner value, rounded down ("half X cards, rounded down").
    Half(Box<EffectValue>),
}

/// A resolving obligation, distinct from the effects chosen by its outcome.
/// Waterbend pays only generic mana; taps never count as mana production.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum PaymentCost {
    Mana(ManaCost),
    Waterbend { generic: u8 },
}

impl PaymentCost {
    pub fn mana_cost(&self) -> ManaCost {
        match self {
            Self::Mana(cost) => cost.clone(),
            Self::Waterbend { generic } => ManaCost::default().with_generic(*generic),
        }
    }
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

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Effect {
    /// A continuous effect from this resolution (CR 611): `modification`
    /// for `affected`, lasting `duration`. The record's amounts are read as
    /// it resolves.
    ApplyContinuous {
        affected: AffectedSpec,
        modification: Modification<EffectValue>,
        duration: EffectDuration,
    },
    /// Put `count` +1/+1 counters on the exact permanent that caused the
    /// trigger, without targeting.
    PutCountersOnTriggeringPermanent {
        count: i32,
    },
    /// Copy the source's copiable characteristics, removing legendary.
    CreateNonlegendaryCopy,
    OptionalPayment {
        cost: PaymentCost,
        if_paid: Vec<Effect>,
        if_declined: Vec<Effect>,
    },
    Sacrifice {
        predicate: CardPredicate,
    },
    /// "You may sacrifice [another] [predicate]. If you do, [then]." The
    /// controller chooses one or declines; `then` runs only after a
    /// sacrifice.
    MaySacrifice {
        predicate: CardPredicate,
        /// "another": the source permanent is not a candidate.
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        another: bool,
        then: Vec<Effect>,
    },
    /// Each opponent loses `amount` life and you gain that much.
    DrainLife {
        amount: EffectValue,
    },
    ReturnGraveyardTarget {
        target: TargetSpec,
        /// Onto the battlefield tapped instead of into its owner's hand.
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        tapped_onto_battlefield: bool,
    },
    MillCards {
        count: EffectValue,
        /// The target player mills instead of the controller.
        target: Option<TargetSpec>,
    },
    MillAndSelect {
        count: usize,
        predicate: CardPredicate,
        if_declined: Vec<Effect>,
        #[serde(skip_serializing_if = "Option::is_none")]
        if_milled: Option<MilledBonus>,
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
    /// Search is a private choice that always shuffles, including failure to find.
    SearchBasicLand {
        subtype: Option<String>,
        /// Up to this many cards, read at resolution.
        #[serde(skip_serializing_if = "is_fixed_one")]
        count: EffectValue,
        #[serde(skip_serializing_if = "is_true")]
        tapped: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        untap_if_lands: Option<usize>,
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        bottom_source: bool,
        /// Landcycling: any card of `subtype` (not only a basic land) is
        /// revealed and put into hand instead of onto the battlefield.
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        to_hand: bool,
    },
    /// Discard up to `up_to` cards, then draw `count` for each discarded.
    OptionalDiscardDraw {
        count: usize,
        #[serde(skip_serializing_if = "is_one")]
        up_to: usize,
    },
    DiscardTargetPlayer {
        count: usize,
        target: TargetSpec,
    },
    Destroy {
        target: TargetSpec,
    },
    Exile {
        target: TargetSpec,
    },
    /// Airbend (CR 701.65): exile the target; its owner may cast that exact
    /// exiled card for {2} rather than its mana cost.
    Airbend {
        target: TargetSpec,
    },
    /// "Exile [target], then return it to the battlefield under its
    /// owner's control": it returns as a new object (CR 400.7).
    ExileAndReturn {
        target: TargetSpec,
    },
    /// Airbend every battlefield permanent matching `predicate` other than
    /// the clause's chosen target, if any.
    AirbendAll {
        predicate: CardPredicate,
    },
    /// Until the controller's next turn, opponents cast spells only from
    /// their hands.
    OpponentsCastOnlyFromHand,
    /// "Exile the top card of your library. You may play that card this turn."
    ExileTopCardPlayThisTurn,
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
    /// "Look at the top card of your library. You may cast that card without
    /// paying its mana cost." Inside `MayOnceEachTurn`, casting spends the
    /// option.
    CastLibraryTopFree,
    /// You may cast a matching spell from your hand without paying its mana cost.
    CastFromHandFree {
        predicate: CardPredicate,
    },
    /// Shuffle your graveyard into your library.
    ShuffleGraveyardIntoLibrary,
    /// The target's owner chooses second from the top of their library or
    /// the bottom (a resolution-time mode of [`Effect::PlaceTargetInLibrary`]).
    OwnerPutsIntoLibrary {
        target: TargetSpec,
    },
    /// The target player reveals their hand (it becomes public knowledge);
    /// the effect's controller chooses a matching card and that player
    /// discards it.
    RevealHandDiscardChosen {
        target: TargetSpec,
        predicate: CardPredicate,
    },
    /// Search for up to `count` land cards with different names; an
    /// opponent puts `opponent_picks` of them into the graveyard and the rest
    /// enter tapped (Elemental Teachings).
    SearchLandsOpponentSplits {
        count: usize,
        opponent_picks: usize,
    },
    /// Put the primary target into its owner's library, second from the top
    /// or on the bottom.
    PlaceTargetInLibrary {
        second_from_top: bool,
    },
    /// Sacrifice the exact source permanent; only if that happened, run `then`.
    SacrificeSource {
        then: Vec<Effect>,
    },
    /// The resolving spell exiles itself instead of going to the graveyard.
    ExileSourceSpell,
    ReturnToHandDrawIfControlled {
        target: TargetSpec,
    },
    Discard,
    DealDamageValue {
        amount: EffectValue,
        target: TargetSpec,
    },
    PutCountersValue {
        count: EffectValue,
        target: TargetSpec,
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
    ReturnToHand {
        target: TargetSpec,
    },
    DealDamage {
        amount: i32,
        target: TargetSpec,
    },
    CounterSpell {
        target: TargetSpec,
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
    /// The source deals `amount` damage to the selected players.
    DealDamageToPlayer {
        player: PlayerSelector,
        amount: EffectValue,
    },
    /// Deal `amount` damage to each creature on the battlefield. No target.
    /// Each creature with mana value at most `max_mana_value` loses all
    /// abilities until end of turn, then those creatures are destroyed
    /// simultaneously (Day of Black Sun).
    DestroyCreaturesLosingAbilities {
        max_mana_value: EffectValue,
    },
    MassDamage {
        amount: i32,
    },
    DamageOpponentCreatures {
        amount: i32,
    },
    CreateTokensValue {
        token_name: String,
        count: EffectValue,
    },
    ExileWithControllerToken {
        target: TargetSpec,
        token_name: String,
    },
    /// Create `count` tokens from the registered token definition
    /// `token_name`, under the resolving player's control.
    CreateToken {
        token_name: String,
        count: usize,
        tapped_and_attacking: bool,
        /// "then attach this Equipment to it" (one token).
        attach_source: bool,
    },
    /// Attach the source Equipment to the target creature (CR 701.3). A
    /// resolving Aura spell instead enters attached to it (CR 303.4f).
    AttachSourceTo {
        target: TargetSpec,
    },
    /// Exile the object the source Aura or Equipment is attached to.
    ExileAttached,
    /// Tap the object the source is attached to.
    TapAttached,
    /// Remove all counters from the object the source is attached to.
    RemoveAllCountersFromAttached,
    /// "Distribute `count` [kind] counters among any number of tapped
    /// creatures your opponents control": the controller places them one
    /// at a time, choosing among the eligible creatures at each placement.
    DistributeCountersToTappedOpponentCreatures {
        kind: CounterKind,
        count: usize,
    },
    /// Return the permanent that granted this ability to its owner's hand
    /// ("Return Trusty Boomerang"); nothing if it has left the battlefield.
    ReturnGrantorToHand,
    /// The owner of the object the source is attached to shuffles it into
    /// their library.
    ShuffleAttachedIntoLibrary,
    /// Put `count` counters of `kind` on the source permanent, `count` read
    /// at resolution.
    PutCountersOnSource {
        count: EffectValue,
        kind: CounterKind,
    },
    /// "Put its counters on this creature": every counter the triggering
    /// permanent had as it last existed, kind by kind, on the source.
    PutTriggeringCountersOnSource,
    /// "[player] [effects]": the effects resolve with that player as "you"
    /// ("Its controller may search their library"), then the controller
    /// returns to the resolving one.
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
    /// "Put its counters on [target]": every counter the source had as it
    /// last existed, kind by kind, on the target (a leaves-the-battlefield
    /// ability, CR 603.10a).
    PutSourceCountersOnTarget {
        target: TargetSpec,
    },
    /// Put `count` counters of `kind` on the target.
    PutCounters {
        count: i32,
        kind: CounterKind,
        target: TargetSpec,
    },
    /// Create a delayed triggered ability (CR 603.7): when `event` occurs,
    /// `effects` trigger from this spell or ability's source. Their targets
    /// are chosen as the delayed ability is put on the stack.
    CreateDelayedTrigger {
        event: DelayedEvent,
        effects: Vec<Effect>,
    },
    /// Sacrifice the permanent the trigger names ("sacrifice that token"),
    /// if its controller still controls it.
    SacrificeTriggeringPermanent,
    /// Tap the source permanent.
    TapSource,
    /// Untap the source permanent.
    UntapSource,
    /// "Return this card from your graveyard to your hand": only the exact
    /// graveyard object the ability came from (CR 400.7).
    ReturnSourceToHand,
    /// "Return this card from your graveyard to the battlefield with a
    /// [counter] counter on it": only the exact graveyard object the ability
    /// came from, under its owner's control.
    ReturnSourceToBattlefield {
        counter: CounterKind,
    },
    /// Resolving player gains `amount` life.
    GainLife {
        amount: i32,
    },
    GainLifeValue {
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
    /// "Look at the top `look` cards of your library. Put up to
    /// `max_select` cards matching [predicate] from among them into your
    /// hand and the rest on the bottom of your library in a random order."
    /// `min_select` > 0 makes the selection mandatory when possible.
    LookAndSelect {
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        reveal: bool,
        look: EffectValue,
        min_select: usize,
        max_select: usize,
        predicate: CardPredicate,
    },
    /// Put the top `count` cards of the controller's library into their
    /// hand (not a draw — no draw triggers).
    PutTopCardsInHand {
        count: usize,
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
    /// card (`CardDefinition::targeting`) when a choice is required.
    CounterUnlessPays {
        cost: PaymentCost,
    },
    /// Allies at Last: all chosen targets except the last each deal damage
    /// equal to their power to the last target. Targeting is declared on
    /// the card (`CardDefinition::targeting`).
    TargetCreaturesDealPowerDamageToLastTarget,
    /// Earthbend N: target land you control becomes a 0/0 creature with
    /// haste that's still a land; put `count` +1/+1 counters on it. A
    /// delayed trigger returns it to the battlefield tapped when it dies
    /// or is exiled.
    Earthbend {
        count: EffectValue,
        target: TargetSpec,
    },
    /// Delayed-trigger payload (earthbend): return the source card from
    /// the graveyard or exile to the battlefield tapped.
    ReturnSourceToBattlefieldTapped,
    /// "Exile [target] until this creature leaves the battlefield." The
    /// exile only happens if the source is still on the battlefield when
    /// this resolves (CR 603.6e pragmatics); the linked card returns
    /// immediately (no stack) when the source leaves.
    ExileUntilSourceLeaves {
        target: TargetSpec,
    },
    /// Add mana to the resolving player's pool. `until_end_of_combat`
    /// routes it to the combat pool (firebending — persists across combat
    /// steps, empties at end of combat).
    AddMana {
        mana: Mana,
        until_end_of_combat: bool,
    },
    /// "Add X mana of any one color", X read as the mana ability resolves.
    AddOneColorManaValue {
        amount: EffectValue,
    },
    /// Add `mana` `amount` times, `amount` read at resolution;
    /// `until_end_of_combat` as for [`Effect::AddMana`] (firebending X).
    AddManaValue {
        mana: Mana,
        amount: EffectValue,
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        until_end_of_combat: bool,
    },
    /// The first chosen creature fights the last (CR 701.14); records the
    /// excess damage dealt to the last.
    FightLastTarget,
    /// Search your library for a card matching `predicate`, reveal it, put
    /// it into your hand, then shuffle (CR 701.23).
    SearchLibraryToHand {
        predicate: CardPredicate,
    },
    /// Untap the target.
    UntapTarget {
        target: TargetSpec,
    },
    /// Tap the target.
    TapTarget {
        target: TargetSpec,
    },
    /// "Remove up to `up_to` counters from [target]." With `chosen` unset the
    /// controller picks a present kind or stops; with it set, one counter of
    /// that kind is removed and the choice repeats for `up_to - 1`.
    RemoveCounters {
        up_to: i32,
        chosen: Option<CounterKind>,
        target: TargetSpec,
    },
    /// Run the inner effects once per chosen target, with that target as
    /// the primary target (Fancy Footwork: "untap one or two target
    /// creatures; they each get +2/+2"). Inner effects must not suspend.
    ForEachTarget {
        effects: Vec<Effect>,
    },
    /// Put `count` +1/+1 counters on each battlefield permanent the
    /// resolving player controls matching [predicate]; `other` excludes
    /// the source permanent.
    PutCountersOnEachMatching {
        count: EffectValue,
        predicate: CardPredicate,
        other: bool,
    },
}

/// The event a delayed triggered ability created by
/// [`Effect::CreateDelayedTrigger`] waits for (CR 603.7).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum DelayedEvent {
    /// "When that creature dies this turn": the frame's primary target.
    TargetDiesThisTurn,
    /// "Whenever a creature dies this turn."
    CreatureDiesThisTurn,
    /// "At the beginning of the next end step", about the token the
    /// resolution created last ("sacrifice that token").
    NextEndStepForCreatedToken,
    /// "When you next cast a [predicate] spell this turn"; that spell is
    /// the delayed ability's context.
    YouNextCastThisTurn(CardPredicate),
}

/// Effects that run when any card milled by the same instruction matches.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct MilledBonus {
    pub predicate: CardPredicate,
    pub effects: Vec<Effect>,
}

impl Modification<EffectValue> {
    /// "+power/+toughness".
    pub fn pump(power: i32, toughness: i32) -> Self {
        Modification::ModifyPt {
            power: EffectValue::Fixed(power),
            toughness: EffectValue::Fixed(toughness),
        }
    }

    /// Fixed base power and/or toughness (layer 7b).
    pub fn base_pt(power: Option<i32>, toughness: Option<i32>) -> Self {
        Modification::SetBasePt {
            power: power.map(EffectValue::Fixed),
            toughness: toughness.map(EffectValue::Fixed),
        }
    }
}

/// Whom a resolving continuous effect affects. A set of permanents is fixed
/// as the effect resolves (CR 611.2c).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum AffectedSpec {
    /// The resolving player ("you").
    You,
    /// The source permanent.
    Source,
    /// The frame's primary target, when it is still a permanent.
    Target(TargetSpec),
    /// The permanent whose event fired the trigger.
    TriggeringPermanent,
    /// Each permanent the resolving player controls matching the predicate.
    EachYouControl(CardPredicate),
    /// Each battlefield permanent matching the predicate ("all creatures").
    EachMatching(CardPredicate),
    /// Each permanent the frame's primary target player controls matching
    /// the predicate ("creatures target player controls").
    EachTargetPlayerControls(CardPredicate),
}

/// The target spec of [`AffectedSpec::EachTargetPlayerControls`].
static TARGET_PLAYER: TargetSpec = TargetSpec::Player;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum TargetSpec {
    PermanentYouControl {
        predicate: CardPredicate,
    },
    GraveyardCardYouOwn {
        predicate: CardPredicate,
    },
    /// "target [predicate] card from a graveyard": any player's.
    GraveyardCard {
        predicate: CardPredicate,
    },
    Player,
    Opponent,
    Permanent {
        predicate: CardPredicate,
    },
    Creature,
    CreatureOrPlayer,
    Spell,
    /// "target spell or permanent with mana value `min_mana_value` or
    /// greater" (Divide by Zero).
    SpellOrPermanent {
        min_mana_value: u8,
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
    /// "Up to X targets": the announced X replaces `max` (CR 601.2c).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub max_x: bool,
}

impl TargetRequirement {
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
fn is_true(value: &bool) -> bool {
    *value
}

impl Default for Effect {
    fn default() -> Self {
        Self::DrawCards {
            count: EffectValue::Fixed(0),
        }
    }
}

impl Effect {
    /// Whether the effect reads the permanent whose event fired its
    /// trigger, which the trigger then keeps as its context.
    pub fn reads_triggering_permanent(&self) -> bool {
        match self {
            Effect::PutCountersOnTriggeringPermanent { .. }
            | Effect::PutTriggeringCountersOnSource
            | Effect::ApplyContinuous {
                affected: AffectedSpec::TriggeringPermanent,
                ..
            } => true,
            Effect::Earthbend { count: value, .. }
            | Effect::DrawCards { count: value }
            | Effect::GainLifeValue { amount: value }
            | Effect::DealDamageValue { amount: value, .. }
            | Effect::PutCountersValue { count: value, .. } => {
                *value == EffectValue::TriggeringPower
            }
            _ => false,
        }
    }

    /// One immediate mana output per mode, preserving the authored color order.
    pub(crate) fn mana_choice(outputs: impl IntoIterator<Item = Mana>) -> Self {
        Self::Modal {
            modes: outputs
                .into_iter()
                .map(|mana| {
                    vec![Self::AddMana {
                        mana,
                        until_end_of_combat: false,
                    }]
                })
                .collect(),
        }
    }

    pub(crate) fn any_color_mana() -> Self {
        Self::mana_choice(["W", "U", "B", "R", "G"].map(Mana::parse))
    }

    /// "[affected] [modification] until end of turn."
    pub fn until_end_of_turn(
        affected: AffectedSpec,
        modification: Modification<EffectValue>,
    ) -> Self {
        Effect::ApplyContinuous {
            affected,
            modification,
            duration: EffectDuration::EndOfTurn,
        }
    }

    /// A continuous effect with no stated duration.
    pub fn indefinitely(affected: AffectedSpec, modification: Modification<EffectValue>) -> Self {
        Effect::ApplyContinuous {
            affected,
            modification,
            duration: EffectDuration::Indefinite,
        }
    }

    pub fn target_spec(&self) -> Option<&TargetSpec> {
        match self {
            Effect::Sacrifice { .. } | Effect::DrainLife { .. } => None,
            Effect::LoseLife { player, .. } | Effect::DealDamageToPlayer { player, .. } => {
                player.target_spec()
            }
            Effect::TargetClause { requirement, .. } => Some(&requirement.spec),
            Effect::MillCards { target, .. } => target.as_ref(),
            Effect::ExileLinked { target, .. } => Some(target),
            Effect::SelectTargetRequirement { .. } | Effect::MillAndSelect { .. } => None,
            Effect::ApplyContinuous {
                affected: AffectedSpec::EachTargetPlayerControls(_),
                ..
            } => Some(&TARGET_PLAYER),
            Effect::ApplyContinuous {
                affected: AffectedSpec::Target(target),
                ..
            }
            | Effect::ReturnGraveyardTarget { target, .. }
            | Effect::ExileWithControllerToken { target, .. }
            | Effect::DiscardTargetPlayer { target, .. }
            | Effect::Destroy { target }
            | Effect::Exile { target }
            | Effect::Airbend { target }
            | Effect::ExileAndReturn { target }
            | Effect::ReturnToHandDrawIfControlled { target } => Some(target),
            Effect::SearchBasicLand { .. }
            | Effect::OptionalDiscardDraw { .. }
            | Effect::ApplyContinuous { .. }
            | Effect::PutCountersOnTriggeringPermanent { .. } => None,
            Effect::DealDamageValue { target, .. } | Effect::PutCountersValue { target, .. } => {
                Some(target)
            }
            Effect::Discard => None,
            Effect::ReturnToHand { target } => Some(target),
            Effect::DealDamage { target, .. } => Some(target),
            Effect::CounterSpell { target } => Some(target),
            Effect::PutCounters { target, .. } | Effect::PutSourceCountersOnTarget { target } => {
                Some(target)
            }
            Effect::Earthbend { target, .. } => Some(target),
            Effect::ExileUntilSourceLeaves { target } => Some(target),
            Effect::UntapTarget { target }
            | Effect::TapTarget { target }
            | Effect::AttachSourceTo { target }
            | Effect::RemoveCounters { target, .. }
            | Effect::OwnerPutsIntoLibrary { target }
            | Effect::RevealHandDiscardChosen { target, .. } => Some(target),
            Effect::OnNthResolutionThisTurn { effect, .. } => effect.target_spec(),
            Effect::OptionalPayment {
                if_paid: then,
                if_declined: otherwise,
                ..
            }
            | Effect::If {
                then, otherwise, ..
            } => then
                .iter()
                .chain(otherwise.iter())
                .find_map(|effect| effect.target_spec()),
            Effect::SacrificeSource { then }
            | Effect::ForPlayer { effects: then, .. }
            | Effect::MaySacrifice { then, .. }
            | Effect::MayOnceEachTurn { effects: then } => {
                then.iter().find_map(|effect| effect.target_spec())
            }
            Effect::CreateNonlegendaryCopy
            | Effect::AirbendAll { .. }
            | Effect::OpponentsCastOnlyFromHand
            | Effect::ExileSourceSpell
            | Effect::ExileTopCardPlayThisTurn
            | Effect::ExileUntilNonlandCastFree { .. }
            | Effect::CastFromHandFree { .. }
            | Effect::CastLibraryTopFree
            | Effect::WhenYouDo { .. }
            | Effect::TriggerModes { .. }
            | Effect::ShuffleGraveyardIntoLibrary
            | Effect::PlaceTargetInLibrary { .. }
            | Effect::SearchLandsOpponentSplits { .. }
            | Effect::DrawCards { .. }
            | Effect::MassDamage { .. }
            | Effect::DestroyCreaturesLosingAbilities { .. }
            | Effect::DamageOpponentCreatures { .. }
            | Effect::CreateTokensValue { .. }
            | Effect::CreateToken { .. }
            | Effect::PutCountersOnSource { .. }
            | Effect::TapSource
            | Effect::CreateDelayedTrigger { .. }
            | Effect::PutTriggeringCountersOnSource
            | Effect::RestoreController(_)
            | Effect::EachPlayerKeepsCreatures { .. }
            | Effect::SacrificeTriggeringPermanent
            | Effect::UntapSource
            | Effect::ExileAttached
            | Effect::TapAttached
            | Effect::RemoveAllCountersFromAttached
            | Effect::DistributeCountersToTappedOpponentCreatures { .. }
            | Effect::ReturnGrantorToHand
            | Effect::ShuffleAttachedIntoLibrary
            | Effect::ReturnSourceToHand
            | Effect::ReturnSourceToBattlefield { .. }
            | Effect::GainLife { .. }
            | Effect::GainLifeValue { .. }
            | Effect::AddManaValue { .. }
            | Effect::FightLastTarget
            | Effect::SearchLibraryToHand { .. }
            | Effect::Scry { .. }
            | Effect::LookAndSelect { .. }
            | Effect::PutTopCardsInHand { .. }
            | Effect::Learn
            | Effect::Modal { .. }
            | Effect::CounterUnlessPays { .. }
            | Effect::ReturnSourceToBattlefieldTapped
            | Effect::AddMana { .. }
            | Effect::AddOneColorManaValue { .. }
            | Effect::ForEachTarget { .. }
            | Effect::PutCountersOnEachMatching { .. }
            | Effect::TargetCreaturesDealPowerDamageToLastTarget => None,
        }
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
            .map(|spec| {
                // The implicit exile-link clause is "up to one target".
                let optional = effects
                    .iter()
                    .any(|effect| matches!(effect, Effect::ExileUntilSourceLeaves { .. }));
                vec![TargetRequirement {
                    min: usize::from(!optional),
                    ..TargetRequirement::one(spec.clone())
                }]
            })
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
