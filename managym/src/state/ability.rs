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
    /// "Whenever a creature you control attacking causes a triggered ability
    /// of that creature to trigger": that ability, once on the stack, is the
    /// trigger's context.
    YourAttackingCreatureTriggers,
    /// "At the beginning of your first main phase" (the precombat main).
    BeginningOfYourFirstMainPhase,
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
    /// Resolution only: a card the resolution's latest discard moved matches
    /// [predicate] ("When you discard a nonland card this way").
    DiscardedThisWay(CardPredicate),
    /// Resolution only: the spell whose casting triggered the ability
    /// matches [predicate] ("If that spell is a Lesson").
    TriggeringSpellMatches(CardPredicate),
    /// Resolution only: the frame's primary target was a card in a
    /// graveyard that this resolution exiled, and it matches [predicate]
    /// ("If a creature card is exiled this way").
    TargetCardExiled(CardPredicate),
    /// Resolution only: the resolution's latest `Move` or `Sacrifice`
    /// moved at least one object ("If you do").
    Moved,
    /// Resolution only: the frame's primary target is a permanent the
    /// resolving player controls, or last controlled once it has left ("If
    /// you controlled it").
    TargetControlledByYou,
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

    /// Whether answering reads a player's mana pools, so a static ability
    /// with this condition follows the `mana_pools` revision.
    pub fn reads_mana_pools(&self) -> bool {
        match self {
            Condition::UnspentManaAtLeast { .. } => true,
            Condition::AnyOf(conditions) => conditions.iter().any(Condition::reads_mana_pools),
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
            | Condition::TargetCardExiled(_)
            | Condition::DiscardedThisWay(_)
            | Condition::TriggeringSpellMatches(_)
            | Condition::Moved
            | Condition::TargetControlledByYou => true,
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
    /// "enchanted creature": the object the source Aura is attached to.
    EnchantedCreature,
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
            | Self::TriggeringPlayer
            | Self::TargetController
            | Self::Owner => None,
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
    /// Permanents `controller` owns that an opponent controls.
    OwnedPermanentsOpponentsControl,
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

/// Which objects a verb acts on, read as the instruction resolves
/// (`Game::select`). A verb acts only on the objects it can: counters, taps
/// and sacrifices need battlefield permanents, a move any card object, a
/// continuous effect a permanent or the player `You` names.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum ObjectSelector {
    /// The source: the exact permanent, graveyard card or spell the ability
    /// or spell belongs to. Once the source has left the battlefield, the
    /// card it became in the graveyard or exile it went to (CR 400.7,
    /// 603.10a: "return this card", "exile it"); never a later object.
    This,
    /// The clause's chosen target. The spec declares the implicit targeting
    /// clause of an effect list without an explicit `TargetClause`.
    Target(TargetSpec),
    /// The permanent whose event fired the trigger ("that creature", "that
    /// token"), which the trigger keeps as its context.
    Triggering,
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
    },
}

/// Whose battlefield permanents an [`ObjectSelector::Each`] ranges over.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
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
}

/// The one permanent a `Chosen` or `Each` selector leaves out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Except {
    /// "another": the source permanent.
    This,
    /// "all other": the clause's chosen target (Avatar's Wrath).
    Target,
}

impl ObjectSelector {
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
        }
    }

    /// "Each other [predicate] you control".
    pub fn each_other_you_control(predicate: CardPredicate) -> Self {
        Self::Each {
            of: Scope::YouControl,
            predicate,
            except: Some(Except::This),
            max_mana_value: None,
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
            Self::Target(spec) => Some(spec),
            Self::Each {
                of: Scope::TargetPlayerControls,
                ..
            } => Some(&TARGET_PLAYER),
            Self::This
            | Self::Triggering
            | Self::Attached
            | Self::CreatedThisWay
            | Self::Grantor
            | Self::Moved
            | Self::You
            | Self::Chosen { .. }
            | Self::Each { .. } => None,
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
        what: ObjectSelector,
        to: ZoneType,
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        tapped: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        under: Option<PlayerSelector>,
        #[serde(skip_serializing_if = "Option::is_none")]
        with_counter: Option<CounterKind>,
        #[serde(skip_serializing_if = "Option::is_none")]
        linked: Option<LinkedExile>,
    },
    /// Tap each permanent `what` names.
    Tap { what: ObjectSelector },
    /// Untap each permanent `what` names.
    Untap { what: ObjectSelector },
    /// Remove every counter from each permanent `from` names.
    RemoveAllCounters { from: ObjectSelector },
    /// Create a token that's a copy of the source (CR 707.2, 111.1). The
    /// token copies the source's copiable values: its definition, including
    /// any copy effect already applied to it, and never its counters, damage,
    /// tapped state or non-copy effects. A source that has left the
    /// battlefield is copied as it last existed there. `exceptions` ("except
    /// it's not legendary") become part of the token's own copiable values
    /// (CR 707.9b), so a copy of it inherits them.
    CreateTokenCopy {
        exceptions: Vec<Modification<EffectValue>>,
    },
    OptionalPayment {
        cost: PaymentCost,
        if_paid: Vec<Effect>,
        if_declined: Vec<Effect>,
    },
    /// Each opponent loses `amount` life and you gain that much.
    DrainLife { amount: EffectValue },
    /// "Return up to one [predicate] card milled this way to the
    /// battlefield under your control": a choice among the cards the
    /// resolution's latest mill moved, still in the graveyard.
    ReturnMilledThisWay { predicate: CardPredicate },
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
    SelectTargetRequirement { index: usize },
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
    /// The player `what` names discards `count` cards, chosen by them.
    Discard {
        what: ObjectSelector,
        #[serde(skip_serializing_if = "is_one")]
        count: usize,
    },
    /// Destroy each permanent `what` names (CR 701.7), as one simultaneous
    /// event: leave-the-battlefield abilities look back at the permanents
    /// as they all were before (CR 603.10a).
    Destroy { what: ObjectSelector },
    /// Airbend (CR 701.65) each permanent `what` names: exile it; its owner
    /// may cast that exact exiled card for {2} rather than its mana cost.
    Airbend { what: ObjectSelector },
    /// "[player] gains control of [what]" for `duration` (CR 613.1b).
    GainControl {
        player: PlayerSelector,
        what: ObjectSelector,
        duration: EffectDuration,
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
    ExileUntilNonlandCastFree { below: EffectValue },
    /// "Look at the top card of your library. You may cast that card without
    /// paying its mana cost." Inside `MayOnceEachTurn`, casting spends the
    /// option.
    CastLibraryTopFree,
    /// You may cast a matching spell from your hand without paying its mana cost.
    CastFromHandFree { predicate: CardPredicate },
    /// Shuffle your graveyard into your library.
    ShuffleGraveyardIntoLibrary,
    /// The target's owner chooses second from the top of their library or
    /// the bottom (a resolution-time mode of [`Effect::PlaceTargetInLibrary`]).
    OwnerPutsIntoLibrary { target: TargetSpec },
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
    SearchLandsOpponentSplits { count: usize, opponent_picks: usize },
    /// Put the primary target into its owner's library, second from the top
    /// or on the bottom.
    PlaceTargetInLibrary { second_from_top: bool },
    /// "You may [effects]. Do this only once each turn." Offered only while
    /// the exact source has not taken this ability's option this turn; only
    /// accepting spends it (the ability itself triggers every time).
    MayOnceEachTurn { effects: Vec<Effect> },
    /// A reflexive trigger (CR 603.12): "When you do, [effects]." It
    /// triggers where it appears, so it only runs if the instruction before
    /// it happened; its targets are chosen as it is put on the stack.
    WhenYouDo { effects: Vec<Effect> },
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
    /// last creature is kept for a following "the excess damage".
    DealDamage {
        amount: EffectValue,
        what: ObjectSelector,
    },
    /// Counter each spell `what` names (CR 701.6a); abilities are not
    /// affected.
    CounterSpell { what: ObjectSelector },
    /// "Counter target activated or triggered ability" (CR 701.6b): the
    /// ability leaves the stack and does nothing; spells are not affected.
    /// A stack ability is no card object, so the clause's chosen target is
    /// read directly.
    CounterAbility { what: ObjectSelector },
    /// Resolving player draws `count` cards, read at resolution. No target.
    DrawCards { count: EffectValue },
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
    /// Create `count` tokens from the registered token definition
    /// `token_name`, under the resolving player's control, `count` read as
    /// the instruction resolves.
    CreateToken {
        token_name: String,
        count: EffectValue,
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        tapped_and_attacking: bool,
        /// "then attach this Equipment to it" (one token).
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        attach_source: bool,
    },
    /// Attach the source Equipment to the target creature (CR 701.3). A
    /// resolving Aura spell instead enters attached to it (CR 303.4f).
    AttachSourceTo { target: TargetSpec },
    /// "Distribute `count` [kind] counters among any number of tapped
    /// creatures your opponents control": the controller places them one
    /// at a time, choosing among the eligible creatures at each placement.
    DistributeCountersToTappedOpponentCreatures { kind: CounterKind, count: usize },
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
    EachPlayerKeepsCreatures { max_total_power: i32 },
    /// Create a delayed triggered ability (CR 603.7): when `event` occurs,
    /// `effects` trigger from this spell or ability's source. Their targets
    /// are chosen as the delayed ability is put on the stack.
    CreateDelayedTrigger {
        event: DelayedEvent,
        effects: Vec<Effect>,
    },
    /// Resolving player gains `amount` life, read as the instruction
    /// resolves.
    GainLife { amount: EffectValue },
    /// Execute the inner effect only if this is the `n`th time this
    /// ability has resolved this turn ("if this is the second time this
    /// ability has resolved this turn, ...").
    OnNthResolutionThisTurn { n: u32, effect: Box<Effect> },
    /// Scry `count` (CR 701.26), with arbitrary top and bottom ordering.
    Scry { count: usize },
    /// Surveil `count` (CR 701.25): look at the top cards and put any
    /// number of them into the graveyard. Kept cards stay on top in their
    /// existing order, which is the whole choice for surveil 1, the only
    /// amount the pool prints.
    Surveil { count: usize },
    /// "Look at the top `look` cards of your library. Put up to
    /// `max_select` cards matching [predicate] from among them into your
    /// hand and the rest on the bottom of your library."
    /// `min_select` > 0 makes the selection mandatory when possible.
    LookAndSelect {
        /// Otherwise the owner chooses the order (CR 401.4).
        random_order: bool,
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        reveal: bool,
        look: EffectValue,
        min_select: usize,
        max_select: usize,
        predicate: CardPredicate,
    },
    /// Put the top `count` cards of the controller's library into their
    /// hand (not a draw — no draw triggers).
    PutTopCardsInHand { count: usize },
    /// "You may discard a card. If you do, [then]."
    MayDiscard { then: Vec<Effect> },
    /// Copy the spell or ability the trigger names (its context): "copy it",
    /// "copy that ability" (CR 707.10). The copy may get new targets.
    CopyTriggeringObject,
    /// "Change the target of target spell or ability with a single target"
    /// (CR 115.7): to another legal target, if there is one.
    ChangeTarget { target: TargetSpec },
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
    Modal { modes: Vec<Vec<Effect>> },
    /// "Counter [the spell or ability] unless its controller pays [cost]."
    /// It is the frame's primary target: either a chosen `TargetSpec::Spell`
    /// target (It'll Quench Ya!) or the trigger-context spell or ability
    /// (ward).
    /// Reports no target spec of its own — targeting is declared on the
    /// card (`CardDefinition::targeting`) when a choice is required.
    CounterUnlessPays { cost: PaymentCost },
    /// Allies at Last: all chosen targets except the last each deal damage
    /// equal to their power to the last target. Targeting is declared on
    /// the card (`CardDefinition::targeting`).
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
    CastExiledWithSource { predicate: CardPredicate },
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
    AddOneColorManaValue { amount: EffectValue },

    /// The first chosen creature fights the last (CR 701.14); records the
    /// excess damage dealt to the last.
    FightLastTarget,
    /// Search your library for a card matching `predicate`, reveal it, put
    /// it into your hand, then shuffle (CR 701.23).
    SearchLibraryToHand { predicate: CardPredicate },
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
    /// creatures; they each get +2/+2"). Inner effects must not suspend.
    ForEachTarget { effects: Vec<Effect> },
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

/// The target spec of [`Scope::TargetPlayerControls`].
static TARGET_PLAYER: TargetSpec = TargetSpec::Player;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
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
    Player,
    Opponent,
    Permanent {
        predicate: CardPredicate,
    },
    Creature,
    /// Creature, player, planeswalker or battle (CR 115.4).
    AnyTarget,
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
        if self
            .selectors()
            .into_iter()
            .any(|selector| *selector == ObjectSelector::Triggering)
        {
            return true;
        }
        match self {
            Effect::Earthbend { count: value, .. }
            | Effect::MillCards { count: value, .. }
            | Effect::DrawCards { count: value }
            | Effect::GainLife { amount: value }
            | Effect::DealDamage { amount: value, .. }
            | Effect::PutCounters { amount: value, .. } => *value == EffectValue::TriggeringPower,
            _ => false,
        }
    }

    /// The selectors a verb acts through, in field order; empty for every
    /// other effect.
    pub fn selectors(&self) -> Vec<&ObjectSelector> {
        match self {
            Effect::PutCounters { on, .. } => vec![on],
            Effect::PutCountersFrom { from, on } => vec![from, on],
            Effect::ApplyContinuous { affected: what, .. }
            | Effect::Sacrifice { what, .. }
            | Effect::Move { what, .. }
            | Effect::Tap { what }
            | Effect::Untap { what }
            | Effect::Destroy { what }
            | Effect::Airbend { what }
            | Effect::GainControl { what, .. }
            | Effect::DealDamage { what, .. }
            | Effect::CounterSpell { what }
            | Effect::CounterAbility { what }
            | Effect::Earthbend { what, .. }
            | Effect::Discard { what, .. } => vec![what],
            Effect::RemoveCounters { from, .. } | Effect::RemoveAllCounters { from } => {
                vec![from]
            }
            _ => Vec::new(),
        }
    }

    /// The choice a `Chosen` selector asks of the resolving player, if the
    /// verb has one; only `Sacrifice` may let them decline.
    pub fn chosen(&self) -> Option<(&CardPredicate, Option<Except>, bool)> {
        let (predicate, except) =
            self.selectors()
                .into_iter()
                .find_map(|selector| match selector {
                    ObjectSelector::Chosen { predicate, except } => Some((predicate, *except)),
                    _ => None,
                })?;
        let optional = matches!(self, Effect::Sacrifice { optional: true, .. });
        Some((predicate, except, optional))
    }

    /// Whether the effect asks about the spell whose casting triggered it,
    /// which the trigger then keeps as its context.
    pub fn asks_triggering_spell(&self) -> bool {
        match self {
            Effect::If {
                condition,
                then,
                otherwise,
            } => {
                matches!(condition, Condition::TriggeringSpellMatches(_))
                    || then
                        .iter()
                        .chain(otherwise)
                        .any(Effect::asks_triggering_spell)
            }
            Effect::TargetClause { effects, .. } => {
                effects.iter().any(Effect::asks_triggering_spell)
            }
            _ => false,
        }
    }

    /// One immediate mana output per mode, preserving the authored color order.
    pub(crate) fn mana_choice(outputs: impl IntoIterator<Item = Mana>) -> Self {
        Self::Modal {
            modes: outputs
                .into_iter()
                .map(|mana| vec![Self::add_mana(mana)])
                .collect(),
        }
    }

    pub(crate) fn any_color_mana() -> Self {
        Self::mana_choice(["W", "U", "B", "R", "G"].map(Mana::parse))
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

    /// A continuous effect with no stated duration.
    pub fn indefinitely(affected: ObjectSelector, modification: Modification<EffectValue>) -> Self {
        Effect::ApplyContinuous {
            affected,
            modification,
            duration: EffectDuration::Indefinite,
        }
    }

    pub fn target_spec(&self) -> Option<&TargetSpec> {
        if let Some(spec) = self
            .selectors()
            .into_iter()
            .find_map(ObjectSelector::target_spec)
        {
            return Some(spec);
        }
        match self {
            Effect::LoseLife { player, .. } | Effect::DealDamageToPlayer { player, .. } => {
                player.target_spec()
            }
            Effect::TargetClause { requirement, .. } => Some(&requirement.spec),
            Effect::MillCards { target, .. } => target.as_ref(),
            Effect::ExileLinked { target, .. } => Some(target),
            Effect::ChangeTarget { target }
            | Effect::AttachSourceTo { target }
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
            Effect::MayDiscard { then }
            | Effect::ForPlayer { effects: then, .. }
            | Effect::MayOnceEachTurn { effects: then } => {
                then.iter().find_map(|effect| effect.target_spec())
            }
            Effect::PutCounters { .. }
            | Effect::PutCountersFrom { .. }
            | Effect::Sacrifice { .. }
            | Effect::Move { .. }
            | Effect::Tap { .. }
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
            | Effect::DrainLife { .. }
            | Effect::SelectTargetRequirement { .. }
            | Effect::MillAndSelect { .. }
            | Effect::ApplyContinuous { .. }
            | Effect::SearchBasicLand { .. }
            | Effect::OptionalDiscardDraw { .. }
            | Effect::CreateTokenCopy { .. }
            | Effect::OpponentsCastOnlyFromHand
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
            | Effect::CreateToken { .. }
            | Effect::CreateDelayedTrigger { .. }
            | Effect::CopyTriggeringObject
            | Effect::ReturnMilledThisWay { .. }
            | Effect::CastExiledWithSource { .. }
            | Effect::RestoreController(_)
            | Effect::EachPlayerKeepsCreatures { .. }
            | Effect::DistributeCountersToTappedOpponentCreatures { .. }
            | Effect::GainLife { .. }
            | Effect::FightLastTarget
            | Effect::SearchLibraryToHand { .. }
            | Effect::Scry { .. }
            | Effect::Surveil { .. }
            | Effect::LookAndSelect { .. }
            | Effect::PutTopCardsInHand { .. }
            | Effect::Learn
            | Effect::Modal { .. }
            | Effect::CounterUnlessPays { .. }
            | Effect::AddMana { .. }
            | Effect::AddOneColorManaValue { .. }
            | Effect::ForEachTarget { .. }
            | Effect::TargetCreaturesDealPowerDamageToLastTarget => None,
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
        }
    }

    /// "You gain [amount] life."
    pub fn gain_life(amount: impl Into<EffectValue>) -> Self {
        Effect::GainLife {
            amount: amount.into(),
        }
    }

    /// "Discard a card."
    pub fn discard() -> Self {
        Effect::Discard {
            what: ObjectSelector::You,
            count: 1,
        }
    }

    /// "Create [count] [token_name] tokens."
    pub fn create_tokens(token_name: &str, count: impl Into<EffectValue>) -> Self {
        Effect::CreateToken {
            token_name: token_name.into(),
            count: count.into(),
            tapped_and_attacking: false,
            attach_source: false,
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
        Effect::Move {
            what,
            to,
            tapped: false,
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

    /// "Return [what] to the battlefield [tapped]" under the resolving
    /// player's control.
    pub fn return_to_battlefield(what: ObjectSelector, tapped: bool) -> Self {
        Effect::Move {
            what,
            to: ZoneType::Battlefield,
            tapped,
            under: None,
            with_counter: None,
            linked: None,
        }
    }

    /// "Exile [what] with this" or "until this leaves the battlefield".
    pub fn exile_linked(what: ObjectSelector, linked: LinkedExile) -> Self {
        Effect::Move {
            what,
            to: ZoneType::Exile,
            tapped: false,
            under: None,
            with_counter: None,
            linked: Some(linked),
        }
    }

    pub fn tap(what: ObjectSelector) -> Self {
        Effect::Tap { what }
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

impl From<i32> for EffectValue {
    fn from(value: i32) -> Self {
        EffectValue::Fixed(value)
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
                let optional = effects.iter().any(|effect| {
                    matches!(
                        effect,
                        Effect::Move {
                            linked: Some(LinkedExile::UntilSourceLeaves),
                            ..
                        }
                    )
                });
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
