use super::card::Keywords;
use super::mana::{Mana, ManaCost};
use super::permanent::{BecomesCreature, CounterKind};
use super::predicate::CardPredicate;
use super::zone::ZoneType;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Ability {
    Triggered {
        condition: TriggerCondition,
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
    /// "When [subject] enters the battlefield."
    EntersTheBattlefield {
        subject: TriggerSubject,
    },
    /// "When [subject] dies." — battlefield to graveyard (CR 700.4).
    Dies {
        subject: TriggerSubject,
    },
    /// "Whenever [subject] attacks." Fires after the declare-attackers
    /// turn-based action (CR 508.1); a subject of `AnotherYouControl` /
    /// `AnyYouControl` fires once per combat ("whenever one or more ...
    /// attack"), not once per attacker.
    Attacks {
        subject: TriggerSubject,
    },
    /// "Whenever [subject] attacks alone": exactly one creature was
    /// declared as an attacker (CR 506.5). The attacker is the context.
    AttacksAlone {
        subject: TriggerSubject,
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
    /// count resets every turn (see `TurnState::cards_drawn_this_turn`).
    YouDrawNthCardThisTurn {
        n: u32,
    },
    /// The inner condition only fires while `active_if` holds for the
    /// ability's controller. Used for conditionally-granted triggered
    /// abilities ("has firebending 2 as long as there's a Lesson card in
    /// your graveyard") and the fire-time half of intervening-if clauses
    /// (CR 603.4; re-check at resolution with an effect branch).
    ActiveIf {
        active_if: StaticCondition,
        condition: Box<TriggerCondition>,
    },
}

/// A game-state condition for conditional statics and intervening-if
/// checks, evaluated for a specific controller ("you").
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum StaticCondition {
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
    /// Copy the source's copiable characteristics, removing legendary.
    CreateNonlegendaryCopy,
    ModifyControlled {
        predicate: CardPredicate,
        power: i32,
        toughness: i32,
    },
    /// Permanents the resolving player controls matching `predicate` have
    /// base power and toughness set until end of turn (CR 613.4b); the
    /// affected set is fixed as the effect resolves (CR 611.2c).
    SetBasePtControlled {
        predicate: CardPredicate,
        power: EffectValue,
        toughness: EffectValue,
    },
    /// Permanents the resolving player controls matching `predicate` gain
    /// `subtypes` until end of turn (Energybending).
    AddSubtypesControlled {
        predicate: CardPredicate,
        subtypes: Vec<String>,
    },
    OptionalPayment {
        cost: PaymentCost,
        if_paid: Vec<Effect>,
        if_declined: Vec<Effect>,
    },
    Sacrifice {
        predicate: CardPredicate,
    },
    DealDamageToController {
        amount: i32,
    },
    DrainLife {
        amount: i32,
    },
    CantBeBlockedTarget {
        target: TargetSpec,
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
    DrainLifeValue {
        amount: EffectValue,
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
    /// Each creature you control gains firebending until end of turn.
    GrantFirebendingUntilEot {
        amount: u8,
    },
    /// Target creature gains firebending until end of turn.
    GrantFirebendingToTarget {
        amount: u8,
        target: TargetSpec,
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
    /// You may cast a matching spell from your hand without paying its mana cost.
    CastFromHandFree {
        predicate: CardPredicate,
    },
    /// You have no maximum hand size for the rest of the game.
    NoMaximumHandSize,
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
    BuffTargetValue {
        power: EffectValue,
        toughness: i32,
        target: TargetSpec,
    },
    ReturnToHandDrawIfControlled {
        target: TargetSpec,
    },
    /// Modify the exact permanent that caused the trigger, without targeting.
    ModifyTriggeringPermanent {
        counters: i32,
        keywords: Keywords,
    },
    /// The permanent whose event fired the trigger gets +X/+Y until end of
    /// turn, with X and Y computed on resolution.
    BuffTriggeringPermanentValue {
        power: EffectValue,
        toughness: EffectValue,
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
    AddSubtypeToTarget {
        target: TargetSpec,
        subtype: String,
    },
    ExileIfDiesThisTurn {
        target: TargetSpec,
    },
    IfCondition {
        condition: StaticCondition,
        effects: Vec<Effect>,
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
    ModifyUntilEot {
        power_delta: i32,
        toughness_delta: i32,
    },
    /// Resolving player draws `count` cards. No target.
    DrawCards {
        count: usize,
    },
    /// Resolving player draws a number of cards read at resolution.
    DrawCardsValue {
        count: EffectValue,
    },
    /// Resolving player loses `amount` life (not damage, CR 119.3).
    LoseLife {
        amount: i32,
    },
    /// The source deals `amount` damage to each opponent of the resolving
    /// player. No target.
    DealDamageToEachOpponent {
        amount: i32,
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
    /// Put `count` counters of `kind` on the source permanent.
    PutCountersOnSource {
        count: i32,
        kind: CounterKind,
    },
    /// Put `count` counters of `kind` on the target.
    PutCounters {
        count: i32,
        kind: CounterKind,
        target: TargetSpec,
    },
    /// Tap the source permanent.
    TapSource,
    /// Untap the source permanent.
    UntapSource,
    /// The source permanent can't be blocked this turn.
    CantBeBlockedThisTurnSource,
    /// The source permanent gains `keywords` until end of turn.
    GrantKeywordsToSource {
        keywords: Keywords,
    },
    /// The source permanent becomes an artifact creature (CR 613.1d).
    BecomeCreatureSource {
        duration: BecomesCreature,
    },
    /// The source permanent has base power and/or toughness `power`/`toughness`
    /// until end of turn (CR 613.4b); an absent value leaves that base alone.
    SetBasePtUntilEot {
        power: Option<i32>,
        toughness: Option<i32>,
    },
    /// The target permanent becomes an artifact creature (CR 613.1d).
    BecomeCreatureTarget {
        target: TargetSpec,
        duration: BecomesCreature,
    },
    /// The target permanent has base power and/or toughness until end of turn.
    SetBasePtTarget {
        target: TargetSpec,
        power: Option<i32>,
        toughness: Option<i32>,
    },
    /// "Return this card from your graveyard to your hand": only the exact
    /// graveyard object the ability came from (CR 400.7).
    ReturnSourceToHand,
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
        cost: ManaCost,
    },
    /// Branch on whether the resolving spell was kicked (CR 702.33).
    IfKicked {
        then: Vec<Effect>,
        otherwise: Vec<Effect>,
    },
    /// Branch on "if there are `count` or more cards matching [predicate]
    /// in your graveyard", evaluated at resolution.
    IfGraveyardAtLeast {
        count: usize,
        predicate: CardPredicate,
        then: Vec<Effect>,
        otherwise: Vec<Effect>,
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
    /// "Until end of turn, you don't lose unspent red mana as steps and
    /// phases end" (CR 106.4).
    KeepUnspentRedMana,
    /// The first chosen creature fights the last (CR 701.14); records the
    /// excess damage dealt to the last.
    FightLastTarget,
    /// Permanents the resolving player controls matching `predicate` gain
    /// `keywords` until end of turn; the set is fixed at resolution.
    GrantKeywordsControlled {
        predicate: CardPredicate,
        keywords: Keywords,
    },
    /// Search your library for a card matching `predicate`, reveal it, put
    /// it into your hand, then shuffle (CR 701.23).
    SearchLibraryToHand {
        predicate: CardPredicate,
    },
    /// The target gets +power/+toughness until end of turn.
    BuffTarget {
        power: i32,
        toughness: i32,
        target: TargetSpec,
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
    /// The target gains the set keywords until end of turn.
    GrantKeywordsToTarget {
        keywords: Keywords,
        target: TargetSpec,
    },
    /// Run `then` only if the frame's primary target currently matches
    /// [predicate] ("if that creature is an Ally, ...").
    IfTargetMatches {
        predicate: CardPredicate,
        then: Vec<Effect>,
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

/// Effects that run when any card milled by the same instruction matches.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct MilledBonus {
    pub predicate: CardPredicate,
    pub effects: Vec<Effect>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum TargetSpec {
    PermanentYouControl {
        predicate: CardPredicate,
    },
    GraveyardCardYouOwn {
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
        Self::DrawCards { count: 0 }
    }
}

impl Effect {
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

    pub fn target_spec(&self) -> Option<&TargetSpec> {
        match self {
            Effect::Sacrifice { .. }
            | Effect::DealDamageToController { .. }
            | Effect::DealDamageToEachOpponent { .. }
            | Effect::LoseLife { .. }
            | Effect::DrawCardsValue { .. }
            | Effect::DrainLife { .. } => None,
            Effect::TargetClause { requirement, .. } => Some(&requirement.spec),
            Effect::MillCards { target, .. } => target.as_ref(),
            Effect::ExileLinked { target, .. } => Some(target),
            Effect::SelectTargetRequirement { .. }
            | Effect::MillAndSelect { .. }
            | Effect::DrainLifeValue { .. } => None,
            Effect::CantBeBlockedTarget { target }
            | Effect::ReturnGraveyardTarget { target, .. }
            | Effect::ExileWithControllerToken { target, .. }
            | Effect::DiscardTargetPlayer { target, .. }
            | Effect::Destroy { target }
            | Effect::Exile { target }
            | Effect::Airbend { target }
            | Effect::BuffTargetValue { target, .. }
            | Effect::ReturnToHandDrawIfControlled { target } => Some(target),
            Effect::SearchBasicLand { .. }
            | Effect::OptionalDiscardDraw { .. }
            | Effect::ModifyTriggeringPermanent { .. }
            | Effect::BuffTriggeringPermanentValue { .. } => None,
            Effect::DealDamageValue { target, .. }
            | Effect::PutCountersValue { target, .. }
            | Effect::ExileIfDiesThisTurn { target }
            | Effect::AddSubtypeToTarget { target, .. } => Some(target),
            Effect::IfCondition {
                effects, otherwise, ..
            } => effects
                .iter()
                .chain(otherwise)
                .find_map(Effect::target_spec),
            Effect::Discard => None,
            Effect::ReturnToHand { target } => Some(target),
            Effect::DealDamage { target, .. } => Some(target),
            Effect::CounterSpell { target } => Some(target),
            Effect::PutCounters { target, .. } => Some(target),
            Effect::Earthbend { target, .. }
            | Effect::BecomeCreatureTarget { target, .. }
            | Effect::SetBasePtTarget { target, .. } => Some(target),
            Effect::ExileUntilSourceLeaves { target } => Some(target),
            Effect::BuffTarget { target, .. } => Some(target),
            Effect::UntapTarget { target }
            | Effect::TapTarget { target }
            | Effect::AttachSourceTo { target }
            | Effect::RemoveCounters { target, .. }
            | Effect::OwnerPutsIntoLibrary { target }
            | Effect::RevealHandDiscardChosen { target, .. } => Some(target),
            Effect::GrantKeywordsToTarget { target, .. }
            | Effect::GrantFirebendingToTarget { target, .. } => Some(target),
            Effect::OnNthResolutionThisTurn { effect, .. } => effect.target_spec(),
            Effect::OptionalPayment {
                if_paid: then,
                if_declined: otherwise,
                ..
            }
            | Effect::IfKicked { then, otherwise }
            | Effect::IfGraveyardAtLeast {
                then, otherwise, ..
            } => then
                .iter()
                .chain(otherwise.iter())
                .find_map(|effect| effect.target_spec()),
            Effect::IfTargetMatches { then, .. } | Effect::SacrificeSource { then } => {
                then.iter().find_map(|effect| effect.target_spec())
            }
            Effect::CreateNonlegendaryCopy
            | Effect::AirbendAll { .. }
            | Effect::OpponentsCastOnlyFromHand
            | Effect::ExileSourceSpell
            | Effect::ExileTopCardPlayThisTurn
            | Effect::GrantFirebendingUntilEot { .. }
            | Effect::ExileUntilNonlandCastFree { .. }
            | Effect::CastFromHandFree { .. }
            | Effect::NoMaximumHandSize
            | Effect::ShuffleGraveyardIntoLibrary
            | Effect::PlaceTargetInLibrary { .. }
            | Effect::SearchLandsOpponentSplits { .. }
            | Effect::ModifyControlled { .. }
            | Effect::SetBasePtControlled { .. }
            | Effect::AddSubtypesControlled { .. }
            | Effect::ModifyUntilEot { .. }
            | Effect::DrawCards { .. }
            | Effect::MassDamage { .. }
            | Effect::DestroyCreaturesLosingAbilities { .. }
            | Effect::DamageOpponentCreatures { .. }
            | Effect::CreateTokensValue { .. }
            | Effect::CreateToken { .. }
            | Effect::PutCountersOnSource { .. }
            | Effect::TapSource
            | Effect::UntapSource
            | Effect::CantBeBlockedThisTurnSource
            | Effect::GrantKeywordsToSource { .. }
            | Effect::SetBasePtUntilEot { .. }
            | Effect::BecomeCreatureSource { .. }
            | Effect::ExileAttached
            | Effect::TapAttached
            | Effect::RemoveAllCountersFromAttached
            | Effect::DistributeCountersToTappedOpponentCreatures { .. }
            | Effect::ReturnGrantorToHand
            | Effect::ShuffleAttachedIntoLibrary
            | Effect::ReturnSourceToHand
            | Effect::GainLife { .. }
            | Effect::GainLifeValue { .. }
            | Effect::AddManaValue { .. }
            | Effect::KeepUnspentRedMana
            | Effect::FightLastTarget
            | Effect::GrantKeywordsControlled { .. }
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
