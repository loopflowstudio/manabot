use super::card::Keywords;
use super::mana::{Mana, ManaCost};
use super::permanent::{BecomesCreature, CounterKind};
use super::predicate::CardPredicate;

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
    /// "Whenever [subject] becomes tapped."
    BecomesTapped {
        subject: TriggerSubject,
    },
    /// "Whenever [subject] is tapped for mana."
    TappedForMana {
        subject: TriggerSubject,
    },
    /// "Whenever [subject] becomes the target of a spell an opponent
    /// controls." (Ward, CR 702.21.) The triggering spell is threaded to the
    /// ability's effects as the trigger context target.
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

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Effect {
    /// Copy the source's copiable characteristics, removing legendary.
    CreateNonlegendaryCopy,
    ModifyControlled {
        predicate: CardPredicate,
        power: i32,
        toughness: i32,
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
    },
    MillCards {
        count: usize,
    },
    MillAndSelect {
        count: usize,
        predicate: CardPredicate,
        if_declined: Vec<Effect>,
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
        #[serde(skip_serializing_if = "is_one")]
        count: usize,
        #[serde(skip_serializing_if = "is_true")]
        tapped: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        untap_if_lands: Option<usize>,
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        bottom_source: bool,
    },
    OptionalDiscardDraw {
        count: usize,
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
    /// Deal `amount` damage to each creature on the battlefield. No target.
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
    /// Earthbend with a count read as the ability resolves.
    EarthbendValue {
        target: TargetSpec,
        count: EffectValue,
    },
    /// Resolving player gains `amount` life.
    GainLife {
        amount: i32,
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
        look: usize,
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
    /// "Counter [the spell] unless its controller pays [cost]." The spell is
    /// the frame's primary target: either a chosen `TargetSpec::Spell`
    /// target (It'll Quench Ya!) or the trigger-context spell (ward).
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
        count: i32,
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
        count: i32,
        predicate: CardPredicate,
        other: bool,
    },
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
    /// "Another target": the source object can't be chosen.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub another: bool,
    pub spec: TargetSpec,
    pub min: usize,
    pub max: usize,
}

impl TargetRequirement {
    pub fn one(spec: TargetSpec) -> Self {
        Self {
            another: false,
            spec,
            min: 1,
            max: 1,
        }
    }

    pub fn up_to(max: usize, spec: TargetSpec) -> Self {
        Self {
            another: false,
            spec,
            min: 0,
            max,
        }
    }
}

fn is_one(value: &usize) -> bool {
    *value == 1
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
            | Effect::DrainLife { .. } => None,
            Effect::TargetClause { requirement, .. } => Some(&requirement.spec),
            Effect::SelectTargetRequirement { .. }
            | Effect::MillCards { .. }
            | Effect::MillAndSelect { .. }
            | Effect::DrainLifeValue { .. } => None,
            Effect::CantBeBlockedTarget { target }
            | Effect::ReturnGraveyardTarget { target }
            | Effect::ExileWithControllerToken { target, .. }
            | Effect::DiscardTargetPlayer { target, .. }
            | Effect::Destroy { target }
            | Effect::Exile { target }
            | Effect::BuffTargetValue { target, .. }
            | Effect::ReturnToHandDrawIfControlled { target } => Some(target),
            Effect::SearchBasicLand { .. }
            | Effect::OptionalDiscardDraw { .. }
            | Effect::ModifyTriggeringPermanent { .. } => None,
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
            | Effect::EarthbendValue { target, .. }
            | Effect::BecomeCreatureTarget { target, .. }
            | Effect::SetBasePtTarget { target, .. } => Some(target),
            Effect::ExileUntilSourceLeaves { target } => Some(target),
            Effect::BuffTarget { target, .. } => Some(target),
            Effect::UntapTarget { target }
            | Effect::TapTarget { target }
            | Effect::AttachSourceTo { target }
            | Effect::RemoveCounters { target, .. } => Some(target),
            Effect::GrantKeywordsToTarget { target, .. } => Some(target),
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
            Effect::IfTargetMatches { then, .. } => {
                then.iter().find_map(|effect| effect.target_spec())
            }
            Effect::CreateNonlegendaryCopy
            | Effect::ModifyControlled { .. }
            | Effect::ModifyUntilEot { .. }
            | Effect::DrawCards { .. }
            | Effect::MassDamage { .. }
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
            | Effect::ShuffleAttachedIntoLibrary
            | Effect::GainLife { .. }
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
                    another: false,
                    spec: spec.clone(),
                    min: usize::from(!optional),
                    max: 1,
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
