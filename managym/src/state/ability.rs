use super::card::Keywords;
use super::mana::{Mana, ManaCost};
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

    /// The target spec of the ability's single targeted effect, if any.
    /// At most one effect per ability may carry a target.
    pub fn target_spec(&self) -> Option<&TargetSpec> {
        self.effects()
            .iter()
            .find_map(|effect| effect.target_spec())
    }

    /// "Up to one target" — the controller may decline choosing a target.
    pub fn target_optional(&self) -> bool {
        self.effects().iter().any(|effect| effect.target_optional())
    }
}

/// The event that causes a triggered ability to trigger (CR 603.1).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum TriggerCondition {
    /// "When [subject] enters the battlefield."
    EntersTheBattlefield { subject: TriggerSubject },
    /// "When [subject] dies." — battlefield to graveyard (CR 700.4).
    Dies { subject: TriggerSubject },
    /// "Whenever [subject] attacks." Fires after the declare-attackers
    /// turn-based action (CR 508.1); a subject of `AnotherYouControl` /
    /// `AnyYouControl` fires once per combat ("whenever one or more ...
    /// attack"), not once per attacker.
    Attacks { subject: TriggerSubject },
    /// "Whenever [subject] becomes tapped."
    BecomesTapped { subject: TriggerSubject },
    /// "Whenever [subject] is tapped for mana."
    TappedForMana { subject: TriggerSubject },
    /// "Whenever [subject] becomes the target of a spell an opponent
    /// controls." (Ward, CR 702.21.) The triggering spell is threaded to the
    /// ability's effects as the trigger context target.
    BecomesTargeted { subject: TriggerSubject },
    /// "At the beginning of your upkeep."
    BeginningOfYourUpkeep,
    /// "Whenever you draw your Nth card each turn." The per-player draw
    /// count resets every turn (see `TurnState::cards_drawn_this_turn`).
    YouDrawNthCardThisTurn { n: u32 },
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
    /// "as long as there are `count` or more [predicate] cards in your
    /// graveyard".
    GraveyardAtLeast {
        count: usize,
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

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Effect {
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
    /// Create `count` tokens from the registered token definition
    /// `token_name`, under the resolving player's control.
    CreateToken {
        token_name: String,
        count: usize,
        tapped_and_attacking: bool,
    },
    /// Put `count` +1/+1 counters on the source permanent.
    PutCountersOnSource {
        count: i32,
    },
    /// Put `count` +1/+1 counters on the target.
    PutCounters {
        count: i32,
        target: TargetSpec,
    },
    /// Tap the source permanent.
    TapSource,
    /// Untap the source permanent.
    UntapSource,
    /// The source permanent can't be blocked this turn.
    CantBeBlockedThisTurnSource,
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
    /// Scry `count` (CR 701.26): the controller looks at the top `count`
    /// cards and decides, one at a time from the top, whether each goes to
    /// the bottom of the library or stays on top. Kept cards retain their
    /// relative order (no reordering — engine simplification).
    Scry {
        count: usize,
    },
    /// "Look at the top `look` cards of your library. Put up to
    /// `max_select` cards matching [predicate] from among them into your
    /// hand and the rest on the bottom of your library in a random order."
    /// `min_select` > 0 makes the selection mandatory when possible.
    LookAndSelect {
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
    /// "Choose one —" modal effect. Mode effects must be targetless (engine
    /// limitation, see Stage 2 notes).
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
    pub spec: TargetSpec,
    pub min: usize,
    pub max: usize,
}

impl TargetRequirement {
    pub fn one(spec: TargetSpec) -> Self {
        Self {
            spec,
            min: 1,
            max: 1,
        }
    }

    pub fn up_to(max: usize, spec: TargetSpec) -> Self {
        Self { spec, min: 0, max }
    }
}

impl Effect {
    pub fn target_spec(&self) -> Option<&TargetSpec> {
        match self {
            Effect::ReturnToHand { target } => Some(target),
            Effect::DealDamage { target, .. } => Some(target),
            Effect::CounterSpell { target } => Some(target),
            Effect::PutCounters { target, .. } => Some(target),
            Effect::Earthbend { target, .. } => Some(target),
            Effect::ExileUntilSourceLeaves { target } => Some(target),
            Effect::BuffTarget { target, .. } => Some(target),
            Effect::UntapTarget { target } => Some(target),
            Effect::GrantKeywordsToTarget { target, .. } => Some(target),
            Effect::OnNthResolutionThisTurn { effect, .. } => effect.target_spec(),
            Effect::IfKicked { then, otherwise }
            | Effect::IfGraveyardAtLeast {
                then, otherwise, ..
            } => then
                .iter()
                .chain(otherwise.iter())
                .find_map(|effect| effect.target_spec()),
            Effect::IfTargetMatches { then, .. } => {
                then.iter().find_map(|effect| effect.target_spec())
            }
            Effect::ModifyUntilEot { .. }
            | Effect::DrawCards { .. }
            | Effect::MassDamage { .. }
            | Effect::CreateToken { .. }
            | Effect::PutCountersOnSource { .. }
            | Effect::TapSource
            | Effect::UntapSource
            | Effect::CantBeBlockedThisTurnSource
            | Effect::GainLife { .. }
            | Effect::Scry { .. }
            | Effect::LookAndSelect { .. }
            | Effect::PutTopCardsInHand { .. }
            | Effect::Learn
            | Effect::Modal { .. }
            | Effect::CounterUnlessPays { .. }
            | Effect::ReturnSourceToBattlefieldTapped
            | Effect::AddMana { .. }
            | Effect::ForEachTarget { .. }
            | Effect::PutCountersOnEachMatching { .. }
            | Effect::TargetCreaturesDealPowerDamageToLastTarget => None,
        }
    }

    /// Whether the effect's target is "up to one" (the trigger's controller
    /// may decline to choose a target).
    pub fn target_optional(&self) -> bool {
        matches!(self, Effect::ExileUntilSourceLeaves { .. })
    }
}
