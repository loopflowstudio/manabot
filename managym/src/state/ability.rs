use super::predicate::CardPredicate;

#[derive(Clone, Debug, PartialEq, Eq)]
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
        self.effects().iter().find_map(|effect| effect.target_spec())
    }
}

/// The event that causes a triggered ability to trigger (CR 603.1).
#[derive(Clone, Debug, PartialEq, Eq)]
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
    /// "At the beginning of your upkeep."
    BeginningOfYourUpkeep,
    /// "Whenever you draw your Nth card each turn." The per-player draw
    /// count resets every turn (see `TurnState::cards_drawn_this_turn`).
    YouDrawNthCardThisTurn { n: u32 },
}

/// Which game objects an event-based trigger condition watches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TriggerSubject {
    /// The permanent this ability is on.
    This,
    /// "another [predicate] you control"
    AnotherYouControl(CardPredicate),
    /// "this or another [predicate] you control"
    AnyYouControl(CardPredicate),
}

#[derive(Clone, Debug, PartialEq, Eq)]
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
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TargetSpec {
    Creature,
    CreatureOrPlayer,
    Spell,
}

impl Effect {
    pub fn target_spec(&self) -> Option<&TargetSpec> {
        match self {
            Effect::ReturnToHand { target } => Some(target),
            Effect::DealDamage { target, .. } => Some(target),
            Effect::CounterSpell { target } => Some(target),
            Effect::PutCounters { target, .. } => Some(target),
            Effect::OnNthResolutionThisTurn { effect, .. } => effect.target_spec(),
            Effect::ModifyUntilEot { .. }
            | Effect::DrawCards { .. }
            | Effect::MassDamage { .. }
            | Effect::CreateToken { .. }
            | Effect::PutCountersOnSource { .. }
            | Effect::TapSource
            | Effect::UntapSource
            | Effect::CantBeBlockedThisTurnSource
            | Effect::GainLife { .. } => None,
        }
    }
}
