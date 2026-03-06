#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ability {
    Triggered {
        condition: TriggerCondition,
        effect: Effect,
        intervening_if: Option<TriggerCondition>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TriggerCondition {
    EntersTheBattlefield { source: TriggerSource },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TriggerSource {
    This,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Effect {
    ReturnToHand { target: TargetSpec },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TargetSpec {
    Creature { controller: TargetController },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TargetController {
    Any,
}
