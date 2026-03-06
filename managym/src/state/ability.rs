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
    DealDamage { amount: i32, target: TargetSpec },
    CounterSpell { target: TargetSpec },
    ModifyUntilEot { power_delta: i32, toughness_delta: i32 },
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
            Effect::ModifyUntilEot { .. } => None,
        }
    }
}
