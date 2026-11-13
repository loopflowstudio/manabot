//! Born of the Gods.
use managym_state::{
    ability::{
        DelayedEvent, DelayedExpiry, DelayedFrequency, Effect, EffectValue, ObjectSelector,
        PlayerSelector, TargetRequirement, TargetSpec,
    },
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![searing_blood()]
}

fn searing_blood() -> CardDefinition {
    let text_box = "Searing Blood deals 2 damage to target creature. When that creature dies this turn, Searing Blood deals 3 damage to the creature's controller.";
    CardDefinition {
        key: "bng.searing_blood",
        scryfall_id: "bb43fd07-d281-447d-88bf-c53498c2cf20",
        printed: PrintedCharacteristics {
            name: "Searing Blood".into(),
            mana_cost: Some(ManaCost::parse("{R}{R}")),
            types: CardTypes::new([CardType::Instant]),
            targeting: vec![TargetRequirement::one(TargetSpec::Creature)],
            // The watcher is created before the damage, so a creature this
            // damage kills is still watched as it dies.
            spell_effects: vec![
                Effect::CreateDelayedTrigger {
                    remember: Some(ObjectSelector::Target(TargetSpec::Creature)),
                    event: DelayedEvent::Dies,
                    frequency: DelayedFrequency::Once,
                    expiry: DelayedExpiry::ThisTurn,
                    effects: vec![Effect::DealDamageToPlayer {
                        player: PlayerSelector::TargetController,
                        amount: EffectValue::Fixed(3),
                    }],
                },
                Effect::deal_damage(2, ObjectSelector::Target(TargetSpec::Creature)),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
