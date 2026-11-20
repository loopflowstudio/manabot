//! Born of the Gods.
use managym_state::{
    ability::{
        DelayedEvent, DelayedExpiry, DelayedFrequency, Effect, EffectValue, ObjectSelector,
        PlayerSelector, TargetRequirement, TargetSpec,
    },
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
};

use super::ths::temple;

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        searing_blood(),
        temple_of_enlightenment(),
        temple_of_malice(),
        temple_of_plenty(),
    ]
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

fn temple_of_enlightenment() -> CardDefinition {
    temple(
        "bng.temple_of_enlightenment",
        "c657a645-f454-4eaf-be0d-15c9989fa4ef",
        "Temple of Enlightenment",
        ["W", "U"],
    )
}

fn temple_of_malice() -> CardDefinition {
    temple(
        "bng.temple_of_malice",
        "52f50818-aede-4667-883a-e0339d86d870",
        "Temple of Malice",
        ["B", "R"],
    )
}

fn temple_of_plenty() -> CardDefinition {
    temple(
        "bng.temple_of_plenty",
        "b0830054-b140-49c3-90cb-24e2502757be",
        "Temple of Plenty",
        ["G", "W"],
    )
}
