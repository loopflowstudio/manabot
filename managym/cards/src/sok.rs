//! Saviors of Kamigawa.
use managym_state::{
    ability::{
        CompareOp, Condition, Effect, EffectValue, PlayerSelector, TargetRequirement, TargetSpec,
    },
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![hidetsugus_second_rite()]
}

fn hidetsugus_second_rite() -> CardDefinition {
    let text_box =
        "If target player has exactly 10 life, Hidetsugu's Second Rite deals 10 damage to that player.";
    let player = || PlayerSelector::Target(Box::new(TargetSpec::Player));
    CardDefinition {
        key: "sok.hidetsugus_second_rite",
        scryfall_id: "2e48eb77-3bd7-444a-9262-799cc706c05a",
        printed: PrintedCharacteristics {
            name: "Hidetsugu's Second Rite".into(),
            mana_cost: Some(ManaCost::parse("{3}{R}")),
            types: CardTypes::new([CardType::Instant]),
            // The life total is asked as the spell resolves (CR 608.2c).
            spell_effects: vec![Effect::If {
                condition: Condition::compare(EffectValue::LifeTotal(player()), CompareOp::Eq, 10),
                then: vec![Effect::DealDamageToPlayer {
                    player: player(),
                    amount: EffectValue::Fixed(10),
                }],
                otherwise: Vec::new(),
            }],
            targeting: vec![TargetRequirement::one(TargetSpec::Player)],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
