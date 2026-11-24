//! Time Spiral.
use managym_state::{
    ability::{Effect, EffectValue},
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics, StaticAbility},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![think_twice()]
}

fn think_twice() -> CardDefinition {
    let rules_text = "Draw a card.\nFlashback {2}{U} (You may cast this card from your graveyard for its flashback cost. Then exile it.)";
    CardDefinition {
        key: "tsp.think_twice",
        scryfall_id: "352d99db-de6d-4405-90ec-b144abbaa5a4",
        printed: PrintedCharacteristics {
            name: "Think Twice".into(),
            mana_cost: Some(ManaCost::parse("{1}{U}")),
            types: CardTypes::new([CardType::Instant]),
            statics: vec![StaticAbility::Flashback(ManaCost::parse("{2}{U}"))],
            spell_effects: vec![Effect::DrawCards {
                count: EffectValue::Fixed(1),
            }],
            text_box: rules_text.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
