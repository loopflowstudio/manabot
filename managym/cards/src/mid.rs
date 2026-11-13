//! Innistrad: Midnight Hunt.
use managym_state::{
    ability::{Effect, EffectValue},
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics, StaticAbility},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![consider(), join_the_dance()]
}

fn consider() -> CardDefinition {
    let text_box = "Surveil 1. (Look at the top card of your library. You may put it into your graveyard.)\nDraw a card.";
    CardDefinition {
        key: "mid.consider",
        scryfall_id: "a211d505-4d40-4914-a9da-220770d6ddbc",
        printed: PrintedCharacteristics {
            name: "Consider".into(),
            mana_cost: Some(ManaCost::parse("{U}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![
                Effect::Surveil { count: 1 },
                Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn join_the_dance() -> CardDefinition {
    let text_box = "Create two 1/1 white Human creature tokens.\nFlashback {3}{G}{W} (You may cast this card from your graveyard for its flashback cost. Then exile it.)";
    CardDefinition {
        key: "mid.join_the_dance",
        scryfall_id: "56b30a99-601b-40b9-b012-30fa4be5fd3c",
        printed: PrintedCharacteristics {
            name: "Join the Dance".into(),
            mana_cost: Some(ManaCost::parse("{G}{W}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![Effect::create_tokens("token.human", 2)],
            statics: vec![StaticAbility::Flashback(ManaCost::parse("{3}{G}{W}"))],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
