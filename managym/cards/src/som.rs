//! Scars of Mirrodin.
use managym_state::{
    ability::{Effect, EffectValue, PlayerSelector},
    card::{CardDefinition, CardType, CardTypes, PrintedCharacteristics},
    mana::ManaCost,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![exsanguinate()]
}

fn exsanguinate() -> CardDefinition {
    let text_box = "Each opponent loses X life. You gain life equal to the life lost this way.";
    CardDefinition {
        key: "som.exsanguinate",
        scryfall_id: "0878b541-a730-49db-b062-5a01656e269d",
        printed: PrintedCharacteristics {
            name: "Exsanguinate".into(),
            mana_cost: Some(ManaCost::parse("{X}{B}{B}")),
            types: CardTypes::new([CardType::Sorcery]),
            text_box: text_box.into(),
            spell_effects: vec![
                Effect::LoseLife {
                    player: PlayerSelector::EachOpponent,
                    amount: EffectValue::X,
                },
                Effect::gain_life(EffectValue::LifeLostThisWay),
            ],
            ..Default::default()
        },
        ..Default::default()
    }
}
