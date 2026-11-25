//! Ravnica cards expressed in the shared proposed-event rules.
use managym_state::{
    card::{
        AdditionalCost, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics, ReplacementEffect,
    },
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![doubling_season(), guardian_of_vitu_ghazi()]
}

fn doubling_season() -> CardDefinition {
    let text_box = "If an effect would create one or more tokens under your control, it creates twice that many of those tokens instead.\nIf an effect would put one or more counters on a permanent you control, it puts twice that many of those counters on that permanent instead.";
    CardDefinition {
        key: "rav.doubling_season",
        scryfall_id: "c7e71299-98f6-494e-b187-8d22ce5f50af",
        printed: PrintedCharacteristics {
            name: "Doubling Season".into(),
            mana_cost: Some(ManaCost::parse("{4}{G}")),
            types: CardTypes::new([CardType::Enchantment]),
            replacement_effects: vec![
                ReplacementEffect::MultiplyTokens { multiplier: 2 },
                ReplacementEffect::ModifyCounters {
                    effects_only: true,
                    kind: None,
                    predicate: CardPredicate::default(),
                    multiplier: 2,
                    additional: 0,
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn guardian_of_vitu_ghazi() -> CardDefinition {
    let text_box = "Convoke (Your creatures can help cast this spell. Each creature you tap while casting this spell pays for {1} or one mana of that creature's color.)\nVigilance";
    CardDefinition {
        key: "rav.guardian_of_vitu_ghazi",
        scryfall_id: "1c293144-df82-481d-8d24-2bdc164a355f",
        printed: PrintedCharacteristics {
            name: "Guardian of Vitu-Ghazi".into(),
            mana_cost: Some(ManaCost::parse("{6}{G}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Elemental".into()],
            power: Some(4),
            toughness: Some(7),
            keywords: Keywords::of(&[Keyword::Vigilance]),
            additional_costs: vec![AdditionalCost::Convoke],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
