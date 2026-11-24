use managym_state::{
    ability::{Ability, Effect, TriggerCondition, TriggerSubject},
    card::{CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics},
    mana::ManaCost,
    predicate::CardPredicate,
    zone::ZoneType,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![vile_entomber(), tireless_provisioner()]
}

fn tireless_provisioner() -> CardDefinition {
    let text_box = "Landfall — Whenever a land you control enters, create a Food token or a Treasure token. (Food is an artifact with \"{2}, {T}, Sacrifice this token: You gain 3 life.\" Treasure is an artifact with \"{T}, Sacrifice this token: Add one mana of any color.\")";

    CardDefinition {
        key: "mh2.tireless_provisioner",
        scryfall_id: "5c5be54d-660e-42ab-b5ea-5e1cf3bad0bc",
        printed: PrintedCharacteristics {
            name: "Tireless Provisioner".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Elf".into(), "Scout".into()],
            power: Some(3),
            toughness: Some(2),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::you_control(CardPredicate {
                        card_type: Some(CardType::Land),
                        ..Default::default()
                    }),
                },
                effects: vec![Effect::Modal {
                    modes: vec![
                        vec![Effect::create_tokens("token.food", 1)],
                        vec![Effect::create_tokens("token.treasure", 1)],
                    ],
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn vile_entomber() -> CardDefinition {
    let text_box = "Deathtouch (Any amount of damage this deals to a creature is enough to destroy it.)\nWhen this creature enters, search your library for a card, put that card into your graveyard, then shuffle.";
    CardDefinition {
        key: "mh2.vile_entomber",
        scryfall_id: "d890ae71-da2b-44fa-8cfa-9c3016c9f696",
        printed: PrintedCharacteristics {
            name: "Vile Entomber".into(),
            mana_cost: Some(ManaCost::parse("{2}{B}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Zombie".into(), "Warlock".into()],
            power: Some(2),
            toughness: Some(2),
            keywords: Keywords::of(&[Keyword::Deathtouch]),
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::search_for_card(ZoneType::Graveyard)],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
