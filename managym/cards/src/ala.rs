//! Shards of Alara.
use managym_state::{
    ability::Effect,
    card::{
        CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics,
        StaticAbility, StaticScope,
    },
    effect::Modification,
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![death_baron(), dragon_fodder()]
}

fn dragon_fodder() -> CardDefinition {
    CardDefinition {
        key: "ala.dragon_fodder",
        scryfall_id: "9eab4120-e7d8-4132-a304-30b88e3175e2",
        printed: PrintedCharacteristics {
            name: "Dragon Fodder".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![Effect::create_tokens("token.goblin_1_1_r", 2)],
            text_box: "Create two 1/1 red Goblin creature tokens.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn death_baron() -> CardDefinition {
    // A Skeleton Zombie receives this bonus once. Baron itself qualifies only as a Skeleton.
    let grant = |scope| {
        StaticAbility::continuous(
            scope,
            None,
            vec![
                Modification::pump(1, 1),
                Modification::AddKeywords(Keywords::of(&[Keyword::Deathtouch])),
            ],
        )
    };
    let text_box = "Skeletons you control and other Zombies you control get +1/+1 and have deathtouch. (Any amount of damage they deal to a creature is enough to destroy it.)";
    CardDefinition {
        key: "ala.death_baron",
        scryfall_id: "4d59b5e5-fc16-4f1a-9f17-f42908473531",
        printed: PrintedCharacteristics {
            name: "Death Baron".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Zombie".into(), "Wizard".into()],
            power: Some(2),
            toughness: Some(2),
            statics: vec![
                grant(StaticScope::YouControl(CardPredicate::subtype("Skeleton"))),
                grant(StaticScope::OtherYouControl(CardPredicate {
                    not_subtypes: vec!["Skeleton".into()],
                    ..CardPredicate::subtype("Zombie")
                })),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
