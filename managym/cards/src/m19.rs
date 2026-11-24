//! Core Set 2019.
use managym_state::{
    ability::{Ability, Effect, ObjectSelector, TargetSpec, TriggerSubject},
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics,
    },
    effect::Modification,
    mana::ManaCost,
    predicate::CardPredicate,
};
pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        meteor_golem(),
        gigantosaurus(),
        lathliss_dragon_queen(),
        heroic_reinforcements(),
    ]
}

fn heroic_reinforcements() -> CardDefinition {
    let text_box = "Create two 1/1 white Soldier creature tokens. Until end of turn, creatures you control get +1/+1 and gain haste. (They can attack and {T} this turn.)";
    CardDefinition {
        key: "m19.heroic_reinforcements",
        scryfall_id: "33c35bf8-ae43-41aa-aae9-4d7513f9058c",
        printed: PrintedCharacteristics {
            name: "Heroic Reinforcements".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}{W}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![
                Effect::create_tokens("token.soldier_1_1_w", 2),
                Effect::until_end_of_turn(
                    ObjectSelector::each_you_control(CardPredicate::creature()),
                    Modification::pump(1, 1),
                ),
                Effect::until_end_of_turn(
                    ObjectSelector::each_you_control(CardPredicate::creature()),
                    Modification::AddKeywords(Keywords::of(&[Keyword::Haste])),
                ),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn lathliss_dragon_queen() -> CardDefinition {
    let text_box = "Flying\nWhenever another nontoken Dragon you control enters, create a 5/5 red Dragon creature token with flying.\n{1}{R}: Dragons you control get +1/+0 until end of turn.";
    CardDefinition {
        key: "m19.lathliss_dragon_queen",
        scryfall_id: "54a4c37d-5eeb-42c9-9688-c2ed0d5044cd",
        printed: PrintedCharacteristics {
            name: "Lathliss, Dragon Queen".into(),
            mana_cost: Some(ManaCost::parse("{4}{R}{R}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Dragon".into()],
            power: Some(6),
            toughness: Some(6),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![Ability::when_enters(
                TriggerSubject::another_you_control(CardPredicate {
                    token: Some(false),
                    ..CardPredicate::subtype("Dragon")
                }),
                vec![Effect::create_tokens("token.dragon_5_5_r_flying", 1)],
            )],
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{1}{R}"),
                effects: vec![Effect::until_end_of_turn(
                    ObjectSelector::each_you_control(CardPredicate::subtype("Dragon")),
                    Modification::pump(1, 0),
                )],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn gigantosaurus() -> CardDefinition {
    CardDefinition {
        key: "m19.gigantosaurus",
        scryfall_id: "c1db84d8-d426-4c0d-b44e-5be7b0f5f5bf",
        printed: PrintedCharacteristics {
            name: "Gigantosaurus".into(),
            mana_cost: Some(ManaCost::parse("{G}{G}{G}{G}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dinosaur".into()],
            power: Some(10),
            toughness: Some(10),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn meteor_golem() -> CardDefinition {
    CardDefinition {
        key: "m19.meteor_golem",
        scryfall_id: "1bdb0b15-d651-4730-8be9-d0e01145311b",
        printed: PrintedCharacteristics {
            name: "Meteor Golem".into(),
            subtypes: vec!["Golem".into()],
            types: CardTypes::new([CardType::Artifact, CardType::Creature]),
            mana_cost: Some(ManaCost::parse("{7}")),
            power: Some(3),
            toughness: Some(3),
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::destroy(ObjectSelector::Target(
                    TargetSpec::PermanentOpponentControls {
                        predicate: CardPredicate {
                            not_card_types: vec![CardType::Land],
                            ..Default::default()
                        },
                    },
                ))],
            )],
            text_box:
                "When this creature enters, destroy target nonland permanent an opponent controls."
                    .into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
