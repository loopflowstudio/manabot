//! Dominaria.
use managym_state::{
    ability::{Ability, Effect, ObjectSelector, TargetRequirement, TargetSpec, TriggerSubject},
    card::{
        CardDefinition, CardType, CardTypes, Keyword, Keywords, LayerValue, PrintedCharacteristics,
        StaticAbility, StaticContinuous, StaticScope,
    },
    effect::Modification,
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![lyra_dawnbringer(), tatyova_benthic_druid(), run_amok()]
}

fn run_amok() -> CardDefinition {
    let text_box = "Target attacking creature gets +3/+3 and gains trample until end of turn.";
    let attacking = || TargetSpec::Permanent {
        predicate: CardPredicate {
            attacking: true,
            ..CardPredicate::creature()
        },
    };
    CardDefinition {
        key: "dom.run_amok",
        scryfall_id: "d3f2aa2c-deec-4927-a2d6-f5dce5967e26",
        printed: PrintedCharacteristics {
            name: "Run Amok".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![
                Effect::until_end_of_turn(
                    ObjectSelector::Target(attacking()),
                    Modification::pump(3, 3),
                ),
                Effect::until_end_of_turn(
                    ObjectSelector::Target(attacking()),
                    Modification::AddKeywords(Keywords::of(&[Keyword::Trample])),
                ),
            ],
            targeting: vec![TargetRequirement::one(attacking())],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn tatyova_benthic_druid() -> CardDefinition {
    CardDefinition {
        key: "dom.tatyova_benthic_druid",
        scryfall_id: "93657aaa-7a0f-49ad-b026-6f79b3bd6768",
        printed: PrintedCharacteristics {
            name: "Tatyova, Benthic Druid".into(),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Merfolk".into(), "Druid".into()],
            types: CardTypes::new([CardType::Creature]),
            mana_cost: Some(ManaCost::parse("{3}{G}{U}")),
            power: Some(3),
            toughness: Some(3),
            abilities: vec![Ability::when_enters(
                TriggerSubject::you_control(CardPredicate {
                    card_type: Some(CardType::Land),
                    ..Default::default()
                }),
                vec![Effect::gain_life(1), Effect::DrawCards { count: 1.into() }],
            )],
            text_box:
                "Landfall — Whenever a land you control enters, you gain 1 life and draw a card."
                    .into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn lyra_dawnbringer() -> CardDefinition {
    CardDefinition {key:"dom.lyra_dawnbringer",scryfall_id:"93be6799-7b9d-44d4-84dc-2961692b5a85",printed:PrintedCharacteristics {statics:vec![StaticAbility::Continuous(StaticContinuous {scope:StaticScope::OtherYouControl(CardPredicate::subtype("Angel")),condition:None,characteristic_defining:false,modifications:vec![Modification::ModifyPt {power:LayerValue::Fixed(1),toughness:LayerValue::Fixed(1)},Modification::AddKeywords(Keywords::of(&[Keyword::Lifelink]))]})],name:"Lyra Dawnbringer".into(),types:CardTypes::new([CardType::Creature]),supertypes:vec!["Legendary".into()],subtypes:vec!["Angel".into()],mana_cost:Some(ManaCost::parse("{3}{W}{W}")),power:Some(5),toughness:Some(5),keywords:Keywords::of(&[Keyword::Flying,Keyword::Lifelink,Keyword::FirstStrike]),text_box:"Flying\nFirst strike (This creature deals combat damage before creatures without first strike.)\nLifelink (Damage dealt by this creature also causes you to gain that much life.)\nOther Angels you control get +1/+1 and have lifelink.".into(),..Default::default()},..Default::default()}
}
