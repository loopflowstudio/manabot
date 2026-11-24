//! Dominaria.
use managym_state::{
    ability::{
        Ability, Condition, Effect, ObjectSelector, TargetRequirement, TargetSpec, TriggerSubject,
    },
    card::{
        CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics,
        StaticAbility, StaticScope,
    },
    effect::Modification,
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        adamant_will(),
        ghitu_lavarunner(),
        lyra_dawnbringer(),
        tatyova_benthic_druid(),
        run_amok(),
        muldrotha(),
    ]
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
            spell_effects: Effect::pump_and_grant_until_end_of_turn(
                ObjectSelector::Target(attacking()),
                3,
                3,
                Keyword::Trample,
            ),
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
    let text_box = "Flying\nFirst strike (This creature deals combat damage before creatures without first strike.)\nLifelink (Damage dealt by this creature also causes you to gain that much life.)\nOther Angels you control get +1/+1 and have lifelink.";
    CardDefinition {
        key: "dom.lyra_dawnbringer",
        scryfall_id: "93be6799-7b9d-44d4-84dc-2961692b5a85",
        printed: PrintedCharacteristics {
            statics: vec![StaticAbility::continuous(
                StaticScope::OtherYouControl(CardPredicate::subtype("Angel")),
                None,
                vec![
                    Modification::pump(1, 1),
                    Modification::AddKeywords(Keywords::of(&[Keyword::Lifelink])),
                ],
            )],
            name: "Lyra Dawnbringer".into(),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Angel".into()],
            mana_cost: Some(ManaCost::parse("{3}{W}{W}")),
            power: Some(5),
            toughness: Some(5),
            keywords: Keywords::of(&[Keyword::Flying, Keyword::Lifelink, Keyword::FirstStrike]),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn ghitu_lavarunner() -> CardDefinition {
    let text_box = "As long as there are two or more instant and/or sorcery cards in your graveyard, this creature gets +1/+0 and has haste. (It can attack and {T} as soon as it comes under your control.)";
    CardDefinition {
        key: "dom.ghitu_lavarunner",
        scryfall_id: "c448ba82-a502-459f-9ebc-fc9e85674e6c",
        printed: PrintedCharacteristics {
            name: "Ghitu Lavarunner".into(),
            mana_cost: Some(ManaCost::parse("{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Wizard".into()],
            power: Some(1),
            toughness: Some(2),
            statics: vec![StaticAbility::continuous(
                StaticScope::This,
                Some(Condition::GraveyardAtLeast {
                    count: 2,
                    predicate: CardPredicate::instant_or_sorcery(),
                }),
                vec![
                    Modification::pump(1, 0),
                    Modification::AddKeywords(Keywords::of(&[Keyword::Haste])),
                ],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn adamant_will() -> CardDefinition {
    let text_box = "Target creature gets +2/+2 and gains indestructible until end of turn. (Damage and effects that say \"destroy\" don't destroy it.)";
    CardDefinition {
        key: "dom.adamant_will",
        scryfall_id: "3dfb8817-ca3c-44ba-92f2-e9d6294cd25d",
        printed: PrintedCharacteristics {
            name: "Adamant Will".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: Effect::pump_and_grant_until_end_of_turn(
                ObjectSelector::Target(TargetSpec::Creature),
                2,
                2,
                Keyword::Indestructible,
            ),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn muldrotha() -> CardDefinition {
    let text_box = "During each of your turns, you may play a land and cast a permanent spell of each permanent type from your graveyard. (If a card has multiple permanent types, choose one as you play it.)";
    CardDefinition {
        key: "dom.muldrotha_the_gravetide",
        scryfall_id: "c654737d-34ac-42ff-ae27-3a3bbb930fc1",
        printed: PrintedCharacteristics {
            name: "Muldrotha, the Gravetide".into(),
            mana_cost: Some(ManaCost::parse("{3}{B}{G}{U}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Elemental".into(), "Avatar".into()],
            power: Some(6),
            toughness: Some(6),
            statics: vec![managym_state::card::StaticAbility::PlayGraveyardByPermanentType],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
