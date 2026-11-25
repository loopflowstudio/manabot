//! Magic 2013.
use managym_state::{
    ability::{
        Effect, EffectValue, ObjectSelector, PlayerSelector, TargetRequirement, TargetSpec,
    },
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PowerCda, PrintedCharacteristics, StaticAbility,
    },
    effect::Modification,
    mana::ManaCost,
    permanent::CounterKind,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        ajani_caller_of_the_pride(),
        crusader_of_odric(),
        krenko_mob_boss(),
        omniscience(),
    ]
}

fn ajani_caller_of_the_pride() -> CardDefinition {
    let text_box = "+1: Put a +1/+1 counter on up to one target creature.\n−3: Target creature gains flying and double strike until end of turn.\n−8: Create X 2/2 white Cat creature tokens, where X is your life total.";
    let loyalty = ActivatedAbilityDefinition::loyalty;
    CardDefinition {
        key: "m13.ajani_caller_of_the_pride",
        scryfall_id: "5e7f410a-7934-48ae-a90b-ffd096aed43d",
        printed: PrintedCharacteristics {
            name: "Ajani, Caller of the Pride".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}{W}")),
            types: CardTypes::new([CardType::Planeswalker]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Ajani".into()],
            loyalty: Some(4),
            activated_abilities: vec![
                loyalty(
                    1,
                    vec![Effect::TargetClause {
                        requirement: TargetRequirement::up_to(1, TargetSpec::Creature),
                        effects: vec![Effect::put_counters(
                            ObjectSelector::Target(TargetSpec::Creature),
                            CounterKind::PlusOnePlusOne,
                            1,
                        )],
                    }],
                ),
                loyalty(
                    -3,
                    vec![Effect::until_end_of_turn(
                        ObjectSelector::Target(TargetSpec::Creature),
                        Modification::AddKeywords(Keywords::of(&[
                            Keyword::Flying,
                            Keyword::DoubleStrike,
                        ])),
                    )],
                ),
                loyalty(
                    -8,
                    vec![Effect::create_tokens(
                        "token.cat_2_2_w",
                        EffectValue::LifeTotal(PlayerSelector::You),
                    )],
                ),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn crusader_of_odric() -> CardDefinition {
    let text_box = "Crusader of Odric's power and toughness are each equal to the number of creatures you control.";
    CardDefinition {
        key: "m13.crusader_of_odric",
        scryfall_id: "295096bb-1857-4224-bc7b-307b38cfd338",
        printed: PrintedCharacteristics {
            name: "Crusader of Odric".into(),
            mana_cost: Some(ManaCost::parse("{2}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Soldier".into()],
            statics: vec![StaticAbility::CharacteristicPt(
                PowerCda::CreaturesYouControl,
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn krenko_mob_boss() -> CardDefinition {
    let text_box = "{T}: Create X 1/1 red Goblin creature tokens, where X is the number of Goblins you control.";
    CardDefinition {
        key: "m13.krenko_mob_boss",
        scryfall_id: "aa078518-0ce2-4c6f-9061-aa7e22ed7493",
        printed: PrintedCharacteristics {
            name: "Krenko, Mob Boss".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}{R}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Goblin".into(), "Warrior".into()],
            power: Some(3),
            toughness: Some(3),
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::create_tokens(
                    "token.goblin_1_1_r",
                    ObjectSelector::each_you_control(CardPredicate::subtype("Goblin")).count(),
                )],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn omniscience() -> CardDefinition {
    let rules_text = "You may cast spells from your hand without paying their mana costs.";
    CardDefinition {
        key: "m13.omniscience",
        scryfall_id: "1088f33e-cb5f-4248-ae8e-280c4e41f291",
        printed: PrintedCharacteristics {
            name: "Omniscience".into(),
            mana_cost: Some(ManaCost::parse("{7}{U}{U}{U}")),
            types: CardTypes::new([CardType::Enchantment]),
            statics: vec![StaticAbility::CastFromHandWithoutManaCost],
            text_box: rules_text.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
