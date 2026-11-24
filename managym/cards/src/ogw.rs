//! Oath of the Gatewatch.
use managym_state::{
    ability::{
        Aggregate, CompareOp, Condition, Effect, EffectValue, ObjectProperty, ObjectSelector,
        PlayerSelector, TargetSpec,
    },
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics, SacrificeCost,
    },
    mana::ManaCost,
    payment::How,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![ayli_eternal_pilgrim()]
}

fn ayli_eternal_pilgrim() -> CardDefinition {
    let text_box = "Deathtouch\n{1}, Sacrifice another creature: You gain life equal to the sacrificed creature's toughness.\n{1}{W}{B}, Sacrifice another creature: Exile target nonland permanent. Activate only if you have at least 10 life more than your starting life total.";
    let another_creature = || {
        Some(SacrificeCost {
            predicate: CardPredicate::creature(),
            another: true,
            or_mana: None,
        })
    };
    let nonland = TargetSpec::Permanent {
        predicate: CardPredicate {
            not_card_types: vec![CardType::Land],
            ..Default::default()
        },
    };
    CardDefinition {
        key: "ogw.ayli_eternal_pilgrim",
        scryfall_id: "e7b5893d-6df6-4cae-ae70-d02d443d1740",
        printed: PrintedCharacteristics {
            name: "Ayli, Eternal Pilgrim".into(),
            mana_cost: Some(ManaCost::parse("{W}{B}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Kor".into(), "Cleric".into()],
            power: Some(2),
            toughness: Some(3),
            keywords: Keywords::of(&[Keyword::Deathtouch]),
            activated_abilities: vec![
                ActivatedAbilityDefinition {
                    mana_cost: ManaCost::parse("{1}"),
                    sacrifice: another_creature(),
                    // Its toughness as it last existed on the battlefield (CR 608.2h).
                    effects: vec![Effect::gain_life(
                        ObjectSelector::Paid(How::Sacrificed)
                            .property(ObjectProperty::Toughness, Aggregate::Sum),
                    )],
                    ..Default::default()
                },
                ActivatedAbilityDefinition {
                    mana_cost: ManaCost::parse("{1}{W}{B}"),
                    sacrifice: another_creature(),
                    condition: Some(Condition::compare(
                        EffectValue::LifeAboveStarting(PlayerSelector::You),
                        CompareOp::Ge,
                        10,
                    )),
                    effects: vec![Effect::exile(ObjectSelector::Target(nonland))],
                    ..Default::default()
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
