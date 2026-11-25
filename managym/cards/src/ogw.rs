//! Oath of the Gatewatch.
use managym_state::ability::{
    Aggregate, CompareOp, Condition, Effect, EffectValue, ObjectProperty, ObjectSelector,
    PlayerSelector, TargetSpec,
};
use managym_state::card::{
    ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
    PrintedCharacteristics, SacrificeCost, StaticAbility, StaticScope,
};
use managym_state::effect::Modification;
use managym_state::mana::ManaCost;
use managym_state::payment::How;
use managym_state::predicate::CardPredicate;

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        sphinx_of_the_final_word(),
        ayli_eternal_pilgrim(),
        untamed_hunger(),
        make_a_stand(),
        tajuru_pathwarden(),
    ]
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

fn tajuru_pathwarden() -> CardDefinition {
    let text_box = "Vigilance (Attacking doesn't cause this creature to tap.)\nTrample (This creature can deal excess combat damage to the player or planeswalker it's attacking.)";
    CardDefinition {
        key: "ogw.tajuru_pathwarden",
        scryfall_id: "b073e75b-b432-41a5-a71e-d169fecf774f",
        printed: PrintedCharacteristics {
            name: "Tajuru Pathwarden".into(),
            mana_cost: Some(ManaCost::parse("{4}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Elf".into(), "Warrior".into(), "Ally".into()],
            power: Some(5),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Vigilance, Keyword::Trample]),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn make_a_stand() -> CardDefinition {
    let text_box = "Creatures you control get +1/+0 and gain indestructible until end of turn. (Damage and effects that say \"destroy\" don't destroy them.)";
    CardDefinition {
        key: "ogw.make_a_stand",
        scryfall_id: "30cace63-91ca-493c-b67f-740fbbf06370",
        printed: PrintedCharacteristics {
            name: "Make a Stand".into(),
            mana_cost: Some(ManaCost::parse("{2}{W}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: Effect::pump_and_grant_until_end_of_turn(
                ObjectSelector::each_you_control(CardPredicate::creature()),
                1,
                0,
                Keyword::Indestructible,
            ),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn untamed_hunger() -> CardDefinition {
    let text_box = "Enchant creature\nEnchanted creature gets +2/+1 and has menace. (It can't be blocked except by two or more creatures.)";
    CardDefinition {
        key: "ogw.untamed_hunger",
        scryfall_id: "595deb86-b6cf-4e4f-a6cf-f5ff128b720d",
        printed: PrintedCharacteristics {
            statics: vec![StaticAbility::continuous(
                StaticScope::Attached,
                None,
                vec![
                    Modification::AddKeywords(Keywords::of(&[Keyword::Menace])),
                    Modification::pump(2, 1),
                ],
            )],
            spell_effects: vec![Effect::enchant(TargetSpec::Creature)],
            name: "Untamed Hunger".into(),
            types: CardTypes::new([CardType::Enchantment]),
            subtypes: vec!["Aura".into()],
            mana_cost: Some(ManaCost::parse("{2}{B}")),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn sphinx_of_the_final_word() -> CardDefinition {
    let text_box = "This spell can't be countered.\nFlying\nHexproof (This creature can't be the target of spells or abilities your opponents control.)\nInstant and sorcery spells you control can't be countered.";
    CardDefinition {
        key: "ogw.sphinx_of_the_final_word",
        scryfall_id: "866f92a3-2738-4e0f-adda-3ff9227dc17a",
        printed: PrintedCharacteristics {
            name: "Sphinx of the Final Word".into(),
            mana_cost: Some(ManaCost::parse("{5}{U}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Sphinx".into()],
            power: Some(5),
            toughness: Some(5),
            keywords: Keywords::of(&[Keyword::Flying, Keyword::Hexproof]),
            statics: vec![
                StaticAbility::CantBeCountered,
                StaticAbility::SpellsCantBeCountered(CardPredicate::instant_or_sorcery()),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
