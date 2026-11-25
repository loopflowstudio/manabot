//! Magic 2012.
use managym_state::{
    ability::{
        Ability, Condition, Effect, EffectValue, ObjectSelector, TargetRequirement, TargetSpec,
        TriggerCondition, TriggerSubject,
    },
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        LayerValue, PrintedCharacteristics, StaticAbility, StaticContinuous, StaticScope,
    },
    effect::Modification,
    mana::ManaCost,
    predicate::CardPredicate,
    zone::ZoneType,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        rune_scarred_demon(),
        swiftfoot_boots(),
        sundial_of_the_infinite(),
        visions_of_beyond(),
        warstorm_surge(),
        angelic_destiny(),
    ]
}

fn sundial_of_the_infinite() -> CardDefinition {
    let text_box = "{1}, {T}: End the turn. Activate only during your turn. (Exile all spells and abilities from the stack. Discard down to your maximum hand size. Damage wears off, and \"this turn\" and \"until end of turn\" effects end.)";
    CardDefinition {
        key: "m12.sundial_of_the_infinite",
        scryfall_id: "36d3da9c-cb7a-4cea-b6e6-6722bd16c73c",
        printed: PrintedCharacteristics {
            name: "Sundial of the Infinite".into(),
            mana_cost: Some(ManaCost::parse("{2}")),
            types: CardTypes::new([CardType::Artifact]),
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{1}"),
                tap_source: true,
                own_turn_only: true,
                effects: vec![Effect::EndTheTurn],
                ..Default::default()
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn visions_of_beyond() -> CardDefinition {
    let text_box =
        "Draw a card. If a graveyard has twenty or more cards in it, draw three cards instead.";
    let draw = |count| Effect::DrawCards {
        count: EffectValue::Fixed(count),
    };
    CardDefinition {
        key: "m12.visions_of_beyond",
        scryfall_id: "75657d26-b0f8-4892-8684-533c103c921d",
        printed: PrintedCharacteristics {
            name: "Visions of Beyond".into(),
            mana_cost: Some(ManaCost::parse("{U}")),
            types: CardTypes::new([CardType::Instant]),
            // The condition is read once, before any card is drawn.
            spell_effects: vec![Effect::If {
                condition: Condition::AnyGraveyardAtLeast { count: 20 },
                then: vec![draw(3)],
                otherwise: vec![draw(1)],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn warstorm_surge() -> CardDefinition {
    let text_box =
        "Whenever a creature you control enters, it deals damage equal to its power to any target.";
    CardDefinition {
        key: "m12.warstorm_surge",
        scryfall_id: "b16443df-52c6-4c9d-a7ff-89a37e593a0a",
        printed: PrintedCharacteristics {
            name: "Warstorm Surge".into(),
            mana_cost: Some(ManaCost::parse("{5}{R}")),
            types: CardTypes::new([CardType::Enchantment]),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::another_you_control(CardPredicate::creature()),
                },
                // The creature is the source of the damage, not this enchantment.
                effects: vec![Effect::DealDamage {
                    amount: EffectValue::TriggeringPower,
                    what: ObjectSelector::Target(TargetSpec::AnyTarget),
                    by: Some(ObjectSelector::Triggering),
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn swiftfoot_boots() -> CardDefinition {
    let text_box = "Equipped creature has hexproof and haste. (It can't be the target of spells or abilities your opponents control. It can attack and {T} no matter when it came under your control.)\nEquip {1} ({1}: Attach to target creature you control. Equip only as a sorcery.)";
    CardDefinition {
        key: "m12.swiftfoot_boots",
        scryfall_id: "8b82753b-284c-44ba-9d48-d28913f02a5f",
        printed: PrintedCharacteristics {
            statics: vec![StaticAbility::continuous(
                StaticScope::Attached,
                None,
                vec![Modification::AddKeywords(Keywords::of(&[
                    Keyword::Hexproof,
                    Keyword::Haste,
                ]))],
            )],
            activated_abilities: vec![ActivatedAbilityDefinition::equip(ManaCost::parse("{1}"))],
            name: "Swiftfoot Boots".into(),
            types: CardTypes::new([CardType::Artifact]),
            subtypes: vec!["Equipment".into()],
            mana_cost: Some(ManaCost::parse("{2}")),
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn rune_scarred_demon() -> CardDefinition {
    let text_box = "Flying\nWhen this creature enters, search your library for a card, put it into your hand, then shuffle.";
    CardDefinition {
        key: "m12.rune_scarred_demon",
        scryfall_id: "509e0f81-0591-4b28-978e-a2f1c46b7427",
        printed: PrintedCharacteristics {
            name: "Rune-Scarred Demon".into(),
            mana_cost: Some(ManaCost::parse("{5}{B}{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Demon".into()],
            power: Some(6),
            toughness: Some(6),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::search_for_card(ZoneType::Hand)],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn angelic_destiny() -> CardDefinition {
    let text_box = "Enchant creature\nEnchanted creature gets +4/+4, has flying and first strike, and is an Angel in addition to its other types.\nWhen enchanted creature dies, return this card to its owner's hand.";
    CardDefinition {
        key: "m12.angelic_destiny",
        scryfall_id: "a0cd7438-fde2-4e26-9c34-52c476a971e9",
        printed: PrintedCharacteristics {
            name: "Angelic Destiny".into(),
            mana_cost: Some(ManaCost::parse("{2}{W}{W}")),
            types: CardTypes::new([CardType::Enchantment]),
            subtypes: vec!["Aura".into()],
            targeting: vec![TargetRequirement::one(TargetSpec::Creature)],
            spell_effects: vec![Effect::AttachSourceTo {
                what: ObjectSelector::Target(TargetSpec::Creature),
            }],
            statics: vec![StaticAbility::Continuous(StaticContinuous {
                scope: StaticScope::Attached,
                condition: None,
                characteristic_defining: false,
                modifications: vec![
                    Modification::ModifyPt {
                        power: LayerValue::Fixed(4),
                        toughness: LayerValue::Fixed(4),
                    },
                    Modification::AddKeywords(Keywords::of(&[
                        Keyword::Flying,
                        Keyword::FirstStrike,
                    ])),
                    Modification::AddSubtypes(vec!["Angel".into()]),
                ],
            })],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::Dies {
                    subject: TriggerSubject::Attached,
                },
                intervening_if: None,
                effects: vec![Effect::return_to_hand(ObjectSelector::This)],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
