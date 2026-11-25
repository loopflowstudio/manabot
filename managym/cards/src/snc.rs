//! Streets of New Capenna.
use managym_state::ability::{
    Ability, Effect, ObjectSelector, PlayerSelector, TargetSpec, TriggerCondition, TriggerSubject,
};
use managym_state::card::{
    CardDefinition, CardType, CardTypes, Keyword, Keywords, LayerValue, PrintedCharacteristics,
    StaticAbility, StaticScope,
};
use managym_state::effect::{EffectDuration, Modification};
use managym_state::mana::{Color, ManaCost};
use managym_state::predicate::CardPredicate;

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        venom_connoisseur(),
        an_offer_you_cant_refuse(),
        witness_protection(),
        inspiring_overseer(),
        involuntary_employment(),
    ]
}

fn venom_connoisseur() -> CardDefinition {
    let grant = |who| {
        Effect::until_end_of_turn(
            who,
            Modification::AddKeywords(Keywords::of(&[Keyword::Deathtouch])),
        )
    };
    let text_box = "Alliance — Whenever another creature you control enters, this creature gains deathtouch until end of turn. If this is the second time this ability has resolved this turn, all creatures you control gain deathtouch until end of turn.";
    CardDefinition {
        key: "snc.venom_connoisseur",
        scryfall_id: "04897d8c-ee05-45eb-80f7-76487dbcc449",
        printed: PrintedCharacteristics {
            name: "Venom Connoisseur".into(),
            mana_cost: Some(ManaCost::parse("{1}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Druid".into()],
            power: Some(2),
            toughness: Some(2),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::another_you_control(CardPredicate::creature()),
                },
                intervening_if: None,
                effects: vec![
                    grant(ObjectSelector::This),
                    Effect::OnNthResolutionThisTurn {
                        n: 2,
                        effect: Box::new(grant(ObjectSelector::each_you_control(
                            CardPredicate::creature(),
                        ))),
                    },
                ],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn an_offer_you_cant_refuse() -> CardDefinition {
    let text_box = "Counter target noncreature spell. Its controller creates two Treasure tokens. (They're artifacts with \"{T}, Sacrifice this token: Add one mana of any color.\")";
    CardDefinition {
        key: "snc.an_offer_you_cant_refuse",
        scryfall_id: "b9d349f3-5be2-4b1f-a4c3-ba94822cf0cf",
        printed: PrintedCharacteristics {
            name: "An Offer You Can't Refuse".into(),
            mana_cost: Some(ManaCost::parse("{U}")),
            types: CardTypes::new([CardType::Instant]),
            spell_effects: vec![
                Effect::CounterSpell {
                    what: ObjectSelector::Target(TargetSpec::SpellMatching {
                        predicate: CardPredicate {
                            not_card_types: vec![CardType::Creature],
                            ..Default::default()
                        },
                    }),
                    exile: false,
                },
                // Its controller as it is or last existed, whether or not
                // the spell could be countered.
                Effect::ForPlayer {
                    player: PlayerSelector::TargetController,
                    effects: vec![Effect::create_tokens("token.treasure", 2)],
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn involuntary_employment() -> CardDefinition {
    let text_box = "Gain control of target creature until end of turn. Untap that creature. It gains haste until end of turn. Create a Treasure token. (It's an artifact with \"{T}, Sacrifice this token: Add one mana of any color.\")";
    CardDefinition {
        key: "snc.involuntary_employment",
        scryfall_id: "af7626e2-0041-4ac5-97b3-2db8dcb094d4",
        printed: PrintedCharacteristics {
            name: "Involuntary Employment".into(),
            mana_cost: Some(ManaCost::parse("{3}{R}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![
                Effect::GainControl {
                    player: PlayerSelector::You,
                    what: ObjectSelector::Target(TargetSpec::Creature),
                    duration: EffectDuration::EndOfTurn,
                },
                Effect::untap(ObjectSelector::Target(TargetSpec::Creature)),
                Effect::until_end_of_turn(
                    ObjectSelector::Target(TargetSpec::Creature),
                    Modification::AddKeywords(Keywords::of(&[Keyword::Haste])),
                ),
                Effect::create_tokens("token.treasure", 1),
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn inspiring_overseer() -> CardDefinition {
    let text_box = "Flying\nWhen this creature enters, you gain 1 life and draw a card.";
    CardDefinition {
        key: "snc.inspiring_overseer",
        scryfall_id: "35d9da1d-8678-4252-b0f8-9960795642f0",
        printed: PrintedCharacteristics {
            name: "Inspiring Overseer".into(),
            mana_cost: Some(ManaCost::parse("{2}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Angel".into(), "Cleric".into()],
            power: Some(2),
            toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![Ability::when_enters(
                TriggerSubject::This,
                vec![Effect::gain_life(1), Effect::DrawCards { count: 1.into() }],
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn witness_protection() -> CardDefinition {
    let text_box = "Enchant creature\nEnchanted creature loses all abilities and is a green and white Citizen creature with base power and toughness 1/1 named Legitimate Businessperson. (It loses all other colors, card types, creature types, and names.)";
    CardDefinition {
        key: "snc.witness_protection",
        scryfall_id: "a2be6f2c-8ad0-402d-a7ca-9fe817e83b72",
        printed: PrintedCharacteristics {
            name: "Witness Protection".into(),
            mana_cost: Some(ManaCost::parse("{U}")),
            types: CardTypes::new([CardType::Enchantment]),
            subtypes: vec!["Aura".into()],
            // One effect across layers 1, 4, 5, 6 and 7b (CR 613.6).
            statics: vec![StaticAbility::continuous(
                StaticScope::Attached,
                None,
                vec![
                    Modification::SetName("Legitimate Businessperson".into()),
                    Modification::SetTypes {
                        types: vec![CardType::Creature],
                        subtypes: vec!["Citizen".into()],
                    },
                    Modification::SetColors([Color::Green, Color::White].into_iter().collect()),
                    Modification::LoseAllAbilities,
                    Modification::SetBasePt {
                        power: Some(LayerValue::Fixed(1)),
                        toughness: Some(LayerValue::Fixed(1)),
                    },
                ],
            )],
            spell_effects: vec![Effect::enchant(TargetSpec::Creature)],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
