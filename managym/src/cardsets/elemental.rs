// elemental.rs
// Non-TLA cards retained alongside the shared typed TLA catalog.

use super::alpha::CardRegistry;
use crate::state::{
    ability::{Ability, Effect, EffectValue, TargetSpec, TriggerCondition, TriggerSubject},
    card::{ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keywords},
    mana::ManaCost,
    predicate::CardPredicate,
};

impl CardRegistry {
    pub(super) fn register_elemental(&mut self) {
        self.register_card(CardDefinition {
            name: "Professor of Symbology".into(),
            mana_cost: Some(ManaCost::parse("{1}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Kor".into(), "Cleric".into()],
            text_box: "When this creature enters, learn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)".into(),
            power: Some(2),
            toughness: Some(1),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::Learn],
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Gnarled Professor".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Treefolk".into(), "Druid".into()],
            text_box: "Trample\nWhen this creature enters, learn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)".into(),
            keywords: Keywords {
                trample: true,
                ..Default::default()
            },
            power: Some(5),
            toughness: Some(4),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::Learn],
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Eyetwitch".into(),
            mana_cost: Some(ManaCost::parse("{B}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Eye".into(), "Bat".into()],
            text_box: "Flying\nWhen this creature dies, learn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)".into(),
            keywords: Keywords {
                flying: true,
                ..Default::default()
            },
            power: Some(1),
            toughness: Some(1),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::Dies {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::Learn],
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Enthusiastic Study".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}")),
            types: CardTypes::new([CardType::Instant]),
            text_box: "Target creature gets +3/+1 and gains trample until end of turn.\nLearn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)".into(),
            spell_effects: vec![
                Effect::BuffTarget {
                    power: 3,
                    toughness: 1,
                    target: TargetSpec::Creature,
                },
                Effect::GrantKeywordsToTarget {
                    keywords: Keywords {
                        trample: true,
                        ..Default::default()
                    },
                    target: TargetSpec::Creature,
                },
                Effect::Learn,
            ],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Monastery Swiftspear".into(),
            mana_cost: Some(ManaCost::parse("{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Human".into(), "Monk".into()],
            text_box: "Haste\nProwess (Whenever you cast a noncreature spell, this creature gets +1/+1 until end of turn.)".into(),
            keywords: Keywords {
                haste: true,
                ..Default::default()
            },
            power: Some(1),
            toughness: Some(2),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::YouCastSpell {
                    predicate: CardPredicate {
                        not_card_types: vec![CardType::Creature],
                        ..Default::default()
                    },
                },
                effects: vec![Effect::ModifyUntilEot {
                    power_delta: 1,
                    toughness_delta: 1,
                }],
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Spectral Sailor".into(),
            mana_cost: Some(ManaCost::parse("{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Spirit".into(), "Pirate".into()],
            text_box: "Flash (You may cast this spell any time you could cast an instant.)\nFlying\n{3}{U}: Draw a card.".into(),
            keywords: Keywords {
                flying: true,
                flash: true,
                ..Default::default()
            },
            power: Some(1),
            toughness: Some(1),
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: false,
                mana_cost: ManaCost::parse("3U"),
                sacrifice_source: false,
                waterbend: false,
                effect: Effect::DrawCards { count: 1 },
            ..Default::default()
}],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Giant Cindermaw".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dinosaur".into(), "Beast".into()],
            text_box: "Trample (This creature can deal excess combat damage to the player or planeswalker it's attacking.)\nPlayers can't gain life.".into(),
            keywords: Keywords {
                trample: true,
                ..Default::default()
            },
            power: Some(4),
            toughness: Some(3),
            prevents_life_gain: true,
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Savor".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Instant]),
            text_box: "Target creature gets -2/-2 until end of turn. Create a Food token. (It's an artifact with \"{2}, {T}, Sacrifice this token: You gain 3 life.\")".into(),
            spell_effects: vec![
                Effect::BuffTarget {
                    power: -2,
                    toughness: -2,
                    target: TargetSpec::Creature,
                },
                Effect::CreateToken {
                    token_name: "Food".into(),
                    count: 1,
                    tapped_and_attacking: false,
                },
            ],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Fists of Flame".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Instant]),
            text_box: "Draw a card. Until end of turn, target creature gains trample and gets +1/+0 for each card you've drawn this turn. (A creature with trample can deal excess combat damage to the player or planeswalker it's attacking.)".into(),
            spell_effects: vec![
                Effect::DrawCards { count: 1 },
                Effect::BuffTargetValue {
                    power: EffectValue::CardsDrawnMinus { subtract: 0 },
                    toughness: 0,
                    target: TargetSpec::Creature,
                },
                Effect::GrantKeywordsToTarget {
                    keywords: Keywords {
                        trample: true,
                        ..Default::default()
                    },
                    target: TargetSpec::Creature,
                },
            ],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Power Word Kill".into(),
            mana_cost: Some(ManaCost::parse("{1}{B}")),
            types: CardTypes::new([CardType::Instant]),
            text_box: "Destroy target non-Angel, non-Demon, non-Devil, non-Dragon creature.".into(),
            spell_effects: vec![Effect::Destroy {
                target: TargetSpec::Permanent {
                    predicate: CardPredicate {
                        card_type: Some(CardType::Creature),
                        not_subtypes: vec![
                            "Angel".into(),
                            "Demon".into(),
                            "Devil".into(),
                            "Dragon".into(),
                        ],
                        ..Default::default()
                    },
                },
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Field Trip".into(),
            mana_cost: Some(ManaCost::parse("{2}{G}")),
            types: CardTypes::new([CardType::Sorcery]),
            text_box: "Search your library for a basic Forest card, put that card onto the battlefield tapped, then shuffle.\nLearn. (You may reveal a Lesson card you own from outside the game and put it into your hand, or discard a card to draw a card.)".into(),
            spell_effects: vec![
                Effect::SearchBasicLand {
                    subtype: Some("Forest".into()), count: 1, tapped: true, untap_if_lands: None, bottom_source: false,
                },
                Effect::Learn,
            ],
            ..Default::default()
        });
    }
}
