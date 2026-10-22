// temur.rs
// Temur mana, graveyard casting, and Dragon registrations. Printed rules are
// independently retained in the Scryfall conformance snapshot.

use super::alpha::CardRegistry;
use crate::state::{
<<<<<<< HEAD
    ability::{Ability, Effect, StaticCondition, TriggerCondition, TriggerSubject},
=======
    ability::{Ability, Effect, PaymentCost, StaticCondition, TriggerCondition, TriggerSubject},
>>>>>>> 20194feb758cd6d1df3e63b8d4fdd567d3ebbe6f
    card::{ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keywords},
    mana::{Mana, ManaCost},
    predicate::CardPredicate,
};

/// Enumerate unordered pairs directly, retaining the authored mode order.
fn lesson_mana_pairs() -> Effect {
    let colors = ["W", "U", "B", "R", "G"].map(Mana::parse);
    Effect::mana_choice(colors.iter().enumerate().flat_map(|(index, first)| {
        colors[index..].iter().map(move |second| {
            let mut pair = first.clone();
            pair.add(second);
            Mana {
                lesson: pair.mana,
                ..Default::default()
            }
        })
    }))
}

fn firebending_two() -> Ability {
    Ability::Triggered {
        condition: TriggerCondition::Attacks {
            subject: TriggerSubject::This,
        },
        effects: vec![Effect::AddMana {
            mana: Mana::parse("RR"),
            until_end_of_combat: true,
        }],
    }
}

impl CardRegistry {
    pub(super) fn register_temur(&mut self) {
        self.register_card(CardDefinition {
            name: "Great Divide Guide".into(),
            mana_cost: Some(ManaCost::parse("{1}{G}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec![],
            subtypes: vec!["Human".into(), "Scout".into(), "Ally".into()],
            text_box: "Each land and Ally you control has \"{T}: Add one mana of any color.\""
                .into(),
            power: Some(2),
            toughness: Some(3),
            mana_grant: vec![
                CardPredicate {
                    card_type: Some(CardType::Land),
                    ..Default::default()
                },
                CardPredicate::subtype("Ally"),
            ],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Hermitic Herbalist".into(),
            mana_cost: Some(ManaCost::parse("{G}{U}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec![],
            subtypes: vec!["Human".into(), "Druid".into(), "Ally".into()],
            text_box: "{T}: Add one mana of any color.\n{T}: Add two mana in any combination of colors. Spend this mana only to cast Lesson spells.".into(),
            power: Some(2),
            toughness: Some(3),
            activated_abilities: vec![
                ActivatedAbilityDefinition {
                    tap_source: true,
                    effect: Effect::any_color_mana(),
                    ..Default::default()
                },
                ActivatedAbilityDefinition {
                    tap_source: true,
                    effect: lesson_mana_pairs(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Iroh, Grand Lotus".into(),
            mana_cost: Some(ManaCost::parse("{3}{G}{U}{R}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["legendary".into()],
            subtypes: vec!["Human".into(), "Noble".into(), "Ally".into()],
            text_box: "Firebending 2\nDuring your turn, each non-Lesson instant and sorcery card in your graveyard has flashback. The flashback cost is equal to that card's mana cost. (You may cast a card from your graveyard for its flashback cost. Then exile it.)\nDuring your turn, each Lesson card in your graveyard has flashback {1}.".into(),
            power: Some(5),
            toughness: Some(5),
            grants_flashback: true,
            abilities: vec![firebending_two()],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Prophetic Prism".into(),
            mana_cost: Some(ManaCost::parse("{2}")),
            types: CardTypes::new([CardType::Artifact]),
            supertypes: vec![],
            subtypes: vec![],
            text_box:
                "When this artifact enters, draw a card.\n{1}, {T}: Add one mana of any color."
                    .into(),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::DrawCards { count: 1 }],
            }],
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                mana_cost: ManaCost::parse("{1}"),
                effect: Effect::any_color_mana(),
                ..Default::default()
            }],
            ..Default::default()
        });
        let dragon_or_lesson = CardPredicate {
            subtypes_any: vec!["Dragon".into(), "Lesson".into()],
            ..Default::default()
        };
        self.register_card(CardDefinition {
            name: "Ran and Shaw".into(),
            mana_cost: Some(ManaCost::parse("{3}{R}{R}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["legendary".into()],
            subtypes: vec!["Dragon".into()],
            text_box: "Flying, firebending 2\nWhen Ran and Shaw enter, if you cast them and there are three or more Dragon and/or Lesson cards in your graveyard, create a token that's a copy of Ran and Shaw, except it's not legendary.\n{3}{R}: Dragons you control get +2/+0 until end of turn.".into(),
            power: Some(4),
            toughness: Some(4),
            keywords: Keywords { flying: true, ..Default::default() },
            abilities: vec![
                firebending_two(),
                Ability::Triggered {
                    condition: TriggerCondition::ActiveIf {
                        active_if: StaticCondition::GraveyardAtLeast {
                            count: 3,
                            predicate: dragon_or_lesson.clone(),
                        },
                        condition: Box::new(TriggerCondition::WasCast {
                            condition: Box::new(TriggerCondition::EntersTheBattlefield {
                                subject: TriggerSubject::This,
                            }),
                        }),
                    },
                    effects: vec![Effect::IfGraveyardAtLeast {
                        count: 3,
                        predicate: dragon_or_lesson,
                        then: vec![Effect::CreateNonlegendaryCopy],
                        otherwise: vec![],
                    }],
                },
            ],
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{3}{R}"),
                effect: Effect::ModifyControlled {
                    predicate: CardPredicate::subtype("Dragon"),
                    power: 2,
                    toughness: 0,
                },
                ..Default::default()
            }],
            ..Default::default()
        });
        self.register_card(CardDefinition {
            name: "Waterbending Lesson".into(),
            mana_cost: Some(ManaCost::parse("{3}{U}")),
            types: CardTypes::new([CardType::Sorcery]),
            supertypes: vec![],
            subtypes: vec!["Lesson".into()],
            text_box: "Draw three cards. Then discard a card unless you waterbend {2}. (While paying a waterbend cost, you can tap your artifacts and creatures to help. Each one pays for {1}.)".into(),
            spell_effects: vec![
                Effect::DrawCards { count: 3 },
<<<<<<< HEAD
                Effect::WaterbendOrDiscard { generic: 2 },
=======
                Effect::OptionalPayment {
                    cost: PaymentCost::Waterbend { generic: 2 },
                    if_paid: vec![],
                    if_declined: vec![Effect::Discard],
                },
>>>>>>> 20194feb758cd6d1df3e63b8d4fdd567d3ebbe6f
            ],
            ..Default::default()
        });
    }
}
