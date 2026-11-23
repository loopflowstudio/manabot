//! Final Fantasy.
use managym_state::{
    ability::{
        Ability, Condition, Effect, ObjectSelector, PlayerSelector, TriggerCondition,
        TriggerSubject,
    },
    card::{
        CardDefinition, CardType, CardTypes, Keyword, Keywords, PrintedCharacteristics,
        StaticAbility,
    },
    effect::Modification,
    mana::{Color, Colors, Mana, ManaCost},
    permanent::CounterKind,
    predicate::CardPredicate,
    zone::ZoneType,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![esper_origins(), sazh_s_chocobo()]
}

fn esper_origins() -> CardDefinition {
    let text_box = "Surveil 2. You gain 2 life. If this spell was cast from a graveyard, exile it, then put it onto the battlefield transformed under its owner's control with a finality counter on it. (If a creature with a finality counter on it would die, exile it instead.)\nFlashback {3}{G} (You may cast this card from your graveyard for its flashback cost. Then exile it.)";
    let back_text = "(As this Saga enters and after your draw step, add a lore counter. Sacrifice after III.)\nI — Reveal the top card of your library. If it's a permanent card, put it into your hand.\nII — Add {G}{G}.\nIII — Other creatures you control get +2/+2 and gain trample until end of turn.";
    let other_creatures = ObjectSelector::each_other_you_control(CardPredicate::creature());
    CardDefinition {
        key: "fin.esper_origins",
        scryfall_id: "0f503360-216a-4629-89b2-d32072850aef",
        printed: PrintedCharacteristics {
            name: "Esper Origins".into(),
            mana_cost: Some(ManaCost::parse("{1}{G}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![
                Effect::Surveil { count: 2 },
                Effect::gain_life(2),
                Effect::If {
                    condition: Condition::CastFromGraveyard,
                    then: vec![
                        Effect::exile(ObjectSelector::This),
                        Effect::Move {
                            what: ObjectSelector::Moved,
                            to: ZoneType::Battlefield,
                            under: Some(PlayerSelector::Owner),
                            transformed: true,
                            with_counter: Some(CounterKind::Finality),
                            tapped: false,
                            library_end: None,
                            linked: None,
                        },
                    ],
                    otherwise: vec![],
                },
            ],
            statics: vec![StaticAbility::Flashback(ManaCost::parse("{3}{G}"))],
            text_box: text_box.into(),
            ..Default::default()
        },
        back_face: Some(Box::new(PrintedCharacteristics {
            name: "Summon: Esper Maduin".into(),
            types: CardTypes::new([CardType::Enchantment, CardType::Creature]),
            subtypes: vec!["Saga".into(), "Elemental".into()],
            color_override: Some(Colors::from_iter([Color::Green])),
            power: Some(4),
            toughness: Some(4),
            abilities: vec![
                Ability::chapter(
                    1,
                    vec![
                        Effect::RevealLibraryTop,
                        Effect::If {
                            condition: Condition::CardsThisWayMatch(CardPredicate {
                                card_types_any: vec![
                                    CardType::Artifact,
                                    CardType::Battle,
                                    CardType::Creature,
                                    CardType::Enchantment,
                                    CardType::Land,
                                    CardType::Planeswalker,
                                ],
                                ..Default::default()
                            }),
                            then: vec![Effect::PutTopCardsInHand { count: 1 }],
                            otherwise: vec![],
                        },
                    ],
                ),
                Ability::chapter(2, vec![Effect::add_mana(Mana::parse("GG"))]),
                Ability::chapter(
                    3,
                    vec![
                        Effect::until_end_of_turn(
                            other_creatures.clone(),
                            Modification::pump(2, 2),
                        ),
                        Effect::until_end_of_turn(
                            other_creatures,
                            Modification::AddKeywords(Keywords::of(&[Keyword::Trample])),
                        ),
                    ],
                ),
            ],
            text_box: back_text.into(),
            ..Default::default()
        })),
    }
}

fn sazh_s_chocobo() -> CardDefinition {
    let text_box =
        "Landfall — Whenever a land you control enters, put a +1/+1 counter on this creature.";

    CardDefinition {
        key: "fin.sazh_s_chocobo",
        scryfall_id: "dda6b4d0-1b60-46b0-b321-b9ffe15afff4",
        printed: PrintedCharacteristics {
            name: "Sazh's Chocobo".into(),
            mana_cost: Some(ManaCost::parse("{G}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Bird".into()],
            power: Some(0),
            toughness: Some(1),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::you_control(CardPredicate {
                        card_type: Some(CardType::Land),
                        ..Default::default()
                    }),
                },
                effects: vec![Effect::put_counters(
                    ObjectSelector::This,
                    CounterKind::PlusOnePlusOne,
                    1,
                )],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
