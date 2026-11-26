//! Ravnica Allegiance.
use managym_state::{
    ability::{
        Ability, Actor, Aggregate, Effect, EffectValue, LibraryEnd, ObjectProperty, ObjectSelector,
        PaymentCost, Per, TargetRequirement, TargetSpec, TriggerCondition, TriggerSubject,
    },
    card::{
        BlockRestriction, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics, StaticAbility,
    },
    effect::Modification,
    mana::ManaCost,
    permanent::CounterKind,
    predicate::CardPredicate,
    zone::ZoneType,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        archway_angel(),
        biogenic_upgrade(),
        gate_colossus(),
        gateway_sneak(),
    ]
}

fn gate_colossus() -> CardDefinition {
    let text_box = "Affinity for Gates (This spell costs {1} less to cast for each Gate you control.)\nThis creature can't be blocked by creatures with power 2 or less.\nWhenever a Gate you control enters, you may put this card from your graveyard on top of your library.";
    CardDefinition {
        key: "rna.gate_colossus",
        scryfall_id: "99767e2f-a558-4d63-b9b6-923d15b433e1",
        printed: PrintedCharacteristics {
            name: "Gate Colossus".into(),
            mana_cost: Some(ManaCost::parse("{8}")),
            types: CardTypes::new([CardType::Artifact, CardType::Creature]),
            subtypes: vec!["Construct".into()],
            power: Some(8),
            toughness: Some(8),
            statics: vec![
                StaticAbility::CostsLessPerPermanent(CardPredicate::subtype("Gate")),
                StaticAbility::BlockRestriction(BlockRestriction::CantBeBlockedBy(CardPredicate {
                    max_power: Some(2),
                    ..CardPredicate::creature()
                })),
            ],
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::InGraveyard {
                    condition: Box::new(TriggerCondition::EntersTheBattlefield {
                        subject: TriggerSubject::you_control(CardPredicate::subtype("Gate")),
                    }),
                },
                intervening_if: None,
                effects: vec![Effect::OptionalPayment {
                    cost: PaymentCost::Mana(ManaCost::default()),
                    if_paid: vec![Effect::Move {
                        unrevealed: false,
                        face_down: false,
                        library_end: Some(LibraryEnd::Top),
                        what: ObjectSelector::This,
                        to: ZoneType::Library,
                        tapped: false,
                        transformed: false,
                        under: None,
                        with_counter: None,
                        linked: None,
                    }],
                    if_declined: vec![],
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn gateway_sneak() -> CardDefinition {
    let text_box = "Whenever a Gate you control enters, this creature can't be blocked this turn.\nWhenever this creature deals combat damage to a player, draw a card.";
    CardDefinition {
        key: "rna.gateway_sneak",
        scryfall_id: "edc0229d-05e6-41b7-b7a9-2a8b2b258add",
        printed: PrintedCharacteristics {
            name: "Gateway Sneak".into(),
            mana_cost: Some(ManaCost::parse("{2}{U}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Vedalken".into(), "Rogue".into()],
            power: Some(1),
            toughness: Some(3),
            abilities: vec![
                Ability::Triggered {
                    condition: TriggerCondition::EntersTheBattlefield {
                        subject: TriggerSubject::you_control(CardPredicate::subtype("Gate")),
                    },
                    intervening_if: None,
                    effects: vec![Effect::until_end_of_turn(
                        ObjectSelector::This,
                        Modification::CantBeBlocked,
                    )],
                },
                Ability::Triggered {
                    condition: TriggerCondition::DealsCombatDamage {
                        subject: TriggerSubject::This,
                        to: Some(Actor::AnyPlayer),
                        per: Per::Object,
                    },
                    intervening_if: None,
                    effects: vec![Effect::DrawCards {
                        count: EffectValue::Fixed(1),
                    }],
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn archway_angel() -> CardDefinition {
    let text_box = "Flying\nWhen this creature enters, you gain 2 life for each Gate you control.";
    CardDefinition {
        key: "rna.archway_angel",
        scryfall_id: "b209d219-b946-4226-a8b4-65a5f3837fac",
        printed: PrintedCharacteristics {
            name: "Archway Angel".into(),
            mana_cost: Some(ManaCost::parse("{5}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Angel".into()],
            power: Some(3),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![Ability::Triggered {
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                intervening_if: None,
                // One gain of twice the Gates (CR 119.9), not one per Gate.
                effects: vec![Effect::gain_life(EffectValue::Scaled {
                    factor: 2,
                    value: Box::new(
                        ObjectSelector::each_you_control(CardPredicate::subtype("Gate")).count(),
                    ),
                })],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn biogenic_upgrade() -> CardDefinition {
    let counters = CounterKind::PlusOnePlusOne;
    let targets = ObjectSelector::Targets(TargetSpec::Creature);
    let counters_on_target = ObjectSelector::Target(TargetSpec::Creature)
        .property(ObjectProperty::Counters(counters), Aggregate::Sum);
    CardDefinition {
        key: "rna.biogenic_upgrade",
        scryfall_id: "0dd73fb2-453f-40b9-8beb-dfa99e6a706e",
        printed: PrintedCharacteristics {
            name: "Biogenic Upgrade".into(),
            mana_cost: Some(ManaCost::parse("{4}{G}{G}")),
            types: CardTypes::new([CardType::Sorcery]),
            division: 3,
            targeting: vec![TargetRequirement {
                max: 3,
                ..TargetRequirement::one(TargetSpec::Creature)
            }],
            spell_effects: vec![
                Effect::put_counters(targets.clone(), counters, EffectValue::TargetShare),
                Effect::put_counters(targets, counters, counters_on_target),
            ],
            text_box: "Distribute three +1/+1 counters among one, two, or three target creatures, then double the number of +1/+1 counters on each of those creatures.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
