//! Modern Horizons.
use managym_state::{
    ability::{Effect, EffectValue, ObjectSelector, TargetSpec},
    card::{
        ActivatedAbilityDefinition, AdditionalCost, CardDefinition, CardType, CardTypes, Keyword,
        Keywords,
    },
    effect::Modification,
    mana::{Color, Mana, ManaCost},
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        fists_of_flame(),
        force_of_negation(),
        prismatic_vista(),
        sunbaked_canyon(),
    ]
}

fn force_of_negation() -> CardDefinition {
    let text_box = "If it's not your turn, you may exile a blue card from your hand rather than pay this spell's mana cost.\nCounter target noncreature spell. If that spell is countered this way, exile it instead of putting it into its owner's graveyard.";
    CardDefinition {
        key: "mh1.force_of_negation",
        scryfall_id: "e9be371c-c688-44ad-ab71-bd4c9f242d58",
        name: "Force of Negation".into(),
        mana_cost: Some(ManaCost::parse("{1}{U}{U}")),
        types: CardTypes::new([CardType::Instant]),
        additional_costs: vec![AdditionalCost::ExileFromHandAlternative {
            predicate: CardPredicate {
                colors_any: vec![Color::Blue],
                ..Default::default()
            },
            not_your_turn: true,
        }],
        spell_effects: vec![Effect::CounterSpell {
            what: ObjectSelector::Target(TargetSpec::SpellMatching {
                predicate: CardPredicate {
                    not_card_types: vec![CardType::Creature],
                    ..Default::default()
                },
            }),
            exile: true,
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn fists_of_flame() -> CardDefinition {
    let text_box = "Draw a card. Until end of turn, target creature gains trample and gets +1/+0 for each card you've drawn this turn. (A creature with trample can deal excess combat damage to the player or planeswalker it's attacking.)";

    CardDefinition {
        key: "mh1.fists_of_flame",
        scryfall_id: "4fa3c0d8-0a3e-43b0-b81e-a3d22ce6a4b2",
        name: "Fists of Flame".into(),
        mana_cost: Some(ManaCost::parse("{1}{R}")),
        types: CardTypes::new([CardType::Instant]),
        spell_effects: vec![
            Effect::DrawCards {
                count: EffectValue::Fixed(1),
            },
            Effect::until_end_of_turn(
                ObjectSelector::Target(TargetSpec::Creature),
                Modification::ModifyPt {
                    power: EffectValue::CardsDrawnMinus { subtract: 0 },
                    toughness: EffectValue::Fixed(0),
                },
            ),
            Effect::until_end_of_turn(
                ObjectSelector::Target(TargetSpec::Creature),
                Modification::AddKeywords(Keywords::of(&[Keyword::Trample])),
            ),
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn prismatic_vista() -> CardDefinition {
    let text_box = "{T}, Pay 1 life, Sacrifice this land: Search your library for a basic land card, put it onto the battlefield, then shuffle.";

    CardDefinition {
        key: "mh1.prismatic_vista",
        scryfall_id: "e37da81e-be12-45a2-9128-376f1ad7b3e8",
        name: "Prismatic Vista".into(),
        types: CardTypes::new([CardType::Land]),
        activated_abilities: vec![ActivatedAbilityDefinition {
            life_cost: 1,
            tap_source: true,
            sacrifice_source: true,
            effects: vec![Effect::search_library(
                CardPredicate::basic_land(),
                EffectValue::Fixed(1),
                vec![Effect::return_to_battlefield(
                    ObjectSelector::Selected,
                    false,
                )],
            )],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn sunbaked_canyon() -> CardDefinition {
    let text_box = "{T}, Pay 1 life: Add {R} or {W}.\n{1}, {T}, Sacrifice this land: Draw a card.";
    CardDefinition {
        key: "mh1.sunbaked_canyon",
        scryfall_id: "c36820fa-ee86-4206-9a0d-737a67cf5208",
        name: "Sunbaked Canyon".into(),
        types: CardTypes::new([CardType::Land]),
        activated_abilities: vec![
            ActivatedAbilityDefinition {
                tap_source: true,
                life_cost: 1,
                effects: vec![Effect::mana_choice(["R", "W"].map(Mana::parse))],
                ..Default::default()
            },
            ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{1}"),
                tap_source: true,
                sacrifice_source: true,
                effects: vec![Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                }],
                ..Default::default()
            },
        ],
        text_box: text_box.into(),
        ..Default::default()
    }
}
