//! Dragons of Tarkir.
use managym_state::{
    ability::{
        Ability, Actor, Condition, Effect, EffectValue, ObjectSelector, PlayerSelector, Scope,
        TargetSpec, TriggerSubject,
    },
    card::{
        CardDefinition, CardType, CardTypes, PrintedCharacteristics, SpellCostReduction,
        StaticAbility,
    },
    mana::{Color, ManaCost},
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        seismic_rupture(),
        clone_legion(),
        dragonlords_servant(),
        impact_tremors(),
        rending_volley(),
    ]
}

fn clone_legion() -> CardDefinition {
    let text_box =
        "For each creature target player controls, create a token that's a copy of that creature.";
    CardDefinition {
        key: "dtk.clone_legion",
        scryfall_id: "45c6a201-1d4d-4ae5-a84e-098e094eaf61",
        printed: PrintedCharacteristics {
            name: "Clone Legion".into(),
            mana_cost: Some(ManaCost::parse("{7}{U}{U}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![Effect::create_token_copies(
                ObjectSelector::each(Scope::TargetPlayerControls, CardPredicate::creature()),
                1,
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn rending_volley() -> CardDefinition {
    let text_box = "This spell can't be countered.\nRending Volley deals 4 damage to target white or blue creature.";
    let white_or_blue_creature = TargetSpec::Permanent {
        predicate: CardPredicate {
            card_type: Some(CardType::Creature),
            colors_any: vec![Color::White, Color::Blue],
            ..Default::default()
        },
    };
    CardDefinition {
        key: "dtk.rending_volley",
        scryfall_id: "8234090e-9df1-4915-90ef-8a4bc6212655",
        printed: PrintedCharacteristics {
            name: "Rending Volley".into(),
            mana_cost: Some(ManaCost::parse("{R}")),
            types: CardTypes::new([CardType::Instant]),
            statics: vec![StaticAbility::CantBeCountered],
            spell_effects: vec![Effect::deal_damage(
                4,
                ObjectSelector::Target(white_or_blue_creature),
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn dragonlords_servant() -> CardDefinition {
    let text_box = "Dragon spells you cast cost {1} less to cast.";
    CardDefinition {
        key: "dtk.dragonlords_servant",
        scryfall_id: "0ffcdd54-b6be-4d42-82c0-ae927037e859",
        printed: PrintedCharacteristics {
            name: "Dragonlord's Servant".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Goblin".into(), "Shaman".into()],
            power: Some(1),
            toughness: Some(3),
            statics: vec![StaticAbility::SpellsCostLess(SpellCostReduction {
                first_on_own_turn: false,
                predicate: CardPredicate::subtype("Dragon"),
                condition: Condition::Always,
                generic: EffectValue::Fixed(1),
            })],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn impact_tremors() -> CardDefinition {
    CardDefinition {
        key: "dtk.impact_tremors",
        scryfall_id: "56fb4035-197b-4d28-9bf7-bb62c304067e",
        printed: PrintedCharacteristics {
            name: "Impact Tremors".into(),
            mana_cost: Some(ManaCost::parse("{1}{R}")),
            types: CardTypes::new([CardType::Enchantment]),
            abilities: vec![Ability::when_enters(
                TriggerSubject::matching(Actor::You, CardPredicate::creature(), false),
                vec![Effect::DealDamageToPlayer { player: PlayerSelector::EachOpponent, amount: 1.into() }],
            )],
            text_box: "Whenever a creature you control enters, this enchantment deals 1 damage to each opponent.".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn seismic_rupture() -> CardDefinition {
    let text_box = "Seismic Rupture deals 2 damage to each creature without flying.";
    CardDefinition {
        key: "dtk.seismic_rupture",
        scryfall_id: "9b952e4e-c1ed-4455-90d5-46b56478e6b0",
        printed: PrintedCharacteristics {
            name: "Seismic Rupture".into(),
            mana_cost: Some(ManaCost::parse("{2}{R}")),
            types: CardTypes::new([CardType::Sorcery]),
            spell_effects: vec![Effect::deal_damage(
                2,
                ObjectSelector::each(
                    Scope::Battlefield,
                    CardPredicate {
                        not_flying: true,
                        ..CardPredicate::creature()
                    },
                ),
            )],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
