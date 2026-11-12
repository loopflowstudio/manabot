//! Magic 2012.
use managym_state::{
    ability::{
        Ability, Condition, Effect, EffectValue, ObjectSelector, TargetSpec, TriggerCondition,
        TriggerSubject,
    },
    card::{ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes},
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        sundial_of_the_infinite(),
        visions_of_beyond(),
        warstorm_surge(),
    ]
}

fn sundial_of_the_infinite() -> CardDefinition {
    let text_box = "{1}, {T}: End the turn. Activate only during your turn. (Exile all spells and abilities from the stack. Discard down to your maximum hand size. Damage wears off, and \"this turn\" and \"until end of turn\" effects end.)";
    CardDefinition {
        key: "m12.sundial_of_the_infinite",
        scryfall_id: "36d3da9c-cb7a-4cea-b6e6-6722bd16c73c",
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
    }
}

fn warstorm_surge() -> CardDefinition {
    let text_box =
        "Whenever a creature you control enters, it deals damage equal to its power to any target.";
    CardDefinition {
        key: "m12.warstorm_surge",
        scryfall_id: "b16443df-52c6-4c9d-a7ff-89a37e593a0a",
        name: "Warstorm Surge".into(),
        mana_cost: Some(ManaCost::parse("{5}{R}")),
        types: CardTypes::new([CardType::Enchantment]),
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::AnotherYouControl(CardPredicate::creature()),
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
    }
}
