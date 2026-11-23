//! Marvel Super Heroes.
use managym_state::{
    card::{
        AdditionalCost, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics, StaticAbility,
    },
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![ironheart_clever_champion(), quicksilver_brash_blur()]
}

fn ironheart_clever_champion() -> CardDefinition {
    let text_box = "Improvise (Your artifacts can help cast this spell. Each artifact you tap after you're done activating mana abilities pays for {1}.)\nFlying\nNoncreature spells you cast have improvise.";
    CardDefinition {
        key: "msh.ironheart_clever_champion",
        scryfall_id: "395e477a-861f-4661-b329-6c1ad5343ed5",
        printed: PrintedCharacteristics {
            name: "Ironheart, Clever Champion".into(),
            mana_cost: Some(ManaCost::parse("{4}{U}")),
            types: CardTypes::new([CardType::Artifact, CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Human".into(), "Hero".into()],
            power: Some(3),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Flying]),
            additional_costs: vec![AdditionalCost::Improvise],
            statics: vec![StaticAbility::SpellsHave {
                predicate: CardPredicate {
                    not_card_types: vec![CardType::Creature],
                    ..Default::default()
                },
                cost: AdditionalCost::Improvise,
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn quicksilver_brash_blur() -> CardDefinition {
    use managym_state::{
        ability::{Effect, ObjectSelector},
        card::ActivatedAbilityDefinition,
        permanent::CounterKind,
    };
    CardDefinition {
        key: "msh.quicksilver_brash_blur",
        scryfall_id: "2d5819ca-165d-4f4c-9500-3ac206994880",
        printed: PrintedCharacteristics {
            name: "Quicksilver, Brash Blur".into(),
            mana_cost: Some(ManaCost::parse("{R}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Mutant".into(), "Hero".into()],
            power: Some(1), toughness: Some(1),
            keywords: Keywords::of(&[Keyword::Haste]),
            statics: vec![StaticAbility::BeginOnBattlefield],
            activated_abilities: vec![ActivatedAbilityDefinition {
                power_up: true, mana_cost: ManaCost::parse("{4}{R}"),
                effects: vec![Effect::put_counters(ObjectSelector::This, CounterKind::PlusOnePlusOne,1), Effect::put_counters(ObjectSelector::This,CounterKind::DoubleStrike,1)],
                ..Default::default()
            }],
            text_box: "If Quicksilver, Brash Blur is in your opening hand, you may begin the game with him on the battlefield.\nHaste\nPower-up — {4}{R}: Put a +1/+1 counter and a double strike counter on Quicksilver. (Activate each power-up ability only once. Reduce the cost by his mana cost if he entered this turn.)".into(),
            ..Default::default()
        }, ..Default::default()
    }
}
