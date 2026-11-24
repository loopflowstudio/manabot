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
    vec![
        ironheart_clever_champion(),
        quicksilver_brash_blur(),
        construct_a_cosmic_cube(),
    ]
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
                predicate: CardPredicate::noncreature(),
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

fn construct_a_cosmic_cube() -> CardDefinition {
    use managym_state::{
        ability::{
            Ability, Actor, Effect, ObjectSelector, PlayerControlDuration, TargetSpec,
            TriggerCondition,
        },
        permanent::CounterKind,
    };
    let text_box = "Whenever you draw your second card each turn, create a 2/1 black Villain creature token with menace and put a plan counter on this enchantment.\nWhen the seventh plan counter is put on this enchantment, sacrifice it. When you do, you control target opponent during their next turn. (You see all cards that player could see and make all decisions for them.)";
    CardDefinition {
        key: "msh.construct_a_cosmic_cube",
        scryfall_id: "448de757-ac16-4529-b851-1a1331b821a5",
        printed: PrintedCharacteristics {
            name: "Construct a Cosmic Cube".into(),
            mana_cost: Some(ManaCost::parse("{2}{B}")),
            types: CardTypes::new([CardType::Enchantment]),
            subtypes: vec!["Plan".into()],
            abilities: vec![
                Ability::Triggered {
                    intervening_if: None,
                    condition: TriggerCondition::Draws {
                        who: Actor::You,
                        nth: Some(2),
                    },
                    effects: vec![
                        Effect::create_tokens("token.villain", 1),
                        Effect::put_counters(ObjectSelector::This, CounterKind::Plan, 1),
                    ],
                },
                Ability::Triggered {
                    intervening_if: None,
                    condition: TriggerCondition::NthCounter {
                        kind: CounterKind::Plan,
                        nth: 7,
                    },
                    effects: vec![
                        Effect::sacrifice_this(),
                        Effect::if_you_do(vec![Effect::WhenYouDo {
                            effects: vec![Effect::ControlPlayer {
                                what: ObjectSelector::Target(TargetSpec::Opponent),
                                duration: PlayerControlDuration::NextTurn,
                            }],
                        }]),
                    ],
                },
            ],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
