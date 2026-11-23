//! Tarkir: Dragonstorm.
use managym_state::{
    ability::{Ability, Actor, Effect, EffectValue, PlayerSelector, TriggerCondition},
    card::{
        AdditionalCost, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics, StaticAbility,
    },
    mana::ManaCost,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![teval_arbiter_of_virtue(), twinmaw_stormbrood()]
}

fn twinmaw_stormbrood() -> CardDefinition {
    use managym_state::ability::TriggerSubject;
    CardDefinition {
        key: "tdm.twinmaw_stormbrood",
        scryfall_id: "2999e3b1-6510-42b2-9429-28c07a64a44f",
        printed: PrintedCharacteristics {
            name: "Twinmaw Stormbrood".into(),
            mana_cost: Some(ManaCost::parse("{5}{W}")),
            types: CardTypes::new([CardType::Creature]),
            subtypes: vec!["Dragon".into()],
            power: Some(5),
            toughness: Some(4),
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::EntersTheBattlefield {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::gain_life(5)],
            }],
            text_box: "Flying\nWhen this creature enters, you gain 5 life.".into(),
            inset: Some(managym_state::card::InsetSpell::Omen(
                charring_bite().into(),
            )),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn charring_bite() -> PrintedCharacteristics {
    use managym_state::ability::{ObjectSelector, TargetSpec};
    let target = TargetSpec::Permanent {
        predicate: CardPredicate {
            flying: Some(false),
            ..CardPredicate::creature()
        },
    };
    PrintedCharacteristics {
        name: "Charring Bite".into(),
        mana_cost: Some(ManaCost::parse("{1}{R}")),
        types: CardTypes::new([CardType::Sorcery]),
        subtypes: vec!["Omen".into()],
        spell_effects: vec![Effect::deal_damage(5, ObjectSelector::Target(target))],
        text_box: "Charring Bite deals 5 damage to target creature without flying. (Then shuffle this card into its owner's library.)".into(),
        ..Default::default()
    }
}

fn teval_arbiter_of_virtue() -> CardDefinition {
    let text_box = "Flying, lifelink\nSpells you cast have delve. (Each card you exile from your graveyard while casting those spells pays for {1}.)\nWhenever you cast a spell, you lose life equal to its mana value.";
    CardDefinition {
        key: "tdm.teval_arbiter_of_virtue",
        scryfall_id: "27a93f5b-7b32-49f0-a179-b897828fe49a",
        printed: PrintedCharacteristics {
            name: "Teval, Arbiter of Virtue".into(),
            mana_cost: Some(ManaCost::parse("{2}{B}{G}{U}")),
            types: CardTypes::new([CardType::Creature]),
            supertypes: vec!["Legendary".into()],
            subtypes: vec!["Spirit".into(), "Dragon".into()],
            power: Some(6),
            toughness: Some(6),
            keywords: Keywords::of(&[Keyword::Flying, Keyword::Lifelink]),
            statics: vec![StaticAbility::SpellsHave {
                predicate: CardPredicate::default(),
                cost: AdditionalCost::Delve,
            }],
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::casts(Actor::You, CardPredicate::default()),
                effects: vec![Effect::LoseLife {
                    player: PlayerSelector::You,
                    amount: EffectValue::TriggeringManaValue,
                }],
            }],
            text_box: text_box.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}
