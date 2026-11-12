//! Every token definition (`token.*`). Tokens are not cards (CR 108.2,
//! 111.1): each is a [`TokenDefinition`] whose [`PrintedCharacteristics`] are
//! the same type a card prints, so copy effects treat tokens and cards alike. Effects
//! create a token by its key, and each carries the Scryfall id of its first
//! token printing, or `None` when it has none.
use managym_state::{
    ability::{Ability, Effect, EffectValue, ObjectSelector, TriggerCondition, TriggerSubject},
    card::{
        ActivatedAbilityDefinition, BlockRestriction, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics, StaticAbility, TokenDefinition,
    },
    effect::Modification,
    mana::{Color, Colors, Mana, ManaCost},
    predicate::CardPredicate,
};

pub(super) fn tokens() -> Vec<TokenDefinition> {
    vec![
        ally(),
        ballistic_boulder(),
        bear(),
        clue(),
        dragon(),
        food(),
        human_soldier(),
        monk(),
        soldier(),
        spirit(),
        treasure(),
    ]
}

/// A token's printed name, type line and text; callers add the rest.
fn face(
    name: &str,
    types: impl IntoIterator<Item = CardType>,
    subtypes: &[&str],
    text_box: &str,
) -> PrintedCharacteristics {
    PrintedCharacteristics {
        name: name.into(),
        types: CardTypes::new(types),
        subtypes: subtypes.iter().map(|&subtype| subtype.into()).collect(),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn token(
    key: &'static str,
    scryfall_id: &'static str,
    printed: PrintedCharacteristics,
) -> TokenDefinition {
    TokenDefinition {
        key,
        scryfall_id: Some(scryfall_id),
        printed,
    }
}

/// A creature token with printed power, toughness and colors.
fn creature(
    definition: PrintedCharacteristics,
    power: i32,
    toughness: i32,
    colors: impl IntoIterator<Item = Color>,
) -> PrintedCharacteristics {
    PrintedCharacteristics {
        power: Some(power),
        toughness: Some(toughness),
        color_override: Some(Colors::from_iter(colors)),
        ..definition
    }
}

fn ally() -> TokenDefinition {
    let base = face("Ally", [CardType::Creature], &["Ally"], "");
    token(
        "token.ally",
        "f842c52a-04b6-4625-ba58-45b043ada340",
        creature(base, 1, 1, [Color::White]),
    )
}

fn ballistic_boulder() -> TokenDefinition {
    let base = face(
        "Ballistic Boulder",
        [CardType::Creature, CardType::Artifact],
        &["Construct"],
        "Flying",
    );
    token(
        "token.ballistic_boulder",
        "b9d70342-9d3c-44cb-9026-db5e5941b9fc",
        PrintedCharacteristics {
            keywords: Keywords::of(&[Keyword::Flying]),
            ..creature(base, 2, 1, [])
        },
    )
}

fn bear() -> TokenDefinition {
    let base = face("Bear", [CardType::Creature], &["Bear"], "");
    token(
        "token.bear",
        "ca3dae7d-3880-4c0a-acfb-8fd227cf9fab",
        creature(base, 4, 4, [Color::Green]),
    )
}

fn clue() -> TokenDefinition {
    token(
        "token.clue",
        "f2c859e1-181e-44d1-afbd-bbd6e52cf42a",
        PrintedCharacteristics {
            activated_abilities: vec![ActivatedAbilityDefinition {
                mana_cost: ManaCost::parse("{2}"),
                sacrifice_source: true,
                effects: vec![Effect::DrawCards {
                    count: EffectValue::Fixed(1),
                }],
                ..Default::default()
            }],
            ..face(
                "Clue",
                [CardType::Artifact],
                &["Clue"],
                "{2}, Sacrifice this token: Draw a card.",
            )
        },
    )
}

fn dragon() -> TokenDefinition {
    let text_box = "Flying, firebending 4 (Whenever this token attacks, add {R}{R}{R}{R} to your mana pool. This mana lasts until end of combat.)";
    let base = face("Dragon", [CardType::Creature], &["Dragon"], text_box);
    token(
        "token.dragon",
        "cddc0746-6d3e-441f-866f-28587bf54801",
        PrintedCharacteristics {
            keywords: Keywords::of(&[Keyword::Flying]),
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::Attacks {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::AddMana {
                    mana: Mana::parse("RRRR"),
                    amount: EffectValue::Fixed(1),
                    until_end_of_combat: true,
                }],
            }],
            ..creature(base, 4, 4, [Color::Red])
        },
    )
}

fn food() -> TokenDefinition {
    token(
        "token.food",
        "bf36408d-ed85-497f-8e68-d3a922c388a0",
        PrintedCharacteristics {
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                mana_cost: ManaCost::parse("{2}"),
                sacrifice_source: true,
                effects: vec![Effect::gain_life(3)],
                ..Default::default()
            }],
            ..face(
                "Food",
                [CardType::Artifact],
                &["Food"],
                "{2}, {T}, Sacrifice this token: You gain 3 life.",
            )
        },
    )
}

fn human_soldier() -> TokenDefinition {
    let base = face(
        "Human Soldier",
        [CardType::Creature],
        &["Human", "Soldier"],
        "",
    );
    token(
        "token.human_soldier",
        "d9cbf36e-4044-4f08-9bae-f0dcb2455716",
        creature(base, 1, 1, [Color::White]),
    )
}

fn monk() -> TokenDefinition {
    let text_box = "Prowess (Whenever you cast a noncreature spell, this creature gets +1/+1 until end of turn.)";
    let base = face("Monk", [CardType::Creature], &["Monk"], text_box);
    token(
        "token.monk",
        "388180bd-cc96-4b18-9ddd-18a3f7e06282",
        PrintedCharacteristics {
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::YouCastSpell {
                    predicate: CardPredicate {
                        not_card_types: vec![CardType::Creature],
                        ..Default::default()
                    },
                },
                effects: vec![Effect::until_end_of_turn(
                    ObjectSelector::This,
                    Modification::pump(1, 1),
                )],
            }],
            ..creature(base, 1, 1, [Color::Red])
        },
    )
}

fn soldier() -> TokenDefinition {
    let text_box = "Firebending 1 (Whenever this token attacks, add {R}. This mana lasts until end of combat.)";
    let base = face("Soldier", [CardType::Creature], &["Soldier"], text_box);
    token(
        "token.soldier",
        "2de43b03-9ac5-4292-ab29-2dc6210ef3d9",
        PrintedCharacteristics {
            abilities: vec![Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::Attacks {
                    subject: TriggerSubject::This,
                },
                effects: vec![Effect::AddMana {
                    mana: Mana::parse("R"),
                    amount: EffectValue::Fixed(1),
                    until_end_of_combat: true,
                }],
            }],
            ..creature(base, 2, 2, [Color::Red])
        },
    )
}

fn spirit() -> TokenDefinition {
    let text_box = "This token can't block or be blocked by non-Spirit creatures.";
    let base = face("Spirit", [CardType::Creature], &["Spirit"], text_box);
    let non_spirit = || CardPredicate {
        not_subtypes: vec!["Spirit".into()],
        ..Default::default()
    };
    token(
        "token.spirit",
        "f59eba51-458a-40e0-b754-999f91d5d839",
        PrintedCharacteristics {
            statics: vec![
                StaticAbility::BlockRestriction(BlockRestriction::CantBlock(non_spirit())),
                StaticAbility::BlockRestriction(BlockRestriction::CantBeBlockedBy(non_spirit())),
            ],
            ..creature(base, 1, 1, [])
        },
    )
}

fn treasure() -> TokenDefinition {
    token(
        "token.treasure",
        "720f3e68-84c0-462e-a0d1-90236ccc494a",
        PrintedCharacteristics {
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                sacrifice_source: true,
                effects: vec![Effect::mana_choice(
                    ["W", "U", "B", "R", "G"].map(Mana::parse),
                )],
                ..Default::default()
            }],
            ..face(
                "Treasure",
                [CardType::Artifact],
                &["Treasure"],
                "{T}, Sacrifice this token: Add one mana of any color.",
            )
        },
    )
}
