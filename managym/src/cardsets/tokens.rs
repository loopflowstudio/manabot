//! Every token definition (`token.*`). A token is an ordinary `CardDefinition`
//! with `is_token: true`, so copy effects treat tokens and cards alike. Each is
//! keyed by name and carries the Scryfall id of its first token printing.
use crate::state::{
    ability::{Ability, Effect, TriggerCondition, TriggerSubject},
    card::{
        ActivatedAbilityDefinition, BlockRestriction, CardDefinition, CardType, CardTypes,
        Keywords, StaticAbility,
    },
    mana::{Color, Colors, Mana, ManaCost},
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
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

/// A token's identity and printed type line; callers add the rest.
fn token(
    key: &'static str,
    scryfall_id: &'static str,
    name: &str,
    types: impl IntoIterator<Item = CardType>,
    subtypes: &[&str],
    text_box: &str,
) -> CardDefinition {
    CardDefinition {
        key,
        scryfall_id,
        name: name.into(),
        types: CardTypes::new(types),
        subtypes: subtypes.iter().map(|&subtype| subtype.into()).collect(),
        is_token: true,
        text_box: text_box.into(),
        ..Default::default()
    }
}

/// A creature token with printed power, toughness and colors.
fn creature(
    definition: CardDefinition,
    power: i32,
    toughness: i32,
    colors: impl IntoIterator<Item = Color>,
) -> CardDefinition {
    CardDefinition {
        power: Some(power),
        toughness: Some(toughness),
        color_override: Some(Colors::from_iter(colors)),
        ..definition
    }
}

fn ally() -> CardDefinition {
    let base = token(
        "token.ally",
        "f842c52a-04b6-4625-ba58-45b043ada340",
        "Ally",
        [CardType::Creature],
        &["Ally"],
        "",
    );
    creature(base, 1, 1, [Color::White])
}

fn ballistic_boulder() -> CardDefinition {
    let base = token(
        "token.ballistic_boulder",
        "b9d70342-9d3c-44cb-9026-db5e5941b9fc",
        "Ballistic Boulder",
        [CardType::Creature, CardType::Artifact],
        &["Construct"],
        "Flying",
    );
    CardDefinition {
        keywords: Keywords {
            flying: true,
            ..Default::default()
        },
        ..creature(base, 2, 1, [])
    }
}

fn bear() -> CardDefinition {
    let base = token(
        "token.bear",
        "ca3dae7d-3880-4c0a-acfb-8fd227cf9fab",
        "Bear",
        [CardType::Creature],
        &["Bear"],
        "",
    );
    creature(base, 4, 4, [Color::Green])
}

fn clue() -> CardDefinition {
    CardDefinition {
        activated_abilities: vec![ActivatedAbilityDefinition {
            mana_cost: ManaCost::parse("{2}"),
            sacrifice_source: true,
            effects: vec![Effect::DrawCards { count: 1 }],
            ..Default::default()
        }],
        ..token(
            "token.clue",
            "f2c859e1-181e-44d1-afbd-bbd6e52cf42a",
            "Clue",
            [CardType::Artifact],
            &["Clue"],
            "{2}, Sacrifice this token: Draw a card.",
        )
    }
}

fn dragon() -> CardDefinition {
    let text_box = "Flying, firebending 4 (Whenever this token attacks, add {R}{R}{R}{R} to your mana pool. This mana lasts until end of combat.)";
    let base = token(
        "token.dragon",
        "cddc0746-6d3e-441f-866f-28587bf54801",
        "Dragon",
        [CardType::Creature],
        &["Dragon"],
        text_box,
    );
    CardDefinition {
        keywords: Keywords {
            flying: true,
            ..Default::default()
        },
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::Attacks {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::AddMana {
                mana: Mana::parse("RRRR"),
                until_end_of_combat: true,
            }],
        }],
        ..creature(base, 4, 4, [Color::Red])
    }
}

fn food() -> CardDefinition {
    CardDefinition {
        activated_abilities: vec![ActivatedAbilityDefinition {
            tap_source: true,
            mana_cost: ManaCost::parse("{2}"),
            sacrifice_source: true,
            effects: vec![Effect::GainLife { amount: 3 }],
            ..Default::default()
        }],
        ..token(
            "token.food",
            "bf36408d-ed85-497f-8e68-d3a922c388a0",
            "Food",
            [CardType::Artifact],
            &["Food"],
            "{2}, {T}, Sacrifice this token: You gain 3 life.",
        )
    }
}

fn human_soldier() -> CardDefinition {
    let base = token(
        "token.human_soldier",
        "d9cbf36e-4044-4f08-9bae-f0dcb2455716",
        "Human Soldier",
        [CardType::Creature],
        &["Human", "Soldier"],
        "",
    );
    creature(base, 1, 1, [Color::White])
}

fn monk() -> CardDefinition {
    let text_box = "Prowess (Whenever you cast a noncreature spell, this creature gets +1/+1 until end of turn.)";
    let base = token(
        "token.monk",
        "388180bd-cc96-4b18-9ddd-18a3f7e06282",
        "Monk",
        [CardType::Creature],
        &["Monk"],
        text_box,
    );
    CardDefinition {
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::YouCastSpell {
                predicate: CardPredicate {
                    not_card_types: vec![CardType::Creature],
                    ..Default::default()
                },
            },
            effects: vec![Effect::ModifyUntilEot {
                power_delta: 1,
                toughness_delta: 1,
            }],
        }],
        ..creature(base, 1, 1, [Color::Red])
    }
}

fn soldier() -> CardDefinition {
    let text_box = "Firebending 1 (Whenever this token attacks, add {R}. This mana lasts until end of combat.)";
    let base = token(
        "token.soldier",
        "2de43b03-9ac5-4292-ab29-2dc6210ef3d9",
        "Soldier",
        [CardType::Creature],
        &["Soldier"],
        text_box,
    );
    CardDefinition {
        abilities: vec![Ability::Triggered {
            condition: TriggerCondition::Attacks {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::AddMana {
                mana: Mana::parse("R"),
                until_end_of_combat: true,
            }],
        }],
        ..creature(base, 2, 2, [Color::Red])
    }
}

fn spirit() -> CardDefinition {
    let text_box = "This token can't block or be blocked by non-Spirit creatures.";
    let base = token(
        "token.spirit",
        "f59eba51-458a-40e0-b754-999f91d5d839",
        "Spirit",
        [CardType::Creature],
        &["Spirit"],
        text_box,
    );
    let non_spirit = || CardPredicate {
        not_subtypes: vec!["Spirit".into()],
        ..Default::default()
    };
    CardDefinition {
        statics: vec![
            StaticAbility::BlockRestriction(BlockRestriction::CantBlock(non_spirit())),
            StaticAbility::BlockRestriction(BlockRestriction::CantBeBlockedBy(non_spirit())),
        ],
        ..creature(base, 1, 1, [])
    }
}

fn treasure() -> CardDefinition {
    let any_color = ["W", "U", "B", "R", "G"].map(|color| {
        vec![Effect::AddMana {
            mana: Mana::parse(color),
            until_end_of_combat: false,
        }]
    });
    CardDefinition {
        activated_abilities: vec![ActivatedAbilityDefinition {
            tap_source: true,
            sacrifice_source: true,
            effects: vec![Effect::Modal {
                modes: any_color.into(),
            }],
            ..Default::default()
        }],
        ..token(
            "token.treasure",
            "720f3e68-84c0-462e-a0d1-90236ccc494a",
            "Treasure",
            [CardType::Artifact],
            &["Treasure"],
            "{T}, Sacrifice this token: Add one mana of any color.",
        )
    }
}
