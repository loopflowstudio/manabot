//! Every token definition (`token.*`). Tokens are not cards (CR 108.2,
//! 111.1): each is a [`TokenDefinition`] whose [`PrintedCharacteristics`] are
//! the same type a card prints, so copy effects treat tokens and cards alike. Effects
//! create a token by its key, and each carries the Scryfall id of its first
//! token printing, or `None` when it has none.
use managym_state::{
    ability::{
        Ability, Actor, Effect, EffectValue, ObjectSelector, Per, TriggerCondition, TriggerSubject,
    },
    card::{
        ActivatedAbilityDefinition, BlockRestriction, CardType, CardTypes, Keyword, Keywords,
        PrintedCharacteristics, StaticAbility, TokenDefinition,
    },
    effect::Modification,
    mana::{Color, Colors, Mana, ManaCost, ManaRestriction},
    predicate::CardPredicate,
};

pub(super) fn tokens() -> Vec<TokenDefinition> {
    vec![
        ally(),
        ballistic_boulder(),
        bear(),
        clue(),
        dragon(),
        elf_warrior(),
        food(),
        human(),
        human_soldier(),
        komas_coil(),
        marit_lage(),
        monk(),
        plain_spirit(),
        powerstone(),
        soldier(),
        spirit(),
        treasure(),
        zeppelin(),
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
                condition: TriggerCondition::attacks(TriggerSubject::This, Per::Object),
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

fn elf_warrior() -> TokenDefinition {
    let base = face("Elf Warrior", [CardType::Creature], &["Elf", "Warrior"], "");
    token(
        "token.elf_warrior",
        "49180301-f72f-48ca-8be1-df295e89cdd7",
        creature(base, 1, 1, [Color::Green, Color::White]),
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

fn human() -> TokenDefinition {
    let base = face("Human", [CardType::Creature], &["Human"], "");
    token(
        "token.human",
        "7d13a93a-a43d-4cf5-8300-8341f3b7f1b1",
        creature(base, 1, 1, [Color::White]),
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

fn komas_coil() -> TokenDefinition {
    let base = face("Koma's Coil", [CardType::Creature], &["Serpent"], "");
    token(
        "token.komas_coil",
        "54a1c6a9-3531-4432-9157-e4400dbc89fd",
        creature(base, 3, 3, [Color::Blue]),
    )
}

fn marit_lage() -> TokenDefinition {
    let base = face(
        "Marit Lage",
        [CardType::Creature],
        &["Avatar"],
        "Flying, indestructible",
    );
    token(
        "token.marit_lage",
        "8f5c3863-c876-48a8-9bc8-be20cd61074c",
        PrintedCharacteristics {
            supertypes: vec!["Legendary".into()],
            keywords: Keywords::of(&[Keyword::Flying, Keyword::Indestructible]),
            ..creature(base, 20, 20, [Color::Black])
        },
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
                condition: TriggerCondition::casts(
                    Actor::You,
                    CardPredicate {
                        not_card_types: vec![CardType::Creature],
                        ..Default::default()
                    },
                ),
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
                condition: TriggerCondition::attacks(TriggerSubject::This, Per::Object),
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

/// Cityscape Leveler's token (first token printing, BRO).
fn powerstone() -> TokenDefinition {
    token(
        "token.powerstone",
        "d45fe4b6-aeaf-4f84-b660-c7b482ed8512",
        PrintedCharacteristics {
            activated_abilities: vec![ActivatedAbilityDefinition {
                tap_source: true,
                effects: vec![Effect::add_mana(Mana::restricted_to(
                    ManaRestriction::ExceptNonartifactSpells,
                    [0, 0, 0, 0, 0, 1],
                ))],
                ..Default::default()
            }],
            ..face(
                "Powerstone",
                [CardType::Artifact],
                &["Powerstone"],
                "{T}: Add {C}. This mana can't be spent to cast a nonartifact spell.",
            )
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

/// Plain colorless Spirit, first printed in the 2004 player rewards tokens.
fn plain_spirit() -> TokenDefinition {
    let base = face("Spirit", [CardType::Creature], &["Spirit"], "");
    token(
        "token.plain_spirit",
        "46b60d95-b9bc-40f8-b986-bfa8e3eb74f3",
        creature(base, 1, 1, []),
    )
}

/// Lita's named Vehicle token has no printed token card.
fn zeppelin() -> TokenDefinition {
    TokenDefinition {
        key: "token.zeppelin",
        scryfall_id: None,
        printed: PrintedCharacteristics {
            power: Some(5),
            toughness: Some(5),
            color_override: Some(Colors::default()),
            keywords: Keywords::of(&[Keyword::Flying]),
            activated_abilities: vec![ActivatedAbilityDefinition {
                crew: Some(3),
                effects: vec![Effect::until_end_of_turn(
                    ObjectSelector::This,
                    Modification::BecomeCreature,
                )],
                ..Default::default()
            }],
            ..face(
                "Zeppelin",
                [CardType::Artifact],
                &["Vehicle"],
                "Flying\nCrew 3",
            )
        },
    }
}
