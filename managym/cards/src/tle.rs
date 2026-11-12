//! Avatar: The Last Airbender Eternal.
use managym_state::{
    ability::{
        Ability, Actor, Effect, EffectValue, ObjectSelector, Per, Scope, TargetRequirement,
        TargetSpec, TriggerCondition, TriggerSubject,
    },
    card::{
        ActivatedAbilityDefinition, CardDefinition, CardType, CardTypes, Keyword, Keywords,
        ReplacementEffect, StaticAbility, StaticKeywordGrant, StaticPtBuff, StaticScope, TapCost,
    },
    effect::Modification,
    mana::{Mana, ManaCost},
    permanent::CounterKind,
    predicate::CardPredicate,
};

pub(super) fn cards() -> Vec<CardDefinition> {
    vec![
        aang_air_nomad(),
        aardvark_sloth(),
        allied_teamwork(),
        capital_guard(),
        deer_dog(),
        dragon_moose(),
        elephant_rat(),
        explosive_shot(),
        fire_nation_ambushers(),
        fire_nation_s_conquest(),
        fire_nation_soldier(),
        flying_dolphin_fish(),
        frog_squirrels(),
        gilacorn(),
        hippo_cows(),
        iroh_firebending_instructor(),
        katara_heroic_healer(),
        kindly_customer(),
        koala_sheep(),
        komodo_rhino(),
        kyoshi_warrior_guard(),
        sledding_otter_penguin(),
        sokka_s_sword_training(),
        sokka_wolf_cove_s_protector(),
        the_cabbage_merchant(),
        the_terror_of_serpent_s_pass(),
        toucan_puffin(),
        tundra_wall(),
        turtle_seals(),
        warship_scout(),
        wolf_cove_villager(),
        zhao_the_seething_flame(),
        zukos_offense(),
    ]
}

fn the_cabbage_merchant() -> CardDefinition {
    let text_box = "Whenever an opponent casts a noncreature spell, create a Food token. (It's an artifact with \"{2}, {T}, Sacrifice this token: You gain 3 life.\")\nWhenever a creature deals combat damage to you, sacrifice a Food token.\nTap two untapped Foods you control: Add one mana of any color.";
    CardDefinition {
        key: "tle.the_cabbage_merchant",
        scryfall_id: "2fea0356-6684-4730-9eb4-0262856bc1f9",
        name: "The Cabbage Merchant".into(),
        mana_cost: Some(ManaCost::parse("{2}{G}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Citizen".into()],
        supertypes: vec!["Legendary".into()],
        power: Some(2),
        toughness: Some(2),
        abilities: vec![
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::Casts {
                    who: Actor::Opponent,
                    predicate: CardPredicate {
                        not_card_types: vec![CardType::Creature],
                        ..Default::default()
                    },
                    nth: None,
                },
                effects: vec![Effect::create_tokens("Food", 1)],
            },
            Ability::Triggered {
                intervening_if: None,
                condition: TriggerCondition::DealsCombatDamage {
                    subject: TriggerSubject::Any(CardPredicate::creature()),
                    to: Actor::You,
                    per: Per::Object,
                },
                effects: vec![Effect::Sacrifice {
                    optional: false,
                    what: ObjectSelector::chosen(CardPredicate {
                        subtype: Some("Food".into()),
                        token: Some(true),
                        ..Default::default()
                    }),
                }],
            },
        ],
        activated_abilities: vec![ActivatedAbilityDefinition {
            tap_cost: Some(TapCost {
                predicate: CardPredicate::subtype("Food"),
                count: 2,
            }),
            effects: vec![Effect::mana_choice(
                ["W", "U", "B", "R", "G"].map(Mana::parse),
            )],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn capital_guard() -> CardDefinition {
    let text_box = "";
    CardDefinition {
        key: "tle.capital_guard",
        scryfall_id: "91cbed11-3b5c-4e7a-9b13-125c1fe5f22f",
        name: "Capital Guard".into(),
        mana_cost: Some(ManaCost::parse("{1}{R}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Soldier".into()],
        power: Some(2),
        toughness: Some(2),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn explosive_shot() -> CardDefinition {
    let text_box = "Explosive Shot deals 4 damage to target creature.";
    CardDefinition {
        key: "tle.explosive_shot",
        scryfall_id: "4ba682a6-1045-4ed6-ae95-bd5af8bd7224",
        name: "Explosive Shot".into(),
        mana_cost: Some(ManaCost::parse("{1}{R}")),
        types: CardTypes::new([CardType::Sorcery]),
        spell_effects: vec![Effect::deal_damage(
            4,
            ObjectSelector::Target(TargetSpec::Creature),
        )],
        targeting: vec![TargetRequirement::one(TargetSpec::Creature)],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn kindly_customer() -> CardDefinition {
    let text_box = "When this creature enters, draw a card.";
    CardDefinition {
        key: "tle.kindly_customer",
        scryfall_id: "84332812-367a-4ac4-9be5-2adc57562c9d",
        name: "Kindly Customer".into(),
        mana_cost: Some(ManaCost::parse("{1}{W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Citizen".into()],
        power: Some(1),
        toughness: Some(1),
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::DrawCards {
                count: EffectValue::Fixed(1),
            }],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn koala_sheep() -> CardDefinition {
    let text_box = "When this creature enters, you gain 3 life.";
    CardDefinition {
        key: "tle.koala_sheep",
        scryfall_id: "7785ddfe-f8ae-473a-8064-549884c5aabc",
        name: "Koala-Sheep".into(),
        mana_cost: Some(ManaCost::parse("{2}{W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Bear".into(), "Sheep".into()],
        power: Some(3),
        toughness: Some(2),
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::gain_life(3)],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn kyoshi_warrior_guard() -> CardDefinition {
    let text_box = "";
    CardDefinition {
        key: "tle.kyoshi_warrior_guard",
        scryfall_id: "7a2acb28-c0e2-492b-a227-ef49e905e0fb",
        name: "Kyoshi Warrior Guard".into(),
        mana_cost: Some(ManaCost::parse("{1}{W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Warrior".into(), "Ally".into()],
        power: Some(2),
        toughness: Some(3),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn warship_scout() -> CardDefinition {
    let text_box = "";
    CardDefinition {
        key: "tle.warship_scout",
        scryfall_id: "b1a95982-be16-465a-9c1b-1f4d875c0c40",
        name: "Warship Scout".into(),
        mana_cost: Some(ManaCost::parse("{R}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Scout".into()],
        power: Some(2),
        toughness: Some(1),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn wolf_cove_villager() -> CardDefinition {
    let text_box = "This creature enters tapped.";
    CardDefinition {
        key: "tle.wolf_cove_villager",
        scryfall_id: "7dbeced9-d27d-476c-92d4-3c14d8a40458",
        name: "Wolf Cove Villager".into(),
        mana_cost: Some(ManaCost::parse("{W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Peasant".into()],
        power: Some(2),
        toughness: Some(2),
        replacement_effects: vec![ReplacementEffect::EntersTapped],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn zukos_offense() -> CardDefinition {
    let text_box = "Zuko's Offense deals 2 damage to any target.";
    CardDefinition {
        key: "tle.zukos_offense",
        scryfall_id: "efc6eea1-9872-47e5-a3bc-3c8c1b9705be",
        name: "Zuko's Offense".into(),
        mana_cost: Some(ManaCost::parse("{R}")),
        types: CardTypes::new([CardType::Sorcery]),
        spell_effects: vec![Effect::deal_damage(
            2,
            ObjectSelector::Target(TargetSpec::AnyTarget),
        )],
        targeting: vec![TargetRequirement::one(TargetSpec::AnyTarget)],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn aardvark_sloth() -> CardDefinition {
    let text_box =
        "Lifelink (Damage dealt by this creature also causes you to gain that much life.)";
    CardDefinition {
        key: "tle.aardvark_sloth",
        scryfall_id: "e16365a2-4969-4ad5-af95-9dd2d0499f06",
        name: "Aardvark Sloth".into(),
        mana_cost: Some(ManaCost::parse("{3}{W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Sloth".into(), "Beast".into()],
        power: Some(3),
        toughness: Some(3),
        keywords: Keywords::of(&[Keyword::Lifelink]),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn deer_dog() -> CardDefinition {
    let text_box =
        "First strike (This creature deals combat damage before creatures without first strike.)";
    CardDefinition {
        key: "tle.deer_dog",
        scryfall_id: "54ec5233-6b27-4655-b68c-4c1aaf64d89b",
        name: "Deer-Dog".into(),
        mana_cost: Some(ManaCost::parse("{1}{R}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Elk".into(), "Dog".into()],
        power: Some(1),
        toughness: Some(3),
        keywords: Keywords::of(&[Keyword::FirstStrike]),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn dragon_moose() -> CardDefinition {
    let text_box =
        "Haste (This creature can attack and {T} as soon as it comes under your control.)";
    CardDefinition {
        key: "tle.dragon_moose",
        scryfall_id: "fd61f469-5377-4bc1-a42b-dd66ca4ede1d",
        name: "Dragon Moose".into(),
        mana_cost: Some(ManaCost::parse("{3}{R}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Dragon".into(), "Elk".into()],
        power: Some(3),
        toughness: Some(3),
        keywords: Keywords::of(&[Keyword::Haste]),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn elephant_rat() -> CardDefinition {
    let text_box = "Menace (This creature can't be blocked except by two or more creatures.)";
    CardDefinition {
        key: "tle.elephant_rat",
        scryfall_id: "0568f28f-e830-4a7a-91d3-43e78bb86c8f",
        name: "Elephant-Rat".into(),
        mana_cost: Some(ManaCost::parse("{1}{B}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Elephant".into(), "Rat".into()],
        power: Some(1),
        toughness: Some(3),
        keywords: Keywords::of(&[Keyword::Menace]),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn fire_nation_ambushers() -> CardDefinition {
    let text_box = "Flash (You may cast this spell any time you could cast an instant.)";
    CardDefinition {
        key: "tle.fire_nation_ambushers",
        scryfall_id: "a20aaa6c-7032-43c1-9e7c-57cf8c8ecfd4",
        name: "Fire Nation Ambushers".into(),
        mana_cost: Some(ManaCost::parse("{2}{B}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Soldier".into()],
        power: Some(3),
        toughness: Some(2),
        keywords: Keywords::of(&[Keyword::Flash]),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn fire_nation_soldier() -> CardDefinition {
    let text_box =
        "Haste (This creature can attack and {T} as soon as it comes under your control.)";
    CardDefinition {
        key: "tle.fire_nation_soldier",
        scryfall_id: "58e59eba-3cf5-4ebe-8bc1-28568921650b",
        name: "Fire Nation Soldier".into(),
        mana_cost: Some(ManaCost::parse("{2}{R}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Soldier".into()],
        power: Some(3),
        toughness: Some(2),
        keywords: Keywords::of(&[Keyword::Haste]),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn flying_dolphin_fish() -> CardDefinition {
    let text_box =
        "Flying (This creature can't be blocked except by creatures with flying or reach.)";
    CardDefinition {
        key: "tle.flying_dolphin_fish",
        scryfall_id: "ccb61a51-cff6-4f88-ac17-f10852a27d06",
        name: "Flying Dolphin-Fish".into(),
        mana_cost: Some(ManaCost::parse("{1}{U}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Whale".into(), "Fish".into()],
        power: Some(1),
        toughness: Some(3),
        keywords: Keywords::of(&[Keyword::Flying]),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn frog_squirrels() -> CardDefinition {
    let text_box = "Reach (This creature can block creatures with flying.)";
    CardDefinition {
        key: "tle.frog_squirrels",
        scryfall_id: "73f4c4b7-5c05-48a5-adb2-ca850df8cd03",
        name: "Frog-Squirrels".into(),
        mana_cost: Some(ManaCost::parse("{1}{G}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Frog".into(), "Squirrel".into()],
        power: Some(2),
        toughness: Some(2),
        keywords: Keywords::of(&[Keyword::Reach]),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn gilacorn() -> CardDefinition {
    let text_box =
        "Deathtouch (Any amount of damage this deals to a creature is enough to destroy it.)";
    CardDefinition {
        key: "tle.gilacorn",
        scryfall_id: "b7f90eef-2393-4e44-9895-00489d3eaa92",
        name: "Gilacorn".into(),
        mana_cost: Some(ManaCost::parse("{B}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Lizard".into()],
        power: Some(1),
        toughness: Some(1),
        keywords: Keywords::of(&[Keyword::Deathtouch]),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn hippo_cows() -> CardDefinition {
    let text_box =
        "Trample (This creature can deal excess combat damage to the player it's attacking.)";
    CardDefinition {
        key: "tle.hippo_cows",
        scryfall_id: "24ca6ca8-e239-4fe5-8711-24098b74e2fe",
        name: "Hippo-Cows".into(),
        mana_cost: Some(ManaCost::parse("{4}{G}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Hippo".into(), "Ox".into()],
        power: Some(5),
        toughness: Some(4),
        keywords: Keywords::of(&[Keyword::Trample]),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn komodo_rhino() -> CardDefinition {
    let text_box =
        "Trample (This creature can deal excess combat damage to the player it's attacking.)";
    CardDefinition {
        key: "tle.komodo_rhino",
        scryfall_id: "e2b84af1-b40b-476a-87b2-ee8e3a018171",
        name: "Komodo Rhino".into(),
        mana_cost: Some(ManaCost::parse("{3}{R}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Lizard".into(), "Rhino".into()],
        power: Some(5),
        toughness: Some(2),
        keywords: Keywords::of(&[Keyword::Trample]),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn the_terror_of_serpent_s_pass() -> CardDefinition {
    let text_box = "Hexproof (This creature can't be the target of spells or abilities your opponents control.)";
    CardDefinition {
        key: "tle.the_terror_of_serpent_s_pass",
        scryfall_id: "f7569861-5b68-4265-bb45-3d54992bfe66",
        name: "The Terror of Serpent's Pass".into(),
        mana_cost: Some(ManaCost::parse("{5}{U}{U}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Serpent".into()],
        supertypes: vec!["Legendary".into()],
        power: Some(8),
        toughness: Some(8),
        keywords: Keywords::of(&[Keyword::Hexproof]),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn tundra_wall() -> CardDefinition {
    let text_box = "Defender (This creature can't attack.)";
    CardDefinition {
        key: "tle.tundra_wall",
        scryfall_id: "49ae7f36-4b37-415e-ab39-5fa1e3f2e3e9",
        name: "Tundra Wall".into(),
        mana_cost: Some(ManaCost::parse("{1}{W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Wall".into()],
        power: Some(0),
        toughness: Some(4),
        keywords: Keywords::of(&[Keyword::Defender]),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn turtle_seals() -> CardDefinition {
    let text_box = "Vigilance (Attacking doesn't cause this creature to tap.)";
    CardDefinition {
        key: "tle.turtle_seals",
        scryfall_id: "3a144e7f-e0f7-4346-a4e8-f4a358c3b4a9",
        name: "Turtle-Seals".into(),
        mana_cost: Some(ManaCost::parse("{3}{U}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Turtle".into(), "Seal".into()],
        power: Some(2),
        toughness: Some(4),
        keywords: Keywords::of(&[Keyword::Vigilance]),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn zhao_the_seething_flame() -> CardDefinition {
    let text_box = "Menace (This creature can't be blocked except by two or more creatures.)";
    CardDefinition {
        key: "tle.zhao_the_seething_flame",
        scryfall_id: "f3fa493e-b58a-4b77-88e8-b8890eefb2b7",
        name: "Zhao, the Seething Flame".into(),
        mana_cost: Some(ManaCost::parse("{4}{R}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Soldier".into()],
        supertypes: vec!["Legendary".into()],
        power: Some(5),
        toughness: Some(5),
        keywords: Keywords::of(&[Keyword::Menace]),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn sokka_wolf_cove_s_protector() -> CardDefinition {
    let text_box = "Vigilance (Attacking doesn't cause this creature to tap.)";
    CardDefinition {
        key: "tle.sokka_wolf_cove_s_protector",
        scryfall_id: "039cb105-f8f7-4d04-a137-34b13491ee9a",
        name: "Sokka, Wolf Cove's Protector".into(),
        mana_cost: Some(ManaCost::parse("{2}{W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Warrior".into(), "Ally".into()],
        supertypes: vec!["Legendary".into()],
        power: Some(3),
        toughness: Some(3),
        keywords: Keywords::of(&[Keyword::Vigilance]),
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn aang_air_nomad() -> CardDefinition {
    let text_box = "Flying (This creature can't be blocked except by creatures with flying or reach.)\nVigilance (Attacking doesn't cause this creature to tap.)\nOther creatures you control have vigilance.";
    CardDefinition {
        key: "tle.aang_air_nomad",
        scryfall_id: "f369827d-e4cd-4bc7-8c5e-72882eff0908",
        name: "Aang, Air Nomad".into(),
        mana_cost: Some(ManaCost::parse("{3}{W}{W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Avatar".into(), "Ally".into()],
        supertypes: vec!["Legendary".into()],
        power: Some(5),
        toughness: Some(4),
        keywords: Keywords::of(&[Keyword::Flying, Keyword::Vigilance]),
        statics: vec![StaticAbility::KeywordGrant(StaticKeywordGrant {
            scope: StaticScope::OtherYouControl(CardPredicate::creature()),
            condition: None,
            keywords: Keywords::of(&[Keyword::Vigilance]),
        })],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn iroh_firebending_instructor() -> CardDefinition {
    let text_box = "Whenever Iroh attacks, attacking creatures get +1/+1 until end of turn.";
    CardDefinition {
        key: "tle.iroh_firebending_instructor",
        scryfall_id: "1ee4bc92-1101-4652-be27-65450abebf00",
        name: "Iroh, Firebending Instructor".into(),
        mana_cost: Some(ManaCost::parse("{2}{R}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Noble".into(), "Ally".into()],
        supertypes: vec!["Legendary".into()],
        power: Some(2),
        toughness: Some(2),
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::Attacks {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::until_end_of_turn(
                ObjectSelector::each(
                    Scope::Battlefield,
                    CardPredicate {
                        attacking: true,
                        ..CardPredicate::creature()
                    },
                ),
                Modification::pump(1, 1),
            )],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn katara_heroic_healer() -> CardDefinition {
    let text_box = "Lifelink (Damage dealt by this creature also causes you to gain that much life.)\nWhen Katara enters, put a +1/+1 counter on each other creature you control.";
    CardDefinition {
        key: "tle.katara_heroic_healer",
        scryfall_id: "bc2ec498-54eb-41a7-bdf2-8772430e177f",
        name: "Katara, Heroic Healer".into(),
        mana_cost: Some(ManaCost::parse("{4}{W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Human".into(), "Warrior".into(), "Ally".into()],
        supertypes: vec!["Legendary".into()],
        power: Some(2),
        toughness: Some(3),
        keywords: Keywords::of(&[Keyword::Lifelink]),
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::put_counters(
                ObjectSelector::each_other_you_control(CardPredicate::creature()),
                CounterKind::PlusOnePlusOne,
                1,
            )],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn sledding_otter_penguin() -> CardDefinition {
    let text_box = "{3}: Put a +1/+1 counter on this creature.";
    CardDefinition {
        key: "tle.sledding_otter_penguin",
        scryfall_id: "d3e0f159-029d-4177-8f20-c96eca0a421c",
        name: "Sledding Otter-Penguin".into(),
        mana_cost: Some(ManaCost::parse("{2}{W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Otter".into(), "Bird".into()],
        power: Some(2),
        toughness: Some(3),
        activated_abilities: vec![ActivatedAbilityDefinition {
            mana_cost: ManaCost::parse("{3}"),
            effects: vec![Effect::put_counters(
                ObjectSelector::This,
                CounterKind::PlusOnePlusOne,
                1,
            )],
            ..Default::default()
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn toucan_puffin() -> CardDefinition {
    let text_box = "Flying\nWhen this creature enters, target creature you control gets +2/+0 until end of turn.";
    CardDefinition {
        key: "tle.toucan_puffin",
        scryfall_id: "10dc44c7-634f-4210-9c22-ce9e8860568b",
        name: "Toucan-Puffin".into(),
        mana_cost: Some(ManaCost::parse("{2}{W}")),
        types: CardTypes::new([CardType::Creature]),
        subtypes: vec!["Bird".into()],
        power: Some(2),
        toughness: Some(2),
        keywords: Keywords::of(&[Keyword::Flying]),
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::until_end_of_turn(
                ObjectSelector::Target(TargetSpec::CreatureYouControl),
                Modification::pump(2, 0),
            )],
        }],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn fire_nation_s_conquest() -> CardDefinition {
    let text_box = "Creatures you control get +1/+0.";
    CardDefinition {
        key: "tle.fire_nation_s_conquest",
        scryfall_id: "b6fcdc4c-547a-4df4-93c8-911fd400c837",
        name: "Fire Nation's Conquest".into(),
        mana_cost: Some(ManaCost::parse("{2}{R}")),
        types: CardTypes::new([CardType::Enchantment]),
        statics: vec![StaticAbility::PtBuff(StaticPtBuff {
            scope: StaticScope::YouControl(CardPredicate::creature()),
            condition: None,
            power: 1,
            toughness: 0,
            per_creature_type_max: None,
            per_color_among: None,
            per_graveyard_matching: None,
        })],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn allied_teamwork() -> CardDefinition {
    let text_box = "When this enchantment enters, create a 1/1 white Ally creature token.\nAllies you control get +1/+1.";
    CardDefinition {
        key: "tle.allied_teamwork",
        scryfall_id: "8a347283-eb16-4339-945e-316a5ce9e2c5",
        name: "Allied Teamwork".into(),
        mana_cost: Some(ManaCost::parse("{2}{W}")),
        types: CardTypes::new([CardType::Enchantment]),
        abilities: vec![Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::create_tokens("Ally", 1)],
        }],
        statics: vec![StaticAbility::PtBuff(StaticPtBuff {
            scope: StaticScope::YouControl(CardPredicate::subtype("Ally")),
            condition: None,
            power: 1,
            toughness: 1,
            per_creature_type_max: None,
            per_color_among: None,
            per_graveyard_matching: None,
        })],
        text_box: text_box.into(),
        ..Default::default()
    }
}

fn sokka_s_sword_training() -> CardDefinition {
    let text_box = "Target creature gets +2/+2 until end of turn.\nCreate a Clue token. (It's an artifact with \"{2}, Sacrifice this token: Draw a card.\")";
    CardDefinition {
        key: "tle.sokka_s_sword_training",
        scryfall_id: "3489930a-208d-4bb2-b200-58a0dd4f6e81",
        name: "Sokka's Sword Training".into(),
        mana_cost: Some(ManaCost::parse("{1}{W}")),
        types: CardTypes::new([CardType::Instant]),
        subtypes: vec!["Lesson".into()],
        spell_effects: vec![
            Effect::until_end_of_turn(
                ObjectSelector::Target(TargetSpec::Creature),
                Modification::pump(2, 2),
            ),
            Effect::create_tokens("Clue", 1),
        ],
        targeting: vec![TargetRequirement::one(TargetSpec::Creature)],
        text_box: text_box.into(),
        ..Default::default()
    }
}
