//! TLE card behavior through ordinary choices and exact Command forks.
use crate::support::{
    game::{
        clear_summoning_sickness, five_color_setup, five_color_unfunded_setup, move_to_hand,
        Scenario,
    },
    inspect::{
        assert_choice_contract, can_activate, can_cast, card_of, counters, has_action, keywords,
        offered_cards, offered_targets, pt_modifications, tapped,
    },
    play::{
        activate, attack_with, cast_owned, choose, choose_mode, pass_until, resolve_everything,
        select_card, settle,
    },
};
use managym::{
    agent::{
        action::{Action, ActionSpaceKind},
        observation::Observation,
        observation_encoder::encode,
    },
    flow::turn::StepKind,
    state::{
        card::Keyword,
        game_object::{PermanentId, PlayerId, Target},
        zone::ZoneType,
    },
};

#[test]
fn tle_baseline_vanilla_creatures_cast_and_deal_printed_combat_damage() {
    for (name, power, toughness) in [
        ("Capital Guard", 2, 2),
        ("Kyoshi Warrior Guard", 2, 3),
        ("Warship Scout", 2, 1),
    ] {
        let mut s = five_color_setup(&[name]);
        let card = cast_owned(&mut s, name);
        settle(&mut s);
        let permanent = s.game().state.card_to_permanent[card].unwrap();
        let view = s.game().characteristics(permanent);
        assert_eq!((view.power, view.toughness), (power, toughness), "{name}");
        assert!(!tapped(&s, permanent), "{name}");
        clear_summoning_sickness(&mut s, permanent);
        let life = s.life(1);
        pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
        attack_with(&mut s, permanent);
        s.advance_to_step(StepKind::EndOfCombat);
        assert_eq!(s.life(1), life - power, "{name}");
    }
}

#[test]
fn tle_baseline_explosive_shot_targets_only_creatures_and_deals_four() {
    let mut s = five_color_setup(&["Explosive Shot", "Craw Wurm"]);
    let wurm = s.force_permanent_on_battlefield(1, "Craw Wurm");
    cast_owned(&mut s, "Explosive Shot");
    assert!(!offered_targets(&s).contains(&Target::Player(PlayerId(1))));
    assert!(s.choose_target(Target::Permanent(wurm)));
    settle(&mut s);
    assert!(s.game().state.permanents[wurm].is_none());
    s.assert_zone_size(1, ZoneType::Graveyard, 1);
    s.assert_life(1, 20);
}

#[test]
fn tle_baseline_zukos_offense_can_damage_player_or_creature() {
    for hit_player in [false, true] {
        let mut s = five_color_setup(&["Zuko's Offense", "Gray Ogre"]);
        let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
        cast_owned(&mut s, "Zuko's Offense");
        let targets = offered_targets(&s);
        assert!(targets.contains(&Target::Permanent(ogre)));
        assert!(targets.contains(&Target::Player(PlayerId(1))));
        assert!(s.choose_target(if hit_player {
            Target::Player(PlayerId(1))
        } else {
            Target::Permanent(ogre)
        }));
        settle(&mut s);
        s.assert_life(1, if hit_player { 18 } else { 20 });
        assert_eq!(s.game().state.permanents[ogre].is_some(), hit_player);
    }
}

#[test]
fn tle_baseline_kindly_customer_draws_on_entry() {
    let mut s = five_color_setup(&["Kindly Customer"]);
    cast_owned(&mut s, "Kindly Customer");
    let hand = s.zone_size(0, ZoneType::Hand);
    let library = s.zone_size(0, ZoneType::Library);
    settle(&mut s);
    s.assert_zone_size(0, ZoneType::Hand, hand + 1);
    s.assert_zone_size(0, ZoneType::Library, library - 1);
    assert_eq!(
        s.battlefield_permanents_named(0, "Kindly Customer").len(),
        1
    );
}

#[test]
fn tle_baseline_koala_sheep_gains_three_on_entry() {
    let mut s = five_color_setup(&["Koala-Sheep"]);
    let life = s.life(0);
    cast_owned(&mut s, "Koala-Sheep");
    settle(&mut s);
    s.assert_life(0, life + 3);
    s.assert_life(1, 20);
}

#[test]
fn tle_baseline_wolf_cove_villager_enters_tapped_and_untaps_normally() {
    let mut s = five_color_setup(&["Wolf Cove Villager"]);
    let card = cast_owned(&mut s, "Wolf Cove Villager");
    settle(&mut s);
    let permanent = s.game().state.card_to_permanent[card].unwrap();
    assert!(tapped(&s, permanent));
    s.advance_to_active_step(1, StepKind::Main);
    assert!(tapped(&s, permanent));
    s.advance_to_active_step(0, StepKind::Main);
    assert!(!tapped(&s, permanent));
}

#[test]
fn tle_baseline_keyword_creatures_have_printed_body_and_keyword() {
    for (name, power, toughness, keyword) in [
        ("Aardvark Sloth", 3, 3, Keyword::Lifelink),
        ("Deer-Dog", 1, 3, Keyword::FirstStrike),
        ("Dragon Moose", 3, 3, Keyword::Haste),
        ("Elephant-Rat", 1, 3, Keyword::Menace),
        ("Fire Nation Ambushers", 3, 2, Keyword::Flash),
        ("Fire Nation Soldier", 3, 2, Keyword::Haste),
        ("Flying Dolphin-Fish", 1, 3, Keyword::Flying),
        ("Frog-Squirrels", 2, 2, Keyword::Reach),
        ("Gilacorn", 1, 1, Keyword::Deathtouch),
        ("Hippo-Cows", 5, 4, Keyword::Trample),
        ("Komodo Rhino", 5, 2, Keyword::Trample),
        ("The Terror of Serpent's Pass", 8, 8, Keyword::Hexproof),
        ("Tundra Wall", 0, 4, Keyword::Defender),
        ("Turtle-Seals", 2, 4, Keyword::Vigilance),
        ("Zhao, the Seething Flame", 5, 5, Keyword::Menace),
        ("Sokka, Wolf Cove's Protector", 3, 3, Keyword::Vigilance),
    ] {
        let mut s = five_color_setup(&[name]);
        let card = cast_owned(&mut s, name);
        settle(&mut s);
        let permanent = s.game().state.card_to_permanent[card].unwrap();
        let view = s.game().characteristics(permanent);
        assert_eq!((view.power, view.toughness), (power, toughness), "{name}");
        assert!(view.keywords.has(keyword), "{name}");
    }
}

#[test]
fn tle_baseline_dragon_moose_attacks_the_turn_it_enters() {
    let mut s = five_color_setup(&["Dragon Moose"]);
    let card = cast_owned(&mut s, "Dragon Moose");
    settle(&mut s);
    let moose = s.game().state.card_to_permanent[card].unwrap();
    pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
    attack_with(&mut s, moose);
    s.advance_to_step(StepKind::EndOfCombat);
    s.assert_life(1, 17);
}

#[test]
fn tle_baseline_tundra_wall_cannot_attack() {
    let mut s = five_color_setup(&["Tundra Wall", "Gray Ogre"]);
    let wall = s.force_permanent_on_battlefield(0, "Tundra Wall");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    clear_summoning_sickness(&mut s, wall);
    clear_summoning_sickness(&mut s, ogre);
    s.game_mut().scenario_refresh_priority().unwrap();
    pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
    assert!(!has_action(&s, |action| matches!(
        action,
        Action::DeclareAttacker { permanent, attack: true, .. } if *permanent == wall
    )));
    assert!(has_action(&s, |action| matches!(
        action,
        Action::DeclareAttacker { permanent, attack: true, .. } if *permanent == ogre
    )));
}

#[test]
fn tle_baseline_fire_nation_ambushers_casts_on_the_opponents_turn() {
    let mut s = five_color_setup(&["Fire Nation Ambushers", "Gray Ogre"]);
    let ambushers = move_to_hand(&mut s, "Fire Nation Ambushers");
    s.advance_to_active_step(1, StepKind::Upkeep);
    s.set_player_mana_pool(0, "2B");
    while s.action_space().player != Some(PlayerId(0))
        || s.action_space().kind != ActionSpaceKind::Priority
    {
        s.advance_default_action();
    }
    assert!(can_cast(&s, ambushers));
}

#[test]
fn tle_baseline_aang_air_nomad_gives_other_creatures_vigilance() {
    let mut s = five_color_setup(&["Aang, Air Nomad", "Gray Ogre"]);
    let aang = s.force_permanent_on_battlefield(0, "Aang, Air Nomad");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let foe = s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(keywords(&s, ogre).has(Keyword::Vigilance));
    assert!(keywords(&s, aang).has(Keyword::Flying));
    assert!(!keywords(&s, foe).has(Keyword::Vigilance));
    clear_summoning_sickness(&mut s, ogre);
    pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
    attack_with(&mut s, ogre);
    s.advance_to_step(StepKind::EndOfCombat);
    assert!(!tapped(&s, ogre));
}

#[test]
fn tle_baseline_iroh_pumps_every_attacking_creature() {
    let mut s = five_color_setup(&["Iroh, Firebending Instructor", "Gray Ogre"]);
    let iroh = s.force_permanent_on_battlefield(0, "Iroh, Firebending Instructor");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let idle = s.force_permanent_on_battlefield(1, "Gray Ogre");
    clear_summoning_sickness(&mut s, iroh);
    clear_summoning_sickness(&mut s, ogre);
    s.game_mut().scenario_refresh_priority().unwrap();
    pass_until(&mut s, ActionSpaceKind::DeclareAttacker);
    while s.action_space().kind == ActionSpaceKind::DeclareAttacker {
        choose(&mut s, |action| {
            matches!(action, Action::DeclareAttacker { permanent, attack, .. }
                if *attack == (*permanent == iroh || *permanent == ogre))
        });
    }
    resolve_everything(&mut s);
    assert_eq!(pt_modifications(&s, iroh), (1, 1));
    assert_eq!(pt_modifications(&s, ogre), (1, 1));
    assert_eq!(pt_modifications(&s, idle), (0, 0));
}

#[test]
fn tle_baseline_katara_heroic_healer_counters_each_other_creature_you_control() {
    let mut s = five_color_setup(&["Katara, Heroic Healer", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let foe = s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.game_mut().scenario_refresh_priority().unwrap();
    let card = cast_owned(&mut s, "Katara, Heroic Healer");
    settle(&mut s);
    let katara = s.game().state.card_to_permanent[card].unwrap();
    assert_eq!(counters(&s, ogre), 1);
    assert_eq!(counters(&s, katara), 0);
    assert_eq!(counters(&s, foe), 0);
    assert!(keywords(&s, katara).has(Keyword::Lifelink));
}

#[test]
fn tle_baseline_sledding_otter_penguin_buys_counters_repeatedly() {
    let mut s = five_color_setup(&["Sledding Otter-Penguin"]);
    let otter = s.force_permanent_on_battlefield(0, "Sledding Otter-Penguin");
    s.game_mut().scenario_refresh_priority().unwrap();
    for expected in 1..=2 {
        activate(&mut s, otter, 0);
        settle(&mut s);
        assert_eq!(counters(&s, otter), expected);
    }
    let view = s.game().characteristics(otter);
    assert_eq!((view.power, view.toughness), (4, 5));
}

#[test]
fn tle_baseline_toucan_puffin_pumps_only_your_creature() {
    let mut s = five_color_setup(&["Toucan-Puffin", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let foe = s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.game_mut().scenario_refresh_priority().unwrap();
    cast_owned(&mut s, "Toucan-Puffin");
    for _ in 0..4 {
        if !offered_targets(&s).is_empty() {
            break;
        }
        s.pass_priority();
    }
    let targets = offered_targets(&s);
    assert!(targets.contains(&Target::Permanent(ogre)));
    assert!(!targets.contains(&Target::Permanent(foe)));
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    assert_eq!(pt_modifications(&s, ogre), (2, 0));
}

#[test]
fn tle_baseline_fire_nation_s_conquest_pumps_your_creatures() {
    let mut s = five_color_setup(&["Fire Nation's Conquest", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let foe = s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.game_mut().scenario_refresh_priority().unwrap();
    cast_owned(&mut s, "Fire Nation's Conquest");
    settle(&mut s);
    let view = s.game().characteristics(ogre);
    assert_eq!((view.power, view.toughness), (3, 2));
    let view = s.game().characteristics(foe);
    assert_eq!((view.power, view.toughness), (2, 2));
}

#[test]
fn tle_baseline_allied_teamwork_makes_and_pumps_allies() {
    let mut s = five_color_setup(&["Allied Teamwork", "Kyoshi Warrior Guard", "Gray Ogre"]);
    let guard = s.force_permanent_on_battlefield(0, "Kyoshi Warrior Guard");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.game_mut().scenario_refresh_priority().unwrap();
    cast_owned(&mut s, "Allied Teamwork");
    settle(&mut s);
    let tokens = s.battlefield_permanents_named(0, "Ally");
    assert_eq!(tokens.len(), 1);
    let token = s.game().characteristics(tokens[0]);
    assert_eq!((token.power, token.toughness), (2, 2));
    let view = s.game().characteristics(guard);
    assert_eq!((view.power, view.toughness), (3, 4));
    let view = s.game().characteristics(ogre);
    assert_eq!((view.power, view.toughness), (2, 2));
}

#[test]
fn tle_baseline_sokka_s_sword_training_pumps_and_makes_a_clue() {
    let mut s = five_color_setup(&["Sokka's Sword Training", "Gray Ogre"]);
    let foe = s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.game_mut().scenario_refresh_priority().unwrap();
    cast_owned(&mut s, "Sokka's Sword Training");
    assert!(s.choose_target(Target::Permanent(foe)));
    settle(&mut s);
    assert_eq!(pt_modifications(&s, foe), (2, 2));
    assert_eq!(s.battlefield_permanents_named(0, "Clue").len(), 1);
}

fn merchant(foods: usize) -> (Scenario, PermanentId, Vec<PermanentId>) {
    let mut s =
        five_color_unfunded_setup(&["The Cabbage Merchant", "Prophetic Prism", "Gray Ogre"]);
    let source = s.force_permanent_on_battlefield(0, "The Cabbage Merchant");
    let foods = create_foods(&mut s, foods);
    s.game_mut().scenario_refresh_priority().unwrap();
    (s, source, foods)
}

fn create_foods(s: &mut Scenario, count: usize) -> Vec<PermanentId> {
    (0..count)
        .map(|_| {
            s.game_mut()
                .create_token("Food", PlayerId(0), false)
                .unwrap()
        })
        .collect()
}

#[test]
fn merchant_repeats_with_four_foods_and_preserves_exact_tap_commands() {
    let (mut s, source, foods) = merchant(4);
    // This ability has no source {T} cost; sickness and a tapped Merchant do not prevent it.
    s.game_mut().state.permanents[source]
        .as_mut()
        .unwrap()
        .tapped = true;
    let prism = move_to_hand(&mut s, "Prophetic Prism");
    assert!(
        can_cast(&s, prism),
        "the payment needs two uses of the same Merchant"
    );
    for (activation, pair) in foods.chunks_exact(2).enumerate() {
        assert!(can_activate(&s, source));
        activate(&mut s, source, 0);
        assert_eq!(s.action_space().kind, ActionSpaceKind::LookAndSelect);
        for food in pair {
            assert_choice_contract(&s);
            let card = card_of(&s, *food);
            assert!(offered_cards(&s).contains(&card));
            select_card(&mut s, card);
            assert!(tapped(&s, *food));
            assert!(!offered_cards(&s).contains(&card));
        }
        assert_eq!(s.action_space().kind, ActionSpaceKind::Modal);
        assert_choice_contract(&s);
        choose_mode(&mut s, 4);
        assert_eq!(
            s.game().state.players[0].mana_pool.total(),
            (activation + 1) as u16
        );
    }
    assert!(!can_activate(&s, source));
    assert!(can_cast(&s, prism));
}

#[test]
fn merchant_three_foods_fund_only_one_activation() {
    let (mut s, source, foods) = merchant(3);
    let prism = move_to_hand(&mut s, "Prophetic Prism");
    assert!(!can_cast(&s, prism));
    activate(&mut s, source, 0);
    for food in &foods[..2] {
        let card = card_of(&s, *food);
        select_card(&mut s, card);
    }
    choose_mode(&mut s, 4);
    assert!(!can_activate(&s, source));
    assert!(!can_cast(&s, prism));
    assert!(!tapped(&s, foods[2]));
}

#[test]
fn merchant_cannot_reuse_the_food_reserved_for_its_own_activation() {
    let (s, _, foods) = merchant(4);
    assert!(
        !can_activate(&s, foods[0]),
        "its own tap leaves only three Foods for mana"
    );

    let (mut s, source, foods) = merchant(5);
    let payer = foods[0];
    let payer_card = card_of(&s, payer);
    assert!(can_activate(&s, payer));
    activate(&mut s, payer, 0);
    for pair in foods[1..].chunks_exact(2) {
        activate(&mut s, source, 0);
        for food in pair {
            assert!(!offered_cards(&s).contains(&payer_card));
            assert_choice_contract(&s);
            let card = card_of(&s, *food);
            select_card(&mut s, card);
        }
        choose_mode(&mut s, 4);
    }
    choose(&mut s, |action| matches!(action, Action::PayCost { .. }));
    settle(&mut s);
    assert_eq!(s.game().state.players[0].life, 23);
    assert!(s.game().state.permanents[payer].is_none());
    assert!(foods[1..].iter().all(|food| tapped(&s, *food)));
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
}

#[test]
fn merchant_tap_count_reaches_the_acting_viewer_and_policy() {
    let (mut s, source, foods) = merchant(2);
    activate(&mut s, source, 0);
    for (index, food) in foods.iter().enumerate() {
        for viewer in [PlayerId(0), PlayerId(1)] {
            let observation = Observation::for_player(s.game(), viewer);
            if viewer == PlayerId(0) {
                let pending = observation.continuation.last().unwrap();
                assert_eq!(pending.mode, Some(2 - index));
            } else {
                assert!(observation.continuation.is_empty());
                assert!(observation.action_space.actions.is_empty());
            }
            encode(&observation, &Default::default()).unwrap();
        }
        let card = card_of(&s, *food);
        select_card(&mut s, card);
    }
    assert_eq!(s.action_space().kind, ActionSpaceKind::Modal);
}

#[test]
fn merchant_creates_food_only_for_an_opponents_noncreature_spell() {
    let (mut s, _, _) = merchant(0);
    s.force_permanent_on_battlefield(1, "The Cabbage Merchant");
    s.set_player_mana_pool(0, "10RRR");
    cast_owned(&mut s, "Gray Ogre");
    settle(&mut s);
    assert!(s.battlefield_permanents_named(0, "Food").is_empty());
    assert!(s.battlefield_permanents_named(1, "Food").is_empty());
    cast_owned(&mut s, "Prophetic Prism");
    settle(&mut s);
    assert!(s.battlefield_permanents_named(0, "Food").is_empty());
    assert_eq!(s.battlefield_permanents_named(1, "Food").len(), 1);
}

#[test]
fn merchant_sacrifices_a_food_for_each_creature_dealing_combat_damage_to_you() {
    let (mut s, _, foods) = merchant(2);
    let enemy_merchant = s.force_permanent_on_battlefield(1, "The Cabbage Merchant");
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    clear_summoning_sickness(&mut s, enemy_merchant);
    clear_summoning_sickness(&mut s, ogre);
    s.advance_to_active_step(1, StepKind::DeclareAttackers);
    while s.action_space().kind == ActionSpaceKind::DeclareAttacker {
        choose(&mut s, |action| {
            matches!(action, Action::DeclareAttacker { attack: true, .. })
        });
    }
    s.advance_to_active_step(1, StepKind::DeclareBlockers);
    while s.action_space().kind == ActionSpaceKind::DeclareBlocker {
        s.decline_block();
    }
    for food in foods {
        pass_until(&mut s, ActionSpaceKind::LookAndSelect);
        assert_choice_contract(&s);
        let card = card_of(&s, food);
        select_card(&mut s, card);
    }
    settle(&mut s);
    assert!(s.battlefield_permanents_named(0, "Food").is_empty());
    assert_eq!(s.game().state.players[0].life, 16);
}

#[test]
fn merchant_paid_foods_are_not_offered_again_for_waterbending() {
    let mut s = five_color_unfunded_setup(&["The Cabbage Merchant", "Waterbending Lesson"]);
    let source = s.force_permanent_on_battlefield(0, "The Cabbage Merchant");
    let foods = create_foods(&mut s, 3);
    s.set_player_mana_pool(0, "3U");
    cast_owned(&mut s, "Waterbending Lesson");
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    s.set_player_mana_pool(0, "0");
    pass_until(&mut s, ActionSpaceKind::Waterbend);
    activate(&mut s, source, 0);
    for food in &foods[..2] {
        let card = card_of(&s, *food);
        select_card(&mut s, card);
    }
    choose_mode(&mut s, 4);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Waterbend);
    assert!(!s.action_space().actions.iter().any(
        |a| matches!(a, Action::WaterbendTap { permanent, .. } if foods[..2].contains(permanent))
    ));
    choose(
        &mut s,
        |a| matches!(a, Action::WaterbendTap { permanent, .. } if *permanent == foods[2]),
    );
    assert_choice_contract(&s);
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    assert!(foods.iter().all(|food| tapped(&s, *food)));
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
}
