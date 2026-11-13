//! TLE card behavior through ordinary choices and exact Command forks.
use crate::support::{
    game::{
        clear_summoning_sickness, enter_battlefield, five_color_setup, five_color_unfunded_setup,
        move_from_library, move_to_hand, plant_in_graveyard, Scenario,
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
        mana::{Mana, ManaClass, ManaSpent, SpendEffect},
        stack_object::StackObject,
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
                .create_token("token.food", PlayerId(0), false)
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
    for (activation, pair) in foods.as_chunks::<2>().0.iter().enumerate() {
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
    for pair in foods[1..].as_chunks::<2>().0 {
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

// ---------------------------------------------------------------------------
// Mana spent to cast: Toph, Uncle's Musings, Animal Attendant
// ---------------------------------------------------------------------------

/// Player 0 has exactly `mana` floating and no untapped permanent.
fn only_pool(s: &mut Scenario, mana: &str) {
    for permanent in s.game_mut().state.permanents.iter_mut().flatten() {
        permanent.tapped = true;
    }
    s.set_player_mana_pool(0, mana);
}

fn spell_spent(s: &Scenario, card: managym::state::game_object::CardId) -> ManaSpent {
    s.game()
        .state
        .stack_objects
        .iter()
        .find_map(|object| match object {
            StackObject::Spell(spell) if spell.card == card => Some(spell.spent),
            _ => None,
        })
        .expect("spell on the stack")
}

/// Resolve everything, choosing `land` whenever a target is asked for.
fn settle_targeting(s: &mut Scenario, land: PermanentId) {
    for _ in 0..50 {
        if s.action_space().kind == ActionSpaceKind::ChooseTarget {
            assert!(s.choose_target(Target::Permanent(land)));
        } else if s.action_space().kind == ActionSpaceKind::Priority
            && s.game().state.stack_objects.is_empty()
            && s.game().state.pending_triggers.is_empty()
        {
            return;
        } else {
            s.pass_priority();
        }
    }
    panic!("did not settle");
}

/// Toph's X is the mana her spell's payment took (CR 601.2h), carried from
/// the stack to her own entering trigger; the land she earthbends is a land
/// creature, so it has double strike.
#[test]
fn toph_earthbends_the_amount_of_mana_spent_to_cast_her() {
    let mut s = five_color_unfunded_setup(&["Toph, Greatest Earthbender"]);
    let land = s.force_permanent_on_battlefield(0, "Forest");
    only_pool(&mut s, "RGGG");
    let toph = cast_owned(&mut s, "Toph, Greatest Earthbender");
    assert_eq!(spell_spent(&s, toph).amounts, [0, 0, 0, 1, 3, 0]);
    assert_eq!(spell_spent(&s, toph).total(), 4);
    settle_targeting(&mut s, land);
    assert_eq!(counters(&s, land), 4);
    let view = s.game().characteristics(land);
    assert!(view.is_creature());
    assert!(view.keywords.has(Keyword::DoubleStrike));
    assert!(s.game().state.players[0].mana_pool.total() == 0);
}

/// A Toph that was not cast spent no mana: her trigger's X is 0.
#[test]
fn toph_put_onto_the_battlefield_earthbends_zero() {
    let mut s = five_color_unfunded_setup(&["Toph, Greatest Earthbender"]);
    s.force_permanent_on_battlefield(0, "Forest");
    let toph = move_from_library(
        &mut s,
        0,
        "Toph, Greatest Earthbender",
        ZoneType::Battlefield,
    );
    s.game_mut().scenario_refresh_priority().unwrap();
    let permanent = s.game().state.card_to_permanent[toph].unwrap();
    assert!(s.game().state.permanents[permanent]
        .as_ref()
        .unwrap()
        .cast_spent
        .is_empty());
    assert!(s
        .game()
        .state
        .pending_triggers
        .iter()
        .all(|trigger| trigger.spent.is_empty()));
}

/// Converge counts the colors of the mana actually spent: the same printed
/// cost returns up to three, one or one permanent cards as the generic part
/// is paid with two other colors, with green, or with colorless.
#[test]
fn uncles_musings_counts_the_colors_of_mana_spent_and_exiles_itself() {
    for (mana, colors) in [("WUGG", 3), ("GGGG", 1), ("2GG", 1), ("RRGG", 2)] {
        let mut s = five_color_unfunded_setup(&["Uncle's Musings", "Craw Wurm", "Explosive Shot"]);
        let permanents = [
            plant_in_graveyard(&mut s, 0, "Craw Wurm"),
            plant_in_graveyard(&mut s, 0, "Forest"),
            plant_in_graveyard(&mut s, 0, "Island"),
        ];
        let shot = plant_in_graveyard(&mut s, 0, "Explosive Shot");
        only_pool(&mut s, mana);
        let musings = cast_owned(&mut s, "Uncle's Musings");
        assert_eq!(spell_spent(&s, musings).colors(), colors, "{mana}");
        pass_until(&mut s, ActionSpaceKind::LookAndSelect);
        let offered = offered_cards(&s);
        assert!(!offered.contains(&shot), "{mana}: only permanent cards");
        assert!(permanents.iter().all(|card| offered.contains(card)));
        let mut returned = 0;
        for card in permanents {
            if s.action_space().kind != ActionSpaceKind::LookAndSelect
                || !offered_cards(&s).contains(&card)
            {
                break;
            }
            select_card(&mut s, card);
            returned += 1;
        }
        settle(&mut s);
        assert_eq!(returned, colors as usize, "{mana}");
        let in_hand = permanents
            .iter()
            .filter(|card| s.game().state.zones.zone_of(**card) == Some(ZoneType::Hand))
            .count();
        assert_eq!(in_hand, colors as usize, "{mana}");
        assert_eq!(
            s.game().state.zones.zone_of(musings),
            Some(ZoneType::Exile),
            "{mana}"
        );
    }
}

fn attendant_class() -> ManaClass {
    SpendEffect::CreatureEntersWithPlusOneCounter {
        without_subtype: "Human".into(),
    }
    .into()
}

/// A ready Animal Attendant tapped for green; the rest of the pool is `mana`.
fn attendant_mana(s: &mut Scenario, mana: &str, attendants: usize) -> Vec<PermanentId> {
    only_pool(s, mana);
    (0..attendants)
        .map(|_| {
            let attendant = enter_battlefield(s, "Animal Attendant");
            clear_summoning_sickness(s, attendant);
            activate(s, attendant, 0);
            choose_mode(s, 4);
            attendant
        })
        .collect()
}

/// Each Attendant mana spent on a non-Human creature spell adds a counter to
/// the creature it becomes; the mana keeps its rider in the pool until then.
#[test]
fn animal_attendant_mana_adds_a_counter_to_a_non_human_creature_spell() {
    for attendants in [1, 2] {
        let mut s = five_color_unfunded_setup(&["Animal Attendant", "Koala-Sheep"]);
        attendant_mana(&mut s, if attendants == 1 { "1W" } else { "W" }, attendants);
        let pool = &s.game().state.players[0].mana_pool;
        assert_eq!(
            pool.class_of(&attendant_class()),
            [0, 0, 0, 0, attendants as u8, 0]
        );
        let sheep = cast_owned(&mut s, "Koala-Sheep");
        let spent = spell_spent(&s, sheep);
        assert_eq!(spent.entry_counters, attendants as u8);
        assert_eq!(spent.total(), 3);
        settle(&mut s);
        let permanent = s.game().state.card_to_permanent[sheep].unwrap();
        assert_eq!(counters(&s, permanent), attendants as i32);
        assert!(s.game().state.players[0].mana_pool.classes().is_empty());
    }
}

/// The rider is captured in the mana: it applies after the Attendant has
/// left, and does nothing for a Human creature or a noncreature spell.
#[test]
fn animal_attendant_mana_needs_no_attendant_and_skips_humans_and_noncreatures() {
    let mut s = five_color_unfunded_setup(&["Animal Attendant", "Koala-Sheep"]);
    let attendant = attendant_mana(&mut s, "1W", 1)[0];
    let card = card_of(&s, attendant);
    s.game_mut().move_card(card, ZoneType::Graveyard);
    let sheep = cast_owned(&mut s, "Koala-Sheep");
    settle(&mut s);
    let permanent = s.game().state.card_to_permanent[sheep].unwrap();
    assert_eq!(counters(&s, permanent), 1);

    let mut s = five_color_unfunded_setup(&["Animal Attendant", "Capital Guard"]);
    attendant_mana(&mut s, "R", 1);
    let guard = cast_owned(&mut s, "Capital Guard");
    assert_eq!(spell_spent(&s, guard).entry_counters, 0);
    settle(&mut s);
    let permanent = s.game().state.card_to_permanent[guard].unwrap();
    assert_eq!(counters(&s, permanent), 0);

    let mut s = five_color_unfunded_setup(&["Animal Attendant", "Explosive Shot", "Craw Wurm"]);
    s.force_permanent_on_battlefield(1, "Craw Wurm");
    attendant_mana(&mut s, "R", 1);
    let shot = cast_owned(&mut s, "Explosive Shot");
    if s.action_space().kind == ActionSpaceKind::ChooseTarget {
        s.step_action(0);
    }
    assert_eq!(spell_spent(&s, shot).entry_counters, 0);
}

/// Mana whose rider would be wasted is spent after ordinary mana of its
/// color, so it stays for a spell it helps.
#[test]
fn animal_attendant_mana_is_kept_when_ordinary_mana_pays_a_human() {
    let mut s = five_color_unfunded_setup(&["Animal Attendant", "Capital Guard"]);
    attendant_mana(&mut s, "RG", 1);
    cast_owned(&mut s, "Capital Guard");
    let pool = &s.game().state.players[0].mana_pool;
    assert_eq!(pool.class_of(&attendant_class()), [0, 0, 0, 0, 1, 0]);
    assert_eq!(pool.mana, [0; 6]);
}

/// The counter belongs to the spell that was cast: the same card returning
/// to the battlefield later, without being cast, enters with none.
#[test]
fn animal_attendant_counter_does_not_follow_the_card_to_a_later_entry() {
    let mut s = five_color_unfunded_setup(&["Animal Attendant", "Koala-Sheep"]);
    attendant_mana(&mut s, "1W", 1);
    let sheep = cast_owned(&mut s, "Koala-Sheep");
    settle(&mut s);
    let permanent = s.game().state.card_to_permanent[sheep].unwrap();
    assert_eq!(counters(&s, permanent), 1);

    s.game_mut().move_card(sheep, ZoneType::Graveyard);
    s.game_mut().move_card(sheep, ZoneType::Battlefield);
    let returned = s.game().state.card_to_permanent[sheep].unwrap();
    assert_eq!(counters(&s, returned), 0);
    assert!(s.game().state.permanents[returned]
        .as_ref()
        .unwrap()
        .cast_spent
        .is_empty());
}

/// Until-end-of-combat mana keeps its class: Attendant mana in the combat
/// pool pays first within its class and still adds its counter.
#[test]
fn animal_attendant_mana_in_the_combat_pool_keeps_its_spend_effect() {
    let mut s = five_color_unfunded_setup(&["Animal Attendant", "Koala-Sheep"]);
    only_pool(&mut s, "1W");
    s.game_mut().state.players[0].combat_mana_pool =
        Mana::of_class(attendant_class(), [0, 0, 0, 0, 1, 0]);
    s.game_mut().state.players[0]
        .mana_pool
        .add(&Mana::of_class(attendant_class(), [0, 0, 0, 0, 1, 0]));
    let sheep = cast_owned(&mut s, "Koala-Sheep");
    assert_eq!(spell_spent(&s, sheep).entry_counters, 1);
    let player = &s.game().state.players[0];
    assert_eq!(player.combat_mana_pool.total(), 0);
    assert_eq!(
        player.mana_pool.class_of(&attendant_class()),
        [0, 0, 0, 0, 1, 0]
    );
}
