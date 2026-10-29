//! Avatar: The Last Airbender (`src/cardsets/tla.rs`). Each test drives
//! legal actions through the same typed programs as the world pool.
use crate::{
    helpers::Scenario,
    support::{
        activate, activate_card, can_activate, can_activate_any, can_activate_card, can_cast,
        card_of, cast, cast_card, choose, counters, counters_of, has_action, library_card, mode,
        offered_targets, pass_until, play_land, ready, settle, setup, setup_unfunded, tapped,
        to_hand,
    },
};
use managym::{
    agent::{
        action::{Action, ActionSpaceKind, ActionType},
        observation::{EventEntityKind, KeywordData, Observation, PermanentData},
        policy_history::HistoryReference,
    },
    flow::{event::GameEvent, turn::StepKind},
    state::{
        card::Keywords,
        game_object::{CardId, PermanentId, PlayerId, Target},
        mana::{Mana, ManaCost, ManaUse},
        permanent::CounterKind,
        zone::ZoneType,
    },
};

fn is_land_creature(s: &Scenario, id: PermanentId) -> bool {
    s.game().state.permanents[id].as_ref().unwrap().animated
}

/// Earthbend N: only lands you control are offered; the land becomes a 0/0
/// creature that is still a land, with N +1/+1 counters.
#[test]
fn earthbending_lesson_animates_only_an_own_land_with_four_counters() {
    let mut s = setup(&["Earthbending Lesson"]);
    let forest = s.force_permanent_on_battlefield(0, "Forest");
    let opposing = s.force_permanent_on_battlefield(1, "Forest");
    cast(&mut s, "Earthbending Lesson");
    let targets = offered_targets(&s);
    assert!(targets.contains(&Target::Permanent(forest)));
    assert!(!targets.contains(&Target::Permanent(opposing)));
    assert!(s.choose_target(Target::Permanent(forest)));
    settle(&mut s);
    assert!(is_land_creature(&s, forest));
    assert_eq!(counters(&s, forest), 4);
    assert_eq!(s.game().effective_pt(forest), (4, 4));
    let land = s.game().state.permanents[forest].as_ref().unwrap().card;
    assert_eq!(
        s.game().state.zones.zone_of(land),
        Some(ZoneType::Battlefield)
    );
}

/// Modes choose distinct targets; the destroy mode accepts only creatures with
/// power 4 or greater, judged on current power.
#[test]
fn sandbenders_storm_destroys_large_creatures_or_earthbends_three() {
    let mut s = setup(&["Sandbenders' Storm", "Gray Ogre", "Craw Wurm"]);
    let small = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let large = s.force_permanent_on_battlefield(1, "Craw Wurm");
    cast(&mut s, "Sandbenders' Storm");
    mode(&mut s, 0);
    let targets = offered_targets(&s);
    assert!(targets.contains(&Target::Permanent(large)));
    assert!(!targets.contains(&Target::Permanent(small)));
    assert!(s.choose_target(Target::Permanent(large)));
    settle(&mut s);
    assert!(s.game().state.permanents[large].is_none());
    assert!(s.game().state.permanents[small].is_some());

    // A pumped Ogre has power 4 and becomes a legal destroy target.
    s.game_mut().state.permanents[small]
        .as_mut()
        .unwrap()
        .plus1_counters = 2;
    cast(&mut s, "Sandbenders' Storm");
    mode(&mut s, 0);
    assert!(offered_targets(&s).contains(&Target::Permanent(small)));
    assert!(s.choose_target(Target::Permanent(small)));
    settle(&mut s);
    assert!(s.game().state.permanents[small].is_none());

    let plains = s.force_permanent_on_battlefield(0, "Plains");
    cast(&mut s, "Sandbenders' Storm");
    mode(&mut s, 1);
    assert!(s.choose_target(Target::Permanent(plains)));
    settle(&mut s);
    assert!(is_land_creature(&s, plains));
    assert_eq!(counters(&s, plains), 3);
}

/// A dies trigger still earthbends after its source left the battlefield.
#[test]
fn earth_village_ruffians_earthbends_two_when_it_dies() {
    let mut s = setup(&["Earth Village Ruffians", "Lightning Strike"]);
    let ruffians = s.force_permanent_on_battlefield(0, "Earth Village Ruffians");
    let swamp = s.force_permanent_on_battlefield(0, "Swamp");
    cast(&mut s, "Lightning Strike");
    assert!(s.choose_target(Target::Permanent(ruffians)));
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.game().state.permanents[ruffians].is_none());
    assert_eq!(offered_targets(&s), vec![Target::Permanent(swamp)]);
    assert!(s.choose_target(Target::Permanent(swamp)));
    settle(&mut s);
    assert!(is_land_creature(&s, swamp));
    assert_eq!(counters(&s, swamp), 2);
}

/// Hybrid {B/G} is payable with either color.
#[test]
fn earth_village_ruffians_hybrid_cost_accepts_black_or_green() {
    for pool in ["2B", "2G"] {
        let mut s = setup(&["Earth Village Ruffians"]);
        s.set_player_mana_pool(0, pool);
        cast(&mut s, "Earth Village Ruffians");
        assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
        settle(&mut s);
        assert_eq!(
            s.battlefield_permanents_named(0, "Earth Village Ruffians")
                .len(),
            1
        );
    }
}

/// Waterbend {4} may be paid with creature taps (which make no mana); the
/// resolution draws first, then requires a discard from the new hand.
#[test]
fn geyser_leaper_waterbends_to_draw_then_discard() {
    let mut s = setup(&["Geyser Leaper", "Gray Ogre"]);
    let leaper = s.force_permanent_on_battlefield(0, "Geyser Leaper");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.set_player_mana_pool(0, "3");
    assert!(can_activate_any(&mut s, leaper));
    activate(&mut s, leaper, 0);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Waterbend);
    assert!(s.choose_waterbend_tap(ogre));
    assert!(s.take_action_by_type(ActionType::PayCost));
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
    assert!(s.game().state.permanents[ogre].as_ref().unwrap().tapped);
    let library = s.zone_size(0, ZoneType::Library);
    pass_until(&mut s, ActionSpaceKind::Discard);
    assert_eq!(s.zone_size(0, ZoneType::Library), library - 1);
    assert_eq!(s.zone_size(0, ZoneType::Hand), 1);
    s.step_action(0);
    assert_eq!(s.zone_size(0, ZoneType::Hand), 0);
    assert_eq!(s.zone_size(0, ZoneType::Graveyard), 1);
    assert!(s.game().state.stack_objects.is_empty());
}

/// Activation is restricted to its controller's turn; blockers with current
/// power 2 or less cannot block it.
#[test]
fn foggy_swamp_vinebender_grows_on_own_turn_and_evades_small_blockers() {
    let mut s = setup(&["Foggy Swamp Vinebender", "Gray Ogre", "Hill Giant"]);
    let vinebender = s.force_permanent_on_battlefield(0, "Foggy Swamp Vinebender");
    ready(&mut s, vinebender);
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let pumped = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let giant = s.force_permanent_on_battlefield(1, "Hill Giant");
    s.game_mut().state.permanents[pumped]
        .as_mut()
        .unwrap()
        .plus1_counters = 1;

    s.set_player_mana_pool(0, "5");
    activate(&mut s, vinebender, 0);
    if s.action_space().kind == ActionSpaceKind::Waterbend {
        assert!(s.take_action_by_type(ActionType::PayCost));
    }
    settle(&mut s);
    assert_eq!(counters(&s, vinebender), 1);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    choose(
        &mut s,
        |a| matches!(a, Action::DeclareAttacker { permanent, attack: true, .. } if *permanent == vinebender),
    );
    for _ in 0..10 {
        if s.action_space().kind == ActionSpaceKind::DeclareBlocker {
            break;
        }
        s.advance_default_action();
    }
    let can_block = |s: &Scenario, blocker: PermanentId| {
        s.action_space().actions.iter().any(|a| {
            matches!(a, Action::DeclareBlocker { blocker: b, attacker: Some(at), .. }
            if *b == blocker && *at == vinebender)
        })
    };
    // Blocker declaration offers one blocker at a time; read every offer.
    let mut seen = Vec::new();
    for _ in 0..3 {
        assert_eq!(s.action_space().kind, ActionSpaceKind::DeclareBlocker);
        for id in [ogre, pumped, giant] {
            if can_block(&s, id) {
                seen.push(id);
            }
        }
        let current = s.action_space().actions.iter().find_map(|a| match a {
            Action::DeclareBlocker { blocker, .. } => Some(*blocker),
            _ => None,
        });
        choose(&mut s, |a| {
            matches!(a, Action::DeclareBlocker { attacker: None, .. })
        });
        if current.is_none() || s.action_space().kind != ActionSpaceKind::DeclareBlocker {
            break;
        }
    }
    assert!(!seen.contains(&ogre));
    assert!(seen.contains(&pumped));
    assert!(seen.contains(&giant));

    // Not on the opponent's turn, even while holding priority there.
    s.advance_to_active_step(1, StepKind::Main);
    s.pass_priority();
    assert_eq!(s.game().active_player(), PlayerId(1));
    assert_eq!(s.action_space().player, Some(PlayerId(0)));
    s.set_player_mana_pool(0, "5");
    assert!(!can_activate_any(&mut s, vinebender));
}

/// Printed flashback (CR 702.34): the graveyard copy costs {8}{R}, may be cast
/// at instant speed on either turn, and is exiled as it resolves.
#[test]
fn fire_nation_attacks_makes_firebending_soldiers_and_flashes_back_once() {
    let mut s = setup(&["Fire Nation Attacks"]);
    s.set_player_mana_pool(0, "4R");
    let spell = cast(&mut s, "Fire Nation Attacks");
    settle(&mut s);
    let soldiers = s.battlefield_permanents_named(0, "Soldier");
    assert_eq!(soldiers.len(), 2);
    assert_eq!(s.game().effective_pt(soldiers[0]), (2, 2));
    assert_eq!(
        s.game().state.zones.zone_of(spell),
        Some(ZoneType::Graveyard)
    );

    // {4}{R} pays the printed cost but not the flashback cost.
    s.set_player_mana_pool(0, "4R");
    assert!(!can_cast(&mut s, spell));

    // On the opponent's turn, with priority, the instant flashes back.
    s.advance_to_active_step(1, StepKind::Main);
    s.pass_priority();
    assert_eq!(s.action_space().player, Some(PlayerId(0)));
    s.set_player_mana_pool(0, "8R");
    assert!(can_cast(&mut s, spell));
    cast_card(&mut s, spell);
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Soldier").len(), 4);
    assert_eq!(s.game().state.zones.zone_of(spell), Some(ZoneType::Exile));
}

/// Soldier tokens attack for firebending 1: one red mana until end of combat.
#[test]
fn fire_nation_attacks_soldiers_add_red_when_attacking() {
    let mut s = setup(&["Fire Nation Attacks"]);
    cast(&mut s, "Fire Nation Attacks");
    settle(&mut s);
    let soldiers = s.battlefield_permanents_named(0, "Soldier");
    for soldier in &soldiers {
        ready(&mut s, *soldier);
    }
    s.set_player_mana_pool(0, "");
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    // Attackers are offered one at a time: attack with the first, hold the other.
    choose(&mut s, |a| {
        matches!(a, Action::DeclareAttacker { attack: true, .. })
    });
    choose(&mut s, |a| {
        matches!(a, Action::DeclareAttacker { attack: false, .. })
    });
    pass_until(&mut s, ActionSpaceKind::Priority);
    while !s.game().state.stack_objects.is_empty() {
        s.pass_priority();
    }
    assert_eq!(s.current_step(), StepKind::DeclareAttackers);
    assert_eq!(s.game().state.players[0].combat_mana_pool.total(), 1);
}

/// X counts Forests you control on resolution, including an earthbent Forest
/// and excluding other lands and opponents' Forests; flashback reuses it.
#[test]
fn rockalanche_earthbends_by_forests_you_control_and_flashes_back() {
    let mut s = setup(&["Rockalanche"]);
    let forest = s.force_permanent_on_battlefield(0, "Forest");
    s.force_permanent_on_battlefield(0, "Forest");
    s.force_permanent_on_battlefield(0, "Plains");
    s.force_permanent_on_battlefield(1, "Forest");
    let spell = cast(&mut s, "Rockalanche");
    assert!(s.choose_target(Target::Permanent(forest)));
    // A third Forest arriving before resolution counts.
    s.force_permanent_on_battlefield(0, "Forest");
    settle(&mut s);
    assert!(is_land_creature(&s, forest));
    assert_eq!(counters(&s, forest), 3);
    assert_eq!(
        s.game().state.zones.zone_of(spell),
        Some(ZoneType::Graveyard)
    );

    let plains = s.battlefield_permanents_named(0, "Plains")[0];
    s.set_player_mana_pool(0, "5G");
    assert!(can_cast(&mut s, spell));
    cast_card(&mut s, spell);
    assert!(s.choose_target(Target::Permanent(plains)));
    settle(&mut s);
    assert_eq!(counters(&s, plains), 3);
    assert_eq!(s.game().state.zones.zone_of(spell), Some(ZoneType::Exile));
}

/// Put a library copy of `name` into player 0's `zone`.
fn put(s: &mut Scenario, name: &str, zone: ZoneType) -> CardId {
    let id = library_card(s, 0, name);
    s.game_mut().move_card(id, zone);
    id
}

/// Printed keywords plus those granted until end of turn.
fn keywords(s: &Scenario, id: PermanentId) -> managym::state::card::Keywords {
    let permanent = s.game().state.permanents[id].as_ref().unwrap();
    s.game().state.cards[permanent.card]
        .keywords
        .union(&permanent.temp_keywords)
}

fn cycling_index(s: &Scenario, card: CardId) -> usize {
    s.game().state.cards[card]
        .activated_abilities
        .iter()
        .position(|ability| ability.discard_source)
        .unwrap()
}

fn zone_of(s: &Scenario, card: CardId) -> Option<ZoneType> {
    s.game().state.zones.zone_of(card)
}

/// Landcycling (CR 702.29e) functions only from hand: {2} and discarding the
/// card are costs, the search offers only cards of the land subtype, and the
/// found card is revealed into hand before the library is shuffled.
#[test]
fn landcyclers_discard_for_two_and_find_their_basic_type() {
    for (name, subtype) in [
        ("Canyon Crawler", "Swamp"),
        ("Mongoose Lizard", "Mountain"),
        ("Saber-Tooth Moose-Lion", "Forest"),
        ("Giant Koi", "Island"),
        ("Rabaroo Troop", "Plains"),
    ] {
        let mut s = setup(&[name]);
        let card = put(&mut s, name, ZoneType::Hand);
        s.set_player_mana_pool(0, "1");
        assert!(!can_activate_card(&mut s, card), "{name} costs {{2}}");
        s.set_player_mana_pool(0, "2");
        // Programs compile in key order; cycling is the card's only hand ability.
        let index = cycling_index(&s, card);
        activate_card(&mut s, card, index);
        assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
        assert_eq!(zone_of(&s, card), Some(ZoneType::Graveyard), "{name}");
        assert_eq!(s.game().state.stack_objects.len(), 1);

        let library = s.zone_size(0, ZoneType::Library);
        pass_until(&mut s, ActionSpaceKind::LookAndSelect);
        let offered: Vec<CardId> = s
            .action_space()
            .actions
            .iter()
            .filter_map(|a| match a {
                Action::SelectCard { card, .. } => Some(*card),
                _ => None,
            })
            .collect();
        assert_eq!(offered.len(), 16, "{name}: every {subtype} in the library");
        assert!(offered.iter().all(|c| s.game().state.cards[*c]
            .subtypes
            .iter()
            .any(|t| t == subtype)));
        choose(
            &mut s,
            |a| matches!(a, Action::SelectCard { card, .. } if *card == offered[0]),
        );
        assert_eq!(zone_of(&s, offered[0]), Some(ZoneType::Hand));
        assert_eq!(s.zone_size(0, ZoneType::Library), library - 1);
        let definition = s.game().state.cards[offered[0]].definition_id;
        assert_eq!(
            s.game().state.players[0].known_hand.get(&definition),
            Some(&1)
        );
        settle(&mut s);

        // On the battlefield the cycling ability does not function.
        let permanent = s.force_permanent_on_battlefield(0, name);
        s.set_player_mana_pool(0, "2");
        s.game_mut().scenario_refresh_priority().unwrap();
        assert!(!s.action_space().actions.iter().any(|a| matches!(
            a,
            Action::ActivateAbility { permanent: p, ability_index, .. }
                if *p == permanent && *ability_index == index
        )));
    }
}

/// Cycling is an activated ability, so it is available at instant speed on
/// the opponent's turn.
#[test]
fn landcycling_works_on_the_opponents_turn() {
    let mut s = setup(&["Saber-Tooth Moose-Lion"]);
    let card = put(&mut s, "Saber-Tooth Moose-Lion", ZoneType::Hand);
    s.advance_to_active_step(1, StepKind::Main);
    s.pass_priority();
    assert_eq!(s.action_space().player, Some(PlayerId(0)));
    s.set_player_mana_pool(0, "2");
    assert!(can_activate_card(&mut s, card));
}

#[test]
fn canyon_crawler_has_deathtouch_and_makes_food_on_entering() {
    let mut s = setup(&["Canyon Crawler"]);
    s.set_player_mana_pool(0, "4BB");
    cast(&mut s, "Canyon Crawler");
    settle(&mut s);
    let crawler = s.battlefield_permanents_named(0, "Canyon Crawler")[0];
    assert!(keywords(&s, crawler).deathtouch);
    assert_eq!(s.game().effective_pt(crawler), (6, 6));
    let food = s.battlefield_permanents_named(0, "Food");
    assert_eq!(food.len(), 1);
    let food_card = s.game().state.permanents[food[0]].as_ref().unwrap().card;
    assert!(s.game().state.cards[food_card].types.is_artifact());
}

#[test]
fn mongoose_lizard_deals_one_damage_to_any_target_on_entering() {
    let mut s = setup(&["Mongoose Lizard", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    cast(&mut s, "Mongoose Lizard");
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    let targets = offered_targets(&s);
    assert!(targets.contains(&Target::Player(PlayerId(1))));
    assert!(targets.contains(&Target::Permanent(ogre)));
    let life = s.life(1);
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    settle(&mut s);
    assert_eq!(s.life(1), life - 1);
    let lizard = s.battlefield_permanents_named(0, "Mongoose Lizard")[0];
    assert!(keywords(&s, lizard).menace);
}

#[test]
fn saber_tooth_moose_lion_has_reach() {
    let mut s = setup(&["Saber-Tooth Moose-Lion"]);
    let lion = s.force_permanent_on_battlefield(0, "Saber-Tooth Moose-Lion");
    assert!(keywords(&s, lion).reach);
    assert_eq!(s.game().effective_pt(lion), (7, 7));
}

/// Waterbend {3} may be paid by tapping creatures; the Koi then can't be
/// blocked this turn.
#[test]
fn giant_koi_waterbends_to_become_unblockable() {
    let mut s = setup(&["Giant Koi", "Gray Ogre"]);
    let koi = s.force_permanent_on_battlefield(0, "Giant Koi");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.set_player_mana_pool(0, "2");
    let card = s.game().state.permanents[koi].as_ref().unwrap().card;
    let unblockable = 1 - cycling_index(&s, card);
    activate(&mut s, koi, unblockable);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Waterbend);
    assert!(s.choose_waterbend_tap(ogre));
    assert!(s.take_action_by_type(ActionType::PayCost));
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
    settle(&mut s);
    assert!(
        s.game().state.permanents[koi]
            .as_ref()
            .unwrap()
            .cant_be_blocked_this_turn
    );
}

/// Landfall counts only lands its controller's side gains; the flying grant
/// lasts until end of turn.
#[test]
fn rabaroo_troop_landfall_grants_flying_and_one_life() {
    let mut s = setup(&["Rabaroo Troop"]);
    let troop = s.force_permanent_on_battlefield(0, "Rabaroo Troop");
    let life = s.life(0);
    s.force_permanent_on_battlefield(1, "Plains");
    settle(&mut s);
    assert_eq!(s.life(0), life);
    assert!(!keywords(&s, troop).flying);

    let plains = put(&mut s, "Plains", ZoneType::Hand);
    s.game_mut().scenario_refresh_priority().unwrap();
    choose(
        &mut s,
        |a| matches!(a, Action::PlayLand { card, .. } if *card == plains),
    );
    settle(&mut s);
    assert_eq!(s.life(0), life + 1);
    assert!(keywords(&s, troop).flying);

    s.advance_to_active_step(1, StepKind::Main);
    assert!(!keywords(&s, troop).flying);
}

/// "Attacks alone" fires only when exactly one creature is declared; X counts
/// creatures you control on resolution.
#[test]
fn team_avatar_buffs_a_creature_attacking_alone() {
    for attackers in [1, 2] {
        let mut s = setup(&["Team Avatar", "Gray Ogre"]);
        s.force_permanent_on_battlefield(0, "Team Avatar");
        let first = s.force_permanent_on_battlefield(0, "Gray Ogre");
        let second = s.force_permanent_on_battlefield(0, "Gray Ogre");
        let third = s.force_permanent_on_battlefield(0, "Gray Ogre");
        for id in [first, second, third] {
            ready(&mut s, id);
        }
        s.advance_to_active_step(0, StepKind::DeclareAttackers);
        for declared in 0..3 {
            choose(
                &mut s,
                |a| matches!(a, Action::DeclareAttacker { attack, .. } if *attack == (declared < attackers)),
            );
        }
        pass_until(&mut s, ActionSpaceKind::Priority);
        while !s.game().state.stack_objects.is_empty() {
            s.pass_priority();
        }
        // Alone, the attacker gets +3/+3 (three creatures); otherwise no
        // creature changes.
        let buffed: Vec<_> = [first, second, third]
            .into_iter()
            .map(|id| s.game().effective_pt(id))
            .collect();
        let expected = usize::from(attackers == 1);
        assert_eq!(
            buffed.iter().filter(|pt| **pt == (5, 5)).count(),
            expected,
            "{attackers} attacking: {buffed:?}"
        );
        assert_eq!(
            buffed.iter().filter(|pt| **pt == (2, 2)).count(),
            3 - expected
        );
    }
}

/// From hand only: {2}{W} and discarding Team Avatar pay for damage equal to
/// the creatures you control, to a target creature.
#[test]
fn team_avatar_discards_from_hand_to_deal_damage_per_creature() {
    let mut s = setup(&["Team Avatar", "Gray Ogre", "Craw Wurm"]);
    s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.force_permanent_on_battlefield(0, "Gray Ogre");
    let wurm = s.force_permanent_on_battlefield(1, "Craw Wurm");
    let avatar = put(&mut s, "Team Avatar", ZoneType::Hand);
    s.set_player_mana_pool(0, "2");
    assert!(!can_activate_card(&mut s, avatar));
    s.set_player_mana_pool(0, "2W");
    activate_card(&mut s, avatar, 0);
    let targets = offered_targets(&s);
    assert!(targets.contains(&Target::Permanent(wurm)));
    assert!(!targets.contains(&Target::Player(PlayerId(1))));
    assert!(s.choose_target(Target::Permanent(wurm)));
    assert_eq!(zone_of(&s, avatar), Some(ZoneType::Graveyard));
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
    // A third creature arriving before resolution counts.
    s.force_permanent_on_battlefield(0, "Gray Ogre");
    settle(&mut s);
    assert_eq!(s.game().state.permanents[wurm].as_ref().unwrap().damage, 3);

    // From the battlefield or graveyard the ability does not function.
    let enchantment = s.force_permanent_on_battlefield(0, "Team Avatar");
    s.set_player_mana_pool(0, "2W");
    assert!(!can_activate_any(&mut s, enchantment));
    assert!(!can_activate_card(&mut s, avatar));
}

/// Graveyard-only (CR 113.6): returns that exact card; a card that left and
/// came back is a new object and the old activation does nothing.
#[test]
fn merchant_of_many_hats_returns_itself_from_the_graveyard() {
    let mut s = setup(&["Merchant of Many Hats"]);
    let merchant = put(&mut s, "Merchant of Many Hats", ZoneType::Hand);
    s.set_player_mana_pool(0, "2B");
    assert!(!can_activate_card(&mut s, merchant));
    s.game_mut().move_card(merchant, ZoneType::Graveyard);
    assert!(can_activate_card(&mut s, merchant));
    s.set_player_mana_pool(0, "2");
    assert!(!can_activate_card(&mut s, merchant));
    s.set_player_mana_pool(0, "2B");
    activate_card(&mut s, merchant, 0);
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
    settle(&mut s);
    assert_eq!(zone_of(&s, merchant), Some(ZoneType::Hand));
    let definition = s.game().state.cards[merchant].definition_id;
    assert_eq!(
        s.game().state.players[0].known_hand.get(&definition),
        Some(&1)
    );

    // Leaving the graveyard in response makes the ability's object stale.
    s.game_mut().move_card(merchant, ZoneType::Graveyard);
    s.set_player_mana_pool(0, "2B");
    activate_card(&mut s, merchant, 0);
    s.game_mut().move_card(merchant, ZoneType::Exile);
    s.game_mut().move_card(merchant, ZoneType::Graveyard);
    settle(&mut s);
    assert_eq!(zone_of(&s, merchant), Some(ZoneType::Graveyard));

    let permanent = s.force_permanent_on_battlefield(0, "Merchant of Many Hats");
    assert!(
        s.game().state.cards[s.game().state.permanents[permanent].as_ref().unwrap().card]
            .types
            .is_creature()
    );
    assert!(!can_activate_any(&mut s, permanent));
}

/// Exhaust (CR 702.177a): once activated, this object never offers the
/// ability again, even after it resolves; a new object starts fresh.
#[test]
fn hog_monkey_exhausts_once_for_two_counters() {
    let mut s = setup(&["Hog-Monkey"]);
    let monkey = s.force_permanent_on_battlefield(0, "Hog-Monkey");
    s.set_player_mana_pool(0, "4");
    assert!(!can_activate_any(&mut s, monkey));
    s.set_player_mana_pool(0, "10");
    activate(&mut s, monkey, 0);
    assert!(!can_activate_any(&mut s, monkey), "not even in response");
    settle(&mut s);
    assert_eq!(counters(&s, monkey), 2);
    assert_eq!(s.game().effective_pt(monkey), (5, 4));
    assert!(!can_activate_any(&mut s, monkey));

    let card = s.game().state.permanents[monkey].as_ref().unwrap().card;
    s.game_mut().move_card(card, ZoneType::Hand);
    s.game_mut().move_card(card, ZoneType::Battlefield);
    let again = s.battlefield_permanents_named(0, "Hog-Monkey")[0];
    assert!(can_activate_any(&mut s, again));
}

/// At the beginning of combat on its controller's turn, only creatures they
/// control with a +1/+1 counter are targets; the target gains menace.
#[test]
fn hog_monkey_gives_menace_to_a_creature_with_a_counter_at_combat() {
    let mut s = setup(&["Hog-Monkey", "Gray Ogre"]);
    s.force_permanent_on_battlefield(0, "Hog-Monkey");
    let plain = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let countered = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let opposing = s.force_permanent_on_battlefield(1, "Gray Ogre");
    for id in [countered, opposing] {
        s.game_mut().state.permanents[id]
            .as_mut()
            .unwrap()
            .plus1_counters = 1;
    }
    s.advance_to_active_step(0, StepKind::BeginningOfCombat);
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert_eq!(offered_targets(&s), vec![Target::Permanent(countered)]);
    assert!(s.choose_target(Target::Permanent(countered)));
    settle(&mut s);
    assert!(keywords(&s, countered).menace);
    assert!(!keywords(&s, plain).menace);
}

#[test]
fn rough_rhino_cavalry_firebends_two_and_exhausts_for_counters_and_trample() {
    let mut s = setup(&["Rough Rhino Cavalry"]);
    let rhino = s.force_permanent_on_battlefield(0, "Rough Rhino Cavalry");
    s.set_player_mana_pool(0, "8");
    activate(&mut s, rhino, 0);
    settle(&mut s);
    assert_eq!(counters(&s, rhino), 2);
    assert!(keywords(&s, rhino).trample);
    s.set_player_mana_pool(0, "8");
    assert!(!can_activate_any(&mut s, rhino));

    ready(&mut s, rhino);
    s.set_player_mana_pool(0, "");
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    choose(&mut s, |a| {
        matches!(a, Action::DeclareAttacker { attack: true, .. })
    });
    pass_until(&mut s, ActionSpaceKind::Priority);
    while !s.game().state.stack_objects.is_empty() {
        s.pass_priority();
    }
    assert_eq!(s.game().state.players[0].combat_mana_pool.total(), 2);
}

/// Counters go on the Captives, then earthbend 2 animates a land you control.
#[test]
fn rebellious_captives_exhausts_to_grow_and_earthbend_two() {
    let mut s = setup(&["Rebellious Captives"]);
    let captives = s.force_permanent_on_battlefield(0, "Rebellious Captives");
    let forest = s.force_permanent_on_battlefield(0, "Forest");
    let opposing = s.force_permanent_on_battlefield(1, "Forest");
    s.set_player_mana_pool(0, "6");
    activate(&mut s, captives, 0);
    let targets = offered_targets(&s);
    assert!(targets.contains(&Target::Permanent(forest)));
    assert!(!targets.contains(&Target::Permanent(opposing)));
    assert!(s.choose_target(Target::Permanent(forest)));
    settle(&mut s);
    assert_eq!(counters(&s, captives), 2);
    assert!(is_land_creature(&s, forest));
    assert_eq!(counters(&s, forest), 2);
    s.set_player_mana_pool(0, "6");
    assert!(!can_activate_any(&mut s, captives));
}

fn pool(s: &Scenario) -> &managym::state::mana::Mana {
    &s.game().state.players[0].mana_pool
}

/// The mana ability may be activated once each turn (CR 602.5b); the count
/// resets on the next turn.
#[test]
fn barrels_of_blasting_jelly_filters_one_mana_once_each_turn() {
    let mut s = setup(&["Barrels of Blasting Jelly"]);
    let barrels = s.force_permanent_on_battlefield(0, "Barrels of Blasting Jelly");
    s.set_player_mana_pool(0, "CC");
    activate(&mut s, barrels, 0);
    mode(&mut s, 3);
    assert_eq!(pool(&s).mana, [0, 0, 0, 1, 0, 1]);
    assert!(!can_activate_any(&mut s, barrels), "once this turn");

    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    s.set_player_mana_pool(0, "C");
    activate(&mut s, barrels, 0);
    mode(&mut s, 0);
    assert_eq!(pool(&s).mana, [1, 0, 0, 0, 0, 0]);
}

#[test]
fn barrels_of_blasting_jelly_sacrifices_to_deal_five_to_a_creature() {
    let mut s = setup(&["Barrels of Blasting Jelly", "Craw Wurm"]);
    let barrels = s.force_permanent_on_battlefield(0, "Barrels of Blasting Jelly");
    let wurm = s.force_permanent_on_battlefield(1, "Craw Wurm");
    let card = s.game().state.permanents[barrels].as_ref().unwrap().card;
    s.set_player_mana_pool(0, "4");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!s.action_space().actions.iter().any(|a| matches!(
        a,
        Action::ActivateAbility {
            ability_index: 1,
            ..
        }
    )));
    s.set_player_mana_pool(0, "5");
    activate(&mut s, barrels, 1);
    let targets = offered_targets(&s);
    assert!(targets.contains(&Target::Permanent(wurm)));
    assert!(!targets.contains(&Target::Player(PlayerId(1))));
    assert!(s.choose_target(Target::Permanent(wurm)));
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    assert_eq!(
        zone_of(&s, card),
        Some(ZoneType::Graveyard),
        "sacrificed as a cost"
    );
    settle(&mut s);
    assert!(
        s.game().state.permanents[wurm].is_none(),
        "5 damage kills the 6/4"
    );
}

/// Restricted mana (CR 106.6) pays for Ally spells and abilities of Ally
/// sources, and nothing else.
#[test]
fn jasmine_dragon_tea_shop_mana_pays_only_ally_spells_and_ally_abilities() {
    let ally = ManaUse {
        spells: ManaUse::spell(["Ally"]).spells,
        abilities: ManaUse::ability(["Ally"]).abilities,
    };
    let mut s = setup(&[
        "Jasmine Dragon Tea Shop",
        "Rebellious Captives",
        "Gray Ogre",
    ]);
    let shop = s.force_permanent_on_battlefield(0, "Jasmine Dragon Tea Shop");
    s.set_player_mana_pool(0, "");
    activate(&mut s, shop, 0);
    mode(&mut s, 4);
    assert_eq!(pool(&s).restricted_of(ally), [0, 0, 0, 0, 1, 0]);
    s.game_mut().state.players[0].mana_pool.mana = [0, 0, 0, 1, 0, 1];
    let captives = library_card(&s, 0, "Rebellious Captives");
    let ogre = library_card(&s, 0, "Gray Ogre");
    s.game_mut().move_card(captives, ZoneType::Hand);
    s.game_mut().move_card(ogre, ZoneType::Hand);
    assert!(
        !can_cast(&mut s, ogre),
        "{{2}}{{R}} cannot use the restricted green"
    );
    assert!(can_cast(&mut s, captives));

    let mut s = setup(&[
        "Jasmine Dragon Tea Shop",
        "Rebellious Captives",
        "Waterbending Scroll",
    ]);
    let shop = s.force_permanent_on_battlefield(0, "Jasmine Dragon Tea Shop");
    let captives = s.force_permanent_on_battlefield(0, "Rebellious Captives");
    let scroll = s.force_permanent_on_battlefield(0, "Waterbending Scroll");
    s.set_player_mana_pool(0, "");
    activate(&mut s, shop, 0);
    mode(&mut s, 0);
    s.game_mut().state.players[0].mana_pool.mana = [0, 0, 0, 0, 0, 5];
    assert!(
        !can_activate_any(&mut s, scroll),
        "a non-Ally ability cannot use it"
    );
    assert!(can_activate_any(&mut s, captives));
    activate(&mut s, captives, 0);
    assert!(s.choose_target(Target::Permanent(shop)));
    settle(&mut s);
    assert_eq!(pool(&s).total(), 0, "the Ally ability spent all six");
    assert_eq!(counters(&s, captives), 2);
}

#[test]
fn jasmine_dragon_tea_shop_taps_for_colorless_or_an_ally_token() {
    let mut s = setup(&["Jasmine Dragon Tea Shop"]);
    let shop = s.force_permanent_on_battlefield(0, "Jasmine Dragon Tea Shop");
    assert_eq!(
        s.game_mut().cached_producible_mana(PlayerId(0)).mana,
        [0, 0, 0, 0, 0, 1]
    );
    s.set_player_mana_pool(0, "5");
    activate(&mut s, shop, 1);
    settle(&mut s);
    let tokens = s.battlefield_permanents_named(0, "Ally");
    assert_eq!(tokens.len(), 1);
    assert_eq!(s.game().effective_pt(tokens[0]), (1, 1));
    assert!(s.game().state.permanents[shop].as_ref().unwrap().tapped);
}

#[test]
fn white_lotus_hideout_restricts_to_lessons_and_shrines_and_filters_for_one() {
    let lesson_or_shrine = ManaUse::spell(["Lesson", "Shrine"]);
    let mut s = setup(&[
        "White Lotus Hideout",
        "Firebending Lesson",
        "Lightning Strike",
        "Gray Ogre",
    ]);
    let hideout = s.force_permanent_on_battlefield(0, "White Lotus Hideout");
    s.force_permanent_on_battlefield(1, "Gray Ogre");
    assert_eq!(
        s.game_mut().cached_producible_mana(PlayerId(0)).mana,
        [0, 0, 0, 0, 0, 1]
    );
    s.set_player_mana_pool(0, "");
    activate(&mut s, hideout, 0);
    mode(&mut s, 3);
    assert_eq!(pool(&s).restricted_of(lesson_or_shrine), [0, 0, 0, 1, 0, 0]);
    let shrine = ManaCost {
        purpose: ManaUse::spell(["Shrine"]),
        ..ManaCost::parse("R")
    };
    assert!(pool(&s).can_pay(&shrine));
    let lesson = library_card(&s, 0, "Firebending Lesson");
    let strike = library_card(&s, 0, "Lightning Strike");
    s.game_mut().move_card(lesson, ZoneType::Hand);
    s.game_mut().move_card(strike, ZoneType::Hand);
    assert!(can_cast(&mut s, lesson));
    assert!(!can_cast(&mut s, strike));

    let mut s = setup(&["White Lotus Hideout"]);
    let hideout = s.force_permanent_on_battlefield(0, "White Lotus Hideout");
    s.set_player_mana_pool(0, "C");
    activate(&mut s, hideout, 1);
    mode(&mut s, 1);
    assert_eq!(pool(&s).mana, [0, 1, 0, 0, 0, 0]);
    assert!(pool(&s).restricted().is_empty());
}

/// "Costs {1} less for each Island you control" applies to the generic part
/// as the ability is activated.
#[test]
fn waterbending_scroll_draws_for_one_less_per_island() {
    let mut s = setup(&["Waterbending Scroll"]);
    let scroll = s.force_permanent_on_battlefield(0, "Waterbending Scroll");
    for _ in 0..4 {
        s.force_permanent_on_battlefield(0, "Island");
    }
    s.force_permanent_on_battlefield(1, "Island");
    for island in s.battlefield_permanents_named(0, "Island") {
        s.game_mut().state.permanents[island]
            .as_mut()
            .unwrap()
            .tapped = true;
    }
    s.set_player_mana_pool(0, "1");
    assert!(!can_activate_any(&mut s, scroll));
    s.set_player_mana_pool(0, "2");
    let hand = s.zone_size(0, ZoneType::Hand);
    activate(&mut s, scroll, 0);
    assert_eq!(pool(&s).total(), 0);
    settle(&mut s);
    assert_eq!(s.zone_size(0, ZoneType::Hand), hand + 1);
    assert!(s.game().state.permanents[scroll].as_ref().unwrap().tapped);
}

/// Airbend exiles the target and lets its owner, and only its owner, cast
/// that exiled card for {2} instead of its mana cost. Lands are not offered.
#[test]
fn airbending_lesson_exiles_and_owner_recasts_for_two() {
    let mut s = setup(&["Airbending Lesson", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let land = s.force_permanent_on_battlefield(1, "Mountain");
    let ogre_card = card_of(&s, ogre);
    let hand = s
        .game()
        .state
        .zones
        .zone_cards(ZoneType::Hand, PlayerId(0))
        .len();
    let lesson = cast(&mut s, "Airbending Lesson");
    assert!(!offered_targets(&s).contains(&Target::Permanent(land)));
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    assert_eq!(
        s.game().state.zones.zone_of(ogre_card),
        Some(ZoneType::Exile)
    );
    assert_eq!(
        s.game()
            .state
            .zones
            .zone_cards(ZoneType::Hand, PlayerId(0))
            .len(),
        hand + 1
    );
    assert_eq!(
        s.game().state.zones.zone_of(lesson),
        Some(ZoneType::Graveyard)
    );
    // The airbender's controller cannot cast the opponent's card.
    assert!(!can_cast(&mut s, ogre_card));

    s.advance_to_active_step(1, StepKind::Main);
    // {2} casts Gray Ogre ({2}{R}) from exile.
    s.set_player_mana_pool(1, "2");
    assert!(can_cast(&mut s, ogre_card));
    cast_card(&mut s, ogre_card);
    assert_eq!(s.game().state.players[1].mana_pool.total(), 0);
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(1, "Gray Ogre").len(), 1);
}

/// The permission belongs to the exact exiled object: once the card leaves
/// exile, a later return to exile does not restore it.
#[test]
fn airbend_permission_ends_when_the_card_leaves_exile() {
    let mut s = setup(&["Airbending Lesson", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let ogre_card = card_of(&s, ogre);
    cast(&mut s, "Airbending Lesson");
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    s.set_player_mana_pool(0, "2");
    assert!(can_cast(&mut s, ogre_card));
    s.game_mut().move_card(ogre_card, ZoneType::Graveyard);
    s.game_mut().move_card(ogre_card, ZoneType::Exile);
    assert!(!can_cast(&mut s, ogre_card));
}

/// Aang's entry airbends up to one other nonland permanent; casting a Lesson
/// gives Aang lifelink until end of turn.
#[test]
fn aang_airbends_another_permanent_and_gains_lifelink_from_lessons() {
    let mut s = setup(&["Aang, the Last Airbender", "Airbending Lesson", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let land = s.force_permanent_on_battlefield(1, "Plains");
    let ogre_card = card_of(&s, ogre);
    cast(&mut s, "Aang, the Last Airbender");
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    let aang = s.battlefield_permanents_named(0, "Aang, the Last Airbender")[0];
    let targets = offered_targets(&s);
    assert!(targets.contains(&Target::Permanent(ogre)));
    assert!(!targets.contains(&Target::Permanent(aang)));
    assert!(!targets.contains(&Target::Permanent(land)));
    assert!(s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::Decline { .. })));
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    assert_eq!(
        s.game().state.zones.zone_of(ogre_card),
        Some(ZoneType::Exile)
    );

    assert!(!keywords(&s, aang).lifelink);
    let blocker = s.force_permanent_on_battlefield(1, "Gray Ogre");
    cast(&mut s, "Airbending Lesson");
    assert!(s.choose_target(Target::Permanent(blocker)));
    // The cast trigger resolves above the Lesson.
    settle(&mut s);
    assert!(keywords(&s, aang).lifelink);
    s.advance_to_active_step(1, StepKind::Main);
    assert!(!keywords(&s, aang).lifelink);
}

/// Mode one destroys only an attacking creature; mode two airbends a creature
/// its caster controls, which can then be recast for {2}.
#[test]
fn airbenders_reversal_destroys_an_attacker_or_airbends_your_creature() {
    let mut s = setup(&["Airbender's Reversal", "Gray Ogre"]);
    let attacker = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let idle = s.force_permanent_on_battlefield(0, "Gray Ogre");
    ready(&mut s, attacker);
    ready(&mut s, idle);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    while s.action_space().kind == ActionSpaceKind::DeclareAttacker {
        choose(&mut s, |a| {
            matches!(a, Action::DeclareAttacker { permanent, attack, .. }
                if *attack == (*permanent == attacker))
        });
    }
    pass_until(&mut s, ActionSpaceKind::Priority);
    s.set_player_mana_pool(0, "20WWWW");
    cast(&mut s, "Airbender's Reversal");
    mode(&mut s, 0);
    let targets = offered_targets(&s);
    assert!(targets.contains(&Target::Permanent(attacker)));
    assert!(!targets.contains(&Target::Permanent(idle)));
    assert!(s.choose_target(Target::Permanent(attacker)));
    settle(&mut s);
    assert!(s.game().state.permanents[attacker].is_none());

    s.set_player_mana_pool(0, "20WWWW");
    let idle_card = card_of(&s, idle);
    cast(&mut s, "Airbender's Reversal");
    mode(&mut s, 1);
    assert!(s.choose_target(Target::Permanent(idle)));
    settle(&mut s);
    assert_eq!(
        s.game().state.zones.zone_of(idle_card),
        Some(ZoneType::Exile)
    );
    s.advance_to_active_step(0, StepKind::Main);
    s.set_player_mana_pool(0, "2");
    assert!(can_cast(&mut s, idle_card));
}

/// Avatar's Wrath spares its optional target, airbends every other creature,
/// exiles itself, and stops opponents casting from outside their hands until
/// its caster's next turn; the caster may recast their own creatures at once.
#[test]
fn avatars_wrath_spares_one_and_restricts_opponents_until_your_next_turn() {
    let mut s = setup(&["Avatar's Wrath", "Gray Ogre", "Craw Wurm"]);
    let spared = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let own = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let wurm = s.force_permanent_on_battlefield(1, "Craw Wurm");
    let land = s.force_permanent_on_battlefield(1, "Forest");
    let (own_card, wurm_card) = (card_of(&s, own), card_of(&s, wurm));
    let wrath = cast(&mut s, "Avatar's Wrath");
    assert!(s.choose_target(Target::Permanent(spared)));
    settle(&mut s);
    assert!(s.game().state.permanents[spared].is_some());
    assert!(s.game().state.permanents[land].is_some());
    assert_eq!(
        s.game().state.zones.zone_of(own_card),
        Some(ZoneType::Exile)
    );
    assert_eq!(
        s.game().state.zones.zone_of(wurm_card),
        Some(ZoneType::Exile)
    );
    assert_eq!(s.game().state.zones.zone_of(wrath), Some(ZoneType::Exile));

    s.set_player_mana_pool(0, "2");
    assert!(can_cast(&mut s, own_card));

    s.advance_to_active_step(1, StepKind::Main);
    s.set_player_mana_pool(1, "20GGGG");
    assert!(!can_cast(&mut s, wurm_card));

    s.advance_to_active_step(0, StepKind::Main);
    s.advance_to_active_step(1, StepKind::Main);
    s.set_player_mana_pool(1, "2");
    assert!(can_cast(&mut s, wurm_card));
}

/// Put the library copy `card` on top of its owner's library.
fn put_on_top(s: &mut Scenario, card: CardId) {
    s.game_mut().move_card(card, ZoneType::Hand);
    s.game_mut().move_card(card, ZoneType::Library);
}

/// Zuko's {3} exiles the top card; that exact card may be played this turn —
/// a land as the turn's land play, a spell for its own mana cost — and the
/// permission ends with the turn.
#[test]
fn zuko_exiles_the_top_card_and_may_play_it_this_turn() {
    let mut s = setup(&["Zuko, Exiled Prince", "Gray Ogre"]);
    let zuko = s.force_permanent_on_battlefield(0, "Zuko, Exiled Prince");
    let mountain = library_card(&s, 0, "Mountain");
    put_on_top(&mut s, mountain);
    activate(&mut s, zuko, 0);
    settle(&mut s);
    assert_eq!(
        s.game().state.zones.zone_of(mountain),
        Some(ZoneType::Exile)
    );
    choose(
        &mut s,
        |a| matches!(a, Action::PlayLand { card, .. } if *card == mountain),
    );
    assert_eq!(
        s.game().state.zones.zone_of(mountain),
        Some(ZoneType::Battlefield)
    );

    let ogre = library_card(&s, 0, "Gray Ogre");
    put_on_top(&mut s, ogre);
    activate(&mut s, zuko, 0);
    settle(&mut s);
    // The played Mountain supplies {R}: {1} more is short of {2}{R}.
    s.set_player_mana_pool(0, "1");
    assert!(!can_cast(&mut s, ogre));
    s.set_player_mana_pool(0, "2");
    assert!(can_cast(&mut s, ogre));

    // Unplayed, the card stays exiled without permission after the turn.
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    s.set_player_mana_pool(0, "20RRRR");
    assert_eq!(s.game().state.zones.zone_of(ogre), Some(ZoneType::Exile));
    assert!(!can_cast(&mut s, ogre));
}

/// Badgermole Cub earthbends 1 on entering; tapping a creature for mana adds
/// an extra {G}.
#[test]
fn badgermole_cub_earthbends_and_adds_green_when_a_creature_taps_for_mana() {
    let mut s = setup(&["Badgermole Cub", "Hermitic Herbalist"]);
    let forest = s.force_permanent_on_battlefield(0, "Forest");
    cast(&mut s, "Badgermole Cub");
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(forest)));
    settle(&mut s);
    assert_eq!(counters(&s, forest), 1);
    assert!(is_land_creature(&s, forest));

    let herbalist = s.force_permanent_on_battlefield(0, "Hermitic Herbalist");
    ready(&mut s, herbalist);
    s.set_player_mana_pool(0, "0");
    activate(&mut s, herbalist, 0);
    mode(&mut s, 1);
    settle(&mut s);
    assert_eq!(*pool(&s), managym::state::mana::Mana::parse("UG"));
}

/// Herbalist's second ability offers every unordered two-color pair once
/// (fifteen modes) and the mana pays only Lesson spells.
#[test]
fn hermitic_herbalist_offers_each_two_color_combination_once() {
    let mut s = setup(&["Hermitic Herbalist"]);
    let herbalist = s.force_permanent_on_battlefield(0, "Hermitic Herbalist");
    ready(&mut s, herbalist);
    s.set_player_mana_pool(0, "0");
    activate(&mut s, herbalist, 1);
    let modes = s
        .action_space()
        .actions
        .iter()
        .filter(|a| matches!(a, Action::ChooseMode { .. }))
        .count();
    assert_eq!(modes, 15);
    mode(&mut s, 1);
    settle(&mut s);
    let player = &s.game().state.players[0].mana_pool;
    assert_eq!(
        player.restricted_of(ManaUse::spell(["Lesson"])),
        [1, 1, 0, 0, 0, 0]
    );
    assert_eq!(player.total(), 2);
}

/// Haru triggers for each other Ally entering, not for itself. Two earthbends
/// on one land stack counters and leave two independent delayed returns: the
/// land returns once, tapped, as a new object.
#[test]
fn haru_earthbends_per_other_ally_and_a_doubly_bent_land_returns_once() {
    let mut s = setup(&[
        "Haru, Hidden Talent",
        "Hermitic Herbalist",
        "Lightning Strike",
    ]);
    let forest = s.force_permanent_on_battlefield(0, "Forest");
    cast(&mut s, "Haru, Hidden Talent");
    settle(&mut s);
    assert_eq!(counters(&s, forest), 0);
    for _ in 0..2 {
        cast(&mut s, "Hermitic Herbalist");
        pass_until(&mut s, ActionSpaceKind::ChooseTarget);
        assert!(s.choose_target(Target::Permanent(forest)));
        settle(&mut s);
    }
    assert_eq!(counters(&s, forest), 2);
    assert_eq!(s.game().state.delayed_triggers.len(), 2);

    let land = card_of(&s, forest);
    cast(&mut s, "Lightning Strike");
    assert!(s.choose_target(Target::Permanent(forest)));
    settle(&mut s);
    assert_eq!(zone_of(&s, land), Some(ZoneType::Battlefield));
    let forests = s.battlefield_permanents_named(0, "Forest");
    assert_eq!(forests.len(), 1);
    let returned = s.game().state.permanents[forests[0]].as_ref().unwrap();
    assert!(returned.tapped && !returned.animated);
    assert_eq!(returned.plus1_counters, 0);
    assert!(s.game().state.delayed_triggers.is_empty());
}

/// Iroh, Grand Lotus attacks for firebending 2. (Its graveyard flashback grants
/// are witnessed in `tests/rules/temur_cards.rs`.)
#[test]
fn iroh_grand_lotus_firebends_two_when_attacking() {
    let mut s = setup(&["Iroh, Grand Lotus"]);
    let iroh = s.force_permanent_on_battlefield(0, "Iroh, Grand Lotus");
    ready(&mut s, iroh);
    s.set_player_mana_pool(0, "");
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    choose(&mut s, |a| {
        matches!(a, Action::DeclareAttacker { attack: true, .. })
    });
    pass_until(&mut s, ActionSpaceKind::Priority);
    while !s.game().state.stack_objects.is_empty() {
        s.pass_priority();
    }
    assert_eq!(
        s.game().state.players[0].combat_mana_pool,
        managym::state::mana::Mana::parse("RR")
    );
}

/// Aang's Iceberg exiles another nonland permanent until it leaves. Its
/// waterbend ability sacrifices it on resolution — the exiled card returns —
/// and scries 2 only if that sacrifice happened.
#[test]
fn aangs_iceberg_exiles_until_it_leaves_and_scries_only_if_sacrificed() {
    let mut s = setup(&["Aang's Iceberg", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let ogre_card = card_of(&s, ogre);
    cast(&mut s, "Aang's Iceberg");
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    let iceberg = s.battlefield_permanents_named(0, "Aang's Iceberg")[0];
    assert!(!offered_targets(&s).contains(&Target::Permanent(iceberg)));
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    assert_eq!(zone_of(&s, ogre_card), Some(ZoneType::Exile));

    // Two activations on the stack: the first sacrifices and scries, the
    // second finds no Iceberg and does nothing.
    activate(&mut s, iceberg, 0);
    if s.action_space().kind != ActionSpaceKind::Priority {
        choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    }
    activate(&mut s, iceberg, 0);
    if s.action_space().kind != ActionSpaceKind::Priority {
        choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    }
    assert_eq!(s.game().state.stack_objects.len(), 2);
    let mut scries = 0;
    for _ in 0..40 {
        if s.action_space().kind == ActionSpaceKind::Priority
            && s.game().state.stack_objects.is_empty()
        {
            break;
        }
        if s.action_space().kind == ActionSpaceKind::Scry {
            scries += 1;
            while s.action_space().kind == ActionSpaceKind::Scry {
                s.step_action(0);
            }
            continue;
        }
        s.pass_priority();
    }
    assert_eq!(scries, 1);
    assert_eq!(zone_of(&s, ogre_card), Some(ZoneType::Battlefield));
    assert!(s
        .battlefield_permanents_named(0, "Aang's Iceberg")
        .is_empty());
}

/// North Pole Patrol untaps another permanent you control, and its waterbend
/// ability taps an opposing creature; the Patrol's own {T} cannot also help
/// pay the waterbend cost.
#[test]
fn north_pole_patrol_untaps_another_and_waterbends_to_tap_an_opponent() {
    let mut s = setup(&["North Pole Patrol", "Gray Ogre"]);
    let patrol = s.force_permanent_on_battlefield(0, "North Pole Patrol");
    let own = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let foe = s.force_permanent_on_battlefield(1, "Gray Ogre");
    ready(&mut s, patrol);
    s.game_mut().state.permanents[own].as_mut().unwrap().tapped = true;
    // Abilities index in program key order: "tap" (0) before "untap" (1).
    activate(&mut s, patrol, 1);
    let targets = offered_targets(&s);
    assert!(!targets.contains(&Target::Permanent(patrol)));
    assert!(!targets.contains(&Target::Permanent(foe)));
    assert!(s.choose_target(Target::Permanent(own)));
    settle(&mut s);
    assert!(!s.game().state.permanents[own].as_ref().unwrap().tapped);

    s.game_mut().state.permanents[patrol]
        .as_mut()
        .unwrap()
        .tapped = false;
    s.set_player_mana_pool(0, "2");
    activate(&mut s, patrol, 0);
    assert_eq!(offered_targets(&s), vec![Target::Permanent(foe)]);
    assert!(s.choose_target(Target::Permanent(foe)));
    assert_eq!(s.action_space().kind, ActionSpaceKind::Waterbend);
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::WaterbendTap { permanent, .. } if *permanent == patrol)));
    assert!(s.choose_waterbend_tap(own));
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    settle(&mut s);
    assert!(s.game().state.permanents[foe].as_ref().unwrap().tapped);
    assert_eq!(pool(&s).total(), 0);
}

/// Zuko's Conviction returns a creature card from your graveyard to hand, or,
/// kicked, puts it onto the battlefield tapped. Noncreature cards are not
/// targets.
#[test]
fn zukos_conviction_returns_a_creature_to_hand_or_kicked_onto_the_battlefield() {
    for kicked in [false, true] {
        let mut s = setup(&["Zuko's Conviction", "Gray Ogre", "Lightning Strike"]);
        let ogre = put(&mut s, "Gray Ogre", ZoneType::Graveyard);
        let strike = put(&mut s, "Lightning Strike", ZoneType::Graveyard);
        cast(&mut s, "Zuko's Conviction");
        assert_eq!(s.action_space().kind, ActionSpaceKind::PayOrNot);
        if kicked {
            choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
        } else {
            choose(&mut s, |a| matches!(a, Action::Decline { .. }));
        }
        let targets = offered_targets(&s);
        assert!(!targets
            .iter()
            .any(|t| matches!(t, Target::GraveyardCard(o) if CardId::from(o.entity) == strike)));
        choose(&mut s, |a| {
            matches!(a, Action::ChooseTarget { target: Target::GraveyardCard(o), .. }
                if CardId::from(o.entity) == ogre)
        });
        settle(&mut s);
        if kicked {
            assert_eq!(zone_of(&s, ogre), Some(ZoneType::Battlefield));
            let id = s.battlefield_permanents_named(0, "Gray Ogre")[0];
            assert!(s.game().state.permanents[id].as_ref().unwrap().tapped);
        } else {
            assert_eq!(zone_of(&s, ogre), Some(ZoneType::Hand));
        }
    }
}

fn hand(s: &Scenario) -> Vec<CardId> {
    s.game()
        .state
        .zones
        .zone_cards(ZoneType::Hand, PlayerId(0))
        .to_vec()
}

/// Sokka discards up to two cards and only then draws that many, so a drawn
/// card cannot be discarded; casting an artifact or a Lesson grows him.
#[test]
fn sokka_discards_up_to_two_then_draws_that_many_and_grows_on_artifacts_and_lessons() {
    let mut s = setup(&[
        "Sokka, Bold Boomeranger",
        "Earthbending Lesson",
        "Barrels of Blasting Jelly",
        "Gray Ogre",
    ]);
    let kept = put(&mut s, "Gray Ogre", ZoneType::Hand);
    let first = put(&mut s, "Forest", ZoneType::Hand);
    let second = put(&mut s, "Island", ZoneType::Hand);
    cast(&mut s, "Sokka, Bold Boomeranger");
    pass_until(&mut s, ActionSpaceKind::Discard);
    choose(
        &mut s,
        |a| matches!(a, Action::SelectCard { card, .. } if *card == first),
    );
    assert_eq!(s.action_space().kind, ActionSpaceKind::Discard);
    assert_eq!(hand(&s).len(), 2, "no draw before the discards finish");
    choose(
        &mut s,
        |a| matches!(a, Action::SelectCard { card, .. } if *card == second),
    );
    settle(&mut s);
    let in_hand = hand(&s);
    assert_eq!(in_hand.len(), 3);
    assert!(in_hand.contains(&kept));
    assert_eq!(zone_of(&s, first), Some(ZoneType::Graveyard));

    let sokka = s.battlefield_permanents_named(0, "Sokka, Bold Boomeranger")[0];
    s.set_player_mana_pool(0, "20WWWWUUUUBBBBRRRRGGGG");
    cast(&mut s, "Gray Ogre");
    settle(&mut s);
    assert_eq!(counters(&s, sokka), 0);
    cast(&mut s, "Barrels of Blasting Jelly");
    settle(&mut s);
    assert_eq!(counters(&s, sokka), 1);
    let forest = s.force_permanent_on_battlefield(0, "Forest");
    cast(&mut s, "Earthbending Lesson");
    assert!(s.choose_target(Target::Permanent(forest)));
    settle(&mut s);
    assert_eq!(counters(&s, sokka), 2);

    // Declining at once discards nothing and draws nothing.
    let mut s = setup(&["Sokka, Bold Boomeranger"]);
    put(&mut s, "Forest", ZoneType::Hand);
    cast(&mut s, "Sokka, Bold Boomeranger");
    pass_until(&mut s, ActionSpaceKind::Discard);
    choose(&mut s, |a| matches!(a, Action::Decline { .. }));
    settle(&mut s);
    assert_eq!(hand(&s).len(), 1);
}

/// Seismic Sense looks at one card per land you control (an earthbent land
/// still counts; opponents' lands do not) and may take a creature or land.
#[test]
fn seismic_sense_looks_at_cards_equal_to_your_lands_and_takes_a_creature_or_land() {
    let mut s = setup(&["Seismic Sense", "Gray Ogre", "Lightning Strike"]);
    for _ in 0..3 {
        s.force_permanent_on_battlefield(0, "Forest");
    }
    s.force_permanent_on_battlefield(1, "Forest");
    let ogre = library_card(&s, 0, "Gray Ogre");
    let strike = library_card(&s, 0, "Lightning Strike");
    let below = library_card(&s, 0, "Island");
    put_on_top(&mut s, below);
    put_on_top(&mut s, ogre);
    put_on_top(&mut s, strike);
    cast(&mut s, "Seismic Sense");
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    let selectable: Vec<_> = s
        .action_space()
        .actions
        .iter()
        .filter_map(|a| match a {
            Action::SelectCard { card, .. } => Some(*card),
            _ => None,
        })
        .collect();
    assert_eq!(
        selectable.len(),
        2,
        "the Ogre and the Island, not the Strike"
    );
    assert!(selectable.contains(&ogre) && selectable.contains(&below));
    choose(
        &mut s,
        |a| matches!(a, Action::SelectCard { card, .. } if *card == ogre),
    );
    settle(&mut s);
    assert_eq!(zone_of(&s, ogre), Some(ZoneType::Hand));
    let library = s
        .game()
        .state
        .zones
        .zone_cards(ZoneType::Library, PlayerId(0))
        .to_vec();
    assert!(library[..2].contains(&strike) && library[..2].contains(&below));
}

fn offered_cards(s: &Scenario) -> Vec<CardId> {
    s.action_space()
        .actions
        .iter()
        .filter_map(|a| match a {
            Action::SelectCard { card, .. } => Some(*card),
            _ => None,
        })
        .collect()
}

/// Aang's Journey finds only a basic land, reveals it into hand and gains 2
/// life; kicked, it also searches for a Shrine card (none in this library, so
/// that search fails to find).
#[test]
fn aangs_journey_finds_a_basic_land_and_kicked_also_a_shrine() {
    for kicked in [false, true] {
        let mut s = setup(&["Aang's Journey", "Gray Ogre"]);
        cast(&mut s, "Aang's Journey");
        assert_eq!(s.action_space().kind, ActionSpaceKind::PayOrNot);
        if kicked {
            choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
        } else {
            choose(&mut s, |a| matches!(a, Action::Decline { .. }));
        }
        pass_until(&mut s, ActionSpaceKind::LookAndSelect);
        let offered = offered_cards(&s);
        assert_eq!(offered.len(), 80, "every basic land, no Gray Ogre");
        assert!(offered
            .iter()
            .all(|c| s.game().state.cards[*c].types.is_land()));
        choose(
            &mut s,
            |a| matches!(a, Action::SelectCard { card, .. } if *card == offered[0]),
        );
        if kicked {
            assert_eq!(s.action_space().kind, ActionSpaceKind::LookAndSelect);
            assert!(offered_cards(&s).is_empty());
        }
        settle(&mut s);
        assert_eq!(zone_of(&s, offered[0]), Some(ZoneType::Hand));
        s.assert_life(0, 22);
    }
}

/// Sparring Dummy cannot attack. Its {T} mills one card: a milled land may go
/// to hand, and a milled Lesson gains 2 life.
#[test]
fn sparring_dummy_mills_a_land_to_hand_or_a_lesson_for_two_life() {
    let mut s = setup(&["Sparring Dummy", "Earthbending Lesson", "Gray Ogre"]);
    let dummy = s.force_permanent_on_battlefield(0, "Sparring Dummy");
    ready(&mut s, dummy);
    let card = s.game().state.permanents[dummy].as_ref().unwrap().card;
    assert!(s.game().state.cards[card].keywords.defender);

    let forest = library_card(&s, 0, "Forest");
    put_on_top(&mut s, forest);
    activate(&mut s, dummy, 0);
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    choose(
        &mut s,
        |a| matches!(a, Action::SelectCard { card, .. } if *card == forest),
    );
    settle(&mut s);
    assert_eq!(zone_of(&s, forest), Some(ZoneType::Hand));
    s.assert_life(0, 20);

    for (name, life) in [("Earthbending Lesson", 22), ("Gray Ogre", 22)] {
        s.game_mut().state.permanents[dummy]
            .as_mut()
            .unwrap()
            .tapped = false;
        let milled = library_card(&s, 0, name);
        put_on_top(&mut s, milled);
        activate(&mut s, dummy, 0);
        settle(&mut s);
        assert_eq!(zone_of(&s, milled), Some(ZoneType::Graveyard));
        s.assert_life(0, life);
    }
}

/// Lost Days: the target's owner (not the caster) chooses second from the top
/// or the bottom of their library; the caster creates a Clue either way.
#[test]
fn lost_days_owner_puts_it_second_from_top_or_on_the_bottom_and_caster_gets_a_clue() {
    for second in [true, false] {
        let mut s = setup(&["Lost Days", "Gray Ogre", "Aang's Iceberg"]);
        let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
        let land = s.force_permanent_on_battlefield(1, "Forest");
        let ogre_card = card_of(&s, ogre);
        cast(&mut s, "Lost Days");
        assert!(!offered_targets(&s).contains(&Target::Permanent(land)));
        assert!(s.choose_target(Target::Permanent(ogre)));
        pass_until(&mut s, ActionSpaceKind::Modal);
        assert_eq!(s.action_space().player, Some(PlayerId(1)));
        mode(&mut s, if second { 0 } else { 1 });
        settle(&mut s);
        let library = s
            .game()
            .state
            .zones
            .zone_cards(ZoneType::Library, PlayerId(1))
            .to_vec();
        let expected = if second { library.len() - 2 } else { 0 };
        assert_eq!(library[expected], ogre_card);
        assert_eq!(s.battlefield_permanents_named(0, "Clue").len(), 1);
    }
}

/// Serpent of the Pass costs {1} less per noncreature, nonland card in your
/// graveyard (creatures and lands there do not count), and with three Lessons
/// there it may be cast at instant speed.
#[test]
fn serpent_of_the_pass_is_cheaper_per_graveyard_spell_and_flashes_with_three_lessons() {
    let mut s = setup(&[
        "Serpent of the Pass",
        "Lightning Strike",
        "Earthbending Lesson",
        "Gray Ogre",
    ]);
    let serpent = put(&mut s, "Serpent of the Pass", ZoneType::Hand);
    put(&mut s, "Gray Ogre", ZoneType::Graveyard);
    put(&mut s, "Forest", ZoneType::Graveyard);
    for _ in 0..2 {
        put(&mut s, "Lightning Strike", ZoneType::Graveyard);
    }
    // {5}{U}{U} less two.
    s.set_player_mana_pool(0, "2UU");
    assert!(!can_cast(&mut s, serpent));
    s.set_player_mana_pool(0, "3UU");
    assert!(can_cast(&mut s, serpent));

    // Two Lessons: sorcery timing only, so not in the beginning of combat.
    for _ in 0..2 {
        put(&mut s, "Earthbending Lesson", ZoneType::Graveyard);
    }
    s.advance_to_active_step(0, StepKind::BeginningOfCombat);
    s.set_player_mana_pool(0, "20UU");
    assert!(!can_cast(&mut s, serpent));
    put(&mut s, "Earthbending Lesson", ZoneType::Graveyard);
    assert!(can_cast(&mut s, serpent));
    // Seven noncreature, nonland cards reduce all {5} generic.
    s.set_player_mana_pool(0, "UU");
    assert!(can_cast(&mut s, serpent));
}

/// Elemental Teachings reveals up to four differently named lands; the
/// opponent chooses two for the graveyard and the rest enter tapped.
#[test]
fn elemental_teachings_finds_distinct_lands_and_the_opponent_bins_two() {
    let mut s = setup(&["Elemental Teachings"]);
    cast(&mut s, "Elemental Teachings");
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    let mut chosen = Vec::new();
    for name in ["Forest", "Island", "Swamp", "Mountain"] {
        let offered = offered_cards(&s);
        assert!(chosen.iter().all(|c| !offered.contains(c)));
        let card = *offered
            .iter()
            .find(|c| s.game().state.cards[**c].name == name)
            .unwrap();
        if name == "Island" {
            // A second Forest is no longer offered.
            assert!(!offered
                .iter()
                .any(|c| s.game().state.cards[*c].name == "Forest"));
        }
        choose(
            &mut s,
            |a| matches!(a, Action::SelectCard { card: c, .. } if *c == card),
        );
        chosen.push(card);
    }
    assert_eq!(s.action_space().player, Some(PlayerId(1)));
    let offered = offered_cards(&s);
    assert_eq!(offered.len(), 4);
    for binned in &chosen[..2] {
        choose(
            &mut s,
            |a| matches!(a, Action::SelectCard { card, .. } if card == binned),
        );
    }
    settle(&mut s);
    for card in &chosen[..2] {
        assert_eq!(zone_of(&s, *card), Some(ZoneType::Graveyard));
    }
    for card in &chosen[2..] {
        assert_eq!(zone_of(&s, *card), Some(ZoneType::Battlefield));
        let id = s.game().state.card_to_permanent[*card].unwrap();
        assert!(s.game().state.permanents[id].as_ref().unwrap().tapped);
    }

    // Stopping after one land: the opponent's choice bins it.
    let mut s = setup(&["Elemental Teachings"]);
    cast(&mut s, "Elemental Teachings");
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    let first = offered_cards(&s)[0];
    choose(
        &mut s,
        |a| matches!(a, Action::SelectCard { card, .. } if *card == first),
    );
    choose(&mut s, |a| matches!(a, Action::Decline { .. }));
    choose(
        &mut s,
        |a| matches!(a, Action::SelectCard { card, .. } if *card == first),
    );
    settle(&mut s);
    assert_eq!(zone_of(&s, first), Some(ZoneType::Graveyard));
}

fn put_for(s: &mut Scenario, player: usize, name: &str, zone: ZoneType) -> CardId {
    let card = library_card(s, player, name);
    s.game_mut().move_card(card, zone);
    card
}

/// Dai Li Indoctrination's first mode reveals the opponent's hand (public
/// definition knowledge) and the caster picks a nonland permanent card for
/// them to discard; the second mode earthbends 2.
#[test]
fn dai_li_indoctrination_reveals_a_hand_and_discards_your_pick_or_earthbends_two() {
    let mut s = setup(&[
        "Dai Li Indoctrination",
        "Gray Ogre",
        "Lightning Strike",
        "Aang's Iceberg",
    ]);
    let ogre = put_for(&mut s, 1, "Gray Ogre", ZoneType::Hand);
    let iceberg = put_for(&mut s, 1, "Aang's Iceberg", ZoneType::Hand);
    let strike = put_for(&mut s, 1, "Lightning Strike", ZoneType::Hand);
    put_for(&mut s, 1, "Forest", ZoneType::Hand);
    cast(&mut s, "Dai Li Indoctrination");
    mode(&mut s, 0);
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    pass_until(&mut s, ActionSpaceKind::Discard);
    assert_eq!(s.action_space().player, Some(PlayerId(0)));
    let offered = offered_cards(&s);
    assert_eq!(offered.len(), 2);
    assert!(offered.contains(&ogre) && offered.contains(&iceberg));
    assert_eq!(
        s.game().state.players[1].known_hand.values().sum::<u32>(),
        4
    );
    // The chooser observes the offered cards in the opponent's hand.
    let view = managym::agent::observation::Observation::for_player(s.game(), PlayerId(0));
    let seen: Vec<i32> = view.opponent_cards.iter().map(|c| c.id).collect();
    for card in [ogre, iceberg] {
        assert!(seen.contains(&(s.game().state.cards[card].id.0 as i32)));
    }
    choose(
        &mut s,
        |a| matches!(a, Action::SelectCard { card, .. } if *card == iceberg),
    );
    settle(&mut s);
    assert_eq!(zone_of(&s, iceberg), Some(ZoneType::Graveyard));
    assert_eq!(zone_of(&s, strike), Some(ZoneType::Hand));
    assert_eq!(
        s.game().state.players[1].known_hand.values().sum::<u32>(),
        3
    );

    let forest = s.force_permanent_on_battlefield(0, "Forest");
    cast(&mut s, "Dai Li Indoctrination");
    mode(&mut s, 1);
    assert!(s.choose_target(Target::Permanent(forest)));
    settle(&mut s);
    assert_eq!(counters(&s, forest), 2);
}

/// Professor Zei's {T}, discard: the discarded card is chosen as a cost
/// (no activation with an empty hand), then a card is drawn. Its sacrifice
/// ability returns an instant or sorcery card, only during your turn.
#[test]
fn professor_zei_discards_as_a_cost_to_draw_and_sacrifices_to_recall_a_spell() {
    let mut s = setup(&[
        "Professor Zei, Anthropologist",
        "Lightning Strike",
        "Gray Ogre",
    ]);
    let zei = s.force_permanent_on_battlefield(0, "Professor Zei, Anthropologist");
    ready(&mut s, zei);
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!s.action_space().actions.iter().any(|a| matches!(
        a, Action::ActivateAbility { permanent, ability_index: 0, .. } if *permanent == zei
    )));
    let pitched = put(&mut s, "Gray Ogre", ZoneType::Hand);
    activate(&mut s, zei, 0);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Discard);
    choose(
        &mut s,
        |a| matches!(a, Action::SelectCard { card, .. } if *card == pitched),
    );
    assert_eq!(zone_of(&s, pitched), Some(ZoneType::Graveyard));
    assert!(s.game().state.permanents[zei].as_ref().unwrap().tapped);
    settle(&mut s);
    assert_eq!(hand(&s).len(), 1);

    let strike = put(&mut s, "Lightning Strike", ZoneType::Graveyard);
    s.game_mut().state.permanents[zei].as_mut().unwrap().tapped = false;
    activate(&mut s, zei, 1);
    let targets = offered_targets(&s);
    assert_eq!(targets.len(), 1, "the Strike, not the Ogre");
    choose(&mut s, |a| matches!(a, Action::ChooseTarget { .. }));
    settle(&mut s);
    assert_eq!(zone_of(&s, strike), Some(ZoneType::Hand));
    assert!(s
        .battlefield_permanents_named(0, "Professor Zei, Anthropologist")
        .is_empty());
}

/// X is announced while casting: offers run from 0 to the most the caster can
/// pay, the stack object keeps it, and the spell makes X Allies before every
/// creature you control gets a counter.
#[test]
fn united_front_announces_x_and_counts_it_on_resolution() {
    use managym::state::stack_object::StackObject;
    let mut s = setup(&["United Front", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let opposing = s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.set_player_mana_pool(0, "3WW");
    cast(&mut s, "United Front");
    assert_eq!(s.action_space().kind, ActionSpaceKind::Modal);
    let offered: Vec<usize> = s
        .action_space()
        .actions
        .iter()
        .filter_map(|a| match a {
            Action::ChooseMode { mode, .. } => Some(*mode),
            _ => None,
        })
        .collect();
    assert_eq!(offered, vec![0, 1, 2, 3]);
    mode(&mut s, 3);
    let Some(StackObject::Spell(spell)) = s.game().state.stack_objects.last() else {
        panic!("United Front is not on the stack");
    };
    assert_eq!(spell.x, 3);
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
    settle(&mut s);
    let allies = s.battlefield_permanents_named(0, "Ally");
    assert_eq!(allies.len(), 3);
    for ally in allies {
        assert_eq!(counters(&s, ally), 1);
    }
    assert_eq!(counters(&s, ogre), 1);
    assert_eq!(counters(&s, opposing), 0);
}

/// The optional waterbend {6} is offered only when payable and may be paid by
/// tapping creatures; paid, the spell reshuffles the graveyard, draws seven
/// and lifts the maximum hand size. Either way it exiles itself.
#[test]
fn spirit_water_revival_waterbends_its_additional_cost() {
    let mut s = setup(&["Spirit Water Revival", "Gray Ogre", "Craw Wurm"]);
    let ogres: Vec<PermanentId> = ["Gray Ogre", "Craw Wurm"]
        .into_iter()
        .flat_map(|name| [name; 3])
        .map(|name| s.force_permanent_on_battlefield(0, name))
        .collect();
    let binned = library_card(&s, 0, "Gray Ogre");
    s.game_mut().move_card(binned, ZoneType::Graveyard);

    // Unkicked: with {1}{U}{U} and no untapped helpers, no choice is offered.
    for ogre in &ogres {
        s.game_mut().state.permanents[*ogre]
            .as_mut()
            .unwrap()
            .tapped = true;
    }
    s.set_player_mana_pool(0, "1UU");
    let hand = s.zone_size(0, ZoneType::Hand);
    let first = cast(&mut s, "Spirit Water Revival");
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
    assert_eq!(s.game().state.stack_objects.len(), 1);
    settle(&mut s);
    assert_eq!(s.zone_size(0, ZoneType::Hand), hand + 2);
    assert_eq!(s.game().state.zones.zone_of(first), Some(ZoneType::Exile));
    assert!(!s.game().state.players[0].no_maximum_hand_size);

    // Kicked: five creatures and {1} more mana pay the {6}.
    for ogre in &ogres[..5] {
        ready(&mut s, *ogre);
        s.game_mut().state.permanents[*ogre]
            .as_mut()
            .unwrap()
            .tapped = false;
    }
    s.set_player_mana_pool(0, "2UU");
    let second = cast(&mut s, "Spirit Water Revival");
    assert_eq!(s.action_space().kind, ActionSpaceKind::PayOrNot);
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    assert_eq!(s.action_space().kind, ActionSpaceKind::Waterbend);
    for ogre in &ogres[..5] {
        choose(
            &mut s,
            |a| matches!(a, Action::WaterbendTap { permanent, .. } if permanent == ogre),
        );
    }
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
    let hand = s.zone_size(0, ZoneType::Hand);
    settle(&mut s);
    assert_eq!(s.zone_size(0, ZoneType::Hand), hand + 7);
    assert_eq!(s.zone_size(0, ZoneType::Graveyard), 0);
    assert_ne!(
        s.game().state.zones.zone_of(binned),
        Some(ZoneType::Graveyard)
    );
    assert_eq!(s.game().state.zones.zone_of(second), Some(ZoneType::Exile));
    assert!(s.game().state.players[0].no_maximum_hand_size);
}

/// Pass priority once so the top of the stack resolves into its decision.
fn resolve_top(s: &mut Scenario) {
    s.pass_priority();
    s.pass_priority();
}

/// Yue's ability offers only noncreature spells from hand; the chosen spell
/// is cast for free after the ability finishes resolving.
#[test]
fn yue_casts_a_noncreature_spell_from_hand_for_free() {
    let mut s = setup(&["Yue, the Moon Spirit", "Lightning Strike", "Gray Ogre"]);
    let yue = s.force_permanent_on_battlefield(0, "Yue, the Moon Spirit");
    ready(&mut s, yue);
    let strike = library_card(&s, 0, "Lightning Strike");
    let ogre = library_card(&s, 0, "Gray Ogre");
    s.game_mut().move_card(strike, ZoneType::Hand);
    s.game_mut().move_card(ogre, ZoneType::Hand);
    s.set_player_mana_pool(0, "5");
    activate(&mut s, yue, 0);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Waterbend);
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    assert!(s.game().state.permanents[yue].as_ref().unwrap().tapped);
    resolve_top(&mut s);
    assert_eq!(s.action_space().kind, ActionSpaceKind::LookAndSelect);
    let offered: Vec<&Action> = s.action_space().actions.iter().collect();
    assert!(offered
        .iter()
        .any(|a| matches!(a, Action::SelectCard { card, .. } if *card == strike)));
    assert!(!offered
        .iter()
        .any(|a| matches!(a, Action::SelectCard { card, .. } if *card == ogre)));
    choose(
        &mut s,
        |a| matches!(a, Action::SelectCard { card, .. } if *card == strike),
    );
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    if s.action_space().kind == ActionSpaceKind::PayOrNot {
        choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    }
    assert_eq!(s.game().state.zones.zone_of(strike), Some(ZoneType::Stack));
    settle(&mut s);
    s.assert_life(1, 17);
}

/// Solstice Revelations exiles to the first nonland card and offers it free
/// only when its mana value is below the Mountains you control; otherwise it
/// goes to hand.
#[test]
fn solstice_revelations_casts_cheap_spells_free_and_keeps_the_rest() {
    let mut s = setup(&["Solstice Revelations", "Lightning Strike"]);
    let on_top = put_on_top;
    let strike = library_card(&s, 0, "Lightning Strike");
    on_top(&mut s, strike);
    let land = library_card(&s, 0, "Plains");
    on_top(&mut s, land);
    for _ in 0..2 {
        s.force_permanent_on_battlefield(0, "Mountain");
    }
    // Two Mountains: Strike's mana value 2 is not less, so it goes to hand.
    cast(&mut s, "Solstice Revelations");
    settle(&mut s);
    assert_eq!(s.game().state.zones.zone_of(land), Some(ZoneType::Exile));
    assert_eq!(s.game().state.zones.zone_of(strike), Some(ZoneType::Hand));

    // Three Mountains: the exiled Strike is offered and cast for free.
    on_top(&mut s, strike);
    s.force_permanent_on_battlefield(0, "Mountain");
    let solstice = cast(&mut s, "Solstice Revelations");
    resolve_top(&mut s);
    assert_eq!(s.action_space().kind, ActionSpaceKind::LookAndSelect);
    choose(
        &mut s,
        |a| matches!(a, Action::SelectCard { card, .. } if *card == strike),
    );
    assert_eq!(
        s.game().state.zones.zone_of(solstice),
        Some(ZoneType::Graveyard)
    );
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    if s.action_space().kind == ActionSpaceKind::PayOrNot {
        choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    }
    settle(&mut s);
    s.assert_life(1, 17);
    assert_eq!(
        s.game().state.zones.zone_of(strike),
        Some(ZoneType::Graveyard)
    );

    // Flashback from the graveyard for {6}{R}, then it is exiled.
    s.set_player_mana_pool(0, "6R");
    on_top(&mut s, strike);
    cast_card(&mut s, solstice);
    resolve_top(&mut s);
    choose(&mut s, |a| matches!(a, Action::Decline { .. }));
    assert_eq!(s.game().state.zones.zone_of(strike), Some(ZoneType::Hand));
    assert_eq!(
        s.game().state.zones.zone_of(solstice),
        Some(ZoneType::Exile)
    );
}

/// Pass priority, targeting the opponent when a trigger asks, until the
/// action space is `kind`.
fn advance_targeting_opponent(s: &mut Scenario, kind: ActionSpaceKind) {
    for _ in 0..10 {
        match s.action_space().kind {
            current if current == kind => return,
            ActionSpaceKind::ChooseTarget => {
                assert!(s.choose_target(Target::Player(PlayerId(1))));
            }
            _ => s.pass_priority(),
        }
    }
    assert_eq!(s.action_space().kind, kind);
}

/// Hama mills the opponent three, exiles a noncreature nonland card from
/// their graveyard, and lets you cast it on your turn by waterbending its
/// mana value.
#[test]
fn hama_exiles_a_milled_spell_and_casts_it_by_waterbending() {
    let mut s = setup(&["Hama, the Bloodbender", "Lightning Strike", "Gray Ogre"]);
    let strike = library_card(&s, 1, "Lightning Strike");
    put_on_top(&mut s, strike);
    let helpers: Vec<PermanentId> = (0..2)
        .map(|_| s.force_permanent_on_battlefield(0, "Gray Ogre"))
        .collect();
    cast(&mut s, "Hama, the Bloodbender");
    advance_targeting_opponent(&mut s, ActionSpaceKind::LookAndSelect);
    assert_eq!(s.zone_size(1, ZoneType::Graveyard), 3);
    assert!(s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::Decline { .. })));
    choose(
        &mut s,
        |a| matches!(a, Action::SelectCard { card, .. } if *card == strike),
    );
    settle(&mut s);
    assert_eq!(s.game().state.zones.zone_of(strike), Some(ZoneType::Exile));

    // Waterbend {2}: both Ogres tap; no mana is spent.
    s.set_player_mana_pool(0, "");
    assert!(can_cast(&mut s, strike));
    cast_card(&mut s, strike);
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    assert_eq!(s.action_space().kind, ActionSpaceKind::Waterbend);
    for helper in &helpers {
        choose(
            &mut s,
            |a| matches!(a, Action::WaterbendTap { permanent, .. } if permanent == helper),
        );
    }
    settle(&mut s);
    s.assert_life(1, 17);
    assert_eq!(
        s.game().state.zones.zone_of(strike),
        Some(ZoneType::Graveyard)
    );
    assert_eq!(s.game().state.cards[strike].owner, PlayerId(1));
}

/// Azula's opponent exiles a nontoken creature and then a nonland graveyard
/// card; on your turn you may cast either with any type of mana. Their
/// creature enters under your control (CR 110.2) and, when it dies, goes to
/// its owner's graveyard.
#[test]
fn azula_has_the_opponent_exile_and_casts_with_any_mana() {
    let mut s = setup(&["Azula, Cunning Usurper", "Lightning Strike", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let ogre_card = s.game().state.permanents[ogre].as_ref().unwrap().card;
    let strike = library_card(&s, 1, "Lightning Strike");
    s.game_mut().move_card(strike, ZoneType::Graveyard);
    cast(&mut s, "Azula, Cunning Usurper");
    advance_targeting_opponent(&mut s, ActionSpaceKind::LookAndSelect);
    assert_eq!(s.action_space().player, Some(PlayerId(1)));
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::Decline { .. })));
    choose(
        &mut s,
        |a| matches!(a, Action::SelectCard { card, .. } if *card == ogre_card),
    );
    assert_eq!(s.action_space().player, Some(PlayerId(1)));
    choose(
        &mut s,
        |a| matches!(a, Action::SelectCard { card, .. } if *card == strike),
    );
    settle(&mut s);
    assert_eq!(
        s.game().state.zones.zone_of(ogre_card),
        Some(ZoneType::Exile)
    );
    assert_eq!(s.game().state.zones.zone_of(strike), Some(ZoneType::Exile));

    s.set_player_mana_pool(0, "2");
    assert!(!can_cast(&mut s, ogre_card));
    assert!(can_cast(&mut s, strike));
    cast_card(&mut s, strike);
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    settle(&mut s);
    s.assert_life(1, 17);
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);

    // Gray Ogre ({2}{R}) for three colorless: it enters under your control.
    s.set_player_mana_pool(0, "3");
    assert!(can_cast(&mut s, ogre_card));
    cast_card(&mut s, ogre_card);
    settle(&mut s);
    let stolen = s.game().state.card_to_permanent[ogre_card].expect("Ogre on battlefield");
    let permanent = s.game().state.permanents[stolen].as_ref().unwrap();
    assert_eq!(permanent.controller, PlayerId(0));
    assert_eq!(s.game().state.cards[ogre_card].owner, PlayerId(1));
    assert!(s
        .game()
        .state
        .zones
        .zone_cards(ZoneType::Battlefield, PlayerId(0))
        .contains(&ogre_card));
    let observation = Observation::for_player(s.game(), PlayerId(0));
    let id = s.game().state.cards[ogre_card].id.0 as i32;
    assert!(observation
        .agent_permanents
        .iter()
        .any(|p| p.id == permanent.id.0 as i32));
    assert!(observation
        .agent_cards
        .iter()
        .any(|c| c.id == id && c.zone == ZoneType::Battlefield));

    // It dies into its owner's graveyard.
    s.set_player_mana_pool(0, "1R");
    cast(&mut s, "Lightning Strike");
    assert!(s.choose_target(Target::Permanent(stolen)));
    settle(&mut s);
    assert!(s
        .game()
        .state
        .zones
        .contains(ogre_card, ZoneType::Graveyard, PlayerId(1)));
}

/// Hama can exile an opposing noncreature permanent card: cast by
/// waterbending, Prophetic Prism enters under your control, its enters
/// trigger draws for you, and its mana ability is yours to activate.
#[test]
fn hama_casts_an_opponents_artifact_under_your_control() {
    let mut s = setup(&["Hama, the Bloodbender", "Prophetic Prism", "Gray Ogre"]);
    let prism = library_card(&s, 1, "Prophetic Prism");
    put_on_top(&mut s, prism);
    let helpers: Vec<PermanentId> = (0..2)
        .map(|_| s.force_permanent_on_battlefield(0, "Gray Ogre"))
        .collect();
    cast(&mut s, "Hama, the Bloodbender");
    advance_targeting_opponent(&mut s, ActionSpaceKind::LookAndSelect);
    choose(
        &mut s,
        |a| matches!(a, Action::SelectCard { card, .. } if *card == prism),
    );
    settle(&mut s);
    assert_eq!(s.game().state.zones.zone_of(prism), Some(ZoneType::Exile));

    s.set_player_mana_pool(0, "");
    let hands = (
        s.zone_size(0, ZoneType::Hand),
        s.zone_size(1, ZoneType::Hand),
    );
    assert!(can_cast(&mut s, prism));
    cast_card(&mut s, prism);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Waterbend);
    for helper in &helpers {
        choose(
            &mut s,
            |a| matches!(a, Action::WaterbendTap { permanent, .. } if permanent == helper),
        );
    }
    settle(&mut s);
    let artifact = s.game().state.card_to_permanent[prism].expect("Prism on battlefield");
    assert_eq!(
        s.game().state.permanents[artifact]
            .as_ref()
            .unwrap()
            .controller,
        PlayerId(0)
    );
    assert_eq!(s.zone_size(0, ZoneType::Hand), hands.0 + 1);
    assert_eq!(s.zone_size(1, ZoneType::Hand), hands.1);
    s.set_player_mana_pool(0, "1");
    assert!(can_activate_any(&mut s, artifact));
    // Each player's view lists the Prism on its controller's side.
    for viewer in [PlayerId(0), PlayerId(1)] {
        let view = Observation::for_player(s.game(), viewer);
        assert!(view.validate());
        let mine = view
            .agent_cards
            .iter()
            .any(|card| card.zone == ZoneType::Battlefield && card.name == "Prophetic Prism");
        assert_eq!(mine, viewer == PlayerId(0));
    }
}

/// Sozin's Comet grants firebending 5 to the creatures you control as it
/// resolves: one attacking adds five red mana that lasts until end of combat.
/// A creature arriving later gains nothing (CR 611.2c).
#[test]
fn sozins_comet_grants_firebending_five_until_end_of_turn() {
    let mut s = setup(&["Sozin's Comet", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    ready(&mut s, ogre);
    cast(&mut s, "Sozin's Comet");
    settle(&mut s);
    let late = s.force_permanent_on_battlefield(0, "Gray Ogre");
    ready(&mut s, late);
    s.set_player_mana_pool(0, "");
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    for _ in 0..2 {
        choose(&mut s, |a| {
            matches!(a, Action::DeclareAttacker { attack: true, .. })
        });
    }
    pass_until(&mut s, ActionSpaceKind::Priority);
    while !s.game().state.stack_objects.is_empty() {
        s.pass_priority();
    }
    assert_eq!(s.current_step(), StepKind::DeclareAttackers);
    let pool = &s.game().state.players[0].combat_mana_pool;
    assert_eq!(pool.total(), 5);
}

fn foretell_offered(s: &mut Scenario, card: CardId) -> bool {
    s.game_mut().scenario_refresh_priority().unwrap();
    s.action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::Foretell { card: c, .. } if *c == card))
}

/// Foretell (CR 702.143a): during its owner's turn, pay {2} and exile Sozin's
/// Comet face down without using the stack. Only its owner sees which card it
/// is; the move is public, its identity is not. On a later turn its owner
/// casts it for {2}{R}.
#[test]
fn sozins_comet_foretells_face_down_and_casts_on_a_later_turn() {
    let mut s = setup(&["Sozin's Comet"]);
    let comet = library_card(&s, 0, "Sozin's Comet");
    s.game_mut().move_card(comet, ZoneType::Hand);
    s.set_player_mana_pool(0, "3");
    assert!(foretell_offered(&mut s, comet));
    s.drain_events();
    choose(
        &mut s,
        |a| matches!(a, Action::Foretell { card, .. } if *card == comet),
    );
    assert_eq!(s.game().state.zones.zone_of(comet), Some(ZoneType::Exile));
    assert!(s.game().state.stack_objects.is_empty());
    assert_eq!(s.game().state.players[0].mana_pool.total(), 1);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);

    // The arrival is public but names no card.
    let events = s.drain_events();
    assert!(events.iter().any(
        |e| matches!(e, GameEvent::CardMoved { card, face_down: true, .. } if *card == comet)
    ));
    let recent = Observation::new(s.game(), &events).recent_events;
    let arrival = recent
        .iter()
        .find(|e| e.to_zone == ZoneType::Exile as i32)
        .expect("exile arrival");
    assert_eq!(arrival.source_kind, EventEntityKind::None as i32);

    let id = s.game().state.cards[comet].id.0 as i32;
    let owner = Observation::for_player(s.game(), PlayerId(0));
    assert!(owner.agent_cards.iter().any(|c| c.id == id && c.face_down));
    let opponent = Observation::for_player(s.game(), PlayerId(1));
    assert!(opponent.opponent_cards.iter().all(|c| c.id != id));
    assert_eq!(opponent.opponent.zone_counts[ZoneType::Exile as usize], 1);
    let definition = s.game().state.cards[comet].definition_id.0;
    assert!(opponent.policy_history.iter().all(|row| !matches!(
        row.source,
        HistoryReference::Card { definition: d, .. } if d == definition
    )));

    // Not castable this turn; castable for {2}{R} on its owner's next turn.
    s.set_player_mana_pool(0, "20RRRR");
    assert!(!can_cast(&mut s, comet));
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    s.set_player_mana_pool(0, "2R");
    assert!(can_cast(&mut s, comet));
    cast_card(&mut s, comet);
    settle(&mut s);
    assert_eq!(
        s.game().state.zones.zone_of(comet),
        Some(ZoneType::Graveyard)
    );
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
}

/// Foretell is a special action of its owner's turn only: holding priority
/// during another player's turn is not enough.
#[test]
fn foretell_is_unavailable_on_an_opponents_turn() {
    let mut s = setup(&["Sozin's Comet"]);
    let comet = library_card(&s, 1, "Sozin's Comet");
    s.game_mut().move_card(comet, ZoneType::Hand);
    s.set_player_mana_pool(1, "20");
    s.pass_priority();
    assert_eq!(s.action_space().player, Some(PlayerId(1)));
    assert!(!foretell_offered(&mut s, comet));
}

/// Determinization resamples an opponent's face-down foretold card from the
/// unseen cards with foretell, keeping its grant on the exiled card.
#[test]
fn determinize_resamples_an_opponents_foretold_card() {
    let mut s = setup(&["Sozin's Comet", "Gray Ogre"]);
    let comet = library_card(&s, 0, "Sozin's Comet");
    s.game_mut().move_card(comet, ZoneType::Hand);
    assert!(foretell_offered(&mut s, comet));
    choose(
        &mut s,
        |a| matches!(a, Action::Foretell { card, .. } if *card == comet),
    );
    let mut resampled = false;
    for seed in 0..32 {
        let mut game = s.game().clone();
        game.determinize(PlayerId(1), seed);
        let exiled = game
            .state
            .zones
            .zone_cards(ZoneType::Exile, PlayerId(0))
            .clone();
        assert_eq!(exiled.len(), 1);
        let foretold = exiled[0];
        assert_eq!(game.state.cards[foretold].name, "Sozin's Comet");
        let grant = &game.state.cast_permissions.exile_grants[0];
        assert_eq!(Some(grant.object), game.current_object_ref(foretold));
        resampled |= foretold != comet;
        if foretold != comet {
            assert!(matches!(
                game.state.zones.zone_of(comet),
                Some(ZoneType::Hand | ZoneType::Library)
            ));
        }
    }
    assert!(resampled);
}

/// The TLA locations with "enters tapped unless you control a basic land",
/// each with a one-drop it can pay for alone.
const LOCATIONS: [(&str, &str, usize); 4] = [
    ("Abandoned Air Temple", "Healer's Hawk", 0),
    ("Agna Qel'a", "Wind Drake", 2),
    ("Ba Sing Se", "Llanowar Elves", 0),
    ("Realm of Koh", "Typhoid Rats", 0),
];

/// CR 614.1c, 614.12: the condition reads the entering player's battlefield
/// as the land moves. Opposing basics and nonbasic lands do not count.
#[test]
fn locations_enter_tapped_unless_you_control_a_basic_land() {
    for (name, _, _) in LOCATIONS {
        let mut s = setup_unfunded(&[name, "Agna Qel'a"]);
        let alone = play_land(&mut s, name);
        assert!(tapped(&s, alone), "{name} enters tapped with no lands");

        let mut s = setup_unfunded(&[name, "Kyoshi Village"]);
        s.force_permanent_on_battlefield(1, "Forest");
        s.force_permanent_on_battlefield(0, "Kyoshi Village");
        let without_basic = play_land(&mut s, name);
        assert!(
            tapped(&s, without_basic),
            "{name}: an opposing basic or own nonbasic land does not count"
        );

        let mut s = setup_unfunded(&[name]);
        s.force_permanent_on_battlefield(0, "Plains");
        let with_basic = play_land(&mut s, name);
        assert!(!tapped(&s, with_basic), "{name} enters untapped");
    }
}

#[test]
fn locations_tap_for_their_color() {
    for (name, spell, generic) in LOCATIONS {
        let mut s = setup_unfunded(&[name, spell]);
        let basic = s.force_permanent_on_battlefield(0, "Plains");
        let land = play_land(&mut s, name);
        s.game_mut().state.permanents[basic]
            .as_mut()
            .unwrap()
            .tapped = true;
        for _ in 0..generic {
            s.force_permanent_on_battlefield(0, "Plains");
            s.force_permanent_on_battlefield(0, "Plains");
        }
        let id = cast(&mut s, spell);
        settle(&mut s);
        assert!(tapped(&s, land), "{name} pays for {spell}");
        assert_eq!(
            s.game().state.zones.zone_of(id),
            Some(ZoneType::Battlefield)
        );
    }
}

#[test]
fn abandoned_air_temple_counters_each_creature_you_control_only() {
    let mut s = setup_unfunded(&["Abandoned Air Temple", "Gray Ogre"]);
    let temple = s.force_permanent_on_battlefield(0, "Abandoned Air Temple");
    let first = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let second = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.set_player_mana_pool(0, "2W");
    assert!(!can_activate(&mut s, temple, 0), "{{3}}{{W}} is required");
    s.set_player_mana_pool(0, "3W");
    activate(&mut s, temple, 0);
    assert!(tapped(&s, temple));
    settle(&mut s);
    assert_eq!(counters(&s, first), 1);
    assert_eq!(counters(&s, second), 1);
    assert_eq!(counters(&s, theirs), 0);
    assert_eq!(counters(&s, temple), 0, "a noncreature land gets none");
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
}

#[test]
fn agna_qel_a_draws_then_discards() {
    let mut s = setup_unfunded(&["Agna Qel'a", "Gray Ogre"]);
    let agna = s.force_permanent_on_battlefield(0, "Agna Qel'a");
    let kept = to_hand(&mut s, "Gray Ogre");
    let top = *s
        .game()
        .state
        .zones
        .zone_cards(ZoneType::Library, PlayerId(0))
        .last()
        .unwrap();
    s.set_player_mana_pool(0, "2U");
    activate(&mut s, agna, 0);
    pass_until(&mut s, ActionSpaceKind::Discard);
    assert_eq!(s.game().state.zones.zone_of(top), Some(ZoneType::Hand));
    // The drawn card is itself a legal discard; discard the older card.
    assert!(has_action(
        &s,
        |a| matches!(a, Action::SelectCard { card, .. } if *card == top)
    ));
    assert!(!has_action(&s, |a| matches!(a, Action::Decline { .. })));
    choose(
        &mut s,
        |a| matches!(a, Action::SelectCard { card, .. } if *card == kept),
    );
    settle(&mut s);
    assert_eq!(
        s.game().state.zones.zone_of(kept),
        Some(ZoneType::Graveyard)
    );
    assert_eq!(s.zone_size(0, ZoneType::Hand), 1);
}

/// CR 602.5d: "activate only as a sorcery" needs your main phase and an
/// empty stack.
#[test]
fn ba_sing_se_earthbends_only_at_sorcery_timing() {
    let mut s = setup_unfunded(&["Ba Sing Se", "Abandon Attachments"]);
    let city = s.force_permanent_on_battlefield(0, "Ba Sing Se");
    let forest = s.force_permanent_on_battlefield(0, "Forest");
    s.set_player_mana_pool(0, "2GUU");

    cast(&mut s, "Abandon Attachments");
    assert_eq!(s.action_space().player, Some(PlayerId(0)));
    assert!(
        !can_activate(&mut s, city, 0),
        "not with a spell on the stack"
    );
    settle(&mut s);

    s.set_player_mana_pool(0, "2G");
    assert!(can_activate(&mut s, city, 0));
    activate(&mut s, city, 0);
    assert!(s.choose_target(Target::Permanent(forest)));
    settle(&mut s);
    assert!(tapped(&s, city));
    assert_eq!(counters(&s, forest), 2);
    let land = s.game().state.permanents[forest].as_ref().unwrap();
    assert!(land.animated, "the land becomes a creature");

    let mut s = setup_unfunded(&["Ba Sing Se"]);
    let city = s.force_permanent_on_battlefield(0, "Ba Sing Se");
    s.force_permanent_on_battlefield(0, "Forest");
    s.advance_to_active_step(1, StepKind::Main);
    if s.action_space().player == Some(PlayerId(1)) {
        s.pass_priority();
    }
    assert_eq!(s.action_space().player, Some(PlayerId(0)));
    s.set_player_mana_pool(0, "2G");
    assert!(!can_activate(&mut s, city, 0), "not on an opponent's turn");
}

#[test]
fn realm_of_koh_creates_a_spirit() {
    let mut s = setup_unfunded(&["Realm of Koh"]);
    let realm = s.force_permanent_on_battlefield(0, "Realm of Koh");
    s.set_player_mana_pool(0, "3B");
    activate(&mut s, realm, 0);
    settle(&mut s);
    assert!(tapped(&s, realm));
    assert_eq!(s.battlefield_permanents_named(0, "Spirit").len(), 1);
    assert!(s.battlefield_permanents_named(1, "Spirit").is_empty());
}

/// The keywords an observer sees on a battlefield permanent.
fn observed(s: &Scenario, id: PermanentId) -> PermanentData {
    let permanent = s.game().state.permanents[id].as_ref().unwrap();
    let observation = Observation::for_player(s.game(), PlayerId(0));
    observation
        .agent_permanents
        .iter()
        .chain(&observation.opponent_permanents)
        .find(|p| p.id == permanent.id.0 as i32)
        .cloned()
        .unwrap()
}

fn observed_keywords(s: &Scenario, id: PermanentId) -> KeywordData {
    observed(s, id).keywords
}

fn mark_damage(s: &mut Scenario, id: PermanentId, damage: i32) {
    s.game_mut().state.permanents[id].as_mut().unwrap().damage = damage;
}

/// CR 702.12b, 704.5g: lethal damage does not destroy an indestructible
/// permanent. The grant covers only other permanents its controller controls
/// and ends when the Walls leave.
#[test]
fn the_walls_of_ba_sing_se_protect_other_permanents_you_control() {
    let mut s = setup_unfunded(&["The Walls of Ba Sing Se", "Gray Ogre"]);
    let walls = s.force_permanent_on_battlefield(0, "The Walls of Ba Sing Se");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let land = s.force_permanent_on_battlefield(0, "Forest");
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    assert!(observed_keywords(&s, walls).defender);
    assert!(!observed_keywords(&s, walls).indestructible);
    assert!(observed_keywords(&s, land).indestructible);
    assert!(!observed_keywords(&s, theirs).indestructible);

    let ogre_card = card_of(&s, ogre);
    let theirs_card = card_of(&s, theirs);
    mark_damage(&mut s, ogre, 5);
    mark_damage(&mut s, theirs, 5);
    s.pass_priority();
    assert_eq!(
        s.game().state.zones.zone_of(ogre_card),
        Some(ZoneType::Battlefield)
    );
    assert_eq!(
        s.game().state.zones.zone_of(theirs_card),
        Some(ZoneType::Graveyard)
    );

    let walls_card = card_of(&s, walls);
    s.game_mut().move_card(walls_card, ZoneType::Graveyard);
    mark_damage(&mut s, ogre, 5);
    s.game_mut().scenario_refresh_priority().unwrap();
    s.pass_priority();
    assert_eq!(
        s.game().state.zones.zone_of(ogre_card),
        Some(ZoneType::Graveyard),
        "the grant ends with its source"
    );
}

/// The anthem reads effective flying, including a temporary grant.
#[test]
fn air_nomad_legacy_makes_a_clue_and_pumps_your_fliers() {
    let mut s = setup_unfunded(&["Air Nomad Legacy", "Wind Drake", "Gray Ogre"]);
    let drake = s.force_permanent_on_battlefield(0, "Wind Drake");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let granted = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let theirs = s.force_permanent_on_battlefield(1, "Wind Drake");
    s.game_mut().state.permanents[granted]
        .as_mut()
        .unwrap()
        .temp_keywords
        .flying = true;
    s.set_player_mana_pool(0, "WU");
    cast(&mut s, "Air Nomad Legacy");
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Clue").len(), 1);
    assert_eq!(s.game().effective_pt(drake), (3, 3));
    assert_eq!(s.game().effective_pt(granted), (3, 3));
    assert_eq!(s.game().effective_pt(ogre), (2, 2));
    assert_eq!(s.game().effective_pt(theirs), (2, 2));
}

/// Earthbend's counters bring the land under the trample grant; a creature
/// without +1/+1 counters (Badgermole, a stunned Ogre) or an opponent's does
/// not have it.
#[test]
fn badgermole_earthbends_and_grants_trample_to_countered_creatures() {
    let mut s = setup_unfunded(&["Badgermole", "Gray Ogre"]);
    let forest = s.force_permanent_on_battlefield(0, "Forest");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let countered = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let stunned = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    for id in [countered, theirs] {
        s.game_mut().state.permanents[id]
            .as_mut()
            .unwrap()
            .plus1_counters = 1;
    }
    add_counters(&mut s, stunned, CounterKind::Stun, 1);
    s.set_player_mana_pool(0, "4G");
    let card = cast(&mut s, "Badgermole");
    for _ in 0..4 {
        if s.choose_target(Target::Permanent(forest)) {
            break;
        }
        s.pass_priority();
    }
    settle(&mut s);
    let badgermole = s.game().state.card_to_permanent[card].unwrap();
    assert_eq!(counters(&s, forest), 2);
    assert!(s.game().state.permanents[forest].as_ref().unwrap().animated);
    assert!(observed_keywords(&s, forest).trample);
    assert!(observed_keywords(&s, countered).trample);
    assert!(!observed_keywords(&s, ogre).trample);
    assert!(!observed_keywords(&s, stunned).trample);
    assert!(!observed_keywords(&s, badgermole).trample);
    assert!(!observed_keywords(&s, theirs).trample);
}

/// CR 205.3m: the bonus counts each creature type once, including granted
/// types, caps at ten, and skips Humans, opponents and animated lands'
/// land types.
#[test]
fn diligent_zookeeper_counts_creature_types_of_non_humans() {
    let mut s = setup_unfunded(&["Diligent Zookeeper", "Gray Ogre", "Momo, Friendly Flier"]);
    let keeper = s.force_permanent_on_battlefield(0, "Diligent Zookeeper");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let momo = s.force_permanent_on_battlefield(0, "Momo, Friendly Flier");
    let many = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let land = s.force_permanent_on_battlefield(0, "Forest");
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.game_mut().state.permanents[many]
        .as_mut()
        .unwrap()
        .temp_subtypes = (0..12).map(|i| format!("Type{i}")).collect();
    {
        let land = s.game_mut().state.permanents[land].as_mut().unwrap();
        land.animated = true;
        land.plus1_counters = 1;
    }
    assert_eq!(s.game().effective_pt(keeper), (4, 4));
    assert_eq!(s.game().effective_pt(ogre), (3, 3));
    assert_eq!(s.game().effective_pt(momo), (4, 4), "Lemur, Bat and Ally");
    assert_eq!(s.game().effective_pt(many), (12, 12), "capped at ten");
    assert_eq!(
        s.game().effective_pt(land),
        (1, 1),
        "Forest is not a creature type"
    );
    assert_eq!(s.game().effective_pt(theirs), (2, 2));

    s.game_mut().state.permanents[ogre]
        .as_mut()
        .unwrap()
        .temp_subtypes = vec!["Ally".into(), "Ogre".into()];
    assert_eq!(
        s.game().effective_pt(ogre),
        (4, 4),
        "a granted type counts once"
    );
}

/// The keywords track the controller's draw count this turn.
#[test]
fn foggy_swamp_hunters_gain_lifelink_and_menace_after_two_draws() {
    let mut s = setup_unfunded(&["Foggy Swamp Hunters"]);
    let hunters = s.force_permanent_on_battlefield(0, "Foggy Swamp Hunters");
    s.game_mut().state.turn.cards_drawn_this_turn[0] = 1;
    let keywords = observed_keywords(&s, hunters);
    assert!(!keywords.lifelink && !keywords.menace);
    s.game_mut().state.turn.cards_drawn_this_turn[0] = 2;
    let keywords = observed_keywords(&s, hunters);
    assert!(keywords.lifelink && keywords.menace);
    s.game_mut().state.turn.cards_drawn_this_turn[1] = 5;
    s.game_mut().state.turn.cards_drawn_this_turn[0] = 0;
    assert!(
        !observed_keywords(&s, hunters).lifelink,
        "only its controller's draws count"
    );
}

/// CR 122.1d: the stun counter replaces the target's next untap, then is gone.
#[test]
fn rowdy_snowballers_stun_replaces_the_next_untap() {
    let mut s = setup_unfunded(&["Rowdy Snowballers", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.set_player_mana_pool(0, "2U");
    cast(&mut s, "Rowdy Snowballers");
    s.pass_priority();
    s.choose_target(Target::Permanent(ogre));
    settle(&mut s);
    assert!(tapped(&s, ogre));
    assert_eq!(counters_of(&s, ogre, CounterKind::Stun), 1);

    s.advance_to_active_step(1, StepKind::Upkeep);
    assert!(
        tapped(&s, ogre),
        "the untap step removes the stun counter instead"
    );
    assert_eq!(counters_of(&s, ogre, CounterKind::Stun), 0);

    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(1, StepKind::Upkeep);
    assert!(!tapped(&s, ogre), "the next untap step untaps it");
}

/// CR 702.177a, 122.1b: exhaust activates once; the double strike counter
/// grants double strike for as long as it remains.
#[test]
fn mai_jaded_edge_exhausts_into_a_double_strike_counter() {
    let mut s = setup_unfunded(&["Mai, Jaded Edge"]);
    let mai = s.force_permanent_on_battlefield(0, "Mai, Jaded Edge");
    assert!(!observed_keywords(&s, mai).double_strike);
    s.set_player_mana_pool(0, "3");
    activate(&mut s, mai, 0);
    settle(&mut s);
    assert_eq!(counters_of(&s, mai, CounterKind::DoubleStrike), 1);
    assert!(observed_keywords(&s, mai).double_strike);

    s.set_player_mana_pool(0, "3");
    assert!(!can_activate(&mut s, mai, 0), "exhaust activates only once");

    ready(&mut s, mai);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::End);
    s.assert_life(1, 18);
}

fn choose_mode(s: &mut Scenario, mode: usize) {
    choose(
        s,
        |a| matches!(a, Action::ChooseMode { mode: m, .. } if *m == mode),
    );
}

fn add_counters(s: &mut Scenario, id: PermanentId, kind: CounterKind, count: i32) {
    s.game_mut().state.permanents[id]
        .as_mut()
        .unwrap()
        .change_counters(kind, count);
}

#[test]
fn heartless_act_destroys_only_a_creature_with_no_counters() {
    let mut s = setup_unfunded(&["Heartless Act", "Gray Ogre"]);
    let plain = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let stunned = s.force_permanent_on_battlefield(1, "Gray Ogre");
    add_counters(&mut s, stunned, CounterKind::Stun, 1);
    s.set_player_mana_pool(0, "1B");
    cast(&mut s, "Heartless Act");
    choose_mode(&mut s, 0);
    assert!(
        !s.choose_target(Target::Permanent(stunned)),
        "a stun counter is a counter"
    );
    assert!(s.choose_target(Target::Permanent(plain)));
    settle(&mut s);
    assert!(s.game().state.permanents[plain].is_none());
    assert!(s.game().state.permanents[stunned].is_some());
}

/// The caster chooses each removed counter's kind and may stop early.
#[test]
fn heartless_act_removes_up_to_three_counters_of_chosen_kinds() {
    let mut s = setup_unfunded(&["Heartless Act", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    add_counters(&mut s, ogre, CounterKind::PlusOnePlusOne, 2);
    add_counters(&mut s, ogre, CounterKind::Stun, 2);
    s.set_player_mana_pool(0, "1B");
    cast(&mut s, "Heartless Act");
    choose_mode(&mut s, 1);
    assert!(s.choose_target(Target::Permanent(ogre)));
    pass_until(&mut s, ActionSpaceKind::Modal);
    // Modes: +1/+1, stun, stop.
    assert_eq!(s.action_space().actions.len(), 3);
    choose_mode(&mut s, 1);
    choose_mode(&mut s, 1);
    // Stun counters are gone; only +1/+1 and stop remain.
    assert_eq!(s.action_space().actions.len(), 2);
    choose_mode(&mut s, 1);
    settle(&mut s);
    assert_eq!(counters_of(&s, ogre, CounterKind::Stun), 0);
    assert_eq!(counters(&s, ogre), 2);

    let mut s = setup_unfunded(&["Heartless Act", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    add_counters(&mut s, ogre, CounterKind::PlusOnePlusOne, 5);
    s.set_player_mana_pool(0, "1B");
    cast(&mut s, "Heartless Act");
    choose_mode(&mut s, 1);
    assert!(s.choose_target(Target::Permanent(ogre)));
    pass_until(&mut s, ActionSpaceKind::Modal);
    for _ in 0..3 {
        choose_mode(&mut s, 0);
    }
    assert_eq!(
        s.action_space().kind,
        ActionSpaceKind::Priority,
        "three is the limit"
    );
    assert_eq!(counters(&s, ogre), 2);
}

/// CR 613.4b-c: the waterbend setter replaces the printed base; a +1/+1
/// counter still applies on top of it, and the setter ends at cleanup.
#[test]
fn flexible_waterbender_sets_base_pt_under_counters_until_end_of_turn() {
    let mut s = setup_unfunded(&["Flexible Waterbender", "Gray Ogre"]);
    let bender = s.force_permanent_on_battlefield(0, "Flexible Waterbender");
    let helpers = [
        s.force_permanent_on_battlefield(0, "Gray Ogre"),
        s.force_permanent_on_battlefield(0, "Gray Ogre"),
    ];
    add_counters(&mut s, bender, CounterKind::PlusOnePlusOne, 1);
    assert_eq!(s.game().effective_pt(bender), (3, 6));

    s.set_player_mana_pool(0, "1");
    activate(&mut s, bender, 0);
    for helper in helpers {
        assert!(s.choose_waterbend_tap(helper));
    }
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    settle(&mut s);
    assert!(helpers.iter().all(|helper| tapped(&s, *helper)));
    assert_eq!(s.game().effective_pt(bender), (6, 3));

    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.game().effective_pt(bender), (3, 6));
}

/// Only base power changes; toughness keeps its printed base, and trample
/// lasts as long as the setter.
#[test]
fn turtle_duck_sets_base_power_and_gains_trample_until_end_of_turn() {
    let mut s = setup_unfunded(&["Turtle-Duck"]);
    let duck = s.force_permanent_on_battlefield(0, "Turtle-Duck");
    add_counters(&mut s, duck, CounterKind::PlusOnePlusOne, 1);
    assert!(!observed_keywords(&s, duck).trample);

    s.set_player_mana_pool(0, "3");
    activate(&mut s, duck, 0);
    settle(&mut s);
    assert_eq!(s.game().effective_pt(duck), (5, 5));
    assert!(observed_keywords(&s, duck).trample);

    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.game().effective_pt(duck), (1, 5));
    assert!(!observed_keywords(&s, duck).trample);
}

fn equip(s: &mut Scenario, equipment: PermanentId, creature: PermanentId) {
    activate(s, equipment, 0);
    assert!(s.choose_target(Target::Permanent(creature)));
    settle(s);
}

fn battlefield_token(s: &Scenario, player: usize) -> PermanentId {
    s.game()
        .state
        .zones
        .zone_cards(ZoneType::Battlefield, PlayerId(player))
        .iter()
        .find(|card| s.game().state.cards[**card].is_token)
        .and_then(|card| s.game().state.card_to_permanent[*card])
        .expect("a token")
}

/// CR 702.6a, 301.5: the token enters, then the Fan attaches to it; equip
/// moves it at sorcery timing only, and the bonus follows the Equipment.
#[test]
fn kyoshi_battle_fan_attaches_to_its_ally_token_then_equips_at_sorcery_speed() {
    let mut s = setup_unfunded(&["Kyoshi Battle Fan", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.set_player_mana_pool(0, "2");
    let fan = cast(&mut s, "Kyoshi Battle Fan");
    settle(&mut s);
    let fan = s.game().state.card_to_permanent[fan].expect("fan entered");
    let ally = battlefield_token(&s, 0);
    assert_eq!(s.game().effective_pt(ally), (2, 1));

    s.set_player_mana_pool(0, "2");
    equip(&mut s, fan, ogre);
    assert_eq!(s.game().effective_pt(ally), (1, 1));
    assert_eq!(s.game().effective_pt(ogre), (3, 2));

    s.advance_to_active_step(1, StepKind::Main);
    s.set_player_mana_pool(0, "2");
    assert!(
        !can_activate(&mut s, fan, 0),
        "equip only as a sorcery on your own turn"
    );
}

/// CR 704.5n: an Equipment whose creature leaves stays on the battlefield,
/// unattached.
#[test]
fn meteor_sword_destroys_a_permanent_and_stays_when_its_creature_dies() {
    let mut s = setup_unfunded(&["Meteor Sword", "Gray Ogre"]);
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let ours = s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.set_player_mana_pool(0, "7");
    let sword = cast(&mut s, "Meteor Sword");
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(theirs)));
    settle(&mut s);
    assert!(s.game().state.permanents[theirs].is_none());
    let sword = s.game().state.card_to_permanent[sword].expect("sword entered");

    s.set_player_mana_pool(0, "3");
    equip(&mut s, sword, ours);
    assert_eq!(s.game().effective_pt(ours), (5, 5));

    mark_damage(&mut s, ours, 5);
    s.pass_priority();
    assert!(s.game().state.permanents[ours].is_none(), "lethal damage");
    let sword = s.game().state.permanents[sword].as_ref().expect("stays");
    assert_eq!(sword.attached_to, None);
}

/// Flash; the enters trigger attaches to a creature you control and grants
/// double strike until end of turn, while +1/+1 lasts while attached.
#[test]
fn twin_blades_attaches_on_entry_and_grants_double_strike_this_turn() {
    let mut s = setup_unfunded(&["Twin Blades", "Gray Ogre"]);
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let ours = s.force_permanent_on_battlefield(0, "Gray Ogre");
    ready(&mut s, ours);
    s.advance_to_active_step(0, StepKind::BeginningOfCombat);
    s.set_player_mana_pool(0, "2R");
    cast(&mut s, "Twin Blades");
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(
        !s.choose_target(Target::Permanent(theirs)),
        "only a creature you control"
    );
    assert!(s.choose_target(Target::Permanent(ours)));
    settle(&mut s);
    assert_eq!(s.game().effective_pt(ours), (3, 3));
    assert!(observed_keywords(&s, ours).double_strike);

    s.game_mut().state.permanents[theirs]
        .as_mut()
        .unwrap()
        .tapped = true;
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::End);
    s.assert_life(1, 14);

    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.game().effective_pt(ours), (3, 3));
    assert!(!observed_keywords(&s, ours).double_strike);
}

/// Cast an Aura from player 0's hand onto `creature` and resolve it.
fn enchant(s: &mut Scenario, name: &str, mana: &str, creature: PermanentId) -> PermanentId {
    s.set_player_mana_pool(0, mana);
    let aura = cast(s, name);
    pass_until(s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(creature)));
    settle(s);
    let aura = s.game().state.card_to_permanent[aura].expect("aura entered");
    assert_eq!(
        s.game().state.permanents[aura]
            .as_ref()
            .unwrap()
            .attached_to,
        s.game().permanent_object_ref(creature),
        "CR 303.4f: the Aura enters attached to its target"
    );
    aura
}

/// Whether the current combat decision offers `id` as an attacker or blocker.
fn offered_in_combat(s: &Scenario, id: PermanentId) -> bool {
    has_action(s, |a| {
        matches!(a, Action::DeclareAttacker { permanent, .. } if *permanent == id)
            || matches!(a, Action::DeclareBlocker { blocker, .. } if *blocker == id)
    })
}

/// CR 508.1c, 509.1b: the enchanted creature is never offered as an attacker
/// or blocker. The sacrifice ability reads the creature it enchanted after
/// the Aura has left as a cost, and only on its controller's turn.
#[test]
fn path_to_redemption_pacifies_then_exiles_for_an_ally_on_your_turn() {
    let mut s = setup_unfunded(&["Path to Redemption", "Gray Ogre"]);
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let free = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let ours = s.force_permanent_on_battlefield(0, "Gray Ogre");
    for id in [theirs, free, ours] {
        ready(&mut s, id);
    }
    let path = enchant(&mut s, "Path to Redemption", "1W", theirs);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_until(
        |s| s.action_space().kind == ActionSpaceKind::DeclareBlocker,
        "blockers".into(),
    );
    assert!(offered_in_combat(&s, free));
    assert!(!offered_in_combat(&s, theirs), "can't block");

    s.advance_to_active_step(1, StepKind::Main);
    s.set_player_mana_pool(0, "5");
    assert!(!can_activate(&mut s, path, 0), "only during your turn");
    s.advance_until(
        |s| s.action_space().kind == ActionSpaceKind::DeclareAttacker,
        "attackers".into(),
    );
    assert!(offered_in_combat(&s, free));
    assert!(!offered_in_combat(&s, theirs), "can't attack");

    s.advance_to_active_step(0, StepKind::Main);
    let theirs_card = card_of(&s, theirs);
    let path_card = card_of(&s, path);
    s.set_player_mana_pool(0, "5");
    activate(&mut s, path, 0);
    settle(&mut s);
    let zones = &s.game().state.zones;
    assert_eq!(zones.zone_of(theirs_card), Some(ZoneType::Exile));
    assert_eq!(zones.zone_of(path_card), Some(ZoneType::Graveyard));
    let ally = battlefield_token(&s, 0);
    assert_eq!(s.game().effective_pt(ally), (1, 1));
}

/// CR 704.5m: an Aura whose creature leaves is put into its owner's
/// graveyard.
#[test]
fn path_to_redemption_follows_its_creature_to_the_graveyard() {
    let mut s = setup_unfunded(&["Path to Redemption", "Gray Ogre"]);
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let path = enchant(&mut s, "Path to Redemption", "1W", theirs);
    let path_card = card_of(&s, path);
    mark_damage(&mut s, theirs, 2);
    s.pass_priority();
    assert!(s.game().state.permanents[theirs].is_none(), "lethal damage");
    assert_eq!(
        s.game().state.zones.zone_of(path_card),
        Some(ZoneType::Graveyard)
    );
}

/// CR 608.3b: an Aura spell whose target is gone doesn't resolve and never
/// reaches the battlefield.
#[test]
fn path_to_redemption_without_its_target_does_not_resolve() {
    let mut s = setup_unfunded(&["Path to Redemption", "Gray Ogre"]);
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.set_player_mana_pool(0, "1W");
    let spell = cast(&mut s, "Path to Redemption");
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(theirs)));
    mark_damage(&mut s, theirs, 2);
    s.drain_events();
    settle(&mut s);
    assert!(s.game().state.permanents[theirs].is_none(), "lethal damage");
    assert_eq!(
        s.game().state.zones.zone_of(spell),
        Some(ZoneType::Graveyard)
    );
    assert!(s
        .drain_events()
        .iter()
        .any(|event| matches!(event, GameEvent::SpellCountered { card, .. } if *card == spell)));
}

/// CR 502.3: the enchanted creature stays tapped through its controller's
/// untap step. Waterbend {5} shuffles it into its owner's library, and the
/// unattached Aura then goes to the graveyard (704.5m).
#[test]
fn watery_grasp_holds_a_creature_tapped_then_shuffles_it_away() {
    let mut s = setup_unfunded(&["Watery Grasp", "Gray Ogre"]);
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let helpers = [
        s.force_permanent_on_battlefield(0, "Gray Ogre"),
        s.force_permanent_on_battlefield(0, "Gray Ogre"),
    ];
    s.game_mut().state.permanents[theirs]
        .as_mut()
        .unwrap()
        .tapped = true;
    let grasp = enchant(&mut s, "Watery Grasp", "U", theirs);
    let grasp_card = card_of(&s, grasp);

    s.advance_to_active_step(1, StepKind::Main);
    assert!(tapped(&s, theirs), "doesn't untap");
    assert!(helpers.iter().all(|helper| !tapped(&s, *helper)));

    s.advance_to_active_step(0, StepKind::Main);
    let theirs_card = card_of(&s, theirs);
    let library = s.zone_size(1, ZoneType::Library);
    s.set_player_mana_pool(0, "3");
    activate(&mut s, grasp, 0);
    for helper in helpers {
        assert!(s.choose_waterbend_tap(helper));
    }
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    settle(&mut s);
    assert_eq!(
        s.game().state.zones.zone_of(theirs_card),
        Some(ZoneType::Library)
    );
    assert_eq!(s.zone_size(1, ZoneType::Library), library + 1);
    assert_eq!(
        s.game().state.zones.zone_of(grasp_card),
        Some(ZoneType::Graveyard)
    );
}

/// CR 601.2f: the reduction depends on the chosen target, so with only
/// {1}{B} the spell is castable only onto a flier and offers no other
/// target; at full cost any creature may be enchanted.
#[test]
fn swampsnare_trap_costs_less_onto_a_flier_and_shrinks_its_creature() {
    let mut s = setup_unfunded(&["Swampsnare Trap", "Wind Drake", "Craw Wurm"]);
    let wurm = s.force_permanent_on_battlefield(1, "Craw Wurm");
    s.set_player_mana_pool(0, "1B");
    let trap = to_hand(&mut s, "Swampsnare Trap");
    assert!(
        !has_action(
            &s,
            |a| matches!(a, Action::CastSpell { card, .. } if *card == trap)
        ),
        "no flier to earn the reduction"
    );

    let drake = s.force_permanent_on_battlefield(1, "Wind Drake");
    s.game_mut().scenario_refresh_priority().unwrap();
    choose(
        &mut s,
        |a| matches!(a, Action::CastSpell { card, .. } if *card == trap),
    );
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(!has_action(&s, |a| {
        matches!(a, Action::ChooseTarget { target, .. } if *target == Target::Permanent(wurm))
    }));
    assert!(s.choose_target(Target::Permanent(drake)));
    settle(&mut s);
    assert!(s.game().state.permanents[drake].is_none(), "2/2 gets -5/-3");
    assert_eq!(
        s.game().state.zones.zone_of(trap),
        Some(ZoneType::Graveyard)
    );

    enchant(&mut s, "Swampsnare Trap", "2B", wurm);
    assert_eq!(s.game().effective_pt(wurm), (1, 1));
}

fn is_creature(s: &Scenario, id: PermanentId) -> bool {
    let permanent = s.game().state.permanents[id].as_ref().unwrap();
    permanent.is_creature(&s.game().state.cards[permanent.card])
}

/// Tap `creature` toward the pending crew cost.
fn crew_with(s: &mut Scenario, creature: PermanentId) {
    let card = card_of(s, creature);
    assert_eq!(s.action_space().kind, ActionSpaceKind::LookAndSelect);
    choose(
        s,
        |a| matches!(a, Action::SelectCard { card: c, .. } if *c == card),
    );
}

/// CR 702.122a: crew taps other creatures whose total power reaches N; the
/// Vehicle is an artifact creature with its printed P/T until end of turn,
/// then a noncreature artifact again. It dies as a creature and leaves a Clue.
#[test]
fn fire_nation_warship_crews_until_end_of_turn_and_leaves_a_clue() {
    let mut s = setup_unfunded(&["Fire Nation Warship", "Llanowar Elves"]);
    let ship = s.force_permanent_on_battlefield(0, "Fire Nation Warship");
    let first = s.force_permanent_on_battlefield(0, "Llanowar Elves");
    ready(&mut s, ship);
    assert!(!is_creature(&s, ship));
    assert!(!can_activate(&mut s, ship, 0), "power 1 can't crew 2");

    let second = s.force_permanent_on_battlefield(0, "Llanowar Elves");
    activate(&mut s, ship, 0);
    assert!(
        !has_action(&s, |a| matches!(a, Action::SelectCard { card, .. }
            if *card == card_of(&s, ship))),
        "a Vehicle doesn't crew itself"
    );
    crew_with(&mut s, first);
    crew_with(&mut s, second);
    settle(&mut s);
    assert!(tapped(&s, first) && tapped(&s, second));
    assert!(is_creature(&s, ship));
    assert!(
        observed(&s, ship).is_animated,
        "the policy sees the crewed Vehicle"
    );
    assert_eq!(s.game().effective_pt(ship), (4, 4));
    assert!(observed_keywords(&s, ship).reach);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    assert!(offered_in_combat(&s, ship));
    s.advance_to_active_step(1, StepKind::Main);
    assert!(!is_creature(&s, ship), "crew lasts until end of turn");

    s.advance_to_active_step(0, StepKind::Main);
    activate(&mut s, ship, 0);
    crew_with(&mut s, first);
    crew_with(&mut s, second);
    settle(&mut s);
    let ship_card = card_of(&s, ship);
    mark_damage(&mut s, ship, 4);
    s.pass_priority();
    settle(&mut s);
    assert_eq!(
        s.game().state.zones.zone_of(ship_card),
        Some(ZoneType::Graveyard)
    );
    let clue = battlefield_token(&s, 0);
    assert_eq!(s.game().state.cards[card_of(&s, clue)].name, "Clue");
}

/// CR 302.6: a Vehicle that came under your control this turn can't attack
/// when crewed, while a summoning-sick creature can crew (crew is not {T}).
/// The enters trigger makes a creature you control indestructible, and the
/// crewed Tank's firebending adds {R} when it attacks.
#[test]
fn tundra_tank_protects_on_entry_and_firebends_once_crewed() {
    let mut s = setup_unfunded(&["Tundra Tank", "Gray Ogre", "Llanowar Elves"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let elves = s.force_permanent_on_battlefield(0, "Llanowar Elves");
    ready(&mut s, ogre);
    s.set_player_mana_pool(0, "2B");
    let tank = cast(&mut s, "Tundra Tank");
    settle(&mut s);
    let tank = s.game().state.card_to_permanent[tank].expect("tank entered");
    assert!(observed_keywords(&s, ogre).indestructible);

    activate(&mut s, tank, 0);
    crew_with(&mut s, elves);
    settle(&mut s);
    assert!(is_creature(&s, tank));
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    assert!(!offered_in_combat(&s, tank), "summoning sick");

    s.advance_to_active_step(0, StepKind::Main);
    activate(&mut s, tank, 0);
    crew_with(&mut s, elves);
    settle(&mut s);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    choose(
        &mut s,
        |a| matches!(a, Action::DeclareAttacker { permanent, attack: true, .. } if *permanent == tank),
    );
    s.advance_until(
        |s| s.game().state.players[0].mana_pool.mana[3] == 1,
        "firebending".into(),
    );
}

/// CR 613.1d: three fire counters make the Balloon an artifact creature for
/// as long as they stay; two do not.
#[test]
fn war_balloon_flies_once_it_has_three_fire_counters() {
    let mut s = setup_unfunded(&["War Balloon"]);
    let balloon = s.force_permanent_on_battlefield(0, "War Balloon");
    for count in 1..=3 {
        assert!(!is_creature(&s, balloon));
        s.set_player_mana_pool(0, "1");
        activate(&mut s, balloon, 1);
        settle(&mut s);
        assert_eq!(counters_of(&s, balloon, CounterKind::Fire), count);
    }
    assert!(is_creature(&s, balloon));
    assert_eq!(s.game().effective_pt(balloon), (4, 3));
    assert!(observed_keywords(&s, balloon).flying);
    s.advance_to_active_step(1, StepKind::Main);
    assert!(is_creature(&s, balloon), "not limited to this turn");
}

/// "Another target": the entering Submersible can return any other nonland
/// permanent but not itself. Its exhaust waterbend makes it a 3/3 artifact
/// creature for as long as it remains, only once.
#[test]
fn invasion_submersible_bounces_another_permanent_then_surfaces_once() {
    let mut s = setup_unfunded(&["Invasion Submersible", "Gray Ogre"]);
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let theirs_card = card_of(&s, theirs);
    let helpers = [
        s.force_permanent_on_battlefield(0, "Gray Ogre"),
        s.force_permanent_on_battlefield(0, "Gray Ogre"),
    ];
    s.set_player_mana_pool(0, "2U");
    let sub = cast(&mut s, "Invasion Submersible");
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    let sub = s.game().state.card_to_permanent[sub].expect("entered");
    assert!(!s.choose_target(Target::Permanent(sub)), "another target");
    assert!(s.choose_target(Target::Permanent(theirs)));
    settle(&mut s);
    assert_eq!(
        s.game().state.zones.zone_of(theirs_card),
        Some(ZoneType::Hand)
    );
    assert!(!is_creature(&s, sub));

    s.set_player_mana_pool(0, "1");
    activate(&mut s, sub, 0);
    for helper in helpers {
        assert!(s.choose_waterbend_tap(helper));
    }
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    settle(&mut s);
    assert!(is_creature(&s, sub));
    assert_eq!(s.game().effective_pt(sub), (3, 3));
    s.advance_to_active_step(0, StepKind::Main);
    assert!(is_creature(&s, sub), "no duration");
    s.set_player_mana_pool(0, "3");
    assert!(!can_activate(&mut s, sub, 0), "exhaust");
}

fn pool_owned(s: &Scenario) -> Mana {
    s.game().state.players[0].mana_pool.clone()
}

fn untap(s: &mut Scenario, id: PermanentId) {
    s.game_mut().state.permanents[id].as_mut().unwrap().tapped = false;
}

/// CR 613.1f: lands and Allies you control have the granted ability while the
/// Guide remains; an opponent's land does not, and a summoning-sick Ally
/// can't use it (CR 302.6). The chosen color pays a real cost.
#[test]
fn great_divide_guide_grants_lands_and_allies_any_color() {
    let mut s = setup_unfunded(&["Great Divide Guide", "Healer's Hawk"]);
    let guide = s.force_permanent_on_battlefield(0, "Great Divide Guide");
    let forest = s.force_permanent_on_battlefield(0, "Forest");
    assert!(!can_activate(&mut s, guide, 0), "summoning sick");

    activate(&mut s, forest, 0);
    choose_mode(&mut s, 0);
    assert_eq!(pool_owned(&s), Mana::parse("W"));
    cast(&mut s, "Healer's Hawk");
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Healer's Hawk").len(), 1);
    assert_eq!(pool_owned(&s).total(), 0);

    ready(&mut s, guide);
    assert!(can_activate(&mut s, guide, 0));
    untap(&mut s, forest);
    let guide_card = card_of(&s, guide);
    s.game_mut().move_card(guide_card, ZoneType::Graveyard);
    assert!(
        !can_activate(&mut s, forest, 0),
        "the grant ends with its source"
    );

    let mut s = setup_unfunded(&["Great Divide Guide"]);
    s.force_permanent_on_battlefield(0, "Great Divide Guide");
    let theirs = s.force_permanent_on_battlefield(1, "Forest");
    s.advance_to_active_step(1, StepKind::Main);
    assert!(!can_activate(&mut s, theirs, 0), "only lands you control");
}

/// The output reads effective power as the ability resolves (CR 605.3a), so
/// a creature grown to four by counters doubles it; an opponent's does not.
/// Payment for a spell counts the doubled output.
#[test]
fn raucous_audience_adds_more_with_a_four_power_creature() {
    let mut s = setup_unfunded(&["Raucous Audience", "Gray Ogre"]);
    let audience = s.force_permanent_on_battlefield(0, "Raucous Audience");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    ready(&mut s, audience);
    add_counters(&mut s, theirs, CounterKind::PlusOnePlusOne, 2);
    activate(&mut s, audience, 0);
    choose_mode(&mut s, 0);
    assert_eq!(pool_owned(&s), Mana::parse("G"));

    untap(&mut s, audience);
    s.set_player_mana_pool(0, "0");
    to_hand(&mut s, "Raucous Audience");
    assert!(
        !has_action(&s, |a| matches!(a, Action::CastSpell { .. })),
        "{{G}} alone can't pay {{1}}{{G}}"
    );
    add_counters(&mut s, ogre, CounterKind::PlusOnePlusOne, 2);
    s.game_mut().scenario_refresh_priority().unwrap();
    choose(&mut s, |a| matches!(a, Action::CastSpell { .. }));
    for _ in 0..4 {
        match s.action_space().kind {
            ActionSpaceKind::Priority => break,
            ActionSpaceKind::Modal => choose_mode(&mut s, 0),
            _ if has_action(&s, |a| matches!(a, Action::PayCost { .. })) => {
                choose(&mut s, |a| matches!(a, Action::PayCost { .. }))
            }
            _ => choose(
                &mut s,
                |a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == audience),
            ),
        }
    }
    settle(&mut s);
    assert!(tapped(&s, audience));
    assert_eq!(
        s.battlefield_permanents_named(0, "Raucous Audience").len(),
        2
    );
    assert_eq!(pool_owned(&s).total(), 0);
}

/// The Tile enters tapped; X counts the largest group of your creatures
/// sharing a creature type (CR 205.3m), including a granted type, and all of
/// it is one chosen color. With no creatures it adds nothing.
#[test]
fn white_lotus_tile_adds_one_color_for_the_largest_shared_type() {
    let mut s = setup_unfunded(&["White Lotus Tile", "Gray Ogre", "Llanowar Elves"]);
    s.set_player_mana_pool(0, "4");
    let tile = cast(&mut s, "White Lotus Tile");
    settle(&mut s);
    let tile = s.game().state.card_to_permanent[tile].expect("entered");
    assert!(tapped(&s, tile));

    untap(&mut s, tile);
    activate(&mut s, tile, 0);
    choose_mode(&mut s, 0);
    assert!(tapped(&s, tile));
    assert_eq!(pool_owned(&s).total(), 0, "no creatures");

    for _ in 0..2 {
        s.force_permanent_on_battlefield(0, "Gray Ogre");
    }
    let elves = s.force_permanent_on_battlefield(0, "Llanowar Elves");
    for _ in 0..3 {
        s.force_permanent_on_battlefield(1, "Gray Ogre");
    }
    untap(&mut s, tile);
    activate(&mut s, tile, 0);
    choose_mode(&mut s, 1);
    assert_eq!(pool_owned(&s), Mana::parse("UU"));

    s.game_mut().state.permanents[elves]
        .as_mut()
        .unwrap()
        .temp_subtypes = vec!["Ogre".into()];
    s.set_player_mana_pool(0, "0");
    untap(&mut s, tile);
    activate(&mut s, tile, 0);
    choose_mode(&mut s, 3);
    assert_eq!(pool_owned(&s), Mana::parse("RRR"));
}

/// CR 508.1c, 509.1b: the Lion-Turtle can neither attack nor block until
/// three Lesson cards are in its controller's graveyard; vigilance keeps it
/// untapped once it can attack. It gains 3 life on entry and taps for any
/// color.
#[test]
fn the_lion_turtle_waits_for_three_lessons_to_fight() {
    let mut s = setup_unfunded(&["The Lion-Turtle", "Waterbending Lesson", "Gray Ogre"]);
    let attacker = s.force_permanent_on_battlefield(1, "Gray Ogre");
    ready(&mut s, attacker);
    s.set_player_mana_pool(0, "1GU");
    let turtle = cast(&mut s, "The Lion-Turtle");
    settle(&mut s);
    let turtle = s.game().state.card_to_permanent[turtle].expect("entered");
    assert_eq!(s.game().state.players[0].life, 23);
    ready(&mut s, turtle);
    activate(&mut s, turtle, 0);
    choose_mode(&mut s, 2);
    assert_eq!(pool_owned(&s), Mana::parse("B"));
    untap(&mut s, turtle);

    s.advance_until(
        |s| s.action_space().kind == ActionSpaceKind::DeclareAttacker,
        "attackers".into(),
    );
    assert!(!offered_in_combat(&s, turtle), "can't attack");
    s.advance_to_active_step(1, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_until(
        |s| s.action_space().kind == ActionSpaceKind::DeclareBlocker,
        "blockers".into(),
    );
    assert!(!offered_in_combat(&s, turtle), "can't block");

    for _ in 0..3 {
        let lesson = library_card(&s, 0, "Waterbending Lesson");
        s.game_mut().move_card(lesson, ZoneType::Graveyard);
    }
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_until(
        |s| s.action_space().kind == ActionSpaceKind::DeclareAttacker,
        "attackers".into(),
    );
    s.declare_attack();
    s.advance_until(
        |s| s.action_space().kind == ActionSpaceKind::DeclareBlocker,
        "blockers".into(),
    );
    assert!(offered_in_combat(&s, turtle), "blocks with three Lessons");
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.advance_until(
        |s| s.action_space().kind == ActionSpaceKind::DeclareAttacker,
        "attackers".into(),
    );
    choose(
        &mut s,
        |a| matches!(a, Action::DeclareAttacker { permanent, attack: true, .. } if *permanent == turtle),
    );
    assert!(!tapped(&s, turtle), "vigilance");
}

/// CR 105.2: the bonus counts each color among Allies you control once,
/// including its own green; a non-Ally or an opponent's Ally adds nothing.
/// The five-color activation earthbends 5.
#[test]
fn earthen_ally_grows_with_ally_colors_and_earthbends_five() {
    let mut s = setup_unfunded(&["Earthen Ally", "Toph, the First Metalbender", "Gray Ogre"]);
    let ally = s.force_permanent_on_battlefield(0, "Earthen Ally");
    assert_eq!(s.game().effective_pt(ally), (1, 2));
    s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.force_permanent_on_battlefield(1, "Toph, the First Metalbender");
    assert_eq!(s.game().effective_pt(ally), (1, 2));
    s.force_permanent_on_battlefield(0, "Toph, the First Metalbender");
    assert_eq!(s.game().effective_pt(ally), (3, 2), "green, red and white");

    let forest = s.force_permanent_on_battlefield(0, "Forest");
    s.set_player_mana_pool(0, "2WUBRG");
    activate(&mut s, ally, 0);
    assert!(s.choose_target(Target::Permanent(forest)));
    settle(&mut s);
    assert_eq!(counters(&s, forest), 5);
    assert_eq!(s.game().effective_pt(forest), (5, 5));
}

/// Advance to declare attackers and attack with exactly `attackers`.
fn declare_attackers(s: &mut Scenario, attackers: &[PermanentId]) {
    s.advance_until(
        |s| s.action_space().kind == ActionSpaceKind::DeclareAttacker,
        "attackers".into(),
    );
    while s.action_space().kind == ActionSpaceKind::DeclareAttacker {
        choose(s, |a| {
            matches!(a, Action::DeclareAttacker { permanent, attack, .. }
                if *attack == attackers.contains(permanent))
        });
    }
}

fn is_block_of(action: &Action, attacker: PermanentId) -> bool {
    matches!(action, Action::DeclareBlocker { attacker: Some(target), .. } if *target == attacker)
}

/// Whether the current blocker may block `attacker`.
fn may_block(s: &Scenario, attacker: PermanentId) -> bool {
    has_action(s, |a| is_block_of(a, attacker))
}

/// The entry trigger counters each creature you control. CR 509.1b: a
/// creature you control with effective power 4 or more takes one blocker at
/// most; a smaller one can still be blocked by two.
#[test]
fn flopsie_counters_your_team_and_limits_big_attackers_to_one_blocker() {
    let mut s = setup_unfunded(&["Flopsie, Bumi's Buddy", "Gray Ogre", "Llanowar Elves"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.set_player_mana_pool(0, "4GG");
    let flopsie = cast(&mut s, "Flopsie, Bumi's Buddy");
    settle(&mut s);
    let flopsie = s.game().state.card_to_permanent[flopsie].expect("entered");
    assert_eq!(counters(&s, flopsie), 1);
    assert_eq!(counters(&s, ogre), 1);
    assert_eq!(counters(&s, theirs), 0);
    for _ in 0..3 {
        s.force_permanent_on_battlefield(1, "Llanowar Elves");
    }
    ready(&mut s, flopsie);
    ready(&mut s, ogre);

    declare_attackers(&mut s, &[flopsie, ogre]);
    s.advance_until(
        |s| s.action_space().kind == ActionSpaceKind::DeclareBlocker,
        "blockers".into(),
    );
    assert!(may_block(&s, flopsie));
    choose(&mut s, |a| is_block_of(a, flopsie));
    assert!(!may_block(&s, flopsie), "a second blocker is refused");
    assert!(may_block(&s, ogre));
    choose(&mut s, |a| is_block_of(a, ogre));
    assert!(may_block(&s, ogre), "a 3/3 can be blocked by two");
}

/// X counts creatures you control with effective power 4 or more as the
/// trigger resolves, including The Boulder and a countered Ogre.
#[test]
fn the_boulder_earthbends_for_each_four_power_creature_on_attack() {
    let mut s = setup_unfunded(&["The Boulder, Ready to Rumble", "Gray Ogre"]);
    let boulder = s.force_permanent_on_battlefield(0, "The Boulder, Ready to Rumble");
    let big = s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.force_permanent_on_battlefield(0, "Gray Ogre");
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let forest = s.force_permanent_on_battlefield(0, "Forest");
    add_counters(&mut s, big, CounterKind::PlusOnePlusOne, 2);
    add_counters(&mut s, theirs, CounterKind::PlusOnePlusOne, 2);
    ready(&mut s, boulder);

    declare_attackers(&mut s, &[boulder]);
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(forest)));
    settle(&mut s);
    assert_eq!(counters(&s, forest), 2);
}

/// CR 702.3b: defender stops the Bear attacking until a Lesson card is in
/// your graveyard; it still has defender. Its entry mills two.
#[test]
fn platypus_bear_mills_two_and_attacks_once_a_lesson_is_in_your_graveyard() {
    let mut s = setup_unfunded(&["Platypus-Bear", "Abandon Attachments"]);
    s.set_player_mana_pool(0, "1G");
    let bear = cast(&mut s, "Platypus-Bear");
    let library = s.game().state.zones.size(ZoneType::Library, PlayerId(0));
    settle(&mut s);
    let bear = s.game().state.card_to_permanent[bear].expect("entered");
    assert_eq!(
        s.game().state.zones.size(ZoneType::Library, PlayerId(0)),
        library - 2
    );
    let graveyard = s
        .game()
        .state
        .zones
        .zone_cards(ZoneType::Graveyard, PlayerId(0))
        .to_vec();
    assert_eq!(graveyard.len(), 2);
    for card in graveyard {
        s.game_mut().move_card(card, ZoneType::Exile);
    }
    ready(&mut s, bear);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    assert!(!offered_in_combat(&s, bear), "defender");

    let lesson = library_card(&s, 0, "Abandon Attachments");
    s.game_mut().move_card(lesson, ZoneType::Graveyard);
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    assert!(offered_in_combat(&s, bear));
    assert!(observed_keywords(&s, bear).defender);
}

/// CR 205.1b, 613.1d: your nontoken artifacts are also lands, so earthbend
/// can target them, and an earthbent one is a 0/0 before counters. A token
/// artifact and an opponent's artifact are not lands. The end-step trigger
/// is only yours.
#[test]
fn toph_makes_your_nontoken_artifacts_lands_and_earthbends_at_your_end_step() {
    let mut s = setup_unfunded(&["Toph, the First Metalbender", "White Lotus Tile"]);
    s.force_permanent_on_battlefield(0, "Toph, the First Metalbender");
    let tile = s.force_permanent_on_battlefield(0, "White Lotus Tile");
    let theirs = s.force_permanent_on_battlefield(1, "White Lotus Tile");
    let clue = s
        .game_mut()
        .create_token("Clue", PlayerId(0), false)
        .expect("clue");
    let forest = s.force_permanent_on_battlefield(0, "Forest");

    s.advance_until(
        |s| s.action_space().kind == ActionSpaceKind::ChooseTarget,
        "end step trigger".into(),
    );
    assert_eq!(s.current_step(), StepKind::End);
    let offered = |s: &Scenario, id: PermanentId| {
        has_action(s, |a| {
            matches!(a, Action::ChooseTarget { target, .. }
            if *target == Target::Permanent(id))
        })
    };
    assert!(offered(&s, tile) && offered(&s, forest));
    assert!(!offered(&s, clue), "tokens stay nonland");
    assert!(!offered(&s, theirs), "only artifacts you control");
    assert!(s.choose_target(Target::Permanent(tile)));
    settle(&mut s);
    assert!(is_creature(&s, tile));
    assert_eq!(s.game().effective_pt(tile), (2, 2));

    s.advance_to_active_step(1, StepKind::End);
    assert_ne!(
        s.action_space().kind,
        ActionSpaceKind::ChooseTarget,
        "not on an opponent's end step"
    );
}

/// A noncreature spell makes a Clue; a creature spell does not. Until end of
/// turn the tapped Mechanist makes an artifact token you control a 3/1
/// Construct artifact creature with flying; nontoken artifacts can't be
/// chosen.
#[test]
fn the_mechanist_makes_clues_and_flies_them_until_end_of_turn() {
    let mut s = setup_unfunded(&[
        "The Mechanist, Aerial Artisan",
        "Abandon Attachments",
        "Llanowar Elves",
        "White Lotus Tile",
    ]);
    let mechanist = s.force_permanent_on_battlefield(0, "The Mechanist, Aerial Artisan");
    let tile = s.force_permanent_on_battlefield(0, "White Lotus Tile");
    ready(&mut s, mechanist);
    s.set_player_mana_pool(0, "1U");
    cast(&mut s, "Abandon Attachments");
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Clue").len(), 1);
    s.set_player_mana_pool(0, "G");
    cast(&mut s, "Llanowar Elves");
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Clue").len(), 1);

    let clue = battlefield_token(&s, 0);
    activate(&mut s, mechanist, 0);
    assert!(!s.choose_target(Target::Permanent(tile)));
    assert!(s.choose_target(Target::Permanent(clue)));
    settle(&mut s);
    assert!(tapped(&s, mechanist));
    assert!(is_creature(&s, clue));
    assert_eq!(s.game().effective_pt(clue), (3, 1));
    assert!(observed_keywords(&s, clue).flying);
    assert!(observed(&s, clue)
        .temp_subtypes
        .contains(&"Construct".to_owned()));

    s.advance_to_active_step(1, StepKind::Main);
    assert!(!is_creature(&s, clue), "until end of turn");
    assert!(!observed_keywords(&s, clue).flying);
}

/// CR 305.6, 611.2c: lands you control at resolution gain every basic land
/// type until end of turn, and with them "{T}: Add" each new type's color.
/// A Forest's own green stays its printed ability. An opponent's land is
/// unaffected, and the grant ends at cleanup.
#[test]
fn energybending_lets_your_lands_tap_for_any_color_this_turn() {
    let mut s = setup_unfunded(&["Energybending", "Typhoid Rats"]);
    let forest = s.force_permanent_on_battlefield(0, "Forest");
    let theirs = s.force_permanent_on_battlefield(1, "Forest");
    assert!(!can_activate(&mut s, forest, 0), "a Forest only taps for G");

    s.set_player_mana_pool(0, "2");
    cast(&mut s, "Energybending");
    settle(&mut s);
    assert_eq!(
        s.game().state.zones.size(ZoneType::Hand, PlayerId(0)),
        1,
        "draws a card"
    );
    assert!(observed(&s, forest)
        .temp_subtypes
        .contains(&"Swamp".to_owned()));
    assert!(observed(&s, theirs).temp_subtypes.is_empty());

    activate(&mut s, forest, 0);
    choose_mode(&mut s, 2);
    assert_eq!(pool_owned(&s), Mana::parse("B"));
    cast(&mut s, "Typhoid Rats");
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Typhoid Rats").len(), 1);
    assert_eq!(pool_owned(&s).total(), 0);

    untap(&mut s, forest);
    s.advance_to_active_step(1, StepKind::Main);
    assert!(!can_activate(&mut s, forest, 0), "until end of turn");
}

/// CR 106.4, 614.1a: under Ozai unspent mana, firebending's combat mana
/// included, becomes red instead of emptying, keeping its Lesson
/// restriction. Six or more unspent mana gives Ozai flying and
/// indestructible. Once Ozai is gone, mana empties again.
#[test]
fn ozai_keeps_unspent_mana_as_red_and_flies_on_six() {
    let mut s = setup_unfunded(&["Ozai, the Phoenix King"]);
    let ozai = s.force_permanent_on_battlefield(0, "Ozai, the Phoenix King");
    s.set_player_mana_pool(0, "WUB");
    let lesson = ManaUse::spell(["Lesson"]);
    s.game_mut().state.players[0]
        .mana_pool
        .add(&Mana::restricted_to(lesson, [0, 0, 0, 0, 1, 0]));
    assert!(!observed_keywords(&s, ozai).flying);

    declare_attackers(&mut s, &[ozai]);
    settle(&mut s);
    assert_eq!(
        s.game().state.players[0].combat_mana_pool,
        Mana::parse("RRRR")
    );
    let keywords = observed_keywords(&s, ozai);
    assert!(keywords.flying && keywords.indestructible, "eight unspent");

    s.advance_to_active_step(0, StepKind::PostcombatMain);
    let mut red = Mana::parse("RRRRRRR");
    red.add(&Mana::restricted_to(lesson, [0, 0, 0, 1, 0, 0]));
    assert_eq!(pool_owned(&s), red);
    assert_eq!(s.game().state.players[0].combat_mana_pool.total(), 0);
    assert!(observed_keywords(&s, ozai).indestructible);

    s.set_player_mana_pool(0, "RRRRR");
    assert!(!observed_keywords(&s, ozai).flying, "five unspent");
    let card = card_of(&s, ozai);
    s.game_mut().move_card(card, ZoneType::Graveyard);
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(pool_owned(&s).total(), 0);
}

/// CR 601.2b, 107.3m: X is announced as the spell is cast, offering each
/// value its total cost can pay, and the permanent enters with that many
/// fire counters. Damage your sources deal to an opponent or their
/// permanents gains one per fire counter; damage to your own creature is
/// unchanged.
#[test]
fn fated_firepower_enters_with_x_fire_counters_and_adds_them_to_damage() {
    let mut s = setup_unfunded(&[
        "Fated Firepower",
        "Typhoid Rats",
        "Wind Drake",
        "Gray Ogre",
        "Firebending Lesson",
    ]);
    let rats = s.force_permanent_on_battlefield(0, "Typhoid Rats");
    let drake = s.force_permanent_on_battlefield(1, "Wind Drake");
    s.game_mut().state.permanents[rats]
        .as_mut()
        .unwrap()
        .summoning_sick = false;

    s.set_player_mana_pool(0, "RRR2");
    cast(&mut s, "Fated Firepower");
    assert_eq!(s.action_space().kind, ActionSpaceKind::Modal);
    let offered: Vec<usize> = s
        .action_space()
        .actions
        .iter()
        .filter_map(|a| match a {
            Action::ChooseMode { mode, .. } => Some(*mode),
            _ => None,
        })
        .collect();
    assert_eq!(offered, vec![0, 1, 2], "every payable X");
    choose_mode(&mut s, 2);
    settle(&mut s);
    assert_eq!(
        pool_owned(&s).total(),
        0,
        "X = 2 paid with the generic mana"
    );
    let firepower = s.battlefield_permanents_named(0, "Fated Firepower")[0];
    assert_eq!(counters_of(&s, firepower, CounterKind::Fire), 2);

    declare_attackers(&mut s, &[rats]);
    s.advance_until(
        |s| s.action_space().kind == ActionSpaceKind::DeclareBlocker,
        "blockers".into(),
    );
    while s.action_space().kind == ActionSpaceKind::DeclareBlocker {
        choose(&mut s, |a| {
            matches!(a, Action::DeclareBlocker { attacker: None, .. })
        });
    }
    s.advance_to_active_step(0, StepKind::PostcombatMain);
    assert_eq!(s.game().state.players[1].life, 17, "1 combat damage + 2");

    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let lesson_damage = |s: &mut Scenario, target: PermanentId| {
        s.set_player_mana_pool(0, "R");
        cast(s, "Firebending Lesson");
        pass_until(s, ActionSpaceKind::ChooseTarget);
        assert!(s.choose_target(Target::Permanent(target)));
        s.drain_events();
        settle(s);
        s.drain_events()
            .iter()
            .find_map(|event| match event {
                GameEvent::DamageDealt { amount, .. } => Some(*amount),
                _ => None,
            })
            .unwrap()
    };
    assert_eq!(
        lesson_damage(&mut s, ogre),
        2,
        "your own creature: unchanged"
    );
    assert_eq!(lesson_damage(&mut s, drake), 4, "their creature: 2 + 2");
}

/// Day of Black Sun: each creature with mana value X or less loses all
/// abilities until end of turn and is destroyed. Indestructible is lost
/// with the rest, and dies triggers don't fire (release notes); a
/// creature with greater mana value is untouched.
#[test]
fn day_of_black_sun_strips_and_destroys_creatures_with_mana_value_x_or_less() {
    let mut s = setup_unfunded(&[
        "Day of Black Sun",
        "Curious Farm Animals",
        "Pretending Poxbearers",
        "Wind Drake",
    ]);
    let animals = s.force_permanent_on_battlefield(1, "Curious Farm Animals");
    let poxbearers = s.force_permanent_on_battlefield(0, "Pretending Poxbearers");
    let drake = s.force_permanent_on_battlefield(1, "Wind Drake");
    s.game_mut().state.permanents[poxbearers]
        .as_mut()
        .unwrap()
        .temp_keywords
        .indestructible = true;

    s.set_player_mana_pool(0, "BB22");
    cast(&mut s, "Day of Black Sun");
    choose_mode(&mut s, 2);
    settle(&mut s);

    assert!(s.game().state.permanents[animals].is_none());
    assert!(
        s.game().state.permanents[poxbearers].is_none(),
        "indestructible lost"
    );
    assert!(s.game().state.permanents[drake].is_some(), "mana value 3");
    assert_eq!(s.game().state.players[1].life, 20, "no dies trigger");
    assert!(s.battlefield_permanents_named(0, "Ally").is_empty());
    assert!(
        !s.game().state.permanents[drake]
            .as_ref()
            .unwrap()
            .abilities_lost
    );
}

/// Katara's entry makes an Ally. Waterbend {X} announces X (never 0, at
/// most what mana and tappable creatures can pay) and sets the base P/T of
/// the creatures you control as it resolves; counters still apply, the
/// opponent's creatures are untouched, and it lasts until end of turn.
#[test]
fn katara_waterbends_x_to_set_your_creatures_base_pt() {
    let mut s = setup_unfunded(&["Katara, Water Tribe's Hope", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    add_counters(&mut s, ogre, CounterKind::PlusOnePlusOne, 1);
    s.set_player_mana_pool(0, "2WUU");
    cast(&mut s, "Katara, Water Tribe's Hope");
    settle(&mut s);
    let katara = s.battlefield_permanents_named(0, "Katara, Water Tribe's Hope")[0];
    let ally = s.battlefield_permanents_named(0, "Ally")[0];

    s.set_player_mana_pool(0, "1");
    activate(&mut s, katara, 0);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Modal);
    let offered: Vec<usize> = s
        .action_space()
        .actions
        .iter()
        .filter_map(|a| match a {
            Action::ChooseMode { mode, .. } => Some(*mode),
            _ => None,
        })
        .collect();
    assert_eq!(
        offered,
        vec![1, 2, 3, 4],
        "three taps and one mana; X can't be 0"
    );
    choose_mode(&mut s, 2);
    assert!(s.choose_waterbend_tap(ally));
    assert!(s.choose_waterbend_tap(ogre));
    settle(&mut s);
    assert_eq!(pool_owned(&s), Mana::parse("1"), "two taps paid X = 2");
    assert_eq!(s.game().effective_pt(katara), (2, 2));
    assert_eq!(s.game().effective_pt(ally), (2, 2));
    assert_eq!(s.game().effective_pt(ogre), (3, 3), "counter still applies");
    assert_eq!(s.game().effective_pt(theirs), (2, 2));

    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.game().effective_pt(ally), (1, 1));
    s.set_player_mana_pool(0, "5");
    assert!(!can_activate(&mut s, katara, 0), "only during your turn");
}

fn block_with(s: &mut Scenario, blocker: PermanentId, attacker: PermanentId) {
    s.advance_until(
        |s| s.action_space().kind == ActionSpaceKind::DeclareBlocker,
        "blockers".into(),
    );
    while s.action_space().kind == ActionSpaceKind::DeclareBlocker {
        choose(s, |a| match a {
            Action::DeclareBlocker {
                blocker: b,
                attacker: chosen,
                ..
            } if *b == blocker => *chosen == Some(attacker),
            Action::DeclareBlocker { attacker: None, .. } => true,
            _ => false,
        });
    }
}

/// CR 120.4a, 120.6: Razor Rings targets only a creature in combat, and the
/// excess counts damage already marked: 4 to a 2/2 blocker with 1 damage
/// marked is 3 excess.
#[test]
fn razor_rings_gains_life_equal_to_excess_damage_to_a_creature_in_combat() {
    let mut s = setup_unfunded(&["Razor Rings", "Gray Ogre", "Wind Drake"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let drake = s.force_permanent_on_battlefield(1, "Wind Drake");
    ready(&mut s, ogre);
    s.set_player_mana_pool(0, "1W");
    to_hand(&mut s, "Razor Rings");
    assert!(
        !has_action(&s, |a| matches!(a, Action::CastSpell { .. })),
        "no creature is attacking or blocking"
    );

    declare_attackers(&mut s, &[ogre]);
    block_with(&mut s, drake, ogre);
    mark_damage(&mut s, drake, 1);
    s.set_player_mana_pool(0, "1W");
    cast(&mut s, "Razor Rings");
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(drake)));
    settle(&mut s);
    assert!(s.game().state.permanents[drake].is_none());
    assert_eq!(
        s.game().state.players[0].life,
        23,
        "4 - 1 lethal = 3 excess"
    );
}

/// CR 701.14: the fighters deal damage equal to their power to each other.
/// Excess damage to the opponent's creature adds that much {R}, and red
/// mana (only) survives step ends until the turn ends.
#[test]
fn the_last_agni_kai_adds_red_for_excess_fight_damage_and_keeps_red_mana() {
    let mut s = setup_unfunded(&["The Last Agni Kai", "Gray Ogre", "Wind Drake"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let drake = s.force_permanent_on_battlefield(1, "Wind Drake");
    add_counters(&mut s, ogre, CounterKind::PlusOnePlusOne, 3);
    s.set_player_mana_pool(0, "1R");
    cast(&mut s, "The Last Agni Kai");
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(ogre)));
    assert!(s.choose_target(Target::Permanent(drake)));
    settle(&mut s);
    assert!(s.game().state.permanents[drake].is_none());
    assert_eq!(
        s.game().state.permanents[ogre].as_ref().unwrap().damage,
        2,
        "the drake fights back"
    );
    assert_eq!(pool_owned(&s), Mana::parse("RRR"), "5 power - 2 lethal");

    s.set_player_mana_pool(0, "RRRG");
    s.advance_to_active_step(0, StepKind::End);
    assert_eq!(pool_owned(&s), Mana::parse("RRR"), "green is lost");
    s.advance_to_active_step(1, StepKind::Upkeep);
    assert_eq!(pool_owned(&s).total(), 0, "the effect ends with the turn");
}

/// CR 701.14b: a fight with an illegal target deals no damage at all.
#[test]
fn the_last_agni_kai_without_its_victim_deals_no_damage() {
    let mut s = setup_unfunded(&["The Last Agni Kai", "Gray Ogre", "Wind Drake"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let drake = s.force_permanent_on_battlefield(1, "Wind Drake");
    s.set_player_mana_pool(0, "1R");
    cast(&mut s, "The Last Agni Kai");
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(ogre)));
    assert!(s.choose_target(Target::Permanent(drake)));
    let card = card_of(&s, drake);
    s.game_mut().move_card(card, ZoneType::Hand);
    settle(&mut s);
    assert_eq!(s.game().state.permanents[ogre].as_ref().unwrap().damage, 0);
    assert_eq!(pool_owned(&s).total(), 0);
}

/// Lo and Li's entry offers only Lesson and Noble cards from the library
/// (or failing to find); the chosen card is revealed into hand. Noble
/// creatures you control and Lesson spells you control have lifelink: a
/// Lesson spell's damage gains its controller life (release notes).
#[test]
fn lo_and_li_tutor_a_lesson_or_noble_and_grant_lifelink() {
    let mut s = setup_unfunded(&[
        "Lo and Li, Twin Tutors",
        "Uncle Iroh",
        "Firebending Lesson",
        "Gray Ogre",
    ]);
    let theirs = s.force_permanent_on_battlefield(1, "Uncle Iroh");
    s.set_player_mana_pool(0, "4B");
    cast(&mut s, "Lo and Li, Twin Tutors");
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    let offered: Vec<_> = s
        .action_space()
        .actions
        .iter()
        .filter_map(|a| match a {
            Action::SelectCard { card, .. } => Some(s.game().state.cards[*card].name.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(offered.len(), 8, "four Lessons and four Nobles");
    assert!(offered
        .iter()
        .all(|name| name == "Uncle Iroh" || name == "Firebending Lesson"));
    assert!(has_action(&s, |a| matches!(a, Action::Decline { .. })));
    let iroh_card = s
        .action_space()
        .actions
        .iter()
        .find_map(|a| match a {
            Action::SelectCard { card, .. } if s.game().state.cards[*card].name == "Uncle Iroh" => {
                Some(*card)
            }
            _ => None,
        })
        .unwrap();
    choose(
        &mut s,
        |a| matches!(a, Action::SelectCard { card, .. } if *card == iroh_card),
    );
    settle(&mut s);
    let hand = s.game().state.zones.zone_cards(ZoneType::Hand, PlayerId(0));
    assert_eq!(hand.len(), 1);
    assert_eq!(s.game().state.cards[hand[0]].name, "Uncle Iroh");

    let iroh = s.force_permanent_on_battlefield(0, "Uncle Iroh");
    assert!(observed_keywords(&s, iroh).lifelink);
    assert!(!observed_keywords(&s, theirs).lifelink);

    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.set_player_mana_pool(0, "R");
    cast(&mut s, "Firebending Lesson");
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    assert_eq!(
        s.game().state.players[0].life,
        22,
        "the Lesson has lifelink"
    );
}

/// Put a library copy of `name` on top of `player`'s library.
fn library_to_top(s: &mut Scenario, player: usize, name: &str) -> CardId {
    let id = library_card(s, player, name);
    s.game_mut().move_card(id, ZoneType::Exile);
    s.game_mut().move_card(id, ZoneType::Library);
    s.game_mut().scenario_refresh_priority().unwrap();
    id
}

fn sees_library_card(s: &Scenario, player: usize, card: CardId) -> bool {
    let id = s.game().state.cards[card].id.0 as i32;
    let observation = Observation::for_player(s.game(), PlayerId(player));
    observation
        .agent_cards
        .iter()
        .chain(&observation.opponent_cards)
        .any(|c| c.id == id && c.zone == ZoneType::Library)
}

/// CR 401.2: Hakoda's controller sees their top card (the opponent does
/// not) and may cast it if it's an Ally spell, paying its cost; other spells
/// stay put.
#[test]
fn hakoda_reveals_the_top_card_to_you_and_casts_allies_from_it() {
    let mut s = setup_unfunded(&[
        "Hakoda, Selfless Commander",
        "Avatar Enthusiasts",
        "Gray Ogre",
    ]);
    let ally = library_to_top(&mut s, 0, "Avatar Enthusiasts");
    s.set_player_mana_pool(0, "2W");
    assert!(!sees_library_card(&s, 0, ally), "no permission yet");
    assert!(!can_cast(&mut s, ally));

    s.force_permanent_on_battlefield(0, "Hakoda, Selfless Commander");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(sees_library_card(&s, 0, ally));
    assert!(!sees_library_card(&s, 1, ally), "private to its controller");
    assert!(can_cast(&mut s, ally));
    choose(
        &mut s,
        |a| matches!(a, Action::CastSpell { card, .. } if *card == ally),
    );
    settle(&mut s);
    assert_eq!(
        s.game().state.zones.zone_of(ally),
        Some(ZoneType::Battlefield)
    );

    let ogre = library_to_top(&mut s, 0, "Gray Ogre");
    s.set_player_mana_pool(0, "2W");
    assert!(sees_library_card(&s, 0, ogre));
    assert!(!can_cast(&mut s, ogre), "not an Ally");
}

/// Sacrificing Hakoda gives the creatures you control +0/+5 and
/// indestructible until end of turn; the opponent's are untouched.
#[test]
fn hakoda_sacrifices_to_fortify_your_creatures() {
    let mut s = setup_unfunded(&["Hakoda, Selfless Commander", "Gray Ogre"]);
    let hakoda = s.force_permanent_on_battlefield(0, "Hakoda, Selfless Commander");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    activate(&mut s, hakoda, 0);
    settle(&mut s);
    assert!(s.game().state.permanents[hakoda].is_none());
    assert_eq!(s.game().effective_pt(ogre), (2, 7));
    assert!(observed_keywords(&s, ogre).indestructible);
    assert_eq!(s.game().effective_pt(theirs), (2, 2));
    assert!(!observed_keywords(&s, theirs).indestructible);

    s.advance_to_active_step(1, StepKind::Upkeep);
    assert_eq!(s.game().effective_pt(ogre), (2, 2));
    assert!(!observed_keywords(&s, ogre).indestructible);
}

/// Whether player 0's observed action space offers ability `index` of `id`.
fn observed_ability(s: &mut Scenario, id: PermanentId, index: usize) -> bool {
    s.game_mut().scenario_refresh_priority().unwrap();
    let object = s.game().state.permanents[id].as_ref().unwrap().id.0 as i32;
    Observation::for_player(s.game(), PlayerId(0))
        .action_space
        .actions
        .iter()
        .any(|action| action.ability_index == Some(index) && action.focus == [object])
}

/// The equipped creature has the ability (CR 702.6a, 613.1f): it taps the
/// creature as its cost, taps the target and returns the exact Equipment to
/// its owner's hand. Moving the Equipment moves the ability with it.
#[test]
fn trusty_boomerang_grants_its_creature_a_tap_ability_that_returns_it() {
    let mut s = setup_unfunded(&["Trusty Boomerang", "Gray Ogre"]);
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let first = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let second = s.force_permanent_on_battlefield(0, "Gray Ogre");
    ready(&mut s, first);
    let boomerang = s.force_permanent_on_battlefield(0, "Trusty Boomerang");
    let card = card_of(&s, boomerang);
    s.set_player_mana_pool(0, "1");
    assert!(!observed_ability(&mut s, first, 0), "unequipped");

    equip(&mut s, boomerang, first);
    s.set_player_mana_pool(0, "1");
    equip(&mut s, boomerang, second);
    s.set_player_mana_pool(0, "1");
    assert!(
        !observed_ability(&mut s, first, 0),
        "left with the Equipment"
    );
    assert!(
        !observed_ability(&mut s, second, 0),
        "{{T}} needs a creature without summoning sickness"
    );

    s.set_player_mana_pool(0, "1");
    equip(&mut s, boomerang, first);
    s.set_player_mana_pool(0, "1");
    assert!(observed_ability(&mut s, first, 0));
    activate(&mut s, first, 0);
    assert!(s.choose_target(Target::Permanent(theirs)));
    settle(&mut s);
    assert!(tapped(&s, first) && tapped(&s, theirs));
    assert_eq!(s.game().state.zones.zone_of(card), Some(ZoneType::Hand));
    assert!(!can_activate(&mut s, first, 0));
}

/// CR 113.7a, 400.7: the ability resolves without its Equipment, but
/// "Return Trusty Boomerang" cannot find a card that has left the
/// battlefield.
#[test]
fn trusty_boomerang_that_left_the_battlefield_does_not_return() {
    let mut s = setup_unfunded(&["Trusty Boomerang", "Gray Ogre"]);
    let theirs = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    ready(&mut s, ogre);
    let boomerang = s.force_permanent_on_battlefield(0, "Trusty Boomerang");
    let card = card_of(&s, boomerang);
    s.set_player_mana_pool(0, "1");
    equip(&mut s, boomerang, ogre);

    s.set_player_mana_pool(0, "1");
    activate(&mut s, ogre, 0);
    assert!(s.choose_target(Target::Permanent(theirs)));
    s.game_mut().move_card(card, ZoneType::Graveyard);
    settle(&mut s);
    assert!(tapped(&s, theirs));
    assert_eq!(
        s.game().state.zones.zone_of(card),
        Some(ZoneType::Graveyard)
    );
}

/// CR 613.1 (layers 1, 4, 6, 7b): the enchanted creature becomes a 1/1
/// Citizen named Humble Merchant, losing printed keywords and abilities
/// granted before (Trusty Boomerang's), and keeps only "{T}: Add {C}". The
/// entry trigger taps it and removes every counter (CR 122.8).
#[test]
fn honest_work_turns_the_creature_into_a_humble_merchant() {
    let mut s = setup_unfunded(&["Honest Work", "Wind Drake", "Trusty Boomerang"]);
    let drake = s.force_permanent_on_battlefield(1, "Wind Drake");
    let boomerang = s.force_permanent_on_battlefield(1, "Trusty Boomerang");
    let attached = s.game().permanent_object_ref(drake);
    let game = s.game_mut();
    game.state.permanents[boomerang]
        .as_mut()
        .unwrap()
        .attached_to = attached;
    let permanent = game.state.permanents[drake].as_mut().unwrap();
    permanent.change_counters(CounterKind::PlusOnePlusOne, 2);
    permanent.change_counters(CounterKind::Stun, 1);
    assert_eq!(s.game().effective_pt(drake), (4, 4));
    assert!(observed_keywords(&s, drake).flying);

    enchant(&mut s, "Honest Work", "U", drake);
    assert!(tapped(&s, drake), "the entry trigger taps it");
    for kind in CounterKind::ALL {
        assert_eq!(counters_of(&s, drake, kind), 0, "{kind:?} removed");
    }
    assert_eq!(s.game().effective_pt(drake), (1, 1));
    assert!(!observed_keywords(&s, drake).flying);
    assert_eq!(s.game().permanent_name(drake), "Humble Merchant");
    assert!(s.game().permanent_has_subtype(drake, "Citizen"));
    assert!(!s.game().permanent_has_subtype(drake, "Drake"));

    s.advance_to_active_step(1, StepKind::Main);
    ready(&mut s, drake);
    assert!(
        !tapped(&s, drake),
        "no stun counter remains to replace the untap"
    );
    s.set_player_mana_pool(1, "1");
    assert!(
        can_activate(&mut s, drake, 0),
        "the merchant's mana ability"
    );
    assert!(
        !can_activate(&mut s, drake, 1),
        "the Boomerang's grant is lost"
    );
    s.set_player_mana_pool(1, "");
    activate(&mut s, drake, 0);
    choose(&mut s, |a| matches!(a, Action::ChooseMode { .. }));
    assert_eq!(s.game().state.players[1].mana_pool, Mana::parse("C"));
    assert!(tapped(&s, drake));
}

/// A printed mana ability goes with the rest (Llanowar Elves taps for {C},
/// not {G}); when the Aura leaves, the creature's own characteristics return.
#[test]
fn honest_work_replaces_a_printed_mana_ability_until_it_leaves() {
    let mut s = setup_unfunded(&["Honest Work", "Llanowar Elves"]);
    let elves = s.force_permanent_on_battlefield(1, "Llanowar Elves");
    ready(&mut s, elves);
    let green = |s: &mut Scenario| s.game_mut().cached_producible_mana(PlayerId(1));
    assert_eq!(green(&mut s), Mana::parse("G"));

    let aura = enchant(&mut s, "Honest Work", "U", elves);
    s.game_mut().state.permanents[elves]
        .as_mut()
        .unwrap()
        .tapped = false;
    s.game_mut().invalidate_mana_cache(PlayerId(1));
    assert_eq!(green(&mut s), Mana::default(), "no fixed {{G}} remains");
    assert!(!s.game().permanent_has_subtype(elves, "Elf"));

    let aura = card_of(&s, aura);
    s.game_mut().move_card(aura, ZoneType::Graveyard);
    assert_eq!(green(&mut s), Mana::parse("G"));
    assert_eq!(s.game().permanent_name(elves), "Llanowar Elves");
    assert!(s.game().permanent_has_subtype(elves, "Elf"));
}

fn set_sick(s: &mut Scenario, id: PermanentId, sick: bool) {
    let permanent = s.game_mut().state.permanents[id].as_mut().unwrap();
    permanent.tapped = false;
    permanent.summoning_sick = sick;
}

/// CR 302.6, 613.1f: the merchant has lost its printed haste, so a
/// creature that came under its controller's control this turn cannot pay
/// "{T}: Add {C}".
#[test]
fn honest_work_takes_haste_from_a_summoning_sick_merchant() {
    let mut s = setup_unfunded(&["Honest Work", "Raging Goblin"]);
    let goblin = s.force_permanent_on_battlefield(1, "Raging Goblin");
    enchant(&mut s, "Honest Work", "U", goblin);
    s.advance_to_active_step(1, StepKind::Main);

    set_sick(&mut s, goblin, true);
    assert!(!can_activate(&mut s, goblin, 0), "no haste remains");
    set_sick(&mut s, goblin, false);
    assert!(can_activate(&mut s, goblin, 0));
}

/// CR 302.6, 702.10: haste from another permanent's static ability lets a
/// creature tap the turn it enters. No printed card grants haste statically
/// yet, so Badgermole's grant carries haste here instead of trample.
#[test]
fn a_static_haste_grant_lets_a_creature_tap_the_turn_it_enters() {
    let mut s = setup_unfunded(&["Badgermole", "Llanowar Elves"]);
    let badgermole = s.force_permanent_on_battlefield(0, "Badgermole");
    let card = card_of(&s, badgermole);
    s.game_mut().state.cards[card].static_keyword_grants[0].keywords = Keywords {
        haste: true,
        ..Default::default()
    };
    let elves = s.force_permanent_on_battlefield(0, "Llanowar Elves");
    let mana = |s: &mut Scenario| {
        s.game_mut().invalidate_mana_cache(PlayerId(0));
        s.game_mut().cached_producible_mana(PlayerId(0))
    };
    assert_eq!(mana(&mut s), Mana::default(), "sick without the grant");

    add_counters(&mut s, elves, CounterKind::PlusOnePlusOne, 1);
    assert_eq!(mana(&mut s), Mana::parse("G"));
}

fn unblockable(s: &Scenario, id: PermanentId) -> bool {
    s.game().state.permanents[id]
        .as_ref()
        .unwrap()
        .cant_be_blocked_this_turn
}

fn offers_target(s: &Scenario, id: PermanentId) -> bool {
    has_action(
        s,
        |a| matches!(a, Action::ChooseTarget { target, .. } if *target == Target::Permanent(id)),
    )
}

/// CR 601.2c, 602.2b: both targets must share a creature type, so only
/// creatures with a partner are offered first, then only that partner;
/// two targets are required, so there is no early finish.
#[test]
fn secret_tunnel_makes_two_creatures_sharing_a_type_unblockable() {
    let mut s = setup_unfunded(&[
        "Secret Tunnel",
        "Deserter's Disciple",
        "Earth Kingdom Soldier",
        "Gray Ogre",
    ]);
    let tunnel = s.force_permanent_on_battlefield(0, "Secret Tunnel");
    let soldier = s.force_permanent_on_battlefield(0, "Earth Kingdom Soldier");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    assert_eq!(
        s.game_mut().cached_producible_mana(PlayerId(0)),
        Mana::parse("C")
    );
    s.set_player_mana_pool(0, "4");
    assert!(!can_activate(&mut s, tunnel, 0), "no two share a type");

    let disciple = s.force_permanent_on_battlefield(0, "Deserter's Disciple");
    assert!(can_activate(&mut s, tunnel, 0));
    activate(&mut s, tunnel, 0);
    assert_eq!(s.action_space().kind, ActionSpaceKind::ChooseTarget);
    assert!(offers_target(&s, soldier) && offers_target(&s, disciple));
    assert!(!offers_target(&s, ogre), "the Ogre has no partner");
    assert!(s.choose_target(Target::Permanent(soldier)));
    assert_eq!(s.action_space().kind, ActionSpaceKind::ChooseTarget);
    assert!(offers_target(&s, disciple) && !offers_target(&s, soldier));
    assert!(!has_action(&s, |a| matches!(a, Action::Decline { .. })));
    assert!(s.choose_target(Target::Permanent(disciple)));
    settle(&mut s);

    assert!(tapped(&s, tunnel));
    assert!(unblockable(&s, soldier) && unblockable(&s, disciple));
    assert!(!unblockable(&s, ogre));
}

/// CR 608.2b, 608.2h: when one target leaves, the other is still legal
/// because it shares a creature type with the departed creature as it last
/// existed, so the ability resolves for it.
#[test]
fn secret_tunnel_survivor_shares_with_the_departed_target() {
    let mut s = setup_unfunded(&[
        "Secret Tunnel",
        "Deserter's Disciple",
        "Earth Kingdom Soldier",
    ]);
    let tunnel = s.force_permanent_on_battlefield(0, "Secret Tunnel");
    let soldier = s.force_permanent_on_battlefield(0, "Earth Kingdom Soldier");
    let disciple = s.force_permanent_on_battlefield(0, "Deserter's Disciple");
    s.set_player_mana_pool(0, "4");
    activate(&mut s, tunnel, 0);
    assert!(s.choose_target(Target::Permanent(soldier)));
    assert!(s.choose_target(Target::Permanent(disciple)));
    assert_eq!(s.game().state.stack_objects.len(), 1);

    let card = card_of(&s, disciple);
    s.game_mut().move_card(card, ZoneType::Graveyard);
    settle(&mut s);
    assert!(unblockable(&s, soldier));
}

/// CR 509.1b: once animated, the land itself can't be blocked.
#[test]
fn secret_tunnel_cannot_be_blocked_once_animated() {
    let mut s = setup_unfunded(&["Secret Tunnel", "Gray Ogre"]);
    let tunnel = s.force_permanent_on_battlefield(0, "Secret Tunnel");
    s.force_permanent_on_battlefield(1, "Gray Ogre");
    let land = s.game_mut().state.permanents[tunnel].as_mut().unwrap();
    land.animated = true;
    land.summoning_sick = false;
    add_counters(&mut s, tunnel, CounterKind::PlusOnePlusOne, 2);

    declare_attackers(&mut s, &[tunnel]);
    s.advance_until(
        |s| s.action_space().kind == ActionSpaceKind::DeclareBlocker,
        "blockers".into(),
    );
    assert!(!may_block(&s, tunnel));
}
