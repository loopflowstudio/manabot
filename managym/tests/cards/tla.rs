//! TLA (Avatar: The Last Airbender) card behavior. Each test drives legal
//! actions through the same typed programs as the world pool.
use crate::{
    helpers::Scenario,
    support::{
        activate, activate_card, can_activate, can_activate_card, can_cast, cast, cast_card,
        choose, counters, library_card, mode, offered_targets, pass_until, ready, settle, setup,
    },
};
use managym::{
    agent::action::{Action, ActionSpaceKind, ActionType},
    flow::turn::StepKind,
    state::{
        game_object::{CardId, PermanentId, PlayerId, Target},
        mana::{ManaCost, ManaUse},
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
    assert!(can_activate(&mut s, leaper));
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
    assert!(!can_activate(&mut s, vinebender));
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
    assert!(!can_activate(&mut s, enchantment));
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
    assert!(!can_activate(&mut s, permanent));
}

/// Exhaust (CR 702.177a): once activated, this object never offers the
/// ability again, even after it resolves; a new object starts fresh.
#[test]
fn hog_monkey_exhausts_once_for_two_counters() {
    let mut s = setup(&["Hog-Monkey"]);
    let monkey = s.force_permanent_on_battlefield(0, "Hog-Monkey");
    s.set_player_mana_pool(0, "4");
    assert!(!can_activate(&mut s, monkey));
    s.set_player_mana_pool(0, "10");
    activate(&mut s, monkey, 0);
    assert!(!can_activate(&mut s, monkey), "not even in response");
    settle(&mut s);
    assert_eq!(counters(&s, monkey), 2);
    assert_eq!(s.game().effective_pt(monkey), (5, 4));
    assert!(!can_activate(&mut s, monkey));

    let card = s.game().state.permanents[monkey].as_ref().unwrap().card;
    s.game_mut().move_card(card, ZoneType::Hand);
    s.game_mut().move_card(card, ZoneType::Battlefield);
    let again = s.battlefield_permanents_named(0, "Hog-Monkey")[0];
    assert!(can_activate(&mut s, again));
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
    assert!(!can_activate(&mut s, rhino));

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
    assert!(!can_activate(&mut s, captives));
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
    assert!(!can_activate(&mut s, barrels), "once this turn");

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
        !can_activate(&mut s, scroll),
        "a non-Ally ability cannot use it"
    );
    assert!(can_activate(&mut s, captives));
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
    assert!(!can_activate(&mut s, scroll));
    s.set_player_mana_pool(0, "2");
    let hand = s.zone_size(0, ZoneType::Hand);
    activate(&mut s, scroll, 0);
    assert_eq!(pool(&s).total(), 0);
    settle(&mut s);
    assert_eq!(s.zone_size(0, ZoneType::Hand), hand + 1);
    assert!(s.game().state.permanents[scroll].as_ref().unwrap().tapped);
}

fn card_of(s: &Scenario, id: PermanentId) -> CardId {
    s.game().state.permanents[id].as_ref().unwrap().card
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
