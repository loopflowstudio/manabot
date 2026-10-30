//! Tests for cards filed in `src/cardsets/tla.rs`, in the card order of that
//! file's `cards()`; card-specific setups sit just before their card's tests.
//! Tests that exercise one mechanic or contract across many cards follow at the
//! end.

use crate::support::{
    game::{
        clear_hand, clear_summoning_sickness, deck, enter_battlefield,
        enters_battlefield_with_triggers, five_color_setup, force_lands, force_new_card_in_hand,
        move_from_library, pin_top, plant_in_graveyard, put_onto_battlefield, ready,
        ur_lessons_deck, ur_lessons_setup, Scenario,
    },
    inspect::{
        assert_choice_contract, can_activate, can_activate_card, can_cast, card_of, counters,
        find_owned_card, hand, keywords, library, library_card, library_top, offered_cards,
        offered_targets, permanent, permanent_named, stack_spell_card, tapped, zone_of,
    },
    play::{
        activate, activate_card, activate_first, activate_named, attack_with, cast,
        cast_and_resolve, cast_card, cast_only, cast_owned, choose, choose_mode, enter_and_settle,
        pass_until, play_and_settle, resolve_everything, resolve_top, resolve_until_priority,
        select_card, settle, settle_targeting_opponent,
    },
};
use managym::{
    agent::{
        action::{Action, ActionSpaceKind, ActionType},
        observation::{EventEntityKind, Observation},
        policy_history::HistoryReference,
    },
    cards::setups::authored_pack,
    decision::Command,
    flow::{event::GameEvent, turn::StepKind},
    possible_worlds::{viewer_observation, PossibleWorldSpace},
    state::{
        game_object::{CardId, PermanentId, PlayerId, Target},
        mana::{Mana, ManaCost, ManaUse},
        zone::ZoneType,
    },
    Game,
};
use std::collections::BTreeMap;

/// Aang's Iceberg exiles another nonland permanent until it leaves. Its
/// waterbend ability sacrifices it on resolution — the exiled card returns —
/// and scries 2 only if that sacrifice happened.
#[test]
fn aangs_iceberg_exiles_until_it_leaves_and_scries_only_if_sacrificed() {
    let mut s = five_color_setup(&["Aang's Iceberg", "Gray Ogre"]);
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

/// Aang's Journey finds only a basic land, reveals it into hand and gains 2
/// life; kicked, it also searches for a Shrine card (none in this library, so
/// that search fails to find).
#[test]
fn aangs_journey_finds_a_basic_land_and_kicked_also_a_shrine() {
    for kicked in [false, true] {
        let mut s = five_color_setup(&["Aang's Journey", "Gray Ogre"]);
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

/// Aang's entry airbends up to one other nonland permanent; casting a Lesson
/// gives Aang lifelink until end of turn.
#[test]
fn aang_airbends_another_permanent_and_gains_lifelink_from_lessons() {
    let mut s = five_color_setup(&["Aang, the Last Airbender", "Airbending Lesson", "Gray Ogre"]);
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
#[test]
fn hybrid_discard_is_optional_at_resolution_and_draws_exactly_two() {
    for discard in [false, true] {
        let mut s = five_color_setup(&["Abandon Attachments"]);
        let fodder = library_card(&s, 0, "Plains");
        s.game_mut().move_card(fodder, ZoneType::Hand);
        s.set_player_mana_pool(0, "1R");
        cast(&mut s, "Abandon Attachments");
        s.pass_priority();
        s.pass_priority();
        assert_eq!(s.action_space().kind, ActionSpaceKind::Discard);
        let owner = Observation::for_player(s.game(), PlayerId(0));
        let other = Observation::for_player(s.game(), PlayerId(1));
        assert!(owner.agent_cards.iter().any(|c| c.name == "Plains"));
        assert!(!other
            .opponent_cards
            .iter()
            .any(|c| c.name == "Plains" && c.zone == ZoneType::Hand));
        let index = s
            .action_space()
            .actions
            .iter()
            .position(|a| {
                if discard {
                    matches!(a,Action::SelectCard {card,..} if *card==fodder)
                } else {
                    matches!(a, Action::Decline { .. })
                }
            })
            .unwrap();
        s.step_action(index);
        settle(&mut s);
        assert_eq!(s.zone_size(0, ZoneType::Hand), if discard { 2 } else { 1 });
        assert_eq!(
            s.game().state.zones.zone_of(fodder),
            Some(if discard {
                ZoneType::Graveyard
            } else {
                ZoneType::Hand
            })
        );
    }
}

#[test]
fn accumulate_wisdom_look_three_pick_one_rest_to_bottom() {
    let mut s = Scenario::new(
        deck(&[("Island", 24), ("Accumulate Wisdom", 16)]),
        deck(&[("Plains", 40)]),
        71,
    );
    force_lands(&mut s, 0, "Island", 2);
    clear_hand(&mut s, 0);
    s.force_card_in_hand(0, "Accumulate Wisdom");
    s.advance_to_active_step(0, StepKind::Main);
    let top3: Vec<CardId> = library(&s, 0).iter().rev().take(3).copied().collect();
    cast_and_resolve(&mut s);

    let space = s.action_space().clone();
    assert_eq!(space.kind, ActionSpaceKind::LookAndSelect);
    // Mandatory pick: three selectable cards, no decline.
    assert_eq!(space.actions.len(), 3);
    assert!(space
        .actions
        .iter()
        .all(|action| action.action_type() == ActionType::SelectCard));

    // Pick the middle looked card.
    let picked = top3[1];
    let index = space
        .actions
        .iter()
        .position(|action| matches!(action, Action::SelectCard { card, .. } if *card == picked))
        .expect("picked card should be selectable");
    s.step_action(index);

    assert_eq!(s.game().state.zones.zone_of(picked), Some(ZoneType::Hand));
    // The other two are on the bottom of the library (random order).
    let lib = library(&s, 0);
    assert!(lib[..2].contains(&top3[0]));
    assert!(lib[..2].contains(&top3[2]));
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
}

#[test]
fn accumulate_wisdom_with_three_lessons_takes_all_three_without_decision() {
    let mut s = Scenario::new(
        deck(&[("Island", 24), ("Accumulate Wisdom", 16)]),
        deck(&[("Plains", 40)]),
        72,
    );
    force_lands(&mut s, 0, "Island", 2);
    clear_hand(&mut s, 0);
    // Three Lesson cards in the graveyard (Accumulate Wisdom is a Lesson).
    for _ in 0..3 {
        let card = s
            .game()
            .state
            .zones
            .zone_cards(ZoneType::Library, PlayerId(0))
            .iter()
            .copied()
            .find(|card| s.game().state.cards[card].name == "Accumulate Wisdom")
            .expect("library should contain Accumulate Wisdom");
        s.game_mut()
            .state
            .zones
            .move_card(card, PlayerId(0), ZoneType::Graveyard);
    }
    s.force_card_in_hand(0, "Accumulate Wisdom");
    s.advance_to_active_step(0, StepKind::Main);
    let top3: Vec<CardId> = library(&s, 0).iter().rev().take(3).copied().collect();
    let hand_before = hand(&s, 0).len();
    cast_and_resolve(&mut s);

    // No decision — all three go to hand ("put all three into your hand").
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
    assert_eq!(hand(&s, 0).len(), hand_before - 1 + 3);
    for card in top3 {
        assert_eq!(s.game().state.zones.zone_of(card), Some(ZoneType::Hand));
    }
    // The resolving Lesson did not count itself (3 in graveyard, not 4,
    // at check time); it is in the graveyard afterwards.
    assert_eq!(s.zone_size(0, ZoneType::Graveyard), 4);
}

/// Mode one destroys only an attacking creature; mode two airbends a creature
/// its caster controls, which can then be recast for {2}.
#[test]
fn airbenders_reversal_destroys_an_attacker_or_airbends_your_creature() {
    let mut s = five_color_setup(&["Airbender's Reversal", "Gray Ogre"]);
    let attacker = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let idle = s.force_permanent_on_battlefield(0, "Gray Ogre");
    clear_summoning_sickness(&mut s, attacker);
    clear_summoning_sickness(&mut s, idle);
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
    choose_mode(&mut s, 0);
    let targets = offered_targets(&s);
    assert!(targets.contains(&Target::Permanent(attacker)));
    assert!(!targets.contains(&Target::Permanent(idle)));
    assert!(s.choose_target(Target::Permanent(attacker)));
    settle(&mut s);
    assert!(s.game().state.permanents[attacker].is_none());

    s.set_player_mana_pool(0, "20WWWW");
    let idle_card = card_of(&s, idle);
    cast(&mut s, "Airbender's Reversal");
    choose_mode(&mut s, 1);
    assert!(s.choose_target(Target::Permanent(idle)));
    settle(&mut s);
    assert_eq!(
        s.game().state.zones.zone_of(idle_card),
        Some(ZoneType::Exile)
    );
    s.advance_to_active_step(0, StepKind::Main);
    s.set_player_mana_pool(0, "2");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_cast(&s, idle_card));
}

/// Airbend exiles the target and lets its owner, and only its owner, cast
/// that exiled card for {2} instead of its mana cost. Lands are not offered.
#[test]
fn airbending_lesson_exiles_and_owner_recasts_for_two() {
    let mut s = five_color_setup(&["Airbending Lesson", "Gray Ogre"]);
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
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_cast(&s, ogre_card));

    s.advance_to_active_step(1, StepKind::Main);
    // {2} casts Gray Ogre ({2}{R}) from exile.
    s.set_player_mana_pool(1, "2");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_cast(&s, ogre_card));
    cast_card(&mut s, ogre_card);
    assert_eq!(s.game().state.players[1].mana_pool.total(), 0);
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(1, "Gray Ogre").len(), 1);
}

/// The permission belongs to the exact exiled object: once the card leaves
/// exile, a later return to exile does not restore it.
#[test]
fn airbend_permission_ends_when_the_card_leaves_exile() {
    let mut s = five_color_setup(&["Airbending Lesson", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let ogre_card = card_of(&s, ogre);
    cast(&mut s, "Airbending Lesson");
    assert!(s.choose_target(Target::Permanent(ogre)));
    settle(&mut s);
    s.set_player_mana_pool(0, "2");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_cast(&s, ogre_card));
    s.game_mut().move_card(ogre_card, ZoneType::Graveyard);
    s.game_mut().move_card(ogre_card, ZoneType::Exile);
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_cast(&s, ogre_card));
}

#[test]
fn draw_lands_enter_tapped_supply_either_color_and_sacrifice_for_one_card() {
    let lands = [
        ("Airship Engine Room", "UR"),
        ("Boiling Rock Prison", "BR"),
        ("Foggy Bottom Swamp", "BG"),
        ("Kyoshi Village", "GW"),
        ("Meditation Pools", "GU"),
        ("Misty Palms Oasis", "WB"),
        ("North Pole Gates", "WU"),
        ("Omashu City", "RG"),
        ("Serpent's Pass", "UB"),
        ("Sun-Blessed Peak", "RW"),
    ];
    for (name, colors) in lands {
        for color in colors.chars() {
            let spell = match color {
                'W' => "Healer's Hawk",
                'U' => "Wind Drake",
                'B' => "Typhoid Rats",
                'R' => "Raging Goblin",
                'G' => "Llanowar Elves",
                _ => unreachable!(),
            };
            let mut s = five_color_setup(&[name, spell]);
            let land = enter_battlefield(&mut s, name);
            assert!(tapped(&s, land), "{name} enters tapped");
            s.set_player_mana_pool(0, if color == 'U' { "2" } else { "0" });
            let id = library_card(&s, 0, spell);
            s.game_mut().move_card(id, ZoneType::Hand);
            s.game_mut().scenario_refresh_priority().unwrap();
            assert!(!can_cast(&s, id));
            s.game_mut().state.permanents[land].as_mut().unwrap().tapped = false;
            s.game_mut().scenario_refresh_priority().unwrap();
            choose(
                &mut s,
                |a| matches!(a,Action::CastSpell{card,..} if *card==id),
            );
            settle(&mut s);
            assert!(tapped(&s, land), "{name} pays {color}");
            assert_eq!(
                s.game().state.players[0].mana_pool.total(),
                0,
                "{name} paying {color}"
            );
        }
        let mut s = five_color_setup(&[name]);
        let land = enter_battlefield(&mut s, name);
        s.game_mut().state.permanents[land].as_mut().unwrap().tapped = false;
        s.set_player_mana_pool(0, "4");
        let before = s.zone_size(0, ZoneType::Hand);
        activate(&mut s, land, 0);
        assert!(s.game().state.permanents[land].is_none());
        assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
        settle(&mut s);
        assert_eq!(s.zone_size(0, ZoneType::Hand), before + 1);
    }
}

#[test]
fn allies_at_last_affinity_and_two_attackers() {
    let mut s = Scenario::new(
        deck(&[
            ("Forest", 20),
            ("Allies at Last", 8),
            ("Kyoshi Warriors", 12),
        ]),
        deck(&[("Mountain", 24), ("Gray Ogre", 16)]),
        91,
    );
    let w1 = s.force_permanent_on_battlefield(0, "Kyoshi Warriors");
    let w2 = s.force_permanent_on_battlefield(0, "Kyoshi Warriors");
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    // Two Allies -> {2}{G} costs {G}: one forest suffices.
    force_lands(&mut s, 0, "Forest", 1);
    clear_hand(&mut s, 0);
    s.force_card_in_hand(0, "Allies at Last");
    s.advance_to_active_step(0, StepKind::Main);

    cast_only(&mut s);
    // Requirement 0: up to two creatures you control (+ Decline).
    let space = s.action_space().clone();
    assert_eq!(space.kind, ActionSpaceKind::ChooseTarget);
    assert_eq!(space.actions.len(), 3, "two warriors + decline");
    assert!(s.choose_target(Target::Permanent(w1)));
    assert!(s.choose_target(Target::Permanent(w2)));
    // Requirement 1: exactly one creature an opponent controls, no decline.
    let space = s.action_space().clone();
    assert_eq!(space.actions.len(), 1);
    assert!(s.choose_target(Target::Permanent(ogre)));
    // Affinity was applied: the cast succeeded on a single forest.
    assert_eq!(s.game().state.stack_objects.len(), 1);

    s.pass_priority();
    s.pass_priority();
    // Each 3/3 warrior dealt 3 damage to the 2/2 Ogre; it died.
    assert_eq!(s.battlefield_permanents_named(1, "Gray Ogre").len(), 0);
    assert_eq!(s.zone_size(1, ZoneType::Graveyard), 1);
    s.assert_life(1, 20);
}

#[test]
fn allies_at_last_up_to_zero_attackers_is_legal() {
    let mut s = Scenario::new(
        deck(&[
            ("Forest", 20),
            ("Allies at Last", 8),
            ("Kyoshi Warriors", 12),
        ]),
        deck(&[("Mountain", 24), ("Gray Ogre", 16)]),
        92,
    );
    let _w1 = s.force_permanent_on_battlefield(0, "Kyoshi Warriors");
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    force_lands(&mut s, 0, "Forest", 4);
    clear_hand(&mut s, 0);
    s.force_card_in_hand(0, "Allies at Last");
    s.advance_to_active_step(0, StepKind::Main);

    cast_only(&mut s);
    // Decline immediately: zero attackers chosen.
    assert!(s.take_action_by_type(ActionType::DeclineChoice));
    assert!(s.choose_target(Target::Permanent(ogre)));
    s.pass_priority();
    s.pass_priority();
    // Nothing happened; the Ogre survives.
    assert_eq!(s.battlefield_permanents_named(1, "Gray Ogre").len(), 1);
}

/// CR 608.2b: a legal source keeps resolution alive, but never licenses damage
/// to an illegal victim. Damage belongs to each remaining legal source.
#[test]
fn allies_at_last_respects_source_and_victim_legality_and_lifelink() {
    for (protect_victim, remove_hawk) in [(false, false), (true, false), (false, true)] {
        let mut s = five_color_setup(&[
            "Allies at Last",
            "Healer's Hawk",
            "Kyoshi Warriors",
            "Wall of Stone",
        ]);
        let hawk = s.force_permanent_on_battlefield(0, "Healer's Hawk");
        let warrior = s.force_permanent_on_battlefield(0, "Kyoshi Warriors");
        let victim = s.force_permanent_on_battlefield(1, "Wall of Stone");
        // One Ally reduces {2}{G} to {1}{G}; non-Allies do not reduce it.
        s.set_player_mana_pool(0, "1G");
        let life = s.life(0);
        cast(&mut s, "Allies at Last");
        assert!(s.choose_target(Target::Permanent(hawk)));
        assert!(s.choose_target(Target::Permanent(warrior)));
        assert!(s.choose_target(Target::Permanent(victim)));
        assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
        if protect_victim {
            s.game_mut().state.permanents[victim]
                .as_mut()
                .unwrap()
                .temp_keywords
                .hexproof = true;
        }
        if remove_hawk {
            let card = s.game().state.permanents[hawk].as_ref().unwrap().card;
            s.game_mut().move_card(card, ZoneType::Hand);
        }
        settle(&mut s);
        let damage = if protect_victim {
            0
        } else if remove_hawk {
            3
        } else {
            4
        };
        assert_eq!(
            s.game().state.permanents[victim].as_ref().unwrap().damage,
            damage
        );
        assert_eq!(s.life(0), life + i32::from(!protect_victim && !remove_hawk));
        assert!(s.game().state.stack_objects.is_empty());
    }
}

#[test]
fn allies_at_last_can_choose_one_source_and_uses_its_current_power() {
    let mut s = five_color_setup(&["Allies at Last", "Healer's Hawk", "Wall of Stone"]);
    let hawk = s.force_permanent_on_battlefield(0, "Healer's Hawk");
    let _unchosen = s.force_permanent_on_battlefield(0, "Healer's Hawk");
    let victim = s.force_permanent_on_battlefield(1, "Wall of Stone");
    s.set_player_mana_pool(0, "2G");
    let life = s.life(0);
    cast(&mut s, "Allies at Last");
    assert!(s.choose_target(Target::Permanent(hawk)));
    choose(&mut s, |action| matches!(action, Action::Decline { .. }));
    assert!(s.choose_target(Target::Permanent(victim)));
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
    s.game_mut().state.permanents[hawk]
        .as_mut()
        .unwrap()
        .plus1_counters = 2;
    settle(&mut s);
    assert_eq!(
        s.game().state.permanents[victim].as_ref().unwrap().damage,
        3
    );
    assert_eq!(s.life(0), life + 3);
}

/// Restricted mana (CR 106.6) pays for Ally spells and abilities of Ally
/// sources, and nothing else.
#[test]
fn jasmine_dragon_tea_shop_mana_pays_only_ally_spells_and_ally_abilities() {
    let ally = ManaUse {
        spells: ManaUse::spell(["Ally"]).spells,
        abilities: ManaUse::ability(["Ally"]).abilities,
    };
    let mut s = five_color_setup(&[
        "Jasmine Dragon Tea Shop",
        "Rebellious Captives",
        "Gray Ogre",
    ]);
    let shop = s.force_permanent_on_battlefield(0, "Jasmine Dragon Tea Shop");
    s.set_player_mana_pool(0, "");
    activate(&mut s, shop, 0);
    choose_mode(&mut s, 4);
    assert_eq!(pool(&s).restricted_of(ally), [0, 0, 0, 0, 1, 0]);
    s.game_mut().state.players[0].mana_pool.mana = [0, 0, 0, 1, 0, 1];
    let captives = library_card(&s, 0, "Rebellious Captives");
    let ogre = library_card(&s, 0, "Gray Ogre");
    s.game_mut().move_card(captives, ZoneType::Hand);
    s.game_mut().move_card(ogre, ZoneType::Hand);
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(
        !can_cast(&s, ogre),
        "{{2}}{{R}} cannot use the restricted green"
    );
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_cast(&s, captives));

    let mut s = five_color_setup(&[
        "Jasmine Dragon Tea Shop",
        "Rebellious Captives",
        "Waterbending Scroll",
    ]);
    let shop = s.force_permanent_on_battlefield(0, "Jasmine Dragon Tea Shop");
    let captives = s.force_permanent_on_battlefield(0, "Rebellious Captives");
    let scroll = s.force_permanent_on_battlefield(0, "Waterbending Scroll");
    s.set_player_mana_pool(0, "");
    activate(&mut s, shop, 0);
    choose_mode(&mut s, 0);
    s.game_mut().state.players[0].mana_pool.mana = [0, 0, 0, 0, 0, 5];
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(
        !can_activate(&s, scroll),
        "a non-Ally ability cannot use it"
    );
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_activate(&s, captives));
    activate(&mut s, captives, 0);
    assert!(s.choose_target(Target::Permanent(shop)));
    settle(&mut s);
    assert_eq!(pool(&s).total(), 0, "the Ally ability spent all six");
    assert_eq!(counters(&s, captives), 2);
}

/// Avatar Enthusiasts on player 0's battlefield; both decks hold Allies and a non-Ally.
fn enthusiasts_scenario() -> (Scenario, PermanentId) {
    let list = deck(&[
        ("Plains", 20),
        ("Avatar Enthusiasts", 8),
        ("South Pole Voyager", 8),
        ("Gray Ogre", 4),
    ]);
    let mut s = Scenario::new(list.clone(), list, 11);
    let enthusiasts = s.force_permanent_on_battlefield(0, "Avatar Enthusiasts");
    (s, enthusiasts)
}

/// Avatar Enthusiasts gets a +1/+1 counter when another Ally enters under
/// its controller's control...
#[test]
fn avatar_enthusiasts_counter_when_another_ally_enters() {
    let (mut s, enthusiasts) = enthusiasts_scenario();
    assert_eq!(counters(&s, enthusiasts), 0);

    enters_battlefield_with_triggers(&mut s, 0, "South Pole Voyager");
    resolve_until_priority(&mut s);

    assert_eq!(counters(&s, enthusiasts), 1);
    // Counters contribute to P/T: 2/2 base + 1 counter = 3/3.
    assert_eq!(s.game().effective_power(enthusiasts), 3);
    assert_eq!(s.game().effective_toughness(enthusiasts), 3);
}

/// ...but NOT when a non-Ally creature enters...
#[test]
fn avatar_enthusiasts_no_counter_for_non_ally() {
    let (mut s, enthusiasts) = enthusiasts_scenario();

    enters_battlefield_with_triggers(&mut s, 0, "Gray Ogre");
    resolve_until_priority(&mut s);

    assert_eq!(counters(&s, enthusiasts), 0);
}

/// Avatar's Wrath spares its optional target, airbends every other creature,
/// exiles itself, and stops opponents casting from outside their hands until
/// its caster's next turn; the caster may recast their own creatures at once.
#[test]
fn avatars_wrath_spares_one_and_restricts_opponents_until_your_next_turn() {
    let mut s = five_color_setup(&["Avatar's Wrath", "Gray Ogre", "Craw Wurm"]);
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
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_cast(&s, own_card));

    s.advance_to_active_step(1, StepKind::Main);
    s.set_player_mana_pool(1, "20GGGG");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_cast(&s, wurm_card));

    s.advance_to_active_step(0, StepKind::Main);
    s.advance_to_active_step(1, StepKind::Main);
    s.set_player_mana_pool(1, "2");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_cast(&s, wurm_card));
}

/// Azula's opponent exiles a nontoken creature and then a nonland graveyard
/// card; on your turn you may cast either with any type of mana. Their
/// creature enters under your control (CR 110.2) and, when it dies, goes to
/// its owner's graveyard.
#[test]
fn azula_has_the_opponent_exile_and_casts_with_any_mana() {
    let mut s = five_color_setup(&["Azula, Cunning Usurper", "Lightning Strike", "Gray Ogre"]);
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
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_cast(&s, ogre_card));
    assert!(can_cast(&s, strike));
    cast_card(&mut s, strike);
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    settle(&mut s);
    s.assert_life(1, 17);
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);

    // Gray Ogre ({2}{R}) for three colorless: it enters under your control.
    s.set_player_mana_pool(0, "3");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_cast(&s, ogre_card));
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

/// Badgermole Cub earthbends 1 on entering; tapping a creature for mana adds
/// an extra {G}.
#[test]
fn badgermole_cub_earthbends_and_adds_green_when_a_creature_taps_for_mana() {
    let mut s = five_color_setup(&["Badgermole Cub", "Hermitic Herbalist"]);
    let forest = s.force_permanent_on_battlefield(0, "Forest");
    cast(&mut s, "Badgermole Cub");
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(forest)));
    settle(&mut s);
    assert_eq!(counters(&s, forest), 1);
    assert!(is_land_creature(&s, forest));

    let herbalist = s.force_permanent_on_battlefield(0, "Hermitic Herbalist");
    clear_summoning_sickness(&mut s, herbalist);
    s.set_player_mana_pool(0, "0");
    activate(&mut s, herbalist, 0);
    choose_mode(&mut s, 1);
    settle(&mut s);
    assert_eq!(*pool(&s), managym::state::mana::Mana::parse("UG"));
}

fn pool(s: &Scenario) -> &managym::state::mana::Mana {
    &s.game().state.players[0].mana_pool
}

/// The mana ability may be activated once each turn (CR 602.5b); the count
/// resets on the next turn.
#[test]
fn barrels_of_blasting_jelly_filters_one_mana_once_each_turn() {
    let mut s = five_color_setup(&["Barrels of Blasting Jelly"]);
    let barrels = s.force_permanent_on_battlefield(0, "Barrels of Blasting Jelly");
    s.set_player_mana_pool(0, "CC");
    activate(&mut s, barrels, 0);
    choose_mode(&mut s, 3);
    assert_eq!(pool(&s).mana, [0, 0, 0, 1, 0, 1]);
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_activate(&s, barrels), "once this turn");

    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    s.set_player_mana_pool(0, "C");
    activate(&mut s, barrels, 0);
    choose_mode(&mut s, 0);
    assert_eq!(pool(&s).mana, [1, 0, 0, 0, 0, 0]);
}

#[test]
fn barrels_of_blasting_jelly_sacrifices_to_deal_five_to_a_creature() {
    let mut s = five_color_setup(&["Barrels of Blasting Jelly", "Craw Wurm"]);
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

fn cycling_index(s: &Scenario, card: CardId) -> usize {
    s.game().state.cards[card]
        .activated_abilities
        .iter()
        .position(|ability| ability.discard_source)
        .unwrap()
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
        let mut s = five_color_setup(&[name]);
        let card = move_from_library(&mut s, 0, name, ZoneType::Hand);
        s.set_player_mana_pool(0, "1");
        s.game_mut().scenario_refresh_priority().unwrap();
        assert!(!can_activate_card(&s, card), "{name} costs {{2}}");
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

#[test]
fn canyon_crawler_has_deathtouch_and_makes_food_on_entering() {
    let mut s = five_color_setup(&["Canyon Crawler"]);
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

/// Dai Li Indoctrination's first mode reveals the opponent's hand (public
/// definition knowledge) and the caster picks a nonland permanent card for
/// them to discard; the second mode earthbends 2.
#[test]
fn dai_li_indoctrination_reveals_a_hand_and_discards_your_pick_or_earthbends_two() {
    let mut s = five_color_setup(&[
        "Dai Li Indoctrination",
        "Gray Ogre",
        "Lightning Strike",
        "Aang's Iceberg",
    ]);
    let ogre = move_from_library(&mut s, 1, "Gray Ogre", ZoneType::Hand);
    let iceberg = move_from_library(&mut s, 1, "Aang's Iceberg", ZoneType::Hand);
    let strike = move_from_library(&mut s, 1, "Lightning Strike", ZoneType::Hand);
    move_from_library(&mut s, 1, "Forest", ZoneType::Hand);
    cast(&mut s, "Dai Li Indoctrination");
    choose_mode(&mut s, 0);
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
    choose_mode(&mut s, 1);
    assert!(s.choose_target(Target::Permanent(forest)));
    settle(&mut s);
    assert_eq!(counters(&s, forest), 2);
}

/// A dies trigger still earthbends after its source left the battlefield.
#[test]
fn earth_village_ruffians_earthbends_two_when_it_dies() {
    let mut s = five_color_setup(&["Earth Village Ruffians", "Lightning Strike"]);
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
        let mut s = five_color_setup(&["Earth Village Ruffians"]);
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

fn is_land_creature(s: &Scenario, id: PermanentId) -> bool {
    s.game().state.permanents[id].as_ref().unwrap().animated
}

/// Earthbend N: only lands you control are offered; the land becomes a 0/0
/// creature that is still a land, with N +1/+1 counters.
#[test]
fn earthbending_lesson_animates_only_an_own_land_with_four_counters() {
    let mut s = five_color_setup(&["Earthbending Lesson"]);
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

/// Elemental Teachings reveals up to four differently named lands; the
/// opponent chooses two for the graveyard and the rest enter tapped.
#[test]
fn elemental_teachings_finds_distinct_lands_and_the_opponent_bins_two() {
    let mut s = five_color_setup(&["Elemental Teachings"]);
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
    let mut s = five_color_setup(&["Elemental Teachings"]);
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

/// Printed flashback (CR 702.34): the graveyard copy costs {8}{R}, may be cast
/// at instant speed on either turn, and is exiled as it resolves.
#[test]
fn fire_nation_attacks_makes_firebending_soldiers_and_flashes_back_once() {
    let mut s = five_color_setup(&["Fire Nation Attacks"]);
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
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_cast(&s, spell));

    // On the opponent's turn, with priority, the instant flashes back.
    s.advance_to_active_step(1, StepKind::Main);
    s.pass_priority();
    assert_eq!(s.action_space().player, Some(PlayerId(0)));
    s.set_player_mana_pool(0, "8R");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_cast(&s, spell));
    cast_card(&mut s, spell);
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Soldier").len(), 4);
    assert_eq!(s.game().state.zones.zone_of(spell), Some(ZoneType::Exile));
}

/// Soldier tokens attack for firebending 1: one red mana until end of combat.
#[test]
fn fire_nation_attacks_soldiers_add_red_when_attacking() {
    let mut s = five_color_setup(&["Fire Nation Attacks"]);
    cast(&mut s, "Fire Nation Attacks");
    settle(&mut s);
    let soldiers = s.battlefield_permanents_named(0, "Soldier");
    for soldier in &soldiers {
        clear_summoning_sickness(&mut s, *soldier);
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

/// Activation is restricted to its controller's turn; blockers with current
/// power 2 or less cannot block it.
#[test]
fn foggy_swamp_vinebender_grows_on_own_turn_and_evades_small_blockers() {
    let mut s = five_color_setup(&["Foggy Swamp Vinebender", "Gray Ogre", "Hill Giant"]);
    let vinebender = s.force_permanent_on_battlefield(0, "Foggy Swamp Vinebender");
    clear_summoning_sickness(&mut s, vinebender);
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
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_activate(&s, vinebender));
}

/// Waterbend {4} may be paid with creature taps (which make no mana); the
/// resolution draws first, then requires a discard from the new hand.
#[test]
fn geyser_leaper_waterbends_to_draw_then_discard() {
    let mut s = five_color_setup(&["Geyser Leaper", "Gray Ogre"]);
    let leaper = s.force_permanent_on_battlefield(0, "Geyser Leaper");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.set_player_mana_pool(0, "3");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_activate(&s, leaper));
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

/// Waterbend {3} may be paid by tapping creatures; the Koi then can't be
/// blocked this turn.
#[test]
fn giant_koi_waterbends_to_become_unblockable() {
    let mut s = five_color_setup(&["Giant Koi", "Gray Ogre"]);
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
    let mut s = five_color_setup(&["Hama, the Bloodbender", "Lightning Strike", "Gray Ogre"]);
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
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_cast(&s, strike));
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

/// Hama can exile an opposing noncreature permanent card: cast by
/// waterbending, Prophetic Prism enters under your control, its enters
/// trigger draws for you, and its mana ability is yours to activate.
#[test]
fn hama_casts_an_opponents_artifact_under_your_control() {
    let mut s = five_color_setup(&["Hama, the Bloodbender", "Prophetic Prism", "Gray Ogre"]);
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
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_cast(&s, prism));
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
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_activate(&s, artifact));
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

/// Haru triggers for each other Ally entering, not for itself. Two earthbends
/// on one land stack counters and leave two independent delayed returns: the
/// land returns once, tapped, as a new object.
#[test]
fn haru_earthbends_per_other_ally_and_a_doubly_bent_land_returns_once() {
    let mut s = five_color_setup(&[
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

/// Herbalist's second ability offers every unordered two-color pair once
/// (fifteen modes) and the mana pays only Lesson spells.
#[test]
fn hermitic_herbalist_offers_each_two_color_combination_once() {
    let mut s = five_color_setup(&["Hermitic Herbalist"]);
    let herbalist = s.force_permanent_on_battlefield(0, "Hermitic Herbalist");
    clear_summoning_sickness(&mut s, herbalist);
    s.set_player_mana_pool(0, "0");
    activate(&mut s, herbalist, 1);
    let modes = s
        .action_space()
        .actions
        .iter()
        .filter(|a| matches!(a, Action::ChooseMode { .. }))
        .count();
    assert_eq!(modes, 15);
    choose_mode(&mut s, 1);
    settle(&mut s);
    let player = &s.game().state.players[0].mana_pool;
    assert_eq!(
        player.restricted_of(ManaUse::spell(["Lesson"])),
        [1, 1, 0, 0, 0, 0]
    );
    assert_eq!(player.total(), 2);
}

/// Exhaust (CR 702.177a): once activated, this object never offers the
/// ability again, even after it resolves; a new object starts fresh.
#[test]
fn hog_monkey_exhausts_once_for_two_counters() {
    let mut s = five_color_setup(&["Hog-Monkey"]);
    let monkey = s.force_permanent_on_battlefield(0, "Hog-Monkey");
    s.set_player_mana_pool(0, "4");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_activate(&s, monkey));
    s.set_player_mana_pool(0, "10");
    activate(&mut s, monkey, 0);
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_activate(&s, monkey), "not even in response");
    settle(&mut s);
    assert_eq!(counters(&s, monkey), 2);
    assert_eq!(s.game().effective_pt(monkey), (5, 4));
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_activate(&s, monkey));

    let card = s.game().state.permanents[monkey].as_ref().unwrap().card;
    s.game_mut().move_card(card, ZoneType::Hand);
    s.game_mut().move_card(card, ZoneType::Battlefield);
    let again = s.battlefield_permanents_named(0, "Hog-Monkey")[0];
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_activate(&s, again));
}

/// At the beginning of combat on its controller's turn, only creatures they
/// control with a +1/+1 counter are targets; the target gains menace.
#[test]
fn hog_monkey_gives_menace_to_a_creature_with_a_counter_at_combat() {
    let mut s = five_color_setup(&["Hog-Monkey", "Gray Ogre"]);
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

/// Iroh, Grand Lotus attacks for firebending 2. (Its graveyard flashback grants
/// are witnessed in `tests/rules/temur_cards.rs`.)
#[test]
fn iroh_grand_lotus_firebends_two_when_attacking() {
    let mut s = five_color_setup(&["Iroh, Grand Lotus"]);
    let iroh = s.force_permanent_on_battlefield(0, "Iroh, Grand Lotus");
    clear_summoning_sickness(&mut s, iroh);
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

#[test]
fn jasmine_dragon_tea_shop_taps_for_colorless_or_an_ally_token() {
    let mut s = five_color_setup(&["Jasmine Dragon Tea Shop"]);
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

/// Lost Days: the target's owner (not the caster) chooses second from the top
/// or the bottom of their library; the caster creates a Clue either way.
#[test]
fn lost_days_owner_puts_it_second_from_top_or_on_the_bottom_and_caster_gets_a_clue() {
    for second in [true, false] {
        let mut s = five_color_setup(&["Lost Days", "Gray Ogre", "Aang's Iceberg"]);
        let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
        let land = s.force_permanent_on_battlefield(1, "Forest");
        let ogre_card = card_of(&s, ogre);
        cast(&mut s, "Lost Days");
        assert!(!offered_targets(&s).contains(&Target::Permanent(land)));
        assert!(s.choose_target(Target::Permanent(ogre)));
        pass_until(&mut s, ActionSpaceKind::Modal);
        assert_eq!(s.action_space().player, Some(PlayerId(1)));
        choose_mode(&mut s, if second { 0 } else { 1 });
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

/// Graveyard-only (CR 113.6): returns that exact card; a card that left and
/// came back is a new object and the old activation does nothing.
#[test]
fn merchant_of_many_hats_returns_itself_from_the_graveyard() {
    let mut s = five_color_setup(&["Merchant of Many Hats"]);
    let merchant = move_from_library(&mut s, 0, "Merchant of Many Hats", ZoneType::Hand);
    s.set_player_mana_pool(0, "2B");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_activate_card(&s, merchant));
    s.game_mut().move_card(merchant, ZoneType::Graveyard);
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_activate_card(&s, merchant));
    s.set_player_mana_pool(0, "2");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_activate_card(&s, merchant));
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
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_activate(&s, permanent));
}

#[test]
fn mongoose_lizard_deals_one_damage_to_any_target_on_entering() {
    let mut s = five_color_setup(&["Mongoose Lizard", "Gray Ogre"]);
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

/// North Pole Patrol untaps another permanent you control, and its waterbend
/// ability taps an opposing creature; the Patrol's own {T} cannot also help
/// pay the waterbend cost.
#[test]
fn north_pole_patrol_untaps_another_and_waterbends_to_tap_an_opponent() {
    let mut s = five_color_setup(&["North Pole Patrol", "Gray Ogre"]);
    let patrol = s.force_permanent_on_battlefield(0, "North Pole Patrol");
    let own = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let foe = s.force_permanent_on_battlefield(1, "Gray Ogre");
    clear_summoning_sickness(&mut s, patrol);
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

/// Professor Zei's {T}, discard: the discarded card is chosen as a cost
/// (no activation with an empty hand), then a card is drawn. Its sacrifice
/// ability returns an instant or sorcery card, only during your turn.
#[test]
fn professor_zei_discards_as_a_cost_to_draw_and_sacrifices_to_recall_a_spell() {
    let mut s = five_color_setup(&[
        "Professor Zei, Anthropologist",
        "Lightning Strike",
        "Gray Ogre",
    ]);
    let zei = s.force_permanent_on_battlefield(0, "Professor Zei, Anthropologist");
    clear_summoning_sickness(&mut s, zei);
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!s.action_space().actions.iter().any(|a| matches!(
        a, Action::ActivateAbility { permanent, ability_index: 0, .. } if *permanent == zei
    )));
    let pitched = move_from_library(&mut s, 0, "Gray Ogre", ZoneType::Hand);
    activate(&mut s, zei, 0);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Discard);
    choose(
        &mut s,
        |a| matches!(a, Action::SelectCard { card, .. } if *card == pitched),
    );
    assert_eq!(zone_of(&s, pitched), Some(ZoneType::Graveyard));
    assert!(s.game().state.permanents[zei].as_ref().unwrap().tapped);
    settle(&mut s);
    assert_eq!(hand(&s, 0).len(), 1);

    let strike = move_from_library(&mut s, 0, "Lightning Strike", ZoneType::Graveyard);
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

/// Landfall counts only lands its controller's side gains; the flying grant
/// lasts until end of turn.
#[test]
fn rabaroo_troop_landfall_grants_flying_and_one_life() {
    let mut s = five_color_setup(&["Rabaroo Troop"]);
    let troop = s.force_permanent_on_battlefield(0, "Rabaroo Troop");
    let life = s.life(0);
    s.force_permanent_on_battlefield(1, "Plains");
    settle(&mut s);
    assert_eq!(s.life(0), life);
    assert!(!keywords(&s, troop).flying);

    let plains = move_from_library(&mut s, 0, "Plains", ZoneType::Hand);
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

/// Counters go on the Captives, then earthbend 2 animates a land you control.
#[test]
fn rebellious_captives_exhausts_to_grow_and_earthbend_two() {
    let mut s = five_color_setup(&["Rebellious Captives"]);
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
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_activate(&s, captives));
}

/// X counts Forests you control on resolution, including an earthbent Forest
/// and excluding other lands and opponents' Forests; flashback reuses it.
#[test]
fn rockalanche_earthbends_by_forests_you_control_and_flashes_back() {
    let mut s = five_color_setup(&["Rockalanche"]);
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
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_cast(&s, spell));
    cast_card(&mut s, spell);
    assert!(s.choose_target(Target::Permanent(plains)));
    settle(&mut s);
    assert_eq!(counters(&s, plains), 3);
    assert_eq!(s.game().state.zones.zone_of(spell), Some(ZoneType::Exile));
}

#[test]
fn rough_rhino_cavalry_firebends_two_and_exhausts_for_counters_and_trample() {
    let mut s = five_color_setup(&["Rough Rhino Cavalry"]);
    let rhino = s.force_permanent_on_battlefield(0, "Rough Rhino Cavalry");
    s.set_player_mana_pool(0, "8");
    activate(&mut s, rhino, 0);
    settle(&mut s);
    assert_eq!(counters(&s, rhino), 2);
    assert!(keywords(&s, rhino).trample);
    s.set_player_mana_pool(0, "8");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_activate(&s, rhino));

    clear_summoning_sickness(&mut s, rhino);
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

/// Cycling is an activated ability, so it is available at instant speed on
/// the opponent's turn.
#[test]
fn landcycling_works_on_the_opponents_turn() {
    let mut s = five_color_setup(&["Saber-Tooth Moose-Lion"]);
    let card = move_from_library(&mut s, 0, "Saber-Tooth Moose-Lion", ZoneType::Hand);
    s.advance_to_active_step(1, StepKind::Main);
    s.pass_priority();
    assert_eq!(s.action_space().player, Some(PlayerId(0)));
    s.set_player_mana_pool(0, "2");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_activate_card(&s, card));
}

#[test]
fn saber_tooth_moose_lion_has_reach() {
    let mut s = five_color_setup(&["Saber-Tooth Moose-Lion"]);
    let lion = s.force_permanent_on_battlefield(0, "Saber-Tooth Moose-Lion");
    assert!(keywords(&s, lion).reach);
    assert_eq!(s.game().effective_pt(lion), (7, 7));
}

/// Modes choose distinct targets; the destroy mode accepts only creatures with
/// power 4 or greater, judged on current power.
#[test]
fn sandbenders_storm_destroys_large_creatures_or_earthbends_three() {
    let mut s = five_color_setup(&["Sandbenders' Storm", "Gray Ogre", "Craw Wurm"]);
    let small = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let large = s.force_permanent_on_battlefield(1, "Craw Wurm");
    cast(&mut s, "Sandbenders' Storm");
    choose_mode(&mut s, 0);
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
    choose_mode(&mut s, 0);
    assert!(offered_targets(&s).contains(&Target::Permanent(small)));
    assert!(s.choose_target(Target::Permanent(small)));
    settle(&mut s);
    assert!(s.game().state.permanents[small].is_none());

    let plains = s.force_permanent_on_battlefield(0, "Plains");
    cast(&mut s, "Sandbenders' Storm");
    choose_mode(&mut s, 1);
    assert!(s.choose_target(Target::Permanent(plains)));
    settle(&mut s);
    assert!(is_land_creature(&s, plains));
    assert_eq!(counters(&s, plains), 3);
}

/// Seismic Sense looks at one card per land you control (an earthbent land
/// still counts; opponents' lands do not) and may take a creature or land.
#[test]
fn seismic_sense_looks_at_cards_equal_to_your_lands_and_takes_a_creature_or_land() {
    let mut s = five_color_setup(&["Seismic Sense", "Gray Ogre", "Lightning Strike"]);
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

/// Serpent of the Pass costs {1} less per noncreature, nonland card in your
/// graveyard (creatures and lands there do not count), and with three Lessons
/// there it may be cast at instant speed.
#[test]
fn serpent_of_the_pass_is_cheaper_per_graveyard_spell_and_flashes_with_three_lessons() {
    let mut s = five_color_setup(&[
        "Serpent of the Pass",
        "Lightning Strike",
        "Earthbending Lesson",
        "Gray Ogre",
    ]);
    let serpent = move_from_library(&mut s, 0, "Serpent of the Pass", ZoneType::Hand);
    move_from_library(&mut s, 0, "Gray Ogre", ZoneType::Graveyard);
    move_from_library(&mut s, 0, "Forest", ZoneType::Graveyard);
    for _ in 0..2 {
        move_from_library(&mut s, 0, "Lightning Strike", ZoneType::Graveyard);
    }
    // {5}{U}{U} less two.
    s.set_player_mana_pool(0, "2UU");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_cast(&s, serpent));
    s.set_player_mana_pool(0, "3UU");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_cast(&s, serpent));

    // Two Lessons: sorcery timing only, so not in the beginning of combat.
    for _ in 0..2 {
        move_from_library(&mut s, 0, "Earthbending Lesson", ZoneType::Graveyard);
    }
    s.advance_to_active_step(0, StepKind::BeginningOfCombat);
    s.set_player_mana_pool(0, "20UU");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_cast(&s, serpent));
    move_from_library(&mut s, 0, "Earthbending Lesson", ZoneType::Graveyard);
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_cast(&s, serpent));
    // Seven noncreature, nonland cards reduce all {5} generic.
    s.set_player_mana_pool(0, "UU");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_cast(&s, serpent));
}

/// Sokka discards up to two cards and only then draws that many, so a drawn
/// card cannot be discarded; casting an artifact or a Lesson grows him.
#[test]
fn sokka_discards_up_to_two_then_draws_that_many_and_grows_on_artifacts_and_lessons() {
    let mut s = five_color_setup(&[
        "Sokka, Bold Boomeranger",
        "Earthbending Lesson",
        "Barrels of Blasting Jelly",
        "Gray Ogre",
    ]);
    let kept = move_from_library(&mut s, 0, "Gray Ogre", ZoneType::Hand);
    let first = move_from_library(&mut s, 0, "Forest", ZoneType::Hand);
    let second = move_from_library(&mut s, 0, "Island", ZoneType::Hand);
    cast(&mut s, "Sokka, Bold Boomeranger");
    pass_until(&mut s, ActionSpaceKind::Discard);
    choose(
        &mut s,
        |a| matches!(a, Action::SelectCard { card, .. } if *card == first),
    );
    assert_eq!(s.action_space().kind, ActionSpaceKind::Discard);
    assert_eq!(hand(&s, 0).len(), 2, "no draw before the discards finish");
    choose(
        &mut s,
        |a| matches!(a, Action::SelectCard { card, .. } if *card == second),
    );
    settle(&mut s);
    let in_hand = hand(&s, 0);
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
    let mut s = five_color_setup(&["Sokka, Bold Boomeranger"]);
    move_from_library(&mut s, 0, "Forest", ZoneType::Hand);
    cast(&mut s, "Sokka, Bold Boomeranger");
    pass_until(&mut s, ActionSpaceKind::Discard);
    choose(&mut s, |a| matches!(a, Action::Decline { .. }));
    settle(&mut s);
    assert_eq!(hand(&s, 0).len(), 1);
}

/// Solstice Revelations exiles to the first nonland card and offers it free
/// only when its mana value is below the Mountains you control; otherwise it
/// goes to hand.
#[test]
fn solstice_revelations_casts_cheap_spells_free_and_keeps_the_rest() {
    let mut s = five_color_setup(&["Solstice Revelations", "Lightning Strike"]);
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

/// ...nor when an opponent's Ally enters...
#[test]
fn avatar_enthusiasts_no_counter_for_opponents_ally() {
    let (mut s, enthusiasts) = enthusiasts_scenario();

    enters_battlefield_with_triggers(&mut s, 1, "South Pole Voyager");
    resolve_until_priority(&mut s);

    assert_eq!(counters(&s, enthusiasts), 0);
}

/// ...nor for itself entering ("another").
#[test]
fn avatar_enthusiasts_does_not_count_itself() {
    let list = deck(&[("Plains", 24), ("Avatar Enthusiasts", 16)]);
    let mut s = Scenario::new(list.clone(), list, 12);

    let entered = enters_battlefield_with_triggers(&mut s, 0, "Avatar Enthusiasts");
    resolve_until_priority(&mut s);

    assert_eq!(counters(&s, entered), 0);
}

/// A created Ally token is "another Ally you control entering" too.
#[test]
fn avatar_enthusiasts_counts_token_allies() {
    let (mut s, enthusiasts) = enthusiasts_scenario();

    s.game_mut().create_token("Ally", PlayerId(0), false);
    resolve_until_priority(&mut s);

    assert_eq!(counters(&s, enthusiasts), 1);
}

#[test]
fn azula_announces_both_modes_then_binds_independent_targets_including_same_object() {
    for same in [false, true] {
        let mut s = five_color_setup(&["Azula Always Lies", "Gnarled Professor"]);
        let first = s.force_permanent_on_battlefield(0, "Gnarled Professor");
        let second = if same {
            first
        } else {
            s.force_permanent_on_battlefield(1, "Gnarled Professor")
        };
        let spell = cast(&mut s, "Azula Always Lies");
        assert_eq!(s.game().state.zones.zone_of(spell), Some(ZoneType::Hand));
        assert_eq!(s.action_space().actions.len(), 3);
        choose_mode(&mut s, 2);
        assert!(s.choose_target(Target::Permanent(first)));
        assert!(s.choose_target(Target::Permanent(second)));
        settle(&mut s);
        assert_eq!(counters(&s, second), 1);
        assert_eq!(s.game().effective_power(first), if same { 5 } else { 4 });
    }
}

fn cub_deck() -> BTreeMap<String, usize> {
    deck(&[("Forest", 24), ("Badgermole Cub", 16)])
}

/// Cast a Cub with lands out; answer the earthbend target choice with the
/// first offered land. Returns the animated land's permanent id.
fn earthbend_a_land(s: &mut Scenario) -> PermanentId {
    cast_and_resolve(s);
    // ETB trigger needs a target: land you control.
    assert_eq!(s.action_space().kind, ActionSpaceKind::ChooseTarget);
    let target = s
        .action_space()
        .actions
        .iter()
        .find_map(|action| match action {
            Action::ChooseTarget {
                target: managym::state::target::Target::Permanent(p),
                ..
            } => Some(*p),
            _ => None,
        })
        .expect("land target should be offered");
    assert!(s.choose_target(Target::Permanent(target)));
    s.pass_priority();
    s.pass_priority(); // trigger resolves
    target
}

#[test]
fn earthbend_animates_land_with_counter() {
    let mut s = Scenario::new(cub_deck(), deck(&[("Plains", 40)]), 301);
    force_lands(&mut s, 0, "Forest", 3);
    clear_hand(&mut s, 0);
    s.force_card_in_hand(0, "Badgermole Cub");
    s.advance_to_active_step(0, StepKind::Main);

    let land = earthbend_a_land(&mut s);
    let perm = permanent(&s, land);
    assert!(perm.animated, "land should be animated");
    assert_eq!(perm.plus1_counters, 1);
    // 0/0 creature with one +1/+1 counter — a 1/1 that's still a land.
    assert_eq!(s.game().effective_power(land), 1);
    assert_eq!(s.game().effective_toughness(land), 1);
    let card = &s.game().state.cards[perm.card];
    assert!(card.types.is_land(), "still a land");
    assert!(!card.types.is_creature(), "printed types unchanged");
}

#[test]
fn earthbend_targets_only_own_lands() {
    let mut s = Scenario::new(cub_deck(), deck(&[("Plains", 40)]), 302);
    force_lands(&mut s, 0, "Forest", 2);
    force_lands(&mut s, 1, "Plains", 2);
    clear_hand(&mut s, 0);
    s.force_card_in_hand(0, "Badgermole Cub");
    s.advance_to_active_step(0, StepKind::Main);
    cast_and_resolve(&mut s);

    assert_eq!(s.action_space().kind, ActionSpaceKind::ChooseTarget);
    // Only the controller's two forests are legal targets.
    assert_eq!(s.action_space().actions.len(), 2);
}

#[test]
fn earthbend_trigger_removed_without_lands() {
    let mut s = Scenario::new(
        deck(&[("Badgermole Cub", 40)]),
        deck(&[("Plains", 40)]),
        303,
    );
    clear_hand(&mut s, 0);
    s.advance_to_active_step(0, StepKind::Main);
    // Enter the battlefield through the event path so the ETB fires.
    let cub = find_owned_card(&s, 0, "Badgermole Cub");
    s.game_mut().move_card(cub, ZoneType::Battlefield);
    s.pass_priority();
    // No land to target: the trigger is removed (CR 603.3d); play goes on.
    assert!(s.game().state.pending_triggers.is_empty());
    assert!(s.game().state.trigger_placement.is_none());
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
}

#[test]
fn animated_land_attacks_with_haste_and_still_taps_for_mana() {
    let mut s = Scenario::new(cub_deck(), deck(&[("Plains", 40)]), 304);
    force_lands(&mut s, 0, "Forest", 3);
    clear_hand(&mut s, 0);
    s.force_card_in_hand(0, "Badgermole Cub");
    s.advance_to_active_step(0, StepKind::Main);
    // Pay from the pool so no forest is tapped for the cast.
    s.set_player_mana_pool(0, "1G");
    let land = earthbend_a_land(&mut s);

    // Still a land and still untapped (the cast was paid from the pool).
    assert!(!permanent(&s, land).tapped);

    // It can attack this very turn (haste; it was never summoning sick).
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    let mut attacked = false;
    while s.action_space().kind == ActionSpaceKind::DeclareAttacker {
        let space = s.action_space().clone();
        let attack_land = space.actions.iter().position(|action| {
        matches!(action, Action::DeclareAttacker { permanent, attack: true, .. } if *permanent == land)
    });
        match attack_land {
            Some(index) => {
                s.step_action(index);
                attacked = true;
            }
            None => s.decline_attack(),
        }
    }
    assert!(attacked, "animated land should be an eligible attacker");
    s.advance_to_active_step(0, StepKind::EndOfCombat);
    // Unblocked 1/1: one damage to the defender.
    s.assert_life(1, 19);
}

#[test]
fn animated_land_dies_and_returns_tapped_as_plain_land() {
    let mut s = Scenario::new(
        cub_deck(),
        deck(&[("Mountain", 24), ("Lightning Bolt", 16)]),
        305,
    );
    force_lands(&mut s, 0, "Forest", 3);
    force_lands(&mut s, 1, "Mountain", 1);
    clear_hand(&mut s, 0);
    clear_hand(&mut s, 1);
    s.force_card_in_hand(0, "Badgermole Cub");
    s.force_card_in_hand(1, "Lightning Bolt");
    s.advance_to_active_step(0, StepKind::Main);
    let land = earthbend_a_land(&mut s);
    let land_card = permanent(&s, land).card;

    // Opponent bolts the 1/1 land-creature.
    s.pass_priority();
    cast_only(&mut s);
    assert!(s.choose_target(Target::Permanent(land)));
    s.pass_priority();
    s.pass_priority(); // bolt resolves, land dies -> delayed trigger fires
    s.pass_priority();
    s.pass_priority(); // return trigger resolves

    assert_eq!(
        s.game().state.zones.zone_of(land_card),
        Some(ZoneType::Battlefield),
        "land should be returned to the battlefield"
    );
    let new_perm_id = s.game().state.card_to_permanent[land_card].expect("re-entered");
    let new_perm = permanent(&s, new_perm_id);
    assert!(new_perm.tapped, "returns tapped");
    assert!(!new_perm.animated, "returns as a plain land");
    assert_eq!(new_perm.plus1_counters, 0, "counters do not persist");
    assert!(
        s.game().state.delayed_triggers.is_empty(),
        "delayed trigger is one-shot"
    );
}

#[test]
fn animated_land_exiled_returns_tapped() {
    let mut s = Scenario::new(cub_deck(), deck(&[("Plains", 40)]), 306);
    force_lands(&mut s, 0, "Forest", 3);
    clear_hand(&mut s, 0);
    s.force_card_in_hand(0, "Badgermole Cub");
    s.advance_to_active_step(0, StepKind::Main);
    let land = earthbend_a_land(&mut s);
    let land_card = permanent(&s, land).card;

    // Exile it directly (no Milestone-1 card exiles a land; the delayed
    // trigger must still see it).
    s.game_mut().move_card(land_card, ZoneType::Exile);
    resolve_everything(&mut s); // return trigger resolves

    assert_eq!(
        s.game().state.zones.zone_of(land_card),
        Some(ZoneType::Battlefield)
    );
    let new_perm_id = s.game().state.card_to_permanent[land_card].expect("re-entered");
    assert!(permanent(&s, new_perm_id).tapped);
}

#[test]
fn animated_land_bounced_drops_the_delayed_trigger() {
    let mut s = Scenario::new(cub_deck(), deck(&[("Plains", 40)]), 307);
    force_lands(&mut s, 0, "Forest", 3);
    clear_hand(&mut s, 0);
    s.force_card_in_hand(0, "Badgermole Cub");
    s.advance_to_active_step(0, StepKind::Main);
    let land = earthbend_a_land(&mut s);
    let land_card = permanent(&s, land).card;
    assert_eq!(s.game().state.delayed_triggers.len(), 1);

    // To hand is neither "dies" nor "is exiled": the delayed trigger is
    // dropped, not fired.
    s.game_mut().move_card(land_card, ZoneType::Hand);
    assert!(s.game().state.delayed_triggers.is_empty());
    assert!(s.game().state.pending_triggers.is_empty());
    assert_eq!(
        s.game().state.zones.zone_of(land_card),
        Some(ZoneType::Hand)
    );
}

#[test]
fn zero_toughness_animated_land_dies_by_sba_and_returns() {
    let mut s = Scenario::new(cub_deck(), deck(&[("Plains", 40)]), 308);
    force_lands(&mut s, 0, "Forest", 3);
    clear_hand(&mut s, 0);
    s.force_card_in_hand(0, "Badgermole Cub");
    s.advance_to_active_step(0, StepKind::Main);
    let land = earthbend_a_land(&mut s);
    let land_card = permanent(&s, land).card;

    // Strip its counter: a 0/0 creature dies as a state-based action
    // (CR 704.5f) — and the earthbend delayed trigger returns it tapped.
    s.game_mut().state.permanents[land]
        .as_mut()
        .expect("permanent")
        .plus1_counters = 0;
    s.pass_priority(); // any action: SBAs run, the 0/0 dies
    resolve_everything(&mut s); // return trigger resolves

    assert_eq!(
        s.game().state.zones.zone_of(land_card),
        Some(ZoneType::Battlefield)
    );
    let new_perm_id = s.game().state.card_to_permanent[land_card].expect("re-entered");
    assert!(permanent(&s, new_perm_id).tapped);
    assert!(!permanent(&s, new_perm_id).animated);
}

#[test]
fn boomerang_draws_only_when_its_target_was_controlled_by_the_caster() {
    for player in [0, 1] {
        let mut s = five_color_setup(&["Boomerang Basics", "Gnarled Professor"]);
        let target = s.force_permanent_on_battlefield(player, "Gnarled Professor");
        let target_card = s.game().state.permanents[target].as_ref().unwrap().card;
        cast(&mut s, "Boomerang Basics");
        assert!(s.choose_target(Target::Permanent(target)));
        settle(&mut s);
        assert_eq!(
            s.game().state.zones.zone_of(target_card),
            Some(ZoneType::Hand)
        );
        assert_eq!(
            s.zone_size(0, ZoneType::Hand),
            if player == 0 { 2 } else { 0 }
        );
    }
}

#[test]
fn boomerang_and_zuko_use_resolution_time_control_after_target_announcement() {
    for spell in ["Boomerang Basics", "Zuko's Exile"] {
        let mut s = five_color_setup(&[spell, "Gnarled Professor"]);
        let target = s.force_permanent_on_battlefield(1, "Gnarled Professor");
        cast(&mut s, spell);
        assert!(s.choose_target(Target::Permanent(target)));
        // The scenario seam models a completed control-change effect in response.
        s.game_mut().state.permanents[target]
            .as_mut()
            .unwrap()
            .controller = PlayerId(0);
        let hand = s.zone_size(0, ZoneType::Hand);
        settle(&mut s);
        if spell == "Boomerang Basics" {
            assert_eq!(s.zone_size(0, ZoneType::Hand), hand + 1);
            assert_eq!(s.zone_size(1, ZoneType::Hand), 1); // owner's hand, controller draws
        } else {
            assert_eq!(s.battlefield_permanents_named(0, "Clue").len(), 1);
            assert!(s.battlefield_permanents_named(1, "Clue").is_empty());
        }
    }
}

#[test]
fn cat_gator_uses_swamps_and_lifelink_on_its_enter_trigger() {
    let mut s = five_color_setup(&["Cat-Gator"]);
    s.force_permanent_on_battlefield(0, "Swamp");
    s.force_permanent_on_battlefield(0, "Swamp");
    cast(&mut s, "Cat-Gator");
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    settle(&mut s);
    assert_eq!(s.life(1), 18);
    assert_eq!(s.life(0), 22);
}

#[test]
fn announced_enter_trigger_keeps_its_controller_when_its_source_changes_control() {
    let mut s = five_color_setup(&["Cat-Gator"]);
    s.force_permanent_on_battlefield(0, "Swamp");
    cast(&mut s, "Cat-Gator");
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    let source = permanent_named(&s, "Cat-Gator");
    s.game_mut().state.permanents[source]
        .as_mut()
        .unwrap()
        .controller = PlayerId(1);
    let life = s.life(1);
    settle(&mut s);
    // The trigger counts its controller's Swamps, while lifelink follows the live source.
    assert_eq!(s.life(1), life);
    assert_eq!(s.life(0), 20);
}

#[test]
fn cat_owl_attack_untaps_artifacts_or_creatures_in_either_control() {
    for artifact in [false, true] {
        let mut s = five_color_setup(&["Cat-Owl", "Prophetic Prism", "Gray Ogre"]);
        let owl = s.force_permanent_on_battlefield(0, "Cat-Owl");
        assert!(keywords(&s, owl).flying);
        s.game_mut().state.permanents[owl]
            .as_mut()
            .unwrap()
            .summoning_sick = false;
        let target = s.force_permanent_on_battlefield(
            if artifact { 1 } else { 0 },
            if artifact {
                "Prophetic Prism"
            } else {
                "Gray Ogre"
            },
        );
        s.game_mut().state.permanents[target]
            .as_mut()
            .unwrap()
            .tapped = true;
        let land = s.force_permanent_on_battlefield(1, "Island");
        s.advance_to_active_step(0, StepKind::DeclareAttackers);
        choose(
            &mut s,
            |a| matches!(a,Action::DeclareAttacker{permanent,attack:true,..} if *permanent==owl),
        );
        pass_until(&mut s, ActionSpaceKind::ChooseTarget);
        assert!(!s
            .action_space()
            .actions
            .iter()
            .any(|a| matches!(a,Action::ChooseTarget{target:Target::Permanent(p),..} if *p==land)));
        assert!(s.choose_target(Target::Permanent(target)));
        settle(&mut s);
        assert!(!tapped(&s, target));
    }
}

fn plant_lessons(s: &mut Scenario, player: usize, n: usize) {
    for _ in 0..n {
        let card = find_owned_card(s, player, "Accumulate Wisdom");
        s.game_mut().move_card(card, ZoneType::Graveyard);
    }
}

/// Combustion Technique cast at a Water Tribe Rallier with five +1/+1 counters,
/// with `lessons` Lessons in player 0's graveyard.
fn combustion(lessons: usize) -> (Scenario, CardId, PermanentId) {
    let mut s = ur_lessons_setup();
    plant_lessons(&mut s, 0, lessons);
    plant_lessons(&mut s, 1, 3); // opponent's Lessons are irrelevant
    put_onto_battlefield(&mut s, 0, "Mountain");
    put_onto_battlefield(&mut s, 0, "Mountain");
    let victim = put_onto_battlefield(&mut s, 1, "Water Tribe Rallier"); // 3/2
    s.game_mut().state.permanents[victim]
        .as_mut()
        .unwrap()
        .plus1_counters = 5;
    let spell = cast_owned(&mut s, "Combustion Technique");
    assert!(s.choose_target(Target::Permanent(victim)));
    resolve_top(&mut s);
    (s, spell, victim)
}

#[test]
fn combustion_damage_counts_own_lessons_excludes_resolving_spell_and_exiles_later_death() {
    for lessons in [0, 3] {
        let (mut s, spell, victim) = combustion(lessons);
        let permanent = s.game().state.permanents[victim].as_ref().unwrap();
        let card = permanent.card;
        assert_eq!(permanent.damage, 2 + lessons as i32);
        assert!(permanent.exile_if_dies_this_turn);
        assert_eq!(
            s.game().state.zones.zone_of(spell),
            Some(ZoneType::Graveyard)
        );
        let obs = Observation::new(s.game(), &[]);
        assert!(obs
            .opponent_permanents
            .iter()
            .any(|p| p.exile_if_dies_this_turn));
        // Any death, including sacrifice, is replaced (CR 614.1a).
        s.game_mut().move_card(card, ZoneType::Graveyard);
        assert_eq!(s.game().state.zones.zone_of(card), Some(ZoneType::Exile));
    }
}

#[test]
fn combustion_lethal_damage_exiles_without_a_graveyard_move() {
    let mut s = ur_lessons_setup();
    put_onto_battlefield(&mut s, 0, "Mountain");
    put_onto_battlefield(&mut s, 0, "Mountain");
    let victim = put_onto_battlefield(&mut s, 1, "Gran-Gran");
    let card = card_of(&s, victim);
    cast_owned(&mut s, "Combustion Technique");
    assert!(s.choose_target(Target::Permanent(victim)));
    resolve_top(&mut s);
    assert_eq!(s.game().state.zones.zone_of(card), Some(ZoneType::Exile));
}

#[test]
fn combustion_replacement_expires_at_cleanup_and_does_not_follow_reentry() {
    for bounce in [false, true] {
        let (mut s, _, victim) = combustion(0);
        let card = card_of(&s, victim);
        if bounce {
            s.game_mut().move_card(card, ZoneType::Hand);
            s.game_mut().move_card(card, ZoneType::Battlefield);
        } else {
            s.advance_to_active_step(1, StepKind::Upkeep);
        }
        let current = s.game().state.card_to_permanent[card].unwrap();
        assert!(
            !s.game().state.permanents[current]
                .as_ref()
                .unwrap()
                .exile_if_dies_this_turn
        );
        s.game_mut().move_card(card, ZoneType::Graveyard);
        assert_eq!(
            s.game().state.zones.zone_of(card),
            Some(ZoneType::Graveyard)
        );
    }
}

#[test]
fn combustion_is_an_owned_learn_candidate() {
    let mut s = Scenario::with_sideboards(
        ur_lessons_deck(),
        ur_lessons_deck(),
        BTreeMap::from([("Combustion Technique".into(), 1)]),
        BTreeMap::new(),
        88,
    );
    s.advance_to_active_step(0, StepKind::Main);
    for _ in 0..3 {
        put_onto_battlefield(&mut s, 0, "Island");
    }
    cast_owned(&mut s, "Pop Quiz");
    resolve_top(&mut s);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Learn);
    let (index, card) = s
        .action_space()
        .actions
        .iter()
        .enumerate()
        .find_map(|(i, a)| match a {
            Action::LearnTakeLesson { card, .. } => Some((i, *card)),
            _ => None,
        })
        .unwrap();
    assert_eq!(s.game().state.cards[card].name, "Combustion Technique");
    s.step_action(index);
    assert_eq!(s.game().state.zones.zone_of(card), Some(ZoneType::Hand));
}

#[test]
fn combustion_invalid_target_does_not_damage_or_mark_new_incarnation() {
    let mut s = ur_lessons_setup();
    put_onto_battlefield(&mut s, 0, "Mountain");
    put_onto_battlefield(&mut s, 0, "Mountain");
    let gran = put_onto_battlefield(&mut s, 1, "Gran-Gran");
    let spell = cast_owned(&mut s, "Combustion Technique");
    assert!(s.choose_target(Target::Permanent(gran)));
    let card = card_of(&s, gran);
    s.game_mut().move_card(card, ZoneType::Hand);
    s.game_mut().move_card(card, ZoneType::Battlefield);
    let new = s.game().state.card_to_permanent[card].unwrap();
    resolve_top(&mut s);
    assert_eq!(
        s.game().state.zones.zone_of(spell),
        Some(ZoneType::Graveyard)
    );
    let permanent = s.game().state.permanents[new].as_ref().unwrap();
    assert_eq!(permanent.damage, 0);
    assert!(!permanent.exile_if_dies_this_turn);
}

#[test]
fn combustion_replacement_is_hashed_visible_and_rolls_back_with_its_damage() {
    use managym::search_state::{BenchCommand, BranchDriver, ClonePlusUndoDriver};
    let mut s = ur_lessons_setup();
    put_onto_battlefield(&mut s, 0, "Mountain");
    put_onto_battlefield(&mut s, 0, "Mountain");
    let gran = put_onto_battlefield(&mut s, 1, "Gran-Gran");
    s.game_mut().state.permanents[gran]
        .as_mut()
        .unwrap()
        .plus1_counters = 3;
    cast_owned(&mut s, "Combustion Technique");
    assert!(s.choose_target(Target::Permanent(gran)));
    s.pass_priority();
    let driver = ClonePlusUndoDriver::default();
    let root = driver.witness(s.game());
    let mut branch = driver.fork_exact(s.game());
    let mark = driver.mark(&mut branch);
    let index = branch
        .action_space()
        .unwrap()
        .actions
        .iter()
        .position(|a| matches!(a, Action::PassPriority { .. }))
        .unwrap();
    driver
        .apply(
            &mut branch,
            BenchCommand {
                action_index: index,
                expected_action_hash: None,
                expected_state_hash: None,
            },
        )
        .unwrap();
    assert!(
        branch.state.permanents[gran]
            .as_ref()
            .unwrap()
            .exile_if_dies_this_turn
    );
    assert_ne!(driver.witness(&branch), root);
    driver.rollback(&mut branch, mark);
    assert_eq!(driver.witness(&branch), root);
    assert_eq!(driver.witness(s.game()), root);
}

#[test]
fn healer_tap_gains_life_and_scries() {
    let mut s = Scenario::new(
        deck(&[("Plains", 24), ("Compassionate Healer", 16)]),
        deck(&[("Plains", 40)]),
        351,
    );
    let healer = s.force_permanent_on_battlefield(0, "Compassionate Healer");
    clear_summoning_sickness(&mut s, healer);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, healer);
    // Attacking without vigilance taps the Healer -> trigger.
    s.pass_priority();
    s.pass_priority(); // trigger resolves: gain 1 life, then scry decision
    s.assert_life(0, 21);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Scry);
    // Keep the top card.
    let top = *s
        .game()
        .state
        .zones
        .zone_cards(ZoneType::Library, PlayerId(0))
        .last()
        .expect("library nonempty");
    let keep = s
        .action_space()
        .actions
        .iter()
        .position(|action| {
            matches!(
                action,
                Action::ScryCard {
                    to_bottom: false,
                    ..
                }
            )
        })
        .expect("keep action");
    s.step_action(keep);
    assert_eq!(
        s.game()
            .state
            .zones
            .zone_cards(ZoneType::Library, PlayerId(0))
            .last(),
        Some(&top)
    );
}

#[test]
fn cunning_maneuver_buffs_and_investigates_but_all_invalid_targets_stop_it() {
    for remove in [false, true] {
        let mut s = five_color_setup(&["Cunning Maneuver", "Gray Ogre"]);
        let creature = s.force_permanent_on_battlefield(0, "Gray Ogre");
        cast(&mut s, "Cunning Maneuver");
        assert!(s.choose_target(Target::Permanent(creature)));
        if remove {
            let id = s.game().state.permanents[creature].as_ref().unwrap().card;
            s.game_mut().move_card(id, ZoneType::Graveyard);
        }
        settle(&mut s);
        assert_eq!(
            s.battlefield_permanents_named(0, "Clue").len(),
            usize::from(!remove)
        );
        if !remove {
            assert_eq!(s.game().effective_pt(creature), (5, 3));
            s.advance_to_active_step(1, StepKind::Main);
            assert_eq!(s.game().effective_power(creature), 2);
        }
    }
}

#[test]
fn farm_animals_can_decline_target_and_still_pay_sacrifice() {
    let mut s = five_color_setup(&["Curious Farm Animals"]);
    play_and_settle(&mut s, "Curious Farm Animals");
    let id = permanent_named(&s, "Curious Farm Animals");
    activate_first(&mut s, id);
    choose(&mut s, |a| matches!(a, Action::Decline { .. }));
    settle(&mut s);
    assert_eq!(s.game().state.players[0].life, 23);
    assert!(s
        .battlefield_permanents_named(0, "Curious Farm Animals")
        .is_empty());
}

#[test]
fn farm_animals_destroy_an_artifact_without_targeting_a_creature() {
    let mut s = five_color_setup(&["Curious Farm Animals", "Callous Inspector"]);
    play_and_settle(&mut s, "Curious Farm Animals");
    play_and_settle(&mut s, "Callous Inspector");
    s.game_mut().create_token("Food", PlayerId(0), false);
    let food = permanent_named(&s, "Food");
    let animal = permanent_named(&s, "Curious Farm Animals");
    let creature = permanent_named(&s, "Callous Inspector");
    activate_first(&mut s, animal);
    assert!(!s.action_space().actions.iter().any(
        |a| matches!(a,Action::ChooseTarget {target:Target::Permanent(id),..} if *id==creature)
    ));
    choose(
        &mut s,
        |a| matches!(a,Action::ChooseTarget {target:Target::Permanent(id),..} if *id==food),
    );
    assert!(s
        .battlefield_permanents_named(0, "Curious Farm Animals")
        .is_empty());
    assert_eq!(s.battlefield_permanents_named(0, "Food").len(), 1);
    settle(&mut s);
    assert!(s.battlefield_permanents_named(0, "Food").is_empty());
    assert_eq!(s.game().state.players[0].life, 23);
}

#[test]
fn cycle_sacrifices_during_resolution_and_searches_two_distinct_basics() {
    let mut s = five_color_setup(&["Cycle of Renewal"]);
    let sacrificed = play_and_settle(&mut s, "Forest");
    cast(&mut s, "Cycle of Renewal");
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    select_card(&mut s, sacrificed);
    let one = library_card(&s, 0, "Forest");
    select_card(&mut s, one);
    let two = library_card(&s, 0, "Swamp");
    select_card(&mut s, two);
    settle(&mut s);
    for found in [one, two] {
        let id = s.game().state.card_to_permanent[found].unwrap();
        assert!(s.game().state.permanents[id].as_ref().unwrap().tapped);
    }
    assert_eq!(
        s.game().state.zones.zone_of(sacrificed),
        Some(ZoneType::Graveyard)
    );
}

#[test]
fn dai_li_targets_same_land_twice_then_drains_for_creatures_with_counters() {
    let mut s = five_color_setup(&["Dai Li Agents"]);
    let land = s.force_permanent_on_battlefield(0, "Forest");
    cast(&mut s, "Dai Li Agents");
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(land)));
    assert!(s.choose_target(Target::Permanent(land)));
    settle(&mut s);
    assert_eq!(counters(&s, land), 2);
    let agents = permanent_named(&s, "Dai Li Agents");
    s.game_mut().state.permanents[agents]
        .as_mut()
        .unwrap()
        .summoning_sick = false;
    s.advance_to_active_step(0, StepKind::BeginningOfCombat);
    s.pass_priority();
    s.pass_priority();
    while s.action_space().kind == ActionSpaceKind::DeclareAttacker {
        let index = s.action_space().actions.iter().position(|a| matches!(a, Action::DeclareAttacker { permanent, attack, .. } if (*permanent == agents) == *attack)).unwrap();
        s.step_action(index);
    }
    settle(&mut s);
    s.assert_life(0, 21);
    s.assert_life(1, 19);
}

#[test]
fn precision_has_a_real_four_mana_alternative() {
    let mut s = five_color_setup(&["Deadly Precision", "Callous Inspector"]);
    let victim = play_and_settle(&mut s, "Callous Inspector");
    let id = permanent_named(&s, "Callous Inspector");
    // Generic payment must leave black available for the base spell cost.
    s.set_player_mana_pool(0, "BGGGG");
    cast(&mut s, "Deadly Precision");
    choose(
        &mut s,
        |a| matches!(a,Action::ChooseTarget {target:Target::Permanent(p),..} if *p==id),
    );
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    assert_eq!(
        s.game().state.zones.zone_of(victim),
        Some(ZoneType::Battlefield)
    );
    assert_eq!(s.game().state.players[0].mana_pool.mana, [0; 6]);
    settle(&mut s);
    assert_eq!(
        s.game().state.zones.zone_of(victim),
        Some(ZoneType::Graveyard)
    );
}

#[test]
fn treasure_completes_an_announced_cost_without_a_priority_window() {
    let mut s = five_color_setup(&["Deadly Precision", "Callous Inspector"]);
    let victim = play_and_settle(&mut s, "Callous Inspector");
    let target = permanent_named(&s, "Callous Inspector");
    s.game_mut().create_token("Treasure", PlayerId(0), false);
    s.set_player_mana_pool(0, "0");
    cast(&mut s, "Deadly Precision");
    choose(
        &mut s,
        |a| matches!(a,Action::ChooseTarget {target:Target::Permanent(p),..} if *p==target),
    );
    assert_choice_contract(&s);
    select_card(&mut s, victim);
    let context = s.game().decision_context(PlayerId(0)).unwrap();
    assert_eq!(context.stage, "payment");
    assert_eq!(context.mana_cost.0.as_deref(), Some("{B}"));
    assert_eq!(
        context.reserved_sacrifice.0.unwrap().entity_id,
        victim.0 as u32
    );
    assert_eq!(context.targets[0].clause, 0);
    let public = s.game().decision_context(PlayerId(1)).unwrap();
    assert_eq!(public.actor, 0);
    assert!(public.source.0.is_none() && public.targets.is_empty());
    let treasure = permanent_named(&s, "Treasure");
    activate_first(&mut s, treasure);
    assert_eq!(s.action_space().actions.len(), 1);
    let context = s.game().decision_context(PlayerId(0)).unwrap();
    assert_eq!(context.stage, "mana_color");
    assert_eq!(context.source.0.unwrap().label, "Deadly Precision");
    assert!(context
        .recent_facts
        .iter()
        .any(|fact| fact.contains("sacrificed Treasure")));
    let frame = s.game().semantic_decision_frame().unwrap();
    assert_eq!(frame.offers[0].label, "Add {B}");
    assert_choice_contract(&s);
    choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 2, .. }));
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    assert_eq!(
        s.game().state.zones.zone_of(victim),
        Some(ZoneType::Graveyard)
    );
    assert!(s.battlefield_permanents_named(0, "Treasure").is_empty());
}

#[test]
fn deadly_precision_sacrifices_artifact_before_destroying_a_different_creature() {
    let mut s = five_color_setup(&["Deadly Precision", "Gray Ogre"]);
    let victim = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let victim_card = s.game().state.permanents[victim].as_ref().unwrap().card;
    s.game_mut().create_token("Clue", PlayerId(0), false);
    let clue = permanent_named(&s, "Clue");
    let clue_card = s.game().state.permanents[clue].as_ref().unwrap().card;
    s.set_player_mana_pool(0, "B");
    cast(&mut s, "Deadly Precision");
    assert!(s.choose_target(Target::Permanent(victim)));
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::PayCost { .. })));
    choose(
        &mut s,
        |a| matches!(a, Action::SelectCard { card, .. } if *card == clue_card),
    );
    assert!(s.game().state.permanents[clue].is_none());
    assert!(s.game().state.permanents[victim].is_some());
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
    settle(&mut s);
    assert_eq!(
        s.game().state.zones.zone_of(victim_card),
        Some(ZoneType::Graveyard)
    );
}

#[test]
fn disciple_targets_another_small_creature_pays_tap_and_survives_source_departure() {
    let mut s = five_color_setup(&[
        "Deserter's Disciple",
        "Monastery Swiftspear",
        "Gnarled Professor",
    ]);
    let disciple = s.force_permanent_on_battlefield(0, "Deserter's Disciple");
    let small = s.force_permanent_on_battlefield(0, "Monastery Swiftspear");
    s.force_permanent_on_battlefield(0, "Gnarled Professor");
    s.game_mut().state.permanents[disciple]
        .as_mut()
        .unwrap()
        .summoning_sick = false;
    s.game_mut().scenario_refresh_priority().unwrap();
    let index = s
        .action_space()
        .actions
        .iter()
        .position(
            |a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == disciple),
        )
        .unwrap();
    s.step_action(index);
    assert_eq!(s.action_space().actions.len(), 1);
    assert!(s.choose_target(Target::Permanent(small)));
    let source = s.game().state.permanents[disciple].as_ref().unwrap().card;
    assert!(s.game().state.permanents[disciple].as_ref().unwrap().tapped);
    s.game_mut().move_card(source, ZoneType::Graveyard);
    // Resolve through the branch driver as well: the independent target
    // mutation must disappear when the search branch rolls back.
    assert_choice_contract(&s);
    settle(&mut s);
    assert!(
        s.game().state.permanents[small]
            .as_ref()
            .unwrap()
            .cant_be_blocked_this_turn
    );
}

#[test]
fn disciple_rechecks_power_and_its_effect_expires() {
    for increased_before_resolution in [false, true] {
        let mut s = five_color_setup(&["Deserter's Disciple", "Gray Ogre"]);
        let disciple = s.force_permanent_on_battlefield(0, "Deserter's Disciple");
        let target = s.force_permanent_on_battlefield(0, "Gray Ogre");
        s.game_mut().state.permanents[disciple]
            .as_mut()
            .unwrap()
            .summoning_sick = false;
        activate(&mut s, disciple, 0);
        assert!(!s.choose_target(Target::Permanent(disciple)));
        assert!(s.choose_target(Target::Permanent(target)));
        if increased_before_resolution {
            s.game_mut().state.permanents[target]
                .as_mut()
                .unwrap()
                .plus1_counters = 1;
        }
        settle(&mut s);
        if !increased_before_resolution {
            s.game_mut().state.permanents[target]
                .as_mut()
                .unwrap()
                .plus1_counters = 1;
        }
        assert_eq!(
            s.game().state.permanents[target]
                .as_ref()
                .unwrap()
                .cant_be_blocked_this_turn,
            !increased_before_resolution
        );
        s.advance_to_active_step(1, StepKind::Main);
        assert!(
            !s.game().state.permanents[target]
                .as_ref()
                .unwrap()
                .cant_be_blocked_this_turn
        );
    }
}

#[test]
fn dragon_and_soldier_tokens_firebend_their_printed_amount() {
    for (name, amount) in [("Dragon", 4), ("Soldier", 1)] {
        let mut s = five_color_setup(&[]);
        s.game_mut().create_token(name, PlayerId(0), false);
        let token = permanent_named(&s, name);
        s.game_mut().state.permanents[token]
            .as_mut()
            .unwrap()
            .summoning_sick = false;
        s.advance_to_active_step(0, StepKind::DeclareAttackers);
        choose(
            &mut s,
            |a| matches!(a,Action::DeclareAttacker{permanent,attack:true,..} if *permanent==token),
        );
        pass_until(&mut s, ActionSpaceKind::Priority);
        settle(&mut s);
        assert_eq!(s.game().state.players[0].combat_mana_pool.total(), amount);
        s.advance_to_active_step(0, StepKind::PostcombatMain);
        assert_eq!(s.game().state.players[0].combat_mana_pool.total(), 0);
    }
}

#[test]
fn dragonfly_swarm_power_counts_noncreature_nonland_graveyard_cards() {
    let mut s = Scenario::new(
        deck(&[
            ("Island", 12),
            ("Mountain", 8),
            ("Dragonfly Swarm", 4),
            ("Pop Quiz", 8),
            ("Otter-Penguin", 8),
        ]),
        deck(&[("Plains", 40)]),
        332,
    );
    let swarm = s.force_permanent_on_battlefield(0, "Dragonfly Swarm");
    assert_eq!(s.game().effective_pt(swarm), (0, 3));

    plant_in_graveyard(&mut s, 0, "Pop Quiz");
    plant_in_graveyard(&mut s, 0, "Pop Quiz");
    assert_eq!(s.game().effective_pt(swarm), (2, 3));

    // Creature and land cards in the graveyard do not count.
    plant_in_graveyard(&mut s, 0, "Otter-Penguin");
    plant_in_graveyard(&mut s, 0, "Island");
    assert_eq!(s.game().effective_pt(swarm), (2, 3));
}

#[test]
fn dragonfly_swarm_death_trigger_needs_lesson_at_fire_time() {
    let mut s = Scenario::new(
        deck(&[
            ("Island", 12),
            ("Dragonfly Swarm", 4),
            ("It'll Quench Ya!", 12),
            ("Otter-Penguin", 12),
        ]),
        deck(&[("Mountain", 24), ("Lightning Bolt", 16)]),
        333,
    );
    // Without a Lesson in the graveyard: dying does not trigger.
    let swarm = s.force_permanent_on_battlefield(0, "Dragonfly Swarm");
    let swarm_card = permanent(&s, swarm).card;
    let hand_before = s.zone_size(0, ZoneType::Hand);
    s.game_mut().move_card(swarm_card, ZoneType::Graveyard);
    assert!(s.game().state.pending_triggers.is_empty());
    assert_eq!(s.zone_size(0, ZoneType::Hand), hand_before);

    // With a Lesson: dying draws a card.
    plant_in_graveyard(&mut s, 0, "It'll Quench Ya!");
    let swarm2 = s.force_permanent_on_battlefield(0, "Dragonfly Swarm");
    let swarm2_card = permanent(&s, swarm2).card;
    let hand_before = s.zone_size(0, ZoneType::Hand);
    s.game_mut().move_card(swarm2_card, ZoneType::Graveyard);
    resolve_everything(&mut s); // death trigger resolves
    assert_eq!(s.zone_size(0, ZoneType::Hand), hand_before + 1);
}

/// Earth Kingdom Jailer in hand with three Plains out; player 1 holds creatures and Bolts.
fn jailer_scenario(seed: u64) -> Scenario {
    let mut s = Scenario::new(
        deck(&[("Plains", 24), ("Earth Kingdom Jailer", 16)]),
        deck(&[
            ("Mountain", 16),
            ("Gray Ogre", 8),
            ("Raging Goblin", 8),
            ("Lightning Bolt", 8),
        ]),
        seed,
    );
    force_lands(&mut s, 0, "Plains", 3);
    clear_hand(&mut s, 0);
    clear_hand(&mut s, 1);
    s.force_card_in_hand(0, "Earth Kingdom Jailer");
    s
}

#[test]
fn jailer_exiles_and_returns_on_leaving() {
    let mut s = jailer_scenario(311);
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let ogre_card = permanent(&s, ogre).card;
    s.advance_to_active_step(0, StepKind::Main);
    cast_and_resolve(&mut s);

    // ETB: choose the Ogre (mv 3).
    assert_eq!(s.action_space().kind, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(ogre)));
    s.pass_priority();
    s.pass_priority(); // trigger resolves

    assert_eq!(
        s.game().state.zones.zone_of(ogre_card),
        Some(ZoneType::Exile)
    );
    assert_eq!(s.game().state.exile_links.len(), 1);

    // Kill the Jailer: the Ogre returns immediately, under its owner.
    let jailer = s.battlefield_permanents_named(0, "Earth Kingdom Jailer")[0];
    let jailer_card = permanent(&s, jailer).card;
    s.game_mut().move_card(jailer_card, ZoneType::Graveyard);

    assert_eq!(
        s.game().state.zones.zone_of(ogre_card),
        Some(ZoneType::Battlefield)
    );
    let returned = s.game().state.card_to_permanent[ogre_card].expect("re-entered");
    assert_eq!(permanent(&s, returned).controller, PlayerId(1));
    assert!(s.game().state.exile_links.is_empty());
}

#[test]
fn jailer_target_is_up_to_one_and_mv_gated() {
    let mut s = jailer_scenario(312);
    let _ogre = s.force_permanent_on_battlefield(1, "Gray Ogre"); // mv 3
    let _goblin = s.force_permanent_on_battlefield(1, "Raging Goblin"); // mv 1
    s.advance_to_active_step(0, StepKind::Main);
    cast_and_resolve(&mut s);

    let space = s.action_space().clone();
    assert_eq!(space.kind, ActionSpaceKind::ChooseTarget);
    // One legal target (the Ogre; the Goblin's mana value is too small)
    // plus Decline ("up to one").
    assert_eq!(space.actions.len(), 2);
    assert!(s.take_action_by_type(ActionType::DeclineChoice));
    s.pass_priority();
    s.pass_priority(); // trigger resolves with no target

    assert_eq!(s.game().state.zones.size(ZoneType::Exile, PlayerId(1)), 0);
    assert!(s.game().state.exile_links.is_empty());
}

#[test]
fn jailer_trigger_still_fires_with_no_legal_targets() {
    let mut s = jailer_scenario(313);
    let _goblin = s.force_permanent_on_battlefield(1, "Raging Goblin"); // mv 1 only
    s.advance_to_active_step(0, StepKind::Main);
    cast_and_resolve(&mut s);
    // "Up to one target" with zero legal targets: no choice surfaces, the
    // trigger resolves as a no-op, play continues.
    s.advance_until(
        |scenario| scenario.action_space().kind == ActionSpaceKind::Priority,
        "expected play to continue".to_string(),
    );
    assert_eq!(s.game().state.zones.size(ZoneType::Exile, PlayerId(1)), 0);
}

#[test]
fn jailer_killed_in_response_exiles_nothing() {
    let mut s = jailer_scenario(314);
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let ogre_card = permanent(&s, ogre).card;
    force_lands(&mut s, 1, "Mountain", 1);
    s.force_card_in_hand(1, "Lightning Bolt");
    s.advance_to_active_step(0, StepKind::Main);
    cast_and_resolve(&mut s);

    assert!(s.choose_target(Target::Permanent(ogre)));
    // Trigger is on the stack. In response, the opponent bolts the 3/3
    // Jailer.
    let jailer = s.battlefield_permanents_named(0, "Earth Kingdom Jailer")[0];
    s.pass_priority();
    cast_only(&mut s);
    assert!(s.choose_target(Target::Permanent(jailer)));
    s.pass_priority();
    s.pass_priority(); // bolt resolves, Jailer dies
    s.pass_priority();
    s.pass_priority(); // exile trigger resolves — source gone, no exile

    assert_eq!(
        s.game().state.zones.zone_of(ogre_card),
        Some(ZoneType::Battlefield),
        "nothing is exiled when the Jailer left first (CR 603.6e)"
    );
    assert!(s.game().state.exile_links.is_empty());
}

#[test]
fn jailer_old_trigger_does_not_follow_reentered_source() {
    let mut s = jailer_scenario(315);
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let ogre_card = permanent(&s, ogre).card;
    s.advance_to_active_step(0, StepKind::Main);
    cast_and_resolve(&mut s);

    assert!(s.choose_target(Target::Permanent(ogre)));
    let jailer = s.battlefield_permanents_named(0, "Earth Kingdom Jailer")[0];
    let jailer_card = permanent(&s, jailer).card;
    let old_ref = s
        .game()
        .current_object_ref(jailer_card)
        .expect("old Jailer ref");

    s.game_mut().move_card(jailer_card, ZoneType::Hand);
    s.game_mut().move_card(jailer_card, ZoneType::Battlefield);
    // Ignore the later incarnation's ETB trigger and resolve only the old
    // trigger already on the stack.
    s.game_mut().state.pending_triggers.clear();
    let new_ref = s
        .game()
        .current_object_ref(jailer_card)
        .expect("new Jailer ref");
    assert_eq!(old_ref.entity, new_ref.entity);
    assert_ne!(old_ref.incarnation, new_ref.incarnation);

    s.pass_priority();
    s.pass_priority();

    assert_eq!(
        s.game().state.zones.zone_of(ogre_card),
        Some(ZoneType::Battlefield),
        "the old trigger must not attach its duration to the new Jailer object"
    );
    assert!(s.game().state.exile_links.is_empty());
}

#[test]
fn protectors_sacrifice_is_committed_and_only_another_own_ally_is_legal() {
    let mut s = five_color_setup(&[
        "Earth Kingdom Protectors",
        "Avatar Enthusiasts",
        "Gray Ogre",
    ]);
    let protector = s.force_permanent_on_battlefield(0, "Earth Kingdom Protectors");
    let ally = s.force_permanent_on_battlefield(0, "Avatar Enthusiasts");
    let other = s.force_permanent_on_battlefield(1, "Avatar Enthusiasts");
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    assert!(keywords(&s, protector).vigilance);
    activate(&mut s, protector, 0);
    for illegal in [protector, other, ogre] {
        assert!(!s.action_space().actions.iter().any(
            |a| matches!(a,Action::ChooseTarget{target:Target::Permanent(p),..} if *p==illegal)
        ));
    }
    assert!(s.choose_target(Target::Permanent(ally)));
    settle(&mut s);
    assert!(s.game().state.permanents[protector].is_none());
    assert!(keywords(&s, ally).indestructible);
    s.advance_to_active_step(1, StepKind::Main);
    assert!(!keywords(&s, ally).indestructible);
}

#[test]
fn earth_kingdom_soldier_offers_zero_one_or_two_distinct_own_creatures() {
    for count in 0..=2 {
        let mut s = five_color_setup(&["Earth Kingdom Soldier", "Gray Ogre"]);
        let a = s.force_permanent_on_battlefield(0, "Gray Ogre");
        let b = s.force_permanent_on_battlefield(0, "Gray Ogre");
        let opponent = s.force_permanent_on_battlefield(1, "Gray Ogre");
        cast(&mut s, "Earth Kingdom Soldier");
        pass_until(&mut s, ActionSpaceKind::ChooseTarget);
        assert!(!s.action_space().actions.iter().any(
            |act| matches!(act,Action::ChooseTarget{target:Target::Permanent(p),..} if *p==opponent)
        ));
        if count >= 1 {
            assert!(s.choose_target(Target::Permanent(a)));
            assert!(!s.action_space().actions.iter().any(
                |act| matches!(act,Action::ChooseTarget{target:Target::Permanent(p),..} if *p==a)
            ));
        }
        if count == 2 {
            assert!(s.choose_target(Target::Permanent(b)));
        } else {
            choose(&mut s, |act| matches!(act, Action::Decline { .. }));
        }
        settle(&mut s);
        assert_eq!(counters(&s, a), i32::from(count >= 1));
        assert_eq!(counters(&s, b), i32::from(count == 2));
        assert_eq!(counters(&s, opponent), 0);
        assert!(keywords(&s, permanent_named(&s, "Earth Kingdom Soldier")).vigilance);
    }
}

#[test]
fn lieutenant_counters_each_other_ally_and_grows() {
    let mut s = Scenario::new(
        deck(&[
            ("Plains", 8),
            ("Forest", 8),
            ("Earth King's Lieutenant", 8),
            ("Kyoshi Warriors", 8),
            ("Otter-Penguin", 4),
            ("Invasion Reinforcements", 4),
        ]),
        deck(&[("Plains", 40)]),
        361,
    );
    let kyoshi = s.force_permanent_on_battlefield(0, "Kyoshi Warriors");
    let penguin = s.force_permanent_on_battlefield(0, "Otter-Penguin");
    force_lands(&mut s, 0, "Plains", 2);
    force_lands(&mut s, 0, "Forest", 2);
    clear_hand(&mut s, 0);
    s.force_card_in_hand(0, "Earth King's Lieutenant");
    s.advance_to_active_step(0, StepKind::Main);
    cast_and_resolve(&mut s);
    // Force the next castable into hand *before* the trigger resolves so
    // the recomputed action space can see it.
    force_new_card_in_hand(&mut s, 0, "Invasion Reinforcements");
    s.pass_priority();
    s.pass_priority(); // ETB trigger resolves

    let lieutenant = s.battlefield_permanents_named(0, "Earth King's Lieutenant")[0];
    assert_eq!(
        permanent(&s, kyoshi).plus1_counters,
        1,
        "other Ally gets a counter"
    );
    assert_eq!(
        permanent(&s, penguin).plus1_counters,
        0,
        "non-Ally does not"
    );
    assert_eq!(permanent(&s, lieutenant).plus1_counters, 0, "not itself");

    // Another Ally enters: the Lieutenant grows — twice, because the
    // Reinforcements' Ally token is itself another entering Ally.
    cast_and_resolve(&mut s);
    resolve_everything(&mut s); // Lieutenant + Reinforcements triggers
    assert_eq!(permanent(&s, lieutenant).plus1_counters, 2);
}

#[test]
fn avatar_state_grants_a_visible_incarnation_local_subtype_and_cleans_up() {
    let mut s = five_color_setup(&["Enter the Avatar State", "Gnarled Professor"]);
    let target = s.force_permanent_on_battlefield(0, "Gnarled Professor");
    settle(&mut s);
    cast(&mut s, "Enter the Avatar State");
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert_choice_contract(&s);
    s.choose_target(Target::Permanent(target));
    use managym::search_state::{witness, BenchCommand, BranchDriver, ClonePlusUndoDriver};
    let driver = ClonePlusUndoDriver::default();
    let mut branch = driver.fork_exact(s.game());
    let root = witness(&branch);
    let mark = driver.mark(&mut branch);
    for _ in 0..8 {
        if !branch.state.permanents[target]
            .as_ref()
            .unwrap()
            .temp_subtypes
            .is_empty()
        {
            break;
        }
        let index = branch
            .action_space()
            .unwrap()
            .actions
            .iter()
            .position(|action| matches!(action, Action::PassPriority { .. }))
            .unwrap();
        driver
            .apply(
                &mut branch,
                BenchCommand {
                    action_index: index,
                    expected_action_hash: None,
                    expected_state_hash: None,
                },
            )
            .unwrap();
    }
    assert_eq!(
        branch.state.permanents[target]
            .as_ref()
            .unwrap()
            .temp_subtypes,
        vec!["Avatar"]
    );
    driver.rollback(&mut branch, mark);
    assert_eq!(witness(&branch), root);
    settle(&mut s);
    let permanent = s.game().state.permanents[target].as_ref().unwrap();
    assert_eq!(permanent.temp_subtypes, vec!["Avatar"]);
    let keywords = permanent.effective_keywords(&s.game().state.cards[permanent.card]);
    assert!(keywords.flying && keywords.first_strike && keywords.lifelink && keywords.hexproof);
    let obs = Observation::for_player(s.game(), PlayerId(0));
    assert_eq!(
        obs.agent_permanents
            .iter()
            .find(|p| p.id == permanent.id.0 as i32)
            .unwrap()
            .temp_subtypes,
        vec!["Avatar"]
    );
    let mut branch = s.game().clone();
    let card_id = branch.state.permanents[target].as_ref().unwrap().card;
    branch.move_card(card_id, ZoneType::Hand);
    branch.move_card(card_id, ZoneType::Battlefield);
    let reentered = branch.state.card_to_permanent[card_id].unwrap();
    assert!(branch.state.permanents[reentered]
        .as_ref()
        .unwrap()
        .temp_subtypes
        .is_empty());
    assert_eq!(
        s.game().state.permanents[target]
            .as_ref()
            .unwrap()
            .temp_subtypes,
        vec!["Avatar"]
    );
    s.advance_to_active_step(1, StepKind::Main);
    assert!(s.game().state.permanents[target]
        .as_ref()
        .unwrap()
        .temp_subtypes
        .is_empty());
}

#[test]
fn avatar_state_grants_keywords_and_hexproof_blocks_opponent_targeting() {
    let mut s = Scenario::new(
        deck(&[
            ("Plains", 16),
            ("Enter the Avatar State", 12),
            ("Kyoshi Warriors", 12),
        ]),
        deck(&[("Mountain", 24), ("Lightning Bolt", 16)]),
        373,
    );
    let kyoshi = s.force_permanent_on_battlefield(0, "Kyoshi Warriors");
    force_lands(&mut s, 0, "Plains", 1);
    force_lands(&mut s, 1, "Mountain", 1);
    clear_hand(&mut s, 0);
    clear_hand(&mut s, 1);
    s.force_card_in_hand(0, "Enter the Avatar State");
    s.force_card_in_hand(1, "Lightning Bolt");
    s.advance_to_active_step(0, StepKind::Main);

    cast_only(&mut s);
    assert!(s.choose_target(Target::Permanent(kyoshi)));
    s.pass_priority();
    s.pass_priority();

    let keywords = permanent(&s, kyoshi).temp_keywords;
    assert!(keywords.flying && keywords.first_strike && keywords.lifelink && keywords.hexproof);

    // All four grants are agent-visible on the permanent entry.
    let obs = Observation::new(s.game(), &[]);
    let kyoshi_object_id = permanent(&s, kyoshi).id.0 as i32;
    let kyoshi_perm = obs
        .agent_permanents
        .iter()
        .chain(obs.opponent_permanents.iter())
        .find(|perm| perm.id == kyoshi_object_id)
        .expect("Kyoshi Warriors permanent should be observable");
    assert!(
        kyoshi_perm.keywords.flying
            && kyoshi_perm.keywords.first_strike
            && kyoshi_perm.keywords.lifelink
            && kyoshi_perm.keywords.hexproof,
        "granted keywords (incl. hexproof) must be encoded"
    );

    // The opponent's Bolt cannot target the hexproof creature: the only
    // remaining legal target is a player, so targeting it is impossible.
    s.pass_priority();
    cast_only(&mut s);
    assert!(
        !s.choose_target(Target::Permanent(kyoshi)),
        "hexproof forbids opponent targeting"
    );
    // Bolt the controller instead to unwind the pending choice.
    assert!(s.choose_target(Target::Player(PlayerId(0))));
    s.pass_priority();
    s.pass_priority();
    s.assert_life(0, 17);

    // Wears off at cleanup.
    s.advance_to_active_step(1, StepKind::Upkeep);
    assert!(!permanent(&s, kyoshi).temp_keywords.hexproof);
}

#[test]
fn fancy_footwork_untaps_and_buffs_one_or_two_targets() {
    let mut s = Scenario::new(
        deck(&[
            ("Plains", 16),
            ("Fancy Footwork", 12),
            ("Kyoshi Warriors", 12),
        ]),
        deck(&[("Plains", 40)]),
        372,
    );
    let w1 = s.force_permanent_on_battlefield(0, "Kyoshi Warriors");
    let w2 = s.force_permanent_on_battlefield(0, "Kyoshi Warriors");
    s.game_mut().state.permanents[w1]
        .as_mut()
        .expect("w1")
        .tapped = true;
    s.game_mut().state.permanents[w2]
        .as_mut()
        .expect("w2")
        .tapped = true;
    force_lands(&mut s, 0, "Plains", 3);
    clear_hand(&mut s, 0);
    s.force_card_in_hand(0, "Fancy Footwork");
    s.advance_to_active_step(0, StepKind::Main);

    cast_only(&mut s);
    assert!(s.choose_target(Target::Permanent(w1)));
    // Min 1 met; the second target is optional but we take it.
    assert!(s.choose_target(Target::Permanent(w2)));
    s.pass_priority();
    s.pass_priority();

    assert!(!permanent(&s, w1).tapped);
    assert!(!permanent(&s, w2).tapped);
    assert_eq!(s.game().effective_pt(w1), (5, 5));
    assert_eq!(s.game().effective_pt(w2), (5, 5));
}

#[test]
fn fancy_footwork_one_target_or_one_remaining_target_expires() {
    for two in [false, true] {
        let mut s = five_color_setup(&["Fancy Footwork", "Gray Ogre"]);
        let own = s.force_permanent_on_battlefield(0, "Gray Ogre");
        let other = s.force_permanent_on_battlefield(1, "Gray Ogre");
        s.game_mut().state.permanents[other]
            .as_mut()
            .unwrap()
            .tapped = true;
        cast(&mut s, "Fancy Footwork");
        assert!(!s
            .action_space()
            .actions
            .iter()
            .any(|a| matches!(a, Action::Decline { .. })));
        assert!(s.choose_target(Target::Permanent(other)));
        assert!(!s.choose_target(Target::Permanent(other)));
        if two {
            assert!(s.choose_target(Target::Permanent(own)));
            let id = s.game().state.permanents[own].as_ref().unwrap().card;
            s.game_mut().move_card(id, ZoneType::Hand);
        } else {
            choose(&mut s, |a| matches!(a, Action::Decline { .. }));
        }
        settle(&mut s);
        assert!(!tapped(&s, other));
        assert_eq!(s.game().effective_pt(other), (4, 4));
        s.advance_to_active_step(1, StepKind::Main);
        assert_eq!(s.game().effective_pt(other), (2, 2));
    }
}

/// Fire Nation Cadets ready to attack, optionally with a Lesson in the graveyard
/// so its firebending is granted.
fn cadets_scenario(seed: u64, with_lesson: bool) -> (Scenario, PermanentId) {
    let mut s = Scenario::new(
        deck(&[
            ("Mountain", 12),
            ("Fire Nation Cadets", 8),
            ("Firebending Lesson", 12),
            ("It'll Quench Ya!", 8),
        ]),
        deck(&[("Plains", 24), ("Wall of Stone", 16)]),
        seed,
    );
    let cadets = s.force_permanent_on_battlefield(0, "Fire Nation Cadets");
    clear_summoning_sickness(&mut s, cadets);
    clear_hand(&mut s, 0);
    if with_lesson {
        plant_in_graveyard(&mut s, 0, "It'll Quench Ya!");
    }
    (s, cadets)
}

#[test]
fn firebending_only_triggers_with_lesson_in_graveyard() {
    let (mut s, cadets) = cadets_scenario(341, false);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, cadets);
    // No Lesson: the granted ability does not exist, no trigger fires.
    assert!(s.game().state.pending_triggers.is_empty());
    assert_eq!(s.game().state.players[0].combat_mana_pool.total(), 0);
}

#[test]
fn firebending_mana_lasts_until_end_of_combat_and_casts_spells() {
    let (mut s, cadets) = cadets_scenario(342, true);
    s.force_card_in_hand(0, "Firebending Lesson");
    let wall = s.force_permanent_on_battlefield(1, "Wall of Stone");
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    attack_with(&mut s, cadets);
    s.pass_priority();
    s.pass_priority(); // firebending trigger resolves

    assert_eq!(
        s.game().state.players[0].combat_mana_pool.total(),
        2,
        "firebending 2 added {{R}}{{R}}"
    );

    // The mana survives into the declare-blockers step (a normal pool
    // would have emptied at the step boundary)...
    s.advance_to_step(StepKind::DeclareBlockers);
    while s.action_space().kind == ActionSpaceKind::DeclareBlocker {
        s.decline_block();
    }
    assert_eq!(s.game().state.players[0].combat_mana_pool.total(), 2);

    // ...and casts an instant with no lands at all (Firebending Lesson,
    // {R}, unkicked — kicker {4} is unaffordable).
    cast_only(&mut s);
    assert!(s.choose_target(Target::Permanent(wall)));
    s.pass_priority();
    s.pass_priority();
    assert_eq!(permanent(&s, wall).damage, 2);
    assert_eq!(
        s.game().state.players[0].combat_mana_pool.total(),
        1,
        "one {{R}} spent from the combat pool"
    );

    // Whatever's left empties as combat ends.
    s.advance_to_active_step(0, StepKind::PostcombatMain);
    assert_eq!(s.game().state.players[0].combat_mana_pool.total(), 0);
}

#[test]
fn cadets_pump_ability_buffs_until_end_of_turn() {
    let (mut s, cadets) = cadets_scenario(343, false);
    force_lands(&mut s, 0, "Mountain", 2);
    s.advance_to_active_step(0, StepKind::Main);
    assert!(s.take_action_by_type(ActionType::PriorityActivateAbility));
    s.pass_priority();
    s.pass_priority(); // ability resolves
    assert_eq!(s.game().effective_pt(cadets), (2, 2));
    s.advance_to_active_step(1, StepKind::Upkeep);
    assert_eq!(s.game().effective_pt(cadets), (1, 2));
}

#[test]
fn cadets_keep_announced_firebending_after_last_lesson_leaves() {
    let mut s = five_color_setup(&["Fire Nation Cadets", "Firebending Lesson"]);
    let cadets = s.force_permanent_on_battlefield(0, "Fire Nation Cadets");
    let lesson = library_card(&s, 0, "Firebending Lesson");
    s.game_mut().move_card(lesson, ZoneType::Graveyard);
    s.game_mut().state.permanents[cadets]
        .as_mut()
        .unwrap()
        .summoning_sick = false;
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    choose(
        &mut s,
        |a| matches!(a, Action::DeclareAttacker { permanent, attack: true, .. } if *permanent == cadets),
    );
    pass_until(&mut s, ActionSpaceKind::Priority);
    s.game_mut().move_card(lesson, ZoneType::Exile);
    settle(&mut s);
    // The ability existed when attacking. Losing it cannot remove its trigger.
    assert_eq!(s.game().state.players[0].combat_mana_pool.total(), 2);
}

#[test]
fn fire_sages_counter_cost_and_firebending_mana_lifetime() {
    let mut s = five_color_setup(&["Fire Sages"]);
    let sages = s.force_permanent_on_battlefield(0, "Fire Sages");
    s.set_player_mana_pool(0, "1RR");
    activate(&mut s, sages, 0);
    settle(&mut s);
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
    assert_eq!(counters(&s, sages), 1);
    s.game_mut().state.permanents[sages]
        .as_mut()
        .unwrap()
        .summoning_sick = false;
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    choose(
        &mut s,
        |a| matches!(a,Action::DeclareAttacker { permanent,attack:true,.. } if *permanent==sages),
    );
    pass_until(&mut s, ActionSpaceKind::Priority);
    settle(&mut s);
    assert_eq!(s.game().state.players[0].combat_mana_pool.total(), 1);
    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    assert_eq!(s.game().state.players[0].combat_mana_pool.total(), 1);
    s.advance_to_active_step(0, StepKind::PostcombatMain);
    assert_eq!(s.game().state.players[0].combat_mana_pool.total(), 0);
}

/// Firebending Lesson in hand at player 0's main phase, with a Wall of Stone to target.
fn firebending_setup(seed: u64, mountains: usize) -> (Scenario, PermanentId) {
    let mut s = Scenario::new(
        deck(&[("Mountain", 24), ("Firebending Lesson", 16)]),
        deck(&[("Plains", 24), ("Wall of Stone", 16)]),
        seed,
    );
    let wall = s.force_permanent_on_battlefield(1, "Wall of Stone");
    force_lands(&mut s, 0, "Mountain", mountains);
    clear_hand(&mut s, 0);
    s.force_card_in_hand(0, "Firebending Lesson");
    s.advance_to_active_step(0, StepKind::Main);
    (s, wall)
}

#[test]
fn treasure_is_available_when_announcing_a_kicker_cost() {
    let mut s = five_color_setup(&["Firebending Lesson", "Gnarled Professor"]);
    let victim = s.force_permanent_on_battlefield(1, "Gnarled Professor");
    s.game_mut().create_token("Treasure", PlayerId(0), false);
    s.set_player_mana_pool(0, "3R");
    cast(&mut s, "Firebending Lesson");
    assert_eq!(s.action_space().kind, ActionSpaceKind::PayOrNot);
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    choose(
        &mut s,
        |a| matches!(a,Action::ChooseTarget {target:Target::Permanent(id),..} if *id==victim),
    );
    let treasure = permanent_named(&s, "Treasure");
    activate_first(&mut s, treasure);
    choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 4, .. }));
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    settle(&mut s);
    assert!(s
        .battlefield_permanents_named(1, "Gnarled Professor")
        .is_empty());
}

#[test]
fn firebending_lesson_kicked_deals_five() {
    let (mut s, wall) = firebending_setup(31, 5);
    cast_only(&mut s);
    let space = s.action_space();
    assert_eq!(space.kind, ActionSpaceKind::PayOrNot);
    assert_eq!(space.player, Some(PlayerId(0)));
    assert!(s.take_action_by_type(ActionType::PayCost));
    assert_eq!(s.action_space().kind, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(wall)));
    // All five mountains paid for {R} + kicker {4}.
    assert_eq!(s.untapped_permanents_named(0, "Mountain"), 0);
    s.pass_priority();
    s.pass_priority();
    assert_eq!(permanent(&s, wall).damage, 5);
}

#[test]
fn firebending_lesson_unkicked_deals_two() {
    let (mut s, wall) = firebending_setup(32, 5);
    cast_only(&mut s);
    assert_eq!(s.action_space().kind, ActionSpaceKind::PayOrNot);
    assert!(s.take_action_by_type(ActionType::DeclineChoice));
    assert!(s.choose_target(Target::Permanent(wall)));
    s.pass_priority();
    s.pass_priority();
    assert_eq!(permanent(&s, wall).damage, 2);
}

/// With only enough mana for the base cost, the kicker choice is not
/// offered - casting goes straight to targeting.
#[test]
fn firebending_lesson_kicker_not_offered_when_unaffordable() {
    let (mut s, wall) = firebending_setup(33, 2);
    cast_only(&mut s);
    assert_eq!(s.action_space().kind, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(wall)));
    s.pass_priority();
    s.pass_priority();
    assert_eq!(permanent(&s, wall).damage, 2);
}

#[test]
fn first_time_flyer_conditional_static_recomputes_on_read() {
    let mut s = Scenario::new(
        deck(&[
            ("Island", 16),
            ("First-Time Flyer", 12),
            ("It'll Quench Ya!", 12),
        ]),
        deck(&[("Plains", 40)]),
        323,
    );
    let flyer = s.force_permanent_on_battlefield(0, "First-Time Flyer");
    assert_eq!(s.game().effective_pt(flyer), (1, 2));

    // A Lesson hits the graveyard: the buff switches on...
    let lesson = plant_in_graveyard(&mut s, 0, "It'll Quench Ya!");
    assert_eq!(s.game().effective_pt(flyer), (2, 3));

    // ...and off again when it leaves (recomputed on read, no state).
    s.game_mut()
        .state
        .zones
        .move_card(lesson, PlayerId(0), ZoneType::Library);
    assert_eq!(s.game().effective_pt(flyer), (1, 2));
}

#[test]
fn fortune_teller_clue_sacrifices_to_draw_and_ceases_to_exist() {
    let mut s = Scenario::new(
        deck(&[("Island", 24), ("Forecasting Fortune Teller", 16)]),
        deck(&[("Island", 40)]),
        31,
    );
    force_lands(&mut s, 0, "Island", 4);
    s.force_card_in_hand(0, "Forecasting Fortune Teller");
    s.advance_to_active_step(0, StepKind::Main);
    cast_and_resolve(&mut s);
    resolve_until_priority(&mut s);

    let clues = s.battlefield_permanents_named(0, "Clue");
    assert_eq!(clues.len(), 1, "Clue token should be created");
    let clue_card = permanent(&s, clues[0]).card;
    assert!(s.game().state.cards[clue_card].is_token);
    assert!(s.game().state.cards[clue_card].types.is_artifact());

    // "{2}, Sacrifice this token: Draw a card."
    let hand_before = s.zone_size(0, ZoneType::Hand);
    s.assert_action_available(ActionType::PriorityActivateAbility);
    assert!(s.take_action_by_type(ActionType::PriorityActivateAbility));
    // The sacrifice is paid on activation: the Clue is already gone.
    assert!(s.battlefield_permanents_named(0, "Clue").is_empty());
    s.pass_priority();
    s.pass_priority();
    assert_eq!(s.zone_size(0, ZoneType::Hand), hand_before + 1);

    // CR 704.5d — the token ceased to exist instead of staying in a zone.
    assert_eq!(s.game().state.zones.zone_of(clue_card), None);
    assert_eq!(s.zone_size(0, ZoneType::Graveyard), 0);
}

#[test]
fn gather_creates_per_plains_and_scry_can_reverse_top_cards_privately() {
    let mut s = five_color_setup(&["Gather the White Lotus"]);
    s.force_permanent_on_battlefield(0, "Plains");
    s.force_permanent_on_battlefield(0, "Plains");
    let cards = pin_top(&mut s, &["Island", "Forest"]);
    cast(&mut s, "Gather the White Lotus");
    pass_until(&mut s, ActionSpaceKind::Scry);
    assert_eq!(s.battlefield_permanents_named(0, "Ally").len(), 2);
    assert_eq!(s.action_space().actions.len(), 4);
    let other = Observation::for_player(s.game(), PlayerId(1));
    assert!(other
        .opponent_cards
        .iter()
        .all(|c| c.zone != ZoneType::Library));
    for card in cards.iter().rev() {
        let index = s
            .action_space()
            .actions
            .iter()
            .position(
                |a| matches!(a, Action::ScryCard { card: c, to_bottom: false, .. } if c == card),
            )
            .unwrap();
        s.step_action(index);
    }
    let library = s
        .game()
        .state
        .zones
        .zone_cards(ZoneType::Library, PlayerId(0));
    assert_eq!(library[library.len() - 1], cards[1]);
    assert_eq!(library[library.len() - 2], cards[0]);
}

#[test]
fn scry_possible_world_materialization_preserves_inspected_cards_and_order_choices() {
    use managym::possible_worlds::{MaterializeMode, PossibleWorldSpace};
    let mut s = five_color_setup(&["Gather the White Lotus"]);
    cast(&mut s, "Gather the White Lotus");
    pass_until(&mut s, ActionSpaceKind::Scry);
    for partial in [false, true] {
        if partial {
            s.step_action(0);
        }
        let view = Observation::for_player(s.game(), PlayerId(0)).to_json();
        let space = PossibleWorldSpace::for_viewer(s.game(), PlayerId(0));
        for seed in 0..5 {
            let branch = space
                .materialize_index(s.game(), 0, seed, MaterializeMode::PreserveViewerRoot)
                .unwrap();
            assert_eq!(
                Observation::for_player(&branch, PlayerId(0)).to_json(),
                view
            );
            assert_eq!(branch.action_space(), s.game().action_space());
        }
    }
}

/// Glider Kids resolved with three Plains out, stopped at its enter trigger's scry 1.
fn glider_kids_scry_setup(seed: u64) -> Scenario {
    let mut s = Scenario::new(
        deck(&[("Plains", 24), ("Glider Kids", 16)]),
        deck(&[("Island", 40)]),
        seed,
    );
    force_lands(&mut s, 0, "Plains", 3);
    clear_hand(&mut s, 0);
    s.force_card_in_hand(0, "Glider Kids");
    s.advance_to_active_step(0, StepKind::Main);
    cast_and_resolve(&mut s); // creature resolves, ETB trigger on stack
    s.pass_priority();
    s.pass_priority(); // trigger resolves -> scry decision
    s
}

#[test]
fn glider_kids_scry_surfaces_binary_decision_to_controller() {
    let s = glider_kids_scry_setup(21);
    let space = s.action_space();
    assert_eq!(space.kind, ActionSpaceKind::Scry);
    assert_eq!(space.player, Some(PlayerId(0)));
    assert_eq!(space.actions.len(), 2);
    assert_eq!(space.actions[0].action_type(), ActionType::ScryKeep);
    assert_eq!(space.actions[1].action_type(), ActionType::ScryBottom);
}

#[test]
fn glider_kids_scry_bottom_moves_top_card_to_bottom() {
    let mut s = glider_kids_scry_setup(22);
    let top = library_top(&s, 0);
    assert!(s.take_action_by_type(ActionType::ScryBottom));
    let lib = library(&s, 0);
    assert_eq!(lib[0], top, "scried card should be on the bottom");
    assert_ne!(library_top(&s, 0), top);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
}

#[test]
fn glider_kids_scry_keep_leaves_top_card() {
    let mut s = glider_kids_scry_setup(23);
    let top = library_top(&s, 0);
    assert!(s.take_action_by_type(ActionType::ScryKeep));
    assert_eq!(library_top(&s, 0), top);
}

/// Glider Kids has flying (oracle: {2}{W} 2/3 "Flying / When this creature
/// enters, scry 1.") — a ground creature can't block it.
#[test]
fn glider_kids_flies_over_ground_blockers() {
    let mut s = Scenario::new(
        deck(&[("Plains", 24), ("Glider Kids", 16)]),
        deck(&[("Forest", 24), ("Llanowar Elves", 16)]),
        25,
    );
    let kids = s.force_permanent_on_battlefield(0, "Glider Kids");
    let _elf = s.force_permanent_on_battlefield(1, "Llanowar Elves");
    s.game_mut().state.permanents[kids]
        .as_mut()
        .expect("permanent should exist")
        .summoning_sick = false;

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::DeclareBlockers);

    assert!(
        !s.action_space().actions.iter().any(|action| {
            matches!(
                action,
                Action::DeclareBlocker {
                    attacker: Some(_),
                    ..
                }
            )
        }),
        "ground creature should have no legal block against the flying Glider Kids"
    );
}

/// The deciding agent sees the revealed card in its observation (zone
/// Library) and the scry actions focus it.
#[test]
fn scry_revealed_card_is_observation_encodable() {
    let mut s = glider_kids_scry_setup(24);
    let top = library_top(&s, 0);
    let top_object_id = s.game().state.cards[top].id.0 as i32;

    let events = s.drain_events();
    let obs = Observation::new(s.game(), &events);
    assert!(obs.validate());
    let revealed: Vec<_> = obs
        .agent_cards
        .iter()
        .filter(|card| card.zone == ZoneType::Library)
        .collect();
    assert_eq!(revealed.len(), 1, "exactly the revealed card is visible");
    assert_eq!(revealed[0].id, top_object_id);
    // Both scry actions focus the revealed card.
    for action in &obs.action_space.actions {
        assert_eq!(action.focus, vec![top_object_id]);
    }
}

/// Determinize during a scry pins the revealed card back on top — it is
/// known information for the deciding player.
#[test]
fn determinize_pins_scry_revealed_cards_on_top() {
    let mut s = glider_kids_scry_setup(25);
    let top = library_top(&s, 0);
    s.game_mut().determinize(PlayerId(0), 12345);
    assert_eq!(library_top(&s, 0), top);
    // The decision is still answerable.
    assert!(s.take_action_by_type(ActionType::ScryBottom));
    assert_eq!(library(&s, 0)[0], top);
}

#[test]
fn gran_gran_taps_to_loot_mandatorily_and_can_discard_the_drawn_card() {
    let mut s = ur_lessons_setup();
    let gran = put_onto_battlefield(&mut s, 0, "Gran-Gran");
    s.game_mut().state.permanents[gran]
        .as_mut()
        .unwrap()
        .summoning_sick = false;
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    let top = *s
        .game()
        .state
        .zones
        .zone_cards(ZoneType::Library, PlayerId(0))
        .last()
        .unwrap();
    let before = s.zone_size(0, ZoneType::Hand);
    let index = s.action_space().actions.iter().position(|a| matches!(a, Action::DeclareAttacker { permanent, attack: true, .. } if *permanent == gran)).unwrap();
    s.step_action(index);
    resolve_top(&mut s);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Discard);
    assert_eq!(s.zone_size(0, ZoneType::Hand), before + 1);
    assert!(s
        .action_space()
        .actions
        .iter()
        .all(|a| matches!(a, Action::SelectCard { .. })));
    let index = s
        .action_space()
        .actions
        .iter()
        .position(|a| matches!(a, Action::SelectCard { card, .. } if *card == top))
        .unwrap();
    s.step_action(index);
    assert_eq!(s.game().state.zones.zone_of(top), Some(ZoneType::Graveyard));
    assert_eq!(s.zone_size(0, ZoneType::Hand), before);
}

#[test]
fn gran_gran_reduction_threshold_noncreature_and_generic_floor() {
    let mut s = ur_lessons_setup();
    let gran = put_onto_battlefield(&mut s, 0, "Gran-Gran");
    let proft = find_owned_card(&s, 0, "Proft's Eidetic Memory");
    let tiger = find_owned_card(&s, 0, "Otter-Penguin");
    for card in [proft, tiger] {
        s.game_mut().move_card(card, ZoneType::Hand);
    }
    put_onto_battlefield(&mut s, 0, "Island");
    plant_lessons(&mut s, 0, 2);
    s.game_mut().scenario_refresh_priority().unwrap();
    let offered = |s: &Scenario, card| {
        s.action_space()
            .actions
            .iter()
            .any(|a| matches!(a, Action::CastSpell { card: c, .. } if *c == card))
    };
    assert!(!offered(&s, proft));
    plant_lessons(&mut s, 0, 1);
    let wisdom = find_owned_card(&s, 0, "Accumulate Wisdom");
    s.game_mut().move_card(wisdom, ZoneType::Hand);
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(offered(&s, proft));
    assert!(!offered(&s, tiger)); // {1}{U} creature is never reduced.
    assert!(offered(&s, wisdom)); // {U} is still {U}.
    let gran_card = card_of(&s, gran);
    s.game_mut().move_card(gran_card, ZoneType::Hand);
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!offered(&s, proft));
}

#[test]
fn legend_choice_and_lethal_damage_share_one_sba_batch() {
    let mut s = ur_lessons_setup();
    let first = put_onto_battlefield(&mut s, 0, "Gran-Gran");
    let doomed = put_onto_battlefield(&mut s, 0, "Gran-Gran");
    let first_card = card_of(&s, first);
    let doomed_card = card_of(&s, doomed);
    s.game_mut().state.permanents[doomed]
        .as_mut()
        .unwrap()
        .damage = 2;
    s.pass_priority();
    assert_eq!(s.action_space().kind, ActionSpaceKind::LegendRule);
    // Keeping the lethally damaged legend does not rescue the other one.
    let i = s
        .action_space()
        .actions
        .iter()
        .position(|a| matches!(a, Action::SelectCard { card, .. } if *card == doomed_card))
        .unwrap();
    s.step_action(i);
    for card in [first_card, doomed_card] {
        assert_eq!(
            s.game().state.zones.zone_of(card),
            Some(ZoneType::Graveyard)
        );
    }
}

#[test]
fn guide_grants_tap_mana_with_sickness_and_removes_it_on_departure() {
    let mut s = five_color_setup(&["Great Divide Guide", "Hermitic Herbalist"]);
    let guide = enter_and_settle(&mut s, "Great Divide Guide");
    s.set_player_mana_pool(0, "0");
    s.game_mut().scenario_refresh_priority().unwrap();
    let source = permanent_named(&s, "Great Divide Guide");
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == source)));
    ready(&mut s, "Great Divide Guide");
    activate_named(&mut s, "Great Divide Guide", 0);
    choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 4, .. }));
    assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("G"));
    let forest = enter_and_settle(&mut s, "Forest");
    let land = s.game().state.card_to_permanent[forest].unwrap();
    assert!(s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == land)));
    s.game_mut().move_card(guide, ZoneType::Graveyard);
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == land)));
}

/// Sozin's Comet grants firebending 5 to the creatures you control as it
/// resolves: one attacking adds five red mana that lasts until end of combat.
/// A creature arriving later gains nothing (CR 611.2c).
#[test]
fn sozins_comet_grants_firebending_five_until_end_of_turn() {
    let mut s = five_color_setup(&["Sozin's Comet", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    clear_summoning_sickness(&mut s, ogre);
    cast(&mut s, "Sozin's Comet");
    settle(&mut s);
    let late = s.force_permanent_on_battlefield(0, "Gray Ogre");
    clear_summoning_sickness(&mut s, late);
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
    let mut s = five_color_setup(&["Sozin's Comet"]);
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
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_cast(&s, comet));
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    s.set_player_mana_pool(0, "2R");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_cast(&s, comet));
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
    let mut s = five_color_setup(&["Sozin's Comet"]);
    let comet = library_card(&s, 1, "Sozin's Comet");
    s.game_mut().move_card(comet, ZoneType::Hand);
    s.set_player_mana_pool(1, "20");
    s.pass_priority();
    assert_eq!(s.action_space().player, Some(PlayerId(1)));
    assert!(!foretell_offered(&mut s, comet));
}

/// With two foretold cards, determinization fills each face-down slot with a
/// distinct foretell-bearing card: one slot never takes the other's card, so
/// every card stays in exactly one zone and each foretold card keeps one grant.
#[test]
fn determinize_fills_two_foretold_slots_with_distinct_cards() {
    let mut s = five_color_setup(&["Sozin's Comet"]);
    let a = library_card(&s, 0, "Sozin's Comet");
    s.game_mut().move_card(a, ZoneType::Hand);
    let b = library_card(&s, 0, "Sozin's Comet");
    s.game_mut().move_card(b, ZoneType::Hand);
    let comets = [a, b];
    s.set_player_mana_pool(0, "4");
    for comet in comets {
        assert!(foretell_offered(&mut s, comet));
        choose(
            &mut s,
            |a| matches!(a, Action::Foretell { card, .. } if *card == comet),
        );
    }
    let zones = [
        ZoneType::Library,
        ZoneType::Hand,
        ZoneType::Battlefield,
        ZoneType::Graveyard,
        ZoneType::Stack,
        ZoneType::Exile,
        ZoneType::Command,
    ];
    let mut resampled = false;
    for seed in 0..256 {
        let mut game = s.game().clone();
        game.determinize(PlayerId(1), seed);
        let mut seen = BTreeMap::new();
        for player in [PlayerId(0), PlayerId(1)] {
            for zone in zones {
                for &card in game.state.zones.zone_cards(zone, player) {
                    assert_eq!(game.state.zones.zone_of(card), Some(zone));
                    *seen.entry(card.0).or_insert(0) += 1;
                }
            }
        }
        for index in 0..game.state.cards.len() {
            assert_eq!(seen.get(&index), Some(&1), "card {index} in one zone");
        }
        let foretold = game
            .state
            .zones
            .zone_cards(ZoneType::Exile, PlayerId(0))
            .clone();
        assert_eq!(foretold.len(), 2);
        let grants = &game.state.cast_permissions.exile_grants;
        assert_eq!(grants.iter().filter(|grant| grant.foretold).count(), 2);
        for card in &foretold {
            assert_eq!(game.state.cards[*card].name, "Sozin's Comet");
            let object = game.current_object_ref(*card);
            assert_eq!(
                grants
                    .iter()
                    .filter(|grant| Some(grant.object) == object)
                    .count(),
                1
            );
        }
        resampled |= foretold.iter().any(|card| !comets.contains(card));
    }
    assert!(resampled);
}

/// The possible-world space counts an opponent's foretold card among the
/// unseen cards: worlds fix only the hand, each weighted by the ways to fill
/// the face-down slot with a Sozin's Comet the hand leaves. Which copy was
/// foretold changes neither the space nor its weights, and materialized
/// branches refill the slot rather than keeping the real card.
#[test]
fn possible_worlds_resample_an_opponents_foretold_card() {
    let foretell_one = |first: bool| {
        let mut s = five_color_setup(&["Sozin's Comet"]);
        let a = library_card(&s, 0, "Sozin's Comet");
        s.game_mut().move_card(a, ZoneType::Hand);
        let b = library_card(&s, 0, "Sozin's Comet");
        s.game_mut().move_card(b, ZoneType::Hand);
        let foretold = if first { a } else { b };
        assert!(foretell_offered(&mut s, foretold));
        choose(
            &mut s,
            |a| matches!(a, Action::Foretell { card, .. } if *card == foretold),
        );
        (s, foretold)
    };
    let (s, comet) = foretell_one(true);
    let (twin, _) = foretell_one(false);
    let viewer = PlayerId(1);
    let space = PossibleWorldSpace::for_viewer(s.game(), viewer);
    assert_eq!(
        space.projection(),
        PossibleWorldSpace::for_viewer(twin.game(), viewer).projection()
    );

    // 80 basics and 4 comets are unseen; one card is in hand, one comet is
    // foretold: 4 ways to fill the slot, then 83 hands.
    let projection = space.projection();
    assert_eq!(projection.foretold, 1);
    assert_eq!(space.pool()["Sozin's Comet"], 4);
    assert_eq!(space.total_weight(), 4 * 83);
    let with_comet = space
        .worlds()
        .iter()
        .find(|world| world.hand.get("Sozin's Comet") == Some(&1))
        .expect("a world holds a comet");
    assert_eq!(with_comet.weight, 4 * 3);

    let observation = viewer_observation(s.game(), viewer);
    let mut resampled = false;
    for seed in 0..32 {
        let branch = space.materialize(s.game(), with_comet, seed).unwrap();
        assert_eq!(viewer_observation(&branch, viewer), observation);
        let exiled = branch
            .state
            .zones
            .zone_cards(ZoneType::Exile, PlayerId(0))
            .clone();
        assert_eq!(exiled.len(), 1);
        let foretold = exiled[0];
        assert_eq!(branch.state.cards[foretold].name, "Sozin's Comet");
        let grant = &branch.state.cast_permissions.exile_grants[0];
        assert_eq!(Some(grant.object), branch.current_object_ref(foretold));
        resampled |= foretold != comet;
    }
    assert!(resampled);
}

/// Sparring Dummy cannot attack. Its {T} mills one card: a milled land may go
/// to hand, and a milled Lesson gains 2 life.
#[test]
fn sparring_dummy_mills_a_land_to_hand_or_a_lesson_for_two_life() {
    let mut s = five_color_setup(&["Sparring Dummy", "Earthbending Lesson", "Gray Ogre"]);
    let dummy = s.force_permanent_on_battlefield(0, "Sparring Dummy");
    clear_summoning_sickness(&mut s, dummy);
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

/// The optional waterbend {6} is offered only when payable and may be paid by
/// tapping creatures; paid, the spell reshuffles the graveyard, draws seven
/// and lifts the maximum hand size. Either way it exiles itself.
#[test]
fn spirit_water_revival_waterbends_its_additional_cost() {
    let mut s = five_color_setup(&["Spirit Water Revival", "Gray Ogre", "Craw Wurm"]);
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
        clear_summoning_sickness(&mut s, *ogre);
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

/// "Attacks alone" fires only when exactly one creature is declared; X counts
/// creatures you control on resolution.
#[test]
fn team_avatar_buffs_a_creature_attacking_alone() {
    for attackers in [1, 2] {
        let mut s = five_color_setup(&["Team Avatar", "Gray Ogre"]);
        s.force_permanent_on_battlefield(0, "Team Avatar");
        let first = s.force_permanent_on_battlefield(0, "Gray Ogre");
        let second = s.force_permanent_on_battlefield(0, "Gray Ogre");
        let third = s.force_permanent_on_battlefield(0, "Gray Ogre");
        for id in [first, second, third] {
            clear_summoning_sickness(&mut s, id);
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
    let mut s = five_color_setup(&["Team Avatar", "Gray Ogre", "Craw Wurm"]);
    s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.force_permanent_on_battlefield(0, "Gray Ogre");
    let wurm = s.force_permanent_on_battlefield(1, "Craw Wurm");
    let avatar = move_from_library(&mut s, 0, "Team Avatar", ZoneType::Hand);
    s.set_player_mana_pool(0, "2");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_activate_card(&s, avatar));
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
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_activate(&s, enchantment));
    assert!(!can_activate_card(&s, avatar));
}

/// X is announced while casting: offers run from 0 to the most the caster can
/// pay, the stack object keeps it, and the spell makes X Allies before every
/// creature you control gets a counter.
#[test]
fn united_front_announces_x_and_counts_it_on_resolution() {
    use managym::state::stack_object::StackObject;
    let mut s = five_color_setup(&["United Front", "Gray Ogre"]);
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
    choose_mode(&mut s, 3);
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

#[test]
fn ally_entry_context_buffs_the_entering_ally_and_haru_earthbends() {
    let mut s = five_color_setup(&[
        "Wartime Protestors",
        "Haru, Hidden Talent",
        "Pretending Poxbearers",
    ]);
    s.force_permanent_on_battlefield(0, "Wartime Protestors");
    s.force_permanent_on_battlefield(0, "Haru, Hidden Talent");
    let land = s.force_permanent_on_battlefield(0, "Forest");
    cast(&mut s, "Pretending Poxbearers");
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(land)));
    settle(&mut s);
    let ally = permanent_named(&s, "Pretending Poxbearers");
    assert_eq!(counters(&s, ally), 1);
    assert_eq!(counters(&s, land), 1);
    let view = Observation::for_player(s.game(), PlayerId(0));
    assert!(
        view.agent_permanents
            .iter()
            .find(|p| p.id == s.game().state.permanents[ally].as_ref().unwrap().id.0 as i32)
            .unwrap()
            .keywords
            .haste
    );
}

#[test]
fn herbalist_restriction_survives_color_selection_and_cannot_pay_abilities() {
    let mut s = five_color_setup(&[
        "Hermitic Herbalist",
        "Firebending Lesson",
        "Lightning Strike",
        "Prophetic Prism",
    ]);
    enter_and_settle(&mut s, "Hermitic Herbalist");
    enter_and_settle(&mut s, "Prophetic Prism");
    ready(&mut s, "Hermitic Herbalist");
    s.set_player_mana_pool(0, "0");
    activate_named(&mut s, "Hermitic Herbalist", 1);
    assert_choice_contract(&s);
    // The two-color choice is unordered; red/red is one distinct output.
    choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 12, .. }));
    assert_eq!(
        s.game().state.players[0]
            .mana_pool
            .restricted_of(ManaUse::spell(["Lesson"])),
        [0, 0, 0, 2, 0, 0]
    );
    let lesson = library_card(&s, 0, "Firebending Lesson");
    let strike = library_card(&s, 0, "Lightning Strike");
    s.game_mut().move_card(lesson, ZoneType::Hand);
    s.game_mut().move_card(strike, ZoneType::Hand);
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_cast(&s, lesson));
    assert!(!can_cast(&s, strike));
    let prism = permanent_named(&s, "Prophetic Prism");
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == prism)));
}

#[test]
fn a_committed_nonlesson_cost_excludes_herbalists_restricted_activation() {
    let mut s = five_color_setup(&["Hermitic Herbalist", "Lightning Strike"]);
    enter_and_settle(&mut s, "Hermitic Herbalist");
    ready(&mut s, "Hermitic Herbalist");
    s.set_player_mana_pool(0, "R");
    cast(&mut s, "Lightning Strike");
    choose(&mut s, |a| {
        matches!(
            a,
            Action::ChooseTarget {
                target: managym::state::game_object::Target::Player(PlayerId(1)),
                ..
            }
        )
    });
    let herbalist = permanent_named(&s, "Hermitic Herbalist");
    assert!(!s.action_space().actions.iter().any(|a| matches!(a, Action::ActivateAbility { permanent, ability_index: 1, .. } if *permanent == herbalist)));
    assert_choice_contract(&s);
    activate_named(&mut s, "Hermitic Herbalist", 0);
    choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 1, .. }));
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    settle(&mut s);
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
}

#[test]
fn iguana_and_boar_trigger_only_on_own_noncreature_casts() {
    let mut s = five_color_setup(&[
        "Iguana Parrot",
        "Boar-q-pine",
        "Gray Ogre",
        "Abandon Attachments",
    ]);
    let parrot = s.force_permanent_on_battlefield(0, "Iguana Parrot");
    let boar = s.force_permanent_on_battlefield(0, "Boar-q-pine");
    let opponent = s.force_permanent_on_battlefield(1, "Boar-q-pine");
    cast(&mut s, "Gray Ogre");
    settle(&mut s);
    assert_eq!(counters(&s, boar), 0);
    assert_eq!(s.game().effective_power(parrot), 2);
    cast(&mut s, "Abandon Attachments");
    settle(&mut s);
    assert_eq!(counters(&s, boar), 1);
    assert_eq!(counters(&s, opponent), 0);
    assert_eq!(s.game().effective_pt(parrot), (3, 3));
    let keywords = keywords(&s, parrot);
    assert!(keywords.flying && keywords.vigilance);
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.game().effective_power(parrot), 2);
    assert_eq!(s.game().effective_power(boar), 3);
}

/// Invasion Reinforcements has flash: castable during the opponent's turn,
/// and its ETB still makes an Ally token.
#[test]
fn invasion_reinforcements_flash_cast_on_opponents_turn() {
    let mut s = Scenario::new(
        deck(&[("Plains", 24), ("Invasion Reinforcements", 16)]),
        deck(&[("Island", 40)]),
        13,
    );
    force_lands(&mut s, 0, "Plains", 2);
    s.force_card_in_hand(0, "Invasion Reinforcements");

    // Opponent's turn, opponent holds priority first; pass to player 0.
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.action_space().player, Some(PlayerId(1)));
    s.pass_priority();
    assert_eq!(s.action_space().player, Some(PlayerId(0)));

    s.assert_action_available(ActionType::PriorityCastSpell);
    cast_and_resolve(&mut s);
    resolve_until_priority(&mut s);

    assert_eq!(
        s.battlefield_permanents_named(0, "Invasion Reinforcements")
            .len(),
        1
    );
    assert_eq!(s.battlefield_permanents_named(0, "Ally").len(), 1);
}

#[test]
fn iroh_casts_lessons_for_one_and_exiles_on_resolution_or_bounce() {
    for bounce in [false, true] {
        let mut s = five_color_setup(&["Iroh, Grand Lotus", "Waterbending Lesson"]);
        enter_and_settle(&mut s, "Iroh, Grand Lotus");
        let lesson = library_card(&s, 0, "Waterbending Lesson");
        s.game_mut().move_card(lesson, ZoneType::Graveyard);
        s.set_player_mana_pool(0, "1");
        s.game_mut().scenario_refresh_priority().unwrap();
        assert!(can_cast(&s, lesson));
        assert_choice_contract(&s);
        choose(
            &mut s,
            |a| matches!(a, Action::CastSpell { card, .. } if *card == lesson),
        );
        assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
        if bounce {
            s.game_mut().move_card(lesson, ZoneType::Hand);
        } else {
            pass_until(&mut s, ActionSpaceKind::Waterbend);
            choose(&mut s, |a| matches!(a, Action::Decline { .. }));
            settle(&mut s);
        }
        assert_eq!(s.game().state.zones.zone_of(lesson), Some(ZoneType::Exile));
    }
}

#[test]
fn iroh_preserves_sorcery_timing_and_nonlesson_printed_cost() {
    let mut s = five_color_setup(&[
        "Iroh, Grand Lotus",
        "Lightning Strike",
        "Waterbending Lesson",
    ]);
    enter_and_settle(&mut s, "Iroh, Grand Lotus");
    let strike = library_card(&s, 0, "Lightning Strike");
    let lesson = library_card(&s, 0, "Waterbending Lesson");
    s.game_mut().move_card(strike, ZoneType::Graveyard);
    s.game_mut().move_card(lesson, ZoneType::Graveyard);
    s.set_player_mana_pool(0, "1");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_cast(&s, strike));
    s.set_player_mana_pool(0, "1R");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_cast(&s, strike));
    s.advance_to_active_step(1, managym::flow::turn::StepKind::Main);
    s.game_mut().state.priority.holder = PlayerId(0);
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_cast(&s, strike));
    assert!(!can_cast(&s, lesson));
}

#[test]
fn flashback_is_exiled_when_countered_after_iroh_leaves() {
    let mut s = five_color_setup(&["Iroh, Grand Lotus", "Waterbending Lesson", "Counterspell"]);
    let iroh = enter_and_settle(&mut s, "Iroh, Grand Lotus");
    let lesson = library_card(&s, 0, "Waterbending Lesson");
    s.game_mut().move_card(lesson, ZoneType::Graveyard);
    s.game_mut().scenario_refresh_priority().unwrap();
    choose(
        &mut s,
        |a| matches!(a, Action::CastSpell { card, .. } if *card == lesson),
    );
    s.game_mut().move_card(iroh, ZoneType::Graveyard);
    let counter = library_card(&s, 1, "Counterspell");
    s.game_mut().move_card(counter, ZoneType::Hand);
    s.pass_priority();
    s.game_mut().scenario_refresh_priority().unwrap();
    choose(
        &mut s,
        |a| matches!(a, Action::CastSpell { card, .. } if *card == counter),
    );
    choose(
        &mut s,
        |a| matches!(a, Action::ChooseTarget { target: managym::state::game_object::Target::StackSpell(id), .. } if *id == lesson),
    );
    settle(&mut s);
    assert_eq!(s.game().state.zones.zone_of(lesson), Some(ZoneType::Exile));
}

#[test]
fn iroh_modes_gate_targets_and_damage_only_opponent_creatures() {
    let mut s = five_color_setup(&[
        "Iroh's Demonstration",
        "Gnarled Professor",
        "Monastery Swiftspear",
    ]);
    let own = s.force_permanent_on_battlefield(0, "Gnarled Professor");
    let other = s.force_permanent_on_battlefield(1, "Gnarled Professor");
    cast(&mut s, "Iroh's Demonstration");
    choose_mode(&mut s, 0);
    settle(&mut s);
    assert_eq!(s.game().state.permanents[own].as_ref().unwrap().damage, 0);
    assert_eq!(s.game().state.permanents[other].as_ref().unwrap().damage, 1);
    cast(&mut s, "Iroh's Demonstration");
    choose_mode(&mut s, 1);
    assert!(s.choose_target(Target::Permanent(other)));
    settle(&mut s);
    assert!(s.game().state.permanents[other].is_none());
    let mut empty = five_color_setup(&["Iroh's Demonstration"]);
    cast(&mut empty, "Iroh's Demonstration");
    assert_eq!(empty.action_space().actions.len(), 1);
    assert!(matches!(
        empty.action_space().actions[0],
        Action::ChooseMode { mode: 0, .. }
    ));
}

/// It'll Quench Ya! resolving against player 1's Gray Ogre, stopped at player 1's
/// pay-{2}-or-not decision.
fn quench_setup(seed: u64, p1_mountains: usize) -> Scenario {
    let mut s = Scenario::new(
        deck(&[("Island", 24), ("It'll Quench Ya!", 16)]),
        deck(&[("Mountain", 24), ("Gray Ogre", 16)]),
        seed,
    );
    force_lands(&mut s, 0, "Island", 2);
    force_lands(&mut s, 1, "Mountain", p1_mountains);
    clear_hand(&mut s, 0);
    clear_hand(&mut s, 1);
    s.force_card_in_hand(0, "It'll Quench Ya!");
    s.force_card_in_hand(1, "Gray Ogre");
    s.advance_to_active_step(1, StepKind::Main);
    // P1 casts Gray Ogre, then passes; P0 responds with the counter.
    cast_only(&mut s);
    s.pass_priority(); // P1 passes
    cast_only(&mut s); // P0 casts It'll Quench Ya!
    assert_eq!(s.action_space().kind, ActionSpaceKind::ChooseTarget);
    let ogre = stack_spell_card(&s, "Gray Ogre");
    assert!(s.choose_target(Target::StackSpell(ogre)));
    s.pass_priority();
    s.pass_priority(); // It'll Quench Ya! resolves -> pay-or-not for P1
    s
}

#[test]
fn quench_controller_declines_and_spell_is_countered() {
    let mut s = quench_setup(41, 3);
    let space = s.action_space();
    assert_eq!(space.kind, ActionSpaceKind::PayOrNot);
    assert_eq!(
        space.player,
        Some(PlayerId(1)),
        "the spell's controller decides"
    );
    assert!(s.take_action_by_type(ActionType::DeclineChoice));

    // Gray Ogre countered; both spells in graveyards; stack empty.
    assert_eq!(s.game().state.stack_objects.len(), 0);
    assert_eq!(s.zone_size(1, ZoneType::Graveyard), 1);
    assert_eq!(s.zone_size(0, ZoneType::Graveyard), 1);
    assert_eq!(s.battlefield_permanents_named(1, "Gray Ogre").len(), 0);
}

#[test]
fn quench_controller_pays_and_spell_resolves() {
    let mut s = quench_setup(42, 5);
    assert_eq!(s.action_space().kind, ActionSpaceKind::PayOrNot);
    assert!(s.take_action_by_type(ActionType::PayCost));
    // Ogre still on the stack; let it resolve.
    assert_eq!(s.game().state.stack_objects.len(), 1);
    s.pass_priority();
    s.pass_priority();
    assert_eq!(s.battlefield_permanents_named(1, "Gray Ogre").len(), 1);
}

#[test]
fn quench_pay_option_absent_when_controller_cannot_pay() {
    let mut s = quench_setup(43, 3);
    // All three of P1's mountains were tapped casting the Ogre.
    let space = s.action_space();
    assert_eq!(space.kind, ActionSpaceKind::PayOrNot);
    assert_eq!(space.actions.len(), 1);
    assert_eq!(space.actions[0].action_type(), ActionType::DeclineChoice);
    assert!(s.take_action_by_type(ActionType::DeclineChoice));
    assert_eq!(s.battlefield_permanents_named(1, "Gray Ogre").len(), 0);
}

/// Jeong Jeong's Deserters puts a +1/+1 counter on target creature; the
/// counter raises toughness for the lethal-damage check (CR 704.5g).
#[test]
fn jeong_jeongs_deserters_targeted_counter_and_lethal_sba() {
    let mut s = Scenario::new(
        deck(&[("Plains", 24), ("Jeong Jeong's Deserters", 16)]),
        deck(&[("Plains", 40)]),
        17,
    );
    force_lands(&mut s, 0, "Plains", 2);
    s.force_card_in_hand(0, "Jeong Jeong's Deserters");
    s.advance_to_active_step(0, StepKind::Main);
    cast_and_resolve(&mut s);

    // The ETB trigger needs a target.
    assert_eq!(s.action_space().kind, ActionSpaceKind::ChooseTarget);
    s.choose_target_named("Jeong Jeong's Deserters");
    resolve_until_priority(&mut s);

    let deserters = s.battlefield_permanents_named(0, "Jeong Jeong's Deserters")[0];
    assert_eq!(counters(&s, deserters), 1);

    // 1/2 with a +1/+1 counter is 2/3: 2 damage is not lethal...
    s.game_mut().state.permanents[deserters]
        .as_mut()
        .unwrap()
        .damage = 2;
    s.pass_priority();
    assert!(!s
        .battlefield_permanents_named(0, "Jeong Jeong's Deserters")
        .is_empty());

    // ...but 3 damage is.
    s.game_mut().state.permanents[deserters]
        .as_mut()
        .unwrap()
        .damage = 3;
    s.pass_priority();
    assert!(s
        .battlefield_permanents_named(0, "Jeong Jeong's Deserters")
        .is_empty());
    assert_eq!(
        s.game()
            .state
            .zones
            .zone_cards(ZoneType::Graveyard, PlayerId(0))
            .len(),
        1
    );
}

#[test]
fn jet_entry_damage_and_death_choose_up_to_two_distinct_creatures() {
    let mut s = five_color_setup(&[
        "Jet, Freedom Fighter",
        "Gnarled Professor",
        "Lightning Strike",
    ]);
    let own = s.force_permanent_on_battlefield(0, "Gnarled Professor");
    let other = s.force_permanent_on_battlefield(1, "Gnarled Professor");
    cast(&mut s, "Jet, Freedom Fighter");
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(other)));
    settle(&mut s);
    assert_eq!(s.game().state.permanents[other].as_ref().unwrap().damage, 2);
    let jet = permanent_named(&s, "Jet, Freedom Fighter");
    cast(&mut s, "Lightning Strike");
    assert!(s.choose_target(Target::Permanent(jet)));
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(own)));
    assert!(!s.action_space().actions.iter().any(|a| matches!(a, Action::ChooseTarget { target: managym::state::target::Target::Permanent(id), .. } if *id == own)));
    assert!(s.choose_target(Target::Permanent(other)));
    settle(&mut s);
    assert_eq!(counters(&s, own), 1);
    assert_eq!(counters(&s, other), 1);
}

#[test]
fn june_sacrifices_another_creature_only_on_its_controllers_turn() {
    let mut s = five_color_setup(&["June, Bounty Hunter", "Callous Inspector"]);
    let june = play_and_settle(&mut s, "June, Bounty Hunter");
    let fodder = play_and_settle(&mut s, "Callous Inspector");
    let source = permanent_named(&s, "June, Bounty Hunter");
    activate_first(&mut s, source);
    assert!(s
        .action_space()
        .actions
        .iter()
        .all(|a| !matches!(a,Action::SelectCard {card,..} if *card==june)));
    select_card(&mut s, fodder);
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Clue").len(), 2);
    s.advance_to_active_step(1, managym::flow::turn::StepKind::Main);
    s.pass_priority();
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a,Action::ActivateAbility {permanent,..} if *permanent==source)));
}

#[test]
fn june_is_unblockable_after_two_draws() {
    for draws in [0, 2] {
        let mut s = five_color_setup(&["June, Bounty Hunter", "Callous Inspector"]);
        let june = s.force_permanent_on_battlefield(0, "June, Bounty Hunter");
        s.force_permanent_on_battlefield(1, "Callous Inspector");
        s.game_mut().state.permanents[june]
            .as_mut()
            .unwrap()
            .summoning_sick = false;
        s.game_mut().draw_cards(PlayerId(0), draws);
        s.advance_to_active_step(0, managym::flow::turn::StepKind::DeclareAttackers);
        s.declare_attack();
        s.advance_to_active_step(0, managym::flow::turn::StepKind::DeclareBlockers);
        assert_eq!(
            s.action_space()
                .actions
                .iter()
                .any(|a| matches!(a,Action::DeclareBlocker {attacker:Some(id),..} if *id==june)),
            draws == 0
        );
    }
}

/// Kyoshi Warriors: "When this creature enters, create a 1/1 white Ally
/// creature token." The targetless trigger still uses the stack and both
/// players receive priority before it resolves (CR 603.3, 117.3d).
#[test]
fn kyoshi_warriors_etb_creates_ally_token_via_stack() {
    let mut s = Scenario::new(
        deck(&[("Plains", 24), ("Kyoshi Warriors", 16)]),
        deck(&[("Plains", 40)]),
        7,
    );
    force_lands(&mut s, 0, "Plains", 4);
    s.force_card_in_hand(0, "Kyoshi Warriors");
    s.advance_to_active_step(0, StepKind::Main);

    cast_and_resolve(&mut s);

    // The ETB trigger is on the stack; both players get a response window.
    assert_eq!(s.game().state.stack_objects.len(), 1);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
    assert_eq!(s.action_space().player, Some(PlayerId(0)));
    s.pass_priority();
    assert_eq!(s.action_space().player, Some(PlayerId(1)));
    assert_eq!(s.game().state.stack_objects.len(), 1);
    s.pass_priority();

    let tokens = s.battlefield_permanents_named(0, "Ally");
    assert_eq!(tokens.len(), 1, "one Ally token should have been created");
    let token = permanent(&s, tokens[0]);
    let token_card = &s.game().state.cards[token.card];
    assert!(token_card.is_token);
    assert!(token_card.has_subtype("Ally"));
    assert!(token_card.types.is_creature());
    assert_eq!(token_card.power, Some(1));
    assert_eq!(token_card.toughness, Some(1));
    assert_eq!(
        token_card.colors(),
        managym::state::mana::Colors::from([managym::state::mana::Color::White])
    );
    assert_eq!(token.controller, PlayerId(0));
}

#[test]
fn master_piandao_reveals_a_selected_ally_and_bottoms_the_rest() {
    let mut s = five_color_setup(&[
        "Master Piandao",
        "Pretending Poxbearers",
        "Abandon Attachments",
    ]);
    let master = s.force_permanent_on_battlefield(0, "Master Piandao");
    s.game_mut().state.permanents[master]
        .as_mut()
        .unwrap()
        .summoning_sick = false;
    let top = pin_top(
        &mut s,
        &[
            "Plains",
            "Pretending Poxbearers",
            "Abandon Attachments",
            "Island",
        ],
    );
    let ally = top[1];
    s.advance_to_active_step(0, StepKind::BeginningOfCombat);
    s.pass_priority();
    s.pass_priority();
    s.declare_attack();
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    assert_eq!(s.action_space().kind, ActionSpaceKind::LookAndSelect);
    assert_eq!(
        s.action_space()
            .actions
            .iter()
            .filter(|a| matches!(a, Action::SelectCard { .. }))
            .count(),
        2
    );
    let index = s
        .action_space()
        .actions
        .iter()
        .position(|a| matches!(a,Action::SelectCard{card,..} if *card==ally))
        .unwrap();
    s.step_action(index);
    settle(&mut s);
    assert_eq!(s.game().state.zones.zone_of(ally), Some(ZoneType::Hand));
    let definition = s.game().state.cards[ally].definition_id.0;
    let other = Observation::for_player(s.game(), PlayerId(1));
    assert_eq!(other.opponent.known_hand.get(&definition), Some(&1));
    assert!(other
        .opponent_cards
        .iter()
        .all(|c| c.zone != ZoneType::Hand));
}

#[test]
fn piandao_can_take_a_lesson_or_equipment_or_decline_without_revealing_remainder() {
    for selection in [None, Some(0), Some(1)] {
        let mut s = five_color_setup(&["Master Piandao", "Abandon Attachments", "Prophetic Prism"]);
        let master = s.force_permanent_on_battlefield(0, "Master Piandao");
        s.game_mut().state.permanents[master]
            .as_mut()
            .unwrap()
            .summoning_sick = false;
        let top = pin_top(
            &mut s,
            &["Abandon Attachments", "Prophetic Prism", "Forest", "Island"],
        );
        // A controlled Equipment fixture isolates that subtype branch without
        // implementing a later batch's equip/attachment mechanics.
        s.game_mut().state.cards[top[1]]
            .subtypes
            .push("Equipment".into());
        s.advance_to_active_step(0, StepKind::DeclareAttackers);
        s.declare_attack();
        pass_until(&mut s, ActionSpaceKind::LookAndSelect);
        assert_eq!(s.action_space().actions.len(), 3);
        choose(&mut s, |a| match selection {
            Some(index) => matches!(a, Action::SelectCard { card, .. } if *card == top[index]),
            None => matches!(a, Action::Decline { .. }),
        });
        settle(&mut s);
        let view = Observation::for_player(s.game(), PlayerId(1));
        let library = s
            .game()
            .state
            .zones
            .zone_cards(ZoneType::Library, PlayerId(0));
        for (index, id) in top.iter().enumerate() {
            let definition = s.game().state.cards[*id].definition_id.0;
            assert_eq!(
                view.opponent
                    .known_hand
                    .get(&definition)
                    .copied()
                    .unwrap_or(0),
                u32::from(selection == Some(index))
            );
            if selection != Some(index) {
                assert!(library[..4 - usize::from(selection.is_some())].contains(id));
            }
        }
    }
}

#[test]
fn hybrid_draw_payoffs_use_draw_count_and_trigger_on_the_second_draw_only() {
    let mut s = five_color_setup(&[
        "Messenger Hawk",
        "Knowledge Seeker",
        "Foggy Swamp Spirit Keeper",
        "Spectral Sailor",
    ]);
    let hawk = s.force_permanent_on_battlefield(0, "Messenger Hawk");
    let fox = s.force_permanent_on_battlefield(0, "Knowledge Seeker");
    s.force_permanent_on_battlefield(0, "Foggy Swamp Spirit Keeper");
    let sailor = s.force_permanent_on_battlefield(0, "Spectral Sailor");
    for draw in 1..=3 {
        s.game_mut().scenario_refresh_priority().unwrap();
        let index = s
            .action_space()
            .actions
            .iter()
            .position(|a| matches!(a,Action::ActivateAbility{permanent,..} if *permanent==sailor))
            .unwrap();
        s.step_action(index);
        settle(&mut s);
        assert_eq!(counters(&s, fox), if draw >= 2 { 1 } else { 0 });
        assert_eq!(
            s.battlefield_permanents_named(0, "Spirit").len(),
            if draw >= 2 { 1 } else { 0 }
        );
        assert_eq!(
            s.game().effective_power(hawk),
            if draw >= 2 { 3 } else { 1 }
        );
    }
}

#[test]
fn messenger_hawk_accepts_either_hybrid_color_and_investigates_on_entry() {
    for pool in ["2U", "2B"] {
        let mut s = five_color_setup(&["Messenger Hawk"]);
        s.set_player_mana_pool(0, pool);
        cast(&mut s, "Messenger Hawk");
        settle(&mut s);
        assert_eq!(s.battlefield_permanents_named(0, "Clue").len(), 1);
        assert_eq!(
            s.game()
                .effective_power(permanent_named(&s, "Messenger Hawk")),
            1
        );
    }
}

#[test]
fn momo_reduces_first_matching_spell_only_and_grows_when_another_flier_enters() {
    let mut s = five_color_setup(&["Momo, Friendly Flier", "Messenger Hawk"]);
    let momo = s.force_permanent_on_battlefield(0, "Momo, Friendly Flier");
    let hawk = library_card(&s, 0, "Messenger Hawk");
    s.game_mut().move_card(hawk, ZoneType::Hand);
    s.set_player_mana_pool(0, "1U");
    s.game_mut().scenario_refresh_priority().unwrap();
    let index = s
        .action_space()
        .actions
        .iter()
        .position(|a| matches!(a, Action::CastSpell { card, .. } if *card == hawk))
        .unwrap();
    s.step_action(index);
    settle(&mut s);
    assert_eq!(s.game().effective_power(momo), 2);
    let next = library_card(&s, 0, "Messenger Hawk");
    s.game_mut().move_card(next, ZoneType::Hand);
    s.set_player_mana_pool(0, "1U");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::CastSpell { card, .. } if *card == next)));
}

#[test]
fn momo_counts_earlier_matching_casts_before_it_entered_and_resets_next_turn() {
    let mut s = five_color_setup(&["Momo, Friendly Flier", "Messenger Hawk"]);
    cast(&mut s, "Messenger Hawk");
    settle(&mut s);
    s.force_permanent_on_battlefield(0, "Momo, Friendly Flier");
    let next = library_card(&s, 0, "Messenger Hawk");
    s.game_mut().move_card(next, ZoneType::Hand);
    s.set_player_mana_pool(0, "1U");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|action| matches!(action, Action::CastSpell { card, .. } if *card == next)));
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    s.set_player_mana_pool(0, "1U");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(s
        .action_space()
        .actions
        .iter()
        .any(|action| matches!(action, Action::CastSpell { card, .. } if *card == next)));
}

#[test]
fn momo_excludes_lemurs_and_opponents_turn_and_its_pump_expires() {
    let mut s = five_color_setup(&["Momo, Friendly Flier", "Messenger Hawk"]);
    let momo = s.force_permanent_on_battlefield(0, "Momo, Friendly Flier");
    let other = s.force_permanent_on_battlefield(1, "Momo, Friendly Flier");
    let lemur = library_card(&s, 0, "Momo, Friendly Flier");
    let hawk = library_card(&s, 0, "Messenger Hawk");
    s.game_mut().move_card(lemur, ZoneType::Hand);
    s.game_mut().move_card(hawk, ZoneType::Hand);
    s.set_player_mana_pool(0, "1");
    s.game_mut().scenario_refresh_priority().unwrap();
    // A generic reduction must never remove white.
    assert!(!can_cast(&s, lemur));
    // Give the more expensive flier a Lemur subtype so exclusion has a
    // measurable generic component instead of being hidden by Momo's {W}.
    s.game_mut().state.cards[hawk].subtypes.push("Lemur".into());
    s.set_player_mana_pool(0, "1U");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_cast(&s, hawk));
    s.game_mut().state.cards[hawk]
        .subtypes
        .retain(|t| t != "Lemur");
    s.game_mut().scenario_refresh_priority().unwrap();
    choose(
        &mut s,
        |a| matches!(a, Action::CastSpell { card, .. } if *card == hawk),
    );
    settle(&mut s);
    assert_eq!(s.game().effective_power(momo), 2);
    assert_eq!(s.game().effective_power(other), 1);
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.game().effective_power(momo), 1);
    let next = library_card(&s, 0, "Messenger Hawk");
    s.game_mut().move_card(next, ZoneType::Hand);
    s.game_mut().state.cards[next].keywords.flash = true;
    s.set_player_mana_pool(0, "1U");
    s.game_mut().scenario_refresh_priority().unwrap();
    s.pass_priority();
    assert!(!can_cast(&s, next));
    s.set_player_mana_pool(0, "2U");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_cast(&s, next));
}

#[test]
fn octopus_form_untaps_only_own_target_and_expires() {
    let mut s = five_color_setup(&["Octopus Form", "Gray Ogre"]);
    let own = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let other = s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.game_mut().state.permanents[own].as_mut().unwrap().tapped = true;
    cast(&mut s, "Octopus Form");
    assert!(!s.choose_target(Target::Permanent(other)));
    assert!(s.choose_target(Target::Permanent(own)));
    settle(&mut s);
    assert!(!tapped(&s, own));
    assert_eq!(s.game().effective_pt(own), (3, 3));
    assert!(keywords(&s, own).hexproof);
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.game().effective_pt(own), (2, 2));
    assert!(!keywords(&s, own).hexproof);
}

#[test]
fn origin_destroys_artifacts_or_protects_with_counter_and_indestructible() {
    let mut s = five_color_setup(&[
        "Origin of Metalbending",
        "Lightning Strike",
        "Power Word Kill",
        "Eyetwitch",
    ]);
    s.game_mut().create_token("Clue", PlayerId(1), false);
    let clue = s.battlefield_permanents_named(1, "Clue")[0];
    cast(&mut s, "Origin of Metalbending");
    choose_mode(&mut s, 0);
    assert!(s.choose_target(Target::Permanent(clue)));
    settle(&mut s);
    assert!(s.game().state.permanents[clue].is_none());
    let creature = s.force_permanent_on_battlefield(0, "Eyetwitch");
    cast(&mut s, "Origin of Metalbending");
    choose_mode(&mut s, 1);
    assert!(s.choose_target(Target::Permanent(creature)));
    settle(&mut s);
    assert_eq!(counters(&s, creature), 1);
    for removal in ["Power Word Kill", "Lightning Strike"] {
        cast(&mut s, removal);
        assert!(s.choose_target(Target::Permanent(creature)));
        settle(&mut s);
        assert!(s.game().state.permanents[creature].is_some());
    }
}

#[test]
fn origin_indestructible_expires_but_counter_remains() {
    let mut s = five_color_setup(&["Origin of Metalbending", "Gray Ogre"]);
    let target = s.force_permanent_on_battlefield(0, "Gray Ogre");
    cast(&mut s, "Origin of Metalbending");
    choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 1, .. }));
    assert!(s.choose_target(Target::Permanent(target)));
    settle(&mut s);
    assert!(keywords(&s, target).indestructible);
    s.advance_to_active_step(1, StepKind::Main);
    assert!(!keywords(&s, target).indestructible);
    assert_eq!(counters(&s, target), 1);
}

#[test]
fn ostrich_mills_then_selects_only_a_milled_land_or_grows_when_declined() {
    for take in [false, true] {
        let mut s = five_color_setup(&["Ostrich-Horse", "Lightning Strike"]);
        let top = pin_top(&mut s, &["Lightning Strike", "Forest", "Island"]);
        cast(&mut s, "Ostrich-Horse");
        pass_until(&mut s, ActionSpaceKind::LookAndSelect);
        assert_eq!(s.action_space().actions.len(), 3);
        assert!(top
            .iter()
            .all(|card| s.game().state.zones.zone_of(*card) == Some(ZoneType::Graveyard)));
        let index = s
            .action_space()
            .actions
            .iter()
            .position(|a| {
                if take {
                    matches!(a, Action::SelectCard { card, .. } if *card == top[1])
                } else {
                    matches!(a, Action::Decline { .. })
                }
            })
            .unwrap();
        s.step_action(index);
        settle(&mut s);
        assert_eq!(
            counters(&s, permanent_named(&s, "Ostrich-Horse")),
            if take { 0 } else { 1 }
        );
        assert_eq!(
            s.game().state.zones.zone_of(top[1]),
            Some(if take {
                ZoneType::Hand
            } else {
                ZoneType::Graveyard
            })
        );
    }
}

#[test]
fn otter_penguin_buffs_on_second_draw_only_and_resets() {
    let mut s = Scenario::new(
        deck(&[("Island", 24), ("Otter-Penguin", 16)]),
        deck(&[("Island", 40)]),
        23,
    );
    let otter = s.force_permanent_on_battlefield(0, "Otter-Penguin");
    // Player 0 skips their draw on turn 1; run the test from their second
    // turn, where the draw step provides the first card of the turn.
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);

    // Draw-step draw was the first card this turn — no buff yet.
    assert_eq!(s.game().state.turn.cards_drawn_this_turn[0], 1);
    assert_eq!(permanent(&s, otter).temp_power, 0);

    // Second draw: +1/+2 and can't be blocked this turn.
    s.game_mut().draw_cards(PlayerId(0), 1);
    resolve_until_priority(&mut s);
    assert_eq!(permanent(&s, otter).temp_power, 1);
    assert_eq!(permanent(&s, otter).temp_toughness, 2);
    assert!(permanent(&s, otter).cant_be_blocked_this_turn);

    // Third draw: no further trigger.
    s.game_mut().draw_cards(PlayerId(0), 1);
    resolve_until_priority(&mut s);
    assert_eq!(permanent(&s, otter).temp_power, 1);

    // Cleanup clears the until-EOT buff and the unblockable flag; the
    // per-turn draw count resets.
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(permanent(&s, otter).temp_power, 0);
    assert_eq!(permanent(&s, otter).temp_toughness, 0);
    assert!(!permanent(&s, otter).cant_be_blocked_this_turn);
    assert_eq!(s.game().state.turn.cards_drawn_this_turn[0], 0);

    // Next turn the "second card each turn" trigger works again.
    s.advance_to_active_step(0, StepKind::Main);
    s.game_mut().draw_cards(PlayerId(0), 1);
    resolve_until_priority(&mut s);
    assert_eq!(permanent(&s, otter).temp_power, 1);
}

/// The opponent drawing their second card must not trigger player 0's
/// Otter-Penguin.
#[test]
fn otter_penguin_ignores_opponent_draws() {
    let mut s = Scenario::new(
        deck(&[("Island", 24), ("Otter-Penguin", 16)]),
        deck(&[("Island", 40)]),
        29,
    );
    let otter = s.force_permanent_on_battlefield(0, "Otter-Penguin");
    s.advance_to_active_step(1, StepKind::Main);

    s.game_mut().draw_cards(PlayerId(1), 1);
    resolve_until_priority(&mut s);
    assert_eq!(s.game().state.turn.cards_drawn_this_turn[1], 2);
    assert_eq!(permanent(&s, otter).temp_power, 0);
}

#[test]
fn court_official_and_ozai_make_the_target_player_choose_discards() {
    for name in ["Corrupt Court Official", "Ozai's Cruelty"] {
        let mut s = five_color_setup(&[name]);
        for _ in 0..3 {
            let id = library_card(&s, 1, "Plains");
            s.game_mut().move_card(id, ZoneType::Hand);
        }
        cast(&mut s, name);
        if name == "Corrupt Court Official" {
            pass_until(&mut s, ActionSpaceKind::ChooseTarget);
        }
        assert!(s.choose_target(Target::Player(PlayerId(1))));
        pass_until(&mut s, ActionSpaceKind::Discard);
        assert_eq!(s.action_space().player, Some(PlayerId(1)));
        settle(&mut s);
        assert_eq!(
            s.zone_size(1, ZoneType::Hand),
            if name == "Corrupt Court Official" {
                2
            } else {
                1
            }
        );
        assert_eq!(
            s.life(1),
            if name == "Corrupt Court Official" {
                20
            } else {
                18
            }
        );
    }
}

#[test]
fn pillar_launch_untaps_and_grants_temporary_reach_and_stats() {
    let mut s = five_color_setup(&["Pillar Launch", "Gray Ogre"]);
    let creature = s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.game_mut().state.permanents[creature]
        .as_mut()
        .unwrap()
        .tapped = true;
    cast(&mut s, "Pillar Launch");
    assert!(s.choose_target(Target::Permanent(creature)));
    settle(&mut s);
    assert!(!tapped(&s, creature));
    assert_eq!(s.game().effective_pt(creature), (4, 4));
    assert!(keywords(&s, creature).reach);
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.game().effective_power(creature), 2);
    assert!(!keywords(&s, creature).reach);
}

#[test]
fn ran_and_shaw_copy_is_nonlegendary_uncast_and_has_printed_abilities() {
    let mut s = five_color_setup(&["Ran and Shaw", "Waterbending Lesson"]);
    for _ in 0..3 {
        let lesson = library_card(&s, 0, "Waterbending Lesson");
        s.game_mut().move_card(lesson, ZoneType::Graveyard);
    }
    cast(&mut s, "Ran and Shaw");
    settle(&mut s);
    let dragons = s.battlefield_permanents_named(0, "Ran and Shaw");
    assert_eq!(dragons.len(), 2);
    let token = dragons
        .iter()
        .find_map(|id| {
            let p = s.game().state.permanents[*id].as_ref().unwrap();
            s.game().state.cards[p.card].is_token.then_some(p)
        })
        .unwrap();
    assert!(!token.was_cast);
    let definition = &s.game().state.cards[token.card];
    assert!(definition.supertypes.is_empty());
    assert!(definition.keywords.flying);
    assert_eq!(definition.abilities.len(), 2);
    assert_eq!(definition.activated_abilities.len(), 1);
    assert_eq!(definition.mana_cost.as_ref().unwrap().mana_value, 5);
    activate_named(&mut s, "Ran and Shaw", 0);
    settle(&mut s);
    for id in dragons {
        assert_eq!(s.game().effective_power(id), 6);
    }
}

#[test]
fn ran_and_shaw_requires_a_cast_and_rechecks_graveyard_at_resolution() {
    for mode in 0..3 {
        let mut s = five_color_setup(&["Ran and Shaw", "Waterbending Lesson"]);
        let mut lessons = Vec::new();
        for _ in 0..3 {
            let lesson = library_card(&s, 0, "Waterbending Lesson");
            s.game_mut().move_card(lesson, ZoneType::Graveyard);
            lessons.push(lesson);
        }
        if mode == 0 {
            enter_and_settle(&mut s, "Ran and Shaw");
        } else {
            cast(&mut s, "Ran and Shaw");
            // Resolve the creature but leave its entry trigger on the stack.
            for _ in 0..10 {
                if s.battlefield_permanents_named(0, "Ran and Shaw").len() == 1 {
                    break;
                }
                s.pass_priority();
            }
            assert_eq!(s.battlefield_permanents_named(0, "Ran and Shaw").len(), 1);
            assert_choice_contract(&s);
            if mode == 1 {
                s.game_mut().move_card(lessons[0], ZoneType::Exile);
            }
            if mode == 2 {
                let source = permanent_named(&s, "Ran and Shaw");
                s.game_mut().state.permanents[source]
                    .as_mut()
                    .unwrap()
                    .plus1_counters = 3;
                let source_card = s.game().state.permanents[source].as_ref().unwrap().card;
                s.game_mut().move_card(source_card, ZoneType::Graveyard);
            }
            settle(&mut s);
        }
        let dragons = s.battlefield_permanents_named(0, "Ran and Shaw");
        assert_eq!(dragons.len(), 1);
        let dragon = s.game().state.permanents[dragons[0]].as_ref().unwrap();
        assert_eq!(s.game().state.cards[dragon.card].is_token, mode == 2);
        assert_eq!(s.game().effective_power(dragons[0]), 4);
    }
}

#[test]
fn rumble_arena_scries_then_filters_one_paid_mana_without_using_the_stack() {
    let mut s = five_color_setup(&["Rumble Arena", "Crossroads of Destiny"]);
    let arena = enter_battlefield(&mut s, "Rumble Arena");
    pass_until(&mut s, ActionSpaceKind::Scry);
    let top = *s
        .game()
        .state
        .zones
        .zone_cards(ZoneType::Library, PlayerId(0))
        .last()
        .unwrap();
    choose(&mut s, |a| {
        matches!(
            a,
            Action::ScryCard {
                to_bottom: true,
                ..
            }
        )
    });
    settle(&mut s);
    assert_eq!(
        s.game()
            .state
            .zones
            .zone_cards(ZoneType::Library, PlayerId(0))[0],
        top
    );
    assert!(!tapped(&s, arena));
    assert!(keywords(&s, arena).vigilance);
    s.set_player_mana_pool(0, "0");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(
        !s.action_space()
            .actions
            .iter()
            .any(|a| matches!(a,Action::ActivateAbility{permanent,..} if *permanent==arena)),
        "filter cannot pay for itself"
    );
    s.set_player_mana_pool(0, "1");
    activate(&mut s, arena, 0);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Modal);
    assert!(tapped(&s, arena));
    assert!(s.game().state.stack_objects.is_empty());
    choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 4, .. }));
    assert_eq!(s.game().state.players[0].mana_pool.total(), 1);
    assert_eq!(s.game().state.players[0].mana_pool.mana[4], 1);
    s.game_mut().state.permanents[arena]
        .as_mut()
        .unwrap()
        .tapped = false;
    s.set_player_mana_pool(0, "0");
    cast(&mut s, "Crossroads of Destiny");
    settle(&mut s);
    assert!(tapped(&s, arena));
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
}

#[test]
fn shared_roots_search_can_find_or_decline_without_leaking_the_library() {
    for take in [false, true] {
        let mut s = five_color_setup(&["Shared Roots", "Rumble Arena"]);
        cast(&mut s, "Shared Roots");
        pass_until(&mut s, ActionSpaceKind::LookAndSelect);
        let forest = library_card(&s, 0, "Forest");
        let nonbasic = library_card(&s, 0, "Rumble Arena");
        assert!(!s
            .action_space()
            .actions
            .iter()
            .any(|a| matches!(a, Action::SelectCard { card, .. } if *card == nonbasic)));
        assert!(Observation::for_player(s.game(), PlayerId(1))
            .opponent_cards
            .iter()
            .all(|c| c.zone != ZoneType::Library));
        choose(&mut s, |a| {
            if take {
                matches!(a, Action::SelectCard { card, .. } if *card == forest)
            } else {
                matches!(a, Action::Decline { .. })
            }
        });
        settle(&mut s);
        assert_eq!(
            s.battlefield_permanents_named(0, "Forest").len(),
            usize::from(take)
        );
        if take {
            assert!(tapped(&s, permanent_named(&s, "Forest")));
        }
        assert!(s.game().state.suspended_decision.is_none());
    }
}

#[test]
fn sokka_binds_spell_and_land_separately_and_resolves_with_one_illegal_target() {
    let mut s = five_color_setup(&["Sokka's Haiku", "Lightning Strike"]);
    let land = s.force_permanent_on_battlefield(1, "Forest");
    s.game_mut().state.permanents[land].as_mut().unwrap().tapped = true;
    let strike = cast(&mut s, "Lightning Strike");
    assert!(s.choose_target(Target::Player(PlayerId(1))));
    cast(&mut s, "Sokka's Haiku");
    assert!(s.choose_target(Target::StackSpell(strike)));
    assert!(s.choose_target(Target::Permanent(land)));
    // Remove one target in response: draw/mill and the other clause still resolve.
    s.game_mut().move_card(strike, ZoneType::Graveyard);
    s.game_mut().state.stack_objects.retain(|object| !matches!(object, managym::state::stack_object::StackObject::Spell(spell) if spell.card == strike));
    let library = s.zone_size(0, ZoneType::Library);
    settle(&mut s);
    assert_eq!(s.zone_size(0, ZoneType::Library), library - 4);
    assert!(!s.game().state.permanents[land].as_ref().unwrap().tapped);
}

#[test]
fn sokkas_haiku_counters_draws_mills_and_untaps_and_needs_a_legal_target() {
    for invalid in [false, true] {
        let mut s = five_color_setup(&["Sokka's Haiku", "Lightning Strike"]);
        let land = s.force_permanent_on_battlefield(0, "Forest");
        let land_card = s.game().state.permanents[land].as_ref().unwrap().card;
        s.game_mut().state.permanents[land].as_mut().unwrap().tapped = true;
        let strike = cast(&mut s, "Lightning Strike");
        assert!(s.choose_target(Target::Player(PlayerId(1))));
        cast(&mut s, "Sokka's Haiku");
        assert!(s.choose_target(Target::StackSpell(strike)));
        assert!(s.choose_target(Target::Permanent(land)));
        let library = s.zone_size(0, ZoneType::Library);
        if invalid {
            s.game_mut().move_card(land_card, ZoneType::Hand);
            s.game_mut().move_card(strike, ZoneType::Graveyard);
            s.game_mut().state.stack_objects.retain(|o| !matches!(o, managym::state::stack_object::StackObject::Spell(spell) if spell.card == strike));
        }
        settle(&mut s);
        assert_eq!(s.life(1), 20);
        assert_eq!(
            s.zone_size(0, ZoneType::Library),
            library - if invalid { 0 } else { 4 }
        );
        assert_eq!(
            s.game().state.zones.zone_of(strike),
            Some(ZoneType::Graveyard)
        );
        if !invalid {
            assert!(!tapped(&s, land));
        }
        assert_eq!(
            s.zone_size(0, ZoneType::Graveyard),
            if invalid { 2 } else { 5 }
        );
    }
}

#[test]
fn south_pole_voyager_second_resolution_draws() {
    let mut s = Scenario::new(
        deck(&[("Plains", 24), ("South Pole Voyager", 16)]),
        deck(&[("Plains", 40)]),
        41,
    );
    s.force_permanent_on_battlefield(0, "South Pole Voyager");
    s.advance_to_active_step(0, StepKind::Main);

    // First Ally entering this turn: gain 1 life, no draw.
    let hand_before = s.zone_size(0, ZoneType::Hand);
    let life_before = s.life(0);
    s.game_mut().create_token("Ally", PlayerId(0), false);
    resolve_until_priority(&mut s);
    assert_eq!(s.life(0), life_before + 1);
    assert_eq!(s.zone_size(0, ZoneType::Hand), hand_before);

    // Second time the ability resolves this turn: gain 1 life AND draw.
    s.game_mut().create_token("Ally", PlayerId(0), false);
    resolve_until_priority(&mut s);
    assert_eq!(s.life(0), life_before + 2);
    assert_eq!(s.zone_size(0, ZoneType::Hand), hand_before + 1);

    // Third time: life only.
    s.game_mut().create_token("Ally", PlayerId(0), false);
    resolve_until_priority(&mut s);
    assert_eq!(s.life(0), life_before + 3);
    assert_eq!(s.zone_size(0, ZoneType::Hand), hand_before + 1);

    // The per-turn resolution count resets: next turn's first resolution
    // does not draw.
    s.advance_to_active_step(1, StepKind::Main);
    let hand_now = s.zone_size(0, ZoneType::Hand);
    s.game_mut().create_token("Ally", PlayerId(0), false);
    resolve_until_priority(&mut s);
    assert_eq!(s.life(0), life_before + 4);
    assert_eq!(s.zone_size(0, ZoneType::Hand), hand_now);
}

#[test]
fn spirit_tokens_cannot_block_or_be_blocked_by_nonspirits() {
    for spirit_attacking in [false, true] {
        let mut s = five_color_setup(&["Gnarled Professor"]);
        s.game_mut().create_token("Spirit", PlayerId(0), false);
        s.game_mut().create_token("Spirit", PlayerId(1), false);
        let spirit = s.battlefield_permanents_named(0, "Spirit")[0];
        let defender = s.battlefield_permanents_named(1, "Spirit")[0];
        let ordinary = s.force_permanent_on_battlefield(0, "Gnarled Professor");
        let other = s.force_permanent_on_battlefield(1, "Gnarled Professor");
        for id in [spirit, ordinary] {
            s.game_mut().state.permanents[id]
                .as_mut()
                .unwrap()
                .summoning_sick = false;
        }
        s.advance_to_active_step(0, StepKind::BeginningOfCombat);
        s.pass_priority();
        s.pass_priority();
        let attacker = if spirit_attacking { spirit } else { ordinary };
        for _ in 0..2 {
            if s.action_space().kind != ActionSpaceKind::DeclareAttacker {
                break;
            }
            let index = s.action_space().actions.iter().position(|action| matches!(
            action,
            Action::DeclareAttacker { permanent, attack, .. } if (*permanent == attacker) == *attack
        )).unwrap();
            s.step_action(index);
        }
        s.advance_to_active_step(0, StepKind::DeclareBlockers);
        let mut checked = 0;
        while s.action_space().kind == ActionSpaceKind::DeclareBlocker {
            let blocker = s
                .action_space()
                .actions
                .iter()
                .find_map(|a| {
                    if let Action::DeclareBlocker { blocker, .. } = a {
                        Some(*blocker)
                    } else {
                        None
                    }
                })
                .unwrap();
            let allowed =
                s.action_space().actions.iter().any(
                    |a| matches!(a,Action::DeclareBlocker{attacker:Some(id),..} if *id==attacker),
                );
            assert_eq!(
                allowed,
                if spirit_attacking {
                    blocker == defender
                } else {
                    blocker == other
                }
            );
            checked += 1;
            s.decline_block();
        }
        assert_eq!(checked, 2);
    }
}

#[test]
fn suki_power_tracks_creature_count_and_attack_makes_a_token() {
    let mut s = Scenario::new(
        deck(&[
            ("Plains", 12),
            ("Forest", 12),
            ("Suki, Kyoshi Warrior", 8),
            ("Kyoshi Warriors", 8),
        ]),
        deck(&[("Plains", 40)]),
        331,
    );
    let suki = s.force_permanent_on_battlefield(0, "Suki, Kyoshi Warrior");
    // Alone: power = 1 (herself).
    assert_eq!(s.game().effective_pt(suki), (1, 4));
    let _kyoshi = s.force_permanent_on_battlefield(0, "Kyoshi Warriors");
    assert_eq!(s.game().effective_pt(suki), (2, 4));

    clear_summoning_sickness(&mut s, suki);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    // Attack with Suki only.
    let mut attacked = false;
    while s.action_space().kind == ActionSpaceKind::DeclareAttacker {
        let space = s.action_space().clone();
        let attack_suki = space.actions.iter().position(|action| {
        matches!(action, Action::DeclareAttacker { permanent, attack: true, .. } if *permanent == suki)
    });
        match attack_suki {
            Some(index) => {
                s.step_action(index);
                attacked = true;
            }
            None => s.decline_attack(),
        }
    }
    assert!(attacked, "Suki should attack");
    s.pass_priority();
    s.pass_priority(); // attack trigger resolves: tapped-and-attacking token

    let allies = s.battlefield_permanents_named(0, "Ally");
    assert_eq!(allies.len(), 1);
    let token = allies[0];
    assert!(permanent(&s, token).tapped);
    assert!(permanent(&s, token).attacking);
    // CDA recomputed mid-combat: 3 creatures now.
    assert_eq!(s.game().effective_pt(suki), (3, 4));

    s.advance_to_active_step(0, StepKind::EndOfCombat);
    // Suki (3) + token (1) both attacked unblocked.
    s.assert_life(1, 16);
}

#[test]
fn legend_rule_recognizes_existing_lowercase_supertypes() {
    let name = "Suki, Kyoshi Warrior";
    let mut cards = ur_lessons_deck();
    cards.insert(name.into(), 2);
    let mut s = Scenario::new(cards.clone(), cards, 88);
    s.advance_to_active_step(0, StepKind::Main);
    let first = put_onto_battlefield(&mut s, 0, name);
    let second = put_onto_battlefield(&mut s, 0, name);
    let first_card = card_of(&s, first);
    let second_card = card_of(&s, second);
    s.pass_priority();
    assert_eq!(s.action_space().kind, ActionSpaceKind::LegendRule);
    let index = s
        .action_space()
        .actions
        .iter()
        .position(
            |action| matches!(action, Action::SelectCard { card, .. } if *card == second_card),
        )
        .unwrap();
    s.step_action(index);
    assert_eq!(
        s.game().state.zones.zone_of(first_card),
        Some(ZoneType::Graveyard)
    );
    assert_eq!(
        s.game().state.zones.zone_of(second_card),
        Some(ZoneType::Battlefield)
    );
}

#[test]
fn tiger_dillo_needs_another_four_power_creature_to_be_an_attacker() {
    for enabled in [false, true] {
        let mut s = five_color_setup(&["Tiger-Dillo", "Gnarled Professor"]);
        let tiger = s.force_permanent_on_battlefield(0, "Tiger-Dillo");
        s.game_mut().state.permanents[tiger]
            .as_mut()
            .unwrap()
            .summoning_sick = false;
        if enabled {
            let other = s.force_permanent_on_battlefield(0, "Gnarled Professor");
            s.game_mut().state.permanents[other]
                .as_mut()
                .unwrap()
                .summoning_sick = false;
        }
        s.advance_to_active_step(0, StepKind::BeginningOfCombat);
        s.pass_priority();
        s.pass_priority();
        let mut offered = false;
        for _ in 0..3 {
            if s.action_space().kind != ActionSpaceKind::DeclareAttacker {
                break;
            }
            offered |= s.action_space().actions.iter().any(|action| {
                matches!(
                    action,
                    Action::DeclareAttacker { permanent, attack: true, .. } if *permanent == tiger
                )
            });
            s.decline_attack();
        }
        assert_eq!(offered, enabled);
    }
}

#[test]
fn tiger_dillo_blocking_needs_another_four_power_creature() {
    for enabled in [false, true] {
        let mut s = five_color_setup(&["Tiger-Dillo", "Gray Ogre", "Master Piandao"]);
        let attacker = s.force_permanent_on_battlefield(0, "Gray Ogre");
        let tiger = s.force_permanent_on_battlefield(1, "Tiger-Dillo");
        if enabled {
            s.force_permanent_on_battlefield(1, "Master Piandao");
        }
        s.game_mut().state.permanents[attacker]
            .as_mut()
            .unwrap()
            .summoning_sick = false;
        s.advance_to_active_step(0, StepKind::DeclareAttackers);
        s.declare_attack();
        s.advance_to_active_step(0, StepKind::DeclareBlockers);
        let mut offered = false;
        for _ in 0..2 {
            if s.action_space().kind != ActionSpaceKind::DeclareBlocker {
                break;
            }
            offered |= s.action_space().actions.iter().any(|a| matches!(a, Action::DeclareBlocker { blocker, attacker: Some(_), .. } if *blocker == tiger));
            s.decline_block();
        }
        assert_eq!(offered, enabled);
    }
}

#[test]
fn tiger_seal_taps_at_upkeep_and_untaps_on_second_draw() {
    let mut s = Scenario::new(
        deck(&[("Island", 24), ("Tiger-Seal", 16)]),
        deck(&[("Island", 40)]),
        19,
    );
    force_lands(&mut s, 0, "Island", 1);
    s.force_card_in_hand(0, "Tiger-Seal");
    s.advance_to_active_step(0, StepKind::Main);
    cast_and_resolve(&mut s);
    let seal = s.battlefield_permanents_named(0, "Tiger-Seal")[0];
    assert!(!permanent(&s, seal).tapped);

    // Next upkeep of its controller: "At the beginning of your upkeep, tap
    // this creature." (It does not fire on the opponent's upkeep.)
    s.advance_to_active_step(1, StepKind::Draw);
    assert!(
        !permanent(&s, seal).tapped,
        "opponent's upkeep must not tap it"
    );
    s.advance_to_active_step(0, StepKind::Draw);
    assert!(
        permanent(&s, seal).tapped,
        "should be tapped by its upkeep trigger"
    );

    // The draw-step draw was this turn's first card; the second draw untaps.
    assert_eq!(s.game().state.turn.cards_drawn_this_turn[0], 1);
    s.game_mut().draw_cards(PlayerId(0), 1);
    resolve_until_priority(&mut s);
    assert!(!permanent(&s, seal).tapped, "second draw should untap");

    // A third draw the same turn does not trigger again (it stays untapped,
    // and no new trigger goes on the stack).
    s.game_mut().draw_cards(PlayerId(0), 1);
    s.pass_priority();
    assert!(s.game().state.stack_objects.is_empty());
    assert!(s.game().state.pending_triggers.is_empty());
}

#[test]
fn tolls_limits_triggers_while_mayhem_sees_opponent_sacrifices() {
    let mut s = five_color_setup(&["Tolls of War", "Mayhem Devil"]);
    play_and_settle(&mut s, "Tolls of War");
    play_and_settle(&mut s, "Mayhem Devil");
    let clue = permanent_named(&s, "Clue");
    activate_first(&mut s, clue);
    settle_targeting_opponent(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Ally").len(), 1);
    assert_eq!(s.game().state.players[1].life, 19);
    s.game_mut().create_token("Clue", PlayerId(1), false);
    let opponent_clue = s.battlefield_permanents_named(1, "Clue")[0];
    // Give the other player priority through the ordinary pass.
    s.game_mut().scenario_refresh_priority().unwrap();
    s.pass_priority();
    activate_first(&mut s, opponent_clue);
    settle_targeting_opponent(&mut s);
    assert_eq!(s.game().state.players[1].life, 18);
    assert_eq!(s.battlefield_permanents_named(0, "Ally").len(), 1);
}

#[test]
fn tolls_fires_once_in_a_turn_and_resets_on_the_next_turn() {
    let mut s = five_color_setup(&["Tolls of War"]);
    play_and_settle(&mut s, "Tolls of War");
    let first = permanent_named(&s, "Clue");
    activate_first(&mut s, first);
    settle(&mut s);
    s.game_mut().create_token("Clue", PlayerId(0), false);
    let second = permanent_named(&s, "Clue");
    activate_first(&mut s, second);
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Ally").len(), 1);
    s.advance_to_active_step(1, managym::flow::turn::StepKind::Main);
    s.advance_to_active_step(0, managym::flow::turn::StepKind::Main);
    s.set_player_mana_pool(0, "2");
    s.game_mut().create_token("Clue", PlayerId(0), false);
    let third = permanent_named(&s, "Clue");
    activate_first(&mut s, third);
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Ally").len(), 2);
}

#[test]
fn toph_enters_earthbends_and_counts_counters_on_lands() {
    let mut s = five_color_setup(&["Toph, the Blind Bandit"]);
    let land = s.force_permanent_on_battlefield(0, "Forest");
    cast(&mut s, "Toph, the Blind Bandit");
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(land)));
    settle(&mut s);
    assert_eq!(counters(&s, land), 2);
    assert_eq!(
        s.game()
            .effective_power(permanent_named(&s, "Toph, the Blind Bandit")),
        2
    );
}

#[test]
fn treetop_creates_one_white_ally_and_can_attack_immediately() {
    let mut s = five_color_setup(&["Treetop Freedom Fighters"]);
    cast(&mut s, "Treetop Freedom Fighters");
    settle(&mut s);
    let fighter = permanent_named(&s, "Treetop Freedom Fighters");
    let allies = s.battlefield_permanents_named(0, "Ally");
    assert_eq!(allies.len(), 1);
    assert_eq!(s.game().effective_pt(allies[0]), (1, 1));
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    choose(
        &mut s,
        |a| matches!(a,Action::DeclareAttacker { permanent,attack:true,.. } if *permanent==fighter),
    );
    assert!(tapped(&s, fighter));
}

#[test]
fn true_ancestry_returns_only_a_current_permanent_card_and_can_decline() {
    for take in [false, true] {
        let mut s = five_color_setup(&["True Ancestry", "Lightning Strike"]);
        let land = library_card(&s, 0, "Forest");
        let instant = library_card(&s, 0, "Lightning Strike");
        s.game_mut().move_card(land, ZoneType::Graveyard);
        s.game_mut().move_card(instant, ZoneType::Graveyard);
        let object = s.game().current_object_ref(land).unwrap();
        cast(&mut s, "True Ancestry");
        assert_eq!(s.action_space().actions.len(), 2);
        if take {
            assert!(s.choose_target(Target::GraveyardCard(object)));
        } else {
            let index = s
                .action_space()
                .actions
                .iter()
                .position(|a| matches!(a, Action::Decline { .. }))
                .unwrap();
            s.step_action(index);
        }
        settle(&mut s);
        assert_eq!(
            s.game().state.zones.zone_of(land),
            Some(if take {
                ZoneType::Hand
            } else {
                ZoneType::Graveyard
            })
        );
        assert_eq!(s.battlefield_permanents_named(0, "Clue").len(), 1);
    }
}

#[test]
fn ancestry_does_not_follow_a_graveyard_target_into_a_new_incarnation() {
    let mut s = five_color_setup(&["True Ancestry"]);
    let land = library_card(&s, 0, "Forest");
    s.game_mut().move_card(land, ZoneType::Graveyard);
    let old = s.game().current_object_ref(land).unwrap();
    cast(&mut s, "True Ancestry");
    assert!(s.choose_target(Target::GraveyardCard(old)));
    s.game_mut().move_card(land, ZoneType::Exile);
    s.game_mut().move_card(land, ZoneType::Graveyard);
    assert_ne!(s.game().current_object_ref(land), Some(old));
    settle(&mut s);
    assert_eq!(
        s.game().state.zones.zone_of(land),
        Some(ZoneType::Graveyard)
    );
    assert!(
        s.battlefield_permanents_named(0, "Clue").is_empty(),
        "all targets illegal: the entire spell fails to resolve"
    );
}

#[test]
fn uncle_iroh_reduces_lesson_spells_without_reducing_ordinary_spells() {
    let mut s = five_color_setup(&["Uncle Iroh", "Abandon Attachments", "Lightning Strike"]);
    s.force_permanent_on_battlefield(0, "Uncle Iroh");
    let lesson = library_card(&s, 0, "Abandon Attachments");
    let ordinary = library_card(&s, 0, "Lightning Strike");
    s.game_mut().move_card(lesson, ZoneType::Hand);
    s.game_mut().move_card(ordinary, ZoneType::Hand);
    s.set_player_mana_pool(0, "R");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a,Action::CastSpell{card,..} if *card==lesson)));
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a,Action::CastSpell{card,..} if *card==ordinary)));
}

#[test]
fn uncle_iroh_firebending_survives_steps_and_expires_after_combat() {
    let mut s = five_color_setup(&["Uncle Iroh"]);
    let iroh = s.force_permanent_on_battlefield(0, "Uncle Iroh");
    s.game_mut().state.permanents[iroh]
        .as_mut()
        .unwrap()
        .summoning_sick = false;
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    choose(
        &mut s,
        |a| matches!(a, Action::DeclareAttacker { permanent, attack: true, .. } if *permanent == iroh),
    );
    pass_until(&mut s, ActionSpaceKind::Priority);
    settle(&mut s);
    assert_eq!(s.game().state.players[0].combat_mana_pool.total(), 1);
    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    assert_eq!(s.game().state.players[0].combat_mana_pool.total(), 1);
    s.advance_to_active_step(0, StepKind::PostcombatMain);
    assert_eq!(s.game().state.players[0].combat_mana_pool.total(), 0);
}

#[test]
fn merchant_search_returns_the_exact_source_even_on_failure_to_find() {
    for search in [false, true] {
        let mut s = five_color_setup(&["Unlucky Cabbage Merchant"]);
        let merchant = play_and_settle(&mut s, "Unlucky Cabbage Merchant");
        let food = permanent_named(&s, "Food");
        activate_first(&mut s, food);
        pass_until(&mut s, ActionSpaceKind::Modal);
        choose(
            &mut s,
            |a| matches!(a,Action::ChooseMode {mode,..} if *mode==usize::from(!search)),
        );
        if search {
            assert_eq!(s.action_space().kind, ActionSpaceKind::LookAndSelect);
            let owner = managym::agent::observation::Observation::for_player(s.game(), PlayerId(0));
            let other = managym::agent::observation::Observation::for_player(s.game(), PlayerId(1));
            assert!(owner
                .agent_cards
                .iter()
                .any(|c| c.zone == ZoneType::Library));
            assert!(other
                .opponent_cards
                .iter()
                .all(|c| c.zone != ZoneType::Library));
            choose(&mut s, |a| matches!(a, Action::Decline { .. }));
        }
        settle(&mut s);
        assert_eq!(
            s.game().state.zones.zone_of(merchant),
            Some(if search {
                ZoneType::Library
            } else {
                ZoneType::Battlefield
            })
        );
        assert_eq!(s.game().state.players[0].life, 23);
    }
}

#[test]
fn merchant_returns_to_its_owners_bottom_while_its_controller_shuffles() {
    let mut s = five_color_setup(&["Unlucky Cabbage Merchant"]);
    let source = s.force_permanent_on_battlefield(1, "Unlucky Cabbage Merchant");
    let merchant = s.game().state.permanents[source].as_ref().unwrap().card;
    // Model an earlier control change without changing physical ownership.
    s.game_mut().state.permanents[source]
        .as_mut()
        .unwrap()
        .controller = PlayerId(0);
    s.game_mut().create_token("Food", PlayerId(0), false);
    let food = permanent_named(&s, "Food");
    activate_first(&mut s, food);
    pass_until(&mut s, ActionSpaceKind::Modal);
    choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 0, .. }));
    choose(&mut s, |a| matches!(a, Action::Decline { .. }));
    settle(&mut s);
    assert_eq!(
        s.game()
            .state
            .zones
            .zone_cards(ZoneType::Library, PlayerId(1))
            .first(),
        Some(&merchant)
    );
}

#[test]
fn merchant_successful_search_is_optional_and_returns_source_before_shuffle() {
    let mut s = five_color_setup(&["Unlucky Cabbage Merchant"]);
    cast(&mut s, "Unlucky Cabbage Merchant");
    settle(&mut s);
    let merchant = permanent_named(&s, "Unlucky Cabbage Merchant");
    let merchant_card = s.game().state.permanents[merchant].as_ref().unwrap().card;
    let food = permanent_named(&s, "Food");
    activate(&mut s, food, 0);
    pass_until(&mut s, ActionSpaceKind::Modal);
    choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 0, .. }));
    let forest = library_card(&s, 0, "Forest");
    choose(
        &mut s,
        |a| matches!(a, Action::SelectCard { card, .. } if *card == forest),
    );
    settle(&mut s);
    assert_eq!(
        s.game().state.zones.zone_of(merchant_card),
        Some(ZoneType::Library)
    );
    let land = s.game().state.card_to_permanent[forest].unwrap();
    assert!(tapped(&s, land));
    assert_eq!(s.life(0), 23);
    assert!(s.battlefield_permanents_named(0, "Food").is_empty());
}

#[test]
fn walltop_intervening_lesson_condition_is_checked_at_death_and_resolution() {
    for lesson_at_death in [false, true] {
        for remove_lesson in [false, true] {
            let mut s = five_color_setup(&["Walltop Sentries", "Firebending Lesson"]);
            let sentries = s.force_permanent_on_battlefield(0, "Walltop Sentries");
            let key = keywords(&s, sentries);
            assert!(key.reach && key.deathtouch);
            let lesson = library_card(&s, 0, "Firebending Lesson");
            if lesson_at_death {
                s.game_mut().move_card(lesson, ZoneType::Graveyard);
            }
            let before = s.life(0);
            let source = s.game().state.permanents[sentries].as_ref().unwrap().card;
            s.game_mut().move_card(source, ZoneType::Graveyard);
            s.game_mut().scenario_refresh_priority().unwrap();
            // No retroactive trigger if the Lesson arrives after death. An
            // intervening-if trigger also checks again when it resolves.
            s.game_mut().move_card(
                lesson,
                if remove_lesson {
                    ZoneType::Exile
                } else {
                    ZoneType::Graveyard
                },
            );
            if !s.game().state.pending_triggers.is_empty() {
                s.pass_priority();
            }
            settle(&mut s);
            assert_eq!(
                s.life(0),
                before
                    + if lesson_at_death && !remove_lesson {
                        2
                    } else {
                        0
                    }
            );
        }
    }
}

#[test]
fn wandering_musicians_pump_all_own_creatures_for_the_turn() {
    let mut s = five_color_setup(&["Wandering Musicians", "Gray Ogre"]);
    let band = s.force_permanent_on_battlefield(0, "Wandering Musicians");
    let own = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let other = s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.game_mut().state.permanents[band]
        .as_mut()
        .unwrap()
        .summoning_sick = false;
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    choose(
        &mut s,
        |a| matches!(a,Action::DeclareAttacker{permanent,attack:true,..} if *permanent==band),
    );
    // Decline the remaining fixture creature's attack.
    if s.action_space().kind == ActionSpaceKind::DeclareAttacker {
        choose(&mut s, |a| {
            matches!(a, Action::DeclareAttacker { attack: false, .. })
        });
    }
    pass_until(&mut s, ActionSpaceKind::Priority);
    settle(&mut s);
    assert_eq!(s.game().effective_pt(band), (3, 5));
    assert_eq!(s.game().effective_power(own), 3);
    assert_eq!(s.game().effective_power(other), 2);
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.game().effective_power(own), 2);
}

#[test]
fn wartime_protestors_haste_grant_is_temporary_and_only_for_other_own_allies() {
    let mut s = five_color_setup(&["Wartime Protestors", "Gray Ogre", "Pretending Poxbearers"]);
    cast(&mut s, "Wartime Protestors");
    settle(&mut s);
    let protestors = permanent_named(&s, "Wartime Protestors");
    assert_eq!(counters(&s, protestors), 0);
    cast(&mut s, "Gray Ogre");
    settle(&mut s);
    let nonally = permanent_named(&s, "Gray Ogre");
    assert_eq!(counters(&s, nonally), 0);
    cast(&mut s, "Pretending Poxbearers");
    settle(&mut s);
    let ally = permanent_named(&s, "Pretending Poxbearers");
    assert_eq!(counters(&s, ally), 1);
    assert!(keywords(&s, ally).haste);
    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    let mut attackers = Vec::new();
    for _ in 0..3 {
        if s.action_space().kind != ActionSpaceKind::DeclareAttacker {
            break;
        }
        for action in &s.action_space().actions {
            if let Action::DeclareAttacker {
                permanent,
                attack: true,
                ..
            } = action
            {
                attackers.push(*permanent);
            }
        }
        s.decline_attack();
    }
    assert!(attackers.contains(&protestors));
    assert!(attackers.contains(&ally));
    assert!(!attackers.contains(&nonally));
    s.advance_to_active_step(1, StepKind::Main);
    assert!(!keywords(&s, ally).haste);
    assert!(keywords(&s, protestors).haste);
    assert_eq!(counters(&s, ally), 1);
}

#[test]
fn wartime_granted_haste_enables_tap_cost_without_erasing_summoning_sickness() {
    let mut s = five_color_setup(&["Wartime Protestors", "Deserter's Disciple", "Gray Ogre"]);
    s.force_permanent_on_battlefield(0, "Wartime Protestors");
    let target = s.force_permanent_on_battlefield(0, "Gray Ogre");
    cast(&mut s, "Deserter's Disciple");
    settle(&mut s);
    let disciple = permanent_named(&s, "Deserter's Disciple");
    activate(&mut s, disciple, 0);
    assert!(s.choose_target(Target::Permanent(target)));
    settle(&mut s);
    assert!(tapped(&s, disciple));
    assert!(
        s.game().state.permanents[disciple]
            .as_ref()
            .unwrap()
            .summoning_sick
    );
    s.advance_to_active_step(1, StepKind::Main);
    // Model an untap after the grant expires; age has not changed on an
    // opponent's turn, so the tap-symbol ability must again be unavailable.
    s.game_mut().state.permanents[disciple]
        .as_mut()
        .unwrap()
        .tapped = false;
    s.game_mut().scenario_refresh_priority().unwrap();
    s.pass_priority();
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == disciple)));
}

#[test]
fn water_tribe_captain_pays_five_and_buffs_only_own_creatures_until_cleanup() {
    let mut s = five_color_setup(&["Water Tribe Captain", "Gray Ogre"]);
    let captain = s.force_permanent_on_battlefield(0, "Water Tribe Captain");
    let own = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let other = s.force_permanent_on_battlefield(1, "Gray Ogre");
    s.set_player_mana_pool(0, "4");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == captain)));
    s.set_player_mana_pool(0, "5");
    activate(&mut s, captain, 0);
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
    settle(&mut s);
    assert_eq!(s.game().effective_pt(captain), (4, 4));
    assert_eq!(s.game().effective_pt(own), (3, 3));
    assert_eq!(s.game().effective_power(other), 2);
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.game().effective_power(captain), 3);
    assert_eq!(s.game().effective_power(own), 2);
}

fn rallier_deck() -> BTreeMap<String, usize> {
    deck(&[
        ("Forest", 20),
        ("Water Tribe Rallier", 12),
        ("Badgermole Cub", 8),
    ])
}

#[test]
fn waterbend_pays_entirely_with_creature_taps() {
    let mut s = Scenario::new(rallier_deck(), deck(&[("Plains", 40)]), 81);
    let rallier = s.force_permanent_on_battlefield(0, "Water Tribe Rallier");
    let mut creatures = vec![rallier];
    for _ in 0..4 {
        creatures.push(s.force_permanent_on_battlefield(0, "Water Tribe Rallier"));
    }
    clear_hand(&mut s, 0);
    s.advance_to_active_step(0, StepKind::Main);

    // No lands — the ability is affordable only through waterbend.
    assert!(s.take_action_by_type(ActionType::PriorityActivateAbility));
    for expected_remaining in (0..5).rev() {
        let space = s.action_space().clone();
        assert_eq!(space.kind, ActionSpaceKind::Waterbend);
        // No mana available: only tap actions.
        assert!(space
            .actions
            .iter()
            .all(|action| action.action_type() == ActionType::TapForCost));
        assert!(s.take_action_by_type(ActionType::TapForCost));
        if expected_remaining > 0 {
            assert_eq!(s.action_space().kind, ActionSpaceKind::Waterbend);
        }
    }
    // All five creatures tapped; ability on the stack.
    for creature in &creatures {
        assert!(permanent(&s, *creature).tapped);
    }
    assert_eq!(s.game().state.stack_objects.len(), 1);
    s.pass_priority();
    s.pass_priority();

    // Look at the top four, may put a creature with power <= 3 into hand.
    let space = s.action_space().clone();
    assert_eq!(space.kind, ActionSpaceKind::LookAndSelect);
    assert_eq!(
        space.actions.last().map(Action::action_type),
        Some(ActionType::DeclineChoice),
        "the reveal is optional"
    );
    let top4: Vec<CardId> = library(&s, 0).iter().rev().take(4).copied().collect();
    // Selectable cards are exactly the creatures with power <= 3 among them.
    let selectable = space
        .actions
        .iter()
        .filter(|action| action.action_type() == ActionType::SelectCard)
        .count();
    let matching = top4
        .iter()
        .filter(|card| {
            let card = &s.game().state.cards[**card];
            card.types.is_creature() && card.power.unwrap_or(0) <= 3
        })
        .count();
    assert_eq!(selectable, matching);

    if selectable > 0 {
        let hand_before = hand(&s, 0).len();
        assert!(s.take_action_by_type(ActionType::SelectCard));
        assert_eq!(hand(&s, 0).len(), hand_before + 1);
    } else {
        assert!(s.take_action_by_type(ActionType::DeclineChoice));
    }
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);
}

#[test]
fn waterbend_taps_do_not_trigger_badgermole_cub() {
    let mut s = Scenario::new(rallier_deck(), deck(&[("Plains", 40)]), 82);
    let rallier = s.force_permanent_on_battlefield(0, "Water Tribe Rallier");
    let cub = s.force_permanent_on_battlefield(0, "Badgermole Cub");
    force_lands(&mut s, 0, "Forest", 3);
    clear_hand(&mut s, 0);
    s.advance_to_active_step(0, StepKind::Main);

    assert!(s.take_action_by_type(ActionType::PriorityActivateAbility));
    assert_eq!(s.action_space().kind, ActionSpaceKind::Waterbend);
    // Two creature taps pay {2} without producing mana.
    assert!(s.choose_waterbend_tap(rallier));
    assert!(s.choose_waterbend_tap(cub));
    assert_eq!(
        s.game().state.players[0].mana_pool.total(),
        0,
        "waterbend is not tapping for mana"
    );
    // All three forests are still needed for the remaining {3}.
    assert!(s.take_action_by_type(ActionType::PayCost));
    let untapped_forests = s
        .battlefield_permanents_named(0, "Forest")
        .iter()
        .filter(|id| !permanent(&s, **id).tapped)
        .count();
    assert_eq!(untapped_forests, 0);
    assert_eq!(s.game().state.stack_objects.len(), 1);
}

#[test]
fn rallier_reveals_only_the_selected_card_and_can_decline() {
    for take in [false, true] {
        let mut s = five_color_setup(&["Water Tribe Rallier", "Gray Ogre", "Wall of Stone"]);
        let rallier = s.force_permanent_on_battlefield(0, "Water Tribe Rallier");
        let top = pin_top(&mut s, &["Forest", "Gray Ogre", "Wall of Stone", "Island"]);
        s.set_player_mana_pool(0, "5");
        activate(&mut s, rallier, 0);
        // Paying in mana is a deliberate waterbend choice.
        choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
        assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
        pass_until(&mut s, ActionSpaceKind::LookAndSelect);
        assert!(s
            .action_space()
            .actions
            .iter()
            .any(|a| matches!(a, Action::SelectCard { card, .. } if *card == top[1])));
        assert!(!s
            .action_space()
            .actions
            .iter()
            .any(|a| matches!(a, Action::SelectCard { card, .. } if *card == top[0])));
        let other = Observation::for_player(s.game(), PlayerId(1));
        assert!(other
            .opponent_cards
            .iter()
            .all(|c| c.zone != ZoneType::Library));
        choose(&mut s, |a| {
            if take {
                matches!(a, Action::SelectCard { card, .. } if *card == top[1])
            } else {
                matches!(a, Action::Decline { .. })
            }
        });
        settle(&mut s);
        let other = Observation::for_player(s.game(), PlayerId(1));
        let definition = s.game().state.cards[top[1]].definition_id.0;
        assert_eq!(
            other
                .opponent
                .known_hand
                .get(&definition)
                .copied()
                .unwrap_or(0),
            u32::from(take)
        );
        assert_eq!(s.zone_size(0, ZoneType::Hand), usize::from(take));
        let library = s
            .game()
            .state
            .zones
            .zone_cards(ZoneType::Library, PlayerId(0));
        for id in &top {
            if !take || *id != top[1] {
                assert!(library[..4 - usize::from(take)].contains(id));
            }
        }
    }
}

/// Cast Waterbending Lesson with `names` in play, refill the pool, and stop at
/// its optional waterbend payment after it drew three cards.
fn lesson_payment(names: &[&str], pool: &str) -> (Scenario, CardId, usize) {
    let mut s = five_color_setup(names);
    for name in names.iter().filter(|name| **name != "Waterbending Lesson") {
        enter_and_settle(&mut s, name);
        if *name == "Great Divide Guide" {
            ready(&mut s, name);
        }
    }
    let before = s.zone_size(0, ZoneType::Hand);
    let spell = cast(&mut s, "Waterbending Lesson");
    if s.action_space().kind == ActionSpaceKind::PayOrNot {
        choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    }
    s.set_player_mana_pool(0, pool);
    pass_until(&mut s, ActionSpaceKind::Waterbend);
    assert_eq!(s.zone_size(0, ZoneType::Hand), before + 3);
    (s, spell, before)
}

#[test]
fn waterbend_lesson_draws_before_optional_payment_and_taps_without_mana() {
    let mut s = five_color_setup(&[
        "Waterbending Lesson",
        "Hermitic Herbalist",
        "Prophetic Prism",
    ]);
    enter_and_settle(&mut s, "Hermitic Herbalist");
    enter_and_settle(&mut s, "Prophetic Prism");
    let before = s.game().state.zones.size(ZoneType::Hand, PlayerId(0));
    let spell = cast(&mut s, "Waterbending Lesson");
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    pass_until(&mut s, ActionSpaceKind::Waterbend);
    assert_eq!(
        s.game().state.zones.size(ZoneType::Hand, PlayerId(0)),
        before + 3
    );
    assert_choice_contract(&s);
    for name in ["Hermitic Herbalist", "Prophetic Prism"] {
        let source = permanent_named(&s, name);
        choose(
            &mut s,
            |a| matches!(a, Action::WaterbendTap { permanent, .. } if *permanent == source),
        );
    }
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    assert_eq!(
        s.game().state.zones.zone_of(spell),
        Some(ZoneType::Graveyard)
    );
    assert_eq!(
        s.game().state.zones.size(ZoneType::Hand, PlayerId(0)),
        before + 3
    );
}

#[test]
fn waterbend_cannot_partially_pay_then_discard() {
    let mut s = five_color_setup(&["Waterbending Lesson", "Prophetic Prism"]);
    enter_and_settle(&mut s, "Prophetic Prism");
    cast(&mut s, "Waterbending Lesson");
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    pass_until(&mut s, ActionSpaceKind::Waterbend);
    s.set_player_mana_pool(0, "1");
    let prism = permanent_named(&s, "Prophetic Prism");
    choose(
        &mut s,
        |a| matches!(a, Action::WaterbendTap { permanent, .. } if *permanent == prism),
    );
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::Decline { .. })));
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
}

#[test]
fn optional_lesson_all_mana_mixed_and_all_taps_keep_the_same_continuation() {
    for taps in 0..=2 {
        let (mut s, spell, before) = lesson_payment(
            &[
                "Waterbending Lesson",
                "Prophetic Prism",
                "Hermitic Herbalist",
            ],
            ["2", "1", "0"][taps],
        );
        assert_choice_contract(&s);
        for name in ["Prophetic Prism", "Hermitic Herbalist"].iter().take(taps) {
            let source = permanent_named(&s, name);
            choose(
                &mut s,
                |a| matches!(a, Action::WaterbendTap { permanent, .. } if *permanent == source),
            );
            assert_choice_contract(&s);
        }
        choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
        assert_eq!(s.zone_size(0, ZoneType::Hand), before + 3);
        assert_eq!(
            s.game().state.zones.zone_of(spell),
            Some(ZoneType::Graveyard)
        );
        assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
    }
}

#[test]
fn optional_lesson_without_resources_only_declines_then_discards_one_drawn_card() {
    let (mut s, spell, before) = lesson_payment(&["Waterbending Lesson"], "0");
    assert_eq!(
        s.action_space().actions,
        vec![Action::Decline {
            player: PlayerId(0)
        }]
    );
    choose(&mut s, |a| matches!(a, Action::Decline { .. }));
    assert_eq!(s.action_space().kind, ActionSpaceKind::Discard);
    assert_choice_contract(&s);
    let other = managym::agent::observation::Observation::for_player(s.game(), PlayerId(1));
    assert!(!other
        .opponent_cards
        .iter()
        .any(|c| c.zone == ZoneType::Hand));
    choose(&mut s, |a| matches!(a, Action::SelectCard { .. }));
    assert_eq!(s.zone_size(0, ZoneType::Hand), before + 2);
    assert_eq!(
        s.game().state.zones.zone_of(spell),
        Some(ZoneType::Graveyard)
    );
}

#[test]
fn optional_lesson_does_not_offer_a_tap_without_a_complete_payment() {
    let (s, _, _) = lesson_payment(&["Waterbending Lesson", "Prophetic Prism"], "0");
    assert_eq!(
        s.action_space().actions,
        vec![Action::Decline {
            player: PlayerId(0)
        }]
    );
}

#[test]
fn committed_optional_payment_resumes_after_granted_chosen_color_mana() {
    let (mut s, spell, before) = lesson_payment(
        &[
            "Waterbending Lesson",
            "Great Divide Guide",
            "Prophetic Prism",
        ],
        "0",
    );
    let prism = permanent_named(&s, "Prophetic Prism");
    choose(
        &mut s,
        |a| matches!(a, Action::WaterbendTap { permanent, .. } if *permanent == prism),
    );
    activate_named(&mut s, "Great Divide Guide", 0);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Modal);
    assert_choice_contract(&s);
    choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 4, .. }));
    assert_eq!(s.action_space().kind, ActionSpaceKind::Waterbend);
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::Decline { .. })));
    assert_choice_contract(&s);
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    assert_eq!(s.zone_size(0, ZoneType::Hand), before + 3);
    assert_eq!(
        s.game().state.zones.zone_of(spell),
        Some(ZoneType::Graveyard)
    );
    for name in ["Prophetic Prism", "Great Divide Guide"] {
        assert!(
            s.game().state.permanents[permanent_named(&s, name)]
                .as_ref()
                .unwrap()
                .tapped
        );
    }
}

#[test]
fn optional_decline_retains_mana_activation_costs_and_chosen_output() {
    let (mut s, _, before) = lesson_payment(&["Waterbending Lesson", "Prophetic Prism"], "1");
    activate_named(&mut s, "Prophetic Prism", 0);
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
    assert_choice_contract(&s);
    choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 1, .. }));
    choose(&mut s, |a| matches!(a, Action::Decline { .. }));
    choose(&mut s, |a| matches!(a, Action::SelectCard { .. }));
    assert_eq!(s.zone_size(0, ZoneType::Hand), before + 2);
    assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("U"));
    assert!(
        s.game().state.permanents[permanent_named(&s, "Prophetic Prism")]
            .as_ref()
            .unwrap()
            .tapped
    );
}

#[test]
fn waterbend_and_mana_taps_have_distinct_bonus_accounting() {
    use managym::state::{card::TriggeredManaAbility, predicate::CardPredicate};

    for with_artifact in [false, true] {
        let names = if with_artifact {
            vec![
                "Waterbending Lesson",
                "Hermitic Herbalist",
                "Prophetic Prism",
            ]
        } else {
            vec!["Waterbending Lesson", "Hermitic Herbalist"]
        };
        let mut s = five_color_setup(&names);
        for name in names.iter().filter(|name| **name != "Waterbending Lesson") {
            enter_and_settle(&mut s, name);
        }
        ready(&mut s, "Hermitic Herbalist");
        let herbalist = permanent_named(&s, "Hermitic Herbalist");
        let source_card = s.game().state.permanents[herbalist].as_ref().unwrap().card;
        s.game_mut().state.cards[source_card]
            .triggered_mana_abilities
            .push(TriggeredManaAbility {
                predicate: CardPredicate::creature(),
                mana: Mana::parse("G"),
            });
        cast(&mut s, "Waterbending Lesson");
        if s.action_space().kind == ActionSpaceKind::PayOrNot {
            choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
        }
        s.set_player_mana_pool(0, "0");
        pass_until(&mut s, ActionSpaceKind::Waterbend);
        assert_eq!(
            s.action_space().actions.iter().any(
                |a| matches!(a, Action::WaterbendTap { permanent, .. } if *permanent == herbalist)
            ),
            with_artifact
        );
        if with_artifact {
            choose(
                &mut s,
                |a| matches!(a, Action::WaterbendTap { permanent, .. } if *permanent == herbalist),
            );
            assert_eq!(
                s.game().state.players[0].mana_pool.total(),
                0,
                "waterbend never fires a mana bonus"
            );
            let prism = permanent_named(&s, "Prophetic Prism");
            assert!(!s.action_space().actions.iter().any(
                |a| matches!(a, Action::ActivateAbility { permanent, .. } if *permanent == prism)
            ));
            choose(
                &mut s,
                |a| matches!(a, Action::WaterbendTap { permanent, .. } if *permanent == prism),
            );
        } else {
            activate_named(&mut s, "Hermitic Herbalist", 0);
            assert_eq!(
                s.game().state.players[0].mana_pool,
                Mana::parse("G"),
                "exactly one immediate tap bonus"
            );
            choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 1, .. }));
            assert_eq!(s.game().state.players[0].mana_pool, Mana::parse("GU"));
        }
        choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
        assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
    }
}

#[test]
fn committed_optional_payment_resumes_after_two_nested_mana_decisions() {
    let (mut s, lesson, drawn) = lesson_payment(
        &[
            "Waterbending Lesson",
            "Prophetic Prism",
            "Callous Inspector",
            "Tireless Provisioner",
        ],
        "0",
    );
    s.game_mut().create_token("Treasure", PlayerId(0), false);
    // Inject the extra resource before payment begins. The initial waterbend
    // offer is already legal from the creatures and Prism; its normal tap
    // transition publishes the offers that include this Treasure.
    let creature = permanent_named(&s, "Callous Inspector");
    choose(
        &mut s,
        |a| matches!(a, Action::WaterbendTap { permanent, .. } if *permanent == creature),
    );
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::Decline { .. })));
    activate_named(&mut s, "Prophetic Prism", 0);
    activate_named(&mut s, "Treasure", 0);
    assert_eq!(
        s.game().state.stack_objects.len(),
        1,
        "only the resolving Lesson is on the stack"
    );
    choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 1, .. }));
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
    choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 4, .. }));
    assert_eq!(s.action_space().kind, ActionSpaceKind::Waterbend);
    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::Decline { .. })));
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    assert_eq!(s.zone_size(0, ZoneType::Hand), drawn + 3);
    assert_eq!(
        s.game().state.zones.zone_of(lesson),
        Some(ZoneType::Graveyard)
    );
    assert_eq!(s.game().state.players[0].mana_pool.total(), 0);
}

/// "Costs {1} less for each Island you control" applies to the generic part
/// as the ability is activated.
#[test]
fn waterbending_scroll_draws_for_one_less_per_island() {
    let mut s = five_color_setup(&["Waterbending Scroll"]);
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
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_activate(&s, scroll));
    s.set_player_mana_pool(0, "2");
    let hand = s.zone_size(0, ZoneType::Hand);
    activate(&mut s, scroll, 0);
    assert_eq!(pool(&s).total(), 0);
    settle(&mut s);
    assert_eq!(s.zone_size(0, ZoneType::Hand), hand + 1);
    assert!(s.game().state.permanents[scroll].as_ref().unwrap().tapped);
}

#[test]
fn white_lotus_hideout_restricts_to_lessons_and_shrines_and_filters_for_one() {
    let lesson_or_shrine = ManaUse::spell(["Lesson", "Shrine"]);
    let mut s = five_color_setup(&[
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
    choose_mode(&mut s, 3);
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
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_cast(&s, lesson));
    assert!(!can_cast(&s, strike));

    let mut s = five_color_setup(&["White Lotus Hideout"]);
    let hideout = s.force_permanent_on_battlefield(0, "White Lotus Hideout");
    s.set_player_mana_pool(0, "C");
    activate(&mut s, hideout, 1);
    choose_mode(&mut s, 1);
    assert_eq!(pool(&s).mana, [0, 1, 0, 0, 0, 0]);
    assert!(pool(&s).restricted().is_empty());
}

#[test]
fn anthem_buffs_other_allies_only() {
    let mut s = Scenario::new(
        deck(&[
            ("Plains", 16),
            ("White Lotus Reinforcements", 8),
            ("Kyoshi Warriors", 8),
            ("Otter-Penguin", 8),
        ]),
        deck(&[("Plains", 24), ("Kyoshi Warriors", 16)]),
        321,
    );
    let anthem = s.force_permanent_on_battlefield(0, "White Lotus Reinforcements");
    let kyoshi = s.force_permanent_on_battlefield(0, "Kyoshi Warriors");
    let penguin = s.force_permanent_on_battlefield(0, "Otter-Penguin");
    let enemy_ally = s.force_permanent_on_battlefield(1, "Kyoshi Warriors");

    // Other Ally you control: 3/3 -> 4/4.
    assert_eq!(s.game().effective_pt(kyoshi), (4, 4));
    // Not itself ("other").
    assert_eq!(s.game().effective_pt(anthem), (2, 3));
    // Not a non-Ally.
    assert_eq!(s.game().effective_pt(penguin), (2, 1));
    // Not an opponent's Ally.
    assert_eq!(s.game().effective_pt(enemy_ally), (3, 3));
}

#[test]
fn two_anthems_buff_each_other_and_stack() {
    let mut s = Scenario::new(
        deck(&[
            ("Plains", 16),
            ("White Lotus Reinforcements", 12),
            ("Kyoshi Warriors", 12),
        ]),
        deck(&[("Plains", 40)]),
        322,
    );
    let a1 = s.force_permanent_on_battlefield(0, "White Lotus Reinforcements");
    let a2 = s.force_permanent_on_battlefield(0, "White Lotus Reinforcements");
    let kyoshi = s.force_permanent_on_battlefield(0, "Kyoshi Warriors");

    // Each anthem is an "other Ally" for the other one.
    assert_eq!(s.game().effective_pt(a1), (3, 4));
    assert_eq!(s.game().effective_pt(a2), (3, 4));
    // The Kyoshi gets both anthems: 3/3 -> 5/5.
    assert_eq!(s.game().effective_pt(kyoshi), (5, 5));
}

#[test]
fn layers_compose_counters_anthem_and_until_eot() {
    let mut s = Scenario::new(
        deck(&[
            ("Plains", 12),
            ("White Lotus Reinforcements", 8),
            ("Kyoshi Warriors", 8),
            ("Yip Yip!", 12),
        ]),
        deck(&[("Plains", 40)]),
        324,
    );
    let _anthem = s.force_permanent_on_battlefield(0, "White Lotus Reinforcements");
    let kyoshi = s.force_permanent_on_battlefield(0, "Kyoshi Warriors");
    s.game_mut().state.permanents[kyoshi]
        .as_mut()
        .expect("permanent")
        .plus1_counters = 1;
    force_lands(&mut s, 0, "Plains", 1);
    clear_hand(&mut s, 0);
    s.force_card_in_hand(0, "Yip Yip!");
    s.advance_to_active_step(0, StepKind::Main);

    cast_only(&mut s);
    assert!(s.choose_target(Target::Permanent(kyoshi)));
    s.pass_priority();
    s.pass_priority();

    // 3/3 base + anthem (+1/+1) + counter (+1/+1) + Yip Yip (+2/+2) = 7/7.
    assert_eq!(s.game().effective_pt(kyoshi), (7, 7));
    // Until-EOT part expires at cleanup; statics and counters persist.
    s.advance_to_active_step(1, StepKind::Upkeep);
    assert_eq!(s.game().effective_pt(kyoshi), (5, 5));
}

#[test]
fn yip_yip_buffs_and_grants_flying_to_allies_only() {
    let mut s = Scenario::new(
        deck(&[
            ("Plains", 16),
            ("Yip Yip!", 12),
            ("Kyoshi Warriors", 6),
            ("Otter-Penguin", 6),
        ]),
        deck(&[("Plains", 40)]),
        371,
    );
    let kyoshi = s.force_permanent_on_battlefield(0, "Kyoshi Warriors");
    let penguin = s.force_permanent_on_battlefield(0, "Otter-Penguin");
    force_lands(&mut s, 0, "Plains", 2);
    clear_hand(&mut s, 0);
    force_new_card_in_hand(&mut s, 0, "Yip Yip!");
    force_new_card_in_hand(&mut s, 0, "Yip Yip!");
    s.advance_to_active_step(0, StepKind::Main);

    // On the Ally: +2/+2 and flying.
    cast_only(&mut s);
    assert!(s.choose_target(Target::Permanent(kyoshi)));
    s.pass_priority();
    s.pass_priority();
    assert_eq!(s.game().effective_pt(kyoshi), (5, 5));
    assert!(permanent(&s, kyoshi).temp_keywords.flying);

    // The granted keyword is agent-visible: the permanent entry carries
    // effective keywords (flying), while the card entry keeps the printed
    // keywords (no flying on Kyoshi Warriors).
    let obs = Observation::new(s.game(), &[]);
    let kyoshi_object_id = permanent(&s, kyoshi).id.0 as i32;
    let kyoshi_perm = obs
        .agent_permanents
        .iter()
        .chain(obs.opponent_permanents.iter())
        .find(|perm| perm.id == kyoshi_object_id)
        .expect("Kyoshi Warriors permanent should be observable");
    assert!(
        kyoshi_perm.keywords.flying,
        "granted flying must be encoded"
    );
    assert!(!kyoshi_perm.keywords.hexproof);
    let kyoshi_card = obs
        .agent_cards
        .iter()
        .chain(obs.opponent_cards.iter())
        .find(|card| card.name == "Kyoshi Warriors")
        .expect("Kyoshi Warriors card should be observable");
    assert!(
        !kyoshi_card.keywords.flying,
        "card entry keeps printed keywords"
    );

    // On the non-Ally: +2/+2 but no flying.
    cast_only(&mut s);
    assert!(s.choose_target(Target::Permanent(penguin)));
    s.pass_priority();
    s.pass_priority();
    assert_eq!(s.game().effective_pt(penguin), (4, 3));
    assert!(!permanent(&s, penguin).temp_keywords.flying);

    // Both wear off at cleanup.
    s.advance_to_active_step(1, StepKind::Upkeep);
    assert_eq!(s.game().effective_pt(kyoshi), (3, 3));
    assert!(!permanent(&s, kyoshi).temp_keywords.flying);
}

/// Yue's ability offers only noncreature spells from hand; the chosen spell
/// is cast for free after the ability finishes resolving.
#[test]
fn yue_casts_a_noncreature_spell_from_hand_for_free() {
    let mut s = five_color_setup(&["Yue, the Moon Spirit", "Lightning Strike", "Gray Ogre"]);
    let yue = s.force_permanent_on_battlefield(0, "Yue, the Moon Spirit");
    clear_summoning_sickness(&mut s, yue);
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

#[test]
fn yuyan_rummage_may_be_declined_or_discard_then_draw() {
    for discard in [false, true] {
        let mut s = five_color_setup(&["Yuyan Archers", "Gray Ogre"]);
        let chosen = library_card(&s, 0, "Gray Ogre");
        s.game_mut().move_card(chosen, ZoneType::Hand);
        cast(&mut s, "Yuyan Archers");
        pass_until(&mut s, ActionSpaceKind::Discard);
        let top = *s
            .game()
            .state
            .zones
            .zone_cards(ZoneType::Library, PlayerId(0))
            .last()
            .unwrap();
        if discard {
            choose(
                &mut s,
                |a| matches!(a,Action::SelectCard { card,.. } if *card==chosen),
            );
        } else {
            choose(&mut s, |a| matches!(a, Action::Decline { .. }));
        }
        settle(&mut s);
        assert_eq!(
            s.game().state.zones.zone_of(chosen),
            Some(if discard {
                ZoneType::Graveyard
            } else {
                ZoneType::Hand
            })
        );
        assert_eq!(
            s.game().state.zones.zone_of(top),
            Some(if discard {
                ZoneType::Hand
            } else {
                ZoneType::Library
            })
        );
        assert!(keywords(&s, permanent_named(&s, "Yuyan Archers")).reach);
    }
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
    let mut s = five_color_setup(&["Zuko, Exiled Prince", "Gray Ogre"]);
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
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_cast(&s, ogre));
    s.set_player_mana_pool(0, "2");
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(can_cast(&s, ogre));

    // Unplayed, the card stays exiled without permission after the turn.
    s.advance_to_active_step(1, StepKind::Main);
    s.advance_to_active_step(0, StepKind::Main);
    s.set_player_mana_pool(0, "20RRRR");
    assert_eq!(s.game().state.zones.zone_of(ogre), Some(ZoneType::Exile));
    s.game_mut().scenario_refresh_priority().unwrap();
    assert!(!can_cast(&s, ogre));
}

/// Zuko's Conviction returns a creature card from your graveyard to hand, or,
/// kicked, puts it onto the battlefield tapped. Noncreature cards are not
/// targets.
#[test]
fn zukos_conviction_returns_a_creature_to_hand_or_kicked_onto_the_battlefield() {
    for kicked in [false, true] {
        let mut s = five_color_setup(&["Zuko's Conviction", "Gray Ogre", "Lightning Strike"]);
        let ogre = move_from_library(&mut s, 0, "Gray Ogre", ZoneType::Graveyard);
        let strike = move_from_library(&mut s, 0, "Lightning Strike", ZoneType::Graveyard);
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

#[test]
fn zuko_exiles_and_gives_clue_to_the_targets_current_controller() {
    let mut s = five_color_setup(&["Zuko's Exile", "Gnarled Professor"]);
    let victim = s.force_permanent_on_battlefield(1, "Gnarled Professor");
    let card = s.game().state.permanents[victim].as_ref().unwrap().card;
    cast(&mut s, "Zuko's Exile");
    assert!(s.choose_target(Target::Permanent(victim)));
    settle(&mut s);
    assert_eq!(s.game().state.zones.zone_of(card), Some(ZoneType::Exile));
    assert_eq!(s.battlefield_permanents_named(1, "Clue").len(), 1);
    assert!(s.battlefield_permanents_named(0, "Clue").is_empty());
}

#[test]
fn elemental_new_decisions_replay_reject_stale_commands_and_restore_exact_forks() {
    let mut s = five_color_setup(&["Azula Always Lies", "Gnarled Professor"]);
    let target = s.force_permanent_on_battlefield(0, "Gnarled Professor");
    cast(&mut s, "Azula Always Lies");
    assert_choice_contract(&s);
    choose_mode(&mut s, 2);
    assert_choice_contract(&s);
    assert!(s.choose_target(Target::Permanent(target)));
    assert_choice_contract(&s);

    let mut s = five_color_setup(&["Gather the White Lotus"]);
    cast(&mut s, "Gather the White Lotus");
    pass_until(&mut s, ActionSpaceKind::Scry);
    assert_choice_contract(&s);
    s.step_action(0);
    assert_choice_contract(&s);

    let mut s = five_color_setup(&["Ostrich-Horse"]);
    pin_top(&mut s, &["Forest", "Island", "Plains"]);
    cast(&mut s, "Ostrich-Horse");
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    assert_choice_contract(&s);

    let mut s = five_color_setup(&["True Ancestry"]);
    let land = library_card(&s, 0, "Forest");
    s.game_mut().move_card(land, ZoneType::Graveyard);
    cast(&mut s, "True Ancestry");
    assert_choice_contract(&s);
}

#[test]
fn corrected_card_characteristics_match_oracle() {
    let s = Scenario::new(
        deck(&[
            ("Water Tribe Rallier", 2),
            ("Allies at Last", 2),
            ("Firebending Lesson", 2),
            ("It'll Quench Ya!", 2),
            ("Badgermole Cub", 2),
            ("Suki, Kyoshi Warrior", 2),
        ]),
        deck(&[("Plains", 40)]),
        381,
    );
    let card = |name: &str| {
        s.game()
            .state
            .cards
            .iter()
            .find(|card| card.name == name)
            .unwrap_or_else(|| panic!("{name} missing"))
            .clone()
    };

    let rallier = card("Water Tribe Rallier");
    assert_eq!(rallier.mana_cost.as_ref().unwrap().mana_value, 2);
    assert_eq!((rallier.power, rallier.toughness), (Some(2), Some(2)));
    assert!(rallier.has_subtype("Soldier"));

    let allies = card("Allies at Last");
    assert_eq!(allies.mana_cost.as_ref().unwrap().mana_value, 3);
    assert!(allies.types.is_instant());

    let lesson = card("Firebending Lesson");
    assert_eq!(lesson.mana_cost.as_ref().unwrap().mana_value, 1);
    assert!(lesson.types.is_instant());
    assert_eq!(lesson.optional_cost().unwrap().mana_value, 4);
    assert!(lesson.has_subtype("Lesson"));

    assert!(card("It'll Quench Ya!").has_subtype("Lesson"));

    let cub = card("Badgermole Cub");
    assert_eq!((cub.power, cub.toughness), (Some(2), Some(2)));
    assert_eq!(cub.abilities.len(), 1, "earthbend ETB registered");

    let suki = card("Suki, Kyoshi Warrior");
    assert!(suki.supertypes.iter().any(|s| s == "legendary"));
    assert_eq!(suki.toughness, Some(4));
    assert!(suki.power_cda().is_some());
}

#[test]
fn required_fixed_token_templates_have_the_printed_sizes_and_keyword_effects() {
    let mut s = five_color_setup(&["Abandon Attachments"]);
    for (name, power, toughness, flying) in [
        ("Ally", 1, 1, false),
        ("Spirit", 1, 1, false),
        ("Dragon", 4, 4, true),
        ("Monk", 1, 1, false),
        ("Soldier", 2, 2, false),
        ("Bear", 4, 4, false),
        ("Ballistic Boulder", 2, 1, true),
    ] {
        s.game_mut().create_token(name, PlayerId(0), false);
        let token = permanent_named(&s, name);
        assert_eq!(s.game().effective_pt(token), (power, toughness), "{name}");
        assert_eq!(keywords(&s, token).flying, flying, "{name}");
    }
    let monk = permanent_named(&s, "Monk");
    cast(&mut s, "Abandon Attachments");
    settle(&mut s);
    assert_eq!(s.game().effective_pt(monk), (2, 2));
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.game().effective_power(monk), 1);
}

#[test]
fn qualified_fliers_evade_ground_and_vigilant_attackers_remain_untapped() {
    for (name, flying, vigilant) in [
        ("Momo, Friendly Flier", true, false),
        ("First-Time Flyer", true, false),
        ("Messenger Hawk", true, false),
        ("Knowledge Seeker", false, true),
        ("Tiger-Seal", false, true),
        ("White Lotus Reinforcements", false, true),
    ] {
        let mut s = five_color_setup(&[name, "Gray Ogre"]);
        let attacker = s.force_permanent_on_battlefield(0, name);
        s.force_permanent_on_battlefield(1, "Gray Ogre");
        s.game_mut().state.permanents[attacker]
            .as_mut()
            .unwrap()
            .summoning_sick = false;
        s.advance_to_active_step(0, StepKind::DeclareAttackers);
        s.declare_attack();
        assert_eq!(tapped(&s, attacker), !vigilant, "{name}");
        s.advance_to_active_step(0, StepKind::DeclareBlockers);
        assert_eq!(s.action_space().actions.iter().any(|a| matches!(a, Action::DeclareBlocker { attacker: Some(id), .. } if *id == attacker)), !flying, "{name}");
    }
}

#[test]
fn printed_trample_first_strike_lifelink_and_menace_affect_combat() {
    for name in [
        "Earth King's Lieutenant",
        "Master Piandao",
        "Foggy Swamp Spirit Keeper",
        "Callous Inspector",
    ] {
        let mut s = five_color_setup(&[name, "Gray Ogre"]);
        let attacker = s.force_permanent_on_battlefield(0, name);
        let blocker = s.force_permanent_on_battlefield(1, "Gray Ogre");
        s.game_mut().state.permanents[attacker]
            .as_mut()
            .unwrap()
            .summoning_sick = false;
        if name == "Earth King's Lieutenant" {
            s.game_mut().state.permanents[attacker]
                .as_mut()
                .unwrap()
                .plus1_counters = 3;
        }
        s.advance_to_active_step(0, StepKind::DeclareAttackers);
        s.declare_attack();
        s.advance_to_active_step(0, StepKind::DeclareBlockers);
        s.declare_block();
        s.advance_to_active_step(0, StepKind::EndOfCombat);
        match name {
            "Earth King's Lieutenant" => assert_eq!(s.life(1), 18),
            "Master Piandao" => {
                assert!(s.game().state.permanents[blocker].is_none());
                assert_eq!(
                    s.game().state.permanents[attacker].as_ref().unwrap().damage,
                    0
                );
            }
            "Foggy Swamp Spirit Keeper" => assert_eq!(s.life(0), 22),
            "Callous Inspector" => {
                assert_eq!(s.life(1), 19);
                assert!(s.game().state.permanents[attacker].is_some());
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn artifact_enchantment_removal_covers_both_types_and_excludes_lands() {
    for removal in [
        "Zuko's Exile",
        "Origin of Metalbending",
        "Curious Farm Animals",
    ] {
        for enchantment in [false, true] {
            let mut s = five_color_setup(&[removal, "Tolls of War"]);
            let target = if enchantment {
                s.force_permanent_on_battlefield(1, "Tolls of War")
            } else {
                s.game_mut().create_token("Clue", PlayerId(1), false);
                s.battlefield_permanents_named(1, "Clue")[0]
            };
            let land = s.force_permanent_on_battlefield(1, "Forest");
            let target_card = s.game().state.permanents[target].as_ref().unwrap().card;
            if removal == "Curious Farm Animals" {
                let source = s.force_permanent_on_battlefield(0, removal);
                activate(&mut s, source, 0);
            } else {
                cast(&mut s, removal);
                if removal == "Origin of Metalbending" {
                    choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 0, .. }));
                }
            }
            assert!(!s.choose_target(Target::Permanent(land)));
            assert!(s.choose_target(Target::Permanent(target)));
            settle(&mut s);
            assert!(s.game().state.permanents[target].is_none());
            if enchantment {
                assert_eq!(
                    s.game().state.zones.zone_of(target_card),
                    Some(if removal == "Zuko's Exile" {
                        ZoneType::Exile
                    } else {
                        ZoneType::Graveyard
                    })
                );
            }
            if removal == "Zuko's Exile" {
                assert_eq!(s.battlefield_permanents_named(1, "Clue").len(), 1);
            }
            if removal == "Curious Farm Animals" {
                assert_eq!(s.life(0), 23);
            }
        }
    }
}

#[test]
fn cleanup_sba_grants_priority_and_then_repeats_mandatory_discard() {
    let mut s = ur_lessons_setup();
    let land = put_onto_battlefield(&mut s, 0, "Island");
    let card = card_of(&s, land);
    // A 0/0 animated land survives only while its until-end-of-turn buff lasts.
    let permanent = s.game_mut().state.permanents[land].as_mut().unwrap();
    permanent.animated = true;
    permanent.temp_toughness = 1;
    s.game_mut().draw_cards(PlayerId(0), 4);
    s.advance_to_active_step(0, StepKind::End);
    resolve_top(&mut s);
    while s.action_space().kind == ActionSpaceKind::Discard {
        s.step_action(0);
    }
    assert_eq!(s.zone_size(0, ZoneType::Hand), 7);
    assert_eq!(
        s.game().state.zones.zone_of(card),
        Some(ZoneType::Graveyard)
    );
    assert_eq!(s.game().state.turn.current_step_kind(), StepKind::Cleanup);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Priority);

    // CR 514.3a: cards drawn during exceptional cleanup priority must be
    // discarded in a new cleanup step before the next turn starts.
    let turn = s.game().state.turn.turn_number;
    s.game_mut().draw_cards(PlayerId(0), 2);
    resolve_top(&mut s);
    assert_eq!(s.action_space().kind, ActionSpaceKind::Discard);
    assert_eq!(s.game().state.turn.turn_number, turn);
    while s.action_space().kind == ActionSpaceKind::Discard {
        s.step_action(0);
    }
    assert_eq!(s.zone_size(0, ZoneType::Hand), 7);
    assert!(s.game().state.turn.turn_number > turn);
}

#[test]
fn revised_forty_card_deck_completes_and_replays_every_command_exactly() {
    let semantic = authored_pack("two_deck").unwrap();
    let mut ur = semantic
        .player_config("UR candidate", "ur_lessons")
        .unwrap();
    for (name, n) in [
        ("It'll Quench Ya!", 2),
        ("First-Time Flyer", 2),
        ("Pop Quiz", 1),
        ("Igneous Inspiration", 1),
    ] {
        *ur.deck.main.get_mut(name).unwrap() -= n;
    }
    ur.deck.main.retain(|_, n| *n > 0);
    for (name, n) in [
        ("Combustion Technique", 2),
        ("Proft's Eidetic Memory", 1),
        ("Gran-Gran", 1),
        ("Accumulate Wisdom", 1),
    ] {
        *ur.deck.main.entry(name.into()).or_default() += n;
    }
    assert_eq!(ur.deck.main.values().sum::<usize>(), 40);
    let gw = semantic.player_config("GW", "gw_allies").unwrap();
    for seed in 0..4 {
        for reverse in [false, true] {
            let configs = if reverse {
                vec![gw.clone(), ur.clone()]
            } else {
                vec![ur.clone(), gw.clone()]
            };
            let mut game = Game::new(configs.clone(), seed, true);
            let mut replay = Game::new(configs, seed, true);
            for step in 0..20_000_u64 {
                if game.is_game_over() {
                    break;
                }
                let frame = game.semantic_decision_frame().unwrap();
                let i = ((step
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(seed + 1)) as usize)
                    % frame.offers.len();
                let command = Command {
                    command_id: format!("etu88-{step}"),
                    expected_revision: frame.revision,
                    offer_id: frame.offers[i].id.0,
                    answers: vec![],
                    object_preconditions: vec![],
                };
                game.execute_semantic_command(&command).unwrap();
                replay.execute_semantic_command(&command).unwrap();
                assert_eq!(
                    game.state.deterministic_hash(),
                    replay.state.deterministic_hash()
                );
                assert_eq!(
                    serde_json::to_value(&game.state.events).unwrap(),
                    serde_json::to_value(&replay.state.events).unwrap()
                );
                assert_eq!(
                    Observation::new(&game, &[]).to_json(),
                    Observation::new(&replay, &[]).to_json()
                );
            }
            assert!(game.is_game_over(), "seed {seed}, reverse {reverse}");
        }
    }
}
