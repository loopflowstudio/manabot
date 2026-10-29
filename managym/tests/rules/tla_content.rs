//! Independent TLA clause outcomes through the same typed programs as the pool.
//! Directed legal actions cover costs, choices, combat and effect lifetimes.
use super::{
    elemental_cards::{card, cast, counters, pass_until, permanent, pin_top, settle, setup},
    helpers::Scenario,
};
use managym::{
    agent::{
        action::{Action, ActionSpaceKind},
        observation::Observation,
    },
    flow::turn::StepKind,
    state::{
        game_object::{CardId, PermanentId, PlayerId, Target},
        zone::ZoneType,
    },
};

fn choose(s: &mut Scenario, predicate: impl Fn(&Action) -> bool) {
    let i = s
        .action_space()
        .actions
        .iter()
        .position(predicate)
        .unwrap_or_else(|| panic!("missing action: {:?}", s.action_space()));
    s.step_action(i);
}

fn activate(s: &mut Scenario, source: PermanentId, index: usize) {
    s.game_mut().scenario_refresh_priority().unwrap();
    choose(s, |a| {
        matches!(a, Action::ActivateAbility { permanent, ability_index, .. }
        if *permanent == source && *ability_index == index)
    });
}

fn enter(s: &mut Scenario, name: &str) -> PermanentId {
    let id = card(s, 0, name);
    s.game_mut().move_card(id, ZoneType::Battlefield);
    s.game_mut().scenario_refresh_priority().unwrap();
    s.game().state.card_to_permanent[id].unwrap()
}

fn tapped(s: &Scenario, id: PermanentId) -> bool {
    s.game().state.permanents[id].as_ref().unwrap().tapped
}

fn keywords(s: &Scenario, id: PermanentId) -> managym::state::card::Keywords {
    let permanent = s.game().state.permanents[id].as_ref().unwrap();
    permanent.effective_keywords(&s.game().state.cards[permanent.card])
}

fn can_cast(s: &Scenario, id: CardId) -> bool {
    s.action_space()
        .actions
        .iter()
        .any(|a| matches!(a, Action::CastSpell { card, .. } if *card == id))
}

/// CR 608.2b: a legal source keeps resolution alive, but never licenses damage
/// to an illegal victim. Damage belongs to each remaining legal source.
#[test]
fn allies_at_last_respects_source_and_victim_legality_and_lifelink() {
    for (protect_victim, remove_hawk) in [(false, false), (true, false), (false, true)] {
        let mut s = setup(&[
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
    let mut s = setup(&["Allies at Last", "Healer's Hawk", "Wall of Stone"]);
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

#[test]
fn water_tribe_captain_pays_five_and_buffs_only_own_creatures_until_cleanup() {
    let mut s = setup(&["Water Tribe Captain", "Gray Ogre"]);
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

#[test]
fn iguana_and_boar_trigger_only_on_own_noncreature_casts() {
    let mut s = setup(&[
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

#[test]
fn treetop_creates_one_white_ally_and_can_attack_immediately() {
    let mut s = setup(&["Treetop Freedom Fighters"]);
    cast(&mut s, "Treetop Freedom Fighters");
    settle(&mut s);
    let fighter = permanent(&s, "Treetop Freedom Fighters");
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
fn fire_sages_counter_cost_and_firebending_mana_lifetime() {
    let mut s = setup(&["Fire Sages"]);
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

#[test]
fn yuyan_rummage_may_be_declined_or_discard_then_draw() {
    for discard in [false, true] {
        let mut s = setup(&["Yuyan Archers", "Gray Ogre"]);
        let chosen = card(&s, 0, "Gray Ogre");
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
        assert!(keywords(&s, permanent(&s, "Yuyan Archers")).reach);
    }
}

#[test]
fn cunning_maneuver_buffs_and_investigates_but_all_invalid_targets_stop_it() {
    for remove in [false, true] {
        let mut s = setup(&["Cunning Maneuver", "Gray Ogre"]);
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
fn pillar_launch_untaps_and_grants_temporary_reach_and_stats() {
    let mut s = setup(&["Pillar Launch", "Gray Ogre"]);
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
fn protectors_sacrifice_is_committed_and_only_another_own_ally_is_legal() {
    let mut s = setup(&[
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
            let mut s = setup(&[name, spell]);
            let land = enter(&mut s, name);
            assert!(tapped(&s, land), "{name} enters tapped");
            s.set_player_mana_pool(0, if color == 'U' { "2" } else { "0" });
            let id = card(&s, 0, spell);
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
        let mut s = setup(&[name]);
        let land = enter(&mut s, name);
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
fn rumble_arena_scries_then_filters_one_paid_mana_without_using_the_stack() {
    let mut s = setup(&["Rumble Arena", "Crossroads of Destiny"]);
    let arena = enter(&mut s, "Rumble Arena");
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
fn walltop_intervening_lesson_condition_is_checked_at_death_and_resolution() {
    for lesson_at_death in [false, true] {
        for remove_lesson in [false, true] {
            let mut s = setup(&["Walltop Sentries", "Firebending Lesson"]);
            let sentries = s.force_permanent_on_battlefield(0, "Walltop Sentries");
            let key = keywords(&s, sentries);
            assert!(key.reach && key.deathtouch);
            let lesson = card(&s, 0, "Firebending Lesson");
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
fn cat_owl_attack_untaps_artifacts_or_creatures_in_either_control() {
    for artifact in [false, true] {
        let mut s = setup(&["Cat-Owl", "Prophetic Prism", "Gray Ogre"]);
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

#[test]
fn earth_kingdom_soldier_offers_zero_one_or_two_distinct_own_creatures() {
    for count in 0..=2 {
        let mut s = setup(&["Earth Kingdom Soldier", "Gray Ogre"]);
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
        assert!(keywords(&s, permanent(&s, "Earth Kingdom Soldier")).vigilance);
    }
}

#[test]
fn wandering_musicians_pump_all_own_creatures_for_the_turn() {
    let mut s = setup(&["Wandering Musicians", "Gray Ogre"]);
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
fn required_fixed_token_templates_have_the_printed_sizes_and_keyword_effects() {
    let mut s = setup(&["Abandon Attachments"]);
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
        let token = permanent(&s, name);
        assert_eq!(s.game().effective_pt(token), (power, toughness), "{name}");
        assert_eq!(keywords(&s, token).flying, flying, "{name}");
    }
    let monk = permanent(&s, "Monk");
    cast(&mut s, "Abandon Attachments");
    settle(&mut s);
    assert_eq!(s.game().effective_pt(monk), (2, 2));
    s.advance_to_active_step(1, StepKind::Main);
    assert_eq!(s.game().effective_power(monk), 1);
}

#[test]
fn dragon_and_soldier_tokens_firebend_their_printed_amount() {
    for (name, amount) in [("Dragon", 4), ("Soldier", 1)] {
        let mut s = setup(&[]);
        s.game_mut().create_token(name, PlayerId(0), false);
        let token = permanent(&s, name);
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

// Full-card clauses: private selection, resolution and temporary effects.
#[test]
fn rallier_reveals_only_the_selected_card_and_can_decline() {
    for take in [false, true] {
        let mut s = setup(&["Water Tribe Rallier", "Gray Ogre", "Wall of Stone"]);
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

#[test]
fn cadets_keep_announced_firebending_after_last_lesson_leaves() {
    let mut s = setup(&["Fire Nation Cadets", "Firebending Lesson"]);
    let cadets = s.force_permanent_on_battlefield(0, "Fire Nation Cadets");
    let lesson = card(&s, 0, "Firebending Lesson");
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
fn octopus_form_untaps_only_own_target_and_expires() {
    let mut s = setup(&["Octopus Form", "Gray Ogre"]);
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
fn fancy_footwork_one_target_or_one_remaining_target_expires() {
    for two in [false, true] {
        let mut s = setup(&["Fancy Footwork", "Gray Ogre"]);
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

#[test]
fn uncle_iroh_firebending_survives_steps_and_expires_after_combat() {
    let mut s = setup(&["Uncle Iroh"]);
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
fn deadly_precision_sacrifices_artifact_before_destroying_a_different_creature() {
    let mut s = setup(&["Deadly Precision", "Gray Ogre"]);
    let victim = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let victim_card = s.game().state.permanents[victim].as_ref().unwrap().card;
    s.game_mut().create_token("Clue", PlayerId(0), false);
    let clue = permanent(&s, "Clue");
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
fn sokkas_haiku_counters_draws_mills_and_untaps_and_needs_a_legal_target() {
    for invalid in [false, true] {
        let mut s = setup(&["Sokka's Haiku", "Lightning Strike"]);
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
fn qualified_fliers_evade_ground_and_vigilant_attackers_remain_untapped() {
    for (name, flying, vigilant) in [
        ("Momo, Friendly Flier", true, false),
        ("First-Time Flyer", true, false),
        ("Messenger Hawk", true, false),
        ("Knowledge Seeker", false, true),
        ("Tiger-Seal", false, true),
        ("White Lotus Reinforcements", false, true),
    ] {
        let mut s = setup(&[name, "Gray Ogre"]);
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
        let mut s = setup(&[name, "Gray Ogre"]);
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
fn disciple_rechecks_power_and_its_effect_expires() {
    for increased_before_resolution in [false, true] {
        let mut s = setup(&["Deserter's Disciple", "Gray Ogre"]);
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
fn merchant_successful_search_is_optional_and_returns_source_before_shuffle() {
    let mut s = setup(&["Unlucky Cabbage Merchant"]);
    cast(&mut s, "Unlucky Cabbage Merchant");
    settle(&mut s);
    let merchant = permanent(&s, "Unlucky Cabbage Merchant");
    let merchant_card = s.game().state.permanents[merchant].as_ref().unwrap().card;
    let food = permanent(&s, "Food");
    activate(&mut s, food, 0);
    pass_until(&mut s, ActionSpaceKind::Modal);
    choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 0, .. }));
    let forest = card(&s, 0, "Forest");
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
fn shared_roots_search_can_find_or_decline_without_leaking_the_library() {
    for take in [false, true] {
        let mut s = setup(&["Shared Roots", "Rumble Arena"]);
        cast(&mut s, "Shared Roots");
        pass_until(&mut s, ActionSpaceKind::LookAndSelect);
        let forest = card(&s, 0, "Forest");
        let nonbasic = card(&s, 0, "Rumble Arena");
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
            assert!(tapped(&s, permanent(&s, "Forest")));
        }
        assert!(s.game().state.suspended_decision.is_none());
    }
}

#[test]
fn tiger_dillo_blocking_needs_another_four_power_creature() {
    for enabled in [false, true] {
        let mut s = setup(&["Tiger-Dillo", "Gray Ogre", "Master Piandao"]);
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
fn wartime_protestors_haste_grant_is_temporary_and_only_for_other_own_allies() {
    let mut s = setup(&["Wartime Protestors", "Gray Ogre", "Pretending Poxbearers"]);
    cast(&mut s, "Wartime Protestors");
    settle(&mut s);
    let protestors = permanent(&s, "Wartime Protestors");
    assert_eq!(counters(&s, protestors), 0);
    cast(&mut s, "Gray Ogre");
    settle(&mut s);
    let nonally = permanent(&s, "Gray Ogre");
    assert_eq!(counters(&s, nonally), 0);
    cast(&mut s, "Pretending Poxbearers");
    settle(&mut s);
    let ally = permanent(&s, "Pretending Poxbearers");
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
fn origin_indestructible_expires_but_counter_remains() {
    let mut s = setup(&["Origin of Metalbending", "Gray Ogre"]);
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
fn momo_excludes_lemurs_and_opponents_turn_and_its_pump_expires() {
    let mut s = setup(&["Momo, Friendly Flier", "Messenger Hawk"]);
    let momo = s.force_permanent_on_battlefield(0, "Momo, Friendly Flier");
    let other = s.force_permanent_on_battlefield(1, "Momo, Friendly Flier");
    let lemur = card(&s, 0, "Momo, Friendly Flier");
    let hawk = card(&s, 0, "Messenger Hawk");
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
    let next = card(&s, 0, "Messenger Hawk");
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
fn piandao_can_take_a_lesson_or_equipment_or_decline_without_revealing_remainder() {
    for selection in [None, Some(0), Some(1)] {
        let mut s = setup(&["Master Piandao", "Abandon Attachments", "Prophetic Prism"]);
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
fn wartime_granted_haste_enables_tap_cost_without_erasing_summoning_sickness() {
    let mut s = setup(&["Wartime Protestors", "Deserter's Disciple", "Gray Ogre"]);
    s.force_permanent_on_battlefield(0, "Wartime Protestors");
    let target = s.force_permanent_on_battlefield(0, "Gray Ogre");
    cast(&mut s, "Deserter's Disciple");
    settle(&mut s);
    let disciple = permanent(&s, "Deserter's Disciple");
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
fn artifact_enchantment_removal_covers_both_types_and_excludes_lands() {
    for removal in [
        "Zuko's Exile",
        "Origin of Metalbending",
        "Curious Farm Animals",
    ] {
        for enchantment in [false, true] {
            let mut s = setup(&[removal, "Tolls of War"]);
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
