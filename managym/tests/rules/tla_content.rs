//! Independent outcomes for the first content-only TLA additions. These tests
//! use the same typed programs as the pool, through ordinary legal actions.
use super::{
    elemental_cards::{card, cast, counters, pass_until, permanent, settle, setup},
    helpers::Scenario,
};
use managym::{
    agent::action::{Action, ActionSpaceKind},
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
    assert_eq!(
        (
            s.game().effective_power(captain),
            s.game().effective_toughness(captain)
        ),
        (4, 4)
    );
    assert_eq!(
        (
            s.game().effective_power(own),
            s.game().effective_toughness(own)
        ),
        (3, 3)
    );
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
    assert_eq!(
        (
            s.game().effective_power(parrot),
            s.game().effective_toughness(parrot)
        ),
        (3, 3)
    );
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
    assert_eq!(
        (
            s.game().effective_power(allies[0]),
            s.game().effective_toughness(allies[0])
        ),
        (1, 1)
    );
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
            assert_eq!(
                (
                    s.game().effective_power(creature),
                    s.game().effective_toughness(creature)
                ),
                (5, 3)
            );
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
    assert_eq!(
        (
            s.game().effective_power(creature),
            s.game().effective_toughness(creature)
        ),
        (4, 4)
    );
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
    assert_eq!(
        (
            s.game().effective_power(band),
            s.game().effective_toughness(band)
        ),
        (3, 5)
    );
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
        assert_eq!(
            (
                s.game().effective_power(token),
                s.game().effective_toughness(token)
            ),
            (power, toughness),
            "{name}"
        );
        assert_eq!(keywords(&s, token).flying, flying, "{name}");
    }
    let monk = permanent(&s, "Monk");
    cast(&mut s, "Abandon Attachments");
    settle(&mut s);
    assert_eq!(
        (
            s.game().effective_power(monk),
            s.game().effective_toughness(monk)
        ),
        (2, 2)
    );
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
