use std::collections::BTreeMap;

use managym::{
    agent::action::{Action, ActionSpaceKind, ActionType},
    flow::turn::StepKind,
    state::{
        ability::{Ability, Effect, TargetSpec, TriggerCondition, TriggerSubject},
        game_object::{PermanentId, PlayerId, Target},
        zone::ZoneType,
    },
};

use crate::support::{
    game::*,
    inspect::{card_of, find_owned_card_off_battlefield, permanent},
    play::{activate_first, pass_until, select_card},
};

/// CR 702.9 — Creatures with flying can't be blocked except by flying/reach creatures.
#[test]
fn cr_702_flying_cant_be_blocked_by_ground() {
    let mut s = Scenario::new(
        deck_of("Island", "Wind Drake"),
        deck(&[("Forest", 24), ("Llanowar Elves", 36)]),
        70201,
    );
    let drake = s.force_permanent_on_battlefield(0, "Wind Drake");
    let _elf = s.force_permanent_on_battlefield(1, "Llanowar Elves");
    clear_summoning_sickness(&mut s, drake);

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
        "ground creature should not have a legal block action against flyer"
    );
}

/// CR 702.9 / negative — Non-flyers can be blocked by ground creatures.
#[test]
fn cr_702_flying_negative_nonflyer_is_blockable() {
    let mut s = Scenario::new(
        deck_of("Mountain", "Gray Ogre"),
        deck(&[("Forest", 24), ("Llanowar Elves", 36)]),
        70202,
    );
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let _elf = s.force_permanent_on_battlefield(1, "Llanowar Elves");
    clear_summoning_sickness(&mut s, ogre);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::DeclareBlockers);

    assert!(s.action_space().actions.iter().any(|action| {
        matches!(
            action,
            Action::DeclareBlocker {
                attacker: Some(_),
                ..
            }
        )
    }));
}

/// CR 702.17 — Reach allows a creature to block creatures with flying.
#[test]
fn cr_702_reach_blocks_flying() {
    let mut s = Scenario::new(
        deck_of("Island", "Wind Drake"),
        deck_of("Forest", "Giant Spider"),
        70203,
    );
    let drake = s.force_permanent_on_battlefield(0, "Wind Drake");
    let _spider = s.force_permanent_on_battlefield(1, "Giant Spider");
    clear_summoning_sickness(&mut s, drake);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::DeclareBlockers);

    assert!(s.action_space().actions.iter().any(|action| {
        matches!(
            action,
            Action::DeclareBlocker {
                attacker: Some(_),
                ..
            }
        )
    }));
}

/// Negative: reach does not grant evasion while attacking.
#[test]
fn cr_702_reach_negative_no_attacking_evasion() {
    let mut s = Scenario::new(
        deck_of("Forest", "Giant Spider"),
        deck(&[("Forest", 24), ("Llanowar Elves", 36)]),
        70204,
    );
    let spider = s.force_permanent_on_battlefield(0, "Giant Spider");
    let _elf = s.force_permanent_on_battlefield(1, "Llanowar Elves");
    clear_summoning_sickness(&mut s, spider);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::DeclareBlockers);

    assert!(s.action_space().actions.iter().any(|action| {
        matches!(
            action,
            Action::DeclareBlocker {
                attacker: Some(_),
                ..
            }
        )
    }));
}

/// CR 702.10 — Haste allows attacking on the turn the creature enters.
#[test]
fn cr_702_haste_can_attack_turn_it_enters() {
    let mut s = Scenario::new(
        deck_of("Mountain", "Raging Goblin"),
        deck(&[("Mountain", 40)]),
        70205,
    );
    s.play_land_and_cast_creature(0, "Mountain", "Raging Goblin");
    s.advance_to_active_step(0, StepKind::DeclareAttackers);

    assert!(s
        .action_space()
        .actions
        .iter()
        .any(|action| matches!(action, Action::DeclareAttacker { attack: true, .. })));
}

/// Negative: non-haste creatures remain summoning sick.
#[test]
fn cr_702_haste_negative_nonhaste_cant_attack_immediately() {
    let mut s = Scenario::new(
        deck(&[("Forest", 24), ("Llanowar Elves", 36)]),
        deck(&[("Mountain", 40)]),
        70206,
    );
    s.play_land_and_cast_creature(0, "Forest", "Llanowar Elves");
    s.advance_to_active_step(0, StepKind::DeclareAttackers);

    assert!(!s
        .action_space()
        .actions
        .iter()
        .any(|action| matches!(action, Action::DeclareAttacker { attack: true, .. })));
}

/// CR 702.20 — Vigilance attackers don't tap.
#[test]
fn cr_702_vigilance_attacker_stays_untapped() {
    let mut s = Scenario::new(
        deck_of("Plains", "Serra Angel"),
        deck(&[("Mountain", 40)]),
        70207,
    );
    let serra = s.force_permanent_on_battlefield(0, "Serra Angel");
    clear_summoning_sickness(&mut s, serra);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();

    assert!(
        !s.game().state.permanents[serra]
            .as_ref()
            .expect("serra should exist")
            .tapped
    );
}

/// Negative: non-vigilance attackers tap.
#[test]
fn cr_702_vigilance_negative_attacker_taps_without_vigilance() {
    let mut s = Scenario::new(
        deck_of("Mountain", "Gray Ogre"),
        deck(&[("Mountain", 40)]),
        70208,
    );
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    clear_summoning_sickness(&mut s, ogre);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();

    assert!(
        s.game().state.permanents[ogre]
            .as_ref()
            .expect("ogre should exist")
            .tapped
    );
}

/// CR 702.3 — Defender creatures can't attack.
#[test]
fn cr_702_defender_cant_attack() {
    let mut s = Scenario::new(
        BTreeMap::from([
            ("Mountain".to_string(), 20),
            ("Wall of Stone".to_string(), 10),
            ("Raging Goblin".to_string(), 10),
        ]),
        deck_of("Mountain", "Raging Goblin"),
        70209,
    );
    let wall = s.force_permanent_on_battlefield(0, "Wall of Stone");
    let goblin = s.force_permanent_on_battlefield(0, "Raging Goblin");
    clear_summoning_sickness(&mut s, wall);
    clear_summoning_sickness(&mut s, goblin);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);

    assert!(!s.action_space().actions.iter().any(|action| {
        matches!(
            action,
            Action::DeclareAttacker {
                permanent,
                attack: true,
                ..
            } if *permanent == wall
        )
    }));
    assert!(s.action_space().actions.iter().any(|action| {
        matches!(
            action,
            Action::DeclareAttacker {
                permanent,
                attack: true,
                ..
            } if *permanent == goblin
        )
    }));
}

/// Defender can still block.
#[test]
fn cr_702_defender_can_block() {
    let mut s = Scenario::new(
        deck_of("Mountain", "Raging Goblin"),
        deck_of("Mountain", "Wall of Stone"),
        70210,
    );
    let goblin = s.force_permanent_on_battlefield(0, "Raging Goblin");
    let _wall = s.force_permanent_on_battlefield(1, "Wall of Stone");
    clear_summoning_sickness(&mut s, goblin);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::DeclareBlockers);

    assert!(s.action_space().actions.iter().any(|action| {
        matches!(
            action,
            Action::DeclareBlocker {
                attacker: Some(_),
                ..
            }
        )
    }));
}

/// CR 702.19 — Trample assigns excess damage to defending player.
#[test]
fn cr_702_trample_excess_hits_player() {
    let mut s = Scenario::new(
        deck_of("Forest", "War Mammoth"),
        deck(&[("Forest", 24), ("Llanowar Elves", 36)]),
        70211,
    );
    let mammoth = s.force_permanent_on_battlefield(0, "War Mammoth");
    let _elf = s.force_permanent_on_battlefield(1, "Llanowar Elves");
    clear_summoning_sickness(&mut s, mammoth);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    s.declare_block();

    let life_before = s.life(1);
    s.advance_to_active_step(0, StepKind::CombatDamage);
    assert_eq!(s.life(1), life_before - 2);
}

/// Negative: without trample, excess damage isn't dealt to defending player.
#[test]
fn cr_702_trample_negative_no_excess_without_trample() {
    let mut s = Scenario::new(
        deck_of("Mountain", "Gray Ogre"),
        deck(&[("Forest", 24), ("Llanowar Elves", 36)]),
        70212,
    );
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let _elf = s.force_permanent_on_battlefield(1, "Llanowar Elves");
    clear_summoning_sickness(&mut s, ogre);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    s.declare_block();

    let life_before = s.life(1);
    s.advance_to_active_step(0, StepKind::CombatDamage);
    assert_eq!(s.life(1), life_before);
}

/// Interaction: trample + deathtouch only needs 1 lethal damage assignment.
#[test]
fn cr_702_trample_with_deathtouch_assigns_one_lethal() {
    let mut s = Scenario::new(
        deck_of("Forest", "War Mammoth"),
        deck_of("Mountain", "Gray Ogre"),
        70213,
    );
    let mammoth = s.force_permanent_on_battlefield(0, "War Mammoth");
    let _ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    clear_summoning_sickness(&mut s, mammoth);
    let mammoth_card = s.game().state.permanents[mammoth]
        .as_ref()
        .expect("mammoth should exist")
        .card;
    s.game_mut().state.cards[mammoth_card].keywords.deathtouch = true;

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    s.declare_block();

    let life_before = s.life(1);
    s.advance_to_active_step(0, StepKind::CombatDamage);
    assert_eq!(s.life(1), life_before - 2);
}

/// CR 702.2 — Any nonzero deathtouch damage is lethal to a creature.
#[test]
fn cr_702_deathtouch_one_damage_is_lethal() {
    let mut s = Scenario::new(
        deck_of("Forest", "Craw Wurm"),
        deck_of("Swamp", "Typhoid Rats"),
        70214,
    );
    let wurm = s.force_permanent_on_battlefield(0, "Craw Wurm");
    let _rats = s.force_permanent_on_battlefield(1, "Typhoid Rats");
    clear_summoning_sickness(&mut s, wurm);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    s.declare_block();
    s.advance_to_active_step(0, StepKind::CombatDamage);
    s.pass_priority();
    s.pass_priority();

    assert!(
        s.battlefield_permanents_named(0, "Craw Wurm").is_empty(),
        "deathtouch damage should destroy the wurm"
    );
}

/// Negative: without deathtouch, 1 damage doesn't kill large creatures.
#[test]
fn cr_702_deathtouch_negative_normal_one_damage_not_lethal() {
    let mut s = Scenario::new(
        deck_of("Forest", "Craw Wurm"),
        deck(&[("Forest", 24), ("Llanowar Elves", 36)]),
        70215,
    );
    let wurm = s.force_permanent_on_battlefield(0, "Craw Wurm");
    let _elf = s.force_permanent_on_battlefield(1, "Llanowar Elves");
    clear_summoning_sickness(&mut s, wurm);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    s.declare_block();
    s.advance_to_active_step(0, StepKind::CombatDamage);
    s.pass_priority();
    s.pass_priority();

    assert!(!s.battlefield_permanents_named(0, "Craw Wurm").is_empty());
}

/// CR 702.15 — Lifelink causes its controller to gain life with damage dealt.
#[test]
fn cr_702_lifelink_gains_life_on_damage() {
    let mut s = Scenario::new(
        deck_of("Plains", "Healer's Hawk"),
        deck(&[("Mountain", 40)]),
        70216,
    );
    let hawk = s.force_permanent_on_battlefield(0, "Healer's Hawk");
    clear_summoning_sickness(&mut s, hawk);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::CombatDamage);

    assert_eq!(s.life(0), 21);
    assert_eq!(s.life(1), 19);
}

/// Negative: non-lifelink damage doesn't gain life.
#[test]
fn cr_702_lifelink_negative_no_life_gain_without_lifelink() {
    let mut s = Scenario::new(
        deck_of("Island", "Wind Drake"),
        deck(&[("Mountain", 40)]),
        70217,
    );
    let drake = s.force_permanent_on_battlefield(0, "Wind Drake");
    clear_summoning_sickness(&mut s, drake);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::CombatDamage);

    assert_eq!(s.life(0), 20);
    assert_eq!(s.life(1), 18);
}

/// CR 702.111 + 509.1b — Menace attacker can't be blocked by exactly one creature.
#[test]
fn cr_702_menace_single_block_is_removed() {
    let mut s = Scenario::new(
        deck_of("Mountain", "Boggart Brute"),
        deck(&[("Forest", 24), ("Llanowar Elves", 36)]),
        70218,
    );
    let brute = s.force_permanent_on_battlefield(0, "Boggart Brute");
    let _elf = s.force_permanent_on_battlefield(1, "Llanowar Elves");
    clear_summoning_sickness(&mut s, brute);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    s.declare_block();

    let combat = s.game().state.combat.as_ref().expect("combat should exist");
    let attacker = *combat.attackers.first().expect("attacker should exist");
    let blockers = combat
        .attacker_to_blockers
        .get(&attacker)
        .expect("blockers should exist");
    assert!(blockers.is_empty());
}

/// Menace is satisfied by two or more blockers.
#[test]
fn cr_702_menace_two_blockers_is_legal() {
    let mut s = Scenario::new(
        deck_of("Mountain", "Boggart Brute"),
        deck(&[("Forest", 24), ("Llanowar Elves", 36)]),
        70219,
    );
    let brute = s.force_permanent_on_battlefield(0, "Boggart Brute");
    let _elf_one = s.force_permanent_on_battlefield(1, "Llanowar Elves");
    let _elf_two = s.force_permanent_on_battlefield(1, "Llanowar Elves");
    clear_summoning_sickness(&mut s, brute);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    s.declare_block();
    s.declare_block();

    let combat = s.game().state.combat.as_ref().expect("combat should exist");
    let attacker = *combat.attackers.first().expect("attacker should exist");
    let blockers = combat
        .attacker_to_blockers
        .get(&attacker)
        .expect("blockers should exist");
    assert_eq!(blockers.len(), 2);
}

/// CR 702.7 — First strike damage is dealt before normal combat damage.
#[test]
fn cr_702_first_strike_kills_before_normal_damage() {
    let mut s = Scenario::new(
        deck_of("Plains", "Youthful Knight"),
        deck_of("Mountain", "Gray Ogre"),
        70220,
    );
    let knight = s.force_permanent_on_battlefield(0, "Youthful Knight");
    let _ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    clear_summoning_sickness(&mut s, knight);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    s.declare_block();
    s.advance_to_active_step(0, StepKind::CombatDamage);
    s.pass_priority();
    s.pass_priority();

    assert!(!s
        .battlefield_permanents_named(0, "Youthful Knight")
        .is_empty());
    assert!(s.battlefield_permanents_named(1, "Gray Ogre").is_empty());
}

/// Negative: without first strike, creatures deal combat damage simultaneously.
#[test]
fn cr_702_first_strike_negative_normal_combat_is_simultaneous() {
    let mut s = Scenario::new(
        deck_of("Mountain", "Gray Ogre"),
        deck_of("Mountain", "Gray Ogre"),
        70221,
    );
    let attacker = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let _blocker = s.force_permanent_on_battlefield(1, "Gray Ogre");
    clear_summoning_sickness(&mut s, attacker);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    s.declare_block();
    s.advance_to_active_step(0, StepKind::CombatDamage);
    s.pass_priority();
    s.pass_priority();

    assert!(s.battlefield_permanents_named(0, "Gray Ogre").is_empty());
    assert!(s.battlefield_permanents_named(1, "Gray Ogre").is_empty());
}

/// CR 702.4 — Double strike deals damage in both combat damage steps.
#[test]
fn cr_702_double_strike_deals_damage_twice() {
    let mut s = Scenario::new(
        deck_of("Plains", "Fencing Ace"),
        deck(&[("Mountain", 40)]),
        70222,
    );
    let ace = s.force_permanent_on_battlefield(0, "Fencing Ace");
    clear_summoning_sickness(&mut s, ace);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    let life_before = s.life(1);
    s.advance_to_active_step(0, StepKind::CombatDamage);

    assert_eq!(s.life(1), life_before - 2);
}

/// Interaction: double strike + trample can deal player damage in second sub-step.
#[test]
fn cr_702_double_strike_with_trample_interaction() {
    let mut s = Scenario::new(
        deck_of("Plains", "Fencing Ace"),
        deck(&[("Forest", 24), ("Llanowar Elves", 36)]),
        70223,
    );
    let ace = s.force_permanent_on_battlefield(0, "Fencing Ace");
    let _elf = s.force_permanent_on_battlefield(1, "Llanowar Elves");
    clear_summoning_sickness(&mut s, ace);
    let ace_card = s.game().state.permanents[ace]
        .as_ref()
        .expect("ace should exist")
        .card;
    s.game_mut().state.cards[ace_card].keywords.trample = true;

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    s.declare_block();
    let life_before = s.life(1);
    s.advance_to_active_step(0, StepKind::CombatDamage);

    assert_eq!(s.life(1), life_before - 1);
}

/// Regression: multi-blocker assignment uses total attacker power, not full power per blocker.
#[test]
fn cr_702_regression_multi_blocker_damage_is_split_not_duplicated() {
    let mut s = Scenario::new(
        deck_of("Mountain", "Gray Ogre"),
        deck_of("Mountain", "Gray Ogre"),
        70224,
    );
    let ogre = s.force_permanent_on_battlefield(0, "Gray Ogre");
    let _ogre_one = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let _ogre_two = s.force_permanent_on_battlefield(1, "Gray Ogre");
    clear_summoning_sickness(&mut s, ogre);

    s.advance_to_active_step(0, StepKind::DeclareAttackers);
    s.declare_attack();
    s.advance_to_active_step(0, StepKind::DeclareBlockers);
    s.declare_block();
    s.declare_block();
    s.advance_to_active_step(0, StepKind::CombatDamage);
    s.pass_priority();
    s.pass_priority();

    let surviving_ogres = s.battlefield_permanents_named(1, "Gray Ogre").len();
    assert_eq!(surviving_ogres, 1);
}

/// Sanity: deathtouch-enabled creature should still be castable/attackable via normal priority.
#[test]
fn cr_702_deathtouch_card_still_uses_normal_actions() {
    let mut s = Scenario::new(
        deck_of("Swamp", "Typhoid Rats"),
        deck(&[("Mountain", 40)]),
        70225,
    );
    s.advance_to_active_step(0, StepKind::Main);
    s.force_card_in_hand(0, "Swamp");
    s.force_card_in_hand(0, "Typhoid Rats");
    assert!(s.take_action_by_type(ActionType::PriorityPlayLand));
    assert!(s.take_action_by_type(ActionType::PriorityCastSpell));
}

/// Player 1's Weaponize the Monsters (with a Gray Ogre to sacrifice) faces
/// player 0's Waterfall Aerialist, which has ward {2}.
fn ward_ability_setup(seed: u64) -> (Scenario, PermanentId) {
    let both = deck(&[
        ("Island", 20),
        ("Waterfall Aerialist", 8),
        ("Weaponize the Monsters", 8),
        ("Gray Ogre", 8),
    ]);
    let mut s = Scenario::new(both.clone(), both, seed);
    let aerialist = s.force_permanent_on_battlefield(0, "Waterfall Aerialist");
    s.force_permanent_on_battlefield(1, "Weaponize the Monsters");
    s.force_permanent_on_battlefield(1, "Gray Ogre");
    clear_hand(&mut s, 0);
    clear_hand(&mut s, 1);
    s.advance_to_active_step(1, StepKind::Main);
    (s, aerialist)
}

fn activate_weaponize_at(s: &mut Scenario, player: usize, victim: PermanentId, mana: &str) {
    let weapon = s.battlefield_permanents_named(player, "Weaponize the Monsters")[0];
    let ogre = s.battlefield_permanents_named(player, "Gray Ogre")[0];
    s.set_player_mana_pool(player, mana);
    activate_first(s, weapon);
    assert!(s.choose_target(Target::Permanent(victim)));
    select_card(s, card_of(s, ogre));
}

/// CR 702.21a — ward triggers for an opponent's activated ability too, and
/// declining its payment counters the ability.
#[test]
fn cr_702_ward_counters_an_opponents_activated_ability_unless_paid() {
    let (mut s, aerialist) = ward_ability_setup(70221);
    activate_weaponize_at(&mut s, 1, aerialist, "2");
    assert_eq!(
        s.game().state.stack_objects.len(),
        2,
        "ward above the ability"
    );
    s.pass_priority();
    s.pass_priority();
    assert_eq!(s.action_space().kind, ActionSpaceKind::PayOrNot);
    assert_eq!(s.action_space().player, Some(PlayerId(1)));
    assert!(s.take_action_by_type(ActionType::DeclineChoice));
    assert!(
        s.game().state.stack_objects.is_empty(),
        "the ability was countered"
    );
    assert_eq!(permanent(&s, aerialist).damage, 0);
}

#[test]
fn cr_702_ward_paid_lets_the_ability_resolve() {
    let (mut s, aerialist) = ward_ability_setup(70222);
    activate_weaponize_at(&mut s, 1, aerialist, "4");
    s.pass_priority();
    s.pass_priority();
    assert_eq!(s.action_space().kind, ActionSpaceKind::PayOrNot);
    assert!(s.take_action_by_type(ActionType::PayCost));
    s.pass_priority();
    s.pass_priority();
    assert!(
        s.game().state.permanents[aerialist].is_none(),
        "2 damage kills it"
    );
}

/// CR 702.21a — a triggered ability an opponent controls is also an ability.
#[test]
fn cr_702_ward_counters_an_opponents_triggered_ability() {
    let (mut s, aerialist) = ward_ability_setup(70223);
    let ogre = find_owned_card_off_battlefield(&s, 1, "Gray Ogre");
    inject_ability(
        &mut s,
        ogre,
        Ability::Triggered {
            intervening_if: None,
            condition: TriggerCondition::EntersTheBattlefield {
                subject: TriggerSubject::This,
            },
            effects: vec![Effect::DealDamage {
                amount: 2,
                target: TargetSpec::Creature,
            }],
        },
    );
    s.game_mut().move_card(ogre, ZoneType::Battlefield);
    s.game_mut().scenario_refresh_priority().unwrap();
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
    assert!(s.choose_target(Target::Permanent(aerialist)));
    pass_until(&mut s, ActionSpaceKind::PayOrNot);
    assert!(s.take_action_by_type(ActionType::DeclineChoice));
    assert!(s.game().state.stack_objects.is_empty());
    assert_eq!(permanent(&s, aerialist).damage, 0);
}

/// CR 702.21a — only opponents' abilities trigger ward.
#[test]
fn cr_702_ward_ignores_its_controllers_own_ability() {
    let both = deck(&[
        ("Island", 20),
        ("Waterfall Aerialist", 8),
        ("Weaponize the Monsters", 8),
        ("Gray Ogre", 8),
    ]);
    let mut s = Scenario::new(both.clone(), both, 70224);
    let aerialist = s.force_permanent_on_battlefield(0, "Waterfall Aerialist");
    s.force_permanent_on_battlefield(0, "Weaponize the Monsters");
    s.force_permanent_on_battlefield(0, "Gray Ogre");
    s.advance_to_active_step(0, StepKind::Main);
    activate_weaponize_at(&mut s, 0, aerialist, "2");
    assert_eq!(s.game().state.stack_objects.len(), 1);
}
