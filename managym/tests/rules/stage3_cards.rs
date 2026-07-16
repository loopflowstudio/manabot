// stage3_cards.rs
// Trace tests for the Stage-3 specials and the cards that exercise them:
// earthbend (Badgermole Cub), exile-until-this-leaves (Earth Kingdom
// Jailer), static continuous effects (White Lotus Reinforcements,
// First-Time Flyer), dynamic P/T (Suki, Dragonfly Swarm), until-end-of-
// combat mana (Fire Nation Cadets), death trigger with graveyard
// condition (Dragonfly Swarm), plus Compassionate Healer, Earth King's
// Lieutenant, Yip Yip!, Fancy Footwork, Enter the Avatar State and the
// corrected Firebending Lesson / Water Tribe Rallier / Allies at Last.

use std::collections::BTreeMap;

use managym::{
    agent::{
        action::{Action, ActionSpaceKind, ActionType},
        observation::Observation,
    },
    flow::turn::StepKind,
    state::{
        game_object::{CardId, PermanentId, PlayerId, Target},
        zone::ZoneType,
    },
};

use super::helpers::*;

fn deck(entries: &[(&str, usize)]) -> BTreeMap<String, usize> {
    entries
        .iter()
        .map(|(name, qty)| ((*name).to_string(), *qty))
        .collect()
}

fn force_lands(s: &mut Scenario, player: usize, land: &str, count: usize) {
    for _ in 0..count {
        s.force_permanent_on_battlefield(player, land);
    }
}

fn permanent(s: &Scenario, id: PermanentId) -> &managym::state::permanent::Permanent {
    s.game().state.permanents[id]
        .as_ref()
        .expect("permanent should exist")
}

fn hand(s: &Scenario, player: usize) -> Vec<CardId> {
    s.game()
        .state
        .zones
        .zone_cards(ZoneType::Hand, PlayerId(player))
        .clone()
}

fn clear_hand(s: &mut Scenario, player: usize) {
    for card in hand(s, player) {
        s.game_mut()
            .state
            .zones
            .move_card(card, PlayerId(player), ZoneType::Library);
    }
}

fn cast_only(s: &mut Scenario) {
    assert!(
        s.take_action_by_type(ActionType::PriorityCastSpell),
        "cast action should be available"
    );
}

fn cast_and_resolve(s: &mut Scenario) {
    cast_only(s);
    s.pass_priority();
    s.pass_priority();
}

/// Move an owned card straight into a player's graveyard (silent test
/// setup — e.g. planting a Lesson).
fn plant_in_graveyard(s: &mut Scenario, player: usize, card_name: &str) -> CardId {
    let card = find_owned_card(s, player, card_name);
    s.game_mut()
        .state
        .zones
        .move_card(card, PlayerId(player), ZoneType::Graveyard);
    card
}

fn find_owned_card(s: &Scenario, player: usize, card_name: &str) -> CardId {
    s.game()
        .state
        .cards
        .iter()
        .enumerate()
        .find_map(|(index, card)| {
            (card.owner == PlayerId(player)
                && card.name == card_name
                && matches!(
                    s.game().state.zones.zone_of(CardId(index)),
                    Some(ZoneType::Hand) | Some(ZoneType::Library)
                ))
            .then_some(CardId(index))
        })
        .unwrap_or_else(|| panic!("card {card_name} not found for player {player}"))
}

/// Put a fresh library copy into the hand — the stock helper re-finds
/// the same copy when called twice.
fn force_new_card_in_hand(s: &mut Scenario, player: usize, card_name: &str) -> CardId {
    let card = s
        .game()
        .state
        .cards
        .iter()
        .enumerate()
        .find_map(|(index, card)| {
            (card.owner == PlayerId(player)
                && card.name == card_name
                && s.game().state.zones.zone_of(CardId(index)) == Some(ZoneType::Library))
            .then_some(CardId(index))
        })
        .unwrap_or_else(|| panic!("no library copy of {card_name} for player {player}"));
    s.game_mut()
        .state
        .zones
        .move_card(card, PlayerId(player), ZoneType::Hand);
    card
}

/// Advance until the stack, pending triggers, and decisions are drained.
fn resolve_everything(s: &mut Scenario) {
    s.advance_until(
        |sc| {
            sc.game().state.pending_triggers.is_empty()
                && sc.game().state.pending_trigger_choice.is_none()
                && sc.game().state.stack_objects.is_empty()
                && sc.game().state.suspended_decision.is_none()
        },
        "stack and triggers should drain".to_string(),
    );
}

fn clear_summoning_sickness(s: &mut Scenario, id: PermanentId) {
    s.game_mut().state.permanents[id]
        .as_mut()
        .expect("permanent")
        .summoning_sick = false;
}

// ---------------------------------------------------------------------------
// Earthbend — Badgermole Cub.
// ---------------------------------------------------------------------------

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
    assert!(s.game().state.pending_trigger_choice.is_none());
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

// ---------------------------------------------------------------------------
// Exile-until-this-leaves — Earth Kingdom Jailer.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Static continuous effects — White Lotus Reinforcements (anthem),
// First-Time Flyer (conditional self-buff), layer composition.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Dynamic P/T (CDAs) — Suki and Dragonfly Swarm.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Until-end-of-combat mana — Fire Nation Cadets (firebending 2).
// ---------------------------------------------------------------------------

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

fn attack_with(s: &mut Scenario, attacker: PermanentId) {
    let mut attacked = false;
    while s.action_space().kind == ActionSpaceKind::DeclareAttacker {
        let space = s.action_space().clone();
        let index = space.actions.iter().position(|action| {
            matches!(action, Action::DeclareAttacker { permanent, attack: true, .. } if *permanent == attacker)
        });
        match index {
            Some(index) => {
                s.step_action(index);
                attacked = true;
            }
            None => s.decline_attack(),
        }
    }
    assert!(attacked, "expected to attack");
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

// ---------------------------------------------------------------------------
// Compassionate Healer — becomes-tapped trigger with mid-resolution scry.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Earth King's Lieutenant — counters on each other Ally + grows with Allies.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Yip Yip! and Fancy Footwork and Enter the Avatar State.
// ---------------------------------------------------------------------------

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

    let keywords = permanent(&s, kyoshi).temp_keywords.clone();
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

// ---------------------------------------------------------------------------
// Corrected registrations — oracle sanity checks.
// ---------------------------------------------------------------------------

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
    assert_eq!(lesson.kicker.as_ref().unwrap().mana_value, 4);
    assert!(lesson.has_subtype("Lesson"));

    assert!(card("It'll Quench Ya!").has_subtype("Lesson"));

    let cub = card("Badgermole Cub");
    assert_eq!((cub.power, cub.toughness), (Some(2), Some(2)));
    assert_eq!(cub.abilities.len(), 1, "earthbend ETB registered");

    let suki = card("Suki, Kyoshi Warrior");
    assert!(suki.supertypes.iter().any(|s| s == "legendary"));
    assert_eq!(suki.toughness, Some(4));
    assert!(suki.power_cda.is_some());
}
