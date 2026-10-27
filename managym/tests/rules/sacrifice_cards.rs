// sacrifice_cards.rs
// Directed sacrifice-cost, death, fetch and landfall behavior through Commands.
use super::{
    elemental_cards::{card, cast, counters, pass_until, permanent, settle, setup},
    helpers::Scenario,
};
use managym::{
    agent::action::{Action, ActionSpaceKind},
    state::{
        game_object::{CardId, PermanentId, PlayerId, Target},
        zone::ZoneType,
    },
};

fn enter(s: &mut Scenario, name: &str) -> CardId {
    let id = card(s, 0, name);
    if s.game().state.cards[id].types.is_land() {
        s.game_mut().move_card(id, ZoneType::Battlefield);
        s.game_mut().scenario_refresh_priority().unwrap();
    } else {
        cast(s, name);
    }
    settle(s);
    id
}

fn choose(s: &mut Scenario, predicate: impl Fn(&Action) -> bool) {
    let i = s
        .action_space()
        .actions
        .iter()
        .position(predicate)
        .unwrap_or_else(|| panic!("missing action: {:?}", s.action_space()));
    s.step_action(i);
}
fn activate(s: &mut Scenario, id: PermanentId) {
    if s.action_space().kind == ActionSpaceKind::Priority {
        s.game_mut().scenario_refresh_priority().unwrap();
    }
    choose(
        s,
        |a| matches!(a,Action::ActivateAbility {permanent,..} if *permanent==id),
    );
}
fn target_opponent(s: &mut Scenario) {
    choose(s, |a| {
        matches!(
            a,
            Action::ChooseTarget {
                target: Target::Player(PlayerId(1)),
                ..
            }
        )
    });
}
/// Resolve damage triggers with a fixed target; fail if the stack never settles.
fn settle_targeting_opponent(s: &mut Scenario) {
    for _ in 0..20 {
        if s.action_space().kind == ActionSpaceKind::ChooseTarget {
            target_opponent(s);
        } else if s.action_space().kind == ActionSpaceKind::Priority
            && s.game().state.stack_objects.is_empty()
        {
            return;
        } else {
            s.pass_priority();
        }
    }
    panic!("damage triggers did not settle");
}

fn select(s: &mut Scenario, card: CardId) {
    choose(
        s,
        |a| matches!(a,Action::SelectCard {card: selected,..} if *selected==card),
    );
}

#[test]
fn death_rewards_and_selected_activation_costs() {
    let mut s = setup(&[
        "Bastion of Remembrance",
        "Callous Inspector",
        "Weaponize the Monsters",
    ]);
    enter(&mut s, "Bastion of Remembrance");
    assert_eq!(s.battlefield_permanents_named(0, "Human Soldier").len(), 1);
    let fodder = enter(&mut s, "Callous Inspector");
    enter(&mut s, "Weaponize the Monsters");
    let life = s.game().state.players[0].life;
    let weapon = permanent(&s, "Weaponize the Monsters");
    activate(&mut s, weapon);
    target_opponent(&mut s);
    assert_eq!(
        s.game().state.zones.zone_of(fodder),
        Some(ZoneType::Battlefield)
    );
    select(&mut s, fodder);
    assert_eq!(
        s.game().state.zones.zone_of(fodder),
        Some(ZoneType::Graveyard)
    );
    settle(&mut s);
    assert_eq!(s.game().state.players[1].life, 17);
    assert_eq!(s.game().state.players[0].life, life);
    assert_eq!(s.battlefield_permanents_named(0, "Clue").len(), 1);
}

#[test]
fn farm_animals_can_decline_target_and_still_pay_sacrifice() {
    let mut s = setup(&["Curious Farm Animals"]);
    enter(&mut s, "Curious Farm Animals");
    let id = permanent(&s, "Curious Farm Animals");
    activate(&mut s, id);
    choose(&mut s, |a| matches!(a, Action::Decline { .. }));
    settle(&mut s);
    assert_eq!(s.game().state.players[0].life, 23);
    assert!(s
        .battlefield_permanents_named(0, "Curious Farm Animals")
        .is_empty());
}

#[test]
fn dispute_sacrifices_before_responses_draws_two_and_makes_live_treasure() {
    let mut s = setup(&["Deadly Dispute", "Unlucky Cabbage Merchant"]);
    enter(&mut s, "Unlucky Cabbage Merchant");
    let food = permanent(&s, "Food");
    let food_card = s.game().state.permanents[food].as_ref().unwrap().card;
    cast(&mut s, "Deadly Dispute");
    let hand = s.game().state.zones.size(ZoneType::Hand, PlayerId(0));
    select(&mut s, food_card);
    assert_ne!(
        s.game().state.zones.zone_of(food_card),
        Some(ZoneType::Battlefield)
    );
    settle(&mut s);
    assert_eq!(
        s.game().state.zones.size(ZoneType::Hand, PlayerId(0)),
        hand + 1
    );
    let treasure = permanent(&s, "Treasure");
    let stack = s.game().state.stack_objects.len();
    activate(&mut s, treasure);
    assert_eq!(s.game().state.stack_objects.len(), stack);
    choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 4, .. }));
    assert!(s.battlefield_permanents_named(0, "Treasure").is_empty());
}

#[test]
fn precision_has_a_real_four_mana_alternative() {
    let mut s = setup(&["Deadly Precision", "Callous Inspector"]);
    let victim = enter(&mut s, "Callous Inspector");
    let id = permanent(&s, "Callous Inspector");
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
fn fetch_lands_have_distinct_entry_and_life_rules() {
    for (name, tapped, life) in [("Fabled Passage", true, 20), ("Prismatic Vista", false, 19)] {
        let mut s = setup(&[name]);
        enter(&mut s, name);
        let source = permanent(&s, name);
        activate(&mut s, source);
        pass_until(&mut s, ActionSpaceKind::LookAndSelect);
        let found = card(&s, 0, "Forest");
        select(&mut s, found);
        settle(&mut s);
        let id = s.game().state.card_to_permanent[found].unwrap();
        assert_eq!(
            s.game().state.permanents[id].as_ref().unwrap().tapped,
            tapped
        );
        assert_eq!(s.game().state.players[0].life, life);
        assert!(s.battlefield_permanents_named(0, name).is_empty());
    }
}

#[test]
fn cycle_sacrifices_during_resolution_and_searches_two_distinct_basics() {
    let mut s = setup(&["Cycle of Renewal"]);
    let sacrificed = enter(&mut s, "Forest");
    cast(&mut s, "Cycle of Renewal");
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    select(&mut s, sacrificed);
    let one = card(&s, 0, "Forest");
    select(&mut s, one);
    let two = card(&s, 0, "Swamp");
    select(&mut s, two);
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
fn landfall_creates_resources_and_counters() {
    let mut s = setup(&[
        "Sazh's Chocobo",
        "Tireless Tracker",
        "Lotus Cobra",
        "Tireless Provisioner",
    ]);
    for name in [
        "Sazh's Chocobo",
        "Tireless Tracker",
        "Lotus Cobra",
        "Tireless Provisioner",
    ] {
        enter(&mut s, name);
    }
    // The Cobra's color and Provisioner's token are separate resolution choices.
    let land = card(&s, 0, "Forest");
    s.game_mut().move_card(land, ZoneType::Battlefield);
    s.game_mut().scenario_refresh_priority().unwrap();
    s.pass_priority();
    for _ in 0..20 {
        if s.action_space().kind == ActionSpaceKind::Modal {
            choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 0, .. }));
        } else if s.game().state.stack_objects.is_empty()
            && s.action_space().kind == ActionSpaceKind::Priority
        {
            break;
        } else {
            s.pass_priority();
        }
    }
    assert_eq!(counters(&s, permanent(&s, "Sazh's Chocobo")), 1);
    assert_eq!(s.battlefield_permanents_named(0, "Clue").len(), 1);
    assert_eq!(s.battlefield_permanents_named(0, "Food").len(), 1);
    let clue = permanent(&s, "Clue");
    activate(&mut s, clue);
    settle(&mut s);
    assert_eq!(counters(&s, permanent(&s, "Tireless Tracker")), 1);
}

#[test]
fn tolls_limits_triggers_while_mayhem_sees_opponent_sacrifices() {
    let mut s = setup(&["Tolls of War", "Mayhem Devil"]);
    enter(&mut s, "Tolls of War");
    enter(&mut s, "Mayhem Devil");
    let clue = permanent(&s, "Clue");
    activate(&mut s, clue);
    settle_targeting_opponent(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Ally").len(), 1);
    assert_eq!(s.game().state.players[1].life, 19);
    s.game_mut().create_token("Clue", PlayerId(1), false);
    let opponent_clue = s.battlefield_permanents_named(1, "Clue")[0];
    // Give the other player priority through the ordinary pass.
    s.game_mut().scenario_refresh_priority().unwrap();
    s.pass_priority();
    activate(&mut s, opponent_clue);
    settle_targeting_opponent(&mut s);
    assert_eq!(s.game().state.players[1].life, 18);
    assert_eq!(s.battlefield_permanents_named(0, "Ally").len(), 1);
}

#[test]
fn june_sacrifices_another_creature_only_on_its_controllers_turn() {
    let mut s = setup(&["June, Bounty Hunter", "Callous Inspector"]);
    let june = enter(&mut s, "June, Bounty Hunter");
    let fodder = enter(&mut s, "Callous Inspector");
    let source = permanent(&s, "June, Bounty Hunter");
    activate(&mut s, source);
    assert!(s
        .action_space()
        .actions
        .iter()
        .all(|a| !matches!(a,Action::SelectCard {card,..} if *card==june)));
    select(&mut s, fodder);
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
fn merchant_search_returns_the_exact_source_even_on_failure_to_find() {
    for search in [false, true] {
        let mut s = setup(&["Unlucky Cabbage Merchant"]);
        let merchant = enter(&mut s, "Unlucky Cabbage Merchant");
        let food = permanent(&s, "Food");
        activate(&mut s, food);
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
fn treasure_completes_an_announced_cost_without_a_priority_window() {
    let mut s = setup(&["Deadly Precision", "Callous Inspector"]);
    let victim = enter(&mut s, "Callous Inspector");
    let target = permanent(&s, "Callous Inspector");
    s.game_mut().create_token("Treasure", PlayerId(0), false);
    s.set_player_mana_pool(0, "0");
    cast(&mut s, "Deadly Precision");
    choose(
        &mut s,
        |a| matches!(a,Action::ChooseTarget {target:Target::Permanent(p),..} if *p==target),
    );
    super::elemental_cards::assert_choice_contract(&s);
    select(&mut s, victim);
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
    let treasure = permanent(&s, "Treasure");
    activate(&mut s, treasure);
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
    super::elemental_cards::assert_choice_contract(&s);
    choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 2, .. }));
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    assert_eq!(
        s.game().state.zones.zone_of(victim),
        Some(ZoneType::Graveyard)
    );
    assert!(s.battlefield_permanents_named(0, "Treasure").is_empty());
}

#[test]
fn fetch_land_counts_the_found_land_and_triggers_tracker_twice() {
    let mut s = setup(&["Fabled Passage", "Tireless Tracker"]);
    for _ in 0..3 {
        s.force_permanent_on_battlefield(0, "Forest");
    }
    enter(&mut s, "Tireless Tracker");
    enter(&mut s, "Fabled Passage");
    s.pass_priority();
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Clue").len(), 1);
    let fetch = permanent(&s, "Fabled Passage");
    activate(&mut s, fetch);
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    super::elemental_cards::assert_choice_contract(&s);
    let found = card(&s, 0, "Forest");
    select(&mut s, found);
    settle(&mut s);
    let id = s.game().state.card_to_permanent[found].unwrap();
    assert!(!s.game().state.permanents[id].as_ref().unwrap().tapped);
    assert_eq!(s.battlefield_permanents_named(0, "Clue").len(), 2);
}

#[test]
fn bastion_uses_a_sacrificed_lands_last_known_creature_type() {
    let mut s = setup(&["Bastion of Remembrance", "Cycle of Renewal"]);
    enter(&mut s, "Bastion of Remembrance");
    let land = s.force_permanent_on_battlefield(0, "Forest");
    let source = s.game_mut().state.permanents[land].as_mut().unwrap();
    source.animated = true;
    source.plus1_counters = 1;
    let card = source.card;
    cast(&mut s, "Cycle of Renewal");
    pass_until(&mut s, ActionSpaceKind::LookAndSelect);
    select(&mut s, card);
    choose(&mut s, |a| matches!(a, Action::Decline { .. }));
    settle(&mut s);
    assert_eq!(s.game().state.players[0].life, 21);
    assert_eq!(s.game().state.players[1].life, 19);
}

#[test]
fn june_is_unblockable_after_two_draws() {
    for draws in [0, 2] {
        let mut s = setup(&["June, Bounty Hunter", "Callous Inspector"]);
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

#[test]
fn mana_activation_keeps_the_nonactive_players_priority() {
    let mut s = setup(&[]);
    s.game_mut().create_token("Treasure", PlayerId(1), false);
    s.game_mut().scenario_refresh_priority().unwrap();
    s.pass_priority();
    let source = s.battlefield_permanents_named(1, "Treasure")[0];
    activate(&mut s, source);
    choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 1, .. }));
    assert_eq!(s.action_space().player, Some(PlayerId(1)));
    assert_eq!(s.game().state.players[1].mana_pool.mana[1], 5);
}

#[test]
fn tolls_fires_once_in_a_turn_and_resets_on_the_next_turn() {
    let mut s = setup(&["Tolls of War"]);
    enter(&mut s, "Tolls of War");
    let first = permanent(&s, "Clue");
    activate(&mut s, first);
    settle(&mut s);
    s.game_mut().create_token("Clue", PlayerId(0), false);
    let second = permanent(&s, "Clue");
    activate(&mut s, second);
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Ally").len(), 1);
    s.advance_to_active_step(1, managym::flow::turn::StepKind::Main);
    s.advance_to_active_step(0, managym::flow::turn::StepKind::Main);
    s.set_player_mana_pool(0, "2");
    s.game_mut().create_token("Clue", PlayerId(0), false);
    let third = permanent(&s, "Clue");
    activate(&mut s, third);
    settle(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Ally").len(), 2);
}

#[test]
fn farm_animals_destroy_an_artifact_without_targeting_a_creature() {
    let mut s = setup(&["Curious Farm Animals", "Callous Inspector"]);
    enter(&mut s, "Curious Farm Animals");
    enter(&mut s, "Callous Inspector");
    s.game_mut().create_token("Food", PlayerId(0), false);
    let food = permanent(&s, "Food");
    let animal = permanent(&s, "Curious Farm Animals");
    let creature = permanent(&s, "Callous Inspector");
    activate(&mut s, animal);
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
fn mayhem_keeps_granted_lifelink_after_it_is_sacrificed() {
    let mut s = setup(&["Mayhem Devil", "Weaponize the Monsters"]);
    let card = enter(&mut s, "Mayhem Devil");
    enter(&mut s, "Weaponize the Monsters");
    let devil = permanent(&s, "Mayhem Devil");
    // Model a resolved until-end-of-turn grant before the sacrifice.
    s.game_mut().state.permanents[devil]
        .as_mut()
        .unwrap()
        .temp_keywords
        .lifelink = true;
    let weapon = permanent(&s, "Weaponize the Monsters");
    activate(&mut s, weapon);
    target_opponent(&mut s);
    select(&mut s, card);
    settle_targeting_opponent(&mut s);
    assert_eq!(s.game().state.players[0].life, 21);
    assert_eq!(s.game().state.players[1].life, 17);
}

#[test]
fn merchant_returns_to_its_owners_bottom_while_its_controller_shuffles() {
    let mut s = setup(&["Unlucky Cabbage Merchant"]);
    let source = s.force_permanent_on_battlefield(1, "Unlucky Cabbage Merchant");
    let merchant = s.game().state.permanents[source].as_ref().unwrap().card;
    // Model an earlier control change without changing physical ownership.
    s.game_mut().state.permanents[source]
        .as_mut()
        .unwrap()
        .controller = PlayerId(0);
    s.game_mut().create_token("Food", PlayerId(0), false);
    let food = permanent(&s, "Food");
    activate(&mut s, food);
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
fn treasure_is_available_when_announcing_a_kicker_cost() {
    let mut s = setup(&["Firebending Lesson", "Gnarled Professor"]);
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
    let treasure = permanent(&s, "Treasure");
    activate(&mut s, treasure);
    choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 4, .. }));
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    settle(&mut s);
    assert!(s
        .battlefield_permanents_named(1, "Gnarled Professor")
        .is_empty());
}

#[test]
fn weaponize_keeps_its_exact_damage_source_after_removal() {
    let mut s = setup(&["Weaponize the Monsters", "Callous Inspector"]);
    let weapon = enter(&mut s, "Weaponize the Monsters");
    let fodder = enter(&mut s, "Callous Inspector");
    let source = permanent(&s, "Weaponize the Monsters");
    s.game_mut().state.permanents[source]
        .as_mut()
        .unwrap()
        .temp_keywords
        .lifelink = true;
    activate(&mut s, source);
    target_opponent(&mut s);
    select(&mut s, fodder);
    // Model a completed removal response before the activated ability resolves.
    s.game_mut().move_card(weapon, ZoneType::Graveyard);
    settle(&mut s);
    assert_eq!(s.game().state.players[0].life, 21); // Inspector -1, Weaponize lifelink +2.
    assert_eq!(s.game().state.players[1].life, 18);
    assert!(s.game().state.events.iter().any(|event|matches!(event,managym::flow::event::GameEvent::DamageDealt {source:Some(card),target:managym::flow::event::DamageTarget::Player(PlayerId(1)),amount:2} if *card==weapon)));
}
