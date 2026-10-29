//! Tokens created by cards outside TLA (`src/cardsets/tokens.rs`).
use crate::support::{
    activate_any, assert_choice_contract, cast, choose, enter_or_cast, permanent, select, settle,
    setup,
};
use managym::{
    agent::action::{Action, ActionSpaceKind},
    state::{
        game_object::{PlayerId, Target},
        zone::ZoneType,
    },
};

#[test]
fn treasure_completes_an_announced_cost_without_a_priority_window() {
    let mut s = setup(&["Deadly Precision", "Callous Inspector"]);
    let victim = enter_or_cast(&mut s, "Callous Inspector");
    let target = permanent(&s, "Callous Inspector");
    s.game_mut().create_token("Treasure", PlayerId(0), false);
    s.set_player_mana_pool(0, "0");
    cast(&mut s, "Deadly Precision");
    choose(
        &mut s,
        |a| matches!(a,Action::ChooseTarget {target:Target::Permanent(p),..} if *p==target),
    );
    assert_choice_contract(&s);
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
    activate_any(&mut s, treasure);
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
fn mana_activation_keeps_the_nonactive_players_priority() {
    let mut s = setup(&[]);
    s.game_mut().create_token("Treasure", PlayerId(1), false);
    s.game_mut().scenario_refresh_priority().unwrap();
    s.pass_priority();
    let source = s.battlefield_permanents_named(1, "Treasure")[0];
    activate_any(&mut s, source);
    choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 1, .. }));
    assert_eq!(s.action_space().player, Some(PlayerId(1)));
    assert_eq!(s.game().state.players[1].mana_pool.mana[1], 5);
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
    activate_any(&mut s, treasure);
    choose(&mut s, |a| matches!(a, Action::ChooseMode { mode: 4, .. }));
    choose(&mut s, |a| matches!(a, Action::PayCost { .. }));
    settle(&mut s);
    assert!(s
        .battlefield_permanents_named(1, "Gnarled Professor")
        .is_empty());
}
