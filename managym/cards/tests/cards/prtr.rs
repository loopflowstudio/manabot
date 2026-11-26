//! Return to Ravnica promos.
use crate::support::{
    game::{five_color_setup, move_to_hand, Scenario},
    inspect::{card_of, top_spell, zone_of},
    play::{cast, choose_mode, pass_until, select_card, settle},
};
use managym::{
    agent::action::ActionSpaceKind,
    state::{
        game_object::{PlayerId, Target},
        payment::{How, Toward},
        zone::ZoneType,
    },
};

fn militants(names: &[&str]) -> Scenario {
    let mut cards = names.to_vec();
    cards.push("Dryad Militant");
    let mut s = five_color_setup(&cards);
    for player in 0..2 {
        s.force_permanent_on_battlefield(player, "Dryad Militant");
    }
    s
}

fn choose_militant(s: &mut Scenario, player: PlayerId, mode: usize) {
    assert_eq!(s.action_space().kind, ActionSpaceKind::Modal);
    assert_eq!(s.action_space().player, Some(player));
    assert_eq!(s.action_space().actions.len(), 2);
    choose_mode(s, mode);
}

#[test]
fn discard_payment_waits_for_replacement_then_records_one_discard() {
    let mut s = militants(&["Seize the Spoils", "Lightning Bolt"]);
    s.set_player_mana_pool(0, "2R");
    let bolt = move_to_hand(&mut s, "Lightning Bolt");
    cast(&mut s, "Seize the Spoils");
    select_card(&mut s, bolt);
    assert_eq!(zone_of(&s, bolt), Some(ZoneType::Hand));
    for mode in 0..2 {
        let mut branch = s.clone();
        choose_militant(&mut branch, PlayerId(0), mode);
        assert_eq!(zone_of(&branch, bolt), Some(ZoneType::Exile));
        let receipt = &top_spell(&branch).spent;
        assert_eq!(
            receipt
                .contributed(How::Discarded, Toward::Printed)
                .collect::<Vec<_>>(),
            [crate::support::inspect::now(&branch, bolt)]
        );
        settle(&mut branch);
    }
}

#[test]
fn fizzled_spell_waits_on_stack_for_replacement() {
    let mut s = militants(&["Lightning Bolt", "Gray Ogre"]);
    let ogre = s.force_permanent_on_battlefield(1, "Gray Ogre");
    let bolt = cast(&mut s, "Lightning Bolt");
    assert!(s.choose_target(Target::Permanent(ogre)));
    let target = card_of(&s, ogre);
    s.game_mut().move_object(target, ZoneType::Exile);
    if let [managym::state::stack_object::StackObject::Spell(spell)] =
        s.game_mut().state.stack_objects.as_mut_slice()
    {
        spell.controller = PlayerId(1);
    }
    pass_until(&mut s, ActionSpaceKind::Modal);
    assert_eq!(zone_of(&s, bolt), Some(ZoneType::Stack));
    choose_militant(&mut s, PlayerId(1), 1);
    assert_eq!(zone_of(&s, bolt), Some(ZoneType::Exile));
    assert!(s.game().state.stack_objects.is_empty());
}

#[test]
fn militant_exiles_both_players_spells_on_resolution_and_countering() {
    use crate::support::{game::move_from_library, inspect::now, play::choose};
    use managym::agent::action::Action;
    let mut s = five_color_setup(&["Dryad Militant", "Fumigate", "Counterspell"]);
    let militant = s.force_permanent_on_battlefield(0, "Dryad Militant");
    let counter = move_from_library(&mut s, 1, "Counterspell", ZoneType::Hand);
    s.set_player_mana_pool(0, "3WW");
    s.set_player_mana_pool(1, "UU");
    let fumigate = cast(&mut s, "Fumigate");
    s.pass_priority();
    choose(
        &mut s,
        |a| matches!(a, Action::CastSpell { object, .. } if *object == counter),
    );
    let spell = now(&s, fumigate);
    assert!(s.choose_target(Target::StackSpell(spell)));
    settle(&mut s);
    assert_eq!(zone_of(&s, fumigate), Some(ZoneType::Exile));
    assert_eq!(zone_of(&s, counter), Some(ZoneType::Exile));
    assert!(
        s.game().state.permanents[militant].is_some(),
        "Fumigate did not resolve"
    );
}

#[test]
fn both_milled_instants_wait_for_ordering_before_the_batch_commits() {
    use crate::support::game::pin_top;
    let mut s = militants(&["Dread Summons", "Lightning Bolt"]);
    s.set_player_mana_pool(0, "2BB");
    let top = pin_top(&mut s, &["Lightning Bolt", "Lightning Bolt"]);
    cast(&mut s, "Dread Summons");
    choose_mode(&mut s, 2);
    pass_until(&mut s, ActionSpaceKind::Modal);
    choose_militant(&mut s, PlayerId(0), 0);
    assert!(top
        .iter()
        .all(|o| zone_of(&s, *o) == Some(ZoneType::Library)));
    choose_militant(&mut s, PlayerId(0), 1);
    settle(&mut s);
    assert!(top.iter().all(|o| zone_of(&s, *o) == Some(ZoneType::Exile)));
}

#[test]
fn ward_discard_or_decline_resumes_after_its_own_replacement_choice() {
    use crate::support::{inspect::permanent, play::choose};
    use managym::agent::action::Action;
    let mut s = militants(&["Scavenger Regent", "Lightning Bolt"]);
    let dragon = s.force_permanent_on_battlefield(1, "Scavenger Regent");
    let discard = move_to_hand(&mut s, "Lightning Bolt");
    let bolt = cast(&mut s, "Lightning Bolt");
    assert!(s.choose_target(Target::Permanent(dragon)));
    s.pass_priority();
    s.pass_priority();
    for pay in [false, true] {
        let mut branch = s.clone();
        if pay {
            select_card(&mut branch, discard);
        } else {
            choose(&mut branch, |a| matches!(a, Action::Decline { .. }));
        }
        choose_militant(&mut branch, PlayerId(0), 0);
        settle(&mut branch);
        assert_eq!(permanent(&branch, dragon).damage, if pay { 3 } else { 0 });
        assert_eq!(
            zone_of(&branch, discard),
            Some(if pay { ZoneType::Exile } else { ZoneType::Hand })
        );
        assert_eq!(zone_of(&branch, bolt), Some(ZoneType::Exile));
    }
}
