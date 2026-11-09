//! Doubling Season replaces creation and effect-caused counter placement.
use crate::support::{
    game::{enters_battlefield_with_triggers, five_color_setup},
    play::{pass_until, resolve_everything},
};
use managym::{
    agent::action::ActionSpaceKind,
    state::{game_object::PlayerId, permanent::CounterKind, zone::ZoneType},
};

#[test]
fn doubling_season_doubles_entry_lore_and_tokens_but_not_precombat_lore() {
    let mut s = five_color_setup(&["Doubling Season", "Leaves from the Vine", "Gray Ogre"]);
    s.force_permanent_on_battlefield(0, "Doubling Season");
    s.force_permanent_on_battlefield(0, "Gray Ogre");
    let saga = enters_battlefield_with_triggers(&mut s, 0, "Leaves from the Vine");
    assert_eq!(
        s.game().state.permanents[saga]
            .as_ref()
            .unwrap()
            .counter_count(CounterKind::Lore),
        2
    );
    s.game_mut().scenario_refresh_priority().unwrap();
    resolve_everything(&mut s);
    assert_eq!(s.battlefield_permanents_named(0, "Food").len(), 2);
    // Use a second Saga with enough chapters remaining after its doubled entry.
    // Removing Doubling Season while placing then returning it isolates the
    // actual next precombat turn action from entry replacement.
    let season = s.battlefield_permanents_named(0, "Doubling Season")[0];
    let season_card = s.game().state.permanents[season].as_ref().unwrap().card;
    s.game_mut().move_card(season_card, ZoneType::Hand);
    let second = enters_battlefield_with_triggers(&mut s, 0, "Leaves from the Vine");
    s.game_mut().scenario_refresh_priority().unwrap();
    resolve_everything(&mut s);
    s.game_mut().move_card(season_card, ZoneType::Battlefield);
    s.game_mut().scenario_refresh_priority().unwrap();
    let turn = s.game().state.turn.turn_number;
    s.advance_until(
        |s| s.game().state.turn.turn_number > turn && s.game().active_player() == PlayerId(0),
        "next own turn".into(),
    );
    s.advance_to_active_step(0, managym::flow::turn::StepKind::Main);
    assert_eq!(
        s.game().state.permanents[second]
            .as_ref()
            .unwrap()
            .counter_count(CounterKind::Lore),
        2
    );
    pass_until(&mut s, ActionSpaceKind::ChooseTarget);
}
