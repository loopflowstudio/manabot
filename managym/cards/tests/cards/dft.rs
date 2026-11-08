//! Caradora's optional search and counter predicate use the shared rules paths.
use crate::support::{
    game::{five_color_setup, put_onto_battlefield},
    inspect::offered_cards,
    play::{choose_mode, pass_until, select_card, settle},
};
use managym::{
    agent::action::ActionSpaceKind,
    flow::event::GameEvent,
    state::{game_object::PlayerId, zone::ZoneType},
};

#[test]
fn caradora_search_reveals_a_vehicle_and_declining_does_not_shuffle() {
    for search in [false, true] {
        let mut s = five_color_setup(&["Caradora, Heart of Alacria", "Phoenix Fleet Airship"]);
        let before = s
            .game()
            .state
            .zones
            .zone_cards(ZoneType::Library, PlayerId(0))
            .clone();
        put_onto_battlefield(&mut s, 0, "Caradora, Heart of Alacria");
        pass_until(&mut s, ActionSpaceKind::Modal);
        choose_mode(&mut s, if search { 0 } else { 1 });
        if search {
            assert_eq!(s.action_space().kind, ActionSpaceKind::LookAndSelect);
            let offered = offered_cards(&s);
            assert!(!offered.is_empty());
            assert!(offered
                .iter()
                .all(|card| s.game().state.cards[*card].name == "Phoenix Fleet Airship"));
            let selected = offered[0];
            select_card(&mut s, selected);
            settle(&mut s);
            assert_eq!(s.game().state.zones.zone_of(selected), Some(ZoneType::Hand));
            let definition = s.game().state.cards[selected].definition_id;
            assert!(s.game().state.events.iter().any(|event| matches!(event,
                GameEvent::CardRevealed { owner: PlayerId(0), definition: shown } if *shown == definition
            )));
            assert_eq!(
                s.game().state.players[0].known_hand.get(&definition),
                Some(&1)
            );
        } else {
            settle(&mut s);
            // The entry helper can take Caradora out of the library; every
            // remaining member keeps its order when the optional search is declined.
            let after = s
                .game()
                .state
                .zones
                .zone_cards(ZoneType::Library, PlayerId(0));
            let retained: Vec<_> = before
                .into_iter()
                .filter(|card| after.contains(card))
                .collect();
            assert_eq!(&retained, after);
        }
    }
}
