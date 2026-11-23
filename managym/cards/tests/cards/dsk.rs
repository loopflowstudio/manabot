//! Manifest Dread's battlefield after its controller chooses between the top two cards.
use crate::support::game::{five_color_setup, pin_top};
use managym::{
    agent::{
        observation::Observation,
        observation_encoder::{encode, ObservationEncoderConfig},
    },
    state::{game_object::PlayerId, zone::ZoneType},
};

#[test]
fn manifest_dread_hides_the_printing_at_opponent_priority() {
    let projections = [("Gray Ogre", "Forest"), ("Forest", "Gray Ogre")].map(|(hidden, other)| {
        let mut s = five_color_setup(&["Gray Ogre", "Lightning Bolt"]);
        s.force_card_in_hand(1, "Lightning Bolt");
        let cards = pin_top(&mut s, &[hidden, "Plains", other]);
        let printing = s.game().state[cards[0]].definition_id;
        let permanent = s.game_mut().manifest(cards[0], PlayerId(0));
        s.game_mut().move_object(cards[1], ZoneType::Graveyard);
        s.game_mut().scenario_refresh_priority().unwrap();
        s.pass_priority();
        assert_eq!(s.action_space().player, Some(PlayerId(1)));

        let mut fork = s.snapshot().restore(Default::default()).unwrap();
        for game in [s.game(), &*fork.benchmark_game_mut()] {
            assert_eq!(game.state[permanent].retained_base().1.name, hidden);
            assert_eq!(game.state.zones.zone_of(cards[2]), Some(ZoneType::Library));
            let controller = Observation::for_player(game, PlayerId(0));
            assert_eq!(
                controller.agent_permanents[0].registry_key,
                printing.0 as i32
            );
        }
        let opponent = Observation::for_player(s.game(), PlayerId(1));
        let encoded = encode(&opponent, &ObservationEncoderConfig::default()).unwrap();
        let rows = opponent.opponent_permanents;
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.registry_key, -1);
        assert_eq!(row.object_id, Some(row.id));
        assert_eq!(
            (row.characteristics.power, row.characteristics.toughness),
            (2, 2)
        );
        assert!(row.characteristics.name.is_empty());
        (rows, encoded.opponent_permanents)
    });
    assert_eq!(projections[0], projections[1]);
}
