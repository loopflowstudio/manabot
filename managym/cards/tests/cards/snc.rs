//! Streets of New Capenna triggers through the public decision interface.
use crate::support::{
    game::{five_color_setup, move_card},
    inspect::{card_of, keywords},
    play::{command, resolve_until_priority},
};
use managym::{
    agent::{action::Action, observation::Observation},
    state::{card::Keyword, game_object::PlayerId, zone::ZoneType},
};

#[test]
fn venom_connoisseur_commands_replay_old_and_new_resolutions() {
    let mut s = five_color_setup(&["Venom Connoisseur"]);
    let source = s.force_permanent_on_battlefield(0, "Venom Connoisseur");
    let player = PlayerId(0);
    let ally = s
        .game_mut()
        .create_token("token.ally", player, false)
        .unwrap();
    resolve_until_priority(&mut s);
    assert!(keywords(&s, source).has(Keyword::Deathtouch));
    assert!(!keywords(&s, ally).has(Keyword::Deathtouch));

    s.game_mut().create_token("token.ally", player, false);
    s.advance_until(
        |s| s.game().state.stack_objects.len() == 1,
        "original trigger on stack".into(),
    );
    let card = card_of(&s, source);
    let exiled = move_card(&mut s, card, ZoneType::Exile);
    let returned = move_card(&mut s, exiled, ZoneType::Battlefield);
    s.game_mut().create_token("token.ally", player, false);
    s.advance_until(
        |s| s.game().state.stack_objects.len() == 2,
        "both triggers on stack".into(),
    );
    let pass = |action: &Action| matches!(action, Action::PassPriority { .. });
    for remaining in [1, 0] {
        for _ in 0..2 {
            command(&mut s, pass);
            if s.game().state.stack_objects.len() == remaining {
                break;
            }
        }
        assert_eq!(s.game().state.stack_objects.len(), remaining);
        let observation = Observation::for_player(s.game(), player);
        let public = s.game().state.permanents[s.game().state.permanent_of(returned).unwrap()]
            .as_ref()
            .unwrap()
            .id;
        let live = observation
            .agent_permanents
            .iter()
            .find(|p| p.id == public.0 as i32)
            .unwrap();
        assert_eq!(live.ability_resolutions, [(0, 1)]);
        assert_eq!(keywords(&s, ally).has(Keyword::Deathtouch), remaining == 0);
    }
}
