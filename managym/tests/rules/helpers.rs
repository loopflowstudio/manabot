use std::collections::BTreeMap;

use managym::{
    agent::action::{Action, ActionSpace, ActionSpaceKind, ActionType},
    flow::turn::{PhaseKind, StepKind},
    state::{
        game_object::{CardId, PermanentId, PlayerId},
        mana::Mana,
        player::PlayerConfig,
        zone::ZoneType,
    },
    Game,
};

const MAX_SCENARIO_ACTIONS: usize = 20_000;

pub fn mountain_deck() -> BTreeMap<String, usize> {
    BTreeMap::from([("Mountain".to_string(), 40)])
}

pub fn forest_deck() -> BTreeMap<String, usize> {
    BTreeMap::from([("Forest".to_string(), 40)])
}

pub fn forest_elves_deck() -> BTreeMap<String, usize> {
    BTreeMap::from([
        ("Forest".to_string(), 24),
        ("Llanowar Elves".to_string(), 36),
    ])
}

pub fn ogre_only_deck() -> BTreeMap<String, usize> {
    BTreeMap::from([("Grey Ogre".to_string(), 40)])
}

pub fn empty_deck() -> BTreeMap<String, usize> {
    BTreeMap::new()
}

pub struct Scenario {
    game: Game,
}

impl Scenario {
    pub fn new(
        player0_deck: BTreeMap<String, usize>,
        player1_deck: BTreeMap<String, usize>,
        seed: u64,
    ) -> Scenario {
        let p0 = PlayerConfig::new("p0", player0_deck);
        let p1 = PlayerConfig::new("p1", player1_deck);
        Scenario {
            game: Game::new(vec![p0, p1], seed, false),
        }
    }

    pub fn game(&self) -> &Game {
        &self.game
    }

    pub fn game_mut(&mut self) -> &mut Game {
        &mut self.game
    }

    pub fn action_space(&self) -> &ActionSpace {
        self.game
            .action_space()
            .expect("scenario should have an active action space")
    }

    pub fn current_step(&self) -> StepKind {
        self.game.state.turn.current_step_kind()
    }

    pub fn current_phase(&self) -> PhaseKind {
        self.game.state.turn.current_phase_kind()
    }

    pub fn active_player(&self) -> usize {
        self.game.active_player().0
    }

    pub fn life(&self, player: usize) -> i32 {
        self.game.state.players[player].life
    }

    pub fn zone_size(&self, player: usize, zone: ZoneType) -> usize {
        self.game.state.zones.size(zone, PlayerId(player))
    }

    pub fn assert_life(&self, player: usize, expected: i32) {
        assert_eq!(
            self.life(player),
            expected,
            "unexpected life for player {player}"
        );
    }

    pub fn assert_zone_size(&self, player: usize, zone: ZoneType, expected: usize) {
        assert_eq!(
            self.zone_size(player, zone),
            expected,
            "unexpected zone size for player {player} in {zone:?}"
        );
    }

    pub fn assert_action_available(&self, action_type: ActionType) {
        assert!(
            self.action_index_by_type(action_type).is_some(),
            "expected action {action_type:?} in {:?}",
            self.action_space().actions
        );
    }

    pub fn assert_action_not_available(&self, action_type: ActionType) {
        assert!(
            self.action_index_by_type(action_type).is_none(),
            "did not expect action {action_type:?} in {:?}",
            self.action_space().actions
        );
    }

    pub fn step_action(&mut self, index: usize) {
        self.game.step(index).expect("scenario step should succeed");
    }

    pub fn take_action_by_type(&mut self, action_type: ActionType) -> bool {
        let Some(index) = self.action_index_by_type(action_type) else {
            return false;
        };
        self.step_action(index);
        true
    }

    pub fn pass_priority(&mut self) {
        assert!(
            self.take_action_by_type(ActionType::PriorityPassPriority),
            "priority pass action should be available"
        );
    }

    pub fn assert_game_over(&self) {
        assert!(self.game.is_game_over(), "game should be over");
    }

    pub fn assert_winner(&self, player: usize) {
        assert_eq!(self.game.winner_index(), Some(player));
    }

    pub fn advance_default_action(&mut self) {
        let space = self.action_space().clone();
        let index = match space.kind {
            ActionSpaceKind::Priority => space
                .actions
                .iter()
                .position(|action| action.action_type() == ActionType::PriorityPassPriority)
                .unwrap_or(space.actions.len().saturating_sub(1)),
            ActionSpaceKind::DeclareAttacker => space
                .actions
                .iter()
                .position(|action| matches!(action, Action::DeclareAttacker { attack: false, .. }))
                .unwrap_or(space.actions.len().saturating_sub(1)),
            ActionSpaceKind::DeclareBlocker => space
                .actions
                .iter()
                .position(|action| matches!(action, Action::DeclareBlocker { attacker: None, .. }))
                .unwrap_or(space.actions.len().saturating_sub(1)),
            ActionSpaceKind::GameOver => 0,
        };
        self.step_action(index);
    }

    pub fn advance_to_step(&mut self, target: StepKind) {
        self.advance_until(
            |scenario| scenario.current_step() == target,
            format!("failed to reach step {target:?}"),
        );
    }

    pub fn advance_to_phase(&mut self, target: PhaseKind) {
        self.advance_until(
            |scenario| scenario.current_phase() == target,
            format!("failed to reach phase {target:?}"),
        );
    }

    pub fn advance_to_active_step(&mut self, active_player: usize, target: StepKind) {
        self.advance_until(
            |scenario| {
                scenario.active_player() == active_player && scenario.current_step() == target
            },
            format!("failed to reach step {target:?} for player {active_player}"),
        );
    }

    pub fn force_card_in_hand(&mut self, player: usize, card_name: &str) {
        let Some(index) = self
            .game
            .state
            .cards
            .iter()
            .position(|card| card.owner == PlayerId(player) && card.name == card_name)
        else {
            panic!("card {card_name} not found for player {player}");
        };

        self.game
            .state
            .zones
            .move_card(CardId(index), PlayerId(player), ZoneType::Hand);
    }

    pub fn battlefield_permanents_named(&self, player: usize, card_name: &str) -> Vec<PermanentId> {
        self.game
            .state
            .zones
            .zone_cards(ZoneType::Battlefield, PlayerId(player))
            .iter()
            .filter_map(|card_id| {
                let card = &self.game.state.cards[card_id.0];
                if card.name != card_name {
                    return None;
                }
                self.game.state.card_to_permanent[card_id.0]
            })
            .collect()
    }

    pub fn set_player_mana_pool(&mut self, player: usize, mana: &str) {
        self.game.state.players[player].mana_pool = Mana::parse(mana);
    }

    fn action_index_by_type(&self, action_type: ActionType) -> Option<usize> {
        self.action_space()
            .actions
            .iter()
            .position(|action| action.action_type() == action_type)
    }

    fn advance_until<F>(&mut self, mut predicate: F, failure_message: String)
    where
        F: FnMut(&Scenario) -> bool,
    {
        for _ in 0..MAX_SCENARIO_ACTIONS {
            if self.game.is_game_over() || predicate(self) {
                return;
            }
            self.advance_default_action();
        }
        panic!("{failure_message}");
    }
}
