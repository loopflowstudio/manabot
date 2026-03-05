use crate::{
    agent::action::{Action, ActionSpaceKind, ActionType},
    flow::{
        game::Game,
        turn::{PhaseKind, StepKind},
    },
    state::{
        card::Card,
        game_object::{ObjectId, PlayerId},
        mana::ManaCost,
        permanent::Permanent,
        zone::ZoneType,
    },
};
use serde_json::{json, Value};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TurnData {
    pub turn_number: u32,
    pub phase: PhaseKind,
    pub step: StepKind,
    pub active_player_id: i32,
    pub agent_player_id: i32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlayerData {
    pub player_index: i32,
    pub id: i32,
    pub is_agent: bool,
    pub is_active: bool,
    pub life: i32,
    pub zone_counts: [i32; 7],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CardTypeData {
    pub is_castable: bool,
    pub is_permanent: bool,
    pub is_non_land_permanent: bool,
    pub is_non_creature_permanent: bool,
    pub is_spell: bool,
    pub is_creature: bool,
    pub is_land: bool,
    pub is_planeswalker: bool,
    pub is_enchantment: bool,
    pub is_artifact: bool,
    pub is_kindred: bool,
    pub is_battle: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CardData {
    pub zone: ZoneType,
    pub owner_id: i32,
    pub id: i32,
    pub registry_key: i32,
    pub power: i32,
    pub toughness: i32,
    pub card_types: CardTypeData,
    pub mana_cost: ManaCost,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PermanentData {
    pub id: i32,
    pub controller_id: i32,
    pub tapped: bool,
    pub damage: i32,
    pub is_summoning_sick: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActionOption {
    pub action_type: ActionType,
    pub focus: Vec<i32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActionSpaceData {
    pub action_space_type: ActionSpaceKind,
    pub actions: Vec<ActionOption>,
    pub focus: Vec<i32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Observation {
    pub game_over: bool,
    pub won: bool,
    pub turn: TurnData,
    pub action_space: ActionSpaceData,
    pub agent: PlayerData,
    pub agent_cards: Vec<CardData>,
    pub agent_permanents: Vec<PermanentData>,
    pub opponent: PlayerData,
    pub opponent_cards: Vec<CardData>,
    pub opponent_permanents: Vec<PermanentData>,
}

impl Observation {
    pub fn new(game: &Game) -> Self {
        let agent_player = game.agent_player();
        let opponent_player = PlayerId((agent_player.0 + 1) % 2);

        let game_over = game.is_game_over();
        let won = game
            .winner_index()
            .is_some_and(|winner| winner == agent_player.0);

        let turn = TurnData {
            turn_number: game.state.turn.turn_number,
            phase: game.state.turn.current_phase_kind(),
            step: game.state.turn.current_step_kind(),
            active_player_id: game.state.players[game.active_player().0].id.0 as i32,
            agent_player_id: game.state.players[agent_player.0].id.0 as i32,
        };

        let action_space = game
            .current_action_space
            .as_ref()
            .map(|space| ActionSpaceData {
                action_space_type: space.kind,
                actions: space
                    .actions
                    .iter()
                    .map(|action| ActionOption {
                        action_type: action.action_type(),
                        focus: Self::action_focus(game, action)
                            .into_iter()
                            .map(|id| id.0 as i32)
                            .collect(),
                    })
                    .collect(),
                focus: space.focus.iter().map(|id| id.0 as i32).collect(),
            })
            .unwrap_or(ActionSpaceData {
                action_space_type: ActionSpaceKind::GameOver,
                actions: Vec::new(),
                focus: Vec::new(),
            });

        let agent = Self::player_data(game, agent_player, true);
        let opponent = Self::player_data(game, opponent_player, false);

        let mut obs = Self {
            game_over,
            won,
            turn,
            action_space,
            agent,
            agent_cards: Vec::new(),
            agent_permanents: Vec::new(),
            opponent,
            opponent_cards: Vec::new(),
            opponent_permanents: Vec::new(),
        };

        obs.populate_cards(game, agent_player);
        obs.populate_permanents(game);
        obs
    }

    fn player_data(game: &Game, player: PlayerId, is_agent: bool) -> PlayerData {
        let p = &game.state.players[player.0];
        let mut zone_counts = [0_i32; 7];
        for (zone_index, count) in zone_counts.iter_mut().enumerate() {
            *count = game
                .state
                .zones
                .size(Self::zone_from_index(zone_index), player) as i32;
        }

        PlayerData {
            player_index: p.index as i32,
            id: p.id.0 as i32,
            is_agent,
            is_active: game.active_player() == player,
            life: p.life,
            zone_counts,
        }
    }

    fn populate_cards(&mut self, game: &Game, agent_player: PlayerId) {
        for card in game.state.zones.zone_cards(ZoneType::Hand, agent_player) {
            self.add_card(game, *card, ZoneType::Hand);
        }

        for player in game.players_starting_with_agent() {
            for card in game.state.zones.zone_cards(ZoneType::Graveyard, player) {
                self.add_card(game, *card, ZoneType::Graveyard);
            }
            for card in game.state.zones.zone_cards(ZoneType::Exile, player) {
                self.add_card(game, *card, ZoneType::Exile);
            }
        }

        for card in game.state.zones.stack_order().iter().rev() {
            self.add_card(game, *card, ZoneType::Stack);
        }
    }

    fn populate_permanents(&mut self, game: &Game) {
        for player in game.players_starting_with_agent() {
            for card_id in game.state.zones.zone_cards(ZoneType::Battlefield, player) {
                if let Some(perm_id) = game.state.card_to_permanent[card_id.0] {
                    let Some(permanent) = game.state.permanents[perm_id.0].as_ref() else {
                        continue;
                    };
                    self.add_permanent(game, permanent);
                }
            }
        }
    }

    fn add_card(
        &mut self,
        game: &Game,
        card_id: crate::state::game_object::CardId,
        zone: ZoneType,
    ) {
        let card = &game.state.cards[card_id.0];
        let card_data = Self::card_data(game, card, zone);

        if card_data.owner_id == self.agent.id {
            self.agent_cards.push(card_data);
        } else {
            self.opponent_cards.push(card_data);
        }
    }

    fn add_permanent(&mut self, game: &Game, permanent: &Permanent) {
        let pdat = PermanentData {
            id: permanent.id.0 as i32,
            controller_id: game.state.players[permanent.controller.0].id.0 as i32,
            tapped: permanent.tapped,
            damage: permanent.damage,
            is_summoning_sick: permanent.summoning_sick,
        };

        if permanent.controller.0 == self.agent.player_index as usize {
            self.agent_permanents.push(pdat);
        } else {
            self.opponent_permanents.push(pdat);
        }

        self.add_card(game, permanent.card, ZoneType::Battlefield);
    }

    fn card_data(game: &Game, card: &Card, zone: ZoneType) -> CardData {
        CardData {
            zone,
            owner_id: game.state.players[card.owner.0].id.0 as i32,
            id: card.id.0 as i32,
            registry_key: card.registry_key.0 as i32,
            power: card.power.unwrap_or(0),
            toughness: card.toughness.unwrap_or(0),
            card_types: CardTypeData {
                is_castable: card.types.is_castable(),
                is_permanent: card.types.is_permanent(),
                is_non_land_permanent: card.types.is_non_land_permanent(),
                is_non_creature_permanent: card.types.is_non_creature_permanent(),
                is_spell: card.types.is_spell(),
                is_creature: card.types.is_creature(),
                is_land: card.types.is_land(),
                is_planeswalker: card.types.is_planeswalker(),
                is_enchantment: card.types.is_enchantment(),
                is_artifact: card.types.is_artifact(),
                is_kindred: card.types.is_kindred(),
                is_battle: card.types.is_battle(),
            },
            mana_cost: card.mana_cost.clone().unwrap_or_default(),
        }
    }

    fn zone_from_index(index: usize) -> ZoneType {
        match index {
            0 => ZoneType::Library,
            1 => ZoneType::Hand,
            2 => ZoneType::Battlefield,
            3 => ZoneType::Graveyard,
            4 => ZoneType::Stack,
            5 => ZoneType::Exile,
            6 => ZoneType::Command,
            _ => unreachable!("invalid zone index: {index}"),
        }
    }

    fn action_focus(game: &Game, action: &Action) -> Vec<ObjectId> {
        match action {
            Action::PlayLand { card, .. } | Action::CastSpell { card, .. } => {
                vec![game.state.cards[card.0].id]
            }
            Action::ChooseTarget { target, .. } => match target {
                crate::state::game_object::Target::Player(player) => {
                    vec![game.state.players[player.0].id]
                }
                crate::state::game_object::Target::Permanent(permanent) => game.state.permanents
                    [permanent.0]
                    .as_ref()
                    .map(|perm| vec![perm.id])
                    .unwrap_or_default(),
                crate::state::game_object::Target::StackSpell(card) => {
                    vec![game.state.cards[card.0].id]
                }
            },
            Action::PassPriority { .. } => vec![],
            Action::DeclareAttacker { permanent, .. } => game.state.permanents[permanent.0]
                .as_ref()
                .map(|perm| vec![perm.id])
                .unwrap_or_default(),
            Action::DeclareBlocker {
                blocker, attacker, ..
            } => {
                let mut focus = Vec::new();
                if let Some(perm) = game.state.permanents[blocker.0].as_ref() {
                    focus.push(perm.id);
                }
                if let Some(attacker) = attacker {
                    if let Some(perm) = game.state.permanents[attacker.0].as_ref() {
                        focus.push(perm.id);
                    }
                }
                focus
            }
        }
    }

    pub fn validate(&self) -> bool {
        if self.agent.id == self.opponent.id {
            return false;
        }
        if self.agent.is_agent == self.opponent.is_agent {
            return false;
        }
        for card in &self.agent_cards {
            if card.owner_id != self.agent.id {
                return false;
            }
        }
        for card in &self.opponent_cards {
            if card.owner_id != self.opponent.id {
                return false;
            }
        }
        for permanent in &self.agent_permanents {
            if permanent.controller_id != self.agent.id {
                return false;
            }
        }
        for permanent in &self.opponent_permanents {
            if permanent.controller_id != self.opponent.id {
                return false;
            }
        }
        true
    }

    pub fn to_json(&self) -> String {
        fn player_json(player: &PlayerData) -> Value {
            json!({
                "player_index": player.player_index,
                "id": player.id,
                "is_active": player.is_active,
                "is_agent": player.is_agent,
                "life": player.life,
                "zone_counts": player.zone_counts,
            })
        }

        fn card_json(card: &CardData) -> Value {
            json!({
                "id": card.id,
                "registry_key": card.registry_key,
                "zone": card.zone as i32,
                "owner_id": card.owner_id,
                "power": card.power,
                "toughness": card.toughness,
                "card_types": {
                    "is_castable": card.card_types.is_castable,
                    "is_permanent": card.card_types.is_permanent,
                    "is_non_land_permanent": card.card_types.is_non_land_permanent,
                    "is_non_creature_permanent": card.card_types.is_non_creature_permanent,
                    "is_spell": card.card_types.is_spell,
                    "is_creature": card.card_types.is_creature,
                    "is_land": card.card_types.is_land,
                    "is_planeswalker": card.card_types.is_planeswalker,
                    "is_enchantment": card.card_types.is_enchantment,
                    "is_artifact": card.card_types.is_artifact,
                    "is_kindred": card.card_types.is_kindred,
                    "is_battle": card.card_types.is_battle,
                },
                "mana_cost": {
                    "cost": card.mana_cost.cost[..6]
                        .iter()
                        .map(|v| i32::from(*v))
                        .collect::<Vec<_>>(),
                    "mana_value": i32::from(card.mana_cost.mana_value),
                }
            })
        }

        fn permanent_json(permanent: &PermanentData) -> Value {
            json!({
                "id": permanent.id,
                "controller_id": permanent.controller_id,
                "tapped": permanent.tapped,
                "damage": permanent.damage,
                "is_summoning_sick": permanent.is_summoning_sick,
            })
        }

        let action_json = self
            .action_space
            .actions
            .iter()
            .map(|action| {
                json!({
                    "type": action.action_type as i32,
                    "focus": action.focus,
                })
            })
            .collect::<Vec<_>>();

        json!({
            "game_over": self.game_over,
            "won": self.won,
            "turn": {
                "turn_number": self.turn.turn_number,
                "phase": self.turn.phase as i32,
                "step": self.turn.step as i32,
                "active_player_id": self.turn.active_player_id,
                "agent_player_id": self.turn.agent_player_id,
            },
            "action_space": {
                "type": self.action_space.action_space_type as i32,
                "actions": action_json,
            },
            "agent": player_json(&self.agent),
            "agent_cards": self.agent_cards.iter().map(card_json).collect::<Vec<_>>(),
            "agent_permanents": self
                .agent_permanents
                .iter()
                .map(permanent_json)
                .collect::<Vec<_>>(),
            "opponent": player_json(&self.opponent),
            "opponent_cards": self.opponent_cards.iter().map(card_json).collect::<Vec<_>>(),
            "opponent_permanents": self
                .opponent_permanents
                .iter()
                .map(permanent_json)
                .collect::<Vec<_>>(),
        })
        .to_string()
    }
}
