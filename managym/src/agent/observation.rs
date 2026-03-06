use crate::{
    agent::action::{Action, ActionSpaceKind, ActionType},
    flow::{
        event::GameEvent,
        game::Game,
        turn::{PhaseKind, StepKind},
    },
    state::{
        card::Card,
        game_object::{CardId, ObjectId, PlayerId, Target},
        mana::ManaCost,
        permanent::Permanent,
        stack_object::StackObject,
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
    pub name: String,
    pub power: i32,
    pub toughness: i32,
    pub card_types: CardTypeData,
    pub keywords: KeywordData,
    pub mana_cost: ManaCost,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KeywordData {
    pub flying: bool,
    pub reach: bool,
    pub haste: bool,
    pub vigilance: bool,
    pub trample: bool,
    pub first_strike: bool,
    pub double_strike: bool,
    pub deathtouch: bool,
    pub lifelink: bool,
    pub defender: bool,
    pub menace: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum StackObjectKindData {
    Spell = 0,
    ActivatedAbility = 1,
    TriggeredAbility = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum StackTargetKindData {
    Player = 0,
    Permanent = 1,
    StackObject = 2,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StackTargetData {
    pub kind: StackTargetKindData,
    pub player_id: Option<i32>,
    pub permanent_id: Option<i32>,
    pub stack_object_id: Option<i32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StackObjectData {
    pub stack_object_id: i32,
    pub kind: StackObjectKindData,
    pub controller_id: i32,
    pub source_card_registry_key: i32,
    pub source_permanent_id: Option<i32>,
    pub ability_index: Option<i32>,
    pub targets: Vec<StackTargetData>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PermanentData {
    pub id: i32,
    pub controller_id: i32,
    pub tapped: bool,
    pub damage: i32,
    pub is_summoning_sick: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum EventType {
    CardMoved = 0,
    DamageDealt = 1,
    LifeChanged = 2,
    SpellCast = 3,
    SpellResolved = 4,
    SpellCountered = 5,
    AbilityTriggered = 6,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum EventEntityKind {
    None = 0,
    Card = 1,
    Permanent = 2,
    Player = 3,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EventData {
    pub event_type: i32,
    pub source_kind: i32,
    pub source_id: i32,
    pub target_kind: i32,
    pub target_id: i32,
    pub amount: i32,
    pub controller_id: i32,
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
    pub stack_objects: Vec<StackObjectData>,
    pub recent_events: Vec<EventData>,
}

impl Observation {
    pub fn new(game: &Game, recent_events: &[GameEvent]) -> Self {
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
            stack_objects: Vec::new(),
            recent_events: recent_events.iter().map(Self::event_data).collect(),
        };

        obs.populate_cards(game, agent_player);
        obs.populate_permanents(game);
        obs.populate_stack_objects(game);
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

        for stack_object in game.state.stack_objects.iter().rev() {
            if let StackObject::Spell(spell) = stack_object {
                self.add_card(game, spell.card, ZoneType::Stack);
            }
        }
    }

    fn populate_permanents(&mut self, game: &Game) {
        for player in game.players_starting_with_agent() {
            for card_id in game.state.zones.zone_cards(ZoneType::Battlefield, player) {
                if let Some(perm_id) = game.state.card_to_permanent[card_id] {
                    let Some(permanent) = game.state.permanents[perm_id].as_ref() else {
                        continue;
                    };
                    self.add_permanent(game, permanent);
                }
            }
        }
    }

    fn populate_stack_objects(&mut self, game: &Game) {
        for object in game.state.stack_objects.iter().rev() {
            self.stack_objects
                .push(Self::stack_object_data(game, object));
        }
    }

    fn add_card(
        &mut self,
        game: &Game,
        card_id: crate::state::game_object::CardId,
        zone: ZoneType,
    ) {
        let card = &game.state.cards[card_id];
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
            name: card.name.clone(),
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
            keywords: KeywordData {
                flying: card.keywords.flying,
                reach: card.keywords.reach,
                haste: card.keywords.haste,
                vigilance: card.keywords.vigilance,
                trample: card.keywords.trample,
                first_strike: card.keywords.first_strike,
                double_strike: card.keywords.double_strike,
                deathtouch: card.keywords.deathtouch,
                lifelink: card.keywords.lifelink,
                defender: card.keywords.defender,
                menace: card.keywords.menace,
            },
            mana_cost: card.mana_cost.clone().unwrap_or_default(),
        }
    }

    fn stack_object_data(game: &Game, stack_object: &StackObject) -> StackObjectData {
        StackObjectData {
            stack_object_id: stack_object.id().0 as i32,
            kind: match stack_object {
                StackObject::Spell(_) => StackObjectKindData::Spell,
                StackObject::ActivatedAbility(_) => StackObjectKindData::ActivatedAbility,
                StackObject::TriggeredAbility(_) => StackObjectKindData::TriggeredAbility,
            },
            controller_id: game.state.players[stack_object.controller().0].id.0 as i32,
            source_card_registry_key: stack_object.source_card_registry_key().0 as i32,
            source_permanent_id: stack_object
                .source_permanent_object_id()
                .map(|id| id.0 as i32),
            ability_index: stack_object.ability_index().map(|index| index as i32),
            targets: stack_object
                .targets()
                .iter()
                .filter_map(|target| Self::stack_target_data(game, *target))
                .collect(),
        }
    }

    fn stack_target_data(game: &Game, target: Target) -> Option<StackTargetData> {
        match target {
            Target::Player(player) => Some(StackTargetData {
                kind: StackTargetKindData::Player,
                player_id: Some(game.state.players[player.0].id.0 as i32),
                permanent_id: None,
                stack_object_id: None,
            }),
            Target::Permanent(permanent_id) => {
                let permanent = game.state.permanents[permanent_id].as_ref()?;
                Some(StackTargetData {
                    kind: StackTargetKindData::Permanent,
                    player_id: None,
                    permanent_id: Some(permanent.id.0 as i32),
                    stack_object_id: None,
                })
            }
            Target::StackSpell(card_id) => {
                let stack_object_id = Self::stack_spell_object_id(game, card_id)?.0 as i32;
                Some(StackTargetData {
                    kind: StackTargetKindData::StackObject,
                    player_id: None,
                    permanent_id: None,
                    stack_object_id: Some(stack_object_id),
                })
            }
        }
    }

    fn stack_spell_object_id(game: &Game, card_id: CardId) -> Option<ObjectId> {
        game.state
            .stack_objects
            .iter()
            .find_map(|object| match object {
                StackObject::Spell(spell) if spell.card == card_id => Some(spell.id),
                _ => None,
            })
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

    fn event_data(event: &GameEvent) -> EventData {
        use crate::flow::event::DamageTarget;
        match event {
            GameEvent::CardMoved {
                card, controller, ..
            } => Self::simple_event(EventType::CardMoved, card.0, controller.0),
            GameEvent::DamageDealt {
                source,
                target,
                amount,
            } => {
                let (source_kind, source_id) = match source {
                    Some(card) => (EventEntityKind::Card as i32, card.0 as i32),
                    None => (EventEntityKind::None as i32, -1),
                };
                let (target_kind, target_id) = match target {
                    DamageTarget::Player(p) => (EventEntityKind::Player as i32, p.0 as i32),
                    DamageTarget::Permanent(p) => {
                        (EventEntityKind::Permanent as i32, p.0 as i32)
                    }
                };
                EventData {
                    event_type: EventType::DamageDealt as i32,
                    source_kind,
                    source_id,
                    target_kind,
                    target_id,
                    amount: *amount as i32,
                    controller_id: -1,
                }
            }
            GameEvent::LifeChanged { player, old, new } => EventData {
                event_type: EventType::LifeChanged as i32,
                source_kind: EventEntityKind::Player as i32,
                source_id: player.0 as i32,
                target_kind: EventEntityKind::Player as i32,
                target_id: player.0 as i32,
                amount: new - old,
                controller_id: -1,
            },
            GameEvent::SpellCast { card, .. } => {
                Self::simple_event(EventType::SpellCast, card.0, card.0)
            }
            GameEvent::SpellResolved { card } => {
                Self::simple_event(EventType::SpellResolved, card.0, card.0)
            }
            GameEvent::SpellCountered { card, .. } => {
                Self::simple_event(EventType::SpellCountered, card.0, card.0)
            }
            GameEvent::AbilityTriggered {
                source_card,
                controller,
            } => Self::simple_event(EventType::AbilityTriggered, source_card.0, controller.0),
            GameEvent::TurnStarted { .. } | GameEvent::StepStarted { .. } => EventData {
                event_type: -1,
                source_kind: EventEntityKind::None as i32,
                source_id: -1,
                target_kind: EventEntityKind::None as i32,
                target_id: -1,
                amount: 0,
                controller_id: -1,
            },
        }
    }

    fn simple_event(event_type: EventType, card_id: usize, controller_idx: usize) -> EventData {
        EventData {
            event_type: event_type as i32,
            source_kind: EventEntityKind::Card as i32,
            source_id: card_id as i32,
            target_kind: EventEntityKind::None as i32,
            target_id: -1,
            amount: 0,
            controller_id: controller_idx as i32,
        }
    }

    fn action_focus(game: &Game, action: &Action) -> Vec<ObjectId> {
        match action {
            Action::PlayLand { card, .. } | Action::CastSpell { card, .. } => {
                vec![game.state.cards[card].id]
            }
            Action::ActivateAbility { permanent, .. } => game.state.permanents[*permanent]
                .as_ref()
                .map(|perm| vec![perm.id])
                .unwrap_or_default(),
            Action::PassPriority { .. } => vec![],
            Action::DeclareAttacker { permanent, .. } => game.state.permanents[permanent]
                .as_ref()
                .map(|perm| vec![perm.id])
                .unwrap_or_default(),
            Action::DeclareBlocker {
                blocker, attacker, ..
            } => {
                let mut focus = Vec::new();
                if let Some(perm) = game.state.permanents[blocker].as_ref() {
                    focus.push(perm.id);
                }
                if let Some(attacker) = attacker {
                    if let Some(perm) = game.state.permanents[attacker].as_ref() {
                        focus.push(perm.id);
                    }
                }
                focus
            }
            Action::ChooseTarget { target, .. } => match target {
                crate::state::target::Target::Player(player) => {
                    vec![game.state.players[player.0].id]
                }
                crate::state::target::Target::Permanent(permanent_id) => game.state.permanents
                    [*permanent_id]
                    .as_ref()
                    .map(|perm| vec![perm.id])
                    .unwrap_or_default(),
                crate::state::target::Target::StackSpell(card_id) => {
                    vec![game.state.cards[card_id].id]
                }
            },
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
                "name": card.name,
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
                "keywords": {
                    "flying": card.keywords.flying,
                    "reach": card.keywords.reach,
                    "haste": card.keywords.haste,
                    "vigilance": card.keywords.vigilance,
                    "trample": card.keywords.trample,
                    "first_strike": card.keywords.first_strike,
                    "double_strike": card.keywords.double_strike,
                    "deathtouch": card.keywords.deathtouch,
                    "lifelink": card.keywords.lifelink,
                    "defender": card.keywords.defender,
                    "menace": card.keywords.menace,
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

        fn stack_target_json(target: &StackTargetData) -> Value {
            json!({
                "kind": target.kind as i32,
                "player_id": target.player_id,
                "permanent_id": target.permanent_id,
                "stack_object_id": target.stack_object_id,
            })
        }

        fn stack_object_json(stack_object: &StackObjectData) -> Value {
            json!({
                "stack_object_id": stack_object.stack_object_id,
                "kind": stack_object.kind as i32,
                "controller_id": stack_object.controller_id,
                "source_card_registry_key": stack_object.source_card_registry_key,
                "source_permanent_id": stack_object.source_permanent_id,
                "ability_index": stack_object.ability_index,
                "targets": stack_object.targets.iter().map(stack_target_json).collect::<Vec<_>>(),
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
            "stack_objects": self
                .stack_objects
                .iter()
                .map(stack_object_json)
                .collect::<Vec<_>>(),
            "recent_events": self.recent_events.iter().map(|event| {
                json!({
                    "event_type": event.event_type,
                    "source_kind": event.source_kind,
                    "source_id": event.source_id,
                    "target_kind": event.target_kind,
                    "target_id": event.target_id,
                    "amount": event.amount,
                    "controller_id": event.controller_id,
                })
            }).collect::<Vec<_>>(),
        })
        .to_string()
    }
}
