use crate::state::{
    game_object::{CardId, ObjectId, PermanentId, PlayerId},
    target::Target,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum ActionType {
    PriorityPlayLand = 0,
    PriorityCastSpell = 1,
    PriorityPassPriority = 2,
    DeclareAttacker = 3,
    DeclareBlocker = 4,
    ChooseTarget = 5,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Action {
    PlayLand {
        player: PlayerId,
        card: CardId,
    },
    CastSpell {
        player: PlayerId,
        card: CardId,
    },
    PassPriority {
        player: PlayerId,
    },
    DeclareAttacker {
        player: PlayerId,
        permanent: PermanentId,
        attack: bool,
    },
    DeclareBlocker {
        player: PlayerId,
        blocker: PermanentId,
        attacker: Option<PermanentId>,
    },
    ChooseTarget {
        player: PlayerId,
        target: Target,
    },
}

impl Action {
    pub fn action_type(&self) -> ActionType {
        match self {
            Action::PlayLand { .. } => ActionType::PriorityPlayLand,
            Action::CastSpell { .. } => ActionType::PriorityCastSpell,
            Action::PassPriority { .. } => ActionType::PriorityPassPriority,
            Action::DeclareAttacker { .. } => ActionType::DeclareAttacker,
            Action::DeclareBlocker { .. } => ActionType::DeclareBlocker,
            Action::ChooseTarget { .. } => ActionType::ChooseTarget,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum ActionSpaceKind {
    GameOver = 0,
    Priority = 1,
    DeclareAttacker = 2,
    DeclareBlocker = 3,
    ChooseTarget = 4,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActionSpace {
    pub player: Option<PlayerId>,
    pub kind: ActionSpaceKind,
    pub actions: Vec<Action>,
    pub focus: Vec<ObjectId>,
}

impl ActionSpace {
    pub fn game_over() -> Self {
        Self {
            player: None,
            kind: ActionSpaceKind::GameOver,
            actions: Vec::new(),
            focus: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AgentError(pub String);

impl std::fmt::Display for AgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for AgentError {}
