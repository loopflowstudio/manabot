use crate::state::{
    game_object::{CardId, ObjectId, PermanentId, PlayerId},
    target::Target,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[repr(i32)]
pub enum ActionType {
    PriorityPlayLand = 0,
    PriorityCastSpell = 1,
    PriorityPassPriority = 2,
    DeclareAttacker = 3,
    DeclareBlocker = 4,
    ChooseTarget = 5,
    PriorityActivateAbility = 6,
    /// Scry: keep this card on top of the library.
    ScryKeep = 7,
    /// Scry: put this card on the bottom of the library.
    ScryBottom = 8,
    /// Pick a card (look-and-select to hand, discard for learn).
    SelectCard = 9,
    /// Decline / done: finish an optional selection, decline to pay,
    /// stop choosing "up to N" targets.
    DeclineChoice = 10,
    /// Pay an optional or demanded cost (kicker, ward, "unless ... pays",
    /// waterbend mana remainder).
    PayCost = 11,
    /// Choose a mode of a modal effect.
    ChooseMode = 12,
    /// Tap a permanent to pay {1} of a waterbend cost.
    TapForCost = 13,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Action {
    PlayLand {
        player: PlayerId,
        card: CardId,
    },
    CastSpell {
        player: PlayerId,
        card: CardId,
    },
    ActivateAbility {
        player: PlayerId,
        permanent: PermanentId,
        ability_index: usize,
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
    /// Scry decision for the current top revealed card.
    ScryCard {
        player: PlayerId,
        card: CardId,
        to_bottom: bool,
    },
    /// Select a card in a look-and-select or discard decision.
    SelectCard {
        player: PlayerId,
        card: CardId,
    },
    /// Decline / finish the current optional choice.
    Decline {
        player: PlayerId,
    },
    /// Pay the pending optional/demanded cost.
    PayCost {
        player: PlayerId,
    },
    /// Choose mode `mode` of the pending modal decision.
    ChooseMode {
        player: PlayerId,
        mode: usize,
    },
    /// Tap `permanent` to pay {1} of a pending waterbend cost.
    WaterbendTap {
        player: PlayerId,
        permanent: PermanentId,
    },
}

impl Action {
    pub fn action_type(&self) -> ActionType {
        match self {
            Action::PlayLand { .. } => ActionType::PriorityPlayLand,
            Action::CastSpell { .. } => ActionType::PriorityCastSpell,
            Action::ActivateAbility { .. } => ActionType::PriorityActivateAbility,
            Action::PassPriority { .. } => ActionType::PriorityPassPriority,
            Action::DeclareAttacker { .. } => ActionType::DeclareAttacker,
            Action::DeclareBlocker { .. } => ActionType::DeclareBlocker,
            Action::ChooseTarget { .. } => ActionType::ChooseTarget,
            Action::ScryCard { to_bottom, .. } => {
                if *to_bottom {
                    ActionType::ScryBottom
                } else {
                    ActionType::ScryKeep
                }
            }
            Action::SelectCard { .. } => ActionType::SelectCard,
            Action::Decline { .. } => ActionType::DeclineChoice,
            Action::PayCost { .. } => ActionType::PayCost,
            Action::ChooseMode { .. } => ActionType::ChooseMode,
            Action::WaterbendTap { .. } => ActionType::TapForCost,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[repr(i32)]
pub enum ActionSpaceKind {
    GameOver = 0,
    Priority = 1,
    DeclareAttacker = 2,
    DeclareBlocker = 3,
    ChooseTarget = 4,
    /// Mid-resolution scry: keep or bottom, one card at a time.
    Scry = 5,
    /// Mid-resolution look-at-top-N, select up to K matching to hand.
    LookAndSelect = 6,
    /// "You may pay [cost]" — kicker at cast, ward / "unless ... pays" at
    /// resolution.
    PayOrNot = 7,
    /// "Choose one —" modal effect.
    Modal = 8,
    /// Learn without a sideboard: optionally discard, then draw.
    DiscardThenDraw = 9,
    /// Waterbend cost payment: tap permanents / pay the remainder.
    Waterbend = 10,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
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
