//! Viewer-safe refusals from the same predicates that publish legal actions.
//! These explain a contemplated action at this decision; they are never offers.
use crate::{
    agent::{
        action::{Action, ActionSpaceKind},
        observation::Observation,
    },
    flow::game::Game,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum UnavailableReason {
    ZonePermission,
    Timing,
    LandPlay,
    NotPlayable,
    NoLegalMode,
    NoLegalTarget,
    TotalCost,
    NotCreature,
    Tapped,
    SummoningSick,
    Defender,
    CombatRestriction,
    PaymentNeedsActivation,
}

impl UnavailableReason {
    pub const ALL: [Self; 13] = [
        Self::ZonePermission,
        Self::Timing,
        Self::LandPlay,
        Self::NotPlayable,
        Self::NoLegalMode,
        Self::NoLegalTarget,
        Self::TotalCost,
        Self::NotCreature,
        Self::Tapped,
        Self::SummoningSick,
        Self::Defender,
        Self::CombatRestriction,
        Self::PaymentNeedsActivation,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::ZonePermission => "ZonePermission",
            Self::Timing => "Timing",
            Self::LandPlay => "LandPlay",
            Self::NotPlayable => "NotPlayable",
            Self::NoLegalMode => "NoLegalMode",
            Self::NoLegalTarget => "NoLegalTarget",
            Self::TotalCost => "TotalCost",
            Self::NotCreature => "NotCreature",
            Self::Tapped => "Tapped",
            Self::SummoningSick => "SummoningSick",
            Self::Defender => "Defender",
            Self::CombatRestriction => "CombatRestriction",
            Self::PaymentNeedsActivation => "PaymentNeedsActivation",
        }
    }

    pub fn message(self) -> &'static str {
        match self {
            Self::ZonePermission => "No permission to play this card from its current zone.",
            Self::Timing => "Sorcery timing is required: the player playing this card must be in their main phase with an empty stack.",
            Self::LandPlay => "No land plays remain this turn.",
            Self::NotPlayable => "This object cannot be played as a card.",
            Self::NoLegalMode => "No spell mode has the required legal targets.",
            Self::NoLegalTarget => "A required target has no legal candidate.",
            Self::TotalCost => "The complete casting cost cannot be paid, including additional costs.",
            Self::NotCreature => "Only creatures can attack.",
            Self::Tapped => "This creature is tapped.",
            Self::SummoningSick => "This creature cannot attack yet: it has summoning sickness and no haste.",
            Self::Defender => "Defender prevents this creature from attacking.",
            Self::CombatRestriction => "A current combat restriction prevents this creature from attacking.",
            Self::PaymentNeedsActivation => "The remaining cost cannot be paid automatically. Choose a legal mana activation first.",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "python", pyo3::pyclass(frozen))]
pub struct UnavailableAction {
    /// Exact public appearance, or the rules actor for the payment prompt.
    pub object: i32,
    pub reason: UnavailableReason,
}

#[cfg(feature = "python")]
#[pyo3::pymethods]
impl UnavailableAction {
    #[getter]
    fn object(&self) -> i32 {
        self.object
    }
    #[getter]
    fn reason(&self) -> &'static str {
        self.reason.name()
    }
    #[getter]
    fn message(&self) -> &'static str {
        self.reason.message()
    }
}

pub(super) fn project(
    game: &Game,
    obs: &Observation,
    expose_actions: bool,
) -> Vec<UnavailableAction> {
    let Some(space) = game
        .current_action_space
        .as_ref()
        .filter(|_| expose_actions)
    else {
        return Vec::new();
    };
    let Some(actor) = space.actions.first().map(Action::player) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    match space.kind {
        ActionSpaceKind::Priority => {
            // Candidate enumeration is the native permission boundary, including
            // a visible library top and granted exile cards. The observation
            // check prevents explanations from revealing an unavailable object.
            let _payments = game.payment_scan();
            let visible: std::collections::BTreeSet<i32> = obs
                .agent_cards
                .iter()
                .chain(&obs.opponent_cards)
                .map(|card| card.id)
                .collect();
            let playable: std::collections::BTreeSet<i32> = space
                .actions
                .iter()
                .filter_map(|action| match action {
                    Action::CastSpell { object, .. } | Action::PlayLand { object, .. } => {
                        Some(game.state[*object].id.0 as i32)
                    }
                    _ => None,
                })
                .collect();
            for card in game.priority_candidate_cards(actor) {
                let object = game.state[game.state.object_of(card)].id.0 as i32;
                if !visible.contains(&object) || playable.contains(&object) {
                    continue;
                }
                if let Err(reason) = game.priority_card_action(actor, card) {
                    out.push(UnavailableAction { object, reason });
                }
            }
        }
        ActionSpaceKind::DeclareAttacker => {
            for permanent in game.battlefield_permanents(actor) {
                if !game.characteristics(permanent).is_creature() {
                    continue;
                }
                if let Err(reason) = game.attacker_eligibility(permanent) {
                    let object = game.state.permanents[permanent].as_ref().unwrap().id.0 as i32;
                    out.push(UnavailableAction { object, reason });
                }
            }
        }
        ActionSpaceKind::PayOrNot
            if matches!(
                game.pending_choice,
                Some(crate::flow::game::PendingChoice::ManaPayment { .. })
            ) =>
        {
            if !space
                .actions
                .iter()
                .any(|action| matches!(action, Action::PayCost { .. }))
            {
                out.push(UnavailableAction {
                    object: game.state.players[actor.0].id.0 as i32,
                    reason: UnavailableReason::PaymentNeedsActivation,
                });
            }
        }
        _ => {}
    }
    out
}
