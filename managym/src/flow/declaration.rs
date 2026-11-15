//! Declaration completion evidence and private transaction candidates.
//!
//! Completion is captured before declaration triggers and forced consequences.
//! It is output only: no collector, draft or receipt enters game-state identity.

use crate::{
    agent::action::AgentError,
    decision::{Command, SemanticTransition},
    experience::{DecisionObjectView, ObjectRenderId, RequiredNullable, SubjectRef},
    flow::{game::Game, turn::StepKind},
    state::game_object::{ObjectId, PlayerId},
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeclarationStage {
    Attackers,
    Blockers,
}

/// Engine address for one declaration, separate from transport revisions.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeclarationBinding {
    pub revision: u64,
    pub fingerprint: String,
}

/// One visible creature and its complete pair/destination supports. Supports
/// describe individual eligibility; the whole plan still needs native validation.
#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeclarationSubject {
    pub object: DecisionObjectView,
    pub power: i32,
    pub destinations: Vec<SubjectRef>,
    pub fixed: bool,
    pub assigned_to: Option<SubjectRef>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeclarationProjection {
    pub binding: DeclarationBinding,
    pub stage: DeclarationStage,
    pub actor: u8,
    pub rules_player: u8,
    pub subjects: Vec<DeclarationSubject>,
    pub destinations: Vec<DecisionObjectView>,
    pub constraints: Vec<String>,
}

/// Every editable subject must appear once, including explicit declines.
#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeclarationAnswer {
    pub subject: ObjectRenderId,
    #[serde(deserialize_with = "crate::experience::deserialize_required_nullable")]
    pub destination: RequiredNullable<SubjectRef>,
}

#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeclarationPlan {
    pub binding: DeclarationBinding,
    pub assignments: Vec<DeclarationAnswer>,
}

/// Viewer-safe completion evidence, independent of the board after consequences.
#[derive(Clone, Debug, Deserialize, JsonSchema, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeclarationCommitment {
    pub stage: DeclarationStage,
    pub rules_player: crate::experience::PlayerId,
    pub assignments: Vec<DeclarationAnswer>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DeclarationAssignments {
    Attackers {
        destination: PlayerId,
        attackers: Vec<ObjectId>,
    },
    Blockers {
        /// Exact blocker appearance followed by its exact attacker appearance.
        assignments: Vec<(ObjectId, ObjectId)>,
    },
}

/// One pending blocker's explicit assignment, bound to exact appearances.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockerAssignment {
    pub blocker: ObjectId,
    pub attacker: Option<ObjectId>,
}

/// The complete committed declaration, including any previously committed prefix.
/// Absence from these assignments means unassigned; comparison uses the bound
/// prompt's complete subject domain, including its explicitly declined subjects.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeclarationCompletion {
    pub player: PlayerId,
    pub assignments: DeclarationAssignments,
}

impl DeclarationCompletion {
    pub(crate) fn from_game(game: &Game, stage: DeclarationStage) -> Result<Self, AgentError> {
        let combat = game.state.combat.as_ref().expect("declaration has combat");
        let object = |id| {
            game.permanent_object(id).ok_or_else(|| {
                AgentError::Rules("declared object left before declaration completion".into())
            })
        };
        Ok(match stage {
            DeclarationStage::Attackers => Self {
                player: game.active_player(),
                assignments: DeclarationAssignments::Attackers {
                    destination: game.non_active_player(),
                    attackers: combat
                        .attackers
                        .iter()
                        .copied()
                        .map(object)
                        .collect::<Result<_, _>>()?,
                },
            },
            DeclarationStage::Blockers => Self {
                player: game.non_active_player(),
                assignments: DeclarationAssignments::Blockers {
                    assignments: combat
                        .attacker_to_blockers
                        .iter()
                        .flat_map(|(attacker, blockers)| {
                            blockers
                                .iter()
                                .map(|blocker| Ok((object(*blocker)?, object(*attacker)?)))
                        })
                        .collect::<Result<_, AgentError>>()?,
                },
            },
        })
    }

    fn stage(&self) -> DeclarationStage {
        match self.assignments {
            DeclarationAssignments::Attackers { .. } => DeclarationStage::Attackers,
            DeclarationAssignments::Blockers { .. } => DeclarationStage::Blockers,
        }
    }
}

/// Attached only while executing children of one bound root prompt. Once its
/// declaration finishes, later forced declarations cannot replace its evidence.
pub(crate) struct DeclarationCollector {
    stage: DeclarationStage,
    pub(crate) closed: bool,
    pub(crate) completion: Option<DeclarationCompletion>,
}

impl DeclarationCollector {
    pub(crate) fn new(stage: DeclarationStage) -> Self {
        Self {
            stage,
            closed: false,
            completion: None,
        }
    }

    pub(crate) fn observe(&mut self, completion: DeclarationCompletion) -> Result<(), AgentError> {
        if self.closed || completion.stage() != self.stage {
            return Ok(());
        }
        if self.completion.is_some() {
            return Err(AgentError::Rules(
                "originating declaration completed more than once".into(),
            ));
        }
        self.completion = Some(completion);
        Ok(())
    }

    pub(crate) fn close(&mut self, step: StepKind) {
        if matches!(
            (self.stage, step),
            (DeclarationStage::Attackers, StepKind::DeclareAttackers)
                | (DeclarationStage::Blockers, StepKind::DeclareBlockers)
        ) {
            self.closed = true;
        }
    }
}

/// One ordinary choice with its exact recording root and returned evidence.
/// Keeping these together prevents recording from pairing different children.
pub struct DeclarationChild {
    pub root: Game,
    pub command: Command,
    pub transition: SemanticTransition,
    pub observation: crate::agent::observation::Observation,
}

/// Prepared on a private live candidate; callers can stage recording before
/// publishing `game`. Child transitions retain ordinary replay/fork authority.
pub struct DeclarationTransaction {
    pub game: Game,
    pub completion: DeclarationCompletion,
    /// Stage ordinary replay/Study rows without executing the choices again.
    pub children: Vec<DeclarationChild>,
    pub done: bool,
}

impl DeclarationTransaction {
    /// Project exact completion through the original root, so a trigger removing
    /// or changing a declared object cannot change its recorded public address.
    pub fn commitment(&self) -> Result<DeclarationCommitment, AgentError> {
        use std::collections::BTreeMap;
        let root = &self
            .children
            .first()
            .ok_or_else(|| AgentError::Rules("declaration has no child root".into()))?
            .root;
        let actor = root
            .action_space()
            .and_then(|space| space.player)
            .ok_or_else(|| AgentError::Rules("declaration root has no chooser".into()))?;
        let projection = root
            .declaration_projection(actor)
            .map_err(AgentError::from)?
            .ok_or_else(|| AgentError::Rules("declaration root has no projection".into()))?;
        let public = |object| {
            root.decision_object(object)
                .map(|view| view.subject)
                .ok_or_else(|| {
                    AgentError::Rules("completion object is absent from its root".into())
                })
        };
        let key = |subject| match subject {
            SubjectRef::Object { id } => Ok((id.entity, id.incarnation)),
            _ => Err(AgentError::Rules(
                "declaration subject is not a creature".into(),
            )),
        };
        let mut committed = BTreeMap::new();
        match &self.completion.assignments {
            DeclarationAssignments::Attackers {
                destination,
                attackers,
            } => {
                for attacker in attackers {
                    committed.insert(
                        key(public(*attacker)?)?,
                        SubjectRef::Player {
                            id: crate::experience::PlayerId(destination.0 as u8),
                        },
                    );
                }
            }
            DeclarationAssignments::Blockers { assignments } => {
                for (blocker, attacker) in assignments {
                    committed.insert(key(public(*blocker)?)?, public(*attacker)?);
                }
            }
        }
        let assignments = projection
            .subjects
            .into_iter()
            .map(|row| {
                let SubjectRef::Object { id } = row.object.subject else {
                    return Err(AgentError::Rules(
                        "declaration subject is not a creature".into(),
                    ));
                };
                Ok(DeclarationAnswer {
                    subject: id,
                    destination: RequiredNullable(committed.remove(&(id.entity, id.incarnation))),
                })
            })
            .collect::<Result<_, _>>()?;
        if !committed.is_empty() {
            return Err(AgentError::Rules(
                "completion contains an object outside the bound domain".into(),
            ));
        }
        Ok(DeclarationCommitment {
            stage: projection.stage,
            rules_player: crate::experience::PlayerId(self.completion.player.0 as u8),
            assignments,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty() -> DeclarationCompletion {
        DeclarationCompletion {
            player: PlayerId(0),
            assignments: DeclarationAssignments::Attackers {
                destination: PlayerId(1),
                attackers: vec![],
            },
        }
    }

    #[test]
    fn declaration_completion_rejects_duplicates_but_ignores_later_combats() {
        let mut collector = DeclarationCollector::new(DeclarationStage::Attackers);
        collector.observe(empty()).unwrap();
        assert!(collector.observe(empty()).is_err());
        collector.close(StepKind::DeclareAttackers);
        collector.observe(empty()).unwrap();
        assert_eq!(collector.completion, Some(empty()));
    }

    #[test]
    fn declaration_completion_cannot_be_supplied_by_a_later_combat() {
        let mut collector = DeclarationCollector::new(DeclarationStage::Attackers);
        collector.close(StepKind::DeclareAttackers);
        collector.observe(empty()).unwrap();
        assert!(collector.completion.is_none());
    }
}
