//! Current stack and unfinished decisions, projected once from rules authority.
//! References link only to exact, visible incarnations. Tensor rows and portable
//! text evidence consume these same facts; no strings or JSON enter encoding.

use serde::{Deserialize, Serialize};

use crate::{
    flow::{
        decision::{Decision, SuspendedResolution},
        game::{ActivationSource, Game, Payment, PendingChoice},
    },
    state::{
        ability::PaymentCost,
        game_object::{CardId, ObjectRef, PlayerId, Target},
        mana::ManaCost,
        stack_object::StackObject,
        zone::ZoneType,
    },
};

/// Policy inputs v2/v3/v4: up to 64 relation rows, 32/33 floats per row. Overflow is an
/// admission error, never truncation. Zero is padding/absence for selectors.
pub const RELATION_ROWS: usize = 64;
pub const LEGACY_RELATION_DIM: usize = 32;
pub const RELATION_DIM: usize = 33;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::pyclass(get_all, frozen))]
#[serde(deny_unknown_fields)]
pub struct VisibleReference {
    pub object: Option<i32>,
    pub definition: Option<u32>,
    pub player: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::pyclass(get_all, frozen))]
#[serde(deny_unknown_fields)]
pub struct ClauseTarget {
    pub reference: VisibleReference,
    pub clause: usize,
    /// Retained CR 608.2b result, not current presence. Missing historical data is unknown.
    #[serde(default)]
    pub legal_at_resolution_start: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::pyclass(get_all, frozen))]
#[serde(deny_unknown_fields)]
pub struct ManaOutput {
    pub mode: usize,
    pub mana: [u8; 6],
    /// Spend-restricted output by color, whatever its restriction.
    pub lesson: [u8; 6],
}

/// Public stack entries precede the viewer's private announcement/resolution
/// frames. Parent indices retain interrupted payment relationships.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::pyclass(get_all, frozen))]
#[serde(deny_unknown_fields)]
pub struct ContinuationFrame {
    /// 1 spell, 2 activated stack ability, 3 triggered stack ability,
    /// 4 announcement, 5 resolution, 6 obligation, 7 waiting trigger,
    /// 8 trigger being placed, 9 two-pass placement cursor.
    pub kind: u8,
    pub decision: u8,
    pub parent: Option<usize>,
    pub controller: usize,
    pub source: VisibleReference,
    pub mode: Option<usize>,
    pub ability: Option<usize>,
    pub kicked: bool,
    pub active_clause: Option<usize>,
    pub targets: Vec<ClauseTarget>,
    pub context: Option<VisibleReference>,
    pub sacrifice: Option<VisibleReference>,
    pub cost: Option<[u8; 7]>,
    pub hybrid: Vec<[u8; 2]>,
    /// The payment has a purpose (a cast spell or an activation), so some
    /// spend-restricted mana may pay it. Named for the first restriction.
    pub lesson: bool,
    pub waterbend: bool,
    pub can_decline: bool,
    pub mana_outputs: Vec<ManaOutput>,
    pub resolutions_this_turn: u32,
}

impl ContinuationFrame {
    fn cost(&mut self, cost: &ManaCost) {
        self.cost = Some(cost.cost);
        self.hybrid = cost
            .hybrid
            .iter()
            .map(|pair| [pair[0] as u8, pair[1] as u8])
            .collect();
        self.lesson = !cost.purpose.is_ordinary();
    }
}

impl Game {
    pub(crate) fn visible_reference(
        &self,
        object: ObjectRef,
        viewer: PlayerId,
    ) -> VisibleReference {
        let card_id = CardId::from(object.entity);
        if self.current_object_ref(card_id) != Some(object) {
            return VisibleReference {
                definition: self.object_lki(object).map(|lki| lki.definition_id.0),
                ..VisibleReference::default()
            };
        }
        let card = &self.state.cards[card_id];
        let zone = self.state.zones.zone_of(card_id);
        let visible = matches!(
            zone,
            Some(
                ZoneType::Battlefield
                    | ZoneType::Graveyard
                    | ZoneType::Stack
                    | ZoneType::Exile
                    | ZoneType::Command
            )
        ) || (zone == Some(ZoneType::Hand) && card.owner == viewer);
        if !visible {
            return VisibleReference::default();
        }
        VisibleReference {
            object: Some(
                self.lookup_current_permanent(object)
                    .ok()
                    .and_then(|id| self.state.permanents[id].as_ref())
                    .map_or(card.id.0, |p| p.id.0) as i32,
            ),
            definition: Some(card.definition_id.0),
            player: None,
        }
    }

    fn card_reference(&self, card: CardId, viewer: PlayerId) -> VisibleReference {
        self.current_object_ref(card)
            .map(|object| self.visible_reference(object, viewer))
            .unwrap_or_default()
    }

    fn activation_reference(&self, source: ActivationSource, viewer: PlayerId) -> VisibleReference {
        match source {
            ActivationSource::Permanent(permanent) => {
                self.target_reference(Target::Permanent(permanent), viewer)
            }
            ActivationSource::Card(card) => self.card_reference(card, viewer),
        }
    }

    pub(crate) fn target_reference(&self, target: Target, viewer: PlayerId) -> VisibleReference {
        match target {
            Target::Player(player) => VisibleReference {
                player: Some(player.0),
                ..VisibleReference::default()
            },
            Target::Permanent(id) => self
                .permanent_object_ref(id)
                .map(|object| self.visible_reference(object, viewer))
                .unwrap_or_default(),
            Target::StackSpell(card) => {
                if self.state.zones.zone_of(card) == Some(ZoneType::Stack) {
                    self.card_reference(card, viewer)
                } else {
                    VisibleReference::default()
                }
            }
            Target::GraveyardCard(object) => self.visible_reference(object, viewer),
        }
    }

    fn clause_targets(
        &self,
        targets: &[Target],
        clauses: &[usize],
        viewer: PlayerId,
    ) -> Vec<ClauseTarget> {
        targets
            .iter()
            .enumerate()
            .map(|(index, target)| ClauseTarget {
                reference: self.target_reference(*target, viewer),
                clause: clauses.get(index).copied().unwrap_or(0),
                legal_at_resolution_start: None,
            })
            .collect()
    }

    /// The native observation owns the result. Nonacting viewers receive only
    /// public stack facts, never another player's announced private choices.
    pub fn continuation_frames(&self, viewer: PlayerId) -> Vec<ContinuationFrame> {
        // Same top-to-bottom ordering as Observation.stack_objects.
        let mut frames = Vec::new();
        for object in self.state.stack_objects.iter().rev() {
            frames.push(self.stack_frame(object, viewer));
        }
        if self.current_action_space.as_ref().and_then(|s| s.player) != Some(viewer)
            || self.is_game_over()
        {
            return frames;
        }
        if let Some(pending) = &self.pending_choice {
            frames.push(self.announcement_frame(pending, viewer));
        }
        if let Some(suspended) = &self.state.suspended_decision {
            self.append_suspended_frames(suspended, viewer, &mut frames);
        }
        if let Some(placement) = &self.state.trigger_placement {
            let parent = frames.len();
            frames.push(ContinuationFrame {
                kind: 9,
                decision: placement.group + 1,
                controller: placement.controller().0,
                ..ContinuationFrame::default()
            });
            for trigger in placement
                .remaining
                .iter()
                .chain(placement.placing.iter())
                .filter(|t| t.controller == viewer)
            {
                let mut source = trigger
                    .source_ref
                    .map(|r| self.visible_reference(r, viewer))
                    .unwrap_or_default();
                source.definition = Some(trigger.source_lki.as_ref().map_or(
                    self.state.cards[trigger.source_card].definition_id.0,
                    |lki| lki.definition_id.0,
                ));
                frames.push(ContinuationFrame {
                    kind: if placement
                        .placing
                        .as_ref()
                        .is_some_and(|t| t.id == trigger.id)
                    {
                        8
                    } else {
                        7
                    },
                    parent: Some(parent),
                    controller: viewer.0,
                    source,
                    decision: match trigger.ordering_class {
                        crate::flow::trigger::TriggerOrderingClass::Ordinary => 1,
                        crate::flow::trigger::TriggerOrderingClass::AbilityTriggered => 2,
                    },
                    ability: Some(trigger.ability_index),
                    active_clause: Some(trigger.requirement_index),
                    targets: self.clause_targets(
                        &trigger.targets,
                        &trigger.target_req_indices,
                        viewer,
                    ),
                    context: trigger.context.map(|t| self.target_reference(t, viewer)),
                    ..ContinuationFrame::default()
                });
            }
        }
        frames
    }

    pub(crate) fn trigger_action_frame(
        &self,
        action: &super::action::Action,
        viewer: PlayerId,
    ) -> Option<usize> {
        let super::action::Action::ChooseTrigger { trigger_id, player } = action else {
            return None;
        };
        if *player != viewer {
            return None;
        }
        let placement = self.state.trigger_placement.as_ref()?;
        let index = placement
            .remaining
            .iter()
            .filter(|t| t.controller == viewer)
            .position(|t| t.id == *trigger_id)?;
        Some(
            self.state.stack_objects.len() + usize::from(self.pending_choice.is_some()) + 1 + index,
        )
    }

    fn stack_frame(&self, object: &StackObject, viewer: PlayerId) -> ContinuationFrame {
        let mut frame = ContinuationFrame {
            controller: object.controller().0,
            ability: object.ability_index(),
            ..ContinuationFrame::default()
        };
        let clauses: &[usize] = match object {
            StackObject::Spell(spell) => {
                frame.kind = 1;
                frame.source = self.card_reference(spell.card, viewer);
                frame.mode = spell.mode;
                frame.kicked = spell.kicked;
                &spell.target_req_indices
            }
            StackObject::ActivatedAbility(ability) => {
                frame.kind = 2;
                frame.source = self.visible_reference(ability.source_ref, viewer);
                &[]
            }
            StackObject::TriggeredAbility(ability) => {
                frame.kind = 3;
                frame.source = ability
                    .source_ref
                    .map(|r| self.visible_reference(r, viewer))
                    .unwrap_or_default();
                frame.context = ability.context.map(|t| self.target_reference(t, viewer));
                &ability.target_req_indices
            }
        };
        // Stack definitions remain public after the original source departs.
        frame.source.definition = Some(object.source_definition_id().0);
        frame.targets = self.clause_targets(object.targets(), clauses, viewer);
        frame
    }

    fn announcement_frame(&self, pending: &PendingChoice, viewer: PlayerId) -> ContinuationFrame {
        let mut frame = ContinuationFrame {
            kind: 4,
            controller: viewer.0,
            ..ContinuationFrame::default()
        };
        match pending {
            PendingChoice::ManaPayment {
                payment,
                sacrifice,
                alternative,
                ..
            } => {
                self.payment_frame(payment, viewer, &mut frame);
                frame.sacrifice =
                    sacrifice.map(|id| self.target_reference(Target::Permanent(id), viewer));
                frame.cost(&self.payment_mana_cost(viewer, payment, alternative.as_ref()));
            }
            PendingChoice::SacrificePayment { payment, .. } => {
                self.payment_frame(payment, viewer, &mut frame)
            }
            PendingChoice::ChooseTargets {
                card,
                cast,
                mode,
                requirement_index,
                chosen,
                chosen_req_indices,
                ..
            } => {
                frame.source = self.card_reference(*card, viewer);
                frame.kicked = cast.kicked;
                frame.mode = *mode;
                frame.active_clause = Some(*requirement_index);
                frame.targets = self.clause_targets(chosen, chosen_req_indices, viewer);
            }
            PendingChoice::SpellMode { card, cast, .. } => {
                frame.source = self.card_reference(*card, viewer);
                frame.kicked = cast.kicked;
            }
            PendingChoice::KickerChoice { card, .. } | PendingChoice::AnnounceX { card, .. } => {
                frame.source = self.card_reference(*card, viewer)
            }
            PendingChoice::ActivatedTarget {
                source,
                ability_index,
                chosen,
                ..
            } => {
                frame.source = self.activation_reference(*source, viewer);
                frame.ability = Some(*ability_index);
                frame.active_clause = Some(0);
                frame.targets = self.clause_targets(chosen, &[], viewer);
            }
            PendingChoice::DiscardPayment {
                source,
                ability_index,
                ..
            } => {
                frame.source = self.activation_reference(*source, viewer);
                frame.ability = Some(*ability_index);
            }
            PendingChoice::ActivationX {
                permanent,
                ability_index,
                ..
            }
            | PendingChoice::Crew {
                permanent,
                ability_index,
                ..
            } => {
                frame.source = self.target_reference(Target::Permanent(*permanent), viewer);
                frame.ability = Some(*ability_index);
            }
            PendingChoice::Waterbend {
                player,
                payment,
                tapped,
                ..
            } => {
                self.payment_frame(payment, viewer, &mut frame);
                frame.waterbend = true;
                let owed = self
                    .payment_mana_cost(*player, payment, None)
                    .reduced_generic(*tapped);
                frame.cost(&owed);
            }
        }
        frame
    }

    fn append_suspended_frames(
        &self,
        suspended: &SuspendedResolution,
        viewer: PlayerId,
        frames: &mut Vec<ContinuationFrame>,
    ) {
        let effect = &suspended.frame;
        let parent = if self.pending_choice.is_some() {
            Some(frames.len() - 1)
        } else {
            None
        };
        let frame = ContinuationFrame {
            kind: 5,
            parent,
            controller: effect.controller.0,
            source: effect
                .source_ref
                .map(|r| self.visible_reference(r, viewer))
                .unwrap_or_else(|| {
                    effect
                        .source
                        .map(|card| self.card_reference(card, viewer))
                        .unwrap_or_default()
                }),
            kicked: effect.kicked,
            resolutions_this_turn: effect.resolutions_this_turn,
            active_clause: Some(effect.active_requirement),
            targets: effect
                .targets
                .iter()
                .map(|slot| ClauseTarget {
                    reference: self.target_reference(slot.target, viewer),
                    clause: slot.requirement,
                    legal_at_resolution_start: Some(slot.legal),
                })
                .collect(),
            context: effect
                .context_target
                .map(|t| self.target_reference(t, viewer)),
            ..ContinuationFrame::default()
        };
        // The native chain starts at the active choice and follows resume
        // obligations. Reverse its projection so each interrupted obligation
        // precedes its child and the current decision is the deepest frame.
        frames.push(frame);
        let first_decision = frames.len();
        let mut decision = &suspended.decision;
        loop {
            let mut frame = ContinuationFrame {
                kind: 6,
                controller: decision.player().0,
                ..ContinuationFrame::default()
            };
            let (kind, resume) = match decision {
                Decision::Payment(progress) => {
                    frame.cost(&progress.remaining.mana_cost());
                    frame.waterbend = matches!(progress.remaining, PaymentCost::Waterbend { .. });
                    frame.can_decline = progress.can_decline();
                    (1, None)
                }
                Decision::ManaUpfront {
                    permanent,
                    ability_index,
                    resume,
                    ..
                } => {
                    frame.source = self.target_reference(Target::Permanent(*permanent), viewer);
                    frame.ability = Some(*ability_index);
                    frame.cost(&self.activated_abilities_for(*permanent)[*ability_index].mana_cost);
                    (2, resume.as_deref())
                }
                Decision::ManaChoice {
                    choices, resume, ..
                } => {
                    frame.mana_outputs = choices
                        .iter()
                        .enumerate()
                        .map(|(mode, output)| ManaOutput {
                            mode,
                            mana: output.mana,
                            lesson: output.restricted_colors(),
                        })
                        .collect();
                    (3, resume.as_deref())
                }
                Decision::Sacrifice { .. } => (4, None),
                Decision::SelectMilled { .. } => (5, None),
                Decision::SearchBasicLand { .. } | Decision::SearchLibraryToHand { .. } => {
                    (6, None)
                }
                Decision::OptionalDiscardDraw { .. } => (7, None),
                Decision::Scry { .. } => (8, None),
                Decision::LookAndSelect { .. } => (9, None),
                Decision::Modal { .. } => (10, None),
                Decision::Learn { .. } => (11, None),
                Decision::Discard { .. } => (12, None),
                Decision::LegendRule { .. } => (13, None),
                Decision::SearchDistinctLands { .. } => (14, None),
                Decision::OpponentSplitsRevealed { .. } => (15, None),
                Decision::DiscardFromRevealedHand { .. } => (16, None),
                Decision::CastWithoutPaying { .. } => (17, None),
                Decision::ExileLinked { .. } => (18, None),
            };
            frame.decision = kind;
            frames.push(frame);
            let Some(next) = resume else {
                break;
            };
            decision = next;
        }
        frames[first_decision..].reverse();
        for (index, frame) in frames.iter_mut().enumerate().skip(first_decision) {
            frame.parent = Some(index - 1);
        }
    }

    fn payment_frame(&self, payment: &Payment, viewer: PlayerId, frame: &mut ContinuationFrame) {
        match payment {
            Payment::Spell {
                card,
                cast,
                mode,
                targets,
                target_req_indices,
                ..
            } => {
                frame.source = self.card_reference(*card, viewer);
                frame.kicked = cast.kicked;
                frame.mode = *mode;
                frame.targets = self.clause_targets(targets, target_req_indices, viewer);
            }
            Payment::Activation {
                source,
                ability_index,
                targets,
                ..
            } => {
                frame.source = self.activation_reference(*source, viewer);
                frame.ability = Some(*ability_index);
                frame.targets = self.clause_targets(targets, &[], viewer);
            }
        }
    }
}

/// Write directly into the caller's reusable tensor. References use the same
/// padded object table as action focus; absent/expired links encode as zero.
pub fn encode_into(
    frames: &[ContinuationFrame],
    viewer: usize,
    objects: &std::collections::HashMap<i32, i32>,
    out: &mut [f32],
) -> Result<(), super::observation_encoder::ObservationEncodeError> {
    use super::observation_encoder::ObservationEncodeError;
    let count: usize = frames
        .iter()
        .map(|f| {
            1 + f.targets.len()
                + usize::from(f.context.is_some())
                + usize::from(f.sacrifice.is_some())
                + f.hybrid.len()
                + 2 * f.mana_outputs.len()
        })
        .sum();
    if count > RELATION_ROWS {
        return Err(ObservationEncodeError::Capacity {
            field: "continuation relations",
            capacity: RELATION_ROWS,
            actual: count,
        });
    }
    let dim = out.len() / RELATION_ROWS;
    if ![LEGACY_RELATION_DIM, RELATION_DIM].contains(&dim) || out.len() != RELATION_ROWS * dim {
        return Err(ObservationEncodeError::InvalidLength {
            field: "relations",
            expected: RELATION_ROWS * RELATION_DIM,
            actual: out.len(),
        });
    }
    out.fill(0.0);
    let mut rows = out.chunks_exact_mut(dim);
    let link = |reference: VisibleReference| {
        reference
            .object
            .and_then(|id| objects.get(&id))
            .map_or(0.0, |index| (index + 1) as f32)
    };
    let definition =
        |reference: VisibleReference| reference.definition.map_or(0.0, |id| (id + 1) as f32);
    for (index, frame) in frames.iter().enumerate() {
        let mut header = [0.0; RELATION_DIM];
        header[0] = 1.0;
        header[1] = (index + 1) as f32;
        header[2] = frame.parent.map_or(0.0, |p| (p + 1) as f32);
        header[3] = frame.kind as f32;
        header[4] = if frame.controller == viewer { 1.0 } else { 2.0 };
        header[5] = definition(frame.source);
        header[6] = link(frame.source);
        header[7] = frame.mode.map_or(0.0, |v| (v + 1) as f32);
        header[8] = frame.ability.map_or(0.0, |v| (v + 1) as f32);
        header[9] = f32::from(frame.kicked);
        header[10] = frame.active_clause.map_or(0.0, |v| (v + 1) as f32);
        header[11] = f32::from(frame.cost.is_some());
        if let Some(cost) = frame.cost {
            for (i, amount) in cost.iter().enumerate() {
                header[12 + i] = *amount as f32;
            }
        }
        header[19] = f32::from(frame.lesson);
        header[20] = f32::from(frame.waterbend);
        header[21] = f32::from(frame.can_decline);
        header[28] = frame.decision as f32;
        header[30] = frame.resolutions_this_turn as f32;
        header[31] = 1.0;
        rows.next().unwrap().copy_from_slice(&header[..dim]);
        let mut reference_row = |kind: f32,
                                 reference: VisibleReference,
                                 clause: usize,
                                 ordinal: usize,
                                 legal: Option<bool>| {
            let row = rows.next().unwrap();
            row.copy_from_slice(&header[..dim]);
            row[0] = kind;
            if dim == RELATION_DIM {
                row[32] = legal.map_or(0.0, |value| if value { 1.0 } else { 2.0 });
            }
            row[22] = (clause + 1) as f32;
            row[29] = (ordinal + 1) as f32;
            row[23] = definition(reference);
            row[24] = link(reference);
            row[25] = reference
                .player
                .map_or(0.0, |p| if p == viewer { 1.0 } else { 2.0 });
            if let Some(player) = reference.player {
                row[24] = if player == viewer { 1.0 } else { 2.0 };
            }
        };
        for (ordinal, target) in frame.targets.iter().enumerate() {
            reference_row(
                2.0,
                target.reference,
                target.clause,
                ordinal,
                target.legal_at_resolution_start,
            );
        }
        if let Some(target) = frame.context {
            reference_row(3.0, target, 0, 0, None);
        }
        if let Some(target) = frame.sacrifice {
            reference_row(4.0, target, 0, 0, None);
        }
        for pair in &frame.hybrid {
            let row = rows.next().unwrap();
            row.copy_from_slice(&header[..dim]);
            row[0] = 5.0;
            row[26] = (pair[0] + 1) as f32;
            row[27] = (pair[1] + 1) as f32;
        }
        for output in &frame.mana_outputs {
            for (restricted, amounts) in [(false, output.mana), (true, output.lesson)] {
                let row = rows.next().unwrap();
                row.copy_from_slice(&header[..dim]);
                row[0] = 6.0;
                row[7] = (output.mode + 1) as f32;
                for (index, amount) in amounts.iter().enumerate() {
                    row[12 + index] = *amount as f32;
                }
                row[19] = f32::from(restricted);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn departed_sources_and_stack_targets_never_follow_new_incarnations() {
        use crate::state::player::PlayerConfig;
        let deck = [("Forest".into(), 20), ("Gray Ogre".into(), 20)].into();
        let mut game = Game::new(
            vec![
                PlayerConfig::new("a", deck),
                PlayerConfig::new("b", [("Forest".into(), 40)].into()),
            ],
            23,
            false,
        );
        let permanent = game
            .scenario_force_battlefield(PlayerId(0), "Gray Ogre", true)
            .unwrap();
        let reference = game.permanent_object_ref(permanent).unwrap();
        assert!(game
            .visible_reference(reference, PlayerId(0))
            .object
            .is_some());
        let card = CardId::from(reference.entity);
        game.move_card(card, ZoneType::Hand);
        assert!(game
            .visible_reference(reference, PlayerId(0))
            .object
            .is_none());
        assert!(game
            .target_reference(Target::StackSpell(card), PlayerId(0))
            .object
            .is_none());
        assert!(game.card_reference(card, PlayerId(1)).object.is_none());
    }

    #[test]
    fn mana_output_is_a_child_of_the_payment_it_interrupts() {
        use crate::{
            flow::decision::PaymentProgress,
            state::{mana::Mana, player::PlayerConfig},
        };
        let mut game = Game::new(
            vec![
                PlayerConfig::new("a", [("Forest".into(), 40)].into()),
                PlayerConfig::new("b", [("Forest".into(), 40)].into()),
            ],
            23,
            false,
        );
        let actor = game.agent_player();
        game.suspend_rule_decision(Decision::ManaChoice {
            player: actor,
            choices: vec![Mana::parse("U")],
            resume: Some(Box::new(Decision::Payment(PaymentProgress::new(
                actor,
                PaymentCost::Waterbend { generic: 2 },
                vec![],
                vec![],
            )))),
        });
        let frames = game.continuation_frames(actor);
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[1].decision, 1);
        assert_eq!(frames[1].cost.unwrap()[6], 2);
        assert_eq!(frames[1].parent, Some(0));
        assert_eq!(frames[2].decision, 3);
        assert_eq!(frames[2].parent, Some(1));
        assert_eq!(frames[2].mana_outputs[0].mana, [0, 1, 0, 0, 0, 0]);
    }

    #[test]
    fn relation_capacity_rejects_instead_of_truncating() {
        let frames = vec![ContinuationFrame::default(); RELATION_ROWS + 1];
        let mut out = vec![0.0; RELATION_ROWS * RELATION_DIM];
        assert!(matches!(
            encode_into(&frames, 0, &HashMap::new(), &mut out),
            Err(super::super::observation_encoder::ObservationEncodeError::Capacity { .. })
        ));
    }

    #[test]
    fn target_order_clauses_and_payment_are_preserved() {
        let mut frames = vec![ContinuationFrame {
            kind: 4,
            cost: Some([0, 1, 0, 0, 0, 0, 3]),
            hybrid: vec![[0, 2]],
            lesson: true,
            targets: vec![
                ClauseTarget {
                    reference: VisibleReference {
                        player: Some(0),
                        ..VisibleReference::default()
                    },
                    clause: 0,
                    legal_at_resolution_start: None,
                },
                ClauseTarget {
                    reference: VisibleReference {
                        player: Some(1),
                        ..VisibleReference::default()
                    },
                    clause: 1,
                    legal_at_resolution_start: None,
                },
            ],
            ..ContinuationFrame::default()
        }];
        let mut left = vec![0.0; RELATION_ROWS * RELATION_DIM];
        encode_into(&frames, 0, &HashMap::new(), &mut left).unwrap();
        frames[0].targets.reverse();
        let mut right = left.clone();
        encode_into(&frames, 0, &HashMap::new(), &mut right).unwrap();
        assert_ne!(left, right);
        assert_eq!(&left[12..19], &[0., 1., 0., 0., 0., 0., 3.]);
        assert_eq!(left[19], 1.);
        assert_eq!(
            &left[3 * RELATION_DIM + 26..3 * RELATION_DIM + 28],
            &[1., 3.]
        );
        encode_into(&[], 0, &HashMap::new(), &mut right).unwrap();
        assert!(right.iter().all(|value| *value == 0.));
    }
}
