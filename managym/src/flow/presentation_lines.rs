//! Project a committed event while its subjects still have the appearances and
//! visibility they had when it happened. Only the labels used by one line leave
//! this projection. Later positions (even in the same step) cannot add a name.

use crate::{
    cards::content_pack::default_content_pack,
    experience::{
        LookAction, ObjectRenderId, PlayerId as Seat, PresentationKind as Kind, PresentationLabels,
        RequiredNullable, StackRenderId, SubjectRef,
    },
    flow::{
        event::{DamageTarget, EventSubject, GameEvent, ObjectEventRef},
        game::Game,
        presentation::PresentationLine,
    },
    state::{
        game_object::{ObjectId, PlayerId, Target},
        stack_object::StackObject,
        zone::ZoneType,
    },
};

fn seat(player: PlayerId) -> Seat {
    Seat(player.0 as u8)
}

impl PresentationLabels {
    fn name(&mut self, id: ObjectRenderId, name: String) {
        self.objects
            .insert(format!("{}:{}", id.entity, id.incarnation), name);
    }

    fn line(self, kind: Kind) -> PresentationLine {
        PresentationLine {
            kind,
            labels: self,
            combat: false,
        }
    }
}

impl Game {
    fn can_name_object(&self, object: ObjectId, viewer: PlayerId) -> bool {
        // A departed battlefield object keeps its exact public appearance in LKI.
        self.object_lki(object).is_some() || self.visible_reference(object, viewer).object.is_some()
    }

    fn named_object(
        &self,
        object: ObjectId,
        viewer: PlayerId,
        labels: &mut PresentationLabels,
    ) -> Option<ObjectRenderId> {
        if !self.can_name_object(object, viewer) {
            return None;
        }
        let id = self.object_event_ref(object)?.into();
        labels.name(id, self.observed_object_name(object));
        Some(id)
    }

    /// Combat and death facts already carry a public appearance, including a
    /// face-down alias. Look up only that appearance's effective name.
    fn named_permanent(
        &self,
        reference: ObjectEventRef,
        labels: &mut PresentationLabels,
    ) -> ObjectRenderId {
        let object = self
            .state
            .permanents
            .iter()
            .flatten()
            .find(|p| p.id == reference.entity)
            .map(|p| p.object)
            .or_else(|| {
                self.state
                    .object_lki
                    .iter()
                    .find(|(_, lki)| lki.presentation_id == reference.entity)
                    .map(|(object, _)| *object)
            });
        let id = reference.into();
        if let Some(object) =
            object.filter(|object| self.object_event_ref(*object) == Some(reference))
        {
            labels.name(id, self.observed_object_name(object));
        }
        id
    }

    // The legacy board identifies a stacked spell by its card's display id.
    fn named_spell(
        &self,
        object: ObjectId,
        labels: &mut PresentationLabels,
    ) -> Option<StackRenderId> {
        let stack = StackRenderId(self.object_event_ref(object)?.entity.0.into());
        let info = self.spell_info(object)?;
        let name = default_content_pack()
            .definition(info.definition_id)?
            .printed
            .name
            .clone();
        labels.stacks.insert(stack.0.to_string(), name);
        Some(stack)
    }

    fn named_target(
        &self,
        target: Target,
        viewer: PlayerId,
        labels: &mut PresentationLabels,
    ) -> Option<SubjectRef> {
        Some(match target {
            Target::Player(player) => SubjectRef::Player { id: seat(player) },
            Target::Permanent(p) => SubjectRef::Object {
                id: self.named_object(self.permanent_object(p)?, viewer, labels)?,
            },
            Target::GraveyardCard(card) => SubjectRef::Object {
                id: self.named_object(card, viewer, labels)?,
            },
            Target::StackSpell(spell) => SubjectRef::Stack {
                id: self.named_spell(spell, labels)?,
            },
            Target::StackAbility(id) => {
                let source = self.stack_ability_source(id)?;
                if self.can_name_object(source, viewer) && self.object_event_ref(source).is_some() {
                    labels
                        .stacks
                        .insert(id.0.to_string(), self.observed_object_name(source));
                }
                SubjectRef::Stack {
                    id: StackRenderId(id.0.into()),
                }
            }
        })
    }

    /// One `targeted` line per target a spell or ability on the stack chose,
    /// after the line that put it there.
    fn target_lines(
        &self,
        targets: &[Target],
        stack: StackRenderId,
        name: &str,
        viewer: PlayerId,
    ) -> Vec<PresentationLine> {
        targets
            .iter()
            .filter_map(|target| {
                let mut labels = PresentationLabels::default();
                let target = self.named_target(*target, viewer, &mut labels)?;
                labels.stacks.insert(stack.0.to_string(), name.to_owned());
                Some(labels.line(Kind::Targeted {
                    source: SubjectRef::Stack { id: stack },
                    target,
                }))
            })
            .collect()
    }

    pub(super) fn pass_line(player: PlayerId) -> Vec<PresentationLine> {
        vec![PresentationLabels::default().line(Kind::Passed {
            player: seat(player),
        })]
    }

    pub(super) fn lost_line(player: PlayerId) -> Vec<PresentationLine> {
        vec![PresentationLabels::default().line(Kind::Lost {
            player: seat(player),
        })]
    }

    pub(super) fn draw_line(
        &self,
        player: PlayerId,
        card: ObjectId,
        viewer: PlayerId,
    ) -> Vec<PresentationLine> {
        let mut labels = PresentationLabels::default();
        let object = (self.shares_information(viewer, player)
            && self.state.zones.zone_of(card) == Some(ZoneType::Hand))
        .then(|| self.named_object(card, viewer, &mut labels))
        .flatten();
        vec![labels.line(Kind::Drew {
            player: seat(player),
            object,
        })]
    }

    pub(super) fn stack_line(
        &self,
        stacked: &StackObject,
        viewer: PlayerId,
        resolving: bool,
    ) -> Vec<PresentationLine> {
        let mut labels = PresentationLabels::default();
        let object = stacked.source_object();
        let kind = if matches!(stacked, StackObject::Spell(_)) {
            let Some(stack) = self.named_spell(object, &mut labels) else {
                return vec![];
            };
            Kind::Resolved { stack }
        } else {
            let Some(source) = self.named_object(object, viewer, &mut labels) else {
                return vec![];
            };
            let stack = StackRenderId(stacked.id().0.into());
            let name = self.observed_object_name(object);
            labels.stacks.insert(stack.0.to_string(), name.clone());
            if resolving {
                Kind::AbilityResolved { source, stack }
            } else {
                let mut lines = vec![labels.line(Kind::Activated {
                    source,
                    controller: seat(stacked.controller()),
                    stack,
                })];
                lines.extend(self.target_lines(stacked.targets(), stack, &name, viewer));
                return lines;
            }
        };
        vec![labels.line(kind)]
    }

    pub(super) fn event_lines(&self, event: &GameEvent, viewer: PlayerId) -> Vec<PresentationLine> {
        use GameEvent::*;
        let mut labels = PresentationLabels::default();
        let kind = match event {
            SpellCast {
                object_id,
                targets,
                caster,
                ..
            } => {
                let (Some(object), Some(stack)) = (
                    self.named_object(*object_id, viewer, &mut labels),
                    self.named_spell(*object_id, &mut labels),
                ) else {
                    return vec![];
                };
                let name = labels.stacks[&stack.0.to_string()].clone();
                let mut lines = vec![labels.line(Kind::Cast {
                    object,
                    stack,
                    controller: seat(*caster),
                })];
                lines.extend(self.target_lines(targets, stack, &name, viewer));
                return lines;
            }
            SpellCountered { object_id, .. } => {
                let Some(stack) = self.named_spell(*object_id, &mut labels) else {
                    return vec![];
                };
                Kind::Countered { stack }
            }
            DamageDealt {
                source,
                target,
                amount,
                combat: false,
            } => {
                let target = match target {
                    DamageTarget::Player(p) => Target::Player(*p),
                    DamageTarget::Permanent(p) => Target::Permanent(*p),
                };
                let Some(target) = self.named_target(target, viewer, &mut labels) else {
                    return vec![];
                };
                let source = source.and_then(|id| {
                    if self.spell_on_stack(id).is_some() {
                        self.named_spell(id, &mut labels)
                            .map(|id| SubjectRef::Stack { id })
                    } else {
                        self.named_object(id, viewer, &mut labels)
                            .map(|id| SubjectRef::Object { id })
                    }
                });
                Kind::Damage {
                    source: RequiredNullable(source),
                    target,
                    amount: *amount as i32,
                }
            }
            CombatAttackersDeclared {
                attackers,
                defender,
                ..
            } => Kind::AttackGroup {
                attackers: attackers
                    .iter()
                    .map(|r| self.named_permanent(*r, &mut labels))
                    .collect(),
                defender: SubjectRef::Player {
                    id: seat(*defender),
                },
            },
            BlockersDeclared { assignments } => {
                return assignments
                    .iter()
                    .filter(|(_, blockers)| !blockers.is_empty())
                    .map(|(attacker, blockers)| {
                        let mut labels = PresentationLabels::default();
                        let attacker = self.named_permanent(*attacker, &mut labels);
                        let blockers = blockers
                            .iter()
                            .map(|r| self.named_permanent(*r, &mut labels))
                            .collect();
                        labels.line(Kind::Blocked { attacker, blockers })
                    })
                    .collect()
            }
            CombatDamageDealt {
                source,
                target,
                amount,
            } => {
                let source = SubjectRef::Object {
                    id: self.named_permanent(*source, &mut labels),
                };
                let target = match target {
                    EventSubject::Object(r) => SubjectRef::Object {
                        id: self.named_permanent(*r, &mut labels),
                    },
                    EventSubject::Player(p) => SubjectRef::Player { id: seat(*p) },
                };
                let mut line = labels.line(Kind::Damage {
                    source: RequiredNullable(Some(source)),
                    target,
                    amount: *amount as i32,
                });
                line.combat = true;
                return vec![line];
            }
            PermanentsDied { objects } => {
                let sacrificed = self
                    .presentation
                    .as_deref()
                    .map(|sink| sink.sacrificed.as_slice())
                    .unwrap_or(&[]);
                let objects: Vec<ObjectRenderId> = objects
                    .iter()
                    .filter(|r| !sacrificed.contains(r))
                    .map(|r| self.named_permanent(*r, &mut labels))
                    .collect();
                if objects.is_empty() {
                    return vec![];
                }
                Kind::Died { objects }
            }
            PermanentSacrificed {
                object_id,
                controller,
            } => {
                let Some(object) = self.named_object(*object_id, viewer, &mut labels) else {
                    return vec![];
                };
                Kind::Sacrificed {
                    objects: vec![object],
                    player: seat(*controller),
                }
            }
            LibrarySearched { player } => Kind::Searched {
                player: seat(*player),
            },
            Scried { player } => Kind::Looked {
                player: seat(*player),
                action: LookAction::Scry,
            },
            Surveilled { player } => Kind::Looked {
                player: seat(*player),
                action: LookAction::Surveil,
            },
            TurnStarted {
                player,
                turn_number,
            } => Kind::TurnStarted {
                player: seat(*player),
                turn_number: *turn_number as u32,
            },
            CardMoved {
                object_id,
                from: Some(from),
                departed: Some(departed),
                to,
                controller,
                face_down: false,
            } => {
                let played = *to == ZoneType::Battlefield
                    && self.presentation.as_ref().and_then(|sink| sink.land_play)
                        == Some(*departed);
                let discarded = (*from, *to) == (ZoneType::Hand, ZoneType::Graveyard);
                if !played && !discarded {
                    return vec![];
                }
                let Some(object) = self.named_object(*object_id, viewer, &mut labels) else {
                    return vec![];
                };
                if played {
                    Kind::LandPlayed {
                        object,
                        player: seat(*controller),
                    }
                } else {
                    Kind::Discarded {
                        objects: vec![object],
                        player: seat(*controller),
                    }
                }
            }
            CardRevealed { owner, definition } => {
                let Some(card) = default_content_pack().definition(*definition) else {
                    return vec![];
                };
                Kind::Revealed {
                    player: seat(*owner),
                    cards: vec![card.printed.name.clone()],
                }
            }
            AbilityTriggered {
                source_object,
                controller,
                ability,
                ..
            } => {
                let Some(source) = self.named_object(*source_object, viewer, &mut labels) else {
                    return vec![];
                };
                let stack = StackRenderId(ability.0.into());
                let name = self.observed_object_name(*source_object);
                labels.stacks.insert(stack.0.to_string(), name.clone());
                let targets = self
                    .stack_index_of(Target::StackAbility(*ability))
                    .map(|index| self.state.stack_objects[index].targets().to_vec())
                    .unwrap_or_default();
                let mut lines = vec![labels.line(Kind::Triggered {
                    source,
                    controller: seat(*controller),
                    stack,
                })];
                lines.extend(self.target_lines(&targets, stack, &name, viewer));
                return lines;
            }
            CountersPut {
                permanent,
                kind,
                count,
                total,
                ..
            } => {
                let Some(object) = self
                    .permanent_object(*permanent)
                    .and_then(|o| self.named_object(o, viewer, &mut labels))
                else {
                    return vec![];
                };
                Kind::Counters {
                    object,
                    counter: kind.name().to_owned(),
                    count: *count,
                    total: *total,
                }
            }
            // Damage to a player tells its own life loss; `presentation_event`
            // drops the change that follows it.
            LifeChanged {
                player, old, new, ..
            } => Kind::Life {
                player: seat(*player),
                amount: new - old,
                total: *new,
            },
            _ => return vec![],
        };
        vec![labels.line(kind)]
    }
}
