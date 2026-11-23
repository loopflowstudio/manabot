//! London mulligans. All declarations precede redraws in each round.
use crate::{
    agent::action::{Action, ActionSpace, ActionSpaceKind, AgentError},
    flow::game::Game,
    state::{
        game_object::{ObjectId, PlayerId},
        zone::ZoneType,
    },
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
enum Stage {
    #[default]
    Declare,
    Bottom,
    Enter,
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct OpeningHands {
    kept: [bool; 2],
    mulligans: [usize; 2],
    seat: usize,
    stage: Stage,
    bottomed: usize,
}

impl OpeningHands {
    pub(crate) fn instruction(&self) -> (&'static str, &'static str) {
        match self.stage {
            Stage::Declare => ("opening_hand", "Keep your hand or take a mulligan."),
            Stage::Bottom => (
                "opening_bottom",
                "Choose cards for the bottom of your library, deepest first.",
            ),
            Stage::Enter => (
                "opening_entry",
                "Choose a card to begin on the battlefield, or finish.",
            ),
        }
    }
    /// Seat-relative projection is applied by the relation encoder. Neither frame contains a private card.
    pub(crate) fn frames(
        &self,
    ) -> impl Iterator<Item = crate::agent::continuation::ContinuationFrame> + '_ {
        (0..2).map(|seat| crate::agent::continuation::ContinuationFrame {
            kind: 15,
            controller: seat,
            mode: Some(self.mulligans[seat]),
            decision: match (self.stage, seat == self.seat, self.kept[seat]) {
                (Stage::Enter, true, _) => 5,
                (_, _, true) => 2,
                (Stage::Bottom, true, _) => 4,
                (Stage::Declare, true, _) => 1,
                _ => 3,
            },
            active_clause: (self.stage == Stage::Bottom && seat == self.seat)
                .then(|| self.mulligans[seat] - self.bottomed),
            ..Default::default()
        })
    }
}

impl Game {
    fn opening_entries(&self, player: PlayerId) -> impl Iterator<Item = ObjectId> + '_ {
        self.state
            .zones
            .zone_objects(ZoneType::Hand, player)
            .iter()
            .copied()
            .filter(|id| {
                self.state[*id]
                    .statics
                    .contains(&crate::state::card::StaticAbility::BeginOnBattlefield)
            })
    }

    pub(crate) fn opening_entry(&self) -> bool {
        self.state
            .opening
            .as_ref()
            .is_some_and(|o| o.stage == Stage::Enter)
    }

    pub(crate) fn opening_actions(&self) -> Option<ActionSpace> {
        let opening = self.state.opening.as_ref()?;
        let player = PlayerId(opening.seat);
        let (kind, actions) = if opening.stage == Stage::Bottom {
            (
                ActionSpaceKind::LookAndSelect,
                self.selection_zone_objects(player, ZoneType::Hand)
                    .into_iter()
                    .map(|object| Action::SelectCard { player, object })
                    .collect(),
            )
        } else if opening.stage == Stage::Enter {
            let actions = self
                .opening_entries(player)
                .map(|object| Action::SelectCard { player, object })
                .chain([Action::Decline { player }])
                .collect();
            (ActionSpaceKind::LookAndSelect, actions)
        } else {
            let mut actions = vec![Action::ChooseMode { player, mode: 0 }];
            if !self
                .state
                .zones
                .zone_objects(ZoneType::Hand, player)
                .is_empty()
            {
                actions.push(Action::ChooseMode { player, mode: 1 });
            }
            (ActionSpaceKind::Modal, actions)
        };
        Some(ActionSpace {
            kind,
            actions,
            focus: vec![],
            player: Some(player),
        })
    }

    pub(crate) fn choose_opening(&mut self, action: &Action) -> Result<(), AgentError> {
        let mut opening = self.state.opening.take().unwrap();
        let seat = opening.seat;
        let player = PlayerId(seat);
        match action {
            Action::ChooseMode { mode, .. } if opening.stage == Stage::Declare && *mode <= 1 => {
                opening.kept[seat] = *mode == 0;
                opening.mulligans[seat] += *mode;
            }
            Action::SelectCard { object, .. } if opening.stage == Stage::Bottom => {
                let object = self.move_object(*object, ZoneType::Library);
                self.reposition_library_object(player, object, opening.bottomed);
                opening.bottomed += 1;
                let bottom: Vec<_> = self
                    .state
                    .zones
                    .zone_objects(ZoneType::Library, player)
                    .iter()
                    .take(opening.bottomed)
                    .copied()
                    .collect();
                self.remember_library_objects(player, &bottom);
                if opening.bottomed < opening.mulligans[seat]
                    && self.state.zones.size(ZoneType::Hand, player) > 0
                {
                    self.state.opening = Some(opening);
                    return Ok(());
                }
                opening.bottomed = 0;
                opening.kept[seat] = self
                    .state
                    .zones
                    .zone_objects(ZoneType::Hand, player)
                    .is_empty();
            }
            Action::SelectCard { object, .. } if opening.stage == Stage::Enter => {
                self.move_object(*object, ZoneType::Battlefield);
            }
            Action::Decline { .. } if opening.stage == Stage::Enter => opening.seat += 1,
            _ => {
                self.state.opening = Some(opening);
                return Err(AgentError("invalid opening-hand choice".into()));
            }
        }
        if opening.kept == [true; 2] {
            if opening.stage != Stage::Enter {
                opening.stage = Stage::Enter;
                opening.seat = 0;
            }
            while opening.seat < 2
                && self
                    .opening_entries(PlayerId(opening.seat))
                    .next()
                    .is_none()
            {
                opening.seat += 1;
            }
            if opening.seat == 2 {
                self.state.turn.turn_number = 1;
            } else {
                self.state.opening = Some(opening);
            }
            return Ok(());
        }
        if seat == 0 && !opening.kept[1] {
            opening.seat = 1;
        } else {
            opening.seat = usize::from(opening.kept[0]);
            opening.stage = if opening.stage == Stage::Declare {
                Stage::Bottom
            } else {
                Stage::Declare
            };
            if opening.stage == Stage::Bottom {
                for index in 0..2 {
                    if opening.kept[index] {
                        continue;
                    }
                    let player = PlayerId(index);
                    for object in self
                        .state
                        .zones
                        .zone_objects(ZoneType::Hand, player)
                        .to_vec()
                    {
                        self.move_object(object, ZoneType::Library);
                    }
                    self.shuffle_library(player);
                    for _ in 0..7 {
                        if let Some(object) = self.state.zones.top(ZoneType::Library, player) {
                            self.move_object(object, ZoneType::Hand);
                        }
                    }
                }
            }
        }
        self.state.opening = Some(opening);
        Ok(())
    }
}
