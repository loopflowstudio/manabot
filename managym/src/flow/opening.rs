//! London mulligans. All declarations precede redraws in each round.
use crate::{
    agent::action::{Action, ActionSpace, ActionSpaceKind, AgentError},
    flow::game::{Game, GameProcedure},
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

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct OpeningHands {
    kept: [bool; 2],
    mulligans: [u8; 2],
    seat: u8,
    stage: Stage,
    bottomed: u8,
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
    pub(crate) fn action_description(&self, game: &Game, action: &Action) -> Option<String> {
        match action {
            Action::ChooseMode { mode: 0, .. } => Some("Keep hand".into()),
            Action::ChooseMode { mode: 1, .. } => Some("Mulligan".into()),
            Action::Decline { .. } => Some("Finish opening-hand actions".into()),
            Action::SelectCard { object, .. } if self.stage == Stage::Enter => Some(format!(
                "Begin with {} on the battlefield",
                game.state[*object].name
            )),
            Action::SelectCard { object, .. } => Some(format!(
                "Bottom {} (deepest first)",
                game.state[*object].name
            )),
            _ => None,
        }
    }

    /// Seat-relative projection is applied by the relation encoder. Neither frame contains a private card.
    pub(crate) fn frames(
        &self,
    ) -> impl Iterator<Item = crate::agent::continuation::ContinuationFrame> + '_ {
        (0..2).map(|seat| crate::agent::continuation::ContinuationFrame {
            kind: 15,
            controller: seat,
            mode: Some(usize::from(self.mulligans[seat])),
            decision: match (self.stage, seat == usize::from(self.seat), self.kept[seat]) {
                (Stage::Enter, true, _) => 5,
                (_, _, true) => 2,
                (Stage::Bottom, true, _) => 4,
                (Stage::Declare, true, _) => 1,
                _ => 3,
            },
            active_clause: (self.stage == Stage::Bottom && seat == usize::from(self.seat))
                .then(|| usize::from(self.mulligans[seat] - self.bottomed)),
            ..Default::default()
        })
    }
}

impl Game {
    #[cfg(test)]
    pub(crate) fn keep_opening_hands(&mut self) {
        while self.state.opening().is_some() {
            let action = self
                .action_space()
                .unwrap()
                .actions
                .iter()
                .position(|action| {
                    matches!(
                        action,
                        Action::ChooseMode { mode: 0, .. } | Action::Decline { .. }
                    )
                })
                .expect("fixture needs a keep or finish opening offer");
            self.step(action).unwrap();
        }
    }

    fn redraw_opening_hand(&mut self, player: PlayerId) {
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

    fn opening_entries(&self, player: PlayerId) -> impl Iterator<Item = ObjectId> + '_ {
        self.state
            .zones
            .zone_objects(ZoneType::Hand, player)
            .iter()
            .copied()
            .filter(|&id| {
                self.state[id].statics.iter().any(|ability| {
                    matches!(
                        ability,
                        crate::state::card::StaticAbility::BeginOnBattlefield
                    ) && self.static_ability_functions(id, ability)
                })
            })
    }

    pub(crate) fn opening_actions(&self) -> Option<ActionSpace> {
        let opening = self.state.opening()?;
        let player = PlayerId(usize::from(opening.seat));
        let (kind, actions) = match opening.stage {
            Stage::Bottom => (
                ActionSpaceKind::LookAndSelect,
                self.selection_zone_objects(player, ZoneType::Hand)
                    .into_iter()
                    .map(|object| Action::SelectCard { player, object })
                    .collect(),
            ),
            Stage::Enter => {
                let actions = self
                    .opening_entries(player)
                    .map(|object| Action::SelectCard { player, object })
                    .chain([Action::Decline { player }])
                    .collect();
                (ActionSpaceKind::LookAndSelect, actions)
            }
            Stage::Declare => {
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
            }
        };
        Some(ActionSpace {
            kind,
            actions,
            focus: vec![],
            player: Some(player),
        })
    }

    pub(crate) fn choose_opening(&mut self, action: &Action) -> Result<(), AgentError> {
        let mut opening = *self.state.opening().expect("opening decision");
        let seat = usize::from(opening.seat);
        let player = PlayerId(seat);
        match (opening.stage, action) {
            (Stage::Declare, Action::ChooseMode { mode, .. }) if *mode <= 1 => {
                opening.kept[seat] = *mode == 0;
                opening.mulligans[seat] += *mode as u8;
            }
            (Stage::Bottom, Action::SelectCard { object, .. }) => {
                let object = self.move_object(*object, ZoneType::Library);
                self.reposition_library_object(player, object, usize::from(opening.bottomed));
                opening.bottomed += 1;
                let bottom: Vec<_> = self
                    .state
                    .zones
                    .zone_objects(ZoneType::Library, player)
                    .iter()
                    .take(usize::from(opening.bottomed))
                    .copied()
                    .collect();
                self.remember_library_objects(player, &bottom);
                if opening.bottomed < opening.mulligans[seat]
                    && self.state.zones.size(ZoneType::Hand, player) > 0
                {
                    self.state.procedure = GameProcedure::OpeningHands(opening);
                    return Ok(());
                }
                opening.bottomed = 0;
                opening.kept[seat] = self
                    .state
                    .zones
                    .zone_objects(ZoneType::Hand, player)
                    .is_empty();
            }
            (Stage::Enter, Action::SelectCard { object, .. }) => {
                self.move_object(*object, ZoneType::Battlefield);
            }
            (Stage::Enter, Action::Decline { .. }) => opening.seat += 1,
            _ => {
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
                    .opening_entries(PlayerId(usize::from(opening.seat)))
                    .next()
                    .is_none()
            {
                opening.seat += 1;
            }
            if opening.seat == 2 {
                self.state.procedure = GameProcedure::Playing;
                self.state.turn.turn_number = 1;
            } else {
                self.state.procedure = GameProcedure::OpeningHands(opening);
            }
            return Ok(());
        }
        if seat == 0 && !opening.kept[1] {
            opening.seat = 1;
        } else {
            opening.seat = u8::from(opening.kept[0]);
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
                    self.redraw_opening_hand(PlayerId(index));
                }
            }
        }
        self.state.procedure = GameProcedure::OpeningHands(opening);
        Ok(())
    }
}
