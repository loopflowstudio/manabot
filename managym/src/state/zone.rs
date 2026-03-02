use std::array;

use rand::seq::SliceRandom;

use super::game_object::{CardId, PlayerId};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum ZoneType {
    Library = 0,
    Hand = 1,
    Battlefield = 2,
    Graveyard = 3,
    Stack = 4,
    Exile = 5,
    Command = 6,
}

#[derive(Clone, Debug)]
pub struct ZoneManager {
    library: [Vec<CardId>; 2],
    hand: [Vec<CardId>; 2],
    battlefield: [Vec<CardId>; 2],
    graveyard: [Vec<CardId>; 2],
    stack_by_owner: [Vec<CardId>; 2],
    exile: [Vec<CardId>; 2],
    command: [Vec<CardId>; 2],
    stack_order: Vec<CardId>,
    card_zones: Vec<Option<ZoneType>>,
}

impl Default for ZoneManager {
    fn default() -> Self {
        Self {
            library: array::from_fn(|_| Vec::new()),
            hand: array::from_fn(|_| Vec::new()),
            battlefield: array::from_fn(|_| Vec::new()),
            graveyard: array::from_fn(|_| Vec::new()),
            stack_by_owner: array::from_fn(|_| Vec::new()),
            exile: array::from_fn(|_| Vec::new()),
            command: array::from_fn(|_| Vec::new()),
            stack_order: Vec::new(),
            card_zones: Vec::new(),
        }
    }
}

impl ZoneManager {
    fn ensure_slot(&mut self, card: CardId) {
        if self.card_zones.len() <= card.0 {
            self.card_zones.resize(card.0 + 1, None);
        }
    }

    pub fn zone_of(&self, card: CardId) -> Option<ZoneType> {
        self.card_zones.get(card.0).copied().flatten()
    }

    fn remove_card_in_place(cards: &mut Vec<CardId>, card: CardId) {
        cards.retain(|c| *c != card);
    }

    fn remove_from_zone(&mut self, card: CardId, owner: PlayerId, zone: ZoneType) {
        let idx = owner.0;
        match zone {
            ZoneType::Library => Self::remove_card_in_place(&mut self.library[idx], card),
            ZoneType::Hand => Self::remove_card_in_place(&mut self.hand[idx], card),
            ZoneType::Battlefield => Self::remove_card_in_place(&mut self.battlefield[idx], card),
            ZoneType::Graveyard => Self::remove_card_in_place(&mut self.graveyard[idx], card),
            ZoneType::Stack => {
                Self::remove_card_in_place(&mut self.stack_by_owner[idx], card);
                Self::remove_card_in_place(&mut self.stack_order, card);
            }
            ZoneType::Exile => Self::remove_card_in_place(&mut self.exile[idx], card),
            ZoneType::Command => Self::remove_card_in_place(&mut self.command[idx], card),
        }
    }

    fn insert_to_zone(&mut self, card: CardId, owner: PlayerId, zone: ZoneType) {
        let idx = owner.0;
        match zone {
            ZoneType::Library => self.library[idx].push(card),
            ZoneType::Hand => self.hand[idx].push(card),
            ZoneType::Battlefield => self.battlefield[idx].push(card),
            ZoneType::Graveyard => self.graveyard[idx].push(card),
            ZoneType::Stack => {
                self.stack_by_owner[idx].push(card);
                self.stack_order.push(card);
            }
            ZoneType::Exile => self.exile[idx].push(card),
            ZoneType::Command => self.command[idx].push(card),
        }
    }

    pub fn move_card(&mut self, card: CardId, owner: PlayerId, to_zone: ZoneType) {
        self.ensure_slot(card);
        if let Some(old_zone) = self.zone_of(card) {
            self.remove_from_zone(card, owner, old_zone);
        }
        self.insert_to_zone(card, owner, to_zone);
        self.card_zones[card.0] = Some(to_zone);
    }

    pub fn push_stack(&mut self, card: CardId, owner: PlayerId) {
        self.move_card(card, owner, ZoneType::Stack);
    }

    pub fn pop_stack(&mut self, owner: PlayerId) -> Option<CardId> {
        let card = self.stack_order.pop()?;
        Self::remove_card_in_place(&mut self.stack_by_owner[owner.0], card);
        self.card_zones[card.0] = None;
        Some(card)
    }

    pub fn top(&self, zone: ZoneType, player: PlayerId) -> Option<CardId> {
        let cards = self.zone_cards(zone, player);
        cards.last().copied()
    }

    pub fn contains(&self, card: CardId, zone: ZoneType, player: PlayerId) -> bool {
        self.zone_cards(zone, player).contains(&card)
    }

    pub fn size(&self, zone: ZoneType, player: PlayerId) -> usize {
        self.zone_cards(zone, player).len()
    }

    pub fn total_size(&self, zone: ZoneType) -> usize {
        match zone {
            ZoneType::Stack => self.stack_order.len(),
            _ => self.size(zone, PlayerId(0)) + self.size(zone, PlayerId(1)),
        }
    }

    pub fn zone_cards(&self, zone: ZoneType, player: PlayerId) -> &Vec<CardId> {
        let idx = player.0;
        match zone {
            ZoneType::Library => &self.library[idx],
            ZoneType::Hand => &self.hand[idx],
            ZoneType::Battlefield => &self.battlefield[idx],
            ZoneType::Graveyard => &self.graveyard[idx],
            ZoneType::Stack => &self.stack_by_owner[idx],
            ZoneType::Exile => &self.exile[idx],
            ZoneType::Command => &self.command[idx],
        }
    }

    pub fn zone_cards_mut(&mut self, zone: ZoneType, player: PlayerId) -> &mut Vec<CardId> {
        let idx = player.0;
        match zone {
            ZoneType::Library => &mut self.library[idx],
            ZoneType::Hand => &mut self.hand[idx],
            ZoneType::Battlefield => &mut self.battlefield[idx],
            ZoneType::Graveyard => &mut self.graveyard[idx],
            ZoneType::Stack => &mut self.stack_by_owner[idx],
            ZoneType::Exile => &mut self.exile[idx],
            ZoneType::Command => &mut self.command[idx],
        }
    }

    pub fn stack_order(&self) -> &Vec<CardId> {
        &self.stack_order
    }

    pub fn shuffle<R: rand::Rng + ?Sized>(
        &mut self,
        zone: ZoneType,
        player: PlayerId,
        rng: &mut R,
    ) {
        self.zone_cards_mut(zone, player).shuffle(rng);
    }
}
