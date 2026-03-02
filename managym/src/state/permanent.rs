use super::{
    card::Card,
    game_object::{CardId, ObjectId, PlayerId},
    mana::Mana,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Permanent {
    pub id: ObjectId,
    pub card: CardId,
    pub controller: PlayerId,
    pub tapped: bool,
    pub summoning_sick: bool,
    pub damage: i32,
    pub attacking: bool,
}

impl Permanent {
    pub fn new(id: ObjectId, card_id: CardId, card: &Card) -> Self {
        Self {
            id,
            card: card_id,
            controller: card.owner,
            tapped: false,
            summoning_sick: card.types.is_creature(),
            damage: 0,
            attacking: false,
        }
    }

    pub fn can_tap(&self, card: &Card) -> bool {
        !(self.tapped || (self.summoning_sick && card.types.is_creature()))
    }

    pub fn can_attack(&self, card: &Card) -> bool {
        card.types.is_creature() && !self.tapped && !self.summoning_sick
    }

    pub fn can_block(&self, card: &Card) -> bool {
        card.types.is_creature() && !self.tapped
    }

    pub fn has_lethal_damage(&self, card: &Card) -> bool {
        card.types.is_creature() && self.damage >= card.toughness.unwrap_or(0)
    }

    pub fn producible_mana(&self, card: &Card) -> Mana {
        let mut total = Mana::default();
        if !self.can_tap(card) {
            return total;
        }
        for ability in &card.mana_abilities {
            total.add(&ability.mana);
        }
        total
    }

    pub fn untap(&mut self) {
        self.tapped = false;
    }

    pub fn tap(&mut self) {
        self.tapped = true;
    }

    pub fn clear_damage(&mut self) {
        self.damage = 0;
    }

    pub fn take_damage(&mut self, amount: i32) {
        self.damage += amount;
    }

    pub fn attack(&mut self) {
        self.attacking = true;
        self.tap();
    }
}
