pub mod ability;
pub mod card;
pub mod game_object;
pub mod hash;
pub mod mana;
pub mod permanent;
pub mod player;
pub mod predicate;
pub mod stack_object;
pub mod target;
pub mod zone;

/// Serde `skip_serializing_if` for counts whose default is omitted.
pub(crate) fn is_zero<T: Default + PartialEq>(value: &T) -> bool {
    *value == T::default()
}
