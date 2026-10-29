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

/// Serde `skip_serializing_if` for numeric fields whose zero is omitted, so
/// manifests that predate the field stay byte-for-byte stable.
pub(crate) fn is_zero<T: Default + PartialEq>(value: &T) -> bool {
    *value == T::default()
}
