//! Contiguous inference input tensors and validation.

mod input;
pub mod validation;

pub use input::InputTensor;
pub use validation::{INPUT_SHAPE, INPUT_VALUE_COUNT};
