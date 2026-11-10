use crate::error::AppError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Down,
    Stationary,
    Up,
}

impl Direction {
    pub fn from_class_index(index: usize) -> Result<Self, AppError> {
        match index {
            0 => Ok(Self::Down),
            1 => Ok(Self::Stationary),
            2 => Ok(Self::Up),
            other => Err(AppError::Prediction(format!(
                "class index {other} is outside the three-class range"
            ))),
        }
    }

    pub fn class_index(self) -> usize {
        match self {
            Self::Down => 0,
            Self::Stationary => 1,
            Self::Up => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Prediction {
    class_index: usize,
    direction: Direction,
    scores: [f32; 3],
}

impl Prediction {
    pub fn from_scores(scores: [f32; 3]) -> Result<Self, AppError> {
        if scores.iter().any(|score| !score.is_finite()) {
            return Err(AppError::InvalidInferenceOutput(
                "class scores must all be finite".to_owned(),
            ));
        }

        // max_by with a reversed index comparison selects the lowest class
        // index when two scores tie.
        let class_index = (0..scores.len())
            .max_by(|left, right| {
                scores[*left]
                    .total_cmp(&scores[*right])
                    .then_with(|| right.cmp(left))
            })
            .ok_or_else(|| AppError::Prediction("no class scores were supplied".to_owned()))?;
        let direction = Direction::from_class_index(class_index)?;
        Ok(Self {
            class_index,
            direction,
            scores,
        })
    }

    pub fn class_index(&self) -> usize {
        self.class_index
    }

    pub fn direction(&self) -> Direction {
        self.direction
    }

    pub fn scores(&self) -> [f32; 3] {
        self.scores
    }
}

#[cfg(test)]
mod tests {
    use super::{Direction, Prediction};

    #[test]
    fn maps_argmax_scores_to_the_three_directions() {
        for (scores, expected) in [
            ([0.9, 0.1, 0.2], Direction::Down),
            ([0.1, 0.9, 0.2], Direction::Stationary),
            ([0.1, 0.2, 0.9], Direction::Up),
        ] {
            assert_eq!(
                Prediction::from_scores(scores).unwrap().direction(),
                expected
            );
        }
    }

    #[test]
    fn ties_choose_lowest_index_and_nonfinite_values_fail() {
        let result = Prediction::from_scores([0.9, 0.9, 0.1]).unwrap();
        assert_eq!(result.class_index(), 0);
        assert!(Prediction::from_scores([0.0, f32::INFINITY, 0.0]).is_err());
        assert!(Direction::from_class_index(3).is_err());
    }
}
