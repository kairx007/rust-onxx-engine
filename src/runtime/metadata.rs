use ort::{session::Session, value::Outlet};

use crate::{config::ModelSpecs, error::AppError, models::ModelType};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorMetadata {
    pub name: String,
    pub element_type: String,
    pub dimensions: Option<Vec<i64>>,
}

impl From<&Outlet> for TensorMetadata {
    fn from(outlet: &Outlet) -> Self {
        let value_type = outlet.dtype();
        let dimensions = value_type
            .tensor_shape()
            .map(|shape| shape.iter().copied().collect());
        let element_type = value_type.tensor_type().map_or_else(
            || format!("{value_type:?}"),
            |element_type| format!("{element_type:?}"),
        );
        Self {
            name: outlet.name().to_owned(),
            element_type,
            dimensions,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelMetadata {
    pub inputs: Vec<TensorMetadata>,
    pub outputs: Vec<TensorMetadata>,
}

impl ModelMetadata {
    pub fn from_session(session: &Session) -> Self {
        Self {
            inputs: session.inputs().iter().map(TensorMetadata::from).collect(),
            outputs: session.outputs().iter().map(TensorMetadata::from).collect(),
        }
    }

    pub fn validate(&self, model: ModelType, specs: &ModelSpecs) -> Result<(), AppError> {
        let mut problems = Vec::new();
        if self.inputs.len() != 1 {
            problems.push(format!("expected 1 input, found {}", self.inputs.len()));
        }
        if self.outputs.len() != 1 {
            problems.push(format!("expected 1 output, found {}", self.outputs.len()));
        }

        if let Some(input) = self.inputs.first() {
            if input.element_type != "Float32" {
                problems.push(format!(
                    "input '{}' must be Float32, found {}",
                    input.name, input.element_type
                ));
            }
            validate_dimensions("input", input, &specs.input_shape(), &mut problems);
        }

        if let Some(output) = self.outputs.first() {
            if output.element_type != "Float32" {
                problems.push(format!(
                    "output '{}' must be Float32, found {}",
                    output.name, output.element_type
                ));
            }
            match output.dimensions.as_deref() {
                Some([]) => {
                    problems.push(format!(
                        "output '{}' must have a class dimension",
                        output.name
                    ));
                }
                Some(dimensions) => {
                    if let Some(classes) = dimensions.last()
                        && *classes >= 0
                        && *classes != specs.output_classes
                    {
                        problems.push(format!(
                            "output '{}' has {} classes, expected {}",
                            output.name, classes, specs.output_classes
                        ));
                    }
                }
                None => problems.push(format!(
                    "output '{}' is not a tensor and has no dimensions",
                    output.name
                )),
            }
        }

        if problems.is_empty() {
            Ok(())
        } else {
            Err(AppError::MetadataValidation {
                model,
                details: problems.join("; "),
            })
        }
    }
}

fn validate_dimensions(
    label: &str,
    tensor: &TensorMetadata,
    expected: &[i64],
    problems: &mut Vec<String>,
) {
    let Some(actual) = tensor.dimensions.as_deref() else {
        problems.push(format!("{label} '{}' is not a tensor", tensor.name));
        return;
    };
    if actual.len() != expected.len() {
        problems.push(format!(
            "{label} '{}' has rank {}, expected {}",
            tensor.name,
            actual.len(),
            expected.len()
        ));
        return;
    }
    for (index, (actual_dimension, expected_dimension)) in actual.iter().zip(expected).enumerate() {
        if *actual_dimension >= 0 && actual_dimension != expected_dimension {
            problems.push(format!(
                "{label} '{}' dimension {index} is {}, expected {}",
                tensor.name, actual_dimension, expected_dimension
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ModelMetadata, TensorMetadata};
    use crate::{config::Settings, error::AppError, models::ModelType};

    const DEFAULT_CONFIG: &str = include_str!("../../configs/default.toml");

    fn metadata(input_shape: Vec<i64>, output_shape: Vec<i64>) -> ModelMetadata {
        ModelMetadata {
            inputs: vec![TensorMetadata {
                name: "book_window".to_owned(),
                element_type: "Float32".to_owned(),
                dimensions: Some(input_shape),
            }],
            outputs: vec![TensorMetadata {
                name: "logits".to_owned(),
                element_type: "Float32".to_owned(),
                dimensions: Some(output_shape),
            }],
        }
    }

    #[test]
    fn validates_expected_shapes_and_dynamic_dimensions() {
        let settings = Settings::parse(DEFAULT_CONFIG).expect("default config should parse");
        let exact = metadata(vec![1, 1, 100, 40], vec![1, 3]);
        assert!(
            exact
                .validate(ModelType::DeepLob, &settings.model_specs)
                .is_ok()
        );

        let dynamic = metadata(vec![-1, 1, 100, 40], vec![-1, 3]);
        assert!(
            dynamic
                .validate(ModelType::TransformerLob, &settings.model_specs)
                .is_ok()
        );
    }

    #[test]
    fn rejects_wrong_dimensions_and_element_types() {
        let settings = Settings::parse(DEFAULT_CONFIG).expect("default config should parse");
        let invalid = ModelMetadata {
            inputs: vec![TensorMetadata {
                name: "book_window".to_owned(),
                element_type: "Float16".to_owned(),
                dimensions: Some(vec![1, 1, 100, 39]),
            }],
            outputs: vec![TensorMetadata {
                name: "logits".to_owned(),
                element_type: "Float32".to_owned(),
                dimensions: Some(vec![1, 2]),
            }],
        };
        assert!(matches!(
            invalid.validate(ModelType::DeepLob, &settings.model_specs),
            Err(AppError::MetadataValidation { .. })
        ));
    }
}
