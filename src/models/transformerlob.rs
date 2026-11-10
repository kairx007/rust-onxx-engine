use crate::{
    config::ModelSpecs,
    error::AppError,
    models::ModelType,
    runtime::{ModelMetadata, TensorMetadata},
};

/// TransformerLOB tensor contract resolved from the loaded ONNX graph.
///
/// Without a supplied model, this implementation only supports the established
/// Phase 2 input layout and refuses to guess a reshape or reorder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransformerLobContract {
    input: TensorMetadata,
    output: TensorMetadata,
}

impl TransformerLobContract {
    pub fn from_metadata(metadata: &ModelMetadata, specs: &ModelSpecs) -> Result<Self, AppError> {
        let model = ModelType::TransformerLob;
        let mut problems = Vec::new();
        if metadata.inputs.len() != 1 {
            problems.push(format!(
                "expected exactly 1 input, found {}",
                metadata.inputs.len()
            ));
        }
        if metadata.outputs.len() != 1 {
            problems.push(format!(
                "expected exactly 1 output, found {}",
                metadata.outputs.len()
            ));
        }

        let input = metadata.inputs.first().cloned();
        let output = metadata.outputs.first().cloned();
        if let Some(input) = &input {
            validate_input(input, &specs.input_shape(), &mut problems);
        }
        if let Some(output) = &output {
            validate_output(output, specs.output_classes, &mut problems);
        }
        if !problems.is_empty() {
            return Err(AppError::InferenceMetadata {
                model,
                details: problems.join("; "),
            });
        }

        let (Some(input), Some(output)) = (input, output) else {
            return Err(AppError::InferenceMetadata {
                model,
                details: "unable to resolve the single model input and output".to_owned(),
            });
        };
        Ok(Self { input, output })
    }

    pub fn input_name(&self) -> &str {
        &self.input.name
    }

    pub fn output_name(&self) -> &str {
        &self.output.name
    }

    pub fn input_metadata(&self) -> &TensorMetadata {
        &self.input
    }

    pub fn output_metadata(&self) -> &TensorMetadata {
        &self.output
    }
}

fn validate_input(input: &TensorMetadata, expected: &[i64], problems: &mut Vec<String>) {
    if input.element_type != "Float32" {
        problems.push(format!(
            "input '{}' must be Float32, found {}",
            input.name, input.element_type
        ));
    }
    let Some(dimensions) = input.dimensions.as_deref() else {
        problems.push(format!("input '{}' is not a tensor", input.name));
        return;
    };
    if dimensions.len() != expected.len() {
        problems.push(format!(
            "input '{}' has rank {}, expected Phase 2 rank {}",
            input.name,
            dimensions.len(),
            expected.len()
        ));
        return;
    }
    for (axis, (actual, phase2)) in dimensions.iter().zip(expected).enumerate() {
        if *actual >= 0 && actual != phase2 {
            problems.push(format!(
                "input '{}' axis {axis} is {actual}, but Phase 2 provides {phase2}; no TransformerLOB-specific layout transform has been verified",
                input.name
            ));
        }
    }
}

fn validate_output(output: &TensorMetadata, classes: i64, problems: &mut Vec<String>) {
    if output.element_type != "Float32" {
        problems.push(format!(
            "output '{}' must be Float32, found {}",
            output.name, output.element_type
        ));
    }
    let Some(dimensions) = output.dimensions.as_deref() else {
        problems.push(format!("output '{}' is not a tensor", output.name));
        return;
    };
    if dimensions.is_empty() {
        problems.push(format!("output '{}' must have a class axis", output.name));
        return;
    }
    if dimensions[..dimensions.len() - 1]
        .iter()
        .any(|dimension| *dimension != 1 && *dimension >= 0)
    {
        problems.push(format!(
            "output '{}' has non-singleton fixed prefix dimensions {dimensions:?}; a single-input three-class prediction is required",
            output.name
        ));
    }
    if dimensions
        .last()
        .is_some_and(|dimension| *dimension >= 0 && *dimension != classes)
    {
        problems.push(format!(
            "output '{}' final dimension must contain {classes} classes, found {dimensions:?}",
            output.name
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::TransformerLobContract;
    use crate::{
        config::Settings,
        error::AppError,
        runtime::{ModelMetadata, TensorMetadata},
    };

    fn metadata(input: Vec<i64>, input_type: &str, output: Vec<i64>) -> ModelMetadata {
        ModelMetadata {
            inputs: vec![TensorMetadata {
                name: "transformer_input_from_model".to_owned(),
                element_type: input_type.to_owned(),
                dimensions: Some(input),
            }],
            outputs: vec![TensorMetadata {
                name: "transformer_scores_from_model".to_owned(),
                element_type: "Float32".to_owned(),
                dimensions: Some(output),
            }],
        }
    }

    #[test]
    fn resolves_names_from_metadata_and_accepts_dynamic_phase2_dimensions() {
        let settings = Settings::parse(include_str!("../../configs/default.toml")).unwrap();
        for input in [vec![1, 1, 100, 40], vec![-1, 1, 100, 40]] {
            let contract = TransformerLobContract::from_metadata(
                &metadata(input, "Float32", vec![-1, 3]),
                &settings.model_specs,
            )
            .unwrap();
            assert_eq!(contract.input_name(), "transformer_input_from_model");
            assert_eq!(contract.output_name(), "transformer_scores_from_model");
        }
    }

    #[test]
    fn rejects_unsupported_shapes_types_and_input_counts() {
        let settings = Settings::parse(include_str!("../../configs/default.toml")).unwrap();
        for invalid in [
            metadata(vec![1, 100, 40], "Float32", vec![1, 3]),
            metadata(vec![1, 1, 100, 40], "Float16", vec![1, 3]),
            metadata(vec![1, 1, 100, 40], "Float32", vec![1, 2]),
        ] {
            assert!(matches!(
                TransformerLobContract::from_metadata(&invalid, &settings.model_specs),
                Err(AppError::InferenceMetadata { .. })
            ));
        }

        let mut multiple_inputs = metadata(vec![1, 1, 100, 40], "Float32", vec![1, 3]);
        multiple_inputs
            .inputs
            .push(multiple_inputs.inputs[0].clone());
        assert!(
            TransformerLobContract::from_metadata(&multiple_inputs, &settings.model_specs).is_err()
        );

        let mut no_outputs = metadata(vec![1, 1, 100, 40], "Float32", vec![1, 3]);
        no_outputs.outputs.clear();
        assert!(TransformerLobContract::from_metadata(&no_outputs, &settings.model_specs).is_err());

        let mut multiple_outputs = metadata(vec![1, 1, 100, 40], "Float32", vec![1, 3]);
        multiple_outputs
            .outputs
            .push(multiple_outputs.outputs[0].clone());
        assert!(
            TransformerLobContract::from_metadata(&multiple_outputs, &settings.model_specs)
                .is_err()
        );

        assert!(
            TransformerLobContract::from_metadata(
                &metadata(vec![1, 1, 100, 40], "Float32", vec![1, 4, 3]),
                &settings.model_specs
            )
            .is_err()
        );
    }
}
