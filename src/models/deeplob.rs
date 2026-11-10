use crate::{
    config::ModelSpecs,
    error::AppError,
    runtime::{ModelMetadata, TensorMetadata},
};

/// Model-specific names resolved from the loaded ONNX graph.
///
/// Names are intentionally read from the model metadata instead of assumed in
/// source code, since no DeepLOB ONNX asset is checked into this repository.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepLobContract {
    input_name: String,
    output_name: String,
    input: TensorMetadata,
    output: TensorMetadata,
}

impl DeepLobContract {
    pub fn from_metadata(metadata: &ModelMetadata, specs: &ModelSpecs) -> Result<Self, AppError> {
        metadata
            .validate(crate::models::ModelType::DeepLob, specs)
            .map_err(|error| AppError::InferenceMetadata {
                model: crate::models::ModelType::DeepLob,
                details: error.to_string(),
            })?;

        let input =
            metadata
                .inputs
                .first()
                .cloned()
                .ok_or_else(|| AppError::InferenceMetadata {
                    model: crate::models::ModelType::DeepLob,
                    details: "the ONNX graph has no input".to_owned(),
                })?;
        let output =
            metadata
                .outputs
                .first()
                .cloned()
                .ok_or_else(|| AppError::InferenceMetadata {
                    model: crate::models::ModelType::DeepLob,
                    details: "the ONNX graph has no output".to_owned(),
                })?;

        validate_output_contract(&output, specs.output_classes)?;
        Ok(Self {
            input_name: input.name.clone(),
            output_name: output.name.clone(),
            input,
            output,
        })
    }

    pub fn input_name(&self) -> &str {
        &self.input_name
    }

    pub fn output_name(&self) -> &str {
        &self.output_name
    }

    pub fn input_metadata(&self) -> &TensorMetadata {
        &self.input
    }

    pub fn output_metadata(&self) -> &TensorMetadata {
        &self.output
    }
}

fn validate_output_contract(output: &TensorMetadata, output_classes: i64) -> Result<(), AppError> {
    let dims = output
        .dimensions
        .as_deref()
        .ok_or_else(|| AppError::InferenceMetadata {
            model: crate::models::ModelType::DeepLob,
            details: format!("output '{}' is not a tensor", output.name),
        })?;

    // The class dimension is last. Prefix dimensions may be singleton or
    // dynamic batch-like dimensions; actual dimensions are checked at run time.
    let valid_rank = !dims.is_empty();
    let valid_batch = dims[..dims.len().saturating_sub(1)]
        .iter()
        .all(|dimension| *dimension == 1 || *dimension < 0);
    let valid_classes = dims
        .last()
        .is_some_and(|dimension| *dimension == output_classes || *dimension < 0);
    if !valid_rank || !valid_batch || !valid_classes {
        return Err(AppError::InferenceMetadata {
            model: crate::models::ModelType::DeepLob,
            details: format!(
                "output '{}' has dimensions {dims:?}; expected {output_classes} classes on the final axis with singleton or dynamic prefix dimensions",
                output.name
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::DeepLobContract;
    use crate::{
        config::Settings,
        error::AppError,
        runtime::{ModelMetadata, TensorMetadata},
    };

    fn metadata(input_dims: Vec<i64>, output_dims: Vec<i64>) -> ModelMetadata {
        ModelMetadata {
            inputs: vec![TensorMetadata {
                name: "actual_input".to_owned(),
                element_type: "Float32".to_owned(),
                dimensions: Some(input_dims),
            }],
            outputs: vec![TensorMetadata {
                name: "actual_scores".to_owned(),
                element_type: "Float32".to_owned(),
                dimensions: Some(output_dims),
            }],
        }
    }

    #[test]
    fn resolves_actual_tensor_names_and_accepts_static_or_dynamic_batch() {
        let settings = Settings::parse(include_str!("../../configs/default.toml")).unwrap();
        for output in [vec![3], vec![1, 3], vec![-1, 3]] {
            let contract = DeepLobContract::from_metadata(
                &metadata(vec![1, 1, 100, 40], output),
                &settings.model_specs,
            )
            .unwrap();
            assert_eq!(contract.input_name(), "actual_input");
            assert_eq!(contract.output_name(), "actual_scores");
        }
    }

    #[test]
    fn rejects_wrong_output_layout_and_missing_graph_tensors() {
        let settings = Settings::parse(include_str!("../../configs/default.toml")).unwrap();
        for shape in [vec![1, 3, 1], vec![2, 3], vec![1, 2]] {
            assert!(matches!(
                DeepLobContract::from_metadata(
                    &metadata(vec![1, 1, 100, 40], shape),
                    &settings.model_specs
                ),
                Err(AppError::InferenceMetadata { .. })
            ));
        }

        let no_input = ModelMetadata {
            inputs: vec![],
            outputs: metadata(vec![1, 1, 100, 40], vec![1, 3]).outputs,
        };
        assert!(DeepLobContract::from_metadata(&no_input, &settings.model_specs).is_err());
    }
}
