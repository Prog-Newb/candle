#[test]
#[ignore = "requires an AMD ROCm device"]
fn rocm_f16_matmul_large_linear_shape_is_finite() -> candle_core::Result<()> {
    use candle_core::{DType, Device, Tensor};

    let device = Device::new_rocm(0)?;
    let lhs = Tensor::from_vec(vec![1.0f32; 2560], (1, 2560), &Device::Cpu)?
        .to_dtype(DType::F16)?
        .to_device(&device)?;
    let rhs = Tensor::from_vec(vec![1.0f32; 2560 * 7680], (2560, 7680), &Device::Cpu)?
        .to_dtype(DType::F16)?
        .to_device(&device)?;

    let output = lhs.matmul(&rhs)?;
    let values = output
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .to_vec2::<f32>()?;

    assert_eq!(values.len(), 1);
    assert_eq!(values[0].len(), 7680);
    assert!(values[0].iter().all(|value| value.is_finite()));
    assert!(values[0].iter().all(|value| (*value - 2560.0).abs() < 1.0));

    Ok(())
}
#[test]
#[ignore = "requires an AMD ROCm device"]
fn rocm_batched_attention_value_matmul_odd_shape_is_finite() -> candle_core::Result<()> {
    use candle_core::{DType, Device, Tensor};

    let device = Device::new_rocm(0)?;
    let (batch, heads, query, key, width) = (1usize, 20usize, 157usize, 157usize, 128usize);
    for dtype in [DType::F16, DType::F32] {
        let probabilities = vec![1.0f32 / key as f32; batch * heads * query * key];
        let values = vec![1.0f32; batch * heads * key * width];
        let probabilities = Tensor::from_vec(probabilities, (batch, heads, query, key), &Device::Cpu)?
            .to_dtype(dtype)?
            .to_device(&device)?;
        let values = Tensor::from_vec(values, (batch, heads, key, width), &Device::Cpu)?
            .to_dtype(dtype)?
            .to_device(&device)?;
        let output = probabilities.matmul(&values)?;
        let output = output.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let output = output.flatten_all()?.to_vec1::<f32>()?;
        assert!(output.iter().all(|value| value.is_finite()), "{dtype:?} output contains non-finite values");
        assert!(output.iter().all(|value| (*value - 1.0).abs() < 0.02), "{dtype:?} output deviates from the expected uniform average");
    }
    Ok(())
}