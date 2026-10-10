#[test]
#[ignore]
fn rocm_cpu_to_gpu_copy_test() -> candle_core::Result<()> {
    use candle_core::{DType, Device, Tensor};

    let dev = Device::new_rocm(0)?;

    for shape in [
        vec![2560],
        vec![1, 2560],
        vec![1, 1, 2560],
        vec![1, 2560, 1],
        vec![1, 2560, 3],
    ] {
        let n: usize = shape.iter().product();

        let vals: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.001) - 1.0)
            .collect();

        let cpu = Tensor::from_vec(
            vals.clone(),
            shape.as_slice(),
            &Device::Cpu,
        )?
        .to_dtype(DType::F32)?;

        let gpu = cpu.to_device(&dev)?;

        let roundtrip = gpu
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        let mut bad = 0usize;
        let mut max_err = 0.0f32;

        for (a, b) in vals.iter().zip(roundtrip.iter()) {
            if !a.is_finite() || !b.is_finite() || (a - b).abs() > 1e-6 {
                bad += 1;
                max_err = max_err.max((a - b).abs());
            }
        }

        eprintln!(
            "shape={:?} elements={} bad={} max_err={} first_gpu={:?}",
            shape,
            n,
            bad,
            max_err,
            &roundtrip[..roundtrip.len().min(10)]
        );

        assert_eq!(
            bad, 0,
            "CPU -> ROCm -> CPU corruption for {:?}",
            shape
        );
    }

    Ok(())
}