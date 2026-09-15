use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::time::Instant;

fn main() -> Result<()> {
    println!("=== Candle ROCm GPU functional test ===");

    let device = Device::new_rocm(0)?;
    println!("Device: {:?}", device);

    // ------------------------------------------------------------
    // 1. Original known-good baseline
    // ------------------------------------------------------------
    let a = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)?;
    let b = Tensor::new(&[10.0f32, 20.0, 30.0, 40.0], &device)?;

    let add = (&a + &b)?;
    device.synchronize()?;
    println!("Small add: {:?}", add.to_vec1::<f32>()?);

    // ------------------------------------------------------------
    // 2. One-million-element elementwise tests
    // ------------------------------------------------------------
    let n = 1_000_000usize;

    let x = Tensor::arange(0f32, n as f32, &device)?;
    let ones = Tensor::ones((n,), DType::F32, &device)?;
    let twos = Tensor::new(vec![2.0f32; n], &device)?;

    let add_large = (&x + &ones)?;
    let sub_large = (&x - &ones)?;
    let mul_large = (&x * &twos)?;
    let div_large = (&x / &twos)?;

    device.synchronize()?;

    println!("Large tensor size: {}", n);
    println!(
        "Large add sample: {:?}",
        add_large.narrow(0, 0, 5)?.to_vec1::<f32>()?
    );
    println!(
        "Large sub sample: {:?}",
        sub_large.narrow(0, 0, 5)?.to_vec1::<f32>()?
    );
    println!(
        "Large mul sample: {:?}",
        mul_large.narrow(0, 0, 5)?.to_vec1::<f32>()?
    );
    println!(
        "Large div sample: {:?}",
        div_large.narrow(0, 0, 5)?.to_vec1::<f32>()?
    );

    let sum = x.sum_all()?;
    device.synchronize()?;
    println!("Large sum: {}", sum.to_scalar::<f32>()?);

    // ------------------------------------------------------------
    // 3. Matrix multiplication
    //
    // 256x256 matrix of ones multiplied by a 256x256
    // matrix of twos should produce every element = 512.
    // ------------------------------------------------------------
    let dim = 256usize;

    let m1 = Tensor::ones((dim, dim), DType::F32, &device)?;
    let m2 = Tensor::new(vec![2.0f32; dim * dim], &device)?
    .reshape((dim, dim))?;
    device.synchronize()?;


    let start = Instant::now();

    let mm = m1.matmul(&m2)?;

    device.synchronize()?;
    let elapsed = start.elapsed();

    let mm_sample = mm.narrow(0, 0, 1)?.narrow(1, 0, 5)?.to_vec2::<f32>()?;

    println!("Matmul size: {}x{} @ {}x{}", dim, dim, dim, dim);
    println!("Matmul sample: {:?}", mm_sample);
    println!("Matmul elapsed: {:?}", elapsed);

    // ------------------------------------------------------------
    // 4. Verify deterministic matmul result
    // ------------------------------------------------------------
    let expected = 2.0 * dim as f32;

    let first_row = mm.narrow(0, 0, 1)?.to_vec2::<f32>()?[0].clone();

    for (i, value) in first_row.iter().take(8).enumerate() {
        if (*value - expected).abs() > 1e-3 {
            anyhow::bail!(
                "Matmul verification failed at column {}: got {}, expected {}",
                i,
                value,
                expected
            );
        }
    }

    println!("Matmul verification: PASS");

    println!("=== ROCm functional test complete ===");

    Ok(())
}
