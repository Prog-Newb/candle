use anyhow::Result;
use candle_core::{Device, Tensor};

fn main() -> Result<()> {
    println!("=== Candle ROCm GPU test ===");

    let device = Device::new_rocm(0)?;
    println!("Device: {:?}", device);

    let a = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &device)?;
    let b = Tensor::new(&[10.0f32, 20.0, 30.0, 40.0], &device)?;

    println!("A: {:?}", a.to_vec1::<f32>()?);
    println!("B: {:?}", b.to_vec1::<f32>()?);

    let c = (&a + &b)?;
    device.synchronize()?;

    println!("A + B = {:?}", c.to_vec1::<f32>()?);
    println!("=== ROCm execution successful ===");

    Ok(())
}
