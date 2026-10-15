use rocm_rs::miopen::tensor::{DataType, TensorDescriptor};
use rocm_rs::miopen::{ConvolutionDescriptor, Handle};
use std::collections::HashMap;
use std::sync::Mutex;

/// Everything that determines which forward convolution algorithm MIOpen should pick.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct ConvKey {
    dtype: u32,
    b: usize,
    c_in: usize,
    c_out: usize,
    i_h: usize,
    i_w: usize,
    k_h: usize,
    k_w: usize,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
    dilation_h: usize,
    dilation_w: usize,
}

/// The chosen forward algorithm, per convolution configuration.
///
/// `miopenFindConvolutionForwardAlgorithm` is a **search**: it benchmarks candidate kernels by
/// actually running them. It is meant to be called once per configuration with the caller keeping
/// the answer. Calling it on every forward made the search, not the convolution, the dominant
/// cost — the SeedVR2 VAE decode was running 3x3 128->128 convolutions at roughly 2% of this
/// card's fp16 peak.
///
/// MIOpen's own find-db (`~/.miopen/db/*.ufdb.txt`) makes a repeat search cheaper than a cold one,
/// which is why a warm run was already ~30x faster than a cold one. But "cheaper than
/// benchmarking every candidate" is still vastly more than "already know the answer".
///
/// Keyed by shape rather than by pointer, so it is valid across tiles, frames and tensors: the
/// algorithm MIOpen picks depends only on the convolution's geometry and dtype.
static ALGO_CACHE: Mutex<Option<HashMap<ConvKey, i32>>> = Mutex::new(None);

fn cached_algo(key: &ConvKey) -> Option<i32> {
    ALGO_CACHE
        .lock()
        .ok()?
        .as_ref()
        .and_then(|m| m.get(key).copied())
}

fn store_algo(key: ConvKey, algo: i32) {
    if let Ok(mut guard) = ALGO_CACHE.lock() {
        guard.get_or_insert_with(HashMap::new).insert(key, algo);
    }
}

/// Note the bf16-before-f16 order: `half::bf16`'s type name *contains* "f16", so testing "f16"
/// first matched bf16 as well and the `MiopenBFloat16` arm was unreachable — every bf16
/// convolution was describing its tensors to MIOpen as fp16, which reinterprets the bits rather
/// than converting them.
pub fn miopen_dtype<T: Copy>() -> DataType {
    let name = std::any::type_name::<T>();
    if name.contains("f32") {
        DataType::MiopenFloat
    } else if name.contains("f64") {
        DataType::MiopenDouble
    } else if name.contains("bf16") {
        DataType::MiopenBFloat16
    } else if name.contains("f16") {
        DataType::MiopenHalf
    } else {
        panic!("Unsupported dtype for MIOpen: {name}")
    }
}

/// Stable small discriminant for [`ConvKey`], mirroring [`miopen_dtype`]'s matching (including its
/// bf16-before-f16 order) without depending on `DataType` being castable to an integer.
fn dtype_key<T: Copy>() -> u32 {
    let name = std::any::type_name::<T>();
    if name.contains("f32") {
        0
    } else if name.contains("f64") {
        1
    } else if name.contains("bf16") {
        2
    } else if name.contains("f16") {
        3
    } else {
        4
    }
}

pub fn conv2d_forward<T: Copy>(
    handle: &Handle,
    x_ptr: *mut std::ffi::c_void,
    w_ptr: *mut std::ffi::c_void,
    y_ptr: *mut std::ffi::c_void,
    b: usize,
    c_in: usize,
    c_out: usize,
    i_h: usize,
    i_w: usize,
    k_h: usize,
    k_w: usize,
    out_h: usize,
    out_w: usize,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
    dilation_h: usize,
    dilation_w: usize,
) -> crate::Result<()> {
    let x_desc = TensorDescriptor::new_4d(
        miopen_dtype::<T>(),
        b as i32,
        c_in as i32,
        i_h as i32,
        i_w as i32,
    )
    .map_err(|e| crate::Error::Msg(format!("MIOpen x_desc creation failed: {}", e)))?;

    let w_desc = TensorDescriptor::new_4d(
        miopen_dtype::<T>(),
        c_out as i32,
        c_in as i32,
        k_h as i32,
        k_w as i32,
    )
    .map_err(|e| crate::Error::Msg(format!("MIOpen w_desc creation failed: {}", e)))?;

    let y_desc = TensorDescriptor::new_4d(
        miopen_dtype::<T>(),
        b as i32,
        c_out as i32,
        out_h as i32,
        out_w as i32,
    )
    .map_err(|e| crate::Error::Msg(format!("MIOpen y_desc creation failed: {}", e)))?;

    let mut conv_desc = ConvolutionDescriptor::new()
        .map_err(|e| crate::Error::Msg(format!("MIOpen conv_desc creation failed: {}", e)))?;
    conv_desc
        .init_2d(
            0, // miopenConvolution
            pad_h as i32,
            pad_w as i32,
            stride_h as i32,
            stride_w as i32,
            dilation_h as i32,
            dilation_w as i32,
        )
        .map_err(|e| crate::Error::Msg(format!("MIOpen conv_desc init_2d failed: {}", e)))?;

    let workspace_size = rocm_rs::miopen::convolution::get_convolution_forward_workspace_size(
        handle, &w_desc, &x_desc, &conv_desc, &y_desc,
    )
    .map_err(|e| crate::Error::Msg(format!("MIOpen workspace size query failed: {}", e)))?;

    let alpha: [u8; 4] = 1.0f32.to_le_bytes();
    let beta: [u8; 4] = 0.0f32.to_le_bytes();

    let workspace = if workspace_size > 0 {
        Some(
            rocm_rs::hip::DeviceMemory::<u8>::new(workspace_size).map_err(|e| {
                crate::Error::Msg(format!("MIOpen workspace allocation failed: {}", e))
            })?,
        )
    } else {
        None
    };
    let workspace_ptr = workspace
        .as_ref()
        .map(|w| w.as_ptr() as *mut std::ffi::c_void)
        .unwrap_or(std::ptr::null_mut());

    // Search for the best algorithm only the first time we see this configuration; afterwards reuse
    // what it found. See `ALGO_CACHE` — the search is the expensive part, not the convolution.
    let key = ConvKey {
        dtype: dtype_key::<T>(),
        b,
        c_in,
        c_out,
        i_h,
        i_w,
        k_h,
        k_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
    };

    unsafe {
        let algo = match cached_algo(&key) {
            Some(algo) => algo,
            None => {
                let (_, perf_results) =
                    rocm_rs::miopen::convolution::find_convolution_forward_algorithm(
                        handle,
                        &x_desc,
                        x_ptr,
                        &w_desc,
                        w_ptr,
                        &conv_desc,
                        &y_desc,
                        y_ptr,
                        1,
                        workspace_ptr,
                        workspace_size,
                        false,
                    )
                    .map_err(|e| {
                        crate::Error::Msg(format!(
                            "MIOpen find_convolution_forward_algorithm failed: {}",
                            e
                        ))
                    })?;

                let algo = perf_results
                    .first()
                    .map(|p| p.__bindgen_anon_1.fwd_algo)
                    .unwrap_or(4);
                if std::env::var("CANDLE_ROCM_CONV_LOG").is_ok() {
                    // One line per distinct convolution configuration, printed on the cache miss.
                    eprintln!(
                        "[rocm-conv] in {}x{}x{}x{} k{}x{} -> {}ch pad{} stride{} | algo={} workspace={:.1}MB",
                        b, c_in, i_h, i_w, k_h, k_w, c_out, pad_h, stride_h, algo,
                        workspace_size as f64 / (1024.0 * 1024.0)
                    );
                }
                store_algo(key, algo);
                algo
            }
        };

        rocm_rs::miopen::convolution::convolution_forward(
            handle,
            &alpha,
            &x_desc,
            x_ptr,
            &w_desc,
            w_ptr,
            &conv_desc,
            algo,
            &beta,
            &y_desc,
            y_ptr,
            workspace_ptr,
            workspace_size,
        )
        .map_err(|e| crate::Error::Msg(format!("MIOpen convolution_forward failed: {}", e)))?;
    }

    Ok(())
}

pub fn conv_transpose1d_forward<T: Copy>(
    handle: &Handle,
    x_ptr: *mut std::ffi::c_void,
    w_ptr: *mut std::ffi::c_void,
    y_ptr: *mut std::ffi::c_void,
    b: usize,
    c_in: usize,
    c_out: usize,
    l_in: usize,
    k_size: usize,
    l_out: usize,
    padding: usize,
    output_padding: usize,
    stride: usize,
    dilation: usize,
) -> crate::Result<()> {
    let x_desc =
        TensorDescriptor::new_4d(miopen_dtype::<T>(), b as i32, c_in as i32, 1, l_in as i32)
            .map_err(|e| crate::Error::Msg(format!("MIOpen x_desc creation failed: {}", e)))?;

    let w_desc = TensorDescriptor::new_4d(
        miopen_dtype::<T>(),
        c_in as i32,
        c_out as i32,
        1,
        k_size as i32,
    )
    .map_err(|e| crate::Error::Msg(format!("MIOpen w_desc creation failed: {}", e)))?;

    let y_desc =
        TensorDescriptor::new_4d(miopen_dtype::<T>(), b as i32, c_out as i32, 1, l_out as i32)
            .map_err(|e| crate::Error::Msg(format!("MIOpen y_desc creation failed: {}", e)))?;

    let mut conv_desc = ConvolutionDescriptor::new()
        .map_err(|e| crate::Error::Msg(format!("MIOpen conv_desc creation failed: {}", e)))?;
    conv_desc
        .init_2d(0, padding as i32, 0, stride as i32, 1, dilation as i32, 1)
        .map_err(|e| crate::Error::Msg(format!("MIOpen conv_desc init_2d failed: {}", e)))?;
    conv_desc
        .set_transpose_conv_output_padding(output_padding as i32, 0)
        .map_err(|e| {
            crate::Error::Msg(format!(
                "MIOpen set_transpose_conv_output_padding failed: {}",
                e
            ))
        })?;

    let workspace_size =
        rocm_rs::miopen::convolution::get_convolution_backward_data_workspace_size(
            handle, &x_desc, &w_desc, &conv_desc, &y_desc,
        )
        .map_err(|e| {
            crate::Error::Msg(format!(
                "MIOpen backward data workspace size query failed: {}",
                e
            ))
        })?;

    let alpha: [u8; 4] = 1.0f32.to_le_bytes();
    let beta: [u8; 4] = 0.0f32.to_le_bytes();

    let workspace = if workspace_size > 0 {
        Some(
            rocm_rs::hip::DeviceMemory::<u8>::new(workspace_size).map_err(|e| {
                crate::Error::Msg(format!("MIOpen workspace allocation failed: {}", e))
            })?,
        )
    } else {
        None
    };
    let workspace_ptr = workspace
        .as_ref()
        .map(|w| w.as_ptr() as *mut std::ffi::c_void)
        .unwrap_or(std::ptr::null_mut());

    unsafe {
        let (_, perf_results) =
            rocm_rs::miopen::convolution::find_convolution_backward_data_algorithm(
                handle,
                &x_desc,
                x_ptr,
                &w_desc,
                w_ptr,
                &conv_desc,
                &y_desc,
                y_ptr,
                1,
                workspace_ptr,
                workspace_size,
                false,
            )
            .map_err(|e| {
                crate::Error::Msg(format!(
                    "MIOpen find_convolution_backward_data_algorithm failed: {}",
                    e
                ))
            })?;

        let algo = perf_results
            .first()
            .map(|p| p.__bindgen_anon_1.bwd_data_algo)
            .unwrap_or(0);

        rocm_rs::miopen::convolution::convolution_backward_data(
            handle,
            &alpha,
            &x_desc,
            x_ptr,
            &w_desc,
            w_ptr,
            &conv_desc,
            algo,
            &beta,
            &y_desc,
            y_ptr,
            workspace_ptr,
            workspace_size,
        )
        .map_err(|e| {
            crate::Error::Msg(format!(
                "MIOpen convolution_backward_data (transpose conv) failed: {}",
                e
            ))
        })?;
    }

    Ok(())
}
