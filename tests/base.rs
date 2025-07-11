use dehazing::{model::DehazeNet, Module as _, Tensor, VarBuilder};

#[test]
fn test_model() {
    let base_dir = env!("CARGO_MANIFEST_DIR");
    let weight_path = format!("{base_dir}/dehazer.safetensors");
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[&weight_path],
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .unwrap()
    };

    let model = DehazeNet::new(vb).unwrap();

    // println!("{model:?}");

    let img = image::open(format!("{base_dir}/testdata/test2.png")).unwrap();

    let raw = img.to_rgb8().into_vec();
    let data = Tensor::from_vec(
        raw,
        (img.height() as usize, img.width() as usize, 3),
        &candle_core::Device::Cpu,
    )
    .unwrap()
    .to_dtype(candle_core::DType::F32)
    .unwrap()
    .broadcast_div(&Tensor::new(255f32, &candle_core::Device::Cpu).unwrap())
    .unwrap()
    .permute((2, 0, 1))
    .unwrap()
    .unsqueeze(0)
    .unwrap();

    println!("{data:?}");

    let out = model.forward(&data).unwrap();

    assert_eq!(out.dims(), &[1, 3, 980, 1306])
}
