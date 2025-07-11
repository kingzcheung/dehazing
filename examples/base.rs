use candle_core::Tensor;
use candle_nn::Module;
use dehazing::model::DehazeNet;

fn main() {
    let device = candle_core::Device::cuda_if_available(0).unwrap();
    let base_dir = env!("CARGO_MANIFEST_DIR");

    let model = DehazeNet::with_device(&device).unwrap();

    let img = image::open(format!("{base_dir}/testdata/test2.png")).unwrap();

    let raw = img.to_rgb8().into_vec();
    let data = Tensor::from_vec(
        raw,
        (img.height() as usize, img.width() as usize, 3),
        &device,
    )
    .unwrap()
    .to_dtype(candle_core::DType::F32)
    .unwrap()
    .broadcast_div(&Tensor::new(255f32, &device).unwrap())
    .unwrap()
    .permute((2, 0, 1))
    .unwrap()
    .unsqueeze(0)
    .unwrap();

    println!("{data:?}");

    let out = model.forward(&data).unwrap();

    // 处理输出张量
    let out = out.squeeze(0).unwrap(); // 移除批次维度 [c, h, w]

    let (_, height, width) = out.dims3().unwrap();

    let image_data: Vec<u8> = out
        .permute((1, 2, 0))
        .unwrap() // [H, W, C] 符合图像布局
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0) as u8)
        .collect();

    // 保存图像
    let img_out =
        image::RgbImage::from_raw(width as u32, height as u32, image_data).expect("创建图像失败");

    img_out.save("result/dehazed_output.jpg").expect("保存图像失败");
    println!("去雾结果已保存为 result/dehazed_output.jpg");
}
