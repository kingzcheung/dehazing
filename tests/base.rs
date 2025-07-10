use candle_core::Tensor;
use candle_nn::{Module, VarBuilder};
use dehazing::model::DehazeNet;
use image::GenericImageView;

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

    let img = image::open(format!("{base_dir}/test2.png")).unwrap();

    let raw = img.to_rgb8().into_vec();
    let data = Tensor::from_vec(
        raw,
        (img.height() as usize, img.width() as usize, 3),
        &candle_core::Device::Cpu,
    )
    .unwrap()
    .to_dtype(candle_core::DType::F32)
    .unwrap()
    .broadcast_div(&Tensor::new(255f32, &candle_core::Device::Cpu).unwrap()).unwrap()
    .permute((2, 0, 1))
    .unwrap().unsqueeze(0).unwrap();

    println!("{data:?}");

    let out = model.forward(&data).unwrap();

    // 处理输出张量
let out = out.squeeze(0).unwrap(); // 移除批次维度 [3, 384, 465]
let (channels, height, width) = out.dims3().unwrap();
    
// let out_data: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
let out_data: Vec<u8> = out.flatten_all().unwrap()
    .to_vec1::<f32>().unwrap()
    .iter()
    .map(|&v| (v.clamp(0.0, 1.0) * 255.0) as u8) // 强制裁剪 + 缩放
    .collect();
    
// 计算最小/最大值
// let min = out_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
// let max = out_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
// let scale = max - min;
// println!("最小值: {}, 最大值: {}, 缩放: {}", min, max, scale);
    
// 转换为u8并重新排列为图像格式
// let mut image_data = Vec::with_capacity(height * width * 3);
let image_data: Vec<u8> = out
    .permute((1, 2, 0)).unwrap() // [H, W, C] 符合图像布局
    .flatten_all().unwrap()
    .to_vec1::<f32>().unwrap()
    .iter()
    .map(|&v| (v.clamp(0.0, 1.0) * 255.0) as u8)
    .collect();

// 修复1：改变循环顺序
// for h in 0..height {
//     for w in 0..width {
//         for c in 0..channels {
//             // 修复2：使用正确的索引计算公式
//             let idx = c * (height * width) + h * width + w;
//             let normalized = (out_data[idx] - min) / scale;
//             image_data.push((normalized * 255.0).clamp(0.0, 255.0) as u8);
//         }
//     }
// }
    
// 保存图像
let img_out = image::RgbImage::from_raw(
    width as u32,
    height as u32,
    image_data
).expect("创建图像失败");
    
img_out.save("dehazed_output.jpg").expect("保存图像失败");
println!("去雾结果已保存为 dehazed_output.jpg");
}
