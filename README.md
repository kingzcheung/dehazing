# dehazing

图片去雾工具包，它通过深度卷积网络（DNN）对图片进行去雾处理。

## 特性

- 基于深度学习的高效去雾算法
- 支持 GPU 加速（CUDA）
- 简洁易用的 API 接口
- 支持与 [image](https://crates.io/crates/image) 库集成进行图像处理

## 快速开始

使用 `cargo` 安装 `dehazing` 与 `image` 库：

```bash
cargo add dehazing image
```

## 示例代码

以下是一个完整的示例，展示如何使用 `dehazing` 对图像进行去雾处理：

```rust
let device = candle_core::Device::cuda_if_available(0).unwrap();
let base_dir = env!("CARGO_MANIFEST_DIR");

// 加载预训练模型
let model = DehazeNet::with_device(&device).unwrap();

// 打开输入图像
let img = image::open(format!("{base_dir}/testdata/test2.png")).unwrap();

// 将图像转换为 RGB8 格式并转换为 Tensor
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

// 进行去雾推理
let out = model.forward(&data).unwrap();

// 处理输出张量
let out = out.squeeze(0).unwrap(); // 移除批次维度 [c, h, w]
let (_, height, width) = out.dims3().unwrap();

// 将输出张量转换为图像数据
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
```

## 模型说明

本项目基于 `DehazeNet` 架构实现了一个端到端的去雾模型。该模型使用深度卷积神经网络来预测大气光和透射图，从而恢复清晰图像。

## 设备支持

- **CPU**：默认支持
- **GPU（CUDA）**：可通过 `candle-core` 的 CUDA 支持启用

确保你的系统安装了合适的 CUDA 驱动，并在编译时启用 `cuda` 特性。

## 依赖项

- [`candle-core`](https://crates.io/crates/candle-core)：用于张量计算和深度学习模型构建
- [`image`](https://crates.io/crates/image)：用于图像加载和保存

## 目录结构建议

```bash
.
├── Cargo.toml
├── src/
│   └── lib.rs
├── testdata/           # 存放测试图像
│   └── test2.png
└── result/             # 存放去雾后的图像
    └── dehazed_output.jpg
```

## 编译与运行

确保你已经安装 Rust 工具链，并使用以下命令运行项目：

```bash
cargo run
```

如果希望启用 GPU 支持，请确保 `candle-core` 启用了 `cuda` 特性。

## 注意事项

- 输入图像需为 `.png` 或支持的格式
- 输出路径 `result/` 需提前创建
- 若未找到图像或模型加载失败，程序将抛出错误信息

## 贡献

欢迎提交 PR 和 Issue 反馈！请遵循项目的编码风格和文档规范。

## 许可证

本项目采用 MIT License。详情请参阅 [LICENSE](LICENSE) 文件。

## 更多信息

如需了解模型细节、性能评估或训练方法，请查阅项目的官方文档或相关论文。