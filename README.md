Here's the English translation of your README.md while preserving all code blocks and markdown formatting:

```markdown
# dehazing

An image dehazing toolkit that utilizes Deep Convolutional Neural Networks (DNN) for haze removal.

## Features

- Deep learning-based efficient dehazing algorithm
- GPU acceleration support (CUDA)
- Simple and easy-to-use API interface
- Integration with [image](https://crates.io/crates/image) library for image processing

## Quick Start

Install `dehazing` and `image` using `cargo`:

```bash
cargo add dehazing image
```

## Example Code

Below is a complete example demonstrating how to use `dehazing` for image dehazing:

```rust
let device = candle_core::Device::cuda_if_available(0).unwrap();
let base_dir = env!("CARGO_MANIFEST_DIR");

// Load pre-trained model
let model = DehazeNet::with_device(&device).unwrap();

// Open input image
let img = image::open(format!("{base_dir}/testdata/test2.png")).unwrap();

// Convert image to RGB8 format and transform to Tensor
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

// Perform dehazing inference
let out = model.forward(&data).unwrap();

// Process output tensor
let out = out.squeeze(0).unwrap(); // Remove batch dimension [c, h, w]
let (_, height, width) = out.dims3().unwrap();

// Convert output tensor to image data
let image_data: Vec<u8> = out
    .permute((1, 2, 0))
    .unwrap() // [H, W, C] matches image layout
    .flatten_all()
    .unwrap()
    .to_vec1::<f32>()
    .unwrap()
    .iter()
    .map(|&v| (v.clamp(0.0, 1.0) * 255.0) as u8)
    .collect();

// Save image
let img_out =
    image::RgbImage::from_raw(width as u32, height as u32, image_data).expect("Failed to create image");

img_out.save("result/dehazed_output.jpg").expect("Failed to save image");
println!("Dehazed result saved as result/dehazed_output.jpg");
```

## Model Description

This project implements an end-to-end dehazing model based on the `DehazeNet` architecture. The model uses deep convolutional neural networks to predict atmospheric light and transmission maps for image restoration.

## Device Support

- **CPU**: Supported by default
- **GPU (CUDA)**: Enabled via `features = ['cuda']` with CUDA support

Ensure your system has proper CUDA drivers installed and enable `cuda` feature during compilation.

## Contribution

PRs and Issues are welcome! Please follow the project's coding style and documentation standards.

## License

MIT Licensed. See [LICENSE](LICENSE) for details.
