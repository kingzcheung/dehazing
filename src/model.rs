use candle_core::{DType, Result};
use candle_nn::{Conv2d, Module, VarBuilder};

#[derive(Debug)]
pub struct DehazeNet {
    e_conv1: Conv2d,
    e_conv2: Conv2d,
    e_conv3: Conv2d,
    e_conv4: Conv2d,
    e_conv5: Conv2d,
}

fn conv2d(
    c_in: usize,
    c_out: usize,
    ksize: usize,
    padding: usize,
    stride: usize,
    vb: VarBuilder,
) -> Result<Conv2d> {
    let conv2d_cfg = candle_nn::Conv2dConfig {
        stride,
        padding,
        ..Default::default()
    };
    candle_nn::conv2d(c_in, c_out, ksize, conv2d_cfg, vb)
}

impl DehazeNet {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let e_conv1 = conv2d(3, 3, 1, 0, 1, vb.pp("e_conv1"))?;
        let e_conv2 = conv2d(3, 3, 3, 1, 1, vb.pp("e_conv2"))?;
        let e_conv3 = conv2d(6, 3, 5, 2, 1, vb.pp("e_conv3"))?;
        let e_conv4 = conv2d(6, 3, 7, 3, 1, vb.pp("e_conv4"))?;
        let e_conv5 = conv2d(12, 3, 3, 1, 1, vb.pp("e_conv5"))?;
        Ok(Self {
            e_conv1,
            e_conv2,
            e_conv3,
            e_conv4,
            e_conv5,
        })
    }

    pub fn with_device(device: &candle_core::Device) -> Result<Self> {
        let data = include_bytes!("../dehazer.safetensors");
        let vb = VarBuilder::from_buffered_safetensors(data.to_vec(), DType::F32, device)?;

        Self::new(vb)
    }
}

impl Module for DehazeNet {
    fn forward(&self, xs: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        let x1 = self.e_conv1.forward(xs)?.relu()?;
        let x2 = self.e_conv2.forward(&x1)?.relu()?;

        let concat1 = candle_core::Tensor::cat(&[&x1, &x2], 1)?;
        let x3 = self.e_conv3.forward(&concat1)?.relu()?;
        let concat2 = candle_core::Tensor::cat(&[&x2, &x3], 1)?;
        let x4 = self.e_conv4.forward(&concat2)?.relu()?;

        let concat3 = candle_core::Tensor::cat(&[&x1, &x2, &x3, &x4], 1)?;
        let x5 = self.e_conv5.forward(&concat3)?.relu()?;

        let ones = candle_core::Tensor::new(1.0, xs.device())?.to_dtype(DType::F32)?;

        let ys = ((&x5 * xs)? - &x5)?.broadcast_add(&ones)?.relu()?;

        Ok(ys)
    }
}
