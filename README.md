# Doodle-to-Real Image Translation & Colorization using Pix2Pix

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20Demo-Space-blue)](https://huggingface.co/spaces/ImranAliNaeem/pix2pix-anime)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)

## 📌 Overview
Pix2Pix implementation for sketch-to-realistic and grayscale-to-color image translation using Conditional GAN.

## 🚀 Live Demo
**Try it here:** [Hugging Face Space](https://huggingface.co/spaces/ImranAliNaeem/pix2pix-anime)

## 📊 Results

### Quantitative Metrics
| Dataset | SSIM | PSNR (dB) |
|---------|------|-----------|
| Anime Sketch Colorization | 0.6457 | 14.61 |
| CUHK Face Dataset | 0.3928 | 8.42 |

### Training Performance
Epoch [40/50] → G_loss: 20.44 | D_loss: 0.27
Epoch [45/50] → G_loss: 20.36 | D_loss: 0.27
Epoch [48/50] → G_loss: 20.62 | D_loss: 0.26

text

## 🏗️ Architecture
- **Generator**: U-Net with skip connections (54.4M params)
- **Discriminator**: PatchGAN (2.8M params)
- **Input/Output**: 256×256×3

## 📁 Datasets
- **Anime Sketch Colorization** (16,000+ pairs)
- **CUHK Face Sketch Dataset** (188 pairs)

## 🛠️ Setup

```bash
# Clone
git clone https://github.com/Imran-Ali-Naeem/Doodle-to-Real-Image-Translation.git
cd Doodle-to-Real-Image-Translation

# Install
pip install -r requirements.txt
💻 Usage
Training
bash
python train.py --dataset anime --epochs 50 --batch_size 16
Inference
bash
python test.py --input sketch.jpg --output result.png
Run App
bash
python app.py
⚙️ Hyperparameters
Parameter	Value
Image Size	256×256
Batch Size	16
Learning Rate	0.0002
L1 Lambda	100
Epochs	50
📈 Loss Curves
Generator Loss: Stable convergence ~20.4

Discriminator Loss: Maintains 0.26-0.29

🙏 Acknowledgments
Pix2Pix Paper

Kaggle for GPU resources

📧 Contact
Author: Imran Ali Naeem

GitHub: Imran-Ali-Naeem

Hugging Face: ImranAliNaeem

text

This concise version includes all essential information without excessive detail.
