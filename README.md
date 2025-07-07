🖼️ Image Colorization and Super-Resolution
A deep learning project that transforms grayscale images into color and enhances their resolution using two separate neural networks:

🎨 Image Colorization using a VGG16-based autoencoder (TensorFlow)

🔍 Image Super-Resolution using SRGAN (PyTorch)

🚀 Features
✅ Converts grayscale images to realistic colored images

✅ Enhances image resolution for sharper and high-quality output

✅ Streamlit-powered GUI for real-time inference

✅ Supports custom image input and batch processing

✅ Trained on custom dataset with AB channel prediction (Lab color space)

✅ Modular architecture for easy training and inference

🧠 Model Architectures
🎨 Colorization Model (TensorFlow)
Based on VGG16 encoder

Custom decoder to predict a and b channels of Lab color space

Loss: Mean Squared Error (MSE) between predicted and true ab channels

Input size: 128x128 grayscale image

Output: Colorized image in RGB

🔍 Super-Resolution Model (PyTorch)
Based on SRGAN (Super-Resolution GAN)

Generator + Discriminator architecture

Loss: Content loss + Adversarial loss (VGG perceptual loss)

Input: Low-res (LR) image

Output: High-res (HR) image

🏋️‍♂️ Training Your Own Models
🎨 Colorization Model
bash
Copy
Edit
python train_colorization.py
Input: Custom dataset of color images

Output: Trained .h5 model

🔍 Super-Resolution (SRGAN)
bash
Copy
Edit
python train_srgan.py
Input: LR and HR image pairs

Output: Trained generator.pth

🧪 Sample Results
Grayscale Input	Colorized Output	Super-Resolved Output

🛠️ Technologies Used
🧠 TensorFlow / Keras – VGG16 encoder, custom decoder

🔥 PyTorch – SRGAN architecture

🖼️ OpenCV, PIL – Image loading & manipulation

🌐 Streamlit – Web interface

📊 Matplotlib / Seaborn – Visualization

🎨 scikit-image – Color space conversion

📊 Evaluation Metrics
Colorization:

✅ Mean Squared Error (MSE)

✅ Structural Similarity Index (SSIM)

Super-Resolution:

✅ Peak Signal-to-Noise Ratio (PSNR)

✅ SSIM

✅ Visual Perceptual Quality

📸 Dataset
📁 Custom dataset under sample img/

Grayscale images in sample img/grayscale

Color images in sample img/color

🔓 Public dataset suggestions:

ImageNet

COCO Dataset

DIV2K (for SRGAN)

📌 Future Improvements
🔁 Add attention mechanism to colorization decoder

🎯 Integrate perceptual loss for better visual fidelity

🔄 Replace SRGAN with BSRGAN

🎥 Support real-time video colorization

🧑‍💻 Authors
Ansh Motghare

Swapnil Patil

Prajwal Patole

Rohan Nikiam
