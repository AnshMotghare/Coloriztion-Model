🖼️ Image Colorization and Super-Resolution
A deep learning project that transforms grayscale images into color and enhances their resolution using two separate neural networks:

Image Colorization using a VGG16-based autoencoder (TensorFlow)

Image Super-Resolution using SRGAN (PyTorch)

🚀 Features
✅ Converts grayscale images to realistic colored images

✅ Enhances image resolution for sharper and high-quality output

✅ Streamlit-powered GUI for real-time inference

✅ Supports custom image input and batch processing

✅ Trained on custom dataset with AB channel prediction (Lab color space)

✅ Modular architecture for easy training and inference

📂 Project Structure
pgsql
Copy
Edit
├── models/                  # Contains pre-trained model weights
│   ├── colorization_model.h5
│   └── srgan_generator.pth
├── sample img/
│   ├── grayscale/           # Grayscale test images
│   └── color/               # Ground truth color images (optional)
├── streamlit_app.py         # Unified Streamlit app
├── train_colorization.py    # VGG16-based model training
├── train_srgan.py           # SRGAN training script
├── utils/
│   ├── preprocess.py        # Image loading & preprocessing utilities
│   └── postprocess.py       # Color/Lab to RGB conversion, tensor handling
├── README.md
└── requirements.txt
🧠 Model Architectures
🎨 Colorization Model (TensorFlow)
Based on VGG16 encoder

Custom decoder to predict a and b channels of Lab color space

Loss: MSE between predicted and true ab channels

Input size: 128x128 grayscale image

Output: Colorized image in RGB

🔍 Super-Resolution Model (PyTorch)
Based on SRGAN (Super-Resolution GAN)

Generator + Discriminator architecture

Loss: Content loss + Adversarial loss (VGG perceptual loss)

Trained on low-res (LR) and high-res (HR) image pairs

💻 Setup Instructions
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/yourusername/image-colorization-super-resolution.git
cd image-colorization-super-resolution
2. Create Conda Environments (Optional but Recommended)
For Colorization (TensorFlow)
bash
Copy
Edit
conda create -n colorization-env python=3.9
conda activate colorization-env
pip install -r requirements_colorization.txt
For Super-Resolution (PyTorch)
bash
Copy
Edit
conda create -n srgan-env python=3.9
conda activate srgan-env
pip install -r requirements_srgan.txt
3. Run the Streamlit App
bash
Copy
Edit
streamlit run streamlit_app.py
🏋️‍♂️ Training Your Own Models
Colorization Model
bash
Copy
Edit
python train_colorization.py
Input: Custom dataset of color images

Output: Trained .h5 model

Super-Resolution (SRGAN)
bash
Copy
Edit
python train_srgan.py
Input: LR and HR image pairs (can be generated with downsampling)

Output: Trained generator.pth

🧪 Sample Results
Grayscale Input	Colorized Output	Super-Resolved Output

🛠️ Technologies Used
TensorFlow / Keras – VGG16 encoder, custom decoder

PyTorch – SRGAN architecture

OpenCV, PIL – Image handling

Streamlit – Interactive Web App

Matplotlib / Seaborn – Visualization

scikit-image – Lab color space conversion

📊 Evaluation Metrics
Colorization:

Mean Squared Error (MSE)

Structural Similarity Index (SSIM)

Super-Resolution:

Peak Signal-to-Noise Ratio (PSNR)

SSIM

Perceptual quality (visual inspection)

📸 Dataset
Custom dataset under sample img/

Grayscale images in sample img/grayscale

Corresponding color images in sample img/color

You can also use public datasets like:

ImageNet

COCO Dataset

DIV2K (for SRGAN)

📌 Future Improvements
Add attention mechanisms to colorization decoder

Implement perceptual loss in colorization

Replace SRGAN with BSRGAN for better fidelity

Allow batch uploading and real-time video colorization

🧑‍💻 Authors
Ansh Motghare
Swapnil Patil
Prajwal Patole
Rohan Nikiam
