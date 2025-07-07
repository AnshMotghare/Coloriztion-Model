import streamlit as st
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from models.network_rrdbnet import RRDBNet  # <-- Make sure this file is present in models/

# -------------------------------
# Setup Paths to Model Files
# -------------------------------
prototxt = 'models/colorization_deploy_v2.prototxt'
caffemodel = 'models/colorization_release_v2.caffemodel'
cluster_centers = 'models/pts_in_hull.npy'
bsrgan_model_path = 'model_zoo/BSRGAN.pth'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------------------------------
# Load Colorization Model
# -------------------------------
net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
pts = np.load(cluster_centers).transpose().reshape(2, 313, 1, 1)

net.getLayer(net.getLayerId("class8_ab")).blobs = [pts.astype(np.float32)]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]

def colorize_image(image):
    h, w = image.shape[:2]
    img_rgb = (image[:, :, [2, 1, 0]] / 255.).astype(np.float32)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
    img_l = img_lab[:, :, 0]

    img_l_rs = cv2.resize(img_l, (224, 224)) - 50
    net.setInput(cv2.dnn.blobFromImage(img_l_rs))
    ab_dec = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab_us = cv2.resize(ab_dec, (w, h))

    img_lab_out = np.concatenate((img_l[:, :, np.newaxis], ab_us), axis=2)
    img_bgr_out = cv2.cvtColor(img_lab_out.astype(np.float32), cv2.COLOR_Lab2BGR)
    return np.clip(img_bgr_out * 255, 0, 255).astype(np.uint8)

# -------------------------------
# Load BSRGAN Model
# -------------------------------
def load_bsrgan_model():
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4)
    model.load_state_dict(torch.load(bsrgan_model_path, map_location=device), strict=True)
    model.eval()
    model.to(device)
    return model

sr_model = load_bsrgan_model()

def super_resolve_image(image, model):
    img = np.array(image).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img).clamp(0, 1)
    out_img = output.squeeze().permute(1, 2, 0).cpu().numpy()
    return (out_img * 255).astype(np.uint8)

# -------------------------------
# Streamlit App
# -------------------------------
st.title("🖼️ Image Colorization & Super-Resolution App")
st.markdown("Upload different images for **colorization** and **super-resolution**.")

st.sidebar.header("Choose Task")
task = st.sidebar.radio("Select", ['Colorize Grayscale Image', 'Super-Resolve Color Image'])

if task == "Colorize Grayscale Image":
    uploaded_file = st.file_uploader("Upload a Grayscale Image", type=["png", "jpg", "jpeg"], key="colorize")
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        st.image(gray_3ch, caption="Grayscale Input", channels="BGR", use_column_width=True)

        colorized = colorize_image(gray_3ch)
        st.image(colorized[:, :, ::-1], caption="Colorized Output", use_column_width=True)

elif task == "Super-Resolve Color Image":
    uploaded_file = st.file_uploader("Upload a Low-Res Color Image", type=["png", "jpg", "jpeg"], key="sr")
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Low-Res Input", use_column_width=True)

        output_img = super_resolve_image(img, sr_model)
        st.image(output_img, caption="Super-Resolved Output", use_column_width=True)
