import streamlit as st
from PIL import Image
from model import load_model, preprocess_image
from util import grad_cam, overlay_heatmap
import torch


st.set_page_config(page_title="Indian Classical Dance Classifier", layout="centered")
st.title("📀 Indian Classical Dance Classifier")
st.markdown("Upload a dance image to classify the form and visualize attention (Grad-CAM++)")

model_path = "models/dance_model.pth"
model = load_model(model_path)
layer = model.layer4[-1]  # Grad-CAM target

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    input_tensor = preprocess_image(img)
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)

    class_names = ["Bharatanatyam", "Kathak", "Kuchipudi", "Odissi", "Mohiniyattam", "Manipuri", "Sattriya", "Kathakali"]
    st.subheader(f"Predicted Dance Form: **{class_names[predicted]}**")

    heatmap = grad_cam(model, input_tensor, layer)
    cam_image = overlay_heatmap(img, heatmap)
    st.image(cam_image, caption="Model Focus (Grad-CAM++)", use_column_width=True)

