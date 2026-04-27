import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn

# -------------------------
# MODEL
# -------------------------
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 2)

    def forward(self, X):
        X = self.pool(F.relu(self.conv1(X)))
        X = self.pool(F.relu(self.conv2(X)))
        X = self.pool(F.relu(self.conv3(X)))

        X = self.gap(X)
        X = torch.flatten(X, 1)
        X = self.fc(X)
        return X

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    model = MyModel()
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -------------------------
# TRANSFORMS (NO AUGMENTATION)
# -------------------------
IMG_SIZE = 224

transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale = (0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.1
    ),
    transforms.RandomAdjustSharpness(2, p=0.3),

    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),

    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3)
    ], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
    ]
)

classes = ["AI Generated", "REAL"]

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title("📊 Model Info")
st.sidebar.write("Model: Custom CNN")
st.sidebar.write("Dataset: CIFAKE")
st.sidebar.write("Accuracy: ~82%")
st.sidebar.write("Task: AI vs Real Image Detection")

# -------------------------
# MAIN UI
# -------------------------
st.title("🧠 AI vs Real Image Detector")
st.write("Detect whether an image is AI-generated or real.")

uploaded_files = st.file_uploader(
    "Upload one or more images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# -------------------------
# GRAD-CAM FUNCTION
# -------------------------
def generate_gradcam(model, img_tensor):
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    hook1 = model.conv3.register_forward_hook(forward_hook)
    hook2 = model.conv3.register_backward_hook(backward_hook)

    output = model(img_tensor)
    pred_class = output.argmax()

    model.zero_grad()
    output[0, pred_class].backward()

    grad = gradients[0]
    act = activations[0]

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1).squeeze()

    cam = torch.relu(cam)
    cam = cam / cam.max()

    cam = cam.detach().numpy()

    hook1.remove()
    hook2.remove()

    return cam

# -------------------------
# PROCESS IMAGES
# -------------------------
if uploaded_files:
    for file in uploaded_files:
        try:
            image = Image.open(file).convert("RGB")

            st.image(image, caption=file.name, use_container_width=True)

            # Metadata
            st.write("📌 Image Info:")
            st.write(f"- Size: {image.size}")
            st.write(f"- Mode: {image.mode}")

            img_tensor = transform(image).unsqueeze(0)

            # Prediction
            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.softmax(output, dim=1)

            pred = torch.argmax(probs, dim=1)
            confidence = probs[0][pred.item()].item()

            # Show results
            st.subheader("Prediction:")
            st.success(f"{classes[pred.item()]}")

            st.write("### Confidence Scores")
            st.progress(float(probs[0][1]))  # REAL
            st.write(f"REAL: {round(probs[0][1].item()*100,2)}%")
            st.write(f"AI Generated: {round(probs[0][0].item()*100,2)}%")

            # Smart explanation
            st.subheader("🧠 Explanation")
            if confidence < 0.6:
                st.warning("Model is uncertain. This image has mixed signals.")
            elif pred.item() == 0:
                st.write("""
                Likely AI-generated because:
                - Unnatural textures
                - Repetitive patterns
                - Smooth or inconsistent details
                """)
            else:
                st.write("""
                Likely real because:
                - Natural noise patterns
                - Organic textures
                - Real-world lighting consistency
                """)

            # Grad-CAM
            st.subheader("🔥 Model Attention (Grad-CAM)")

            cam = generate_gradcam(model, img_tensor)

            fig, ax = plt.subplots()
            ax.imshow(image)
            ax.imshow(cam, cmap="jet", alpha=0.4)
            ax.axis("off")

            st.pyplot(fig)

            st.divider()

        except Exception as e:
            st.error(f"Error processing image: {e}")