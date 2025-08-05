import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import os

st.set_page_config(page_title="DenseNet-121 Classifier", layout="centered")

# ---------------------------
# Config
# ---------------------------
MODEL_PATH = "model.pth"   # <- put your DenseNet .pth here
IMG_SIZE = 224
NUM_CLASSES = 5
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

# ---------------------------
# Load model (with robust handling)
# ---------------------------
@st.cache_resource
def load_model(model_path=MODEL_PATH):
    # build architecture
    model = models.densenet121(pretrained=False)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, NUM_CLASSES)

    # load the saved file
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    loaded = torch.load(model_path, map_location=torch.device("cpu"))

    # handle common save formats
    if isinstance(loaded, dict):
        # if saved as {'state_dict': ...} or plain state_dict
        if "state_dict" in loaded:
            state = loaded["state_dict"]
            # sometimes saved with "module." prefixes from DataParallel - strip if needed
        else:
            state = loaded

        # Strip "module." prefix if present
        new_state = {}
        for k, v in state.items():
            new_k = k
            if k.startswith("module."):
                new_k = k[len("module."):]
            new_state[new_k] = v

        model.load_state_dict(new_state)
    else:
        # loaded object might be an actual model (rare). Try to use it.
        try:
            # if loaded is a full model object, return it (but ensure eval mode)
            loaded.eval()
            return loaded
        except Exception:
            raise RuntimeError("Unrecognized model file format. Provide a state_dict or a full model object.")

    model.eval()
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ---------------------------
# Preprocessing
# ---------------------------
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # shape [1,3,H,W]

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("DenseNet-121 — APTOS DR Classifier")
st.write("Upload a retinal fundus image to predict diabetic retinopathy stage (0–4).")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file is None:
    st.info("Please upload an image file.")
else:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as ex:
        st.error(f"Cannot open image: {ex}")
        st.stop()

    st.image(image, caption="Uploaded image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Running inference..."):
            input_tensor = preprocess_image(image)  # cpu tensor
            with torch.no_grad():
                outputs = model(input_tensor)  # shape [1, NUM_CLASSES]
                probs = F.softmax(outputs[0], dim=0).cpu().numpy()
                pred_idx = int(probs.argmax())

        # Display results
        st.success(f"Predicted class: **{pred_idx} — {CLASS_NAMES[pred_idx]}**")
        st.write("Probabilities:")
        for i, p in enumerate(probs):
            st.write(f"- {i} ({CLASS_NAMES[i]}): {p*100:.2f}%")

        # Optional: show top-3 nicely
        topk = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:3]
        st.write("Top-3 predictions:")
        for i, p in topk:
            st.write(f"{i} — {CLASS_NAMES[i]}: {p*100:.2f}%")
