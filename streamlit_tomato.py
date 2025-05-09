import streamlit as st
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

# Import your model architectures here
from models import ANN, CNN, YOLOv7Classifier, create_vit_model

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class names (replace with your actual class list)
class_names = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___healthy',
    'Tomato__Septoria_leaf_spot',
    'Tomato_Spider_mites',
    'Tomato_Target_Spot',
    'Tomato_Mosaic_virus'
]

# Image transformation (must match training preprocessing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Model loading function
def load_model(model_name):
    model_map = {
       
        "CNN": {
            "class": CNN,
            "args": {"num_classes": len(class_names)},
            "file": "best_cnn_model.pth"
        },
        "YOLOv7": {
            "class": YOLOv7Classifier,
            "args": {"num_classes": len(class_names)},
            "file": "best_yolov7_model.pth"
        },
        
    }

    config = model_map[model_name]

    # Instantiate the model
    if model_name == "ViT":
        model = config["class"](**config["args"])  # ViT is a factory function
    else:
        model = config["class"](**config["args"])

    # Load weights
    model.load_state_dict(torch.load(config["file"], map_location=device))
    model = model.to(device)
    model.eval()

    return model


# Prediction function
def predict(image, model):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return outputs.cpu().numpy(), probabilities.cpu().numpy()


# Streamlit UI
st.title("🌿 Tomato Leaf Disease detection")
st.markdown("Compare different deep learning architectures for leaf disease detection.")

# Model selection
model_name = st.selectbox("Select Model Architecture", [ "CNN", "YOLOv7",])

# Load model with spinner
with st.spinner('Loading model...'):
    try:
        model = load_model(model_name)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Image upload
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', width=300)

    # Run prediction
    _, probs = predict(image, model)
    top5_prob, top5_class = torch.topk(torch.from_numpy(probs), 5, dim=1)

    # Display results
    st.subheader("Predictions")
    st.write(f"**Model Used:** {model_name}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Top 5 Predictions:**")
        for i in range(5):
            class_name = class_names[top5_class[0][i]]
            prob = top5_prob[0][i].item()
            st.write(f"{i + 1}. {class_name}: {prob:.2%}")

    with col2:
        st.markdown("**Confidence Distribution**")
        fig, ax = plt.subplots()
        ax.barh(
            [class_names[i] for i in top5_class[0].numpy()[::-1]],
            top5_prob[0].numpy()[::-1]
        )
        ax.set_xlim(0, 1)
        st.pyplot(fig)

# Sidebar
st.sidebar.header("🧠 Model Info")
st.sidebar.subheader("Architecture Details")
st.sidebar.write("""
- **CNN**: 4-layer convolutional network
- **YOLOv7**: Custom CNN backbone with 4 blocks
""")

st.sidebar.subheader("Model Performance")

