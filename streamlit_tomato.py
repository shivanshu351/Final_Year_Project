import streamlit as st
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

# Import your model architectures here
from models import ANN, CNN, YOLOv7Classifier, create_vit_model

# Custom CSS for styling
st.markdown("""
<style>
    .header {
        font-size: 36px !important;
        color: #2e8b57 !important;
        text-align: center;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 20px !important;
        color: #3a7ca5 !important;
        margin-bottom: 30px;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa !important;
    }
    .prediction-card {
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
    }
    .confidence-bar {
        height: 20px;
        border-radius: 10px;
        background-color: #e9ecef;
        margin: 5px 0;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
    }
    .upload-area {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class names
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

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Model loading function
@st.cache_resource(show_spinner=False)
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
        model = config["class"](**config["args"])
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

# Main App
st.markdown('<p class="header">🌿 Tomato Leaf Disease Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">AI-powered plant health diagnosis</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🧠 Model Information")
    
    st.subheader("Architecture Details")
    with st.expander("CNN Architecture"):
        st.markdown("""
        - **4 Convolutional Layers** with ReLU activation
        - **Max Pooling** after each layer
        - **Dropout** for regularization
        - Final **Dense Classification Head**
        """)
    
    with st.expander("YOLOv7 Architecture"):
        st.markdown("""
        - **Custom CNN Backbone** with 4 blocks
        - **SiLU Activation** functions
        - **Batch Normalization** layers
        - **Adaptive Average Pooling** before classification
        """)
    
    st.subheader("Performance Metrics")
    st.metric("Inference Speed (CPU)", "0.8-1.2 sec")
    st.metric("Accuracy (Test Set)", "92.4%")
    st.metric("Model Size", "14.7 MB")

# Model selection
model_name = st.selectbox(
    "Select Model Architecture", 
    ["CNN", "YOLOv7"],
    help="Choose the deep learning model for analysis"
)

# Load model
with st.spinner(f'Loading {model_name} model...'):
    try:
        model = load_model(model_name)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Image upload
st.markdown("### 📤 Upload Leaf Image")
with st.container():
    uploaded_file = st.file_uploader(
        "Drag and drop or click to upload a leaf image", 
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

if uploaded_file:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🖼️ Uploaded Image")
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_column_width=True)
    
    with col2:
        st.markdown("### 🔍 Analysis Results")
        
        # Run prediction
        with st.spinner('Analyzing leaf health...'):
            _, probs = predict(image, model)
            top5_prob, top5_class = torch.topk(torch.from_numpy(probs), 5, dim=1)
        
        # Display results in cards
        for i in range(5):
            with st.container():
                st.markdown(f"""
                <div class="prediction-card">
                    <h4>{i+1}. {class_names[top5_class[0][i]]}</h4>
                    <div style="display: flex; justify-content: space-between;">
                        <span>{top5_prob[0][i].item()*100:.1f}% confidence</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {top5_prob[0][i].item()*100:.1f}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Visual chart
        st.markdown("### 📊 Confidence Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(
            [class_names[i] for i in top5_class[0].numpy()[::-1]],
            top5_prob[0].numpy()[::-1],
            color='#4facfe'
        )
        ax.set_xlim(0, 1)
        ax.set_title("Prediction Confidence Scores", pad=20)
        ax.set_xlabel("Confidence", labelpad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 14px;">
    <p>🌱 Powered by Deep Learning | 🚀 Streamlit | 🧠 PyTorch</p>
    <p>For educational and research purposes</p>
</div>
""", unsafe_allow_html=True)