import streamlit as st
import torch
from PIL import Image
from torchvision import transforms, models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime


# Disease infor database
disease_info = {
    "Tomato_Early_blight": {
        "symptoms": "Dark spots with concentric rings on lower leaves",
        "treatment": ["Remove infected leaves", "Apply chlorothalonil or copper fungicides"],
        "prevention": ["Rotate crops", "Improve air circulation"]
    },
    "Potato_Late_blight": {
        "symptoms": "Water-soaked lesions on leaves, white mold underneath",
        "treatment": ["Apply fungicides containing mancozeb", "Destroy infected plants"],
        "prevention": ["Use resistant varieties", "Avoid overhead irrigation"]
    },
    "Pepper_bell_Bacterial_spot": {
        "symptoms": "Small, water-soaked spots on leaves, fruit, and stems",
        "treatment": ["Copper-based fungicides", "Remove infected plant parts"],
        "prevention": ["Use disease-free seeds", "Improve air circulation"]
    },
    "Pepper_bell_healthy": {
        "symptoms": "Healthy leaves and fruit, no signs of disease",
        "treatment": ["No treatment needed"],
        "prevention": ["Maintain healthy growing conditions"]
    },
    "Potato_Early_blight": {
        "symptoms": "Dark, circular lesions with rings on potato leaves",
        "treatment": ["Fungicides containing chlorothalonil or mancozeb", "Improve soil drainage"],
        "prevention": ["Crop rotation", "Use certified disease-free seed potatoes"]
    },
    "Potato_healthy": {
        "symptoms": "Healthy potato plant with no disease symptoms",
        "treatment": ["No treatment needed"],
        "prevention": ["Maintain healthy growing conditions"]
    },
    "Potato_Late_blight": {
        "symptoms": "Rapid development of dark lesions on leaves and stems, white mold in humid conditions",
        "treatment": ["Systemic fungicides", "Destroy infected plants promptly"],
        "prevention": ["Plant resistant varieties", "Ensure good air circulation"]
    },
    "Tomato_Target_Spot": {
        "symptoms": "Small, round spots with concentric rings on tomato fruit and leaves",
        "treatment": ["Copper fungicides", "Improve air circulation"],
        "prevention": ["Crop rotation", "Avoid overhead watering"]
    },
    "Tomato_Tomato_mosaic_virus": {
        "symptoms": "Mottled leaves, stunted growth, distorted fruit",
        "treatment": ["No cure, remove and destroy infected plants"],
        "prevention": ["Use resistant varieties", "Control aphids"]
    },
    "Tomato_Tomato_YellowLeaf_Curl_Virus": {
        "symptoms": "Upward curling and yellowing of leaves, stunted plant",
        "treatment": ["No cure, remove and destroy infected plants"],
        "prevention": ["Use resistant varieties", "Control whiteflies"]
    },
    "Tomato_Bacterial_spot": {
        "symptoms": "Small, dark, water-soaked spots on tomato leaves and fruit",
        "treatment": ["Copper fungicides", "Remove infected leaves"],
        "prevention": ["Use disease-free seeds", "Avoid overhead irrigation"]
    },
    "Tomato_Early_blight": {
        "symptoms": "Dark, concentric rings on tomato leaves, starting from lower leaves",
        "treatment": ["Chlorothalonil or mancozeb fungicides", "Improve air circulation"],
        "prevention": ["Crop rotation", "Mulching to prevent soil splash"]
    },
    "Tomato_healthy": {
        "symptoms": "Healthy tomato plant with no signs of disease",
        "treatment": ["No treatment needed"],
        "prevention": ["Maintain healthy growing conditions"]
    },
    "Tomato_Late_blight": {
        "symptoms": "Irregular, water-soaked lesions on tomato leaves and fruit, white mold in moist conditions",
        "treatment": ["Mancozeb or copper fungicides", "Improve air circulation"],
        "prevention": ["Plant resistant varieties", "Avoid overhead watering"]
    },
    "Tomato_Leaf_Mold": {
        "symptoms": "Pale green or yellow spots on upper leaf surface with gray-purple mold underneath",
        "treatment": ["Chlorothalonil or mancozeb fungicides", "Improve air circulation"],
        "prevention": ["Ensure good ventilation", "Avoid overhead watering"]
    },
    "Tomato_Septoria_leaf_spot": {
        "symptoms": "Small, circular spots with dark borders and tan centers on tomato leaves",
        "treatment": ["Copper fungicides", "Remove and destroy infected leaves"],
        "prevention": ["Crop rotation", "Mulching to prevent soil splash"]
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "symptoms": "Fine webbing on leaves, leaves appear stippled or bronzed",
        "treatment": ["Insecticidal soap or neem oil", "Increase humidity"],
        "prevention": ["Regularly inspect plants", "Maintain plant vigor"]
    }
}


# Classnames based on PlantVillage
class_names = [
    "Pepper_bell_Bacterial_spot",
    "Pepper_bell_healthy",
    "Potato_Early_blight",
    "Potato_healthy",
    "Potato_Late_blight",
    "Tomato_Target_Spot",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_Tomato_YellowLeaf_Curl_Virus",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite"
]

model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_names))


model.load_state_dict(torch.load('plant_disease_model.pth', map_location=torch.device('cpu'))) # CPU loaded, GPU not available currently
model.eval()  # Model set to evaluation mode

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# History session state 
if 'history' not in st.session_state:
    st.session_state.history = []

st.title("Crop Disease Detector 🌱")
st.sidebar.header("Field Management Tools")

# Image uploader
uploaded_file = st.file_uploader("Upload a crop leaf photo", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True) 

    transformed_image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(transformed_image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_class_idx = torch.max(probabilities, 0)
        predicted_class = class_names[predicted_class_idx]
        confidence = confidence.item()

    # Prediction print
    st.success(f"Prediction: {predicted_class} (Confidence: {confidence:.2f})")

    # Alert system
    if "blight" in predicted_class.lower() or "virus" in predicted_class.lower() or "mold" in predicted_class.lower():
        st.error("🚨 Critical disease detected! Immediate action required!")
    elif confidence < 0.60:
        st.warning("⚠️ Prediction confidence is low. Consider re-uploading image or consulting an expert.")


    # Disease info
    predicted_class_formatted = predicted_class.replace("_", " ").title() # Format for display in disease_info
    if predicted_class_formatted in disease_info:
        with st.expander("Disease Information & Recommendations"):
            info = disease_info[predicted_class_formatted] #  Formatted class name for lookup
            st.subheader("Symptoms")
            st.write(info["symptoms"])

            st.subheader("Recommended Treatment")
            for treatment in info["treatment"]:
                st.write(f"- {treatment}")

            st.subheader("Prevention Measures")
            for prevention in info["prevention"]:
                st.write(f"- {prevention}")

    # Add to history
    st.session_state.history.append({
        "timestamp": datetime.now(),
        "prediction": predicted_class_formatted, # Storing class name in proper format
        "confidence": confidence,
        "image": image.copy()
    })

# Field Health Visualization
st.sidebar.subheader("Field Health Map")
field_size = st.sidebar.slider("Field Size (acres)", 1, 50, 10)
map_type = st.sidebar.selectbox("Map Type", ["Disease Spread", "Soil Moisture", "Yield Potential"])

# Mock Field Data
def generate_field_map(size):
    np.random.seed(42) 
    return np.random.rand(size, size)

field_data = generate_field_map(field_size)

advice_dict = {
    "Disease Spread": {
        "Low": "No immediate action needed. Monitor for any signs of disease.",
        "Moderate": "Consider applying preventative fungicides or rotating crops.",
        "High": "Immediate intervention required! Apply treatment and isolate affected areas."
    },
    "Soil Moisture": {
        "Low": "Increase irrigation and check for drainage issues.",
        "Moderate": "Monitor soil conditions; adjust irrigation as needed.",
        "High": "Reduce watering to prevent root rot and soil erosion."
    },
    "Yield Potential": {
        "Low": "Optimize fertilization and assess pest/disease risk.",
        "Moderate": "Crop growth is stable, but minor improvements are possible.",
        "High": "Excellent yield potential! Maintain current practices."
    }
}

# Determine warning level based on field data
avg_risk = np.mean(field_data)
if avg_risk < 0.3:
    warning_level = "Low"
elif avg_risk < 0.7:
    warning_level = "Moderate"
else:
    warning_level = "High"


fig = px.imshow(field_data,
                color_continuous_scale='RdYlGn_r', # Red-Yellow-Green reversed for health
                title=f"Field Health Map - {map_type}",
                labels=dict(x="Field Width", y="Field Length", color=f"{map_type} Index")) # clearer labels
st.plotly_chart(fig)


# Prediction History
st.subheader("Detection History")
st.info(f"**Severity Level:** {warning_level}\n\n{advice_dict[map_type][warning_level]}")
if st.session_state.history:
    for i, entry in enumerate(reversed(st.session_state.history)):
        cols = st.columns([1, 4])
        cols[0].image(entry['image'], width=100)
        cols[1].write(f"**{entry['prediction']}** ({entry['confidence']:.2f})")
        cols[1].caption(entry['timestamp'].strftime("%Y-%m-%d %H:%M:%S"))
else:
    st.write("No detection history yet")

# Weather Integration
st.sidebar.subheader("Weather Advisory")
#Not able to use API currently.
weather = {
    "temp": 9,
    "humidity": 55,
    "conditions": "Sunny"
}

st.sidebar.metric("Temperature", f"{weather['temp']}°C")
st.sidebar.metric("Humidity", f"{weather['humidity']}%")
st.sidebar.write(f"Conditions: {weather['conditions']}")

# Prevention Tips
with st.expander("General Crop Care Tips"):
    st.write("""
    - Regular crop rotation helps prevent soil depletion
    - Maintain proper spacing between plants for air circulation
    - Monitor soil moisture levels regularly
    - Use organic mulch to retain moisture and prevent weeds
    """)

# Report Generation
if uploaded_file and predicted_class:
    report = f"""Crop Health Report\n
    Date: {datetime.now().strftime("%Y-%m-%d")}\n
    Diagnosis: {predicted_class_formatted}\n
    Confidence: {confidence:.2f}\n
    Recommended Actions:\n""" + "\n".join(disease_info.get(predicted_class_formatted, {}).get("treatment", []))

    st.download_button(
        label="Download Report",
        data=report,
        file_name="crop_health_report.txt",
        mime="text/plain"
    )