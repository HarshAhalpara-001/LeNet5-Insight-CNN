import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Define LeNet-like CNN model
class MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.C1 = nn.Conv2d(1, 6, 5)
        self.S2_S4 = nn.MaxPool2d(2, 2)
        self.C3 = nn.Conv2d(6, 16, 5)
        self.C5 = nn.Conv2d(16, 120, 5)
        self.F6 = nn.Linear(120, 84)
        self.Output = nn.Linear(84, 10)

    def forward(self, x):
        layer_outputs = {}
        
        x = self.C1(x)
        layer_outputs["C1"] = x.clone().detach()
        x = self.relu(x)
        x = self.S2_S4(x)
        layer_outputs["S2_S4"] = x.clone().detach()

        x = self.C3(x)
        layer_outputs["C3"] = x.clone().detach()
        x = self.relu(x)
        x = self.S2_S4(x)
        layer_outputs["S4"] = x.clone().detach()

        x = self.C5(x)
        layer_outputs["C5"] = x.clone().detach()
        x = self.relu(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.F6(x)
        layer_outputs["F6"] = x.clone().detach()
        x = self.relu(x)

        x = self.Output(x)
        layer_outputs["Output"] = x.clone().detach()
        
        return x, layer_outputs

# Load model
model = MNIST()
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

# Streamlit UI
st.title("🖍 Handwritten Digit Classifier - LeNet5")

canvas_result = st_canvas(
    fill_color="red", 
    stroke_width=10, 
    stroke_color="white", 
    background_color="black",
    width=280, 
    height=280, 
    drawing_mode="freedraw", 
    key="canvas"
)

if canvas_result.image_data is not None:
    img = np.array(canvas_result.image_data[:, :, 0])  # Convert to grayscale
    img = cv2.resize(img, (32, 32))  # Resize to 32x32
    img = img / 255.0  # Normalize
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1,1,32,32)

    with torch.no_grad():
        output, layer_outputs = model(img)
        softmax_output = torch.softmax(output, dim=1)  # Apply Softmax

    predicted_digit = torch.argmax(output).item()
    st.subheader(f"🧠 Predicted Digit: {predicted_digit}")

    st.subheader("🔍 Probability Distribution (Softmax Output)")
    
    # Convert Softmax output to DataFrame
    softmax_probs = softmax_output.numpy().flatten()
    df_softmax = pd.DataFrame({"Digit": list(range(10)), "Probability": softmax_probs})

    # Display Table
    # st.write(df_softmax)

    # Seaborn Bar Plot for Softmax Probabilities
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=df_softmax["Digit"], y=df_softmax["Probability"], palette="viridis", ax=ax)
    ax.set_xlabel("Digit")
    ax.set_ylabel("Probability")
    ax.set_title("Softmax Output (Digit Probabilities)")
    st.pyplot(fig)

    st.subheader("🔍 Intermediate Layer Outputs")
    
    for layer, data in layer_outputs.items():
        st.write(f"## {layer} Layer")
        if layer == "C5":  # Special case for C5 layer
            st.write("#### C5 Feature Map as Row-wise Matrix")
            data = data[0].squeeze()
            df = pd.DataFrame(data.reshape(1,120))
            st.write(df)

            # Seaborn Bar Plot for C5 Layer
            fig, ax = plt.subplots(figsize=(12, 5))
            sns.barplot(x=list(range(1, 121)), y=data.numpy().flatten(), palette="viridis", ax=ax)
            ax.set_xlabel("Neuron Index")
            ax.set_ylabel("Activation Value")
            ax.set_title("C5 Layer Activations")
            ax.set_xticks([])
            st.pyplot(fig)
        elif data.ndim == 4:  # Convolutional layers (Feature Maps)
            num_cols = (data.shape[1])  # Limit to 6 images for clarity
            cols = st.columns(num_cols)
            
            for i in range(num_cols):
                feature_map = data[0, i].numpy()
                feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-10)  # Normalize
                cols[i].image(feature_map, caption=f"F.M {i+1}", use_container_width=True)

        else :  # Fully Connected layers
            df = pd.DataFrame(data.numpy())
            st.write("### Table Representation:")
            st.write(df)
      
