from typing import List, Any

import streamlit as st
import supervision as sv
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from PIL import Image, ImageDraw, ImageFont
import yaml
import pages.gamification as g

# initialize session state
g.load_session_state()

# Paths to the dataset folders and yaml file
test_images_directory_path = "streamlit/content/datasets/animalDataset/test/images"
test_annotations_directory_path = "streamlit/content/datasets/animalDataset/test/labels"
data_yaml_path = "streamlit/content/datasets/animalDataset/data.yaml"

# Load the model
model = YOLO("streamlit/models/wildwatchyolov8_v2_finetuning05.pt")

# Load class names from data.yaml
with open(data_yaml_path, 'r') as file:
    data = yaml.safe_load(file)
class_names = data['names']

# Callback function for model predictions
def callback(image: np.ndarray):
    result = model(image)[0]
    return result

# Function to draw bounding boxes on the image
def draw_bounding_boxes(image: np.ndarray, result) -> np.ndarray:
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.load_default()

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        label = int(box.cls.cpu().numpy()[0])
        class_name = class_names[label]
        st.write(class_name)
        st.session_state.ai_result = str(class_name)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1), class_name, fill="red", font=font)

    return np.array(pil_image)

# Streamlit app setup
st.title("WildWatchAI's fine-tuned YOLOv8 model")
st.sidebar.title("Navigation")
options = ["Home", "Evaluation", "Visualizations", "Gamification"]
choice = st.sidebar.radio("Go to", options)

if choice == "Home":
    st.subheader("Upload an image to detect animals")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Running inference...")
        result = callback(image)
        image_with_boxes = draw_bounding_boxes(image, result)
        st.image(image_with_boxes, caption='Detected Animals', use_column_width=True)

elif choice == "Evaluation":
    st.subheader("Model Evaluation Metrics")
    st.write("Computing confusion matrix and precision scores...")

    # Create the test dataset
    dataset = sv.DetectionDataset.from_yolo(
        images_directory_path=test_images_directory_path, 
        annotations_directory_path=test_annotations_directory_path,
        data_yaml_path=data_yaml_path
    )

    def callback(image: np.ndarray) -> sv.Detections:
        result = model(image)[0]
        return sv.Detections.from_ultralytics(result)

    # Evaluate and get the confusion matrix
    confusion_matrix = sv.ConfusionMatrix.benchmark(
        dataset=dataset,
        callback=callback,
    )

    # Plot normalized confusion matrix
    plt.figure(figsize=(10, 10))
    confusion_matrix.plot(normalize=True)
    st.pyplot(plt.gcf())

    """
    # Extract true labels and predicted labels
    true_labels = []
    predicted_labels = []
    for data in dataset:
        image, labels = data.image, data.labels  # Adjust indexing based on the structure
        detections = callback(image)
        true_labels.extend(labels[:, 0].cpu().numpy())  # Adjust to extract the class IDs
        predicted_labels.extend(detections.boxes.cls.cpu().numpy())

    # Calculate precision for each class
    precision = precision_score(true_labels, predicted_labels, average=None)
    precision_dict = {f'Class {i}': p for i, p in enumerate(precision)}

    # Display precision values as a bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(precision_dict.keys(), precision_dict.values())
    plt.xlabel('Class')
    plt.ylabel('Precision')
    plt.title('Precision for Each Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
 
    """

elif choice == "Visualizations":
    st.subheader("Model Visualizations")
    st.write("Visualizations go here")

elif choice == "Gamification":

    col0, col1, col2 = st.columns([0.1, 0.8, 0.1], gap="small")
    with col1:
        st.image(st.session_state.current_image, use_column_width=True)
        g.render_radio()

        if "solve_button_clicked" not in st.session_state:
            st.session_state.solve_button_clicked = False
        if "ai_image" not in st.session_state:
            st.session_state.ai_image = None

        if st.button("Lösung abgeben", disabled=st.session_state.solve_button_disabled):
            st.session_state.solve_button_clicked = True
            image = Image.open(st.session_state.current_image)
            npimage = np.array(image)
            st.write("")
            st.write("Running inference...")
            result = callback(npimage)
            image_with_boxes = draw_bounding_boxes(npimage, result)
            st.image(image_with_boxes, caption='Detected Animals', use_column_width=True)
            st.session_state.ai_image = image_with_boxes
            g.solve_button()

        if st.button(
                "Weiter",
                disabled=st.session_state.continue_button_disabled
        ):
            g.continue_button()

        st.text(f"Punktzahl: {st.session_state.score}")
        if st.session_state.game_over:
            st.text("GAME OVER")