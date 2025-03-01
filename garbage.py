import os
import cv2
import numpy as np
import tensorflow as tf

# Define the model path
model_path = r'garbage_classifier_binary.h5'  # Update this path to the correct location of your .h5 file

# Check if the model file exists
if not os.path.exists(model_path):
    print(f"Error: The file '{model_path}' does not exist.")
    exit()

# Load the saved model
print(f"Loading model from: {model_path}")
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading the model: {e}")
    exit()

# Define class labels for binary classification
classes = ['Non-Biodegradable', 'Biodegradable']  # Order must match how the model was trained

# Confidence threshold for valid predictions
CONFIDENCE_THRESHOLD = 0.7  # Adjust this value as needed (e.g., 0.7 = 70%)

# Initialize the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open webcam.")
    exit()
else:
    print("Webcam successfully opened!")

# Perform real-time inference
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the frame
    try:
        resized_frame = cv2.resize(frame, (150, 150))  # Resize to match model input size
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        normalized_frame = resized_frame / 255.0  # Normalize pixel values
        input_data = np.expand_dims(normalized_frame, axis=0).astype(np.float32)  # Add batch dimension
    except Exception as e:
        print(f"Error preprocessing the frame: {e}")
        continue

    # Run inference
    try:
        predictions = model.predict(input_data)[0]  # Get prediction probabilities
        predicted_class_index = int(predictions >= 0.5)  # Sigmoid output: 0 or 1
        confidence = predictions[0] if predicted_class_index == 1 else 1 - predictions[0]
    except Exception as e:
        print(f"Error during inference: {e}")
        continue

    # Debugging: Print raw predictions and confidence
    print(f"Raw prediction: {predictions}, Predicted class: {classes[predicted_class_index]}, Confidence: {confidence}")

    # Determine the label based on confidence threshold
    if confidence >= CONFIDENCE_THRESHOLD:
        label = f"{classes[predicted_class_index]} (Confidence: {confidence:.2f})"
    else:
        label = "Invalid (Object not recognized)"  # Object is neither biodegradable nor non-biodegradable

    # Display the result on the frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Garbage Classifier', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()