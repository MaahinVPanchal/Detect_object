import cv2
import os
import tensorflow as tf
import numpy as np

# Function to load label map from a .pbtxt file
def load_labels(path):
    labels = {}
    with open(path, 'r') as f:
        for line in f:
            if 'id:' in line:
                id = int(line.split(':')[1].strip())
            if 'name:' in line:
                name = line.split(':')[1].strip().strip("'")
                labels[id] = name
    return labels

# Absolute path to the saved model directory
model_dir = r'D:\Maahin Coding\ML_Project\Tensorflow\Testing\my_model\saved_model'

# Absolute path to the label map file
label_map_path = r'D:\Maahin Coding\ML_Project\Tensorflow\Testing\label_map.pbtxt'

# Load labels
labels = load_labels(label_map_path)

# Print the current working directory for verification
print(f"Current working directory: {os.getcwd()}")

# Check if the model directory exists
if not os.path.isdir(model_dir):
    print(f"Model directory not found: {model_dir}")
    exit()

# Load the TensorFlow model
model = tf.saved_model.load(model_dir)

# Function to run the model on an input image and get detections
def run_inference(model, image):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]  # Add batch dimension

    # Run inference
    detections = model(input_tensor)
    return detections

# Start video capture (0 for the default camera, you can change to the index of your camera)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend (Windows)
if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    # Prepare the frame for the model (convert BGR to RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = np.asarray(rgb_frame)

    # Run inference
    detections = run_inference(model, rgb_frame)
    
    # Extract detection data
    detection_boxes = detections['detection_boxes'].numpy()
    detection_scores = detections['detection_scores'].numpy()
    detection_classes = detections['detection_classes'].numpy()

    num_detections = int(detections['num_detections'][0])

    # Draw bounding box and label
    for i in range(num_detections):
        score = detection_scores[0][i]
        if score > 0.1:  # Adjust confidence threshold as needed
            bbox = detection_boxes[0][i]
            ymin, xmin, ymax, xmax = bbox

            # Convert normalized coordinates to pixel values
            (left, top, right, bottom) = (xmin * frame.shape[1], ymin * frame.shape[0], xmax * frame.shape[1], ymax * frame.shape[0])
            
            # Ensure coordinates are within the frame boundaries
            left, top, right, bottom = int(max(0, left)), int(max(0, top)), int(min(frame.shape[1], right)), int(min(frame.shape[0], bottom))
            
            # Draw the bounding box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), thickness=2)
            
            # Draw the label with score
            class_id = int(detection_classes[0][i])
            label = labels.get(class_id, 'Unknown')
            label_text = f"{label}: {score:.2f}"
            cv2.putText(frame, label_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
