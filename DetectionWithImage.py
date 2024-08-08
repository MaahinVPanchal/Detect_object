import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util

# Load the saved model
def load_model(model_dir):
    return tf.saved_model.load(model_dir)

# Load label map
def load_label_map(label_map_path):
    return label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)

# Detection function
def detect_objects(image, model, category_index):
    input_tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.uint8)
    detections = model(input_tensor)

    # Extract detection results
    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy().astype(np.int32)
    detection_scores = detections['detection_scores'][0].numpy()

    # Debug output
    print("Detection boxes:", detection_boxes)
    print("Detection classes:", detection_classes)
    print("Detection scores:", detection_scores)

    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        detection_boxes,
        detection_classes,
        detection_scores,
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.14)  # Adjust threshold as needed

    return image

# Paths to model and label map
model_dir = r'D:\Maahin Coding\ML_Project\Tensorflow\Testing\my_model\saved_model'
label_map_path = r'D:\Maahin Coding\ML_Project\Tensorflow\Testing\label_map.pbtxt'

# Load model and label map
detection_model = load_model(model_dir)
category_index = load_label_map(label_map_path)

# Read and preprocess the image
image_path = r'D:\Maahin Coding\ML_Project\Tensorflow\Testing\6_jpg.rf.e5df7f34f32b4a6a7dcdc0df1e0a931d.jpg'
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"Image not found at {image_path}")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run detection
output_image = detect_objects(image_rgb, detection_model, category_index)

# Convert the output image back to BGR for OpenCV
output_image_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

# Display the output image using OpenCV
cv2.imshow('Object Detection', output_image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()