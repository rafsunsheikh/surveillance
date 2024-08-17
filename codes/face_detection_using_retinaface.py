import os
import cv2
import time
from matplotlib import pyplot as plt
# %matplotlib inline
import face_detection
from cv2 import dnn_superres
import numpy as np

# Load the detector
detector = face_detection.build_detector("RetinaNetResNet50", confidence_threshold=.5, nms_iou_threshold=.3)

# Read image
image_path = "test_2.jpg"
main_image = cv2.imread(image_path)

# Save time
t0 = time.time()

# Getting the detections
detections = detector.detect(main_image)

# Calculate inference time
inf_time = round(time.time() - t0, 3)

# Print results
print(f"Inference time: {inf_time}s")
print(f"Number of faces detected: {len(detections)}")

# Create the 'faces' directory if it doesn't exist
output_dir = "faces"
os.makedirs(output_dir, exist_ok=True)

# Counter for saved faces
face_counter = 0

# Initialize the Super Resolution model
sr = dnn_superres.DnnSuperResImpl_create()
model_path = "LapSRN_x8.pb"
sr.readModel(model_path)
sr.setModel("lapsrn", 8)  # Setting the model and scale

kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])

# Draw detections and save individual faces
if len(detections) > 0:
    for detection in detections:
        # Extract the bounding box coordinates
        x_min, y_min, x_max, y_max = detection[:4]

        # Convert to integer values
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

        # Crop the face from the image
        face = main_image[y_min:y_max, x_min:x_max]

        # Save the face as a separate image in the 'faces' directory
        face_filename = os.path.join(output_dir, f"face_{face_counter}.jpg")
        cv2.imwrite(face_filename, face)

        # Upscale the face image
        upscaled_face = sr.upsample(face)

        # Sharpen the upscaled face
        upscaled_face = cv2.filter2D(upscaled_face, -1, kernel)
        
        # Save the upscaled face image
        upscaled_face_filename = os.path.join(output_dir, f"face_{face_counter}_upscaled.jpg")
        cv2.imwrite(upscaled_face_filename, upscaled_face)

        # Incremenr the face counter
        face_counter += 1


        # Draw the bounding box
        cv2.rectangle(main_image, (x_min, y_min), (x_max, y_max), (0,255,0),1)

# Write inference time
inference_time_added_image = cv2.putText(main_image, f"Inf Time: {inf_time}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
     


# # Display the original and upscaled images
# cv2.imshow('Original Face', face_image)
# cv2.imshow('Upscaled Face', upscaled_face)
# cv2.waitKey()
# cv2.destroyAllWindows()

# cv2.imshow('Sharpened Result image face', main_image)
# cv2.waitKey()
# cv2.destroyAllWindows()
     

     