import cv2
from cv2 import dnn_superres

# Create the SR object
sr = dnn_superres.DnnSuperResImpl_create()

# Read Image
image_path = 'Monarch-butterfly-tall-l.jpg'
image = cv2.imread(image_path)

# Load the desired model
model_path = "LapSRN_x8.pb"
sr.readModel(model_path)

# Set the desired model and scale to get correct pre- and post-processing
sr.setModel("lapsrn", 8)

# Upscale the image
result = sr.upsample(image)

# Save the image
cv2.imwrite("Monarch-butterfly-tall-l_upscaled_lapsrn.png", result)

