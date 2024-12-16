import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Load the trained model
model = load_model('/home/ramkumar/.h5')

def process_image(file_path):
    # Read and preprocess the image
    img = cv2.imread(file_path)
    img = cv2.resize(img, (224, 224))  # Resize the image to match model input size
    img = img / 255.0  # Normalize pixel values
    return img

def predict_tumor(image_path):
    img = process_image(image_path)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    return prediction[0][0] > 0.5  # Assuming binary classification with threshold 0.5

def mark_tumor(image_path):
    img = cv2.imread(image_path)
    # Apply your tumor detection and marking algorithm here
    # For example, you can use image segmentation techniques
    # and mark the detected tumor area in red
    # This part depends on your specific implementation

def generate_pdf(result, image_path, output_path):
    c = canvas.Canvas(output_path, pagesize=letter)
    c.drawString(100, 750, "Tumor Detection Report")
    c.drawString(100, 730, "Result: " + ("Tumor Detected" if result else "No Tumor Detected"))
    # Draw the MRI image
    c.drawImage(image_path, 100, 500, width=400, height=400)
    c.save()

def main():
    # Provide paths to your dataset folders
    yes_folder = '/home/ramkumar/MediScan_Insight/yes'
    no_folder = '/home/ramkumar/MediScan_Insight/no'
    output_folder = '/home/ramkumar/MediScan_Insight/outputs'

    for file in os.listdir(yes_folder):
        image_path = os.path.join(yes_folder, file)
        result = predict_tumor(image_path)
        if result:
            mark_tumor(image_path)
        output_path = os.path.join(output_folder, file.split('.')[0] + '.pdf')
        generate_pdf(result, image_path, output_path)

    for file in os.listdir(no_folder):
        image_path = os.path.join(no_folder, file)
        result = predict_tumor(image_path)
        output_path = os.path.join(output_folder, file.split('.')[0] + '.pdf')
        generate_pdf(result, image_path, output_path)

if __name__ == "__main__":
    main()
