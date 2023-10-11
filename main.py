import cv2
import io
import os
from google.cloud import vision_v1p3beta1 as vision
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Set the path to your Google Cloud credentials JSON file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'client_key.json'

# Load the OpenPose model (You need to have OpenPose installed)
net = cv2.dnn.readNetFromTensorflow('path_to_openpose/frozen_inference_graph.pb')

def load_face_recognition_model():
    # Load a pre-trained face recognition model (You need to have a pre-trained model)
    model = torch.load('path_to_face_recognition_model.pth')
    model.eval()
    return model

def recognize_body_parts_and_features(image_path, output_image=True, save_to_file=False, face_recognition_model=None):
    img = cv2.imread(image_path)

    # Create a Google Vision client
    client = vision.ImageAnnotatorClient()

    # Read the image file
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Perform label detection
    response = client.label_detection(image=image)
    labels = response.label_annotations

    recognized_parts_and_features = []

    for label in labels:
        desc = label.description.lower()
        score = round(label.score, 2)

        recognized_parts_and_features.append((desc, score))

        if output_image:
            cv2.putText(img, f"{desc.upper()} ({score})", (300, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 200), 2)

    # Detect body parts using OpenPose
    img_cv2 = cv2.imread(image_path)
    blob = cv2.dnn.blobFromImage(img_cv2, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False)
    net.setInput(blob)
    output = net.forward()

    # Process OpenPose output to extract body part information
    body_parts = []
    for i in range(output.shape[1]):
        if i not in [0, 1, 2]:  # Skip background, left/right hands, and left/right feet
            heat_map = output[0, i, :, :]
            _, conf, _, point = cv2.minMaxLoc(heat_map)
            x, y = point
            body_parts.append((i, x, y, conf))

    for part_id, x, y, conf in body_parts:
        part_name = f"Body Part {part_id}"
        recognized_parts_and_features.append((part_name, conf))
        if output_image:
            cv2.putText(img, f"{part_name} ({conf:.2f})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(img, (x, y), 3, (0, 0, 255), thickness=-1)

    # Recognize facial features using a face recognition model
    if face_recognition_model is not None:
        image_pil = Image.open(image_path)
        image_tensor = transforms.ToTensor()(image_pil)
        image_tensor = image_tensor.unsqueeze(0)
        with torch.no_grad():
            embeddings = face_recognition_model(image_tensor)
        # Perform face recognition logic and add results to recognized_parts_and_features

    if save_to_file:
        with open("recognized_parts_and_features.txt", "a") as file:
            file.write(f"Image: {image_path}\n")
            for part_feature, score in recognized_parts_and_features:
                file.write(f"{part_feature}: {score}\n")
            file.write("\n")

    if output_image:
        cv2.imshow('Recognize & Draw', img)
        cv2.waitKey(0)

    return recognized_parts_and_features

if __name__ == "__main__":
    # Single image processing
    image_path = 'human_image.jpg'  # Update with your image path

    # Load the face recognition model (You need to have a pre-trained model)
    face_recognition_model = load_face_recognition_model()

    # Recognize human body parts and features
    recognized_parts_and_features = recognize_body_parts_and_features(image_path, output_image=True, save_to_file=True, face_recognition_model=face_recognition_model)
    print("Recognized Body Parts and Features:", recognized_parts_and_features)
