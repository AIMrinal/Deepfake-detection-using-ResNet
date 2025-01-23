import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm

model = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 1),
    nn.Sigmoid() 
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.load_state_dict(torch.load('Enter the path of Trained Model'))

image_transforms = transforms.Compose([
    transforms.Resize(640), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

def count_real_fake_images(image_folder):
    real_images = 0
    fake_images = 0
    real_images_list = []
    fake_images_list = []
    
    for image_file in tqdm(os.listdir(image_folder), desc="Processing Images"):
        if image_file.endswith(".jpg") or image_file.endswith(".jpeg") :
            image_path = os.path.join(image_folder, image_file)
            label = predict_image(image_path)
            if label =="Fake":
                fake_images_list.append(image_file)
                fake_images = fake_images+1
            if label == "Real":
                real_images_list.append(image_file)
                real_images = real_images+1
    return real_images, fake_images ,real_images_list ,fake_images_list


def predict_image(image_path):
    model.eval()
    image = Image.open(image_path)
    image = image.convert("RGB")
    image_tensor = image_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probability = output.item()
    label = "Fake" if probability > 0.5 else "Real"
    confidence = probability if label == "Fake" else 1 - probability
    return label


image_folder = '/Users/mrinal/Desktop/Test_images'
real_images, fake_images ,real_images_list ,fake_images_list  = count_real_fake_images(image_folder)

print("Real Image:", real_images)
print("List of Real Images:")
for images in real_images_list:
    print(images)
print("Deepfake Image:", fake_images)
print("List of Deepfake Images:")
for images in fake_images_list:
    print(images)

