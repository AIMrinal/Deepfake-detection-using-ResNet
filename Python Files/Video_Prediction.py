import os
import cv2
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm 
from mtcnn import MTCNN
import contextlib

#Model Loading
model = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 1),
    nn.Sigmoid() 
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.load_state_dict(torch.load('Load Model')) 

#Image Transformation
image_transforms = transforms.Compose([
    transforms.Resize(640),  
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
])

detector = MTCNN()

def extract_face(frame):
    faces = detector.detect_faces(frame)
    
    if len(faces) == 0:
        return None

    face_info = faces[0]
    x, y, w, h = face_info['box']

    face = frame[y:y+h, x:x+w]

    return face

# Image Prediction
def predict_image(image):
    model.eval()  
    image_tensor = image_transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probability = output.item()  
    label = "Fake" if probability > 0.5 else "Real"
    return label

# Frame Extraction
def extract_frames(video_path):
    frames = []
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    
    while success:
        frames.append(image)
        success, image = vidcap.read()

    return frames

#Video Processing
def process_video(video_frames):
    Fake = 0
    Real = 0
    
    for video_frame in video_frames:
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            face = extract_face(video_frame)
        
        if face is not None:
            image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            label = predict_image(image)
            if label == "Fake":
                Fake += 1
            else:
                Real += 1
                
    return Real, Fake

# Deepafake Prediction
def count_real_fake_videos(video_folder):
    real_videos = 0
    fake_videos = 0
    real_video_list = []
    fake_video_list = []
    
    for video_file in tqdm(os.listdir(video_folder), desc="Processing Videos"):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(video_folder, video_file)
            video_frames = extract_frames(video_path)
        
            real_frames, fake_frames = process_video(video_frames)
        
            if real_frames > fake_frames and real_frames - fake_frames > 20:
                real_videos += 1
                real_video_list.append(video_file)
            elif real_frames > fake_frames and real_frames - fake_frames < 20:
                fake_videos += 1
                fake_video_list.append(video_file)
            elif fake_frames > real_frames and fake_frames - real_frames > 20:
                fake_videos += 1
                fake_video_list.append(video_file)
            elif fake_frames > real_frames and fake_frames - real_frames < 20:
                real_videos += 1
                real_video_list.append(video_file) 

    
    return real_videos, fake_videos , real_video_list , fake_video_list

# Video Folder 
video_folder = "/Users/mrinal/Desktop/Test_Videos"
real_videos, fake_videos ,real_video_list ,fake_video_list  = count_real_fake_videos(video_folder)

#Printing Results
print("Real Videos:", real_videos)
print("List of Real Videos:")
for videos in real_video_list:
    print(videos)
print("Fake Videos:", fake_videos)
for videos in fake_video_list:
    print(videos)
