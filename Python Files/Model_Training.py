import torch
import torchvision
from torchvision import transforms, datasets 
import torch.nn as nn
import torch.optim as optim
import tqdm
import ssl

# Image Processing
class NestedImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        parent_folder = path.split('/')[-2]  
        label = 0 if parent_folder == 'Real' else 1
        return sample, label

data_transforms = transforms.Compose([
    transforms.Resize(640),  
    transforms.RandomHorizontalFlip(),  
    transforms.RandomRotation(15),  
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
])

# Data-Set Loading
train_dataset = NestedImageFolder(root='Enter Path for Training Data', transform=data_transforms)
valid_dataset = NestedImageFolder(root='Enter Path for Validation set', transform=data_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True) 
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=False) 

# Model Setup 
model = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT') 

num_features = model.fc.in_features  
model.fc = nn.Sequential(
    nn.Linear(num_features, 1),
    nn.Sigmoid()  
) 

device = "cuda" if torch.cuda.is_available() else "cpu"  
model.to(device)

# Training
criterion = nn.BCEWithLogitsLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  
epochs = 10  

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm.tqdm(train_loader, unit="batch", leave=False) 
    for i, (images, labels) in enumerate(loop): 
        images, labels = images.to(device), labels.to(device) 
        optimizer.zero_grad()
        outputs = model(images).squeeze(1)  
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Accuracy calculation
        predicted = (outputs > 0.5).float()  
        total += labels.size(0)
        correct += (predicted == labels.float()).sum().item()

        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=running_loss, acc=100*correct/total) 

    train_acc = 100 * correct / total    
    print(f'Epoch {epoch}, Train Loss: {running_loss/len(train_loader):.3f}, Train Accuracy: {train_acc:.2f}%')

    scheduler.step() # Update learning rate

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        val_loop = tqdm.tqdm(valid_loader, unit="batch", leave=False) 
        for images, labels in val_loop:
            images, labels = images.to(device), labels.to(device) 
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels.float())
            val_loss += loss.item()

            # Accuracy calculation
            predicted = (outputs > 0.5).float()  
            total += labels.size(0)
            correct += (predicted == labels.float()).sum().item()

            val_loop.set_description(f'Epoch {epoch}')
            val_loop.set_postfix(loss=val_loss, acc=100*correct/total) 

    val_acc = 100 * correct / total
    print(f'Epoch {epoch}, Val Loss: {val_loss/len(valid_loader):.3f}, Val Accuracy: {val_acc:.2f}%')
    torch.save(model.state_dict(), f'resnet50_epoch_{epoch}.pth')

# Explicitly close SSL context
ssl._create_default_https_context = ssl._create_unverified_context
