import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
from matplotlib import pyplot as plt


class EdgeDetectionNet(nn.Module):
    def __init__(self):
        super(EdgeDetectionNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class EdgeDetectionDataset(Dataset):
    def __init__(self, image_paths, target_paths, image_transform=None, target_transform=None):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.pool = nn.MaxPool2d(2, 2)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        target = Image.open(self.target_paths[idx])

        if self.image_transform:
            image = self.image_transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        target = target.unsqueeze(0)
        target = self.pool(target)
        target = self.pool(target)
        target = target.squeeze(0)

        return image, target


net = EdgeDetectionNet()
print("Model initialized")

image_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.ToTensor(),
    transforms.GaussianBlur(3, 3),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
target_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.Lambda(lambda x: x.convert("L")),
    transforms.ToTensor()
])

# image_dir = "../../data/UDED/imgs"
# image_dir = "../../data/BIPED/edges/imgs/train/rgbr/real"
image_dir = "../../data/my_dataset/imgs"
# edge_dir = "../../data/UDED/gt"
# edge_dir = "../../data/BIPED/edges/edge_maps/train/rgbr/real"
edge_dir = "../../data/my_dataset/edges"
image_files = sorted(os.listdir(image_dir))
edge_files = sorted(os.listdir(edge_dir))
image_paths = [os.path.join(image_dir, file) for file in image_files]
edge_paths = [os.path.join(edge_dir, file) for file in edge_files]
image_paths_train, image_paths_test, edge_paths_train, edge_paths_test = train_test_split(image_paths, edge_paths, test_size=0.2)

# Create datasets
train_dataset = EdgeDetectionDataset(image_paths_train, edge_paths_train, image_transform=image_transform, target_transform=target_transform)
test_dataset = EdgeDetectionDataset(image_paths_test, edge_paths_test, image_transform=image_transform, target_transform=target_transform)
print(f"Train dataset created with {len(train_dataset)} samples")
print(f"Test dataset created with {len(test_dataset)} samples")

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
print("DataLoader created")

print("Starting training...")
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters())
epochs = 100

for epoch in range(epochs):
    for data in train_loader:
        inputs, targets = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    net.eval()
    with torch.no_grad():
        test_loss = 0
        for data in test_loader:
            inputs, targets = data
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
        print(f"Epoch {epoch + 1}, loss: {test_loss / len(test_loader)}")

print("Training finished")

torch.save(net.state_dict(), "../../models/edge_detection_model_my_data.pth")
print("Model saved")
# net.load_state_dict(torch.load("../../models/edge_detection_model_only_bigger_data_meta.pth"))
# print("Model loaded")

img = Image.open("../../data/img/pebbles.jpg")
orig_w, orig_h = img.size
img = image_transform(img)
img = img.unsqueeze(0)  # Batch size dimension
print("Image loaded and transformed")

output = net(img)
print("Forward pass completed")

output = output.squeeze(0).detach().numpy()
output = output[0, :, :]  # Convert to 2D
output = (output - output.min()) / (output.max() - output.min()) * 255
output = output.astype("uint8")
output = Image.fromarray(output)
output = output.resize((orig_w, orig_h))
print("Output processed")

plt.imshow(output, cmap="gray")
plt.axis("off")
plt.show()

output = np.array(output)
thresholded = np.where(output > 100, output, 0).astype(np.uint8)

plt.imshow(thresholded, cmap="gray")
plt.axis("off")
plt.show()
