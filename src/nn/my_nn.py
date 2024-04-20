import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt


class EdgeDetectionNet(nn.Module):
    def __init__(self):
        super(EdgeDetectionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 1, kernel_size=3, padding=1)  # Change this line
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


def train(net, dataloader, epochs=5):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters())

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, loss: {running_loss / len(dataloader)}')

    print('Finished Training')


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

        # Resize the images and targets to the same size
        image = image.resize((560, 425))
        target = target.resize((560, 425))

        if self.image_transform:
            image = self.image_transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        target = self.pool(target)

        return image, target


net = EdgeDetectionNet()

image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

target_transform = transforms.Compose([
    transforms.ToTensor()
])
image_dir = "../../data/UDED/imgs"
edge_dir = "../../data/UDED/gt"
image_files = sorted(os.listdir(image_dir))
edge_files = sorted(os.listdir(edge_dir))
image_paths = [os.path.join(image_dir, file) for file in image_files]
edge_paths = [os.path.join(edge_dir, file) for file in edge_files]
dataset = EdgeDetectionDataset(image_paths, edge_paths, image_transform=image_transform, target_transform=target_transform)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

train(net, data_loader, epochs=100)

# Save the model
torch.save(net.state_dict(), "../../models/edge_detection_model.pth")

img = Image.open("../../data/img/pebbles.jpg")
img = image_transform(img)
output = net(img)
output = output.squeeze(0).detach().numpy()
output = (255 * output).astype("uint8")
output = Image.fromarray(output)
plt.imshow(output, cmap='gray')
plt.show()
