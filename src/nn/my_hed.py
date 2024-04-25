import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
from matplotlib import pyplot as plt
# https://github.com/s9xie/hed
# https://arxiv.org/abs/1504.06375


class HolisticallyNestedEdgeDetectionNet(nn.Module):
    def __init__(self):
        super(HolisticallyNestedEdgeDetectionNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.score_dsn1 = nn.Conv2d(32, 1, kernel_size=1, padding=0)
        self.score_dsn2 = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.score_dsn3 = nn.Conv2d(128, 1, kernel_size=1, padding=0)

        self.combine = torch.nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        score_dsn1 = F.interpolate(self.score_dsn1(conv1), size=(1024, 1024), mode="bilinear", align_corners=False)
        score_dsn2 = F.interpolate(self.score_dsn2(conv2), size=(1024, 1024), mode="bilinear", align_corners=False)
        score_dsn3 = F.interpolate(self.score_dsn3(conv3), size=(1024, 1024), mode="bilinear", align_corners=False)

        score_final = self.combine(torch.cat([score_dsn1, score_dsn2, score_dsn3], dim=1))

        return score_final


class HolisticallyNestedEdgeDetectionDataset(Dataset):
    def __init__(self, image_paths, target_paths, image_transform=None, target_transform=None):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.image_transform = image_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        target = Image.open(self.target_paths[idx])

        if self.image_transform:
            image = self.image_transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target


if __name__ == "__main__":
    net = HolisticallyNestedEdgeDetectionNet()
    print("Model initialized")

    image_transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
        transforms.GaussianBlur(3, 3),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    target_transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.Lambda(lambda x: x.convert("L")),
        transforms.ToTensor()
    ])

    # image_dir = "../../data/UDED/imgs"
    # image_dir = "../../data/dataset/imgs"
    image_dir = "../../data/my_dataset/imgs"
    # edge_dir = "../../data/UDED/gt"
    # edge_dir = "../../data/dataset/edges"
    edge_dir = "../../data/my_dataset/edges"
    image_files = sorted(os.listdir(image_dir))
    edge_files = sorted(os.listdir(edge_dir))
    image_paths = [os.path.join(image_dir, file) for file in image_files]
    edge_paths = [os.path.join(edge_dir, file) for file in edge_files]
    image_paths_train, image_paths_test, edge_paths_train, edge_paths_test = train_test_split(image_paths, edge_paths, test_size=0.2)

    # Create datasets
    train_dataset = HolisticallyNestedEdgeDetectionDataset(image_paths_train, edge_paths_train, image_transform=image_transform, target_transform=target_transform)
    test_dataset = HolisticallyNestedEdgeDetectionDataset(image_paths_test, edge_paths_test, image_transform=image_transform, target_transform=target_transform)
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

    torch.save(net.state_dict(), "../../models/edge_detection_hed_model_my_data_new.pth")
    print("Model saved")
    # net.load_state_dict(torch.load("../../models/edge_detection_hed_model_meta.pth"))
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

    import numpy as np
    output = np.array(output)
    thresholded = np.where(output > 180, output, 0).astype(np.uint8)

    plt.imshow(thresholded, cmap="gray")
    plt.axis("off")
    plt.show()
