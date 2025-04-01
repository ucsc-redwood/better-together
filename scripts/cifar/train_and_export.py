import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np


# Your AlexNet-style model
class SimpleAlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleAlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 4 * 4, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train_and_export(num_epochs=2, batch_size=128, save_dir="weights_npy"):
    # Data transforms
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )

    # Use your locally downloaded dataset
    dataset_root = "./"
    train_set = torchvision.datasets.CIFAR10(
        root=dataset_root, train=True, download=True, transform=transform
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )

    # Export the CIFAR-10 images as .npy files
    os.makedirs(save_dir, exist_ok=True)

    # Get the raw dataset (without transforms) to export original images
    cifar10_data = torchvision.datasets.CIFAR10(
        root=dataset_root, train=True, download=True, transform=None
    )

    # Export all images (or a subset if dataset is large)
    print("Exporting CIFAR-10 images as individual .npy files...")
    num_images_to_export = 10  # Can adjust this if needed

    # Create subdirectory for images
    images_dir = os.path.join(save_dir, "cifar10_images")
    os.makedirs(images_dir, exist_ok=True)

    # Create a mapping file to store label information
    label_mapping = {}

    for i in range(min(num_images_to_export, len(cifar10_data))):
        img, label = cifar10_data[i]
        # Save each image as individual .npy file
        img_path = os.path.join(images_dir, f"img_{i:05d}.npy")
        # Convert image from HWC format (32,32,3) to CHW format (3,32,32) as expected by the C++ code
        img_array = np.array(img).transpose(2, 0, 1)
        np.save(img_path, img_array)

        # Store label information
        label_mapping[f"img_{i:05d}.npy"] = int(label)

    # Save the label mapping as a numpy array for easy loading
    np.save(os.path.join(save_dir, "cifar10_labels_map.npy"), label_mapping)
    print(
        f"✅ Exported {min(num_images_to_export, len(cifar10_data))} individual CIFAR-10 images to ./{images_dir}/"
    )

    # Model, optimizer, loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleAlexNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop (simple)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f"[{epoch+1}, {i+1}] loss: {running_loss / 100:.4f}")
                running_loss = 0.0

    print("✅ Training complete.")

    # Save weights & biases
    def save(name, tensor):
        np.save(os.path.join(save_dir, f"{name}.npy"), tensor.detach().cpu().numpy())

    save("conv1_w", model.conv1.weight)
    save("conv1_b", model.conv1.bias)
    save("conv2_w", model.conv2.weight)
    save("conv2_b", model.conv2.bias)
    save("conv3_w", model.conv3.weight)
    save("conv3_b", model.conv3.bias)
    save("conv4_w", model.conv4.weight)
    save("conv4_b", model.conv4.bias)
    save("conv5_w", model.conv5.weight)
    save("conv5_b", model.conv5.bias)
    save("linear_w", model.fc.weight)
    save("linear_b", model.fc.bias)

    print(f"✅ Saved all weights to ./{save_dir}/")


if __name__ == "__main__":
    train_and_export()
