import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pandas as pd
import os
from PIL import Image

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Custom dataset class
class PlantDocDataset(Dataset):
    def __init__(self, img_dir, labels_csv, transform=None):
        self.img_dir = img_dir
        self.labels_df = pd.read_csv(labels_csv)
        self.transform = transform
        self.class_to_idx = {cls: i for i, cls in enumerate(sorted(self.labels_df["class"].unique()))}

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_name = row["filename"].split("?")[0]  # Remove query parameters
        img_path = os.path.join(self.img_dir, img_name)

        try:
            # Open image
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: {img_path} not found, skipping this file.")
            return None  # Return None when file not found

        if self.transform:
            image = self.transform(image)

        label = self.class_to_idx[row["class"]]
        return image, label


# ✅ Wrap data loading in "if __name__ == '__main__'" to avoid multiprocessing issues
if __name__ == "__main__":
    data_dir = "PlantDoc"
    train_csv = os.path.join(data_dir, "train_labels.csv")
    test_csv = os.path.join(data_dir, "test_labels.csv")
    train_img_dir = os.path.join(data_dir, "TRAIN")
    test_img_dir = os.path.join(data_dir, "TEST")

    # Load datasets
    train_dataset = PlantDocDataset(train_img_dir, train_csv, transform)
    test_dataset = PlantDocDataset(test_img_dir, test_csv, transform)

    # Compute class weights for balanced training
    train_labels = train_dataset.labels_df["class"].values
    class_counts = pd.Series(train_labels).value_counts().sort_index().values
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    sample_weights = torch.tensor([class_weights[train_dataset.class_to_idx[label]] for label in train_labels])

    # Weighted sampler for class balance
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=2, drop_last=True) # Added drop_last
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, drop_last=True) # Added drop_last

    # Load EfficientNetV2
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(train_dataset.class_to_idx))
    model = model.to(device)

    # Loss function & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    num_epochs = 15
    print("Starting training...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            # Quick fix: Skip batches if images is None (due to missing files)
            if images is None:
                print("Warning: Skipping batch due to None images.") # Optional: More explicit warning during training
                continue

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {correct/total:.4f}")

    print("Training complete.")

    # Save model
    torch.save(model.state_dict(), "plant_disease_model.pth")
    print("Model saved to plant_disease_model.pth")