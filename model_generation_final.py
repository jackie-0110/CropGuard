if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms, models
    from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

    # Define transformations with augmentation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load dataset
    data_dir = 'PlantVillage/'  # Update path as needed
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Handle class imbalance
    train_indices = train_dataset.indices  # Get indices of training set
    train_targets = torch.tensor([dataset.targets[i] for i in train_indices])  # Extract corresponding labels

    class_counts = torch.bincount(train_targets)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[train_targets]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Load pre-trained EfficientNetV2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(dataset.classes))  # Adjust output layer
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    print("Starting training...")

    num_epochs = 15
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
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

    torch.save(model.state_dict(), "plant_disease_model.pth")
    print("Model saved to plant_disease_model.pth")