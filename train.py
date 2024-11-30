import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model.network import SimpleCNN
from datetime import datetime
import os
import matplotlib.pyplot as plt

def show_augmented_images(data_loader):
    # Get a batch of images
    images, labels = next(iter(data_loader))
    
    # Create a figure with 10 subplots
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    # Plot 10 images
    for idx in range(10):
        axes[idx].imshow(images[idx].squeeze(), cmap='gray')
        axes[idx].axis('off')
        axes[idx].set_title(f'Label: {labels[idx]}')
    
    plt.suptitle('Augmented MNIST Images')
    
    # Create directory for visualizations if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    plt.savefig('visualizations/augmented_images.png')
    plt.close()

def train():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Enhanced data augmentation
    transform = transforms.Compose([
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Show augmented images from first batch
    show_augmented_images(train_loader)
    print("Augmented images have been saved to 'visualizations/augmented_images.png'")
    
    # Initialize model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train for 1 epoch
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    # Save model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'models/model_{timestamp}.pth'
    os.makedirs('models', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
        'timestamp': timestamp
    }, save_path)
    print(f"Model saved as {save_path}")

if __name__ == "__main__":
    train() 