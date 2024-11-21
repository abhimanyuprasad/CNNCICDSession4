import torch
from torchvision import datasets, transforms
from model.network import SimpleCNN
import pytest
import numpy as np
import os
import glob

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model_architecture():
    model = SimpleCNN()
    
    # Test 1: Check number of parameters
    num_params = count_parameters(model)
    assert num_params < 100000, f"Model has {num_params} parameters, should be less than 100000"
    
    # Test 2: Check input shape handling
    test_input = torch.randn(1, 1, 28, 28)
    try:
        output = model(test_input)
    except:
        pytest.fail("Model failed to process 28x28 input")
    
    # Test 3: Check output shape
    assert output.shape[1] == 10, f"Output should have 10 classes, got {output.shape[1]}"

def test_model_accuracy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = SimpleCNN().to(device)
    
    # Load the latest model
    model_files = glob.glob('models/*.pth')
    if not model_files:
        pytest.skip("No model files found in models directory")
    latest_model = max(model_files, key=os.path.getctime)
    checkpoint = torch.load(latest_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    # Test 4: Check accuracy
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    assert accuracy > 95, f"Accuracy is {accuracy}%, should be above 95%"

def test_model_robustness():
    """Test 5: Check model's robustness to noise"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    
    # Load the latest model
    model_files = glob.glob('models/*.pth')
    if not model_files:
        pytest.skip("No model files found in models directory")
    latest_model = max(model_files, key=os.path.getctime)
    checkpoint = torch.load(latest_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create noisy input
    test_input = torch.randn(1, 1, 28, 28).to(device)
    noise = torch.randn(1, 1, 28, 28).to(device) * 0.1
    noisy_input = test_input + noise
    
    # Model should not crash with noisy input
    try:
        output = model(noisy_input)
        assert output.shape[1] == 10, "Model failed to handle noisy input"
    except:
        pytest.fail("Model crashed with noisy input")

def test_model_activation_values():
    """Test 6: Check if activation values are reasonable"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    
    # Load model
    model_files = glob.glob('models/*.pth')
    if not model_files:
        pytest.skip("No model files found in models directory")
    latest_model = max(model_files, key=os.path.getctime)
    checkpoint = torch.load(latest_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_input = torch.randn(1, 1, 28, 28).to(device)
    output = model(test_input)
    
    # Check if outputs are in reasonable range
    assert not torch.isnan(output).any(), "Model produced NaN values"
    assert not torch.isinf(output).any(), "Model produced infinite values"
    assert output.abs().mean() < 10, "Activation values are unreasonably large"

def test_model_gradient_flow():
    """Test 7: Check if gradients flow properly"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    
    # Forward pass
    test_input = torch.randn(1, 1, 28, 28, requires_grad=True).to(device)
    output = model(test_input)
    loss = output.sum()
    
    # Backward pass
    loss.backward()
    
    # Check if gradients exist and are not zero
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"