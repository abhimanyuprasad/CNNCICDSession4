import torch
from torchvision import datasets, transforms
from model.network import SimpleCNN
import pytest

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
    import glob
    import os
    model_files = glob.glob('models/*.pth')
    latest_model = max(model_files, key=os.path.getctime)
    model.load_state_dict(torch.load(latest_model))
    
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