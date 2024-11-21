CICD Pipeline Badge:

![MLCICD Badge](https://github.com/abhimanyuprasad/CNNCICDSession4/actions/workflows/ml-pipeline.yml/badge.svg)


Steps to run locally before pushing to GitHub:

Create a virtual environment and activate it: 
python -m venv venv source venv/bin/activate # On Windows: venv\Scripts\activate

Install required packages: 
pip install torch torchvision pytest

Create necessary directories: mkdir data models

Run the training: python train.py

Run the tests: pytest test_model.py -v

If all tests pass, you can commit and push to GitHub: 
git init git add . 
git commit -m "Initial commit" 
git remote add origin 
git push -u origin master

The tests will check for:

Model parameter count (< 100,000)
Input shape compatibility (28x28)
Output shape (10 classes)
Model accuracy (> 95%)
When you push to GitHub, the Actions workflow will:

Set up a Python environment
Install dependencies
Train the model
Run all tests
Archive the trained model as an artifact

Key changes and additions:

In train.py:
Added comprehensive data augmentation (rotation, affine transforms, random erasing)
Enhanced model saving to include optimizer state and training metadata

In test_model.py added three new tests:
test_model_robustness: Tests the model's ability to handle noisy inputs
test_model_activation_values: Ensures the model produces reasonable activation values without NaN or infinite values
test_model_gradient_flow: Verifies that gradients are properly flowing through the model

These new tests will help ensure:
Model stability with perturbed inputs
Numerical stability of the network
Proper gradient flow during training
No vanishing/exploding gradient issues



