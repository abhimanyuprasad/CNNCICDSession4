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
Model accuracy (> 70%)
When you push to GitHub, the Actions workflow will:

Set up a Python environment
Install dependencies
Train the model
Run all tests
Archive the trained model as an artifact
