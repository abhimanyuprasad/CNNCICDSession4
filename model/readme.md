
Input:

create a python script that sets up a baic CI/CD pipeline for a machine learning project. Include automated testing for an
ML model, model validation checks, and a simple automated deployment process. I want you to make 3 layer DNN (Using convolutions
and fully connected layer), use MNIST as dataset (download if required), and train only for 1 epoch. When you test, mention
what testa you are checking for. I would like to check if model is under 100000 parammeters, take 28x28 input without any issues,
has 10 output and accuraciy is more than 70%. Make all the necessary files ( for training as well as github actions related).
Finally I should be able to confirm the build was successful. Tell me steps to run locally as well before i push to github. For deployment, add suffix to model
file to get some details on when it was trained. Add gitignore files to ignore data and setting to upload.


Steps to run locally before pushing to GitHub:
1. Create a virtual environment and activate it:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

2. Install required packages:
pip install torch torchvision pytest

3. Create necessary directories:
mkdir data models

4. Run the training:
python train.py

5. Run the tests:
pytest test_model.py -v

6. If all tests pass, you can commit and push to GitHub:
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-github-repo-url>
git push -u origin master

The tests will check for:
1. Model parameter count (< 100,000)
2. Input shape compatibility (28x28)
3. Output shape (10 classes)
4. Model accuracy (> 70%)

When you push to GitHub, the Actions workflow will:
1. Set up a Python environment
2. Install dependencies
3. Train the model
4. Run all tests
5. Archive the trained model as an artifact

The model files will be saved with timestamps (e.g., model_20240315_143022.pth), making it easy to track when each model was trained.
You can monitor the CI/CD pipeline in the "Actions" tab of your GitHub repository. The build will be marked as successful if all tests pass.