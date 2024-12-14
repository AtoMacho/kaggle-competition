# Kaggle Text Classification Competition 2024
This project is a participation in a Kaggle competition to classify text documents into binary categories using machine learning models. The dataset includes term count vectors, and the evaluation metric is the macro F1 score.

## Project Overview
The goal was to develop a robust text classification algorithm in two phases:

1. Phase 1: Implement a baseline model using only NumPy and Python to beat a logistic regression reference classifier.
2. Phase 2: Develop an advanced model using any tools or methods to optimize performance.

## Results
- Phase 1: Achieved an F1 score of 0.5714 using logistic regression with L2 regularization and class balancing.
- Phase 2: Improved the F1 score to 0.5989 using logistic regression with sklearn, Truncated SVD for dimensionality reduction, and hyperparameter tuning.
- Final Ranking:
    - Public leaderboard: 40th out of 151 teams.

## Features
1. ### Baseline Model:
- Implemented logistic regression from scratch using NumPy.
- Handled class imbalance with oversampling.
- Applied data normalization for stability and convergence.

2. ### Improved Model:
- Used sklearn’s logistic regression with regularization.
- Reduced dimensionality using Truncated SVD.
- Tuned hyperparameters and optimized classification thresholds.

## File Structure
1. ```classement.py```:
- Implements the baseline model with logistic regression from scratch.
- Handles data loading, preprocessing (normalization, splitting, oversampling), training, and evaluation.
- Outputs predictions in ```submission.csv```.

2. ```classement_improved.py```:
- Implements the improved model using sklearn.
- Adds dimensionality reduction with Truncated SVD.
- Includes threshold tuning to maximize the F1 score.
- Outputs predictions in ```submission_improved.csv```.

3. ### Data Files:
- ```data_train.npy```: Training data with term count vectors.
- ```data_test.npy```: Test data for prediction.
- ```vocab_map.npy```: Mapping of terms to indices.
- ```label_train.csv```: Labels for training data (binary).
  
4. ### Report:
- Detailed methodology, results, and discussions are provided in Rapport.pdf.

## Installation
1. Clone the repository:
```
git clone https://github.com/your-repo-name.git
cd your-repo-name
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Place the dataset files (```data_train.npy```, ```data_test.npy```, ```vocab_map.npy```, ```label_train.csv```) in the root directory.

## Usage
## Phase 1 - Baseline Model

Run the baseline model:

```
python classement.py
```

Output predictions will be saved in ```submission.csv```.

## Phase 2 - Improved Model
Run the improved model:

```
python classement_improved.py
```
Output predictions will be saved in ```submission_improved.csv```.

## Future Improvements
- Experiment with ensemble models like Gradient Boosting or XGBoost.
- Explore deep learning models such as transformers for text classification.

## Authors
- Lucky Khounvonsga (20172476)
- Hao Yuan Zhang (20208605)
