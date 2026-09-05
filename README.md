# Titanic Survival Prediction - Kaggle Competition

## Project Overview
This is an independent machine learning project based on the classic Kaggle Titanic competition. The goal is to predict passenger survival using binary classification models. The project covers the full workflow from raw data cleaning, feature engineering, model training to result submission.

This project is built to practice end-to-end ML implementation and strengthen data modeling capabilities for further study in data-driven engineering fields.

## Tech Stack
- **Data Processing**: Pandas, NumPy
- **Modeling**: Scikit-learn (Random Forest Classifier)
- **Validation**: 5-Fold Cross Validation
- **Tools**: Python, Kaggle CLI

## Project Workflow
### 1. Data Preprocessing
- Handled missing values in `Age`, `Fare`, `Embarked` and `Cabin` columns
- Implemented **title-based group-wise median imputation** for age, which reduced estimation bias compared with global median filling

### 2. Feature Engineering
Extracted 13 features from raw data, including:
- Social status title extracted from passenger names (Mr, Mrs, Miss, Master, etc.)
- Family size and alone status derived from `SibSp` and `Parch`
- Age band and fare band via binning
- Cabin deck identifier extracted from cabin number
- Categorical encoding for all text features

### 3. Model Training & Tuning
- Built Random Forest classification model
- Tuned hyperparameters (`max_depth`, `n_estimators`, `min_samples_leaf`) via 5-fold cross validation
- Fixed data path mismatch and empty test set issues during implementation

## Results
| Metric | Score |
|--------|-------|
| Baseline online accuracy | 0.75837 |
| Cross-validation accuracy (after optimization) | 0.8260 |
| Final online submission score | 0.75837 (baseline) / pending optimization |

> The project demonstrates that systematic feature engineering can significantly improve model performance beyond raw data training.

## How to Run
1. Download `train.csv` and `test.csv` from [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic/data)
2. Put the data files into a `data` folder in the project directory
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
4. Run the script:
    ```bash
   python analysis.py
5. The submission file will be generated at `./data/submission.csv`

## File Structure
    
    titanic-survival-prediction/
    ├── README.md               # Project documentation
    ├── analysis.py             # Full code with data processing and modeling
    └── requirements.txt        # Python dependencies
