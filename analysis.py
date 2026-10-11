# ===================== Import Basic Libraries =====================
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===================== Path & Directory Auto-Configuration =====================
# Get the directory of the current script to eliminate runtime working directory dependency
current_dir = os.path.dirname(os.path.abspath(__file__))
# Define paths for each directory
data_dir = os.path.join(current_dir, "data")
report_dir = os.path.join(current_dir, "report")
# Auto-create directories (no error if they already exist)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)
# Full file path for datasets
train_file = os.path.join(data_dir, "train.csv")
cleaned_file = os.path.join(data_dir, "train_cleaned.csv")

# ===================== Plot Font Configuration =====================
plt.rcParams['font.sans-serif'] = ['SimHei']  # Display Chinese on Windows
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Uncomment for Mac users
plt.rcParams['axes.unicode_minus'] = False  # Fix negative sign display issue

# ===================== Main Program Entry =====================
def main():
    # ===================== 1. Data Loading & Preliminary Exploration =====================
    try:
        df = pd.read_csv(train_file)
        print("✅ Raw data loaded successfully")
    except FileNotFoundError:
        print(f"❌ Dataset not found: {train_file}")
        print("Please place train.csv into the data folder and run again")
        return
    print("Data dimensions:", df.shape)
    print("\nFirst 5 rows of data:")
    print(df.head())
    print("\nField information:")
    df.info()
    print("\nStatistical overview of numeric fields:")
    print(df.describe())

    # ===================== 2. Data Cleaning =====================
    # 2.1 Remove duplicate values
    print(f"\nNumber of duplicate samples: {df.duplicated().sum()}")
    df = df.drop_duplicates()
    # 2.2 Handle missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())  # Fill age with median
    df = df.drop('Cabin', axis=1)  # Drop Cabin due to excessively high missing rate
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # Fill embarkation port with mode
    # 2.3 Handle outliers (Fare capping method)
    Q1 = df['Fare'].quantile(0.25)
    Q3 = df['Fare'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df['Fare'] = np.where(df['Fare'] > upper, upper, df['Fare'])
    df['Fare'] = np.where(df['Fare'] < lower, lower, df['Fare'])
    # 2.4 Data type conversion
    df['Survived'] = df['Survived'].astype('category')
    df['Pclass'] = df['Pclass'].astype('category')
    df['Sex'] = df['Sex'].astype('category')
    # 2.5 Drop irrelevant fields
    df = df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
    print("\n✅ Data cleaning completed")

    # ===================== Added: Save cleaned data =====================
    df.to_csv(cleaned_file, index=False, encoding='utf-8-sig')
    print(f"✅ Cleaned data saved to: {cleaned_file}")

    # ===================== 3. Exploratory Data Analysis & Visualization =====================
    # 3.1 Numeric variable distribution: Age, Fare
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df['Age'], kde=True, ax=axes[0]).set_title('Age Distribution')
    sns.histplot(df['Fare'], kde=True, ax=axes[1]).set_title('Fare Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'fig1_age_fare_dist.png'), dpi=300, bbox_inches='tight')
    plt.close()
    # 3.2 Categorical variable proportion: Survival, Gender, Passenger class
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.countplot(x='Survived', data=df, ax=axes[0]).set_title('Survival Count')
    sns.countplot(x='Sex', data=df, ax=axes[1]).set_title('Gender Count')
    sns.countplot(x='Pclass', data=df, ax=axes[2]).set_title('Passenger Class Count')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'fig2_category_count.png'), dpi=300, bbox_inches='tight')
    plt.close()
    # 3.3 Gender vs Survival rate
    plt.figure(figsize=(6, 4))
    sns.barplot(x='Sex', y='Survived', data=df).set_title('Survival Rate by Gender')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'fig3_gender_survival.png'), dpi=300, bbox_inches='tight')
    plt.close()
    # 3.4 Passenger class vs Survival rate
    plt.figure(figsize=(6, 4))
    sns.barplot(x='Pclass', y='Survived', data=df).set_title('Survival Rate by Passenger Class')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'fig4_pclass_survival.png'), dpi=300, bbox_inches='tight')
    plt.close()
    # 3.5 Correlation heatmap of numeric variables
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    plt.figure(figsize=(8, 5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f').set_title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'fig5_correlation_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✅ All visualization charts saved to the report folder")

    # ===================== 4. Feature Engineering =====================
    # 4.1 One-hot encoding for categorical features
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    df['Pclass'] = df['Pclass'].astype(int)
    # 4.2 Standardization of numeric features
    scaler = StandardScaler()
    df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
    # 4.3 Split training and test sets
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("\n✅ Feature engineering completed, training and test sets split")

    # ===================== 5. Basic Modeling & Evaluation =====================
    # 5.1 Model training
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=50)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'report': classification_report(y_test, y_pred)
        }
    # 5.2 Output evaluation results
    print("\n" + "="*50)
    print("Test set accuracy of each model:")
    for name, res in results.items():
        print(f"{name:20s} Accuracy: {res['accuracy']:.4f}")
        print(res['report'])
        print('-'*50)
    # 5.3 Confusion matrix visualization for the best model
    best_model = models['Random Forest']
    y_pred_best = best_model.predict(X_test)
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred_best), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Random Forest)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'fig6_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("\n🎉 Full analysis pipeline completed!")
    print("📊 All charts exported to the report folder")
    print("📈 Model evaluation results printed above")
    print("📄 Cleaned dataset saved to data/train_cleaned.csv")

if __name__ == "__main__":
    main()

# ===================== Generate Kaggle Submission File =====================
def generate_submission():
    # Reload and preprocess training set (to obtain statistics)
    train_df = pd.read_csv(train_file)
    
    # Training set statistics (used for test set processing to avoid data leakage)
    age_median = train_df['Age'].median()
    embarked_mode = train_df['Embarked'].mode()[0]
    fare_q1 = train_df['Fare'].quantile(0.25)
    fare_q3 = train_df['Fare'].quantile(0.75)
    fare_iqr = fare_q3 - fare_q1
    fare_upper = fare_q3 + 1.5 * fare_iqr
    fare_lower = fare_q1 - 1.5 * fare_iqr
    
    # Full preprocessing pipeline for training set
    train_df['Age'] = train_df['Age'].fillna(age_median)
    train_df = train_df.drop('Cabin', axis=1)
    train_df['Embarked'] = train_df['Embarked'].fillna(embarked_mode)
    train_df['Fare'] = np.clip(train_df['Fare'], fare_lower, fare_upper)
    train_df = train_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
    train_df = pd.get_dummies(train_df, columns=['Sex', 'Embarked'], drop_first=True)
    
    X_train_final = train_df.drop('Survived', axis=1)
    y_train_final = train_df['Survived']
    scaler_final = StandardScaler()
    X_train_final[['Age', 'Fare']] = scaler_final.fit_transform(X_train_final[['Age', 'Fare']])
    
    # Train final model (using full training set)
    final_model = RandomForestClassifier(random_state=42, n_estimators=50)
    final_model.fit(X_train_final, y_train_final)
    
    # Load and preprocess test set
    test_file = os.path.join(data_dir, "test.csv")
    if not os.path.exists(test_file):
        print("❌ test.csv not found, please place the test set into the data folder")
        return
    
    test_df = pd.read_csv(test_file)
    passenger_ids = test_df['PassengerId']  # Save IDs for submission
    
    # Process test set using training set statistics
    test_df['Age'] = test_df['Age'].fillna(age_median)
    test_df = test_df.drop('Cabin', axis=1)
    test_df['Embarked'] = test_df['Embarked'].fillna(embarked_mode)
    test_df['Fare'] = test_df['Fare'].fillna(train_df['Fare'].median())  # 1 missing Fare value in test set
    test_df['Fare'] = np.clip(test_df['Fare'], fare_lower, fare_upper)
    test_df = test_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
    test_df = pd.get_dummies(test_df, columns=['Sex', 'Embarked'], drop_first=True)
    
    # Standardization (using scaler fitted on training set)
    test_df[['Age', 'Fare']] = scaler_final.transform(test_df[['Age', 'Fare']])
    
    # Make predictions
    predictions = final_model.predict(test_df)
    
    # Generate submission file
    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions
    })
    
    submission_path = os.path.join(data_dir, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    print(f"\n✅ Submission file generated: {submission_path}")
    print(f"Total {len(submission)} prediction records, meets Kaggle requirements")

# Execute generation
if __name__ == "__main__":
    generate_submission()
