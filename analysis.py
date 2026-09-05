# ===================== 导入基础库 =====================
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

# ===================== 路径与目录自动配置 =====================
# 获取当前脚本所在目录，彻底摆脱运行时工作目录依赖
current_dir = os.path.dirname(os.path.abspath(__file__))

# 定义各目录路径
data_dir = os.path.join(current_dir, "data")
report_dir = os.path.join(current_dir, "report")

# 自动创建目录（已存在也不会报错）
os.makedirs(data_dir, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)

# 数据集文件完整路径
train_file = os.path.join(data_dir, "train.csv")
cleaned_file = os.path.join(data_dir, "train_cleaned.csv")

# ===================== 绘图中文配置 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows显示中文
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac用户取消注释这行
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常

# ===================== 主程序入口 =====================
def main():
    # ===================== 1. 数据加载与初步探查 =====================
    try:
        df = pd.read_csv(train_file)
        print("✅ 原始数据加载成功")
    except FileNotFoundError:
        print(f"❌ 找不到数据集：{train_file}")
        print("请将 train.csv 放入 data 文件夹后重新运行")
        return

    print("数据维度：", df.shape)
    print("\n前5行数据：")
    print(df.head())
    print("\n字段信息：")
    df.info()
    print("\n数值字段统计概览：")
    print(df.describe())

    # ===================== 2. 数据清洗 =====================
    # 2.1 删除重复值
    print(f"\n重复样本数：{df.duplicated().sum()}")
    df = df.drop_duplicates()

    # 2.2 缺失值处理
    df['Age'] = df['Age'].fillna(df['Age'].median())  # 年龄用中位数填充
    df = df.drop('Cabin', axis=1)  # 客舱号缺失率过高，删除
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # 登船港口用众数填充

    # 2.3 异常值处理（票价盖帽法）
    Q1 = df['Fare'].quantile(0.25)
    Q3 = df['Fare'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df['Fare'] = np.where(df['Fare'] > upper, upper, df['Fare'])
    df['Fare'] = np.where(df['Fare'] < lower, lower, df['Fare'])

    # 2.4 数据类型转换
    df['Survived'] = df['Survived'].astype('category')
    df['Pclass'] = df['Pclass'].astype('category')
    df['Sex'] = df['Sex'].astype('category')

    # 2.5 剔除无关字段
    df = df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
    print("\n✅ 数据清洗完成")

    # ===================== 新增：保存清洗后的数据 =====================
    df.to_csv(cleaned_file, index=False, encoding='utf-8-sig')
    print(f"✅ 清洗后数据已保存到：{cleaned_file}")

    # ===================== 3. 探索性数据分析与可视化 =====================
    # 3.1 数值变量分布：年龄、票价
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df['Age'], kde=True, ax=axes[0]).set_title('Age Distribution')
    sns.histplot(df['Fare'], kde=True, ax=axes[1]).set_title('Fare Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'fig1_age_fare_dist.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3.2 分类变量占比：存活情况、性别、客舱等级
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.countplot(x='Survived', data=df, ax=axes[0]).set_title('Survival Count')
    sns.countplot(x='Sex', data=df, ax=axes[1]).set_title('Gender Count')
    sns.countplot(x='Pclass', data=df, ax=axes[2]).set_title('Passenger Class Count')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'fig2_category_count.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3.3 性别 vs 存活率
    plt.figure(figsize=(6, 4))
    sns.barplot(x='Sex', y='Survived', data=df).set_title('Survival Rate by Gender')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'fig3_gender_survival.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3.4 客舱等级 vs 存活率
    plt.figure(figsize=(6, 4))
    sns.barplot(x='Pclass', y='Survived', data=df).set_title('Survival Rate by Passenger Class')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'fig4_pclass_survival.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3.5 数值变量相关性热力图
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    plt.figure(figsize=(8, 5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f').set_title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'fig5_correlation_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("\n✅ 所有可视化图表已保存到 report 文件夹")

    # ===================== 4. 特征工程 =====================
    # 4.1 分类特征独热编码
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    df['Pclass'] = df['Pclass'].astype(int)

    # 4.2 数值特征标准化
    scaler = StandardScaler()
    df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

    # 4.3 划分训练集与测试集
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("\n✅ 特征工程完成，训练集测试集已拆分")

    # ===================== 5. 基础建模与评估 =====================
    # 5.1 模型训练
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

    # 5.2 输出评估结果
    print("\n" + "="*50)
    print("各模型测试集准确率：")
    for name, res in results.items():
        print(f"{name:20s} 准确率：{res['accuracy']:.4f}")
        print(res['report'])
        print('-'*50)

    # 5.3 最优模型混淆矩阵可视化
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

    print("\n🎉 全部分析流程运行完成！")
    print("📊 所有图表已输出到 report 文件夹")
    print("📈 模型评估结果已在上方打印")
    print("📄 清洗后数据集已保存到 data/train_cleaned.csv")

if __name__ == "__main__":
    main()
# ===================== 生成Kaggle提交文件 =====================
def generate_submission():
    # 重新加载并预处理训练集（获取统计量）
    train_df = pd.read_csv(train_file)
    
    # 训练集统计量（用于测试集处理，避免数据泄露）
    age_median = train_df['Age'].median()
    embarked_mode = train_df['Embarked'].mode()[0]
    fare_q1 = train_df['Fare'].quantile(0.25)
    fare_q3 = train_df['Fare'].quantile(0.75)
    fare_iqr = fare_q3 - fare_q1
    fare_upper = fare_q3 + 1.5 * fare_iqr
    fare_lower = fare_q1 - 1.5 * fare_iqr
    
    # 训练集完整预处理流程
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
    
    # 训练最终模型（用全部训练集）
    final_model = RandomForestClassifier(random_state=42, n_estimators=50)
    final_model.fit(X_train_final, y_train_final)
    
    # 加载并预处理测试集
    test_file = os.path.join(data_dir, "test.csv")
    if not os.path.exists(test_file):
        print("❌ 找不到 test.csv，请将测试集放入 data 文件夹")
        return
    
    test_df = pd.read_csv(test_file)
    passenger_ids = test_df['PassengerId']  # 保存ID用于提交
    
    # 用训练集的统计量处理测试集
    test_df['Age'] = test_df['Age'].fillna(age_median)
    test_df = test_df.drop('Cabin', axis=1)
    test_df['Embarked'] = test_df['Embarked'].fillna(embarked_mode)
    test_df['Fare'] = test_df['Fare'].fillna(train_df['Fare'].median())  # 测试集有1个票价缺失
    test_df['Fare'] = np.clip(test_df['Fare'], fare_lower, fare_upper)
    test_df = test_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
    test_df = pd.get_dummies(test_df, columns=['Sex', 'Embarked'], drop_first=True)
    
    # 标准化（用训练集的scaler）
    test_df[['Age', 'Fare']] = scaler_final.transform(test_df[['Age', 'Fare']])
    
    # 预测
    predictions = final_model.predict(test_df)
    
    # 生成提交文件
    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions
    })
    
    submission_path = os.path.join(data_dir, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    print(f"\n✅ 提交文件已生成：{submission_path}")
    print(f"共 {len(submission)} 条预测记录，符合Kaggle要求")

# 执行生成
if __name__ == "__main__":
    generate_submission()
