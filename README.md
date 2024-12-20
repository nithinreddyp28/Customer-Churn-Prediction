# Customer Churn Prediction: Unlocking Insights into Customer Retention

## 🚀 Project Overview

In this project, we explore the world of **customer churn prediction** using **Python** and **machine learning** techniques. The goal is to predict whether a customer will churn based on various features such as their **demographics**, **account details**, and **behavioral data**. With the help of **machine learning models** like **Random Forest**, **Gradient Boosting**, and **Neural Networks**, we can gain valuable insights to help businesses reduce churn and improve customer retention.

Whether you're a data analyst, business strategist, or a data enthusiast looking to learn about customer behavior, this project offers practical solutions for predicting and understanding churn.

## 🔍 Business Problem

**Can we predict whether a customer will churn based on their historical data?**

The main goal is to predict **customer churn** for a telecom company. By analyzing customer features like **age**, **contract type**, **monthly charges**, and **total charges**, we aim to help businesses proactively identify customers who are at risk of leaving and take necessary actions to retain them.

## 🧠 Analysis Process

### 1. **Data Wrangling & Preprocessing**
   - **Loaded** the dataset using **pandas** for data manipulation and **PySpark** for large-scale processing.
   - **Cleaned** the dataset by addressing missing values, encoding categorical variables, and transforming numerical columns for consistency.
   - Balanced the data using **SMOTE (Synthetic Minority Over-sampling Technique)** to handle the class imbalance in churned vs. non-churned customers.
   - **Feature engineering** was done to extract additional insights, such as **tenure** (customer tenure), **contract type**, and **monthly charges**.

### 2. **Machine Learning Models**
   - We used multiple machine learning algorithms to predict churn, including:
     - **Random Forest**
     - **Gradient Boosting**
     - **Neural Networks (with Adam and LBFGS solvers)**
     - **K-Nearest Neighbors (KNN)**
     - **Support Vector Machine (SVM)** with various kernels (RBF, Polynomial)
   - Models were trained on a balanced dataset and evaluated using key metrics like **accuracy**, **precision**, **recall**, **F1-score**, and **AUC-ROC**.

### 3. **Model Evaluation**
   - **Accuracy** ranged from **71%** to **86%**, with **Random Forest** and **Gradient Boosting** providing the best results.
   - The **Neural Network** model performed well with **Adam solver**, achieving an **accuracy of 83.65%** and an **AUC-ROC of 0.84**.
   - **SVM (RBF kernel)** showed promising results, while **KNN** and **Decision Tree** were slightly less effective in predicting churn.

## 🛠️ Tools & Technologies

- **PySpark**: For scalable data processing and machine learning.
- **Python Libraries**:
  - **pandas**: For data manipulation and analysis.
  - **NumPy**: For numerical operations.
  - **matplotlib**, **seaborn**: For data visualization.
  - **sklearn**: For machine learning algorithms like Random Forest, SVM, and KNN.
  - **imblearn**: For implementing SMOTE to balance the dataset.
  - **tensorflow**: For building and training neural network models.

## 🎯 Key Insights & Results

1. **Top Predictors**: Features such as **Monthly Charges**, **Contract Type**, and **Tenure** were found to be the strongest predictors of churn.
2. **High Accuracy Models**: The **Random Forest** and **Gradient Boosting** models achieved the highest accuracy, outperforming others in predicting customer churn.
3. **Churn and Engagement**: We discovered that customers on month-to-month contracts with high monthly charges were at a higher risk of churn.
4. **Neural Network Success**: The **Adam solver** for the Neural Network model provided the best results, with an accuracy of **83.65%** and an AUC-ROC of **0.84**.


