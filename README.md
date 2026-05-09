# Heart Disease Prediction using Logistic Regression and Feature Scaling

## 📌 Project Overview

This project predicts whether an employee is likely to leave the company or stay based on different employee-related factors.

The project uses:

- Logistic Regression
- L1 Regularization (Lasso)
- L2 Regularization (Ridge)

to compare model performance and understand the impact of regularization techniques in Machine Learning.

---

## 📊 Problem Statement

Employee turnover is a major challenge for organizations. Predicting employee attrition helps companies:

- Improve employee retention
- Reduce hiring costs
- Increase workforce stability
- Make better HR decisions

This project builds classification models to predict employee turnover.

---

## 🛠 Technologies Used

- Python
- Pandas
- Scikit-learn
- NumPy
- Jupyter Notebook

---

## 📂 Dataset

The dataset contains employee-related information such as:

- Age
- Salary
- Experience
- Department
- Performance
- Working Hours
- Employee Turnover (Target Variable)

Target Column:

```python
Employee_Turnover
```

- 0 → Employee Stays
- 1 → Employee Leaves

---

## ⚙️ Project Workflow

### 1. Data Loading

```python
df = pd.read_csv("employee_turnover.csv")
```

### 2. Feature and Target Selection

```python
X = df.drop('Employee_Turnover', axis=1)
y = df['Employee_Turnover']
```

### 3. Train-Test Split

```python
train_test_split(X, y, test_size=0.2, random_state=42)
```

### 4. Model Building

#### Baseline Logistic Regression

```python
LogisticRegression()
```

#### L1 Regularization (Lasso)

```python
LogisticRegression(penalty='l1', solver='liblinear')
```

#### L2 Regularization (Ridge)

```python
LogisticRegression(penalty='l2')
```

### 5. Model Evaluation

Models are evaluated using:

- Accuracy Score
- Precision
- Recall
- F1-Score

---

## 📈 Understanding Regularization

### 🔹 L1 Regularization (Lasso)

- Adds penalty using absolute coefficient values
- Can reduce some coefficients to zero
- Performs feature selection automatically

Formula:

```text
Loss + λ Σ|w|
```

---

### 🔹 L2 Regularization (Ridge)

- Adds penalty using squared coefficient values
- Shrinks coefficients but usually does not make them zero
- Helps reduce overfitting

Formula:

```text
Loss + λ Σw²
```

---

## 📌 Key Learning Outcomes

- Understanding Logistic Regression
- Difference between L1 and L2 Regularization
- Preventing Overfitting
- Feature Selection using Lasso
- Classification Model Evaluation

---

## 🚀 Future Improvements

- Hyperparameter Tuning
- Cross Validation
- Feature Engineering
- Deployment using Flask or Streamlit
- Advanced ML Models

---

## 👨‍💻 Author

Shanu

Machine Learning & Data Analytics Enthusiast
