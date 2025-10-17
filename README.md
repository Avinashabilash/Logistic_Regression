# 👥 Employee Attrition Prediction

## 🧠 Description
This project predicts whether an employee will **leave the company** using **Logistic Regression**, based on features like **satisfaction level, average monthly hours, promotion history, and salary**.  
The dataset is analyzed to find patterns in employee behavior, and a machine learning model is trained to predict attrition.

---

## 🎯 AIM
To build a **machine learning model** that can predict **employee turnover** and help HR teams make informed decisions.

---

## 🧩 Goals
- Predict whether an employee will **leave** or **stay** in the company.  
- Analyze patterns in employee satisfaction, workload, promotions, and salary.  
- Evaluate model performance using **accuracy, confusion matrix, and classification report**.  
- Visualize relationships between key features and attrition.  

---

## ⚙️ Technical Details
**Language & Libraries:**  
- Python  
- pandas  
- matplotlib  
- scikit-learn  

---

## 🧾 Data Preprocessing
- Load dataset using `pandas`.  
- Analyze data using **groupby** and **crosstab**.  
- Select key features:  
  `satisfaction_level`, `average_montly_hours`, `promotion_last_5years`, `salary`.  
- Encode categorical variables (`salary`) using **one-hot encoding**.

---

## 🧮 Model Used
**Logistic Regression**  
- Predicts **binary outcome**: 0 = retained, 1 = left.  
- Model Formula:  
  \[
  P(y=1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \dots + \beta_n x_n)}}
  \]

---

 

## 💻 Code Implementation

```python
# Import libraries
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline

# Load dataset
df = pd.read_csv(r"C:\Users\abiav\Classficiation  Logistic Regression\HR_comma_sep.csv")
print(df.head())
```
📊 Results

Model Accuracy: 0.785 (≈78.5%)

Confusion Matrix:

[[7566  419]
 [1839  676]]

| Class            | Precision | Recall | F1-score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| 0 (Retained)     | 0.80      | 0.95   | 0.87     | 7985    |
| 1 (Left)         | 0.62      | 0.27   | 0.37     | 2515    |
| **Accuracy**     |           |        | 0.78     | 10500   |
| **Macro Avg**    | 0.71      | 0.61   | 0.62     | 10500   |
| **Weighted Avg** | 0.76      | 0.78   | 0.75     | 10500   |

📋 Observation

Employees with low satisfaction and longer time in company are more likely to leave.

Promotion history and salary also influence attrition.

Model performs well for retained employees, but recall is lower for employees who leave.

Useful for HR teams to identify at-risk employees and improve retention strategies.
