
# 🩺 Diabetes Prediction with Random Forest

This project uses a Random Forest Classifier trained on the Pima Indians Diabetes dataset to predict whether a patient is likely to have diabetes based on various medical features.

## 🔧 Requirements

- Python 3.x
- pandas
- scikit-learn
- joblib
- Google Colab or Jupyter Notebook

## 📊 Dataset

The dataset is sourced from:
https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv

**Columns:**
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (Target variable)

## 🧠 Model Training

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

# Load data
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
           'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)

# Split data
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model
joblib.dump(model, 'diabetes_model.pkl')
```

## 📤 User CSV Prediction

1. Upload a CSV file containing these columns:
   ```
   Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
   BMI, DiabetesPedigreeFunction, Age
   ```

2. Use the following code to load and predict:

```python
from google.colab import files
import pandas as pd
import joblib

# Upload CSV
uploaded = files.upload()

# Load model
model = joblib.load('diabetes_model.pkl')

# Load data and predict
for filename in uploaded.keys():
    user_df = pd.read_csv(filename)
    predictions = model.predict(user_df)
    user_df['DiabetesPrediction'] = ['Likely Diabetic' if p == 1 else 'Unlikely Diabetic' for p in predictions]
    print(user_df[['DiabetesPrediction']])
    user_df.to_csv("prediction_results.csv", index=False)
    files.download("prediction_results.csv")
```

## 📦 Output

- `diabetes_model.pkl`: Trained model
- `prediction_results.csv`: Results with predictions added to user-uploaded data

## ✅ Example

A sample `input.csv` should look like this:

```csv
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
6,148,72,35,0,33.6,0.627,50
1,85,66,29,0,26.6,0.351,31
```
