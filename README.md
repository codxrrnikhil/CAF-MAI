# 🧠 Model Training Layer (Decision Tree)

## 📌 Overview

This module is responsible for:

* Data preprocessing
* Model training (Decision Tree)
* Prediction generation
* Evaluation metrics computation
* Providing a unified interface for downstream modules

It integrates seamlessly with upstream modules like **Data Contract** and **Bias Detection**.

---

## ⚙️ Features

* ✅ Handles missing values (median/mode strategy)
* ✅ Encodes categorical variables
* ✅ Aligns train-test feature columns
* ✅ Trains Decision Tree model
* ✅ Generates predictions & probabilities
* ✅ Computes evaluation metrics:

  * Accuracy
  * Precision
  * Recall
  * F1-score
* ✅ Provides a unified `run_training()` pipeline

---

## 📁 Module Structure

```
module4/
├── __init__.py
├── train.py        # Training + integration logic
├── predict.py      # Prediction logic
└── utils.py        # Preprocessing utilities
```

---

## 🔄 Workflow

```
DataContract
   ↓
Preprocessing (utils.py)
   ↓
Train Model (train.py)
   ↓
Generate Predictions (predict.py)
   ↓
Compute Metrics
   ↓
Return Structured Output
```

---

## 🚀 Usage

### 1. Import

```python
from module4.train import run_training
```

---

### 2. Prepare DataContract

```python
data_contract = {
    "X": X_dataframe,
    "Y": y_series,
    "metadata": {}
}
```

---

### 3. Run Training

```python
result = run_training(data_contract)
```

---

### 4. Output

```python
{
  "model": trained_model,
  "X_test": X_test,
  "y_test": y_test,
  "predictions": [...],
  "probabilities": [...],
  "metrics": {
      "accuracy": ...,
      "precision": ...,
      "recall": ...,
      "f1_score": ...
  }
}
```

---

## 📊 Model Details

* Algorithm: `DecisionTreeClassifier`
* Parameters:

  * `max_depth=5`
  * `min_samples_split=10`
  * `min_samples_leaf=5`
  * `random_state=42`

---

## 🧹 Preprocessing Details

### Missing Values

* Numerical → median
* Categorical → mode

### Encoding

* One-hot encoding (`pd.get_dummies`)
* `drop_first=True` to avoid multicollinearity

### Column Alignment

* Ensures test data matches training features

---

## ⚠️ Assumptions

* Input data is tabular (DataFrame)
* Target variable is correctly specified
* DataContract structure is valid

---

## 🔌 Integration

This module is designed to:

* Plug into FastAPI backend
* Feed predictions to bias detection modules
* Work as part of a larger AI fairness system

---

## 🧪 Testing

Use a CSV dataset (e.g., Titanic dataset) to validate:

```bash
python test_module4.py
```

---

## 📦 Dependencies

* pandas
* numpy
* scikit-learn

---

## 📈 Future Improvements

* Model persistence (save/load)
* Hyperparameter tuning
* Support for multiple models (XGBoost, RandomForest)
* Advanced preprocessing (scaling, pipelines)
* Bias-aware training

---

## 👨‍💻 Author

Developed as part of the **UnbiasAI Healthcare Project**
Focused on building fair and explainable AI systems.

---

## 📄 License

For academic and research use.
