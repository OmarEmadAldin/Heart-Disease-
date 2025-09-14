# 🫀 Heart Disease Prediction Project

This project is a **machine learning pipeline** for analyzing and predicting **heart disease** based on clinical data.  
It follows a structured workflow, starting from **data preprocessing** all the way to **model training, evaluation, clustering, and hyperparameter tuning**.

---

## 📂 Project Structure

```
Heart_Disease_Project/
│
├── Data/
│   └── heart_disease_reduced_features.csv   # Processed dataset used for modeling
│
├── Notebooks/
│   ├── 01_data_preprocessing.ipynb          # Cleaning, handling missing values, encoding
│   ├── 02_pca_analysis.ipynb                 # Dimensionality reduction via PCA
│   ├── 03_feature_selection.ipynb            # Select most relevant features
│   ├── 04_supervised_learning.ipynb          # Train/test ML models (classification)
│   ├── 05_unsupervised_learning.ipynb        # K-Means & Hierarchical clustering
│   └── 06_hyperparameter_tuning.ipynb        # GridSearchCV for best model selection
│
├── Results/
│   └── evaluation_metrics.txt               # Saved metrics from hypertuning
│
├── Models/
│   └── best_model.pkl                       # Best-performing model saved with joblib
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

1. Clone or download the repository.
2. Create a Python virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Workflow (Notebook Summaries)

### 1️⃣ Data Preprocessing (`01_data_preprocessing.ipynb`)
- Handles missing values and categorical encoding.
- Scales features to ensure fair comparison between variables.
- Produces a **clean dataset** for downstream analysis.

### 2️⃣ PCA Analysis (`02_pca_analysis.ipynb`)
- Applies **Principal Component Analysis (PCA)**.
- Justification:
  - Reduces dimensionality → less noise.
  - Keeps maximum variance with fewer components.
  - Speeds up training.

### 3️⃣ Feature Selection (`03_feature_selection.ipynb`)
- Identifies **most predictive features** using selection methods.
- Justification:
  - Removes irrelevant variables.
  - Improves model accuracy & interpretability.

### 4️⃣ Supervised Learning (`04_supervised_learning.ipynb`)
- Trains classification models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
- Evaluates using:
  - Accuracy, Precision, Recall, F1-score, ROC-AUC
- Justification:
  - Compares different algorithm families (linear, tree-based, ensemble, kernel).

### 5️⃣ Unsupervised Learning (`05_unsupervised_learning.ipynb`)
- K-Means clustering with **Elbow Method** for optimal `K`.
- Hierarchical Clustering with dendrogram visualization.
- Evaluates with **Adjusted Rand Index (ARI)** against true labels.
- Justification:
  - Provides insight into natural groupings in the data.
  - Validates label quality.

### 6️⃣ Hyperparameter Tuning (`06_hyperparameter_tuning.ipynb`)
- Uses **GridSearchCV** to optimize models.
- Saves:
  - Best hyperparameters.
  - Final trained best model (`best_model.pkl`).
  - Evaluation metrics (`evaluation_metrics.txt`).
- Justification:
  - Fine-tunes models for maximum generalization.
  - Ensures reproducibility with saved artifacts.

---

## 📊 Outputs
- **Evaluation Metrics**: Precision, Recall, F1-score, ROC-AUC for each model.  
- **Best Model**: Stored in `Models/best_model.pkl` (loadable with `joblib.load`).  
- **Visualization**: ROC curves, PCA explained variance plots, clustering dendrograms, and elbow plots.

---

## 💡 Usage Example

To load the best model and make predictions:

```python
import joblib
import pandas as pd

# Load saved model
model = joblib.load("Models/best_model.pkl")

# Load new data (must match preprocessing pipeline)
new_data = pd.DataFrame([[63, 1, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1, 1]],
                        columns=["age", "sex", "trestbps", "chol", "fbs", 
                                 "restecg", "thalach", "exang", "oldpeak",
                                 "slope", "ca", "thal", "cp"])

# Predict
prediction = model.predict(new_data)
print("Heart Disease Risk:", prediction)
```

---

## 🛠️ Justification of Methodology

- **Preprocessing** ensures data quality.  
- **PCA + Feature Selection** reduce dimensionality and focus on informative features.  
- **Supervised Learning** evaluates multiple ML models to find the best predictor.  
- **Unsupervised Learning** checks clustering tendencies and validates labels.  
- **Hyperparameter Tuning** optimizes models and ensures reproducibility.  

Together, these steps build a **robust and interpretable ML pipeline** for heart disease prediction.

---

## 👨‍💻 Author
Developed by **Omar 3amora**  
Project focus: **Perception & predictive modeling in healthcare data**.
