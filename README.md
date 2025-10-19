# Wine Quality Prediction — ML vs DL Comparison

Author: Renny Pascal Ihirwe

Repository: https://github.com/leny62/Summative-Assignment---Model-Training-and-Evaluation.git

Demo: https://drive.google.com/file/d/1ulNVW5szCyeDlHAY7DHGx2-FUrEI1MqJ/view?usp=sharing

Report Link(Enhanced): https://docs.google.com/document/d/1BKojFqvMAYzn47SSoUutuhtheA28JbRD0gZ2Q0YQ9Ck/edit?usp=sharing
---

## Project Overview

This project compares traditional machine learning (ML) and deep learning (DL) approaches for predicting the quality of red wines using the UCI / Kaggle Red Wine dataset (1,599 samples, 11 physicochemical features). The task is framed as a binary classification problem: "Good" wines (quality ≥ 7) vs "Not Good" wines (quality < 7). The principal objective is to identify which modeling approach gives the best balance of predictive performance, robustness to class imbalance, interpretability, and computational cost.

Key points:

- Dataset: 1,599 samples, 11 features (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol)
- Task: Binary classification (Good vs Not Good)
- Approaches: Traditional ML (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM) and Deep Learning (Sequential NN, Functional NN)
- Key challenge: Class imbalance (~86.4% Not Good : ~13.6% Good)

## Dataset

The experiments use the Red Wine Quality dataset (Cortez et al., 2009). You can obtain the data from either source below:

- Kaggle (mirror): https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009
- UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/wine+quality

Download note: the notebook expects a CSV named `winequality-red.csv` in the working directory or you can update `DATA_PATH` in the notebook to point to your local copy.

---

## Repository Structure (recommended)

- `Summative_Assignment_Model_Training_and_Evaluation_.ipynb` — Primary notebook with data loading, EDA, modeling pipelines, training, and evaluation
- `Wine_Quality_Scholarly_Report.md` — Submission-ready report (draft) summarising experiments, results, and recommendations
- `README.md` — This overview and run instructions
- `saved_models/` — Directory to hold saved model artifacts (joblib and Keras models)
- `roc_curves.png`, `learning_curves.png`, `confusion_matrix_best.png`, `feature_importance.png` — Suggested locations for exported figures
- `saved_models/all_models/` — contains all trained model artifacts (baseline and deep models)
	- Decision_Tree.joblib
	- Functional_NN.keras
	- Gradient_Boosting.joblib
	- Logistic_Regression.joblib
	- Random_Forest.joblib
	- Sequential_NN.keras
	- SVM.joblib

- `saved_models/best_models/` — contains selected best models
	- Random_Forest_Tuned.joblib
	- Sequential_NN.keras

---

## Quick Results Summary

The experiments show that ensemble tree-based methods outperform the tested deep learning architectures on this dataset:

| Model | Test AUC |
|---|---:|
| Random Forest (baseline) | 0.9497 |
| Random Forest (tuned) | 0.9556 |
| Gradient Boosting | 0.8979 |
| Sequential NN (class-weighted) | 0.8734 |

Best recommendation for deployment: Random Forest (tuned) — highest AUC, fast training and inference, and feature-importance interpretability.

Handling class imbalance: experiments compared SMOTE, ADASYN, class weighting, and threshold tuning. Threshold tuning and class weighting provided the best balance for precision and recall in production-like scenarios; SMOTE/ADASYN increased recall at the cost of precision.

---

## How to Reproduce (Windows PowerShell)

1) Create a virtual environment (recommended)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
```

2) Install dependencies

```powershell
pip install -r requirements.txt
```

If `requirements.txt` is not present, the main packages used are:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- tensorflow
- xgboost
- joblib
- tabulate

3) Run the notebook

Open `Summative_Assignment_Model_Training_and_Evaluation_.ipynb` in Jupyter or VS Code and run cells in order. Alternatively, execute the main pipeline cells that load data, train models, and export figures. The notebook contains code blocks to save models under `saved_models/` and to create `saved_models.zip`.

## How to Use the Saved Models

Saved traditional ML models are stored as `.joblib` files and Keras models as `.keras` or folders. Example: `saved_models/all_models/Random_Forest.joblib` and `saved_models/best_models/Random_Forest_Tuned.joblib`.

To load a joblib model:

```python
import joblib
model = joblib.load(r'saved_models\best_models\Random_Forest_Tuned.joblib')
proba = model.predict_proba(X_new)[:, 1]
```

To load a Keras model:

```python
from tensorflow import keras
model = keras.models.load_model(r'saved_models\all_models\Sequential_NN.keras')
proba = model.predict(X_new_scaled).ravel()
```

---

