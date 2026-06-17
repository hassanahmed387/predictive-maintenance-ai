# predictive-maintenance-ai
ML pipeline to predict industrial machine failures from sensor data — Random Forest 99% accuracy, ROC-AUC 0.978, Wilcoxon validated
# Predictive Maintenance AI — Industrial Machine Fault Detection

Most factories discover machine failures after they happen.
This model catches them before.

Built a complete end-to-end ML pipeline on the AI4I 2020 
Industrial Dataset to predict machine failures from live 
sensor readings — before they cause damage.

---

## Problem

Only 3.4% of machines actually fail — making this a classic 
imbalanced classification problem. A naive model that always 
predicts "no failure" gets 96% accuracy and catches nothing.

This pipeline handles it properly — no synthetic data, 
no SMOTE. Just class-weighted training that keeps the 
data honest and results meaningful.

---

## Dataset

AI4I 2020 Predictive Maintenance Dataset
- 10,000 records
- Features: Air Temperature, Process Temperature, 
  Rotational Speed, Torque, Tool Wear
- Target: Machine Failure (binary)
- Failure rate: 3.4%

---

## Features Engineered

3 physics-based features derived from domain knowledge:

| Feature | Formula | Why |
|---|---|---|
| Power [W] | Torque × Angular Velocity | Mechanical stress indicator |
| Temp Diff [K] | Process Temp - Air Temp | Heat stress on machine |
| Wear Rate | Tool Wear / RPM | Relative degradation speed |

All 3 showed clear separation between healthy and failing machines.

---

## ML Pipeline

- Exploratory Data Analysis (EDA)
- IQR-based outlier analysis (kept — real sensor readings)
- Feature engineering (3 derived features)
- Label encoding for categorical variable
- 70/15/15 Train/Validation/Test split (stratified)
- StandardScaler (fit on train only — no data leakage)
- Class imbalance handled via class_weight='balanced'

---

## Models Trained & Compared

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.87 | 0.19 | 0.88 | 0.31 | 0.933 |
| Random Forest | 0.99 | 0.93 | 0.73 | 0.81 | 0.978 |
| SVM | 0.94 | 0.33 | 0.86 | 0.48 | 0.971 |

Winner: Random Forest
Validated with Wilcoxon Signed-Rank Test (p < 0.05)

---

## Real World Impact

| Event | Cost |
|---|---|
| False Alarm (unnecessary inspection) | ~$500 |
| Missed Failure (unplanned downtime) | ~$10,000+ |

Recall was optimized as the primary metric — 
catching failures matters more than avoiding false alarms.

---

## Stack

Python | Scikit-learn | Pandas | NumPy | 
Matplotlib | Seaborn | SciPy

---

## Results

![Model Comparison](results/fig4_feature_importance_model_comparison.png)
![ROC Curves](results/fig6_roc_curves.png)
![Confusion Matrices](results/fig5_confusion_matrices.png)
