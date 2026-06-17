


#  Predictive Maintenance for Industrial Machines
#  CS-333: Applied AI & Machine Learning - Lab Project
# 
# **Dataset:** AI4I 2020 Predictive Maintenance Dataset  
# **Goal:** Predict whether a machine will fail based on sensor readings and process parameters.
# 
# In this notebook we'll go through the full ML pipeline step by step:
# 1. Load and explore the data
# 2. Clean and preprocess
# 3. Do some feature engineering
# 4. Train 3 different models (from different families)
# 5. Compare them properly with metrics + statistical test
# 6. Pick the best model and explain why


print("Final Project AI&ML")

# ## 1. Import Libraries
 




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
#seaborn lib for better graph create 

# for reproducibility
SEED = 42
np.random.seed(SEED)

# sklearn things is python library used for ML and data analysis    used for data set training
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             roc_curve)

# stats test
from scipy.stats import wilcoxon
#scipy is python lib used for  sceintific computing , math, statistics, signal processing
#stats us module inside scipy that contain statistical tests , prob distrb and stats calc
#wilcoxon is statistical test function (used to compare two sample values before and after)

# plot settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("All libraries loaded successfully!")

#Next step is >>> 2. Load the Dataset
 
# i will be using using the AI4I 2020 Predictive Maintenance Dataset. On Kaggle the file is usually called `ai4i2020.csv`.
 

import pandas as pd

df = pd.read_csv(r"C:\Users\Hassan Ahmed\Desktop\data sc\AI&ML Project\ai4i2020.csv")

print(df.head())
print("Shape of dataset:", df.shape)
df.head()

# quick look at columns
df.info()

# check for any missing values
print("Missing values per column:")
print(df.isnull().sum())

# - **UDI**: Unique identifier 
# - **Product ID**: serial like L47181, M14860 etc (
# - **Type**: Quality of product - L (low), M (medium), H (high) -  **categorical** feature
# - **Air temperature [K]**: ambient air temp
# - **Process temperature [K]**: process temp inside the machine
# - **Rotational speed [rpm]**: how fast the tool spins
# - **Torque [Nm]**: torque applied
# - **Tool wear [min]**: how many minutes the tool has been in use
# - **Machine failure**: 1 if machine failed, 0 otherwise → **TARGET**
# - TWF, HDF, PWF, OSF, RNF: specific failure modes 


# Now next step is >>>> 3.  Data Analysis (DA)


# descriptive statistics which is tool for generating descriptive statistics that summerize central tendency,dispersion,and shape of data set's distribution
df.describe()

#  numeric cols
numeric_cols = ['Air temperature [K]', 'Process temperature [K]', 
                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

#skewness tells abut symmetric distrb
#kurtosis about heavy or low outliers
#std >> standard deviation

stats_df = pd.DataFrame({
    'mean': df[numeric_cols].mean(),
    'std': df[numeric_cols].std(),
    'skewness': df[numeric_cols].skew(),
    'kurtosis': df[numeric_cols].kurtosis()
})
stats_df.round(3)

# check the target distribution - is it balanced?
print("Machine failure distribution:")
print(df['Machine failure'].value_counts())
print("\nFailure rate:", round(df['Machine failure'].mean()*100, 2), "%")

# plot it all which was calc
plt.figure(figsize=(6, 4))
df['Machine failure'].value_counts().plot(kind='bar', color=['steelblue', 'salmon'])
plt.title('Class Distribution - Machine Failure')
plt.xlabel('Failure (0=No, 1=Yes)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

#when we will see graph we will come to know that
# This is a heavily **imbalanced** dataset. Only ~3.4% of machines fail. 


# product type distribution
print(df['Type'].value_counts())

plt.figure(figsize=(6, 4))
df['Type'].value_counts().plot(kind='bar', color=['lightgreen', 'gold', 'tomato'])
plt.title('Product Quality Type Distribution')
plt.xlabel('Type')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()


# histograms for all numeric features
df[numeric_cols].hist(figsize=(14, 8), bins=30, color='steelblue', edgecolor='black')
plt.suptitle('Distribution of Numeric Features', fontsize=14)
plt.tight_layout()
plt.show()


# boxplots to spot outliers
fig, axes = plt.subplots(1, 5, figsize=(18, 5))
for i, col in enumerate(numeric_cols):
    sns.boxplot(y=df[col], ax=axes[i], color='lightblue')
    axes[i].set_title(col, fontsize=10)
plt.tight_layout() #used to prevent overlapping
plt.show()


# Boxplots show that Rotational Speed and Torque have some outliers. But these are real machine readings (not data entry errors), so we'll keep them - they likely contain useful information about failures.


# correlation heatmap
plt.figure(figsize=(10, 7))
corr = df[numeric_cols + ['Machine failure']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()


# Some important  things from the heatmap:
# - Air temp and Process temp are strongly correlated (0.88) - makes sense physically
# - Rotational speed and Torque are negatively correlated (-0.88) - this is also a known relationship in motors (higher torque needs lower speed)
# - Tool wear has a mild positive correlation with failure - older tools fail more


# pairplot to see relationships - using a sample to make it faster
sample = df.sample(2000, random_state=SEED)
sns.pairplot(sample[numeric_cols + ['Machine failure']], hue='Machine failure', 
             diag_kind='hist', plot_kws={'alpha': 0.5, 's': 15})
plt.suptitle('Pairplot of Features (colored by failure)', y=1.02)
plt.show()


# Now, next  4. Data Preprocessing

# Now we will prepare the data for modeling.


# drop columns we don't need
# UDI and Product ID are just identifiers
# TWF, HDF, PWF, OSF, RNF are specific failure modes - using them would be data leakage
df_clean = df.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])

print("Columns we'll use:")
print(df_clean.columns.tolist())

# ### 4.1 Outlier Analysis (IQR method)

# Let's check outliers properly using IQR but as discussed, we won't remove them. bz they are real life things


# IQR-based outlier check
def count_outliers(col):
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df_clean[(df_clean[col] < lower) | (df_clean[col] > upper)]
    return len(outliers)

for col in numeric_cols:
    n = count_outliers(col)
    pct = round(n/len(df_clean)*100, 2)
    print(f"{col}: {n} outliers ({pct}%)")

# **Decision:** We are keeping outliers because:
# 1. They are real sensor readings, not errors
# 2. Failures often happen at extreme operating conditions - removing them could hurt model performance
# 3. Random Forest and SVM (with proper scaling) can handle them


# ### 4.2 Feature Engineering

# The project requires us to create at least 2 new derived features with physical/domain justification. Let me create a few:
 
# 1. **Power [W]** = Torque × Angular velocity → Mechanical power output of the machine. This is the most important derived quantity in rotating machinery.
 
# 2. **Temperature difference [K]** = Process temp - Air temp → Heat generated by the process. Bigger difference often means stress on the machine.

# 3. **Tool wear rate** = Tool wear / Rotational speed → How quickly the tool is wearing relative to its operating speed.


# Feature 1: Mechanical Power (Watts)
# Power = Torque (Nm) * Angular velocity (rad/s)
# Angular velocity = 2*pi*RPM/60
df_clean['Power [W]'] = df_clean['Torque [Nm]'] * (2 * np.pi * df_clean['Rotational speed [rpm]'] / 60)

# Feature 2: Temperature difference (process generates heat above ambient)
df_clean['Temp diff [K]'] = df_clean['Process temperature [K]'] - df_clean['Air temperature [K]']

# Feature 3: Wear-to-speed ratio
df_clean['Wear rate'] = df_clean['Tool wear [min]'] / df_clean['Rotational speed [rpm]']

# quick check
df_clean[['Power [W]', 'Temp diff [K]', 'Wear rate']].describe().round(3)

# let's see if these new features actually correlate with failure
new_features = ['Power [W]', 'Temp diff [K]', 'Wear rate']

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for i, feat in enumerate(new_features):
    sns.boxplot(x='Machine failure', y=feat, data=df_clean, ax=axes[i], palette='Set2')
    axes[i].set_title(f'{feat} vs Failure')
plt.tight_layout()
plt.show()

# Good output - Power and Temp diff both show clear differences between failed vs non-failed machines. Our new features are useful!


# ### 4.3 Encode Categorical Variable
# 
# The `Type` column has L/M/H. We'll use Label Encoding since there's a natural order (Low < Medium < High quality).


# encode Type column
le = LabelEncoder()
df_clean['Type_encoded'] = le.fit_transform(df_clean['Type'])

# check the mapping
print("Label encoding mapping:")
for cls, val in zip(le.classes_, le.transform(le.classes_)):
    print(f"  {cls} -> {val}")

# drop original Type column
df_clean = df_clean.drop(columns=['Type'])
df_clean.head()

# ## 5. Train/Validation/Test Split
# 
# We'll use 70/15/15 split as recommended in the project guidelines. Same split for all models for fair comparison.


# separate features and target
X = df_clean.drop(columns=['Machine failure'])
y = df_clean['Machine failure']

print("Total features:", X.shape[1])
print("Feature names:", X.columns.tolist())

# first split: 70% train, 30% temp (to be split into val + test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)

# second split: split temp into 50/50 -> 15% val, 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
)

print(f"Train size: {X_train.shape[0]}")
print(f"Validation size: {X_val.shape[0]}")
print(f"Test size: {X_test.shape[0]}")
print(f"\nFailure rate - Train: {y_train.mean()*100:.2f}%")
print(f"Failure rate - Val: {y_val.mean()*100:.2f}%")
print(f"Failure rate - Test: {y_test.mean()*100:.2f}%")

# ### 5.1 Feature Scaling
# 
# We scale features because:
# - Logistic Regression is sensitive to feature scale (gradient descent works better with normalized features)
# - SVM definitely needs scaling (it uses distance calculations)
# - Random Forest doesn't strictly need it but it doesn't hurt
# 
# **Important:** Fit scaler ONLY on training data, then apply to val and test (otherwise we leak test info).


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("Scaling done!")
print("Mean of scaled train data (should be ~0):", X_train_scaled.mean().round(3))
print("Std of scaled train data (should be ~1):", X_train_scaled.std().round(3))

# ### 5.2 Handle Class Imbalance with class_weight
# 
# Only 3.4% of our data is failures. If we don't handle this, the model will just predict "no failure" all the time and still get 96% accuracy - useless.
# 
# Instead of creating fake failure samples (SMOTE), we'll use `class_weight='balanced'` in each model. This tells the model to pay ~30x more attention to failure cases during training. Cleaner approach - no synthetic data, just smarter weighting.


# no resampling - we'll use class_weight in each model instead
# this is the weight ratio the models will use internally
n_samples = len(y_train)
n_classes = 2
n_failures = (y_train == 1).sum()
n_normal = (y_train == 0).sum()

print(f"Class 0 (No Failure): {n_normal} samples")
print(f"Class 1 (Failure):    {n_failures} samples")
print(f"\nImbalance ratio: 1 : {n_normal/n_failures:.1f}")
print("Models will use class_weight='balanced' to handle this")

# ## 6. Model Training
# 
# We'll train 3 models from different families:
# 1. **Logistic Regression** - linear model (baseline)
# 2. **Random Forest** - tree-based ensemble  
# 3. **Support Vector Machine** - kernel method
# 
# For each one we'll do hyperparameter tuning using GridSearchCV with 5-fold cross validation.


# ### 6.1 Model 1: Logistic Regression


# define hyperparameters to search
lr_params = {
     'C': [ 1, 10],
    'solver': ['liblinear'],
    'penalty': ['l2']
}

# class_weight='balanced' handles the imbalance automatically
lr = LogisticRegression(max_iter=1000, random_state=SEED, class_weight='balanced')
lr_grid = GridSearchCV(lr, lr_params, cv=3, scoring='f1', n_jobs=1)
lr_grid.fit(X_train_scaled, y_train)

print("Best parameters:", lr_grid.best_params_)
print("Best CV F1 score:", round(lr_grid.best_score_, 4))

best_lr = lr_grid.best_estimator_

# evaluate on test set
lr_pred = best_lr.predict(X_test_scaled)

# print classification report
print("\n" + "="*60)
print("Logistic Regression - Classification Report")
print("="*60)
print(classification_report(y_test, lr_pred, target_names=['No Failure', 'Failure']))

# confusion matrix visualization
cm = confusion_matrix(y_test, lr_pred)
total = cm.sum()
labels = np.array([[f'{val}\n({val/total*100:.1f}%)' for val in row] for row in cm])

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=labels, fmt='', cmap='Blues',
            xticklabels=['No Failure', 'Failure'],
            yticklabels=['No Failure', 'Failure'],
            annot_kws={'size': 12, 'weight': 'bold'},
            linewidths=1, linecolor='white')
plt.title('Logistic Regression - Confusion Matrix', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.tight_layout()
plt.show()


# look at the coefficients - which features matter most for LR?
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': best_lr.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)

print("Logistic Regression coefficients (sorted by importance):")
print(coef_df)

# plot it
plt.figure(figsize=(10, 5))
colors = ['red' if c < 0 else 'green' for c in coef_df['Coefficient']]
plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
plt.title('Logistic Regression Coefficients')
plt.xlabel('Coefficient value')
plt.tight_layout()
plt.show()


# ### 6.2 Model 2: Random Forest


# RF hyperparameters
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

rf = RandomForestClassifier(random_state=SEED, n_jobs=-1, class_weight='balanced')
rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='f1', n_jobs=1)
rf_grid.fit(X_train_scaled, y_train)

print("Best parameters:", rf_grid.best_params_)
print("Best CV F1 score:", round(rf_grid.best_score_, 4))

best_rf = rf_grid.best_estimator_

# evaluate on test set
rf_pred = best_rf.predict(X_test_scaled)

# print classification report
print("\n" + "="*60)
print("Random Forest - Classification Report")
print("="*60)
print(classification_report(y_test, rf_pred, target_names=['No Failure', 'Failure']))

# confusion matrix visualization
cm = confusion_matrix(y_test, rf_pred)
total = cm.sum()
labels = np.array([[f'{val}\n({val/total*100:.1f}%)' for val in row] for row in cm])

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=labels, fmt='', cmap='Greens',
            xticklabels=['No Failure', 'Failure'],
            yticklabels=['No Failure', 'Failure'],
            annot_kws={'size': 12, 'weight': 'bold'},
            linewidths=1, linecolor='white')
plt.title('Random Forest - Confusion Matrix', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.tight_layout()
plt.show()


# feature importance from RF
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("Random Forest feature importance:")
print(importance_df)

plt.figure(figsize=(10, 5))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='forestgreen')
plt.title('Random Forest Feature Importance')
plt.xlabel('Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# Notice how our derived **Power** feature is among the top important features! Good engineering intuition pays off.


# ### 6.3 Model 3: Support Vector Machine
# 
# SVM can be slow on large data so we'll use a smaller hyperparameter grid.


# SVM hyperparameters
svm_params = {
    'C': [1, 10],
    'kernel': ['rbf'],
    'gamma': ['scale']
}

svm = SVC(random_state=SEED, probability=True, class_weight='balanced')
svm_grid = GridSearchCV(svm, svm_params, cv=3, scoring='f1', n_jobs=1)
svm_grid.fit(X_train_scaled, y_train)

print("Best parameters:", svm_grid.best_params_)
print("Best CV F1 score:", round(svm_grid.best_score_, 4))

best_svm = svm_grid.best_estimator_

# evaluate on test set
svm_pred = best_svm.predict(X_test_scaled)

# print classification report
print("\n" + "="*60)
print("SVM - Classification Report")
print("="*60)
print(classification_report(y_test, svm_pred, target_names=['No Failure', 'Failure']))

# confusion matrix visualization
cm = confusion_matrix(y_test, svm_pred)
total = cm.sum()
labels = np.array([[f'{val}\n({val/total*100:.1f}%)' for val in row] for row in cm])

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=labels, fmt='', cmap='Oranges',
            xticklabels=['No Failure', 'Failure'],
            yticklabels=['No Failure', 'Failure'],
            annot_kws={'size': 12, 'weight': 'bold'},
            linewidths=1, linecolor='white')
plt.title('SVM - Confusion Matrix', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.tight_layout()
plt.show()


# SVM doesn't have direct feature importance for RBF kernel
# but we can use permutation importance as an alternative
from sklearn.inspection import permutation_importance

# use a small sample for speed
perm = permutation_importance(best_svm, X_val_scaled[:500], y_val[:500], 
                              n_repeats=3, random_state=SEED, n_jobs=1)

svm_imp_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': perm.importances_mean
}).sort_values('Importance', ascending=False)

print("SVM permutation importance:")
print(svm_imp_df)

# ## 7. Model Evaluation & Comparison
# 
# Now the most important part - let's properly evaluate all three models on the **test set** (which they haven't seen yet).


# helper function to get all metrics in one go
def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba)
    }
    return metrics, y_pred, y_proba

# evaluate all 3
lr_metrics, lr_pred, lr_proba = evaluate_model(best_lr, X_test_scaled, y_test, 'Logistic Regression')
rf_metrics, rf_pred, rf_proba = evaluate_model(best_rf, X_test_scaled, y_test, 'Random Forest')
svm_metrics, svm_pred, svm_proba = evaluate_model(best_svm, X_test_scaled, y_test, 'SVM')

# put into a table
results = pd.DataFrame([lr_metrics, rf_metrics, svm_metrics])
results = results.set_index('Model').round(4)
print("=== Model Comparison Table ===")
print(results)

# visualize the comparison
results.plot(kind='bar', figsize=(12, 5), colormap='viridis', edgecolor='black')
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xticks(rotation=15)
plt.legend(loc='lower right')
plt.ylim(0, 1.05)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


# ### 7.1 Confusion Matrices


fig, axes = plt.subplots(1, 3, figsize=(15, 4))
models_preds = [('Logistic Regression', lr_pred), ('Random Forest', rf_pred), ('SVM', svm_pred)]

for i, (name, pred) in enumerate(models_preds):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['No Fail', 'Fail'], yticklabels=['No Fail', 'Fail'])
    axes[i].set_title(f'{name}')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.tight_layout()
plt.show()


# ### 7.2 ROC Curves


plt.figure(figsize=(8, 6))

for name, proba in [('Logistic Regression', lr_proba), ('Random Forest', rf_proba), ('SVM', svm_proba)]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


#  7.3 Learning Curves
# 
# These tell us if our models are overfitting or underfitting.


from sklearn.model_selection import learning_curve

def plot_learning_curve(model, X, y, title, ax):
    # use StratifiedKFold to make sure both classes appear in each fold
    cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv_strat, scoring='f1', n_jobs=1,
        train_sizes=np.linspace(0.2, 1.0, 5), random_state=SEED
    )
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_std = val_scores.std(axis=1)
    
    ax.plot(train_sizes, train_mean, 'o-', label='Training F1', color='blue')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
    ax.plot(train_sizes, val_mean, 's-', label='Validation F1', color='red')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='red')
    ax.set_title(title)
    ax.set_xlabel('Training samples')
    ax.set_ylabel('F1 Score')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

# IMPORTANT: use X_train_scaled and y_train (the ORIGINAL imbalanced data, NOT SMOTE)
print("Using training data shape:", X_train_scaled.shape)
print("Class distribution:", pd.Series(y_train).value_counts().to_dict())
print("Generating learning curves... (this may take a minute)\n")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
plot_learning_curve(best_lr, X_train_scaled, y_train, 'Logistic Regression', axes[0])
plot_learning_curve(best_rf, X_train_scaled, y_train, 'Random Forest', axes[1])
plot_learning_curve(best_svm, X_train_scaled, y_train, 'SVM', axes[2])
plt.tight_layout()
plt.show()

# ### 7.4 Statistical Significance Test


# get cross-validation F1 scores for all models on the ORIGINAL (non-SMOTE) data
# using class_weight='balanced' inside the models handles the imbalance
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

print("Running cross-validation on original training data (no SMOTE)...")
print(f"Data shape: {X_train_scaled.shape}")
print(f"Class distribution: {pd.Series(y_train).value_counts().to_dict()}\n")

lr_cv = cross_val_score(best_lr, X_train_scaled, y_train, cv=cv, scoring='f1', n_jobs=1)
rf_cv = cross_val_score(best_rf, X_train_scaled, y_train, cv=cv, scoring='f1', n_jobs=1)
svm_cv = cross_val_score(best_svm, X_train_scaled, y_train, cv=cv, scoring='f1', n_jobs=1)

print("CV F1 Scores (mean ± std):")
print(f"  Logistic Regression: {lr_cv.mean():.4f} ± {lr_cv.std():.4f}")
print(f"  Random Forest:       {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")
print(f"  SVM:                 {svm_cv.mean():.4f} ± {svm_cv.std():.4f}")

# Wilcoxon test - comparing best two models
print("=== Wilcoxon Signed-Rank Test ===\n")

# RF vs LR
stat, p_value = wilcoxon(rf_cv, lr_cv)
print(f"RF vs LR: statistic={stat:.4f}, p-value={p_value:.4f}")
if p_value < 0.05:
    print("  -> RF is SIGNIFICANTLY better than LR (p < 0.05)\n")
else:
    print("  -> Difference is NOT statistically significant\n")

# RF vs SVM
stat, p_value = wilcoxon(rf_cv, svm_cv)
print(f"RF vs SVM: statistic={stat:.4f}, p-value={p_value:.4f}")
if p_value < 0.05:
    print("  -> RF is SIGNIFICANTLY better than SVM (p < 0.05)")
else:
    print("  -> Difference is NOT statistically significant")

# ## 8. Engineering Interpretation
# 
# Let's translate the numbers into engineering meaning:
# 
# **What does an F1-Score of 0.80+ mean for a factory?**
# - High **Precision** = when we predict failure, we're usually right → fewer false alarms (which would mean unnecessary maintenance)
# - High **Recall** = we catch most actual failures → fewer machine breakdowns
# 
# **Cost analysis:**
# - A *false positive* costs ~$500 (unnecessary inspection)
# - A *false negative* costs ~$10,000+ (unplanned downtime, broken parts)
# - So Recall is more important than Precision in this context
# 
# **Looking at our best model (Random Forest):**


# detailed report for best model
print("=== Best Model: Random Forest - Detailed Report ===\n")
print(classification_report(y_test, rf_pred, target_names=['No Failure', 'Failure']))

# count how many failures we missed
cm = confusion_matrix(y_test, rf_pred)
tn, fp, fn, tp = cm.ravel()
print(f"True Positives (correctly predicted failures): {tp}")
print(f"False Negatives (MISSED failures - bad!):       {fn}")
print(f"False Positives (false alarms):                 {fp}")
print(f"True Negatives (correctly predicted no fail):   {tn}")

print(f"\nOut of {tp+fn} actual failures, we caught {tp} ({100*tp/(tp+fn):.1f}%)")
