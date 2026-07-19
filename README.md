# Music Genre Classification 2024

An end-to-end Machine Learning pipeline implemented in Python to predict the musical genre (`Class`) of audio tracks based on their acoustic features and metadata.

## 📊 Dataset Overview

The dataset contains **14,396 instances** and **18 initial features** detailing structural and perceptual properties of various tracks.

### Features Description
* **Artist Name / Track Name**: Metadata identifying the creator and song title.
* **Popularity**: Numerical metric of a track's current target reach.
* **Danceability / Energy / Valence**: Perceptual measures (0.0 to 1.0) describing tempo/rhythm suitability for dancing, intensity/activity levels, and musical positiveness.
* **Loudness / Tempo**: Physical properties measured in decibels (dB) and Beats Per Minute (BPM).
* **Key / Mode / Time Signature**: Structural components utilizing Pitch Class notation, major/minor scale modalities (1 or 0), and rhythmic structures.
* **Speechiness / Acousticness / Instrumentalness**: Confidence values detecting spoken words, acoustic instrumentation vs. digital, and the presence or absence of vocals.
* **Class (Target)**: The multi-class categorical target column mapping to specific music genres.

---

## 🛠️ Tech Stack & Libraries

* **Data Manipulation & EDA**: `pandas`, `numpy`, `scipy`
* **Data Visualization**: `matplotlib`, `seaborn`
* **Machine Learning Models**: `scikit-learn`, `xgboost`, `catboost`, `lightgbm`
* **Preprocessing Frameworks**: Custom `ColumnTransformer` pipelines combining `SimpleImputer`, `StandardScaler`, `RobustScaler`, `PowerTransformer`, and `OneHotEncoder`.

---

## 📈 Data Pipeline & Preprocessing Workflow

### 1. Exploratory Data Analysis (EDA)
* Analyzed data distributions using systematic descriptive statistics (Range, IQR, Variance, and Coefficient of Variation).
* Correctly identified hidden categorical representations stored as numeric formats (e.g., `mode`, `key`, `time_signature`).

### 2. Data Cleaning & Feature Engineering
* **Irrelevant Feature Removal**: Dropped the unique tracking sequence `Id` column to reduce model dimensionality.
* **Missing Value Strategy**: Imputed continuous variables (`Popularity`, `instrumentalness`) via their median distributions to mitigate outlier effects, and handled categorical column shifts (`key`) by applying frequentist mode imputation.
* **Integrity Checks**: Confirmed zero duplicate rows to protect downstream cross-validation evaluations.

---

## 🚀 Model Architecture Strategy

The framework loads comprehensive classification stacks to balance bias-variance tradeoffs:
* **Baseline Estimators**: `DecisionTreeClassifier`, `KNeighborsClassifier`, `GaussianNB`, `LogisticRegression`, `LinearSVC`.
* **Advanced Ensemble Ensembles**: `RandomForestClassifier`, `GradientBoostingClassifier`, `XGBClassifier`, `CatBoostClassifier`, `LGBMClassifier`.
* **Meta-Learning Structures**: Configured unified prediction layers via `VotingClassifier` and `StackingClassifier` wrappers to blend baseline strengths.

--- 

## Team Members : 
* [Sedra Merkhan](https://github.com/sedramerkhan)
* [Ahmad Sadik](https://github.com/AhmadSadik1)
* [Mays AlFasfous](https://github.com/Mays-AlFasfous)
* [Hesham Alsaadi](https://github.com/HeshamSaadi)
