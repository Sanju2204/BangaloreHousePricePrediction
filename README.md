# 🏠 Bangalore House Price Prediction

> A complete end-to-end machine learning project that predicts residential house prices in Bangalore based on location, area, BHK, and bathroom count.

---

## 📌 Project Overview

| Detail | Info |
|---|---|
| **Project Type** | Supervised Machine Learning — Regression |
| **Dataset** | Bengaluru House Price Data |
| **Source** | [Kaggle — amitabhajoy](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data) |
| **Total Records** | ~13,000 listings |
| **Final Model** | Linear Regression |
| **Model R² Score** | 0.86 (test set) |
| **Cross-Val R²** | ≈ 0.76 (5-fold average) |
| **Frontend** | Streamlit Web App |

---

## 🎯 Project Objective

The objective of this project is to build an end-to-end machine learning pipeline for estimating house prices in Bangalore using historical housing data.

The project covers:
- Data cleaning and preprocessing
- Feature engineering
- Outlier removal using domain logic
- Training and comparing multiple regression models
- Model evaluation using standard metrics
- Saving and loading the trained model
- Deploying a simple and interactive Streamlit web app

---

## 📁 Folder Structure

```
BangaloreHousePricePrediction/
│
├── app.py                        ← Streamlit web application
├── requirements.txt              ← Python dependencies
├── README.md                     ← Project documentation
│
├── data/
│   └── cleaned_house_prices.csv  ← Preprocessed dataset
│
├── model/
│   ├── house_price_model.pkl     ← Trained ML model
│   ├── columns.json  ← One-hot encoded feature columns
│   └── known_locations.json      ← Valid grouped location names
│
└── notebooks/
    └── eda_and_modeling.ipynb    ← Full EDA, cleaning, and model training
```

---

## 📊 Dataset

The dataset contains Bangalore real estate listings with the following columns:

| Column | Description |
|---|---|
| `area_type` | Type of area measurement |
| `availability` | Ready to move or future date |
| `location` | Neighbourhood in Bangalore |
| `size` | BHK info as text (e.g., "2 BHK") |
| `society` | Housing society name |
| `total_sqft` | Total area in square feet |
| `bath` | Number of bathrooms |
| `balcony` | Number of balconies |
| `price` | **Target variable** — price in lakhs |

---

## ✅ Features Used in Final Model

| Feature | Type | Description |
|---|---|---|
| `location` | Categorical | Neighbourhood in Bangalore |
| `total_sqft` | Numerical | Total area in square feet |
| `bath` | Numerical | Number of bathrooms |
| `bhk` | Numerical | Number of bedrooms (extracted from size) |

**Target Variable:** `price` (in Lakhs ₹)

---

## 🧹 Data Preprocessing Steps

1. Removed unnecessary columns — `area_type`, `availability`, `society`, `balcony`
2. Dropped rows with missing values
3. Extracted `bhk` from the `size` column using regex (`"2 BHK"` → `2`)
4. Cleaned `total_sqft` by converting ranges to numeric averages (`"1000-1500"` → `1250.0`)
5. Stripped extra whitespace from location names
6. Grouped rare locations (fewer than 10 listings) into `other`
7. Removed outliers using four domain-logic rules:
   - Minimum 300 sqft per BHK rule
   - Price-per-sqft filtering within each location
   - BHK-based price consistency check
   - Bathroom count sanity check (`bath < bhk + 2`)

---

## 🤖 Models Compared

| Model | R² | MAE | RMSE |
|---|---|---|---|
| **Linear Regression** | **0.8629** | **17.44** | **31.52** |
| Random Forest | 0.7954 | 16.22 | 38.51 |
| Lasso Regression | 0.7309 | 23.47 | 44.16 |
| Decision Tree | 0.5929 | 18.55 | 54.33 |

---

## 🏆 Final Model — Linear Regression

Linear Regression was selected as the final model because:

- Highest R² score (0.86) on the test set
- Lowest RMSE (31.52) — best performance on high-value properties
- Stable cross-validation scores (Mean R² ≈ 0.76 across 5 folds)
- No overfitting — train/test R² gap within acceptable range
- Fully interpretable — coefficients explain feature impact on price

---

## 📈 Model Evaluation

Metrics used for evaluation:

- **R² Score** — How much price variation the model explains
- **MAE** — Average prediction error in lakhs
- **MSE** — Mean squared error
- **RMSE** — Root mean squared error (penalises large errors)
- **Cross-Validation** — 5-fold CV to check model stability

```
Mean Cross-Validation R² ≈ 0.76
```

---

## 💾 Files Saved During Training

| File | Description |
|---|---|
| `house_price_model.pkl` | Trained Linear Regression model |
| `columns.json` | All feature columns after one-hot encoding |
| `known_locations.json` | Valid grouped location names used during training |

---

## 🌐 Demo App — Streamlit

A Streamlit web app was built where users can input property details and receive a predicted house price instantly.

**User Inputs:**
- Location (dropdown)
- Total Square Feet
- Number of BHK
- Number of Bathrooms

**Output:**
- Predicted house price in Lakhs ₹

### Sample Prediction

| Input | Value |
|---|---|
| Location | Rajaji Nagar |
| Total Square Feet | 2000 |
| BHK | 3 |
| Bathrooms | 3 |
| **Predicted Price** | **₹ 161.39 Lakhs** |

---

## ⚙️ How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Sanju2204/BangaloreHousePricePrediction
cd BangaloreHousePricePrediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

> **Note:** Run the notebook first (`notebooks/eda_and_modeling.ipynb`) to generate the model files before launching the app.

---

## 🧰 Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-white?style=flat&logo=matplotlib&logoColor=black)

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| matplotlib & seaborn | Data visualisation |
| scikit-learn | Model training and evaluation |
| pickle | Saving and loading model |
| Streamlit | Web app frontend |

---

## 💡 Key Learnings

Through this project, I learned:

- How to clean real-world housing data with multiple inconsistency types
- How to handle high-cardinality categorical variables like location
- How to remove outliers using business logic, not just statistics
- How to compare multiple regression models using appropriate metrics
- How to save and reload a trained ML model for deployment
- How to build a simple and interactive frontend using Streamlit

---

## ⚠️ Limitations

- Dataset is from 2017–2019 — predictions do not reflect current market prices
- Model does not include factors like floor level, furnishing, or property view
- Predictions for rare or unseen locations may be less accurate
- External factors like market trends and economy are not captured

---

## 🚀 Future Improvements

- Add more features such as balcony count, area type, and furnishing status
- Explore advanced boosting models like XGBoost or LightGBM
- Deploy on a public cloud platform (Streamlit Cloud, Render, or AWS)
- Improve UI with better design and input validation
- Add model explainability using SHAP values

---

## 👤 Author

**Sanjana Gupta**
B.Tech CSE

[![GitHub](https://img.shields.io/badge/GitHub-Sanju2204-black?style=flat&logo=github)](https://github.com/Sanju2204)

---

> ⭐ If you found this project helpful, please consider giving it a star on GitHub!
