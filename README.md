# Bangalore House Price Prediction

A machine learning project that predicts house prices in Bangalore based on important property features such as location, total square feet, number of bedrooms (BHK), and number of bathrooms.

## Project Objective

The objective of this project is to build an end-to-end machine learning pipeline for estimating house prices in Bangalore using historical housing data. The project includes data cleaning, feature engineering, outlier removal, model training, evaluation, and deployment using a Streamlit web app.

## Dataset

The project uses a Bangalore housing dataset containing features such as:

- location
- total_sqft
- bath
- bhk
- price

Target variable:

- **price**

## Features Used in Final Model

The final model uses the following input features:

- `location`
- `total_sqft`
- `bath`
- `bhk`

Target:

- `price`

## Data Preprocessing Steps

The following preprocessing steps were performed:

1. Removed unnecessary columns such as `area_type`, `availability`, `society`, and `balcony`
2. Removed rows with missing values
3. Extracted `bhk` from the `size` column
4. Cleaned the `total_sqft` column by converting ranges into numeric averages
5. Stripped extra spaces from location names
6. Grouped rare locations into `other`
7. Removed outliers using:
   - minimum square feet per BHK rule
   - price per square foot filtering
   - BHK-based outlier filtering
   - bathroom count sanity check

## Models Compared

The following regression models were trained and compared:

- Linear Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor

## Final Model

The final saved model is:

- **Linear Regression**

It was selected because it provided stable and interpretable performance for this project setup.

## Model Evaluation

Evaluation metrics used:

- R² Score
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)

Cross-validation was also performed to check model stability.

Example result:

- Mean Cross-Validation R² ≈ **0.76**

## Project Structure

```bash
BangaloreHousePricePrediction/
│
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── cleaned_house_prices.csv
├── model/
│   ├── house_price_model.pkl
│   ├── columns.json
│   └── known_locations.json
└── notebooks/