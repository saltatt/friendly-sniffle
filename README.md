# Car Price Prediction — Regression Analysis Project

A hands-on regression project predicting used car prices from tabular listing data. The goal was to work through a full regression workflow end to end: cleaning messy real-world data, exploring it visually, building a baseline linear model, comparing regularized variants (Lasso, Ridge, ElasticNet), incorporating categorical features, experimenting with feature engineering, and evaluating results with a business-relevant metric.

## Project overview

- **Task:** predict `selling_price` for used cars from features like year, mileage, engine specs, fuel type, seller type, transmission, and ownership history.
- **Approach:** linear models only (no tree-based / black-box models), with an emphasis on interpretability — feature scaling, regularization, and coefficient inspection.
- **Data:** train/test CSVs of used car listings ([source](https://github.com/Murcha1990/MLDS_ML_2022/tree/main/Hometasks/HT1)).

## What's inside

```
car-price-prediction/
├── notebooks/
│   └── Car_Price_Prediction_Regression_Project.ipynb   # full analysis, end to end
├── data/                                                # (data is pulled directly from source URLs in the notebook)
├── requirements.txt
├── .gitignore
└── README.md
```

## Workflow

1. **EDA & cleaning** — inspect the raw data, handle duplicates, parse unit-laden columns (`mileage`, `engine`, `max_power`, `torque`), impute missing values, and fix dtypes.
2. **Visualization** — pairplots, correlation heatmap, scatter plots to understand feature relationships and compare train/test distributions.
3. **Baseline model** — linear regression on numeric features, evaluated with R² and MSE on train and test.
4. **Regularization** — Lasso, Ridge, and ElasticNet, each tuned via 10-fold grid search over their regularization hyperparameters.
5. **Categorical features** — one-hot encoding of categorical columns and `seats`.
6. **Feature engineering (bonus)** — ratios (power per liter), polynomial terms (year²), thresholded indicators, missing-value flags, outlier clipping, and log-transforming the target.
7. **Business metric** — the share of predictions within 10% of the true price, a more interpretable stand-in for R²/MSE when reporting to non-technical stakeholders.

## Results at a glance

- A plain linear regression on numeric features alone already picks up a meaningful signal from `max_power`, `year`, and `engine`.
- Standardization is essential once regularization enters the picture, and makes coefficients directly interpretable as feature importances.
- Lasso/Ridge/ElasticNet grid search gave only modest gains over the numeric-only baseline.
- Categorical features (fuel type, seller type, transmission, ownership) and engineered features are the more promising directions for further accuracy.

## Setup

```bash
git clone <this-repo-url>
cd car-price-prediction
pip install -r requirements.txt
jupyter notebook notebooks/Car_Price_Prediction_Regression_Project.ipynb
```

The notebook pulls the train/test CSVs directly from their source URLs, so no manual data download is needed.

## Tech stack

pandas · numpy · scikit-learn · matplotlib · seaborn

## Author

Saltanat — [GitHub](https://github.com/saltatt)
